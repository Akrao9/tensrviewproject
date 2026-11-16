//! Parquet part files under `{store_root}/runs/{run_id}/rank={rank}/attempt={attempt}/`.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt, TryStreamExt};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::async_reader::{MetadataFetch, MetadataSuffixFetch};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;

/// The first read from the end of a part. The server's own parts have footers of about
/// 7.2 to 7.7 KiB (measured Nov 14, 2026 on router-only and full captures), and longer
/// statistics or more columns make them grow; 64 KiB leaves room and stays small next to
/// a part. A larger footer costs a second read, never a wrong answer.
const FOOTER_PREFETCH_BYTES: usize = 64 * 1024;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// One stored part of a rank, as its path says: `attempt=<attempt>/part-<first>-<last>.parquet`.
#[derive(Clone, Debug)]
pub struct StoredPart {
    pub attempt: i64,
    /// The first step in the part; unknown for a part this server did not name.
    pub first_step: Option<i64>,
    pub location: Path,
}

/// What the server asks of a cloud store besides its URL.
#[derive(Clone, Debug)]
pub struct StoreOptions {
    /// How long a failing request is retried (with backoff) before its error is returned.
    /// `object_store` retries a transient error (a timeout, a 5xx, `SlowDown`) up to ten times
    /// over three minutes; the server holds a rank's buffer, and so its stream, for the whole
    /// of that, and at shutdown waits for it. Thirty seconds lets a stream be answered with
    /// `UNAVAILABLE` (the sender keeps its rows and retries) and keeps SIGTERM bounded. A
    /// request that is refused outright (403, 404) is never retried.
    pub retry_timeout: Duration,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            retry_timeout: Duration::from_secs(30),
        }
    }
}

/// The environment variables of one cloud provider, as `object_store` reads them (its own
/// `from_env` takes every `AWS_*` or `GOOGLE_*` variable). Nothing else is passed on: the
/// builders also accept bare names (`token`, `region`, `endpoint`), so handing them the whole
/// environment would let an unrelated `TOKEN` or `ENDPOINT` configure the store.
#[cfg(any(feature = "s3", feature = "gcs"))]
fn provider_env(env: impl IntoIterator<Item = (String, String)>, prefix: &str) -> Vec<(String, String)> {
    env.into_iter()
        .filter(|(key, _)| key.starts_with(prefix))
        .map(|(key, value)| (key.to_ascii_lowercase(), value))
        .collect()
}

#[cfg(any(feature = "s3", feature = "gcs"))]
fn retry_config(options: &StoreOptions) -> object_store::RetryConfig {
    object_store::RetryConfig {
        retry_timeout: options.retry_timeout,
        ..Default::default()
    }
}

#[derive(Clone)]
pub struct Storage {
    store: Arc<dyn ObjectStore>,
    root: Path,
}

impl Storage {
    pub fn new(store: Arc<dyn ObjectStore>, root: Path) -> Self {
        Self { store, root }
    }

    /// `file:///var/lib/tensorquill`, `s3://bucket/prefix`, `gs://bucket/prefix`, or a local
    /// directory path. Cloud credentials come from the usual environment variables and
    /// instance mechanisms, see [`Storage::open`].
    pub fn from_url(location: &str) -> Result<Self, Error> {
        Self::open(location, std::env::vars(), &StoreOptions::default())
    }

    /// Opens a store from `location`, reading cloud settings from `env`.
    ///
    /// S3 (`s3://`, `s3a://`): the `AWS_*` variables that `object_store` knows, among them
    /// `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION` or
    /// `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL` (an S3-compatible service; with `http://`
    /// also `AWS_ALLOW_HTTP=true`), the web-identity and container-credential variables.
    /// Without keys it asks the instance metadata service. Shared credential files and
    /// `AWS_PROFILE` are not read.
    ///
    /// GCS (`gs://`): the `GOOGLE_*` variables that `object_store` knows
    /// (`GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_SERVICE_ACCOUNT`, `GOOGLE_SERVICE_ACCOUNT_KEY`,
    /// `GOOGLE_BASE_URL`), then the application default credentials and the metadata server.
    ///
    /// Values are handed to `object_store` and never printed or logged here.
    pub fn open(
        location: &str,
        env: impl IntoIterator<Item = (String, String)>,
        options: &StoreOptions,
    ) -> Result<Self, Error> {
        let url = match url::Url::parse(location) {
            Ok(url) if url.scheme().len() > 1 => url,
            _ => {
                let dir = std::path::absolute(location)?;
                std::fs::create_dir_all(&dir)?;
                url::Url::from_directory_path(&dir).map_err(|_| format!("bad store path {location}"))?
            }
        };
        let (scheme, root) = object_store::ObjectStoreScheme::parse(&url)?;
        let _ = (&env, options); // used only by the cloud arms, which a local-only build leaves out
        let store: Arc<dyn ObjectStore> = match scheme {
            object_store::ObjectStoreScheme::Local => {
                std::fs::create_dir_all(url.to_file_path().map_err(|_| format!("bad store URL {location}"))?)?;
                Arc::new(object_store::local::LocalFileSystem::new())
            }
            #[cfg(feature = "s3")]
            object_store::ObjectStoreScheme::AmazonS3 => {
                use object_store::aws::AmazonS3Builder;
                let mut builder = AmazonS3Builder::new()
                    .with_url(url.as_str())
                    .with_retry(retry_config(options));
                let settings = provider_env(env, "AWS_");
                let setting = |name: &str| {
                    settings
                        .iter()
                        .find(|(key, _)| key == name)
                        .map(|(_, value)| value.as_str())
                };
                let plain_endpoint = ["aws_endpoint_url_s3", "aws_endpoint_url", "aws_endpoint"]
                    .into_iter()
                    .find_map(setting)
                    .is_some_and(|endpoint| endpoint.starts_with("http://"));
                if plain_endpoint && !setting("aws_allow_http").is_some_and(|value| value.eq_ignore_ascii_case("true"))
                {
                    return Err(
                        "the S3 endpoint (AWS_ENDPOINT_URL) is http://, which needs AWS_ALLOW_HTTP=true".into(),
                    );
                }
                for (key, value) in settings {
                    if let Ok(key) = key.parse() {
                        builder = builder.with_config(key, value);
                    }
                }
                Arc::new(builder.build()?)
            }
            #[cfg(feature = "gcs")]
            object_store::ObjectStoreScheme::GoogleCloudStorage => {
                use object_store::gcp::GoogleCloudStorageBuilder;
                let mut builder = GoogleCloudStorageBuilder::new()
                    .with_url(url.as_str())
                    .with_retry(retry_config(options));
                for (key, value) in provider_env(env, "GOOGLE_") {
                    if let Ok(key) = key.parse() {
                        builder = builder.with_config(key, value);
                    }
                }
                Arc::new(builder.build()?)
            }
            #[cfg(not(feature = "s3"))]
            object_store::ObjectStoreScheme::AmazonS3 => {
                return Err("this build of tensorquill-server has no S3 support (rebuild with --features s3)".into())
            }
            #[cfg(not(feature = "gcs"))]
            object_store::ObjectStoreScheme::GoogleCloudStorage => {
                return Err("this build of tensorquill-server has no GCS support (rebuild with --features gcs)".into())
            }
            _ => return Err("--store must be a directory, file://, s3:// or gs:// location".into()),
        };
        Ok(Self::new(store, root))
    }

    /// Replaces `runs/<run_id>/watchdog.json`, the run's progress status. One `PUT`: a reader
    /// sees the old document or the new one, never a mixture, on local disk (`object_store`
    /// writes a temporary file and renames it), S3 and GCS alike.
    pub async fn write_status(&self, run_id: &str, document: &serde_json::Value) -> Result<(), Error> {
        let path = self.status_path(run_id);
        let bytes = serde_json::to_vec_pretty(document)?;
        self.store.put(&path, PutPayload::from(Bytes::from(bytes))).await?;
        Ok(())
    }

    fn status_path(&self, run_id: &str) -> Path {
        self.root.clone().join("runs").join(run_id).join("watchdog.json")
    }

    /// The run IDs under `runs/`: one delimiter listing (a request per 1,000 runs on S3), not
    /// a walk of every part.
    pub async fn list_runs(&self) -> Result<Vec<String>, Error> {
        let runs = self.root.clone().join("runs");
        let listing = self.store.list_with_delimiter(Some(&runs)).await?;
        Ok(listing
            .common_prefixes
            .iter()
            .filter_map(|prefix| prefix.filename().map(String::from))
            .collect())
    }

    /// `runs/<run_id>/watchdog.json` as it was last written, or none if it was never written or
    /// is not JSON.
    pub async fn read_status(&self, run_id: &str) -> Result<Option<serde_json::Value>, Error> {
        match self.store.get(&self.status_path(run_id)).await {
            Ok(response) => Ok(serde_json::from_slice(&response.bytes().await?).ok()),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Replaces `runs/<run_id>/rank=<rank>/incidents.json`, what the server's detectors found.
    pub async fn write_incidents(&self, run_id: &str, rank: i32, document: &serde_json::Value) -> Result<(), Error> {
        let path = self.rank_dir(run_id, rank).join("incidents.json");
        let bytes = serde_json::to_vec_pretty(document)?;
        self.store.put(&path, PutPayload::from(Bytes::from(bytes))).await?;
        Ok(())
    }

    fn rank_dir(&self, run_id: &str, rank: i32) -> Path {
        self.root.clone().join("runs").join(run_id).join(format!("rank={rank}"))
    }

    /// Every part stored for a rank, ordered by attempt, then first step, then name: the
    /// order `load_records` reads them in.
    pub async fn rank_parts(&self, run_id: &str, rank: i32) -> Result<Vec<StoredPart>, Error> {
        let dir = self.rank_dir(run_id, rank);
        let objects: Vec<_> = self.store.list(Some(&dir)).try_collect().await?;
        let mut parts: Vec<StoredPart> = objects
            .into_iter()
            .filter(|object| {
                object.location.extension() == Some("parquet")
                    && object.location.filename().is_some_and(|name| name.starts_with("part-"))
            })
            .map(|object| StoredPart {
                attempt: object
                    .location
                    .parts()
                    .find_map(|part| {
                        part.as_ref()
                            .strip_prefix("attempt=")
                            .and_then(|attempt| attempt.parse().ok())
                    })
                    .unwrap_or(0),
                first_step: name_min_step(&object.location),
                location: object.location,
            })
            .collect();
        parts.sort_by(|a, b| {
            (a.attempt, a.first_step.unwrap_or(0), a.location.filename()).cmp(&(
                b.attempt,
                b.first_step.unwrap_or(0),
                b.location.filename(),
            ))
        });
        Ok(parts)
    }

    /// The rows of one part, in order.
    pub async fn read_part(&self, location: &Path) -> Result<Vec<RecordBatch>, Error> {
        let bytes = self.store.get(location).await?.bytes().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;
        Ok(reader.collect::<Result<Vec<_>, _>>()?)
    }

    fn attempt_dir(&self, run_id: &str, rank: i32, attempt: i64) -> Path {
        self.root
            .clone()
            .join("runs")
            .join(run_id)
            .join(format!("rank={rank}"))
            .join(format!("attempt={attempt}"))
    }

    /// Writes `batches` as one immutable part and returns its path. All batches share a schema.
    pub async fn write_part(
        &self,
        run_id: &str,
        rank: i32,
        attempt: i64,
        batches: &[RecordBatch],
        min_step: i64,
        max_step: i64,
    ) -> Result<Path, Error> {
        let properties = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, batches[0].schema(), Some(properties))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
        let path = self
            .attempt_dir(run_id, rank, attempt)
            .join(format!("part-{min_step:08}-{max_step:08}.parquet"));
        self.store.put(&path, PutPayload::from(Bytes::from(bytes))).await?;
        Ok(path)
    }

    /// The highest step already stored for this attempt of this rank, from the `step` column statistics in
    /// the Parquet footers. Only the parts whose file name claims the highest last step are
    /// opened; a part with a name this server did not write is always opened.
    pub async fn max_step(&self, run_id: &str, rank: i32, attempt: i64) -> Result<Option<i64>, Error> {
        let dir = self.attempt_dir(run_id, rank, attempt);
        let objects: Vec<_> = self.store.list(Some(&dir)).try_collect().await?;
        let mut named: Vec<(Option<i64>, Path, u64)> = objects
            .into_iter()
            .filter(|object| object.location.extension() == Some("parquet"))
            .map(|object| (name_max_step(&object.location), object.location, object.size))
            .collect();
        let newest = named.iter().filter_map(|(step, _, _)| *step).max();
        named.retain(|(step, _, _)| step.is_none() || *step == newest);
        let mut max = None;
        for (_, path, size) in named {
            // The default PageIndexPolicy::Skip: the row-group statistics in the footer hold
            // the step range, so the page indexes, stored before the footer, are never fetched.
            // (Loading them after a suffix read sliced the wrong bytes before parquet 60.0,
            // arrow-rs#11207, fixed by #11208; the pinned 60.1.0 has the fix.)
            let metadata = ParquetMetaDataReader::new()
                .with_prefetch_hint(Some(FOOTER_PREFETCH_BYTES))
                .load_via_suffix_and_finish(FooterSuffixFetch {
                    store: self.store.as_ref(),
                    path: &path,
                    size,
                })
                .await?;
            let column = metadata
                .file_metadata()
                .schema_descr()
                .columns()
                .iter()
                .position(|column| column.path().string() == "step");
            let Some(column) = column else { continue };
            for row_group in metadata.row_groups() {
                if let Some(Statistics::Int64(stats)) = row_group.column(column).statistics() {
                    if let Some(step) = stats.max_opt() {
                        max = max.max(Some(*step));
                    }
                }
            }
        }
        Ok(max)
    }
}

fn to_parquet_err(err: object_store::Error) -> ParquetError {
    ParquetError::External(Box::new(err))
}

struct FooterSuffixFetch<'a> {
    store: &'a dyn ObjectStore,
    path: &'a Path,
    size: u64,
}

impl MetadataFetch for FooterSuffixFetch<'_> {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.store.get_range(self.path, range).map_err(to_parquet_err).boxed()
    }
}

impl MetadataSuffixFetch for FooterSuffixFetch<'_> {
    fn fetch_suffix(&mut self, suffix: usize) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let options = GetOptions {
            range: Some(GetRange::Suffix(suffix as u64)),
            ..Default::default()
        };
        async move {
            match self.store.get_opts(self.path, options).await {
                Ok(response) => response.bytes().await.map_err(to_parquet_err),
                Err(object_store::Error::NotSupported { .. }) => {
                    let start = self.size.saturating_sub(suffix as u64);
                    self.store
                        .get_range(self.path, start..self.size)
                        .await
                        .map_err(to_parquet_err)
                }
                Err(err) => Err(to_parquet_err(err)),
            }
        }
        .boxed()
    }
}

/// `part-00000000-00000049.parquet` -> `49`.
fn name_max_step(path: &Path) -> Option<i64> {
    let name = path.filename()?.strip_prefix("part-")?.strip_suffix(".parquet")?;
    // Steps are not negative in practice, so the last `-` separates the two numbers.
    name.rsplit_once('-')?.1.parse().ok()
}

/// `part-00000000-00000049.parquet` -> `0`.
fn name_min_step(path: &Path) -> Option<i64> {
    let name = path.filename()?.strip_prefix("part-")?.strip_suffix(".parquet")?;
    name.rsplit_once('-')?.0.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::Mutex;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use futures::stream::BoxStream;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, PutMultipartOptions, PutOptions, PutResult,
    };

    use super::*;

    #[derive(Debug)]
    struct SpyStore {
        inner: Arc<dyn ObjectStore>,
        gets: Mutex<Vec<Option<GetRange>>>,
        reject_suffix: bool,
    }

    impl SpyStore {
        fn new(inner: Arc<dyn ObjectStore>, reject_suffix: bool) -> Self {
            Self {
                inner,
                gets: Mutex::new(Vec::new()),
                reject_suffix,
            }
        }

        fn clear_gets(&self) {
            self.gets.lock().unwrap().clear();
        }

        fn recorded_gets(&self) -> Vec<Option<GetRange>> {
            self.gets.lock().unwrap().clone()
        }
    }

    impl fmt::Display for SpyStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SpyStore({})", self.inner)
        }
    }

    impl ObjectStore for SpyStore {
        fn put_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> BoxFuture<'async_trait, object_store::Result<PutResult>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            self.inner.put_opts(location, payload, opts)
        }

        fn put_multipart_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
            opts: PutMultipartOptions,
        ) -> BoxFuture<'async_trait, object_store::Result<Box<dyn MultipartUpload>>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            self.inner.put_multipart_opts(location, opts)
        }

        fn get_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
            options: GetOptions,
        ) -> BoxFuture<'async_trait, object_store::Result<GetResult>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            self.gets.lock().unwrap().push(options.range.clone());
            if self.reject_suffix && matches!(options.range, Some(GetRange::Suffix(_))) {
                return futures::future::err(object_store::Error::NotSupported {
                    source: "suffix range unsupported".into(),
                })
                .boxed();
            }
            self.inner.get_opts(location, options)
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_delimiter<'life0, 'life1, 'async_trait>(
            &'life0 self,
            prefix: Option<&'life1 Path>,
        ) -> BoxFuture<'async_trait, object_store::Result<ListResult>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            self.inner.list_with_delimiter(prefix)
        }

        fn copy_opts<'life0, 'life1, 'life2, 'async_trait>(
            &'life0 self,
            from: &'life1 Path,
            to: &'life2 Path,
            options: CopyOptions,
        ) -> BoxFuture<'async_trait, object_store::Result<()>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            'life2: 'async_trait,
            Self: 'async_trait,
        {
            self.inner.copy_opts(from, to, options)
        }
    }

    fn make_batch(steps: Range<i64>, payload_bytes: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let step_values: Vec<i64> = steps.collect();
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let payloads: Vec<String> = step_values
            .iter()
            .map(|_| {
                let mut s = String::with_capacity(payload_bytes);
                for _ in 0..payload_bytes {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let ch = (b'A' + ((state >> 32) % 26) as u8) as char;
                    s.push(ch);
                }
                s
            })
            .collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(step_values)),
                Arc::new(StringArray::from(payloads)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn max_step_uses_single_suffix_range_on_local_filesystem() {
        let tmp = tempfile::tempdir().unwrap();
        let local = Arc::new(LocalFileSystem::new_with_prefix(tmp.path()).unwrap());
        let spy = Arc::new(SpyStore::new(local, false));
        let storage = Storage::new(spy.clone(), Path::from("prefix"));

        let batch = make_batch(0..300, 512);
        let path = storage.write_part("run-1", 0, 1, &[batch], 0, 299).await.unwrap();
        let meta = spy.head(&path).await.unwrap();
        assert!(
            meta.size > FOOTER_PREFETCH_BYTES as u64,
            "expected part > {} bytes, got {}",
            FOOTER_PREFETCH_BYTES,
            meta.size
        );
        spy.clear_gets();

        let max = storage.max_step("run-1", 0, 1).await.unwrap();
        assert_eq!(max, Some(299));
        assert_eq!(
            spy.recorded_gets(),
            vec![Some(GetRange::Suffix(FOOTER_PREFETCH_BYTES as u64))]
        );
    }

    #[tokio::test]
    async fn max_step_reads_tiny_and_unnamed_parts_via_suffix_only() {
        let spy = Arc::new(SpyStore::new(Arc::new(InMemory::new()), false));
        let storage = Storage::new(spy.clone(), Path::from("root"));

        storage
            .write_part("run-1", 0, 1, &[make_batch(0..10, 8)], 0, 9)
            .await
            .unwrap();
        storage
            .write_part("run-1", 0, 1, &[make_batch(10..20, 8)], 10, 19)
            .await
            .unwrap();

        // Also place a file whose name does not follow `part-<min>-<max>.parquet`.
        let custom_path = storage.attempt_dir("run-1", 0, 1).join("legacy.parquet");
        let custom_batch = make_batch(20..30, 8);
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, custom_batch.schema(), None).unwrap();
        writer.write(&custom_batch).unwrap();
        writer.close().unwrap();
        spy.put(&custom_path, PutPayload::from(Bytes::from(buf))).await.unwrap();

        let max = storage.max_step("run-1", 0, 1).await.unwrap();
        assert_eq!(max, Some(29));
        // Only the newest named part and the unnamed part should be opened, both via suffix.
        assert_eq!(
            spy.recorded_gets(),
            vec![
                Some(GetRange::Suffix(FOOTER_PREFETCH_BYTES as u64)),
                Some(GetRange::Suffix(FOOTER_PREFETCH_BYTES as u64)),
            ]
        );
    }

    #[tokio::test]
    async fn max_step_falls_back_to_bounded_range_when_suffix_unsupported() {
        let spy = Arc::new(SpyStore::new(Arc::new(InMemory::new()), true));
        let storage = Storage::new(spy.clone(), Path::from("root"));

        let batch = make_batch(0..300, 512);
        let path = storage.write_part("run-1", 0, 1, &[batch], 0, 299).await.unwrap();
        let meta = spy.head(&path).await.unwrap();
        assert!(meta.size > FOOTER_PREFETCH_BYTES as u64);
        spy.clear_gets();

        let max = storage.max_step("run-1", 0, 1).await.unwrap();
        assert_eq!(max, Some(299));
        assert_eq!(
            spy.recorded_gets(),
            vec![
                Some(GetRange::Suffix(FOOTER_PREFETCH_BYTES as u64)),
                Some(GetRange::Bounded((meta.size - FOOTER_PREFETCH_BYTES as u64)..meta.size)),
            ]
        );
    }

    #[tokio::test]
    async fn max_step_handles_footer_larger_than_prefetch_hint() {
        let spy = Arc::new(SpyStore::new(Arc::new(InMemory::new()), false));
        let storage = Storage::new(spy.clone(), Path::from("root"));

        // Many small batches written as separate row groups produce a Thrift footer larger than the first read.
        let batches: Vec<RecordBatch> = (0..1200).map(|step| make_batch(step..step + 1, 8)).collect();
        let properties = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, batches[0].schema(), Some(properties)).unwrap();
        for batch in &batches {
            writer.write(batch).unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
        let path = storage
            .attempt_dir("run-1", 0, 1)
            .join("part-00000000-00001199.parquet");
        spy.put(&path, PutPayload::from(Bytes::from(buf))).await.unwrap();

        let max = storage.max_step("run-1", 0, 1).await.unwrap();
        assert_eq!(max, Some(1199));
        let gets = spy.recorded_gets();
        assert_eq!(
            gets.len(),
            2,
            "expected initial prefetch + exact footer suffix: {gets:?}"
        );
        assert_eq!(gets[0], Some(GetRange::Suffix(FOOTER_PREFETCH_BYTES as u64)));
        assert!(matches!(
            gets[1],
            Some(GetRange::Suffix(n)) if n > FOOTER_PREFETCH_BYTES as u64
        ));
    }

    #[cfg(all(feature = "s3", feature = "gcs"))]
    fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[cfg(all(feature = "s3", feature = "gcs"))]
    #[test]
    fn only_the_providers_own_variables_reach_the_builder() {
        let kept = provider_env(
            env(&[
                ("AWS_REGION", "eu-west-1"),
                ("AWS_ENDPOINT_URL", "http://127.0.0.1:9"),
                // The builders accept these bare names; a stray variable must not set them.
                ("TOKEN", "x"),
                ("REGION", "x"),
                ("ENDPOINT", "x"),
                ("BUCKET", "x"),
                ("PATH", "/bin"),
                ("TENSORQUILL_TOKEN", "x"),
                ("GOOGLE_APPLICATION_CREDENTIALS", "/k.json"),
            ]),
            "AWS_",
        );
        assert_eq!(
            kept,
            env(&[("aws_region", "eu-west-1"), ("aws_endpoint_url", "http://127.0.0.1:9")])
        );
        let google = provider_env(
            env(&[("GOOGLE_BASE_URL", "http://127.0.0.1:9"), ("AWS_REGION", "x")]),
            "GOOGLE_",
        );
        assert_eq!(google, env(&[("google_base_url", "http://127.0.0.1:9")]));
    }

    #[cfg(all(feature = "s3", feature = "gcs"))]
    #[test]
    fn cloud_locations_build_without_touching_the_network_and_say_what_is_wrong() {
        let options = StoreOptions::default();
        // Building a store reads no credentials and sends nothing: the first request does.
        let s3 = Storage::open("s3://bucket/some/prefix", env(&[("AWS_REGION", "eu-west-1")]), &options).unwrap();
        assert_eq!(s3.root.as_ref(), "some/prefix");
        assert!(s3.store.to_string().contains("bucket"), "{}", s3.store);
        let gcs = Storage::open(
            "gs://bucket/some/prefix",
            env(&[("GOOGLE_SKIP_SIGNATURE", "true")]),
            &options,
        )
        .unwrap();
        assert_eq!(gcs.root.as_ref(), "some/prefix");
        assert!(gcs.store.to_string().contains("bucket"), "{}", gcs.store);
        // An emulator on plain HTTP needs the flag; the error names the variables, not their values.
        let plain = env(&[
            ("AWS_ENDPOINT_URL", "http://127.0.0.1:9"),
            ("AWS_ACCESS_KEY_ID", "key-id"),
            ("AWS_SECRET_ACCESS_KEY", "SECRET-VALUE"),
        ]);
        let error = Storage::open("s3://bucket", plain.clone(), &options)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("AWS_ALLOW_HTTP") && !error.contains("SECRET-VALUE"),
            "{error}"
        );
        let mut allowed = plain;
        allowed.push(("AWS_ALLOW_HTTP".into(), "true".into()));
        let built = Storage::open("s3://bucket", allowed, &options);
        assert!(built.is_ok(), "{:?}", built.err());
        // Other schemes are not stores this server writes.
        for location in ["az://container/p", "memory:///", "https://example.com/p"] {
            assert!(Storage::open(location, env(&[]), &options).is_err(), "{location}");
        }
    }

    #[cfg(not(any(feature = "s3", feature = "gcs")))]
    #[test]
    fn a_build_without_cloud_stores_says_so() {
        for location in ["s3://bucket/p", "gs://bucket/p"] {
            let error = Storage::open(location, Vec::new(), &StoreOptions::default())
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.contains("no S3 support") || error.contains("no GCS support"),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn runs_are_listed_by_prefix_and_a_missing_status_is_none() {
        let storage = Storage::new(Arc::new(InMemory::new()), Path::from("prefix"));
        assert!(storage.list_runs().await.unwrap().is_empty());
        for run in ["run-1", "run-10"] {
            storage
                .write_status(run, &serde_json::json!({"run_id": run}))
                .await
                .unwrap();
        }
        storage
            .write_part("run-2", 0, 1, &[make_batch(0..3, 4)], 0, 2)
            .await
            .unwrap();
        let mut runs = storage.list_runs().await.unwrap();
        runs.sort();
        assert_eq!(runs, ["run-1", "run-10", "run-2"]);
        assert_eq!(
            storage.read_status("run-10").await.unwrap().unwrap()["run_id"],
            "run-10"
        );
        assert!(storage.read_status("run-2").await.unwrap().is_none()); // parts, no status
        assert!(storage.read_status("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rank_parts_of_rank_1_do_not_include_rank_10() {
        let storage = Storage::new(Arc::new(InMemory::new()), Path::from("prefix"));
        for rank in [1, 10, 11] {
            storage
                .write_part("run-1", rank, 7, &[make_batch(0..3, 4)], 0, 2)
                .await
                .unwrap();
        }
        let parts = storage.rank_parts("run-1", 1).await.unwrap();
        assert_eq!(parts.len(), 1);
        assert!(
            parts[0].location.as_ref().contains("rank=1/attempt=7/"),
            "{:?}",
            parts[0]
        );
        assert_eq!(storage.max_step("run-1", 10, 7).await.unwrap(), Some(2));
        assert_eq!(storage.max_step("run-1", 2, 7).await.unwrap(), None);
    }

    #[tokio::test]
    async fn a_local_store_lists_no_runs_before_the_first_one() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = Storage::from_url(tmp.path().to_str().unwrap()).unwrap();
        assert!(storage.list_runs().await.unwrap().is_empty());
        storage
            .write_status("run-1", &serde_json::json!({"run_id": "run-1"}))
            .await
            .unwrap();
        assert_eq!(storage.list_runs().await.unwrap(), ["run-1"]);
        assert_eq!(storage.read_status("run-1").await.unwrap().unwrap()["run_id"], "run-1");
    }
}
