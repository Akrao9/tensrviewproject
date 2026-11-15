//! Parquet part files under `{store_root}/runs/{run_id}/rank={rank}/attempt={attempt}/`.

use std::ops::Range;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt, TryStreamExt};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutPayload};
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

#[derive(Clone)]
pub struct Storage {
    store: Arc<dyn ObjectStore>,
    root: Path,
}

impl Storage {
    pub fn new(store: Arc<dyn ObjectStore>, root: Path) -> Self {
        Self { store, root }
    }

    /// `file:///var/lib/tensorlens`, `s3://bucket/prefix`, `gs://bucket/prefix`, or a local
    /// directory path. Cloud credentials come from the usual environment variables.
    pub fn from_url(location: &str) -> Result<Self, Error> {
        let url = match url::Url::parse(location) {
            Ok(url) if url.scheme().len() > 1 => url,
            _ => {
                let dir = std::path::absolute(location)?;
                std::fs::create_dir_all(&dir)?;
                url::Url::from_directory_path(&dir).map_err(|_| format!("bad store path {location}"))?
            }
        };
        if url.scheme() == "file" {
            std::fs::create_dir_all(url.to_file_path().map_err(|_| format!("bad store URL {location}"))?)?;
        }
        let options = std::env::vars().map(|(key, value)| (key.to_ascii_lowercase(), value));
        let (store, root) = object_store::parse_url_opts(&url, options)?;
        Ok(Self::new(Arc::from(store), root))
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
}
