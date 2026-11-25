//! Parquet part files under `{store_root}/runs/{run_id}/rank={rank}/attempt={attempt}/`.
//!
//! Beside the parts, the same folder holds the attempt's scalar-only rows (`WAVE_AL_CONTRACT.md`
//! section 4.5): `scalars-<first_seq:08>-<last_seq:08>.parquet`, each created once and never replaced.
//! They are their own family: `is_part`, `rank_parts`, `max_step` and the replay never list them, so a
//! scalar row never raises the step watermark (D8) and the detectors never read one.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt, TryStreamExt};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::async_reader::{MetadataFetch, MetadataSuffixFetch};
use parquet::arrow::{ArrowWriter, ProjectionMask};
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

/// The columns of protocol 5's layout that the detectors never read: an attention module's map and
/// a probe's histograms (`WAVE_AL_CONTRACT.md` section 2.4, D11). `read_part_for_detect` leaves them
/// out; `read_part` keeps them.
pub const NOT_REPLAYED: [&str; 2] = ["probe_maps", "probe_histograms"];

/// Why a store with no `store.json` whose backend cannot create one atomically gets none, and the
/// two ways out.
fn no_conditional_create(path: &Path) -> String {
    format!(
        "the store has no {STORE_FILE} and its backend cannot create one atomically (a conditional \
         create is not supported), so the server writes none: two servers starting at once could each \
         write their own and give one store two IDs. Either provision the ID once by putting a \
         {STORE_FILE} at the store root (the object `{path}`) holding {{\"store_id\": \"<32 lowercase \
         hex digits, such as `openssl rand -hex 16` prints>\"}}, or turn on conditional writes for the backend (on S3, AWS_CONDITIONAL_PUT=etag, \
         its default, on a service that honours If-None-Match)"
    )
}

/// A new store ID: 128 bits from the operating system's random source, as 32 lowercase hex digits.
fn new_store_id() -> Result<String, Error> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|error| format!("cannot draw a store ID: {error}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// One stored part of a rank, as its path says: `attempt=<attempt>/part-<first>-<last>.parquet`.
#[derive(Clone, Debug)]
pub struct StoredPart {
    pub attempt: i64,
    /// The first step in the part; unknown for a part this server did not name.
    pub first_step: Option<i64>,
    pub location: Path,
}

/// The file at the store root that names the store (L3, decided by the author on Nov 20, 2026):
/// `{"store_id": "<32 lowercase hex digits>", "created_at_ms": <Unix milliseconds>}`. Written once,
/// by the first server to find none, and returned with every acknowledgement, so a sender can tell
/// a store wiped and recreated behind the same address from the one its offset was saved against.
pub const STORE_FILE: &str = "store.json";

/// The file beside a rank's parts that keeps the order of its attempts (contract 1, decided Nov 19,
/// 2026): `runs/<run_id>/rank=<rank>/attempts.json`.
pub const ATTEMPTS_FILE: &str = "attempts.json";

/// How an attempt got its place in a rank's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacedBy {
    /// Its own process named it: a heartbeat's `attempt`, or a batch's `sender_attempt`.
    Claim,
    /// Only its rows were seen (a backlog sent by a later process, or a process the server never heard).
    Rows,
    /// It was in the store before the store kept an order: numeric order among such attempts.
    Store,
}

impl PlacedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            PlacedBy::Claim => "claim",
            PlacedBy::Rows => "rows",
            PlacedBy::Store => "store",
        }
    }

    fn parse(word: &str) -> Option<Self> {
        match word {
            "claim" => Some(PlacedBy::Claim),
            "rows" => Some(PlacedBy::Rows),
            "store" => Some(PlacedBy::Store),
            _ => None,
        }
    }
}

/// One attempt in a rank's order. `first_seen_ms` is for people to read; it never orders anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttemptEntry {
    pub attempt: i64,
    pub by: PlacedBy,
    pub first_seen_ms: i64,
}

/// The attempts of one rank, oldest first (contract 1). The newest owns the rank's live progress.
/// An attempt is placed once and never moves: one its own process claims goes after every attempt
/// known; one seen only in rows goes just before the attempt whose process sent them. When the
/// sender named none (every SDK before Nov 19, 2026), nothing but the attempt's number says whether
/// its rows are a backlog or a new process whose first beat has not arrived: it is the session's
/// start time, so a number above the newest's goes after it (the job started again; until then a
/// new attempt whose rows beat its first heartbeat lost the rank for good) and any other just
/// before it (a backlog). Otherwise attempt numbers order nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttemptOrder {
    pub entries: Vec<AttemptEntry>,
    /// The order is the one in the store's `attempts.json`. False for a rank whose store has no
    /// such file yet (stored before contract 1, or nothing stored): written at the first placement.
    pub on_disk: bool,
}

impl AttemptOrder {
    /// The attempts already stored for a rank with no `attempts.json`: numeric order, except
    /// `owner`, the attempt a restored status names as the rank's own, which goes last (the server
    /// last saw it live; a replacement with a lower number than its predecessor is one).
    pub fn legacy(stored: &[i64], owner: Option<i64>) -> Self {
        let mut attempts: Vec<i64> = stored
            .iter()
            .copied()
            .filter(|attempt| Some(*attempt) != owner)
            .collect();
        attempts.sort_unstable();
        attempts.dedup();
        attempts.extend(owner);
        Self {
            entries: attempts
                .into_iter()
                .map(|attempt| AttemptEntry {
                    attempt,
                    by: PlacedBy::Store,
                    first_seen_ms: 0,
                })
                .collect(),
            on_disk: false,
        }
    }

    pub fn attempts(&self) -> Vec<i64> {
        self.entries.iter().map(|entry| entry.attempt).collect()
    }

    pub fn newest(&self) -> Option<i64> {
        self.entries.last().map(|entry| entry.attempt)
    }

    pub fn position(&self, attempt: i64) -> Option<usize> {
        self.entries.iter().position(|entry| entry.attempt == attempt)
    }

    pub fn contains(&self, attempt: i64) -> bool {
        self.position(attempt).is_some()
    }

    /// Places an attempt its own process named. True when it was not known.
    pub fn place_claim(&mut self, attempt: i64, now_ms: i64) -> bool {
        if self.contains(attempt) {
            return false;
        }
        self.entries.push(AttemptEntry {
            attempt,
            by: PlacedBy::Claim,
            first_seen_ms: now_ms,
        });
        true
    }

    /// Places an attempt seen in rows that `sender` sent (the stream's `sender_attempt`, none from
    /// an older sender). True when it was not known.
    pub fn place_rows(&mut self, attempt: i64, sender: Option<i64>, now_ms: i64) -> bool {
        if self.contains(attempt) {
            return false;
        }
        let entry = AttemptEntry {
            attempt,
            by: PlacedBy::Rows,
            first_seen_ms: now_ms,
        };
        let at = match (sender.and_then(|sender| self.position(sender)), self.entries.last()) {
            (Some(at), _) => at,
            (None, Some(newest)) if attempt < newest.attempt => self.entries.len() - 1,
            (None, _) => self.entries.len(),
        };
        self.entries.insert(at, entry);
        true
    }

    /// What reports say of the order: `first_seen` when the store's file has kept it from the
    /// start, `numeric` when the store has no file, `numeric_then_first_seen` when the file began
    /// with attempts stored before it.
    pub fn kind(&self) -> &'static str {
        if !self.on_disk {
            "numeric"
        } else if self.entries.iter().any(|entry| entry.by == PlacedBy::Store) {
            "numeric_then_first_seen"
        } else {
            "first_seen"
        }
    }

    pub fn to_json(&self, run_id: &str, rank: i32) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "run_id": run_id,
            "rank": rank,
            "attempts": self.entries.iter().map(|entry| serde_json::json!({
                "attempt": entry.attempt,
                "by": entry.by.as_str(),
                "first_seen_ms": entry.first_seen_ms,
            })).collect::<Vec<_>>(),
        })
    }

    /// The order an `attempts.json` holds; `None` when it is not one (another schema, an attempt
    /// twice, a field missing).
    pub fn from_json(document: &serde_json::Value) -> Option<Self> {
        if document["schema_version"] != 1 {
            return None;
        }
        let mut order = Self {
            entries: Vec::new(),
            on_disk: true,
        };
        for entry in document["attempts"].as_array()? {
            let attempt = entry["attempt"].as_i64()?;
            if order.contains(attempt) {
                return None;
            }
            order.entries.push(AttemptEntry {
                attempt,
                by: PlacedBy::parse(entry["by"].as_str()?)?,
                first_seen_ms: entry["first_seen_ms"].as_i64().unwrap_or(0),
            });
        }
        Some(order)
    }
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
                // The values `object_store` itself reads as true: 1, true, on, yes, y (any case).
                let allowed = setting("aws_allow_http").is_some_and(|value| {
                    ["1", "true", "on", "yes", "y"]
                        .iter()
                        .any(|yes| value.eq_ignore_ascii_case(yes))
                });
                if plain_endpoint && !allowed {
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

    /// The store's ID, from `store.json` at its root, written first when there is none (L3). An
    /// existing `store.json` that names an ID is used on any backend. The write is a create that
    /// fails if the file exists (`PutMode::Create`: a hard link on local disk, `If-None-Match` on S3,
    /// a generation match on GCS), so two servers that start on an empty store at once both end with
    /// the one that was written first.
    ///
    /// A backend that cannot create conditionally (`NotImplemented`: S3 with `AWS_CONDITIONAL_PUT`
    /// set to `disabled`; none of the supported backends by default) gets no `store.json` written:
    /// the call fails with an error that says how to provision the ID or turn conditional writes on.
    /// Until Nov 20, 2026 it wrote plainly and read back, and two servers starting at once could each
    /// read back their own ID and acknowledge two IDs for one store. A `store.json` that is not a store
    /// ID is an error, never replaced: replacing it would make every sender start its file again for
    /// nothing.
    pub async fn store_id(&self) -> Result<String, Error> {
        if let Some(id) = self.read_store_id().await? {
            return Ok(id);
        }
        let id = new_store_id()?;
        let created_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as u64);
        let document = serde_json::json!({"store_id": id, "created_at_ms": created_at_ms});
        let payload = || PutPayload::from(Bytes::from(serde_json::to_vec_pretty(&document).expect("JSON")));
        let path = self.store_file();
        let create = PutOptions {
            mode: PutMode::Create,
            ..PutOptions::default()
        };
        match self.store.put_opts(&path, payload(), create).await {
            Ok(_) => return Ok(id),
            Err(object_store::Error::AlreadyExists { .. }) => {}
            Err(object_store::Error::NotImplemented { .. }) => return Err(no_conditional_create(&path).into()),
            Err(error) => return Err(error.into()),
        }
        self.read_store_id()
            .await?
            .ok_or_else(|| "store.json was written but cannot be read back".into())
    }

    /// The ID `store.json` names, none when there is no such file.
    async fn read_store_id(&self) -> Result<Option<String>, Error> {
        let bytes = match self.store.get(&self.store_file()).await {
            Ok(response) => response.bytes().await?,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let id = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|document| document.get("store_id")?.as_str().map(String::from))
            .filter(|id| {
                !id.is_empty() && id.len() <= 128 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
            });
        match id {
            Some(id) => Ok(Some(id)),
            None => Err(format!(
                "{} at the store root does not name a store ID; move it away to let the server write a new one",
                STORE_FILE
            )
            .into()),
        }
    }

    fn store_file(&self) -> Path {
        self.root.clone().join(STORE_FILE)
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

    /// Replaces `runs/<run_id>/run.json`, the run's manifest: its project, name and setup, as
    /// rank 0's sender gave it.
    pub async fn write_manifest(&self, run_id: &str, document: &serde_json::Value) -> Result<(), Error> {
        let path = self.root.clone().join("runs").join(run_id).join("run.json");
        let bytes = serde_json::to_vec_pretty(document)?;
        self.store.put(&path, PutPayload::from(Bytes::from(bytes))).await?;
        Ok(())
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

    /// Every part stored for a rank, ordered by the rank's attempt order (`attempt_order`: its
    /// `attempts.json`, or numeric for a store without one), then first step, then name. Until
    /// Nov 19, 2026 by attempt number (RS05).
    pub async fn rank_parts(&self, run_id: &str, rank: i32) -> Result<Vec<StoredPart>, Error> {
        let mut parts = self.list_parts(run_id, rank).await?;
        let order = self.order_of(run_id, rank, distinct_attempts(&parts)).await?;
        let place = |attempt: i64| order.position(attempt).unwrap_or(usize::MAX);
        parts.sort_by(|a, b| {
            (
                place(a.attempt),
                a.attempt,
                a.first_step.unwrap_or(0),
                a.location.filename(),
            )
                .cmp(&(
                    place(b.attempt),
                    b.attempt,
                    b.first_step.unwrap_or(0),
                    b.location.filename(),
                ))
        });
        Ok(parts)
    }

    /// The attempts a rank has parts of, in numeric order, each once.
    pub async fn stored_attempts(&self, run_id: &str, rank: i32) -> Result<Vec<i64>, Error> {
        Ok(distinct_attempts(&self.list_parts(run_id, rank).await?))
    }

    fn attempts_path(&self, run_id: &str, rank: i32) -> Path {
        self.rank_dir(run_id, rank).join(ATTEMPTS_FILE)
    }

    /// The rank's `attempts.json` (contract 1), `None` when there is none. A file that cannot be
    /// read as one is logged and taken as none: the rank is then read in numeric order, as a store
    /// written before the file existed, rather than refused for good.
    pub async fn read_attempts(&self, run_id: &str, rank: i32) -> Result<Option<AttemptOrder>, Error> {
        match self.store.get(&self.attempts_path(run_id, rank)).await {
            Ok(response) => {
                let bytes = response.bytes().await?;
                let order = serde_json::from_slice(&bytes)
                    .ok()
                    .and_then(|document| AttemptOrder::from_json(&document));
                if order.is_none() {
                    eprintln!(
                        "tensorquill-server: {ATTEMPTS_FILE} of {run_id} rank {rank} cannot be read; its attempts are taken in numeric order"
                    );
                }
                Ok(order)
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Replaces the rank's `attempts.json` with `order`.
    pub async fn write_attempts(&self, run_id: &str, rank: i32, order: &AttemptOrder) -> Result<(), Error> {
        let bytes = serde_json::to_vec_pretty(&order.to_json(run_id, rank))?;
        self.store
            .put(&self.attempts_path(run_id, rank), PutPayload::from(Bytes::from(bytes)))
            .await?;
        Ok(())
    }

    /// The order of a rank's attempts, oldest first, as every reader of the store takes it: its
    /// `attempts.json`, with any attempt stored but not in the file placed as rows are (just before
    /// the newest when that one was claimed). With no file, the stored attempts in numeric order
    /// (`kind` says `numeric`).
    pub async fn attempt_order(&self, run_id: &str, rank: i32) -> Result<AttemptOrder, Error> {
        let stored = self.stored_attempts(run_id, rank).await?;
        self.order_of(run_id, rank, stored).await
    }

    /// `attempt_order` for attempts already listed.
    async fn order_of(&self, run_id: &str, rank: i32, stored: Vec<i64>) -> Result<AttemptOrder, Error> {
        Ok(match self.read_attempts(run_id, rank).await? {
            Some(mut order) => {
                for attempt in stored {
                    order.place_rows(attempt, None, 0);
                }
                order
            }
            None => AttemptOrder::legacy(&stored, None),
        })
    }

    /// Every part of a rank, unordered.
    async fn list_parts(&self, run_id: &str, rank: i32) -> Result<Vec<StoredPart>, Error> {
        let dir = self.rank_dir(run_id, rank);
        let objects: Vec<_> = self.store.list(Some(&dir)).try_collect().await?;
        let parts: Vec<StoredPart> = objects
            .into_iter()
            .filter(|object| is_part(&object.location))
            .map(|object| StoredPart {
                // The partition directly under the rank's folder, read relative to it: a folder
                // above it (in the store's root) that looks like a partition says nothing about
                // the part (until Nov 19, 2026 the first `attempt=` anywhere in the path was taken).
                attempt: object
                    .location
                    .prefix_match(&dir)
                    .and_then(|mut below| below.next())
                    .and_then(|part| {
                        part.as_ref()
                            .strip_prefix("attempt=")
                            .and_then(|attempt| attempt.parse().ok())
                    })
                    .unwrap_or(0),
                first_step: name_min_step(&object.location),
                location: object.location,
            })
            .collect();
        Ok(parts)
    }

    /// The rows of one part, in order, with every column: what was sent, probe arrays included.
    /// Every reader but the detectors' takes this one, so a read loses nothing by default (the
    /// owner's choice, Nov 24, 2026, for Phase 4's read API); `examples/live_streams.rs` sends parts
    /// read this way again.
    pub async fn read_part(&self, location: &Path) -> Result<Vec<RecordBatch>, Error> {
        let bytes = self.store.get(location).await?.bytes().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;
        Ok(reader.collect::<Result<Vec<_>, _>>()?)
    }

    /// The rows of one part, in order, as the detectors read them: every column but the probe's
    /// heavy arrays (`NOT_REPLAYED`), which are never decoded. This is the read `detect::replay`
    /// makes after a restart and on a rebuild, and what `examples/detect_store.rs` times. A part
    /// without those columns (protocol 4's layout) is read whole. The live path is unchanged: a
    /// batch is already decoded from Flight there.
    ///
    /// `ProjectionMask::roots` (arrow-rs `parquet/src/arrow/mod.rs`) selects the part's top-level
    /// columns by position, here looked up by name in the part's own Arrow schema, so the columns
    /// kept stay in the part's order, each with every leaf under it (WAVE_AL_CONTRACT.md section 2.4).
    pub async fn read_part_for_detect(&self, location: &Path) -> Result<Vec<RecordBatch>, Error> {
        let bytes = self.store.get(location).await?.bytes().await?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let fields = builder.schema().fields();
        let builder = if fields.iter().any(|field| NOT_REPLAYED.contains(&field.name().as_str())) {
            let kept = fields
                .iter()
                .enumerate()
                .filter(|(_, field)| !NOT_REPLAYED.contains(&field.name().as_str()))
                .map(|(index, _)| index);
            let mask = ProjectionMask::roots(builder.parquet_schema(), kept);
            builder.with_projection(mask)
        } else {
            builder
        };
        Ok(builder.build()?.collect::<Result<Vec<_>, _>>()?)
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
        let bytes = parquet_bytes(batches)?;
        let path = self
            .attempt_dir(run_id, rank, attempt)
            .join(format!("part-{min_step:08}-{max_step:08}.parquet"));
        self.store.put(&path, PutPayload::from(Bytes::from(bytes))).await?;
        Ok(path)
    }

    /// Writes `batches`, scalar-only rows of one attempt whose `seq` rises from `first_seq` to
    /// `last_seq`, as `attempt=<a>/scalars-<first_seq:08>-<last_seq:08>.parquet` (`WAVE_AL_CONTRACT.md`
    /// section 4.5), and returns its path. All batches share a schema.
    ///
    /// The object is created only if there is none of that name (`PutMode::Create`: a hard link on
    /// local disk, `If-None-Match` on S3, a generation match on GCS), so a stored object is never
    /// replaced. A create that meets an existing object reads it for its `seq` range: the same range
    /// means these rows are stored (a `PUT` that reported failure after storing them, tried again
    /// under the same name), and is `Ok`; another range, or an object that is not one of these, is an
    /// error, and the caller keeps the rows. A backend that cannot create conditionally (S3 with
    /// `AWS_CONDITIONAL_PUT` set to `disabled`, on a store whose `store.json` was provisioned) is
    /// asked whether the object exists and then written plainly: the server holds one buffer per rank,
    /// so only another server on the same rank could write between the two.
    pub async fn write_scalars(
        &self,
        run_id: &str,
        rank: i32,
        attempt: i64,
        batches: &[RecordBatch],
        first_seq: i64,
        last_seq: i64,
    ) -> Result<Path, Error> {
        let bytes = Bytes::from(parquet_bytes(batches)?);
        let path = self
            .attempt_dir(run_id, rank, attempt)
            .join(format!("{SCALARS_PREFIX}{first_seq:08}-{last_seq:08}.parquet"));
        let create = PutOptions {
            mode: PutMode::Create,
            ..PutOptions::default()
        };
        let met = match self
            .store
            .put_opts(&path, PutPayload::from(bytes.clone()), create)
            .await
        {
            Ok(_) => return Ok(path),
            Err(object_store::Error::AlreadyExists { .. }) => true,
            Err(object_store::Error::NotImplemented { .. }) => match self.store.head(&path).await {
                Ok(_) => true,
                Err(object_store::Error::NotFound { .. }) => false,
                Err(error) => return Err(error.into()),
            },
            Err(error) => return Err(error.into()),
        };
        if !met {
            self.store.put(&path, PutPayload::from(bytes)).await?;
            return Ok(path);
        }
        match self.seq_range(&path).await? {
            Some(range) if range == (first_seq, last_seq) => Ok(path),
            held => Err(format!(
                "{path} already exists and holds {}, not the rows of seq {first_seq} to {last_seq}; the rows are kept and nothing is replaced",
                held.map_or_else(|| "no seq".to_string(), |(first, last)| format!("seq {first} to {last}"))
            )
            .into()),
        }
    }

    /// The lowest and highest `seq` in an object of scalar-only rows, read from its `seq` column
    /// alone; none when it has no such column or no `seq` in it.
    async fn seq_range(&self, path: &Path) -> Result<Option<(i64, i64)>, Error> {
        use arrow::array::{Array, Int64Array};
        let bytes = self.store.get(path).await?.bytes().await?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let Some(column) = leaf_index(builder.parquet_schema(), SEQ) else {
            return Ok(None);
        };
        let mask = ProjectionMask::leaves(builder.parquet_schema(), [column]);
        let mut range: Option<(i64, i64)> = None;
        for batch in builder.with_projection(mask).build()? {
            let batch = batch?;
            let Some(seqs) = batch.column(0).as_any().downcast_ref::<Int64Array>() else {
                return Ok(None);
            };
            for seq in seqs.iter().flatten() {
                range = Some(range.map_or((seq, seq), |(low, high)| (low.min(seq), high.max(seq))));
            }
        }
        Ok(range)
    }

    /// The highest step already stored for this attempt of this rank, from the `step` column statistics in
    /// the Parquet footers. Only the parts whose file name claims the highest last step are
    /// opened; a part whose name gives no step range is always opened.
    ///
    /// The parts are those [`rank_parts`](Self::rank_parts) lists and replay reads (`part-*.parquet`),
    /// and no other file: a row that only some other file holds would be below the watermark,
    /// dropped by ingest, and never read (RS09, lane V7, Nov 19, 2026). A part whose footer has
    /// no statistics for the step column is read for its step column.
    pub async fn max_step(&self, run_id: &str, rank: i32, attempt: i64) -> Result<Option<i64>, Error> {
        let dir = self.attempt_dir(run_id, rank, attempt);
        let objects: Vec<_> = self.store.list(Some(&dir)).try_collect().await?;
        let mut named: Vec<(Option<i64>, Path, u64)> = objects
            .into_iter()
            .filter(|object| is_part(&object.location))
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
            let mut complete = true;
            for row_group in metadata.row_groups() {
                match row_group.column(column).statistics() {
                    Some(Statistics::Int64(stats)) if stats.max_opt().is_some() => {
                        max = max.max(stats.max_opt().copied());
                    }
                    // No statistics, or none for this row group (all its steps null, which no
                    // writer makes, gives none too): its steps are read.
                    _ => complete = complete && row_group.num_rows() == 0,
                }
            }
            if !complete {
                max = max.max(self.highest_step(&path, column).await?);
            }
        }
        Ok(max)
    }

    /// The highest step in a part, read from its step column (leaf `column`) alone.
    async fn highest_step(&self, path: &Path, column: usize) -> Result<Option<i64>, Error> {
        use arrow::array::{Array, Int64Array};
        let bytes = self.store.get(path).await?.bytes().await?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let mask = ProjectionMask::leaves(builder.parquet_schema(), [column]);
        let mut max = None;
        for batch in builder.with_projection(mask).build()? {
            let batch = batch?;
            if let Some(steps) = batch.column(0).as_any().downcast_ref::<Int64Array>() {
                max = max.max(steps.iter().flatten().max());
            }
        }
        Ok(max)
    }
}

/// The name every object of scalar-only rows starts with (section 4.5).
pub const SCALARS_PREFIX: &str = "scalars-";

/// The column that numbers a scalar-only row within its attempt.
const SEQ: &str = "seq";

/// `batches` as the bytes of one Parquet file, compressed as every part is.
fn parquet_bytes(batches: &[RecordBatch]) -> Result<Vec<u8>, Error> {
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batches[0].schema(), Some(properties))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(bytes)
}

/// The leaf of a Parquet schema whose path is `name`, a top-level column.
fn leaf_index(schema: &parquet::schema::types::SchemaDescriptor, name: &str) -> Option<usize> {
    schema
        .columns()
        .iter()
        .position(|column| column.path().string() == name)
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

/// Whether `path` is a part: a `part-*.parquet` file, the only files `rank_parts`, `max_step`
/// and Python's `_parquet_parts` read.
/// The attempts `parts` belong to, in numeric order, each once.
fn distinct_attempts(parts: &[StoredPart]) -> Vec<i64> {
    let mut attempts: Vec<i64> = parts.iter().map(|part| part.attempt).collect();
    attempts.sort_unstable();
    attempts.dedup();
    attempts
}

fn is_part(path: &Path) -> bool {
    path.extension() == Some("parquet") && path.filename().is_some_and(|name| name.starts_with("part-"))
}

/// A part's step range as its name gives it: `part-00000000-00000049.parquet` -> `(0, 49)`,
/// and `part--0000010--0000001.parquet` -> `(-10, -1)`. The `-` between the two numbers is
/// the first one after the first number's own digits; each number may start with a sign.
fn name_steps(path: &Path) -> Option<(i64, i64)> {
    let name = path.filename()?.strip_prefix("part-")?.strip_suffix(".parquet")?;
    let at = name.get(1..)?.find('-')? + 1;
    Some((name[..at].parse().ok()?, name[at + 1..].parse().ok()?))
}

/// `part-00000000-00000049.parquet` -> `49`.
fn name_max_step(path: &Path) -> Option<i64> {
    name_steps(path).map(|(_, last)| last)
}

/// `part-00000000-00000049.parquet` -> `0`.
fn name_min_step(path: &Path) -> Option<i64> {
    name_steps(path).map(|(first, _)| first)
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
        /// A create that fails if the object exists (`PutMode::Create`) is `NotImplemented`, as on S3
        /// with conditional puts turned off.
        no_conditional_create: bool,
    }

    impl SpyStore {
        fn new(inner: Arc<dyn ObjectStore>, reject_suffix: bool) -> Self {
            Self {
                inner,
                gets: Mutex::new(Vec::new()),
                reject_suffix,
                no_conditional_create: false,
            }
        }

        fn without_conditional_create(inner: Arc<dyn ObjectStore>) -> Self {
            Self {
                no_conditional_create: true,
                ..Self::new(inner, false)
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
            if self.no_conditional_create && matches!(opts.mode, PutMode::Create) {
                return futures::future::err(object_store::Error::NotImplemented {
                    operation: "`put_opts` with mode `PutMode::Create` when conditional put is disabled".into(),
                    implementer: self.to_string(),
                })
                .boxed();
            }
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

        // Also place a part whose name gives no step range. (Until lane V7, Nov 19, 2026, this
        // was `legacy.parquet`, which replay never reads: RS09.)
        let custom_path = storage.attempt_dir("run-1", 0, 1).join("part-legacy.parquet");
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
        // Any value `object_store` reads as true allows it; the others do not.
        for yes in ["true", "TRUE", "1", "yes", "on", "y"] {
            let mut allowed = plain.clone();
            allowed.push(("AWS_ALLOW_HTTP".into(), yes.into()));
            let built = Storage::open("s3://bucket", allowed, &options);
            assert!(built.is_ok(), "{yes}: {:?}", built.err());
        }
        for no in ["false", "0", ""] {
            let mut refused = plain.clone();
            refused.push(("AWS_ALLOW_HTTP".into(), no.into()));
            assert!(Storage::open("s3://bucket", refused, &options).is_err(), "{no:?}");
        }
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

    /// L3 (lane AA3, Nov 20, 2026): servers that start on an empty store at once end with one ID,
    /// the one written first, and a store's ID is read back as written.
    #[tokio::test]
    async fn servers_starting_at_once_on_an_empty_store_end_with_one_store_id() {
        let store = Arc::new(InMemory::new());
        let servers: Vec<Storage> = (0..8)
            .map(|_| Storage::new(store.clone(), Path::from("prefix")))
            .collect();
        let ids = futures::future::try_join_all(servers.iter().map(Storage::store_id))
            .await
            .unwrap();
        assert!(ids.iter().all(|id| id == &ids[0]), "{ids:?}");
        assert_eq!(ids[0].len(), 32);
        let written: serde_json::Value = serde_json::from_slice(
            &store
                .get(&Path::from("prefix/store.json"))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(written["store_id"], ids[0]);
        assert_eq!(servers[0].store_id().await.unwrap(), ids[0]);
        // Another root in the same bucket is another store.
        let beside = Storage::new(store.clone(), Path::from("other"))
            .store_id()
            .await
            .unwrap();
        assert_ne!(beside, ids[0]);
    }

    /// L3 follow-up (author's review of the trial merge, Nov 20, 2026): a backend that cannot create
    /// `store.json` atomically writes none. Until then it wrote one plainly and read it back, so two
    /// servers starting at once could each write, each read back its own, and acknowledge two IDs for
    /// one store.
    #[tokio::test]
    async fn without_a_conditional_create_an_empty_store_gets_no_id_and_nothing_is_written() {
        let inner = Arc::new(InMemory::new());
        let spy = Arc::new(SpyStore::without_conditional_create(inner.clone()));
        let error = Storage::new(spy, Path::from("prefix"))
            .store_id()
            .await
            .unwrap_err()
            .to_string();
        for said in [
            "store.json",
            "two servers",
            r#"{"store_id": "#,
            "AWS_CONDITIONAL_PUT=etag",
        ] {
            assert!(error.contains(said), "{said:?} not in {error}");
        }
        let written: Vec<_> = inner.list(None).try_collect().await.unwrap();
        assert!(written.is_empty(), "{written:?}");
    }

    /// The same backend holding a valid `store.json` (one an operator put there): its ID is returned.
    #[tokio::test]
    async fn without_a_conditional_create_a_provisioned_store_id_is_used() {
        let inner = Arc::new(InMemory::new());
        let id = "0123456789abcdef0123456789abcdef";
        inner
            .put(
                &Path::from("prefix/store.json"),
                PutPayload::from(format!(r#"{{"store_id": "{id}"}}"#).into_bytes()),
            )
            .await
            .unwrap();
        let spy = Arc::new(SpyStore::without_conditional_create(inner.clone()));
        assert_eq!(Storage::new(spy, Path::from("prefix")).store_id().await.unwrap(), id);
    }

    /// Two servers that start at once on the empty store: neither gets an ID, and the store stays
    /// empty, so no sender is ever told two IDs for it.
    #[tokio::test]
    async fn without_a_conditional_create_two_servers_on_an_empty_store_get_no_id() {
        let inner = Arc::new(InMemory::new());
        let servers: Vec<Storage> = (0..2)
            .map(|_| {
                Storage::new(
                    Arc::new(SpyStore::without_conditional_create(inner.clone())),
                    Path::from("prefix"),
                )
            })
            .collect();
        let (a, b) = futures::join!(servers[0].store_id(), servers[1].store_id());
        assert!(a.is_err() && b.is_err(), "{a:?} {b:?}");
        let written: Vec<_> = inner.list(None).try_collect().await.unwrap();
        assert!(written.is_empty(), "{written:?}");
    }

    /// A `store.json` that names no ID is an error, and is left as it is: replacing it would give
    /// the store a new ID and make every sender start its file again.
    #[tokio::test]
    async fn a_store_file_that_names_no_id_is_an_error_and_is_not_replaced() {
        let store = Arc::new(InMemory::new());
        let path = Path::from("store.json");
        for bad in [&b"not json"[..], br#"{"store_id": 7}"#, br#"{"store_id": "a/b"}"#] {
            store.put(&path, PutPayload::from(bad.to_vec())).await.unwrap();
            let error = Storage::new(store.clone(), Path::from(""))
                .store_id()
                .await
                .unwrap_err();
            assert!(error.to_string().contains("store.json"), "{error}");
            assert_eq!(store.get(&path).await.unwrap().bytes().await.unwrap().as_ref(), bad);
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

    /// Writes `batch` at `path` as a Parquet file, with or without the footer's statistics.
    async fn put_parquet(store: &dyn ObjectStore, path: &Path, batch: &RecordBatch, statistics: bool) {
        use parquet::file::properties::EnabledStatistics;
        let properties = WriterProperties::builder()
            .set_statistics_enabled(if statistics {
                EnabledStatistics::Chunk
            } else {
                EnabledStatistics::None
            })
            .build();
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(properties)).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        store.put(path, PutPayload::from(Bytes::from(buf))).await.unwrap();
    }

    #[tokio::test]
    async fn rs09_the_watermark_counts_the_parts_replay_reads_and_no_other_file() {
        // Lane V7, Nov 19, 2026. A named part holds steps 0-1 and a file this server never
        // names holds 2-3. Replay reads only the part, so the watermark is 1: with 3, ingest
        // would drop the rows of steps 2 and 3 that replay never sees.
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let storage = Storage::new(store.clone(), Path::from("root"));
        storage
            .write_part("run-1", 0, 1, &[make_batch(0..2, 4)], 0, 1)
            .await
            .unwrap();
        let foreign = storage.attempt_dir("run-1", 0, 1).join("legacy.parquet");
        put_parquet(store.as_ref(), &foreign, &make_batch(2..4, 4), true).await;
        let parts = storage.rank_parts("run-1", 0).await.unwrap();
        assert_eq!(parts.len(), 1, "{parts:?}");
        assert_eq!(storage.max_step("run-1", 0, 1).await.unwrap(), Some(1));
    }

    #[tokio::test]
    async fn rs09_a_part_without_footer_statistics_is_read_by_its_step_column() {
        // Lane V7, Nov 19, 2026. A part named as this server names them, written without
        // statistics: the watermark is its highest step, not none.
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let storage = Storage::new(store.clone(), Path::from("root"));
        let path = storage
            .attempt_dir("run-1", 0, 1)
            .join("part-00000000-00000009.parquet");
        put_parquet(store.as_ref(), &path, &make_batch(0..10, 4), false).await;
        assert_eq!(storage.max_step("run-1", 0, 1).await.unwrap(), Some(9));
        // A part whose name gives no step range, also without statistics.
        let path = storage.attempt_dir("run-1", 0, 2).join("part-x.parquet");
        put_parquet(store.as_ref(), &path, &make_batch(5..8, 4), false).await;
        assert_eq!(storage.max_step("run-1", 0, 2).await.unwrap(), Some(7));
    }

    #[tokio::test]
    async fn rust14_negative_steps_round_trip_through_a_parts_name() {
        // Lane V7, Nov 19, 2026. `part--0000010--0000001.parquet` is steps -10 to -1.
        for (name, range) in [
            ("part--0000010--0000001.parquet", (Some(-10), Some(-1))),
            ("part--0000010-00000005.parquet", (Some(-10), Some(5))),
            ("part-00000000-00000049.parquet", (Some(0), Some(49))),
            ("part-00000000.parquet", (None, None)),
            ("part-x-y.parquet", (None, None)),
        ] {
            let path = Path::from(format!("a/{name}"));
            assert_eq!((name_min_step(&path), name_max_step(&path)), range, "{name}");
        }
        let storage = Storage::new(Arc::new(InMemory::new()), Path::from("root"));
        storage
            .write_part("run-1", 0, 1, &[make_batch(-10..0, 4)], -10, -1)
            .await
            .unwrap();
        let parts = storage.rank_parts("run-1", 0).await.unwrap();
        assert_eq!(parts[0].first_step, Some(-10), "{parts:?}");
        assert_eq!(storage.max_step("run-1", 0, 1).await.unwrap(), Some(-1));
    }

    #[tokio::test]
    async fn a_parts_attempt_is_the_partition_under_its_rank_not_one_in_the_store_root() {
        // RS03 (lane U6, Nov 19, 2026). A store whose root has partition-like folders of its own:
        // each part's attempt is the folder directly under its rank's, and nothing above it.
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let storage = Storage::new(store.clone(), Path::from("data/attempt=7/rank=3"));
        storage
            .write_part("prefix", 0, 1, &[make_batch(0..100, 4)], 0, 99)
            .await
            .unwrap();
        storage
            .write_part("prefix", 0, 2, &[make_batch(50..60, 4)], 50, 59)
            .await
            .unwrap();
        // A part straight in the rank's folder, not in an attempt's: attempt 0, as always.
        let loose = Path::from("data/attempt=7/rank=3/runs/prefix/rank=0/part-00000000-00000001.parquet");
        store.put(&loose, PutPayload::from_static(b"x")).await.unwrap();
        let parts = storage.rank_parts("prefix", 0).await.unwrap();
        let read: Vec<(i64, Option<i64>)> = parts.iter().map(|part| (part.attempt, part.first_step)).collect();
        assert_eq!(read, [(0, Some(0)), (1, Some(0)), (2, Some(50))], "{parts:?}");
    }

    // Contract 1 (RS05, decided Nov 19, 2026): the order of a rank's attempts.

    #[test]
    fn w2_a_claim_is_newest_and_rows_go_before_the_attempt_that_sent_them() {
        let mut order = AttemptOrder::default();
        assert!(order.place_claim(900, 1));
        // B claims after A, with a smaller number: it is the newest.
        assert!(order.place_claim(2, 2));
        assert!(!order.place_claim(900, 3), "an attempt is placed once and never moves");
        assert_eq!(order.attempts(), [900, 2]);
        assert_eq!(order.newest(), Some(2));
        // Rows of an attempt never seen, sent by B's process: a backlog, older than B.
        assert!(order.place_rows(500, Some(2), 4));
        assert_eq!(order.attempts(), [900, 500, 2]);
        // Rows with no sender named: only the number (the session's start time) is left. A lower
        // one than the newest's is a backlog, just before it; a higher one is the job started
        // again, whose first beat has not arrived, after it.
        assert!(order.place_rows(1, None, 5));
        assert_eq!(order.attempts(), [900, 500, 1, 2]);
        assert!(order.place_rows(700, None, 6));
        assert_eq!(order.attempts(), [900, 500, 1, 2, 700]);
        // A sender from before claims: its attempts are known from rows only, by number against the newest.
        let mut rows_only = AttemptOrder::default();
        for attempt in [5, 3, 9] {
            rows_only.place_rows(attempt, None, 0);
        }
        assert_eq!(rows_only.attempts(), [3, 5, 9]);
    }

    #[test]
    fn w2_a_store_without_an_order_is_numeric_with_the_restored_owner_last() {
        let order = AttemptOrder::legacy(&[1000, 900, 3], Some(900));
        assert_eq!(order.attempts(), [3, 1000, 900]);
        assert_eq!(order.kind(), "numeric");
        let mut extended = order.clone();
        extended.place_claim(5, 0);
        extended.on_disk = true;
        assert_eq!(extended.attempts(), [3, 1000, 900, 5]);
        assert_eq!(extended.kind(), "numeric_then_first_seen");
        let mut fresh = AttemptOrder::default();
        fresh.place_claim(5, 0);
        fresh.on_disk = true;
        assert_eq!(fresh.kind(), "first_seen");
        // The file round-trips; a file that is not one is none.
        let back = AttemptOrder::from_json(&extended.to_json("r", 0)).unwrap();
        assert_eq!(back, extended);
        assert!(AttemptOrder::from_json(&serde_json::json!({"schema_version": 2, "attempts": []})).is_none());
        assert!(
            AttemptOrder::from_json(&serde_json::json!({"schema_version": 1, "attempts": [
            {"attempt": 1, "by": "claim"}, {"attempt": 1, "by": "rows"}]}))
            .is_none()
        );
    }

    #[tokio::test]
    async fn w2_parts_are_listed_in_the_ranks_order_of_attempts_not_by_number() {
        let storage = Storage::new(Arc::new(InMemory::new()), Path::from("s"));
        storage
            .write_part("r", 0, 900, &[make_batch(0..101, 8)], 0, 100)
            .await
            .unwrap();
        storage
            .write_part("r", 0, 2, &[make_batch(30..31, 8)], 30, 30)
            .await
            .unwrap();
        storage
            .write_part("r", 0, 900, &[make_batch(101..102, 8)], 101, 101)
            .await
            .unwrap();
        let read = |parts: Vec<StoredPart>| {
            parts
                .iter()
                .map(|part| (part.attempt, part.first_step))
                .collect::<Vec<_>>()
        };
        // No order kept: by number, as a store written before contract 1.
        assert_eq!(storage.attempt_order("r", 0).await.unwrap().kind(), "numeric");
        assert_eq!(
            read(storage.rank_parts("r", 0).await.unwrap()),
            [(2, Some(30)), (900, Some(0)), (900, Some(101))]
        );
        let mut order = AttemptOrder::default();
        order.place_claim(900, 1);
        order.place_claim(2, 2);
        storage.write_attempts("r", 0, &order).await.unwrap();
        assert_eq!(
            storage.read_attempts("r", 0).await.unwrap().unwrap().attempts(),
            [900, 2]
        );
        assert_eq!(
            read(storage.rank_parts("r", 0).await.unwrap()),
            [(900, Some(0)), (900, Some(101)), (2, Some(30))]
        );
        assert_eq!(storage.attempt_order("r", 0).await.unwrap().kind(), "first_seen");
        // An attempt stored but missing from the file is placed as rows with no sender are.
        storage
            .write_part("r", 0, 1, &[make_batch(0..5, 8)], 0, 4)
            .await
            .unwrap();
        assert_eq!(storage.attempt_order("r", 0).await.unwrap().attempts(), [900, 1, 2]);
    }
}
