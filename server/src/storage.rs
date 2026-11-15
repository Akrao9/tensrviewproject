//! Parquet part files under `{store_root}/runs/{run_id}/rank={rank}/`.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;

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

    fn rank_dir(&self, run_id: &str, rank: i32) -> Path {
        self.root.clone().join("runs").join(run_id).join(format!("rank={rank}"))
    }

    /// Writes `batches` as one immutable part and returns its path. All batches share a schema.
    pub async fn write_part(
        &self,
        run_id: &str,
        rank: i32,
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
            .rank_dir(run_id, rank)
            .join(format!("part-{min_step:08}-{max_step:08}.parquet"));
        self.store.put(&path, PutPayload::from(Bytes::from(bytes))).await?;
        Ok(path)
    }

    /// The highest step already stored for this rank, from the `step` column statistics in
    /// the Parquet footers. Only the parts whose file name claims the highest last step are
    /// opened; a part with a name this server did not write is always opened.
    pub async fn max_step(&self, run_id: &str, rank: i32) -> Result<Option<i64>, Error> {
        let dir = self.rank_dir(run_id, rank);
        let objects: Vec<_> = self.store.list(Some(&dir)).try_collect().await?;
        let mut named: Vec<(Option<i64>, Path)> = objects
            .into_iter()
            .filter(|object| object.location.extension() == Some("parquet"))
            .map(|object| (name_max_step(&object.location), object.location))
            .collect();
        let newest = named.iter().filter_map(|(step, _)| *step).max();
        named.retain(|(step, _)| step.is_none() || *step == newest);
        let mut max = None;
        for (_, path) in named {
            let bytes = self.store.get(&path).await?.bytes().await?;
            let metadata = ParquetMetaDataReader::new().parse_and_finish(&bytes)?;
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

/// `part-00000000-00000049.parquet` -> `49`.
fn name_max_step(path: &Path) -> Option<i64> {
    let name = path.filename()?.strip_prefix("part-")?.strip_suffix(".parquet")?;
    // Steps are not negative in practice, so the last `-` separates the two numbers.
    name.rsplit_once('-')?.1.parse().ok()
}
