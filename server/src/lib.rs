//! `tensorlens-server`: receives each rank's step records over Arrow Flight `DoPut`, drops
//! rows it already has for `(run_id, rank, step)`, and writes immutable Zstd Parquet parts
//! through `object_store`.

pub mod ingest;
pub mod server;
pub mod storage;
