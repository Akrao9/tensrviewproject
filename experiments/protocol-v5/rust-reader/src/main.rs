//! WAVE_AL_CONTRACT.md's Rust-side measurements. Not product code.
//!
//! `part-reader keys <store> <part> <column>`: reads `<part>` whole with the server's own
//! `Storage::read_part` (every column, `probe_maps` and `probe_histograms` included) and prints,
//! for the first row, the keys of the map column `<column>` in stored order, as JSON `[[...]]`. With
//! `<column>` written `map.field` it prints, for the first entry of the first row, the struct field
//! `field` looked up by name, or `"absent"` when the part's struct has no such field.
//!
//! `part-reader bench <rounds> <part>...`: for every part, the time to read it with
//! `Storage::read_part_for_detect` (the read `detect::replay` makes: it leaves out `probe_maps`
//! and `probe_histograms`, and reads a part without them whole), and the time to decode what the detectors read from its layer and router
//! statistics: from `layers_json` / `moe_stats_json` with a copy of `detect.rs`' own visitors
//! (`Members<LayerStats>`, `Members<RouterStats>`), or from the prototype's typed `layers` /
//! `moe_stats` maps (protocol 5's layout); and of `probe_json` skipped (`IgnoredAny`, what the probe visitor pays to pass the
//! maps) and read whole (`Value`, what `HeadTrends` pays on a probe with a head). Prints one JSON
//! object with the median and the spread over `rounds`.

use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;

use arrow::array::{Array, Float64Array, Int64Array, MapArray, StringArray, StructArray};
use arrow::record_batch::RecordBatch;
use serde::de::{Deserialize, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use tensorquill_server::storage::Storage;

// --- copied from server/src/detect.rs at f71d8e0 (Key, Members, Number, Stat, LayerStats, RouterStats) ---

struct Members<'a, T>(Vec<(Cow<'a, str>, T)>);
struct Key<'a>(Cow<'a, str>);

impl<'de: 'a, 'a> Deserialize<'de> for Key<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = Key<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_borrowed_str<E>(self, text: &'de str) -> Result<Self::Value, E> {
                Ok(Key(Cow::Borrowed(text)))
            }
            fn visit_str<E>(self, text: &str) -> Result<Self::Value, E> {
                Ok(Key(Cow::Owned(text.to_owned())))
            }
        }
        deserializer.deserialize_str(KeyVisitor)
    }
}

impl<'de: 'a, 'a, T: Deserialize<'de>> Deserialize<'de> for Members<'a, T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MembersVisitor<'a, T>(PhantomData<(&'a (), T)>);
        impl<'de: 'a, 'a, T: Deserialize<'de>> Visitor<'de> for MembersVisitor<'a, T> {
            type Value = Members<'a, T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut members = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(Key(key)) = map.next_key()? {
                    members.push((key, map.next_value()?));
                }
                Ok(Members(members))
            }
        }
        deserializer.deserialize_map(MembersVisitor(PhantomData))
    }
}

#[derive(Default)]
struct Number(Option<f64>);

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NumberVisitor;
        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = Number;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_f64<E>(self, number: f64) -> Result<Number, E> {
                Ok(Number(Some(number).filter(|number| number.is_finite())))
            }
            fn visit_i64<E>(self, number: i64) -> Result<Number, E> {
                Ok(Number(Some(number as f64)))
            }
            fn visit_u64<E>(self, number: u64) -> Result<Number, E> {
                Ok(Number(Some(number as f64)))
            }
            fn visit_bool<E>(self, _: bool) -> Result<Number, E> {
                Ok(Number(None))
            }
            fn visit_str<E>(self, _: &str) -> Result<Number, E> {
                Ok(Number(None))
            }
            fn visit_unit<E>(self) -> Result<Number, E> {
                Ok(Number(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Number, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Number(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Number, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Number(None))
            }
        }
        deserializer.deserialize_any(NumberVisitor)
    }
}

struct Stat {
    number: Option<f64>,
    non_finite: bool,
}

impl Stat {
    fn of(number: f64) -> Self {
        Self { number: Some(number).filter(|n| n.is_finite()), non_finite: !number.is_finite() }
    }
    const ABSENT: Self = Self { number: None, non_finite: false };
}

impl<'de> Deserialize<'de> for Stat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StatVisitor;
        impl<'de> Visitor<'de> for StatVisitor {
            type Value = Stat;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_f64<E>(self, number: f64) -> Result<Stat, E> {
                Ok(Stat::of(number))
            }
            fn visit_i64<E>(self, number: i64) -> Result<Stat, E> {
                Ok(Stat::of(number as f64))
            }
            fn visit_u64<E>(self, number: u64) -> Result<Stat, E> {
                Ok(Stat::of(number as f64))
            }
            fn visit_bool<E>(self, _: bool) -> Result<Stat, E> {
                Ok(Stat::ABSENT)
            }
            fn visit_str<E>(self, text: &str) -> Result<Stat, E> {
                Ok(Stat { number: None, non_finite: matches!(text, "nan" | "inf" | "-inf") })
            }
            fn visit_unit<E>(self) -> Result<Stat, E> {
                Ok(Stat::ABSENT)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Stat, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Stat::ABSENT)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Stat, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Stat::ABSENT)
            }
        }
        deserializer.deserialize_any(StatVisitor)
    }
}

#[derive(Default)]
struct LayerStats {
    std: Option<f64>,
    abs_max: Option<f64>,
    zero_frac: Option<f64>,
    non_finite: bool,
}

impl<'de> Deserialize<'de> for LayerStats {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LayerVisitor;
        impl<'de> Visitor<'de> for LayerVisitor {
            type Value = LayerStats;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<LayerStats, A::Error> {
                let mut stats = LayerStats::default();
                while let Some(Key(key)) = map.next_key()? {
                    let stat: Stat = map.next_value()?;
                    match key.as_ref() {
                        "std" => stats.std = stat.number,
                        "abs_max" => stats.abs_max = stat.number,
                        "zero_frac" => stats.zero_frac = stat.number,
                        _ => {}
                    }
                    stats.non_finite |= stat.non_finite;
                }
                Ok(stats)
            }
        }
        deserializer.deserialize_map(LayerVisitor)
    }
}

#[derive(serde::Deserialize)]
struct RouterStats {
    #[serde(default)]
    diversity: Number,
}

// --- the prototype's typed columns, read as the detectors would read them -----------------------

/// What the activation detectors take from one row of a typed `layers` map: the same tuple the
/// JSON visitor gives, for every entry. A null entry or field reads as absent.
fn typed_layers(map: &MapArray, row: usize, out: &mut Vec<(usize, Option<f64>, Option<f64>, Option<f64>, bool)>) {
    out.clear();
    if map.is_null(row) {
        return;
    }
    let entries: &StructArray = map.values().as_any().downcast_ref().expect("struct values");
    let field = |name: &str| entries.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Float64Array>());
    let (mean, std, abs_max, zero_frac) = (field("mean"), field("std"), field("abs_max"), field("zero_frac"));
    let get = |column: Option<&Float64Array>, at: usize| column.filter(|c| !c.is_null(at)).map(|c| c.value(at));
    let offsets = map.value_offsets();
    for at in offsets[row] as usize..offsets[row + 1] as usize {
        if entries.is_null(at) {
            out.push((at, None, None, None, false));
            continue;
        }
        let values = [get(mean, at), get(std, at), get(abs_max, at), get(zero_frac, at)];
        let non_finite = values.iter().flatten().any(|v| !v.is_finite());
        let finite = |v: Option<f64>| v.filter(|v| v.is_finite());
        out.push((at, finite(values[1]), finite(values[2]), finite(values[3]), non_finite));
    }
}

fn typed_diversity(map: &MapArray, row: usize, out: &mut Vec<Option<f64>>) {
    out.clear();
    if map.is_null(row) {
        return;
    }
    let entries: &StructArray = map.values().as_any().downcast_ref().expect("struct values");
    let diversity = entries.column_by_name("diversity").and_then(|c| c.as_any().downcast_ref::<Float64Array>());
    let offsets = map.value_offsets();
    for at in offsets[row] as usize..offsets[row + 1] as usize {
        out.push(diversity.filter(|c| !c.is_null(at) && !entries.is_null(at)).map(|c| c.value(at)).filter(|v| v.is_finite()));
    }
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Option<&'a T> {
    batch.column_by_name(name)?.as_any().downcast_ref()
}

#[derive(Default)]
struct Timings {
    read: Vec<f64>,
    layers: Vec<f64>,
    moe_stats: Vec<f64>,
    probe_skip: Vec<f64>,
    probe_value: Vec<f64>,
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn spread(values: &mut [f64]) -> Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        return Value::Null;
    }
    let median = if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 };
    serde_json::json!({"median_ms": median, "min_ms": values[0], "max_ms": values[n - 1], "rounds": n})
}

async fn bench(rounds: usize, paths: &[String]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let storage = Storage::from_url("/")?;
    let locations: Vec<object_store::path::Path> = paths
        .iter()
        .map(|p| object_store::path::Path::from_absolute_path(std::path::absolute(p).unwrap()).unwrap())
        .collect();
    let mut timings = Timings::default();
    let (mut sink_layers, mut sink_div): (Vec<(usize, Option<f64>, Option<f64>, Option<f64>, bool)>, Vec<Option<f64>>) =
        (Vec::new(), Vec::new());
    let mut checksum = 0.0f64;
    let mut rows = 0usize;
    for _ in 0..rounds {
        let (mut read, mut layers, mut moe, mut skip, mut whole) = (0.0, 0.0, 0.0, 0.0, 0.0);
        rows = 0;
        for location in &locations {
            let started = Instant::now();
            let batches = storage.read_part_for_detect(location).await?;
            read += ms(started);
            for batch in &batches {
                rows += batch.num_rows();
                // layers
                let started = Instant::now();
                if let Some(text) = column::<StringArray>(batch, "layers_json") {
                    for row in 0..batch.num_rows() {
                        if text.is_null(row) {
                            continue;
                        }
                        if let Ok(Members(parsed)) = serde_json::from_str::<Members<LayerStats>>(text.value(row)) {
                            for (_, s) in &parsed {
                                checksum += s.std.unwrap_or(0.0) + f64::from(u8::from(s.non_finite));
                            }
                        }
                    }
                } else if let Some(map) = column::<MapArray>(batch, "layers") {
                    for row in 0..batch.num_rows() {
                        typed_layers(map, row, &mut sink_layers);
                        for (_, std, _, _, bad) in &sink_layers {
                            checksum += std.unwrap_or(0.0) + f64::from(u8::from(*bad));
                        }
                    }
                }
                layers += ms(started);
                // moe_stats
                let started = Instant::now();
                if let Some(text) = column::<StringArray>(batch, "moe_stats_json") {
                    for row in 0..batch.num_rows() {
                        if text.is_null(row) {
                            continue;
                        }
                        if let Ok(Members(parsed)) = serde_json::from_str::<Members<RouterStats>>(text.value(row)) {
                            for (_, s) in &parsed {
                                checksum += s.diversity.0.unwrap_or(0.0);
                            }
                        }
                    }
                } else if let Some(map) = column::<MapArray>(batch, "moe_stats") {
                    for row in 0..batch.num_rows() {
                        typed_diversity(map, row, &mut sink_div);
                        checksum += sink_div.iter().flatten().sum::<f64>();
                    }
                }
                moe += ms(started);
                // probe_json
                if let Some(text) = column::<StringArray>(batch, "probe_json") {
                    for row in 0..batch.num_rows() {
                        if text.is_null(row) {
                            continue;
                        }
                        let started = Instant::now();
                        let _: IgnoredAny = serde_json::from_str(text.value(row))?;
                        skip += ms(started);
                        let started = Instant::now();
                        let value: Value = serde_json::from_str(text.value(row))?;
                        whole += ms(started);
                        checksum += f64::from(u8::from(value.is_object()));
                    }
                }
            }
        }
        timings.read.push(read);
        timings.layers.push(layers);
        timings.moe_stats.push(moe);
        timings.probe_skip.push(skip);
        timings.probe_value.push(whole);
    }
    let _ = column::<Int64Array>;
    println!(
        "{}",
        serde_json::json!({
            "parts": paths.len(), "rows": rows, "checksum": checksum,
            "read_part": spread(&mut timings.read),
            "layers_decode": spread(&mut timings.layers),
            "moe_stats_decode": spread(&mut timings.moe_stats),
            "probe_json_skipped": spread(&mut timings.probe_skip),
            "probe_json_as_value": spread(&mut timings.probe_value),
        })
    );
    Ok(())
}

/// `part-reader write <rounds> <part>...`: each part read once, then written to memory `rounds`
/// times with the WriterProperties of `Storage::write_part` (Zstd at its default level), as the server
/// writes a part before it acknowledges it. Prints the median, spread and bytes per part. The part is
/// read whole (`read_part`): the server writes the probe's maps and histograms with the rest.
async fn write(rounds: usize, paths: &[String]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::WriterProperties;
    let storage = Storage::from_url("/")?;
    let mut out = Vec::new();
    for path in paths {
        let location = object_store::path::Path::from_absolute_path(std::path::absolute(path)?)?;
        let batches = storage.read_part(&location).await?;
        let mut times = Vec::new();
        let mut size = 0;
        for _ in 0..rounds {
            let started = Instant::now();
            let properties = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::default())).build();
            let mut bytes = Vec::new();
            let mut writer = ArrowWriter::try_new(&mut bytes, batches[0].schema(), Some(properties))?;
            for batch in &batches {
                writer.write(batch)?;
            }
            writer.close()?;
            times.push(ms(started));
            size = bytes.len();
        }
        let arrow_bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
        out.push(serde_json::json!({"part": path, "bytes": size, "arrow_bytes": arrow_bytes, "write": spread(&mut times)}));
    }
    println!("{}", Value::Array(out));
    Ok(())
}

async fn keys(part: &str, column_name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let storage = Storage::from_url("/")?;
    let location = object_store::path::Path::from_absolute_path(std::path::absolute(part)?)?;
    let batches = storage.read_part(&location).await?;
    let batch = &batches[0];
    let (name, field) = match column_name.split_once('.') {
        Some((name, field)) => (name, Some(field)),
        None => (column_name, None),
    };
    let map: &MapArray = column(batch, name).ok_or("no such map column")?;
    let offsets = map.value_offsets();
    let range = offsets[0] as usize..offsets[1] as usize;
    if let Some(field) = field {
        let entries: &StructArray = map.values().as_any().downcast_ref().ok_or("not a struct map")?;
        let value = match entries.column_by_name(field) {
            None => Value::String("absent".into()),
            Some(values) => match values.as_any().downcast_ref::<Float64Array>() {
                Some(values) if !values.is_null(range.start) => serde_json::json!(values.value(range.start)),
                _ => Value::Null,
            },
        };
        println!("{value}");
        return Ok(());
    }
    let keys: &StringArray = map.keys().as_any().downcast_ref().ok_or("keys not utf8")?;
    let first: Vec<&str> = range.map(|at| keys.value(at)).collect();
    println!("{}", serde_json::json!([first]));
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("keys") if args.len() == 5 => keys(&args[3], &args[4]).await,
        Some("bench") if args.len() >= 4 => bench(args[2].parse()?, &args[3..]).await,
        Some("write") if args.len() >= 4 => write(args[2].parse()?, &args[3..]).await,
        _ => Err("usage: part-reader keys <store> <part> <column> | part-reader bench <rounds> <part>...".into()),
    }
}
