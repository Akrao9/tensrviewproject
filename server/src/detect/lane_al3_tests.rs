//! Lane AL-3's tests (wave AL, step 1): the detectors read protocol 5's typed `layers` and
//! `moe_stats` columns (`WAVE_AL_CONTRACT.md` sections 1.2 to 1.5), per batch, where a batch has
//! them, and `layers_json` and `moe_stats_json` where it does not; they never read `probe_maps` or
//! `probe_histograms` (section 2.4); and `check_keys` takes row `schema_version` 2 (section 3).
//!
//! Each test builds the same records twice, once in protocol 4's layout (the JSON columns) and once
//! in protocol 5's (the typed maps, with what a column cannot hold in `extra_json` as section 1.4
//! lays it out), and compares the reports. A record no capture writes (an entry that is not an
//! object, a name Arrow cannot encode) is compared with its clean record instead: the typed row
//! reads the rest of the map, where the JSON row today loses every layer. The same cases run
//! through the binary from Python in `sdk/tests/test_lane_al3_typed_detect.py`.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, Int64Builder, ListBuilder, MapArray, MapBuilder,
    StringArray, StringBuilder, StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::{json, Map, Value};

use super::{detect_stream, detect_stream_samples, judge, report, Thresholds};

/// One entry of a struct map as a record holds it: an object's fields in the record's order, or a
/// value that is not an object.
#[derive(Clone, Debug)]
enum Entry {
    Object(Vec<(&'static str, Value)>),
    Other(Value),
}

/// One record, of the keys these tests vary. `layers_other` is a `layers` value that is not an
/// object; `extra` is a row's `extra_json` text written by hand (it then replaces what the layout
/// would put there).
#[derive(Clone, Debug, Default)]
struct Rec {
    step: i64,
    layers: Option<Vec<(String, Entry)>>,
    layers_other: Option<Value>,
    routers: Option<Vec<(String, Vec<i64>)>>,
    moe_stats: Option<Vec<(String, Entry)>>,
    extra: Option<&'static str>,
    probe: Option<String>,
}

/// Section 1.1's fields, in the order the capture writes them; `true` for an Int64 field.
const LAYER: &[(&str, bool)] = &[
    ("mean", false),
    ("std", false),
    ("abs_max", false),
    ("zero_frac", false),
];
const MOE: &[(&str, bool)] = &[
    ("routed_tokens", true),
    ("entropy", false),
    ("diversity", false),
    ("dead_experts", true),
    ("dead_frac", false),
];

#[derive(Clone, Copy)]
enum Layout {
    /// Protocol 4: `layers_json` and `moe_stats_json`, row version 1.
    Json,
    /// Protocol 5: the typed maps with these struct fields, row version 2.
    Typed {
        layers: &'static [(&'static str, bool)],
        moe: &'static [(&'static str, bool)],
    },
}

const TYPED: Layout = Layout::Typed {
    layers: LAYER,
    moe: MOE,
};

/// A value a struct field of this type holds (section 1.4): for Float64 a number that is not a bool,
/// or one of the sink's three strings; for Int64 an integer that fits.
fn held(value: &Value, integer: bool) -> Option<f64> {
    match value {
        Value::Number(number) if integer => number.as_i64().map(|n| n as f64),
        Value::Number(number) => number.as_f64(),
        Value::String(text) if !integer => match text.as_str() {
            "nan" => Some(f64::NAN),
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

/// A typed struct map over `rows`, as section 1.4 lays a record out: what a field cannot hold goes
/// to `extras[row][key]`, an entry that is not an object is a null struct in its place.
fn typed_map(
    rows: &[Option<Vec<(String, Entry)>>],
    fields: &[(&str, bool)],
    key: &str,
    extras: &mut [Map<String, Value>],
) -> ArrayRef {
    let mut names: Vec<String> = Vec::new();
    let mut valid: Vec<bool> = Vec::new();
    let mut values: Vec<Vec<Option<f64>>> = vec![Vec::new(); fields.len()];
    let mut offsets: Vec<i32> = vec![0];
    for (at, row) in rows.iter().enumerate() {
        let mut refused = Map::new();
        for (name, entry) in row.iter().flatten() {
            names.push(name.clone());
            match entry {
                Entry::Other(value) => {
                    valid.push(false);
                    values.iter_mut().for_each(|column| column.push(None));
                    refused.insert(name.clone(), value.clone());
                }
                Entry::Object(given) => {
                    valid.push(true);
                    let mut bad = Map::new();
                    for (field, value) in given {
                        let kept = fields
                            .iter()
                            .find(|(name, _)| name == field)
                            .and_then(|(_, integer)| held(value, *integer));
                        if kept.is_none() {
                            bad.insert((*field).to_owned(), value.clone());
                        }
                    }
                    for (column, (field, integer)) in values.iter_mut().zip(fields) {
                        let value = given.iter().find(|(name, _)| name == field).map(|(_, value)| value);
                        column.push(value.and_then(|value| held(value, *integer)));
                    }
                    if !bad.is_empty() {
                        refused.insert(name.clone(), Value::Object(bad));
                    }
                }
            }
        }
        offsets.push(names.len() as i32);
        if !refused.is_empty() {
            extras[at].insert(key.to_owned(), Value::Object(refused));
        }
    }
    let struct_fields: Fields = fields
        .iter()
        .map(|(name, integer)| {
            let data_type = if *integer { DataType::Int64 } else { DataType::Float64 };
            Field::new(*name, data_type, true)
        })
        .collect();
    let arrays: Vec<ArrayRef> = values
        .into_iter()
        .zip(fields)
        .map(|(column, (_, integer))| -> ArrayRef {
            if *integer {
                Arc::new(column.into_iter().map(|v| v.map(|v| v as i64)).collect::<Int64Array>())
            } else {
                Arc::new(column.into_iter().collect::<Float64Array>())
            }
        })
        .collect();
    let structs = StructArray::new(struct_fields, arrays, Some(NullBuffer::from(valid)));
    let entries = StructArray::new(
        Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", structs.data_type().clone(), true),
        ]),
        vec![Arc::new(StringArray::from(names)), Arc::new(structs)],
        None,
    );
    let row_valid: Vec<bool> = rows.iter().map(Option::is_some).collect();
    Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(offsets.into()),
        entries,
        Some(NullBuffer::from(row_valid)),
        false,
    ))
}

/// A struct map's entries as protocol 4's JSON text, in the record's order.
fn json_text(entries: &[(String, Entry)]) -> String {
    let entry = |entry: &Entry| match entry {
        Entry::Object(fields) => {
            let fields: Vec<String> = fields
                .iter()
                .map(|(name, value)| format!("{}:{value}", Value::from(*name)))
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Entry::Other(value) => value.to_string(),
    };
    let members: Vec<String> = entries
        .iter()
        .map(|(name, value)| format!("{}:{}", Value::from(name.as_str()), entry(value)))
        .collect();
    format!("{{{}}}", members.join(","))
}

fn batch(recs: &[Rec], layout: Layout) -> RecordBatch {
    let mut extras: Vec<Map<String, Value>> = vec![Map::new(); recs.len()];
    let version = if matches!(layout, Layout::Json) { 1 } else { 2 };
    let mut columns: Vec<(&str, ArrayRef)> = vec![
        (
            "run_id",
            Arc::new(StringArray::from_iter_values(recs.iter().map(|_| "run"))),
        ),
        ("rank", Arc::new(Int32Array::from_iter_values(recs.iter().map(|_| 0)))),
        (
            "attempt",
            Arc::new(Int64Array::from_iter_values(recs.iter().map(|_| 1))),
        ),
        (
            "step",
            Arc::new(Int64Array::from_iter_values(recs.iter().map(|r| r.step))),
        ),
        (
            "schema_version",
            Arc::new(Int32Array::from_iter_values(recs.iter().map(|_| version))),
        ),
        (
            "grad_norm",
            Arc::new(Float64Array::from_iter_values(recs.iter().map(|_| 1.0))),
        ),
    ];
    match layout {
        Layout::Json => {
            let layers: StringArray = recs
                .iter()
                .map(|r| {
                    r.layers
                        .as_deref()
                        .map(json_text)
                        .or_else(|| r.layers_other.as_ref().map(Value::to_string))
                })
                .collect();
            let moe: StringArray = recs.iter().map(|r| r.moe_stats.as_deref().map(json_text)).collect();
            columns.push(("layers_json", Arc::new(layers)));
            columns.push(("moe_stats_json", Arc::new(moe)));
        }
        Layout::Typed { layers, moe } => {
            for (at, rec) in recs.iter().enumerate() {
                if let Some(other) = &rec.layers_other {
                    extras[at].insert("layers".to_owned(), other.clone());
                }
            }
            let rows: Vec<_> = recs.iter().map(|r| r.layers.clone()).collect();
            columns.push(("layers", typed_map(&rows, layers, "layers", &mut extras)));
            let rows: Vec<_> = recs.iter().map(|r| r.moe_stats.clone()).collect();
            columns.push(("moe_stats", typed_map(&rows, moe, "moe_stats", &mut extras)));
        }
    }
    if recs.iter().any(|r| r.routers.is_some()) {
        let mut counts = MapBuilder::new(None, StringBuilder::new(), ListBuilder::new(Int64Builder::new()));
        for rec in recs {
            for (name, values) in rec.routers.iter().flatten() {
                counts.keys().append_value(name);
                counts.values().append_value(values.iter().map(|v| Some(*v)));
            }
            counts.append(rec.routers.is_some()).unwrap();
        }
        columns.push(("moe_counts", Arc::new(counts.finish())));
    }
    if recs.iter().any(|r| r.probe.is_some()) {
        let probe: StringArray = recs.iter().map(|r| r.probe.clone()).collect();
        columns.push(("probe_json", Arc::new(probe)));
    }
    let extra: StringArray = recs
        .iter()
        .zip(&extras)
        .map(|(rec, extra)| match rec.extra {
            Some(text) => Some(text.to_owned()),
            None if extra.is_empty() => None,
            None => Some(Value::Object(extra.clone()).to_string()),
        })
        .collect();
    columns.push(("extra_json", Arc::new(extra)));
    RecordBatch::try_new(
        Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        )),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

/// `batch` with one more column.
fn with_column(batch: &RecordBatch, name: &str, array: ArrayRef) -> RecordBatch {
    let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(name, array.data_type().clone(), true));
    let mut columns = batch.columns().to_vec();
    columns.push(array);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn judged(batches: &[RecordBatch]) -> Value {
    report("run", 0, &judge(0, batches, &Arc::<Thresholds>::default()).unwrap(), 0)
}

fn kinds(report: &Value) -> Vec<(String, Value)> {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| (i["kind"].as_str().unwrap().to_owned(), i["step"].clone()))
        .collect()
}

fn check(report: &Value, kind: &str) -> Value {
    report["kind_checks"][kind]["value"].clone()
}

// --- the records: six layers in the model's order (the case `layers_model_order`) ---------------

const NAMES: [&str; 6] = [
    "model.layers.0",
    "model.layers.1",
    "model.layers.2",
    "model.layers.10",
    "model.layers.11",
    "model.layers.3",
];

fn layer(std: f64, abs_max: f64) -> Entry {
    Entry::Object(vec![
        ("mean", json!(0.0)),
        ("std", json!(std)),
        ("abs_max", json!(abs_max)),
        ("zero_frac", json!(0.0)),
    ])
}

/// Forty steps of six steady layers; at step 35, `model.layers.2` and `model.layers.10` spike.
fn run() -> Vec<Rec> {
    (0..40)
        .map(|step| {
            let std = 1.0 + 0.01 * (step % 3) as f64;
            let layers = NAMES
                .iter()
                .map(|name| {
                    let spike = step == 35 && (*name == "model.layers.2" || *name == "model.layers.10");
                    ((*name).to_owned(), layer(std, if spike { 60.0 } else { 3.0 * std }))
                })
                .collect();
            Rec {
                step,
                layers: Some(layers),
                ..Rec::default()
            }
        })
        .collect()
}

/// `recs` with one layer of one step replaced (or removed, for None).
fn changed(mut recs: Vec<Rec>, step: i64, name: &str, entry: Option<Entry>) -> Vec<Rec> {
    let layers = recs[step as usize].layers.as_mut().unwrap();
    let at = layers.iter().position(|(n, _)| n == name).unwrap();
    match entry {
        Some(entry) => layers[at].1 = entry,
        None => {
            layers.remove(at);
        }
    }
    recs
}

/// The first incident of `kind` in a report.
fn incident<'a>(report: &'a Value, kind: &str) -> &'a Value {
    report["incidents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["kind"] == kind)
        .unwrap_or_else(|| panic!("no {kind}: {report}"))
}

fn both(recs: &[Rec]) -> (Value, Value) {
    (judged(&[batch(recs, Layout::Json)]), judged(&[batch(recs, TYPED)]))
}

#[test]
fn a_typed_row_is_judged_as_its_json_row_and_keeps_the_models_order() {
    let (json_report, typed) = both(&run());
    assert_eq!(typed, json_report);
    let spike = &typed["incidents"][0];
    assert_eq!(spike["kind"], "activation_spike", "{typed}");
    let modules: Vec<&str> = spike["modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["module"].as_str().unwrap())
        .collect();
    assert_eq!(modules, ["model.layers.2", "model.layers.10"]);
    for kind in ["activation_spike", "activation_drift", "dead_activations"] {
        assert_eq!(check(&typed, kind), "checked", "{kind}");
    }
}

#[test]
fn a_non_finite_value_held_in_a_struct_field_names_its_layer() {
    let mut fields = match layer(1.0, 3.0) {
        Entry::Object(fields) => fields,
        Entry::Other(_) => unreachable!(),
    };
    fields[0].1 = json!("nan"); // `mean`, which only non_finite reads
    let recs = changed(run(), 10, "model.layers.3", Some(Entry::Object(fields)));
    let (json_report, typed) = both(&recs);
    assert_eq!(typed, json_report);
    let first = incident(&typed, "non_finite");
    assert_eq!(
        (first["step"].clone(), first["layers"].clone()),
        (json!(10), json!(["model.layers.3"])),
        "{typed}"
    );
}

/// Section 1.4 and 1.5: a refused field is null in the struct and absent to every detector; it is
/// read again from `extra_json` by `non_finite` alone.
#[test]
fn a_refused_field_is_absent_and_non_finite_reads_it_from_extra_json() {
    let with = |field: &'static str, value: Value| {
        let mut fields = match layer(1.0, 3.0) {
            Entry::Object(fields) => fields,
            Entry::Other(_) => unreachable!(),
        };
        match fields.iter_mut().find(|(name, _)| *name == field) {
            Some(slot) => slot.1 = value,
            None => fields.push((field, value)),
        }
        Entry::Object(fields)
    };
    for (field, value, named) in [
        ("std", json!(true), false),
        ("abs_max", json!("NaN"), false),
        ("std", Value::Null, false),
        ("p99", json!("nan"), true),
        ("p99", json!("-inf"), true),
        ("p99", json!(2.5), false),
    ] {
        let recs = changed(run(), 9, "model.layers.1", Some(with(field, value.clone())));
        let typed_batch = batch(&recs, TYPED);
        let extra = typed_batch
            .column_by_name("extra_json")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(9)
            .to_owned();
        assert_eq!(
            serde_json::from_str::<Value>(&extra).unwrap(),
            json!({"layers": {"model.layers.1": {field: value}}}),
            "the layout of section 1.4"
        );
        let typed = judged(&[typed_batch]);
        assert_eq!(typed, judged(&[batch(&recs, Layout::Json)]), "{field}: {value}");
        let non_finite = kinds(&typed).contains(&("non_finite".to_owned(), json!(9)));
        assert_eq!(non_finite, named, "{field}: {value}: {typed}");
        if named {
            assert_eq!(incident(&typed, "non_finite")["layers"], json!(["model.layers.1"]));
        }
    }
}

/// Section 1.5's first target, `layers_entry_not_object`: a null struct in its place, the entry in
/// `extra_json`; the rest of the map is read, as on the clean records. As in the case, every record
/// holds it, and today's JSON row then loses every layer of the run (unchanged here: protocol 4's
/// reader is not this lane's to change).
#[test]
fn an_entry_that_is_not_an_object_is_absent_and_the_rest_of_the_map_is_read() {
    let (mut dirty, mut clean) = (run(), run());
    for step in 0..40 {
        dirty = changed(dirty, step, "model.layers.2", Some(Entry::Other(json!(5))));
        clean = changed(clean, step, "model.layers.2", None);
    }
    let typed = judged(&[batch(&dirty, TYPED)]);
    assert_eq!(typed, judged(&[batch(&clean, TYPED)]));
    assert_eq!(typed, judged(&[batch(&clean, Layout::Json)]));
    assert_eq!(check(&typed, "activation_spike"), "checked");
    assert_eq!(
        incident(&typed, "activation_spike")["modules"][0]["module"],
        "model.layers.10"
    );
    let today = judged(&[batch(&dirty, Layout::Json)]);
    assert_eq!(check(&today, "activation_spike"), "not_configured");
}

/// A row whose every entry is a null struct holds no layer, as `{}` holds none.
#[test]
fn a_map_of_null_structs_holds_no_layer() {
    let recs: Vec<Rec> = (0..3)
        .map(|step| Rec {
            step,
            layers: Some(vec![("a".to_owned(), Entry::Other(json!(5)))]),
            ..Rec::default()
        })
        .collect();
    let empty: Vec<Rec> = (0..3)
        .map(|step| Rec {
            step,
            layers: Some(Vec::new()),
            ..Rec::default()
        })
        .collect();
    let typed = judged(&[batch(&recs, TYPED)]);
    assert_eq!(typed, judged(&[batch(&empty, Layout::Json)]));
    assert_eq!(check(&typed, "activation_spike"), "not_configured");
    assert_eq!(check(&typed, "non_finite"), "checked");
}

/// Section 1.5's third target, `layers_name_arrow_cannot_encode`: the entry is only in `extra_json`
/// (its name a lone surrogate, which a Rust string cannot hold, so the text is written by hand);
/// the column's entries are read as on the clean record.
#[test]
fn a_name_only_in_extra_json_leaves_the_column_s_entries_read() {
    let mut recs = run();
    recs[1].extra = Some(r#"{"layers":{"bad\ud800":{"mean":0.0,"std":1.0,"abs_max":3.0,"zero_frac":0.0}}}"#);
    let typed = judged(&[batch(&recs, TYPED)]);
    assert_eq!(typed, judged(&[batch(&run(), Layout::Json)]));
    assert_eq!(check(&typed, "activation_spike"), "checked");
}

/// The limit of a name Arrow cannot encode: serde_json refuses the lone surrogate's escape, so the
/// row's whole `extra_json` reads as nothing, as it does for protocol 4's row today (whose whole
/// map is in `extra_json` then). A non-finite field of that entry names no layer here, where Python
/// names it; no capture writes such a name (contract 1.4).
#[test]
fn a_name_arrow_cannot_encode_is_not_read_by_non_finite() {
    let text = r#"{"layers":{"bad\ud800":{"p99":"nan"}}}"#;
    assert_eq!(super::extra_json::parse(text), None);
    let mut recs = run();
    recs[1].extra = Some(text);
    let typed = judged(&[batch(&recs, TYPED)]);
    assert_eq!(typed, judged(&[batch(&run(), TYPED)]));
    assert!(!kinds(&typed).iter().any(|(kind, _)| kind == "non_finite"), "{typed}");
}

/// Section 1.5's second target, `layers_value_not_object`: the column is null, the value in
/// `extra_json`; nothing is read, as on the JSON row.
#[test]
fn a_layers_value_that_is_not_an_object_reads_no_layer() {
    let recs: Vec<Rec> = (0..3)
        .map(|step| Rec {
            step,
            layers_other: Some(json!([1, 2])),
            ..Rec::default()
        })
        .collect();
    let (json_report, typed) = both(&recs);
    assert_eq!(typed, json_report);
    assert_eq!(check(&typed, "activation_spike"), "not_configured");
}

/// D3: struct fields are looked up by name, in any order, a field the reader does not know is
/// ignored by the activation kinds, and a field a part does not have reads as absent.
#[test]
fn struct_fields_are_looked_up_by_name() {
    const REORDERED: &[(&str, bool)] = &[
        ("zero_frac", false),
        ("p50", false),
        ("abs_max", false),
        ("std", false),
        ("mean", false),
    ];
    let reordered = Layout::Typed {
        layers: REORDERED,
        moe: MOE,
    };
    let typed = judged(&[batch(&run(), TYPED)]);
    assert_eq!(judged(&[batch(&run(), reordered)]), typed);
    assert_eq!(check(&typed, "activation_spike"), "checked");

    const NO_STD: &[(&str, bool)] = &[("mean", false), ("abs_max", false), ("zero_frac", false)];
    let without = Layout::Typed {
        layers: NO_STD,
        moe: MOE,
    };
    let no_std: Vec<Rec> = run()
        .into_iter()
        .map(|mut rec| {
            for (_, entry) in rec.layers.iter_mut().flatten() {
                if let Entry::Object(fields) = entry {
                    fields.retain(|(name, _)| *name != "std");
                }
            }
            rec
        })
        .collect();
    // The part has no `std` field: the record's `std` goes to `extra_json` (a field the struct does
    // not name), and the layers are skipped by the activation kinds, as a record without `std` is.
    let typed = judged(&[batch(&run(), without)]);
    assert_eq!(typed, judged(&[batch(&no_std, Layout::Json)]));
    assert!(kinds(&typed).is_empty(), "{typed}");
}

/// Section 3: a layout is told per batch by its columns. A batch with a typed `layers` column reads
/// it, also where a row of it is null, and never falls back to `layers_json` row by row.
#[test]
fn the_typed_column_is_chosen_per_batch() {
    let recs: Vec<Rec> = run().into_iter().map(|rec| Rec { layers: None, ..rec }).collect();
    let typed_nulls = batch(&recs, TYPED);
    let json_text: StringArray = run().iter().map(|r| r.layers.as_deref().map(json_text)).collect();
    let both_columns = with_column(&typed_nulls, "layers_json", Arc::new(json_text));
    let report = judged(&[both_columns]);
    assert_eq!(check(&report, "activation_spike"), "not_configured", "{report}");
    assert_eq!(report, judged(&[typed_nulls]));
}

/// A rank whose older rows are in protocol 4's layout and newer ones in protocol 5's (section 3: the
/// parts of both layouts that a restart's backlog leaves) is judged as one run.
#[test]
fn json_batches_then_typed_batches_are_judged_as_one_run() {
    let recs = run();
    let mixed = [batch(&recs[..20], Layout::Json), batch(&recs[20..], TYPED)];
    let all_json = judged(&[batch(&recs, Layout::Json)]);
    assert_eq!(judged(&mixed), all_json);
    assert_eq!(kinds(&all_json), [("activation_spike".to_owned(), json!(35))]);
}

// --- moe_stats ----------------------------------------------------------------------------------

fn router(diversity: Option<Value>, extra: Option<(&'static str, Value)>) -> Entry {
    let mut fields = vec![("routed_tokens", json!(40)), ("entropy", json!(1.386))];
    if let Some(diversity) = diversity {
        fields.push(("diversity", diversity));
    }
    fields.push(("dead_experts", json!(0)));
    fields.push(("dead_frac", json!(0.0)));
    fields.extend(extra);
    Entry::Object(fields)
}

fn moe_run(stats: impl Fn(i64) -> Option<Entry>) -> Vec<Rec> {
    (0..8)
        .map(|step| Rec {
            step,
            routers: Some(vec![("r".to_owned(), vec![25, 10, 3, 2])]),
            moe_stats: stats(step).map(|entry| vec![("r".to_owned(), entry)]),
            ..Rec::default()
        })
        .collect()
}

fn stream(batches: &[RecordBatch]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, &batches[0].schema()).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
    drop(writer);
    out
}

/// Each sample's diversity, in step order, as `detect --samples` gives it.
fn diversities(batch: &RecordBatch) -> Vec<Value> {
    let mut out = Vec::new();
    detect_stream_samples(
        stream(std::slice::from_ref(batch)).as_slice(),
        &Arc::<Thresholds>::default(),
        &mut out,
    )
    .unwrap();
    let report: Value = serde_json::from_slice(&out).unwrap();
    report["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["diversity"].clone())
        .collect()
}

/// The router's `diversity` is read from the struct by name; refused (a bool), null, missing, a
/// null struct, or a part without the field, it is taken from the counts, as on the JSON row.
/// A part with K3's `token_entropy` is read as one without it (D3).
#[test]
fn the_router_diversity_is_read_by_name_and_otherwise_from_the_counts() {
    let recs = moe_run(|step| match step {
        0 | 1 => Some(router(Some(json!(0.25)), None)),
        2 => Some(router(Some(json!(true)), None)),
        3 => Some(router(Some(Value::Null), None)),
        4 => Some(router(None, None)),
        5 => Some(Entry::Other(json!(5))),
        6 => None,
        _ => Some(router(Some(json!(0.5)), Some(("token_entropy", json!(0.7))))),
    });
    let from_json = diversities(&batch(&recs, Layout::Json));
    let typed = diversities(&batch(&recs, TYPED));
    let counts = super::diversity_from_counts(&[25, 10, 3, 2]);
    assert_eq!(
        typed,
        [
            json!(0.25),
            json!(0.25),
            json!(counts),
            json!(counts),
            json!(counts),
            json!(counts),
            json!(counts),
            json!(0.5)
        ]
    );
    assert_eq!(typed, from_json);

    const WITH_TOKEN_ENTROPY: &[(&str, bool)] = &[
        ("routed_tokens", true),
        ("entropy", false),
        ("diversity", false),
        ("dead_experts", true),
        ("dead_frac", false),
        ("token_entropy", false),
    ];
    let k3 = Layout::Typed {
        layers: LAYER,
        moe: WITH_TOKEN_ENTROPY,
    };
    assert_eq!(diversities(&batch(&recs, k3)), typed);
    const NO_DIVERSITY: &[(&str, bool)] = &[("routed_tokens", true), ("entropy", false)];
    let without = Layout::Typed {
        layers: LAYER,
        moe: NO_DIVERSITY,
    };
    assert_eq!(diversities(&batch(&recs, without)), vec![json!(counts); 8]);
}

// --- the probe ----------------------------------------------------------------------------------

/// Section 2.4: `probe_maps` and `probe_histograms` are never read; the probe's light part in
/// `probe_json` is read as today. A report is the same with the two columns, of any type, as
/// without them.
#[test]
fn probe_maps_and_probe_histograms_change_nothing() {
    let mut recs = run();
    // The case `probe_split_maps_and_histograms`: a probe every other step, the first module's
    // entropy share falling from 0.8 to 0.1 at step 9 and staying there.
    for step in [1, 3, 5, 7, 9, 11] {
        let share = if step >= 9 { 0.1 } else { 0.8 };
        let module = |share: f64| {
            json!({"max_logit": [5.0, 2.5], "entropy": [1.0, 1.25], "entropy_share": [share, share],
                   "masked_rows": 0, "invalid_rows": 0, "rows": 8, "query_len": 2, "key_len": 3})
        };
        recs[step].probe =
            Some(json!({"attention": {"model.layers.0": module(share), "model.layers.0#1": module(0.8)}}).to_string());
    }
    let plain = batch(&recs, TYPED);
    let want = judged(std::slice::from_ref(&plain));
    assert!(
        kinds(&want).contains(&("attention_collapse".to_owned(), json!(9))),
        "{want}"
    );
    let maps: StringArray = recs.iter().map(|_| Some("not a map")).collect();
    let histograms = Int64Array::from_iter_values(recs.iter().map(|_| 7));
    let with = with_column(&plain, "probe_maps", Arc::new(maps));
    let with = with_column(&with, "probe_histograms", Arc::new(histograms));
    assert_eq!(judged(&[with]), want);
}

// --- the row version ----------------------------------------------------------------------------

/// Section 3: `detect` takes row `schema_version` 2, protocol 5's, as well as 0 and 1, and refuses
/// any other naming the three.
#[test]
fn detect_takes_row_version_2_and_names_the_three_it_reads() {
    let typed = batch(&run(), TYPED);
    let report = detect_stream(stream(std::slice::from_ref(&typed)).as_slice(), &Arc::default()).unwrap();
    assert_eq!(kinds(&report), [("activation_spike".to_owned(), json!(35))]);

    let versions = Int32Array::from_iter_values((0..40).map(|step| if step == 2 { 3 } else { 2 }));
    let mut columns = typed.columns().to_vec();
    let at = typed.schema().index_of("schema_version").unwrap();
    columns[at] = Arc::new(versions);
    let refused = RecordBatch::try_new(typed.schema(), columns).unwrap();
    let error = detect_stream(stream(&[refused]).as_slice(), &Arc::default()).unwrap_err();
    assert_eq!(
        error,
        "row 3: schema_version is 3; this server reads versions 0, 1 and 2; the input is refused"
    );
}

/// A row version this server stores is one `detect` reads: the live server's list (lane AL-2's,
/// `ingest::SUPPORTED_ROW_VERSIONS`) and `detect`'s may differ only until the seam joins them.
#[test]
fn every_version_the_server_stores_is_one_detect_reads() {
    let typed = batch(&run(), TYPED);
    let at = typed.schema().index_of("schema_version").unwrap();
    for version in crate::ingest::SUPPORTED_ROW_VERSIONS.into_iter().chain([2]) {
        let mut columns = typed.columns().to_vec();
        columns[at] = Arc::new(Int32Array::from_iter_values((0..40).map(|_| version)));
        let rows = RecordBatch::try_new(typed.schema(), columns).unwrap();
        detect_stream(stream(&[rows]).as_slice(), &Arc::default()).unwrap_or_else(|error| panic!("{version}: {error}"));
    }
}
