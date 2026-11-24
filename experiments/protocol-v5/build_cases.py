"""Writes sdk/tests/fixtures/wave_al/cases.json, the shared cases of WAVE_AL_CONTRACT.md.

    PYTHONPATH=<worktree>/sdk/src TENSORQUILL_SERVER_BIN=<worktree>/server/target/release/tensorquill-server \
        .venv/bin/python experiments/protocol-v5/build_cases.py

Each case's inputs are written here by hand. What is computed, and from what:

- `row` (the layout every path must give): from `prototype.py`, the layout the contract states. A
  lane's `flight.record_to_row` must give the same typed columns, `probe_json` and `extra_json`.
- `expect` (what every engine path must report): for sections 1 and 2, today's engines on
  `clean_records` (the same records with every refused part removed, or in the form today's code
  reads correctly); the v5 engines must report on `records` what today's report on the clean ones.
  For section 4, written by hand from the contract's rules.
- `today` (what the base does with `records`, run here, never recomputed by the tests): the Python
  engine and the raw Rust report, and the row's round trip through today's `flight.py`.

Only the base's tree is run: nothing here imports a lane's code.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import prototype as p5  # noqa: E402
from tensorquill import evidence  # noqa: E402
from tensorquill import flight as v4  # noqa: E402
from tensorquill.server_detect import find_binary, server_report  # noqa: E402

OUT = ROOT / "sdk" / "tests" / "fixtures" / "wave_al" / "cases.json"
FORWARD = ["model.layers.0", "model.layers.1", "model.layers.2", "model.layers.10", "model.layers.11", "model.layers.3"]
KINDS = sorted(("activation_spike", "activation_drift", "dead_activations", "non_finite", "moe_imbalance",
                "attention_collapse", "attention_uniform", "attention_logit_growth"))
LOCATION_KEYS = ("layers", "modules", "gradients", "routers", "router", "parameters", "experts")


def base_sha() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def rec(run: str, step: int, **fields) -> dict:
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, "step": step, "ts": float(step), **fields}


def layer(std=1.0, abs_max=3.0, zero_frac=0.0, mean=0.0) -> dict:
    return {"mean": mean, "std": std, "abs_max": abs_max, "zero_frac": zero_frac}


def run_of_layers(run: str, steps: int, at: int | None = None, spike: dict | None = None, odd: dict | None = None) -> list[dict]:
    """`steps` healthy records over FORWARD, with `spike` ({name: stats}) at step `at` and `odd`
    ({name: value}) replacing entries on every record (the case's refused input)."""
    out = []
    for s in range(steps):
        layers = {n: layer(std=1.0 + 0.01 * (s % 3), abs_max=3.0 * (1 + 0.01 * (s % 5))) for n in FORWARD}
        if at is not None and s == at:
            layers.update(spike or {})
        if odd:
            layers.update(copy.deepcopy(odd))
        out.append(rec(run, s, layers=layers, grad_norm=1.0))
    return out


def brief(incident: dict) -> dict:
    out = {"kind": incident["kind"], "step": incident.get("step")}
    for key in LOCATION_KEYS:
        if key in incident:
            value = incident[key]
            out[key] = sorted(value) if isinstance(value, dict) else value
    return out


def engines(records: list[dict], rust: bool) -> dict:
    out: dict = {}
    try:
        built = evidence.build_evidence(copy.deepcopy(records), detectors="python")
        out["python"] = {"incidents": [brief(i) for i in built["incidents"]],
                         "kind_checks": {k: (built["kind_checks"].get(k) or {}).get("value") for k in KINDS},
                         "run_state": (built.get("run_state") or {}).get("state")}
    except Exception as error:  # noqa: BLE001
        out["python"] = {"raises": f"{type(error).__name__}: {str(error)[:200]}"}
    if rust:
        try:
            report = server_report(copy.deepcopy(records))
            out["rust_raw"] = {"incidents": [brief(i) for i in report["incidents"]],
                               "kind_checks": {k: (report["kind_checks"].get(k) or {}).get("value") for k in KINDS}}
        except Exception as error:  # noqa: BLE001
            out["rust_raw"] = {"raises": f"{type(error).__name__}: {str(error)[:200]}"}
    return out


def jsonable(value):
    """Arrow-side values as JSON: non-finite floats as the sink's strings, tuples as pairs."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def row_expectation(record: dict, step: int) -> dict:
    row = p5.record_to_row_v5(record)
    out = {"step": step, "schema_version": row["schema_version"], "record_version": row["record_version"], "seq": row["seq"]}
    for column in ("layers", "moe_stats", "probe_maps", "probe_histograms"):
        if column in ("probe_maps", "probe_histograms") and "probe" not in record:
            continue
        if column in ("layers", "moe_stats") and column not in record:
            continue
        out[column] = None if row[column] is None else [[name, jsonable(v)] for name, v in row[column]]
    if "probe" in record:
        out["probe_json"] = row["probe_json"]
    out["extra"] = json.loads(row["extra_json"]) if row.get("extra_json") else None
    import pyarrow as pa
    batch = pa.RecordBatch.from_pylist([row], schema=p5.flight_schema_v5())
    back = p5.row_to_record_v5(batch.to_pylist()[0])
    out["round_trip"] = back == record
    out["order_kept"] = all(list(back.get(k) or {}) == list(record.get(k) or {}) for k in ("layers", "moe_stats"))
    return out


def today_row(record: dict) -> dict:
    try:
        row = v4.record_to_row(record)
        return {"layers_json": row.get("layers_json"), "moe_stats_json": row.get("moe_stats_json"),
                "extra": json.loads(row["extra_json"]) if row.get("extra_json") else None,
                "round_trip": v4.row_to_record(row) == record}
    except Exception as error:  # noqa: BLE001
        return {"raises": f"{type(error).__name__}: {str(error)[:200]}"}


# --- section 1: layers and moe_stats ---------------------------------------------------------------


def section_1() -> list[dict]:
    cases = []

    def add(name, about, records, clean, row_step, paths=("row", "python", "rust_raw", "rust_sdk", "live_server"), **more):
        cases.append({"name": name, "section": 1, "about": about, "paths": list(paths), "records": records,
                      "clean_records": clean, "row_step": row_step, **more})

    spike = {"model.layers.10": layer(abs_max=60.0), "model.layers.2": layer(abs_max=60.0)}
    add("layers_model_order", "activation_spike leads with the earliest block in forward order; the typed map keeps "
        "the model's order (model.layers.2 before model.layers.10), which sorting would not",
        run_of_layers("al_order", 40, at=35, spike=spike), None, 35)
    add("layers_empty_map", "a router-only capture writes layers {} on every record (granite-router-only.jsonl): an "
        "empty typed map, no extra_json", [rec("al_empty", s, layers={}, grad_norm=1.0) for s in range(3)], None, 0)
    nf = {"model.layers.3": {"mean": "nan", "std": "inf", "abs_max": "-inf", "zero_frac": 0.0}}
    add("layers_non_finite_strings", "the sink's strings for NaN and Inf in struct fields are held, as non-finite "
        "floats; non_finite names the layer in both engines", run_of_layers("al_nf", 12, at=10, spike=nf), None, 10)
    add("layers_field_bool", "a bool in a field is refused: the field is null in the struct and travels in "
        "extra_json under layers.<name>.<field>; the layer is skipped by the activation kinds (std absent)",
        run_of_layers("al_bool", 12, odd={"model.layers.2": {**layer(), "std": True}}),
        run_of_layers("al_bool", 12, odd={"model.layers.2": {k: v for k, v in layer().items() if k != "std"}}), 3)
    add("layers_field_other_string", "a string other than the sink's three (\"NaN\") is refused like a bool",
        run_of_layers("al_str", 12, odd={"model.layers.2": {**layer(), "abs_max": "NaN"}}),
        run_of_layers("al_str", 12, odd={"model.layers.2": {k: v for k, v in layer().items() if k != "abs_max"}}), 3)
    add("layers_field_missing", "a missing field is a null struct field and reads back absent; no extra_json",
        run_of_layers("al_missing", 12, odd={"model.layers.2": {k: v for k, v in layer().items() if k != "zero_frac"}}),
        None, 3)
    add("layers_field_extra", "a field the struct does not name travels in extra_json; its value is still read for "
        "non_finite (Python reads every value of a layer; detect.rs' visitor ORs every key today)",
        run_of_layers("al_extra", 12, at=9, spike={"model.layers.1": {**layer(), "p99": "nan"}}), None, 9)
    add("layers_field_null", "an explicit null field is refused, so it reads back as null and not absent",
        run_of_layers("al_null", 12, odd={"model.layers.2": {**layer(), "std": None}}),
        run_of_layers("al_null", 12, odd={"model.layers.2": {k: v for k, v in layer().items() if k != "std"}}), 3)
    add("layers_entry_not_object", "an entry that is not an object keeps its place as a null struct and travels "
        "whole in extra_json; it is absent to every detector, in both engines",
        run_of_layers("al_notobj", 12, odd={"model.layers.2": 5}),
        [{**r, "layers": {k: v for k, v in r["layers"].items() if k != "model.layers.2"}} for r in run_of_layers("al_notobj", 12)], 3)
    add("layers_name_arrow_cannot_encode", "a name with a lone surrogate cannot be a key: the entry travels only in "
        "extra_json and is read after the column's entries", run_of_layers("al_surrogate", 3, odd={"bad\ud800": layer()}),
        [{**r, "layers": r["layers"]} for r in run_of_layers("al_surrogate", 3)], 1, paths=("row", "python", "rust_raw", "rust_sdk"))
    add("layers_int_in_float_field", "an integer in a Float64 field is held as a float: the round trip holds by == "
        "(0 == 0.0), the JSON text does not (\"0\" against \"0.0\"), as for wave AJ's maps",
        run_of_layers("al_int", 3, odd={"model.layers.0": {"mean": 0, "std": 1, "abs_max": 3, "zero_frac": 0}}), None, 1)
    add("layers_value_not_object", "a layers value that is not an object travels whole in extra_json and the column is null",
        [rec("al_list", s, layers=[1, 2], grad_norm=1.0) for s in range(3)],
        [rec("al_list", s, grad_norm=1.0) for s in range(3)], 0, paths=("row", "python", "rust_raw", "rust_sdk"))
    router = "model.layers.0.block_sparse_moe.router"
    def moe_rec(run, s, stats, counts=None):
        counts = counts or [10, 10, 10, 10]
        return rec(run, s, moe={router: counts}, moe_top_k={router: 1}, moe_stats={router: stats}, grad_norm=1.0)
    good = {"routed_tokens": 40, "entropy": math.log(4), "diversity": 1.0, "dead_experts": 0, "dead_frac": 0.0}
    add("moe_stats_real_fields", "the five fields the capture writes (_router_stats): routed_tokens and dead_experts "
        "Int64, entropy, diversity and dead_frac Float64; the architecture page's struct lacks entropy and "
        "dead_experts", [moe_rec("al_moe", s, dict(good)) for s in range(12)], None, 0)
    add("moe_stats_float_in_int_field", "a float in an Int64 field (routed_tokens 40.0) is refused to extra_json",
        [moe_rec("al_moefloat", s, {**good, "routed_tokens": 40.0}) for s in range(12)],
        [moe_rec("al_moefloat", s, {k: v for k, v in good.items() if k != "routed_tokens"}) for s in range(12)], 0)
    add("moe_stats_diversity_refused", "diversity true is refused (a bool is not a number): absent to every detector, so "
        "both engines take the diversity from the counts, as for a record with none (today Python reads float(true) = 1.0 "
        "and Rust reads it as absent)",
        [moe_rec("al_moediv", s, {**good, "diversity": True}, [37, 1, 1, 1] if s >= 14 else None) for s in range(18)],
        [moe_rec("al_moediv", s, {k: v for k, v in good.items() if k != "diversity"}, [37, 1, 1, 1] if s >= 14 else None)
         for s in range(18)], 15)
    add("moe_stats_int_past_int64", "an integer past int64 is refused to extra_json and moved (the sender's count names "
        "moe_stats once per row)", [moe_rec("al_moebig", s, {**good, "dead_experts": 2**70}) for s in range(3)],
        [moe_rec("al_moebig", s, {k: v for k, v in good.items() if k != "dead_experts"}) for s in range(3)], 0,
        paths=("row", "python", "rust_raw", "rust_sdk"))
    add("moe_stats_token_entropy_by_name", "K3's token_entropy is not reserved: a part without the field and a part "
        "with it are read by one reader that looks fields up by name (live.py struct_field_added: pyarrow and the Rust "
        "reader give absent, then the value)", [moe_rec("al_te", 0, dict(good))], None, 0, paths=("row",),
        measured="results/live.json: struct_field_added")
    return cases


# --- section 2: the probe ---------------------------------------------------------------------------


def probe_record(run: str, step: int, probe) -> dict:
    return rec(run, step, layers={"model.layers.0": layer()}, grad_norm=1.0, probe=probe)


def attention_entry(share: float, logit: float, rows: int = 2, cols: int = 3, with_map: bool = True) -> dict:
    entry = {"max_logit": [logit, logit / 2], "entropy": [1.0, 1.25], "entropy_share": [share, share], "masked_rows": 0,
             "invalid_rows": 0, "rows": 8, "query_len": rows, "key_len": cols}
    if with_map:
        entry["map"] = [[0.25 * ((r + c) % 4) for c in range(cols)] for r in range(rows)]
    return entry


def histogram(lo: float = -1.0) -> dict:
    return {"lo": lo, "hi": 1.0, "counts": [1, 2, 3, 2], "nonfinite": 0}


def section_2() -> list[dict]:
    cases = []

    def add(name, about, records, clean, row_step, paths=("row", "python", "rust_raw", "rust_sdk", "live_server"), **more):
        cases.append({"name": name, "section": 2, "about": about, "paths": list(paths), "records": records,
                      "clean_records": clean, "row_step": row_step, **more})

    def probes(run, attention_at):
        out = []
        for s in range(12):
            probe = None
            if s % 2 == 1:
                share = 0.8 if s < attention_at else 0.1
                probe = {"histograms": {"model.layers.0": histogram()},
                         "attention": {"model.layers.0": attention_entry(share, 5.0), "model.layers.0#1": attention_entry(0.8, 5.0, 3, 2)},
                         "head": {"module": "lm_head", "found_by": "name", "norm": 1.0, "mean": 0.0, "log_z": 2.0,
                                  "gap": 0.01, "gap_relative": 0.001}}
            out.append(probe_record(run, s, probe) if probe else rec(run, s, layers={"model.layers.0": layer()}, grad_norm=1.0))
        return out

    add("probe_split_maps_and_histograms", "maps (ragged: 2x3 and 3x2) go to probe_maps by label, histograms whole to "
        "probe_histograms, the rest (per-head lists, row counts, head) stays in probe_json; attention_collapse is judged "
        "the same; the evidence does not change", probes("al_probe", 9), None, 9)
    add("probe_map_not_float32", "a map value that is not exactly a float32 (0.1) keeps that module's map in probe_json",
        [probe_record("al_f32", 1, {"histograms": {}, "attention": {"a": {**attention_entry(0.5, 2.0), "map": [[0.1, 0.9]]}}})],
        None, 1, paths=("row", "python", "rust_raw"))
    add("probe_entry_error", "an attention entry {error} stays whole in probe_json; the probe's other modules split",
        [probe_record("al_entryerr", 1, {"histograms": {"b": histogram()}, "attention": {
            "a": {"error": "RuntimeError: 5-D query"}, "b": attention_entry(0.5, 2.0)}})], None, 1, paths=("row", "python", "rust_raw"))
    add("probe_not_run", "{not_run} (a sharded model) stays in probe_json alone; no heavy column is set",
        [probe_record("al_notrun", 0, {"not_run": "the model is sharded (FSDP)"})], None, 0, paths=("row", "python", "rust_raw"))
    add("probe_error_whole", "a probe that failed, {error}, stays in probe_json; the attention kinds count it failed in "
        "both engines", [probe_record("al_err", 1, {"error": "RuntimeError: boom"})], None, 1, paths=("row", "python", "rust_raw"))
    add("probe_histogram_irregular", "a histogram entry with another key set or a bool count keeps every histogram of "
        "that probe in probe_json (all or none, so the map's order is never split)",
        [probe_record("al_hist", 1, {"histograms": {"a": histogram(), "b": {**histogram(), "counts": [1, True]}},
                                     "attention": {}})], None, 1, paths=("row", "python", "rust_raw"))
    add("probe_attention_note", "attention_note (FlashAttention switched) and an empty attention stay in probe_json",
        [probe_record("al_note", 1, {"histograms": {"a": histogram()}, "attention": {},
                                     "attention_note": "no scaled_dot_product_attention call seen"})], None, 1,
        paths=("row", "python", "rust_raw"))
    return cases


# --- section 4: the scalar-only record --------------------------------------------------------------


def scalar(run: str, step: int, seq: int, values: dict, attempt: int = 1) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run, "rank": 0, "attempt": attempt, "step": step,
            "seq": seq, "ts": 100.0 + seq, "scalars": values}


def section_4() -> list[dict]:
    cases = []

    def add(name, about, records, expect, paths, **more):
        cases.append({"name": name, "section": 4, "about": about, "paths": list(paths), "records": records,
                      "expect": expect, **more})

    steps = [rec("al_sc", s, layers={"model.layers.0": layer()}, grad_norm=1.0, scalars={"train/loss": 2.0 - 0.1 * s}) for s in range(6)]
    add("scalar_record_keys", "session.log(eval_loss=1.8, step=500) at step 3: one line, these keys and no others; the "
        "row carries seq and record_version 2; row_to_record(record_to_row(r)) == r",
        steps[:4] + [scalar("al_sc", 500, 0, {"eval_loss": 1.8})] + steps[4:],
        {"keys": ["schema_version", "record", "run_id", "rank", "attempt", "step", "seq", "ts", "scalars"],
         "steps_judged": 6, "repeated_steps": False, "series": {"eval_loss": [[500, 1.8]]},
         "last_training_step": 5}, ("row", "python", "rust_raw", "rust_sdk", "live_server", "runs", "mcp_get_series"))
    add("scalar_not_a_training_step", "a scalar record at a step that has a training record: no repeated_steps, the same "
        "records judged, kind_checks' records unchanged, in every engine",
        [{**r, "run_id": "al_sc2"} for r in steps[:3]] + [scalar("al_sc2", 2, 0, {"eval/loss": 1.5})] + [{**r, "run_id": "al_sc2"} for r in steps[3:]],
        {"steps_judged": 6, "repeated_steps": False, "series": {"eval/loss": [[2, 1.5]]}, "evidence_equal_to_without": True},
        ("python", "rust_raw", "rust_sdk", "live_server"))
    add("scalar_future_step_keeps_training_rows", "a scalar record for step 1000 logged at step 3, then steps 4 to 9 and "
        "a server restart: every training row is stored (today, a row of step 1000 in a part-* file raises the attempt's "
        "watermark from the part's footer and steps 4 to 9 are dropped as not newer: results/live.json, watermark)",
        [rec("al_future", s, grad_norm=1.0) for s in range(4)] + [scalar("al_future", 1000, 0, {"eval/loss": 1.4})]
        + [rec("al_future", s, grad_norm=1.0) for s in range(4, 10)],
        {"stored_training_steps": list(range(10)), "stored_scalar_seqs": [0], "objects": ["part-*", "scalars-00000000-00000000.parquet"]},
        ("live_server",), measured="results/live.json: watermark")
    add("scalar_replay_from_byte_0", "the sender replays its file from byte 0 (a replaced store's ID, rewritten bytes): "
        "scalar rows at or below the attempt's seq watermark are dropped and counted, nothing is stored twice",
        [rec("al_replay", s, grad_norm=1.0) for s in range(3)] + [scalar("al_replay", 1, 0, {"eval/loss": 1.1}),
                                                                  scalar("al_replay", 1, 1, {"eval/loss": 1.0})],
        {"sent_twice": True, "stored_scalar_seqs": [0, 1], "scalars_not_newer": 2}, ("live_server",))
    add("scalar_seq_watermark_across_restart", "the server restarts between seq 1 and seq 2: the watermark is read back "
        "from the scalars-* objects' names, so seq 2 is stored and a resend of 0 and 1 is not",
        [scalar("al_restart", 5, i, {"eval/loss": 1.0 + i}) for i in range(3)],
        {"stored_scalar_seqs": [0, 1, 2]}, ("live_server",))
    add("scalar_step_never_written", "a scalar for a step no training record has (7, past the last step 5): a point of "
        "its own on its series; no record is made for step 7 and runs reports the last training step, 5",
        [{**r, "run_id": "al_never"} for r in steps] + [scalar("al_never", 7, 0, {"eval/loss": 1.2})],
        {"series": {"eval/loss": [[7, 1.2]]}, "last_training_step": 5, "steps_judged": 6}, ("python", "runs", "mcp_get_series"))
    add("scalar_first_and_last_line", "a file whose first and last lines are scalar records: runs takes its run ID from "
        "the first line and its last step from the last training record", [scalar("al_edges", 0, 0, {"eval/baseline": 3.0})]
        + [{**r, "run_id": "al_edges"} for r in steps] + [scalar("al_edges", 5, 1, {"eval/final": 1.0})],
        {"run_id": "al_edges", "last_training_step": 5, "series": {"eval/baseline": [[0, 3.0]], "eval/final": [[5, 1.0]]}},
        ("runs", "mcp_get_series", "python"))
    add("scalar_same_step_twice", "two scalar records for one step and name: the higher seq is the value read",
        [scalar("al_twice", 4, 0, {"eval/loss": 1.5}), scalar("al_twice", 4, 1, {"eval/loss": 1.25})] +
        [{**r, "run_id": "al_twice"} for r in steps],
        {"series": {"eval/loss": [[4, 1.25]]}}, ("python", "mcp_get_series"))
    add("scalar_rolled_back_by_a_later_attempt", "attempt 1 logs a scalar for step 5; attempt 2 resumes from step 4: "
        "the default read leaves it out with attempt 1's steps 4 and 5, all_attempts keeps it",
        [rec("al_roll", s, grad_norm=1.0) for s in range(6)] + [scalar("al_roll", 5, 0, {"eval/loss": 1.0})]
        + [{**rec("al_roll", s, grad_norm=1.0), "attempt": 2} for s in range(4, 8)],
        {"series_default": {}, "series_all_attempts": {"eval/loss": [[5, 1.0]]}}, ("python", "mcp_get_series"))
    add("scalar_old_reader", "today's readers given a file with a scalar record refuse the whole file, naming its line "
        "(identity.check: schema_version 2)", [rec("al_old", 0, grad_norm=1.0), scalar("al_old", 9, 0, {"eval/loss": 1.0})],
        {"today_raises": "MalformedRecordError"}, ("today",))
    add("scalar_log_refusals", "session.log(..., step=s) refuses, before anything is recorded: a bool, a float, a "
        "negative integer or text as s (TypeError or ValueError, as log refuses a bad value); with no step= the rules "
        "of wave AK hold (a value with no step in memory: the scalar_no_step_* cases, Q5)",
        [], {"refused_steps": [True, 1.5, -1, "3"], "written": 0}, ("capture",))
    # Q5, decided Nov 24: no step in memory and no step=. Left out with one warning per session, which
    # shows the explicit call. Written by hand from the contract's section Q5; lane AL-5's to run.
    for name, about, expect in json.loads('[["scalar_no_step_baseline_warning", "under session.phase(\\"eval\\") before the first step, session.log({\\"eval/loss\\": 2.5}) with no step=: nothing is written, and the one warning names a baseline evaluation and the call that records it at start_step (0)", {"setup": {"start_step": 0, "steps_before": 0, "flush": false, "phase": "eval", "mirrored": false, "at": "log"}, "records_written": 0, "warnings": 1, "warning_contains": ["it was logged before the first training step (a baseline evaluation)", "session.log({\'eval/loss\': value}, step=0), the step training starts at", "Said once per session."]}], ["scalar_no_step_baseline_on_a_resume_warning", "the same on a resumed session (start_step=500): the text suggests step=500, the step training starts at", {"setup": {"start_step": 500, "steps_before": 0, "flush": false, "phase": "eval", "mirrored": false, "at": "log"}, "records_written": 0, "warnings": 1, "warning_contains": ["(a baseline evaluation)", "session.log({\'eval/loss\': value}, step=500), the step training starts at"]}], ["scalar_no_step_after_flush_warning", "steps 0 to 4, flush(), then session.log({\\"eval/loss\\": 1.9}) under \\"eval\\": nothing is written, and the warning names an evaluation after flush() and the last step training finished, 4", {"setup": {"start_step": 0, "steps_before": 5, "flush": true, "phase": "eval", "mirrored": false, "at": "log"}, "records_written": 0, "warnings": 1, "warning_contains": ["it was logged after flush() with no training step since (an evaluation after flush())", "session.log({\'eval/loss\': value}, step=4), the last step training finished"]}], ["scalar_no_step_pending_at_close_after_flush_warning", "steps 0 to 4, flush(), session.log({\\"eval/loss\\": 1.9}) outside \\"eval\\", then close(): the pending value is left out, and the warning says so and suggests step 4", {"setup": {"start_step": 0, "steps_before": 5, "flush": true, "phase": "train", "mirrored": false, "at": "close"}, "records_written": 0, "warnings": 1, "warning_contains": ["it was still pending at close() after flush() with no training step since", "step=4), the last step training finished"]}], ["scalar_no_step_mirrored_warning", "a value wandb.log brings through mirror_scalars under \\"eval\\" before the first step: the warning adds that the mirror cannot pass a step and the user must call session.log(..., step=...) for it (a documented limit, Q7)", {"setup": {"start_step": 0, "steps_before": 0, "flush": false, "phase": "eval", "mirrored": true, "at": "log"}, "records_written": 0, "warnings": 1, "warning_contains": ["(a baseline evaluation)", "It came from your tracker\'s log call through mirror_scalars, which cannot pass a step (the tracker\'s own step is ignored): call session.log(..., step=...) yourself for it."]}], ["scalar_no_step_said_once", "a baseline evaluation left out, then steps 0 to 4, flush() and an evaluation left out: one warning in the session, the baseline\'s; the second value is left out silently", {"setup": {"start_step": 0, "steps_before": 5, "flush": true, "phase": "eval", "mirrored": false, "at": "log", "baseline_first": true}, "records_written": 0, "warnings": 1, "warning_contains": ["(a baseline evaluation)"], "warning_lacks": ["after flush()"]}]]'):
        add(name, about, [], expect, ("capture",))
    # Q5's addition (the coordinator's choice on the owner's word, Nov 24): the Hugging Face callback
    # passes step= when no step is in memory; and the step-0 point a baseline shares (Q6).
    for name, about, expect in json.loads('[["scalar_hf_eval_on_start_baseline", "a Trainer with eval_on_start=True on a fresh run: the callback\'s on_evaluate finds no step in memory and logs the eval/* metrics with step=state.global_step (0), so one scalar-only record at step 0 is written and no warning is said", {"setup": {"adapter": "huggingface", "eval_on_start": true, "resume_global_step": 0}, "records_written": 1, "record": {"step": 0, "seq": 0, "names_start_with": "eval/"}, "warnings": 0}], ["scalar_hf_eval_on_start_on_a_resume", "the same resumed from checkpoint-500: start_step and global_step are 500, so the record is at step 500", {"setup": {"adapter": "huggingface", "eval_on_start": true, "resume_global_step": 500}, "records_written": 1, "record": {"step": 500, "seq": 0, "names_start_with": "eval/"}, "warnings": 0}]]'):
        add(name, about, [], expect, ("capture",))
    series = json.loads('{"name": "scalar_baseline_and_step_0_share_a_point", "section": 4, "about": "a baseline recorded with step=0 and an evaluation after step 0 recorded on step 0\'s own record under the same name are one series point; by Q6 the scalar-only record wins, so the series shows the baseline, and step 0\'s record still holds the later value (a README line, section 9)", "paths": ["python", "mcp_get_series"], "records": [{"schema_version": 2, "record": "scalars", "run_id": "al_share", "rank": 0, "attempt": 1, "step": 0, "seq": 0, "ts": 100.0, "scalars": {"eval/loss": 3.0}}, {"schema_version": 1, "run_id": "al_share", "rank": 0, "attempt": 1, "step": 0, "ts": 0.0, "grad_norm": 1.0, "scalars": {"train/loss": 2.9, "eval/loss": 2.8}}, {"schema_version": 1, "run_id": "al_share", "rank": 0, "attempt": 1, "step": 1, "ts": 1.0, "grad_norm": 1.0, "scalars": {"train/loss": 2.7}}], "expect": {"series": {"eval/loss": [[0, 3.0]], "train/loss": [[0, 2.9], [1, 2.7]]}, "step_0_record_eval_loss": 2.8, "steps_judged": 2}}')
    add(series["name"], series["about"], series["records"], series["expect"], tuple(series["paths"]))
    # Section 4.8 (the owner, Nov 24: "this will be fixed in AL"): a value logged late inside a compiled
    # function, before the flush for a newly trainable parameter. Written by hand from capture.py at
    # f71d8e0, not run here; `today` is what the code does, which the contract test's target and pin show.
    setup = {"model": "Linear(4, 4) then Linear(4, 1), the first frozen at init",
             "init": {"every": [1, 4], "weights": 1, "updates": True}, "optimizer": "AdamW", "steps": 3,
             "evaluated_after_steps": [0, 1]}
    add("scalar_eager_late_before_unfreeze",
        "the reference: a model whose first layer is frozen at init runs steps 0 and 1, each followed by an "
        "evaluation run eagerly under session.phase(\"eval\") that logs eval/x; the first layer is made trainable "
        "before step 2, whose optimizer pre-step hook flushes the ring (_take_unfrozen) before step 2 ends. The "
        "eval/x logged after step 1 is on step 1's own record, at every=1 and every=4, and no left-out warning is "
        "said", [],
        {"setup": {**setup, "compiled": False, "unfrozen_before_step": 2}, "step_1_scalars_hold": ["eval/x"],
         "left_out_warnings": 0, "unfreeze_warnings": 1}, ("capture",),
        today={"capture": "the same as expect (read in capture.py at f71d8e0; pinned by "
                          "test_the_same_value_logged_eagerly_reaches_its_step)"})
    add("scalar_compiled_late_before_unfreeze",
        "the same evaluation under torch.compile(fullgraph=True), compiled once for both calls: the traced log puts "
        "eval/x in "
        "_late_traced; the flush for the newly trainable parameter writes step 1 without draining it, and step 2's "
        "end then finds no step in memory. Fixed: eval/x on step 1's own record, as in the eager twin, and no "
        "left-out warning (section 4.8)", [],
        {"setup": {**setup, "compiled": True, "unfrozen_before_step": 2}, "frames_compiled": 1,
         "step_1_scalars_hold": ["eval/x"], "left_out_warnings": 0, "unfreeze_warnings": 1}, ("capture",),
        today={"capture": "step 1's record lacks eval/x, and one left-out warning is said that names the wrong "
                          "cause (read in capture.py at f71d8e0; the strict target "
                          "test_a_value_logged_late_inside_a_compiled_function_reaches_its_step shows it)",
               "warning_contains": ["with no step in memory (before the first step, or after flush() with no step "
                                    "since)"]})
    add("scalar_compiled_late_at_a_stop",
        "a stated limit: the compiled evaluation logs eval/x after step 1, then capture stops (_stop: a sink error, "
        "a writer behind, a flush that raised) before step 1 is handed to the writer. Step 1 and eval/x are lost "
        "together; the stop's own warning is the one said, and no left-out warning", [],
        {"setup": {"compiled": True, "evaluation_after_step": 1, "stop": "before step 1 is handed to the writer"},
         "records_with_step_1": 0, "left_out_warnings": 0, "stop_warnings": 1,
         "warning_contains": ["the steps already handed to the writer are written, no later step is"]}, ("capture",),
        today={"capture": "the same as expect (read in capture.py at f71d8e0; lane AL-5's own test)"})
    return cases


def main() -> None:
    rust = find_binary() is not None
    cases = section_1() + section_2()
    for case in cases:
        records, clean = case["records"], case.get("clean_records")
        case["row"] = row_expectation(next(r for r in records if r["step"] == case["row_step"]), case["row_step"])
        case["today"] = {"row": today_row(next(r for r in records if r["step"] == case["row_step"])), **engines(records, rust)}
        case["expect"] = engines(clean if clean is not None else records, rust)
        if clean is None:
            case.pop("clean_records")
    for case in section_4():
        if case["name"] in ("scalar_record_keys",):
            row = p5.record_to_row_v5(next(r for r in case["records"] if r.get("schema_version") == 2))
            import pyarrow as pa
            back = p5.row_to_record_v5(pa.RecordBatch.from_pylist([row], schema=p5.flight_schema_v5()).to_pylist()[0])
            case["row"] = {"seq": row["seq"], "record_version": row["record_version"], "schema_version": row["schema_version"],
                           "extra": json.loads(row["extra_json"]) if row.get("extra_json") else None,
                           "round_trip": back == next(r for r in case["records"] if r.get("schema_version") == 2)}
        if case["records"]:
            try:
                evidence.load_records(case["records"])
                case["today"] = {"python": "read"}
            except Exception as error:  # noqa: BLE001
                case["today"] = {"python": f"{type(error).__name__}: {str(error)[:200]}"}
        cases.append(case)
    document = {
        "contract": "WAVE_AL_CONTRACT.md",
        "base": base_sha(),
        "how_to_read": [
            "Base trial/ai-wave f71d8e0 (the base field: the commit these were computed on). The rules are WAVE_AL_CONTRACT.md.",
            "paths: row = flight.record_to_row and row_to_record in protocol 5's layout; python = build_evidence(detectors='python'); "
            "rust_raw = server_detect.server_report; rust_sdk = build_evidence(detectors='rust'); live_server = a protocol 5 release "
            "server fed by flight.send_file, read from the rank's incidents.json and its stored objects; runs = python -m tensorquill runs; "
            "mcp_get_series = the MCP tool get_series; capture = tensorquill.init and Session.log; today = the base's readers.",
            "Sections 1 and 2: row is what record_to_row must give at row.step (typed columns as [name, value] pairs in stored order, "
            "non-finite floats as the sink's strings, null for a null column or a null struct; probe_json as text; extra as extra_json "
            "parsed, null for none), and row_to_record(record_to_row(r)) == r with the map's order kept. expect is today's engines on "
            "clean_records (or on records when there are none): every engine path must report that on records. today is the base on "
            "records (never recomputed).",
            "Section 4: expect is written from the contract's rules; series maps a scalar name to [step, value] points in step order.",
            "measured: the results file under experiments/protocol-v5/ that shows the case's today on a real server.",
        ],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=1, ensure_ascii=True) + "\n")
    print(f"{len(cases)} cases written to {OUT}")


if __name__ == "__main__":
    main()
