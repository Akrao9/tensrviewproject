"""The shared cases of WAVE_AS_CONTRACT.md (the folder route): writes cases.json beside this file.

    PYTHONPATH=sdk/src python sdk/tests/fixtures/folder_route/make_cases.py

Needs pyarrow: a conversion case's row is the one today's sender builds and sends (flight.read_lines,
flight.parse_lines and flight.rows_to_send against flight.flight_schema()), read back as Arrow gives it
(batch.to_pylist()). That is what tensorquill-server stores today when the same file is sent with
flight.send_file, so the Rust converter of the folder route is held to it (contract section 2).

Every other case is data written here by hand: what the page's rules say must be true. The test
(sdk/tests/test_wave_as_contract.py) checks each against the rules written plainly there, and against
today's Python where a Python path exists. Nothing here is product code.

How a value is written in cases.json, so that both pytest and cargo read the file as strict JSON:
- a float that is NaN or infinite is {"$float": "nan"}, {"$float": "inf"} or {"$float": "-inf"};
- a map column is a list of [key, value] pairs in its stored order; a struct is an object;
- a null column, and a null field of a struct, is left out; a null entry of a map stays as null;
- a line is its text ("line"), or its bytes in hex ("line_hex") where they are not UTF-8.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tensorquill import flight

HERE = Path(__file__).resolve().parent
BASE = "1dddcce262373c1b3a552b53dff6ec639a8c2cb7"
ATTEMPT = 1764201600000  # a session's start in Unix milliseconds, as capture writes it
BS = chr(92)  # a backslash, kept out of the literals below
BOM = chr(0xFEFF)


def encode(value):
    """A row value as cases.json holds it (the module docstring)."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def encode_row(row):
    return {column: encode(value) for column, value in row.items() if value is not None}


def line_of(record, **raw):
    """A record as JsonlSink writes it, with each key of raw put in as raw JSON text instead."""
    parts = []
    for key, value in record.items():
        text = raw.pop(key) if key in raw else json.dumps(value, separators=(",", ":"))
        parts.append(json.dumps(key) + ":" + text)
    for key, text in raw.items():
        parts.append(json.dumps(key) + ":" + text)
    return "{" + ",".join(parts) + "}"


def step(run_id, **fields):
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": ATTEMPT, "step": 0, "ts": 1.5,
            "grad_norm": 0.25, **fields}


def run_of(name):
    return "fr-" + name.replace("_", "-")


# Each conversion case: (name, about, a function of the case's run ID that gives the line).
CONVERSION = [
    ("convert_a_plain_step", "a version 1 step record with the two keys capture always writes",
     lambda r: line_of(step(r, lr=[0.001, 0.0005], clip_grad_norm=1.0, step_skipped=False))),
    ("convert_grad_norm_null_is_a_null_column", "capture writes grad_norm as null when it has none: a null column",
     lambda r: line_of(step(r, grad_norm=None))),
    ("convert_a_record_with_no_version", "a record written before schema_version existed: record_version is null",
     lambda r: line_of({k: v for k, v in step(r).items() if k != "schema_version"})),
    ("convert_layers_keep_the_model_order", "layers as a map of structs, in the record's order (10 after 2)",
     lambda r: line_of(step(r, layers={
         "model.layers.2": {"mean": 0.1, "std": 1.0, "abs_max": 3.5, "zero_frac": 0.0},
         "model.layers.10": {"mean": -0.2, "std": 0.9, "abs_max": 2.5, "zero_frac": 0.01},
         "model.layers.3": {"mean": 0.0, "std": 1.1, "abs_max": 4.0, "zero_frac": 0.0}}))),
    ("convert_a_layer_field_the_struct_cannot_hold", "a bool in a float field: null in the struct, the field in extra_json",
     lambda r: line_of(step(r, layers={"blocks.0": {"mean": True, "std": 1.0, "abs_max": 2.0, "zero_frac": 0.0}}))),
    ("convert_a_layer_entry_that_is_not_an_object", "a null struct keeps the entry's place; the entry travels in extra_json",
     lambda r: line_of(step(r, layers={"blocks.0": {"mean": 0.0, "std": 1.0, "abs_max": 1.0, "zero_frac": 0.0},
                                        "blocks.1": "unreadable"}))),
    ("convert_moe_counts_stats_and_top_k", "the router's maps: counts, stats (two Int64 fields) and top_k",
     lambda r: line_of(step(r, moe={"router": [5, 3, 0, 8]}, moe_top_k={"router": 2},
                            moe_stats={"router": {"routed_tokens": 16, "entropy": 1.2, "diversity": 0.75,
                                                  "dead_experts": 1, "dead_frac": 0.25}}))),
    ("convert_the_probe_split_by_column", "an attention map of exact float32 values and histograms leave probe_json",
     lambda r: line_of(step(r, probe={"step": 0, "histograms": {"blocks.0": {"lo": -1.0, "hi": 1.0, "counts": [1, 2, 3],
                                                                             "nonfinite": 0}},
                                      "attention": {"blocks.0.attn": {"heads": 2, "map": [[0.5, 0.25], [0.125, 1.0]]}}}))),
    ("convert_scalars_with_a_bool_and_an_integer_past_2_53", "the bool and the exact integer travel in extra_json too",
     lambda r: line_of(step(r, scalars={"loss": 2.5, "flag": True, "tokens": 9007199254740993}))),
    ("convert_a_scalar_only_record", "record version 2: seq in its column, record_version 2, nothing in extra_json",
     lambda r: line_of({"schema_version": 2, "record": "scalars", "run_id": r, "rank": 0, "attempt": ATTEMPT,
                        "step": 3, "seq": 0, "scalars": {"eval_loss": 1.25}})),
    ("convert_a_key_with_no_column", "a key no column names travels in extra_json, as sorted JSON text",
     lambda r: line_of(step(r, custom={"z": 1, "a": [1, 2]}, data_wait=0.004))),
    ("convert_an_explicit_null_of_another_key", "null for a key other than ts and grad_norm travels in extra_json",
     lambda r: line_of(step(r, clip_grad_norm=None))),
    ("convert_nan_and_infinity_literals", "bare NaN, Infinity and -Infinity, which Python's json reads",
     lambda r: line_of(step(r), ts="NaN", grad_norm="Infinity", lr="[-Infinity]")),
    ("convert_a_number_past_the_float_range", "1e400 reads as infinity, as Python's json reads it",
     lambda r: line_of(step(r), ts="1e400")),
    ("convert_an_integer_past_64_bits", "a scalar of 30 digits: the nearest float in the column, the exact digits in extra_json",
     lambda r: line_of(step(r), scalars='{"big":123456789012345678901234567890}')),
    ("convert_a_duplicate_key", "the last value wins, at the place of the first: ts and a layers entry",
     lambda r: line_of(step(r), ts="1", grad_norm='0.5,"ts":2',
                       layers='{"b":{"mean":1},"a":{"mean":2},"b":{"mean":3}}')),
    ("convert_a_lone_surrogate_in_a_list", "Arrow cannot hold a lone surrogate: the list travels in extra_json",
     lambda r: line_of(step(r), weights_unchanged='["' + BS + 'ud800"]')),
    ("convert_top_k_past_int32", "moe_top_k is Int32: 3000000000 travels in extra_json and the key is counted as moved",
     lambda r: line_of(step(r, moe_top_k={"router": 3000000000}))),
    ("convert_a_whole_float_where_an_integer_is_read", "2.0 in moe_top_k is not an integer: the map travels in extra_json",
     lambda r: line_of(step(r), moe_top_k='{"router":2.0}')),
    ("convert_float_text_in_a_json_column", "data_json is Python's text (1e+16, 1e-07): the Rust text is compared as JSON values",
     lambda r: line_of(step(r, data={"z": 1e-7, "a": 1e16, "b": 0.1}))),
    ("convert_non_ascii_text", "UTF-8 and an escape in a key with no column: extra_json escapes them (ensure_ascii)",
     lambda r: line_of(step(r, note="caf" + chr(233)), label='"caf' + BS + 'u00e9"')),
    ("convert_a_line_ending_in_a_carriage_return", "a CRLF line: the carriage return is whitespace to the JSON reader",
     lambda r: line_of(step(r)) + chr(13)),
    ("convert_a_line_with_a_byte_order_mark", "Python's json reads a line that starts with a UTF-8 byte order mark",
     lambda r: BOM + line_of(step(r))),
    ("convert_a_line_that_is_not_utf8", "a byte that is not UTF-8: not a record, skipped and counted",
     lambda r: line_of(step(r)).encode()[:-1] + b',"x":"' + bytes([0xFF]) + b'"}'),
    ("convert_a_line_that_is_not_an_object", "a JSON array: skipped and counted",
     lambda r: "[1,2]"),
    ("convert_a_line_of_spaces", "a blank line: passed over and not counted",
     lambda r: "   "),
    ("convert_a_line_that_is_not_json", "half a record, as a killed process leaves it: skipped and counted",
     lambda r: '{"schema_version":1,"run_id":"' + r + '","st'),
    ("convert_a_line_of_another_run", "a record of another run ID than the file's: skipped and counted",
     lambda r: line_of(step("another-run"))),
    ("convert_a_line_with_no_run_id", "a record with no run_id: skipped and counted",
     lambda r: line_of({k: v for k, v in step(r).items() if k != "run_id"})),
    ("convert_a_negative_step", "an identity field the sender cannot send: left out and counted",
     lambda r: line_of(step(r, step=-1))),
    ("convert_a_whole_float_step", "a step of 1.0 is not an integer: left out and counted",
     lambda r: line_of(step(r), step="1.0")),
    ("convert_a_step_past_int64", "a step past Int64: left out and counted",
     lambda r: line_of(step(r, step=2**63))),
    ("convert_a_rank_past_int32", "a rank past Int32: left out and counted",
     lambda r: line_of(step(r, rank=2**31))),
    ("convert_a_record_version_this_package_does_not_read",
     "schema_version 7 is sent with row version 7, for the server to drop and count",
     lambda r: line_of(step(r, schema_version=7))),
    ("convert_a_version_2_record_that_is_not_scalar_only",
     "record version 2 without record: scalars is left out and counted",
     lambda r: line_of(step(r, schema_version=2))),
]


def conversion_case(name, about, make):
    import pyarrow  # noqa: F401  the generator needs it; the test reads the file without it

    run_id = run_of(name)
    line = make(run_id)
    data = line if isinstance(line, bytes) else line.encode("utf-8")
    records, skipped = flight.parse_lines([data], run_id)
    moved, refused = [], []
    batch, steps, left_out = flight.rows_to_send(records, flight.flight_schema(), moved, refused)
    rows = batch.to_pylist()
    case = {"name": name, "section": 2, "kind": "conversion", "about": about, "run_id": run_id}
    try:
        case["line"] = data.decode("utf-8")
    except UnicodeDecodeError:
        case["line_hex"] = data.hex()
    case.update(skipped=skipped, left_out=left_out, moved=[key for _, key in moved],
                row=encode_row(rows[0]) if rows else None)
    # Whether the row without Arrow's check (flight._row, no pyarrow) is already the row sent: where it is not,
    # Arrow refused a value and the sender's strict path moved it to extra_json (wave AJ).
    plain = None
    if records and not left_out:
        try:
            plain = encode_row(flight._row(records[0]))
        except Exception:  # noqa: BLE001
            plain = None
    case["arrow_moves"] = bool(rows) and plain != case["row"]
    # What flight.record_to_row makes of the record, given the schema (section 2.1): the sender's row, or the
    # identity field it refuses, where the sender leaves the line out or sends it for the server to count.
    if records:
        try:
            same = encode_row(flight.record_to_row(records[0], flight.flight_schema())) == case["row"]
            case["record_to_row"] = "same" if same else "differs"
        except flight.MalformedRecordError as error:
            case["record_to_row"] = {"refuses": error.field}
    else:
        case["record_to_row"] = None
    return case


def mixed_case():
    """Section 2.5: one file holding a record with no version, a step record and a scalar-only record."""
    run_id = "fr-mixed"
    lines = [line_of({k: v for k, v in step(run_id, step=0).items() if k != "schema_version"}),
             line_of(step(run_id, step=1)),
             line_of({"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": ATTEMPT,
                      "step": 1, "seq": 0, "scalars": {"eval_loss": 1.25}}),
             line_of(step(run_id, step=2))]
    records, skipped = flight.parse_lines([line.encode() for line in lines], run_id)
    batch, steps, left_out = flight.rows_to_send(records, flight.flight_schema())
    rows = batch.to_pylist()
    return {"name": "lines_a_file_of_step_and_scalar_only_records", "section": 2, "kind": "mixed",
            "about": "step records of versions 0 and 1 and a scalar-only record in one file, in one batch",
            "run_id": run_id, "file": "\n".join(lines) + "\n", "training_steps": steps,
            "rows": [{"step": r["step"], "seq": r["seq"], "record_version": r["record_version"]} for r in rows],
            "skipped": skipped, "left_out": left_out}


def lines_cases():
    a = line_of(step("fr-lines", step=0))
    b = line_of(step("fr-lines", step=1))
    cr = chr(13)
    return [
        {"name": "lines_a_last_line_with_no_newline", "section": 2, "kind": "lines",
         "about": "a last line with no newline is not read until its newline comes; the offset stops before it",
         "run_id": "fr-lines", "max_bytes": 4 * 1024 * 1024,
         "reads": [{"file": a + "\n" + b, "lines": [a], "offset": len(a) + 1},
                   {"file": a + "\n" + b + "\n", "lines": [b], "offset": len(a) + len(b) + 2}]},
        {"name": "lines_blank_lines_and_crlf", "section": 2, "kind": "lines",
         "about": "a blank line is read past and not counted; a CRLF line is one line",
         "run_id": "fr-lines", "max_bytes": 4 * 1024 * 1024,
         "reads": [{"file": a + cr + "\n\n" + b + "\n", "lines": [a + cr, "", b], "offset": len(a) + len(b) + 4}]},
        {"name": "lines_one_line_longer_than_a_batch", "section": 2, "kind": "lines",
         "about": "a line longer than the batch's byte cap is read alone, whole",
         "run_id": "fr-lines", "max_bytes": 16,
         "reads": [{"file": a + "\n" + b + "\n", "lines": [a], "offset": len(a) + 1}]},
    ]


def torn_case():
    good = line_of(step("fr-torn", step=0))
    return {"name": "killed_job_torn_tail_then_a_new_session", "section": 6, "kind": "torn",
            "about": ("a killed process left half a line; the next session's sink gives it a newline first, so the "
                      "fragment is a line of its own that is skipped and counted, and the new record is read"),
            "run_id": "fr-torn", "file": good + "\n" + '{"schema_version":1,"run_id":"fr-torn","st',
            "appended": step("fr-torn", step=1), "expect": {"records": [0, 1], "skipped": 1}}


def names_cases():
    run, rank, pid = "run-slurm1234.0", 3, 4242
    stem, second = f"{run}.rank{rank}", f"{run}.rank{rank}.{pid}"
    return [
        {"name": "names_records_and_second_writer", "section": 7, "kind": "names",
         "about": "the two names a rank's records take, read from the right; a copy is not a records file",
         "files": {f"{stem}.jsonl": [run, rank, None], f"{second}.jsonl": [run, rank, pid],
                   f"{stem}.bak.jsonl": None, f"{run}.rank{rank}.x.jsonl": None}},
        {"name": "names_beat_file_and_stop_notice_are_no_records_file", "section": 5, "kind": "names",
         "about": "the proposed beat file and stop notice names match no name a reader takes for something else",
         "run_id": run, "rank": rank, "pid": pid,
         "proposed": [f"{stem}.beat.json", f"{stem}.stop.json", f"{second}.beat.json", f"{second}.stop.json"],
         "taken": [f"{stem}.jsonl", f"{stem}.end.json", f"{stem}.offset", f"{run}.run.json"]},
    ]


# The end states a folder reader gives a rank (sections 5 and 6), from what the folder holds and the beat
# file's freshness by the server's own clock. Expectations written by hand from the page's rules.
STOPPED = {"state": "capture_stopped", "reason": "sink_error", "step": 7}
CLOSED = {"state": "closed", "outcome": "closed"}
END_STATES = [
    ("folder_closed_reads_finished", {"marker": "closed"}, {"state": "finished", "ended": CLOSED}),
    ("folder_closed_by_an_exception_reads_finished", {"marker": "raised"},
     {"state": "finished", "ended": {"state": "closed", "outcome": "raised", "exception": "RuntimeError"}}),
    ("folder_stop_notice_reads_capture_stopped", {"stop": "present"}, {"state": "capture_stopped", "ended": STOPPED}),
    ("folder_stop_notice_and_marker_reads_capture_stopped", {"marker": "closed", "stop": "present"},
     {"state": "capture_stopped", "ended": STOPPED}),
    ("folder_marker_written_after_close_is_no_end", {"marker": "written_after_close", "beat": "fresh"}, {"state": "active"}),
    ("folder_stop_notice_of_another_size_is_no_end", {"stop": "stale", "beat": "fresh"}, {"state": "active"}),
    ("folder_beat_fresh_and_working_reads_active", {"beat": "fresh", "work_still_for": 10}, {"state": "active"}),
    ("folder_beat_fresh_and_work_still_reads_silent", {"beat": "fresh", "work_still_for": 400}, {"state": "silent"}),
    ("killed_job_beat_stale_inside_the_floor_reads_reconnecting", {"beat": "stale", "stale_for": 60},
     {"state": "reconnecting"}),
    ("killed_job_beat_stale_past_the_floor_reads_lost", {"beat": "stale", "stale_for": 400}, {"state": "lost"}),
    ("killed_job_never_reads_finished", {"beat": "stale", "stale_for": 100000}, {"state": "lost"}),
    ("folder_no_beat_file_is_not_checked", {"beat": "none"}, {"state": "active", "liveness": "not_checked"}),
    ("folder_no_beat_file_and_closed_reads_finished", {"beat": "none", "marker": "closed"},
     {"state": "finished", "ended": CLOSED}),
    ("folder_network_filesystem_is_not_checked", {"beat": "stale", "stale_for": 400, "fs": "network"},
     {"state": "active", "liveness": "not_checked"}),
    ("folder_network_filesystem_stop_notice_still_reads", {"stop": "present", "fs": "network"},
     {"state": "capture_stopped", "ended": STOPPED}),
]

# The beat file's freshness at each of the server's looks (section 5): the server's own monotonic clock in
# seconds, the file's counter, and a wall time inside the file, which is never read.
BEATS = [
    ("beat_a_counter_that_moves_is_fresh", 15,
     [(0, 1, None), (15, 2, None), (30, 3, None), (45, 4, None)], ["fresh", "fresh", "fresh", "fresh"]),
    ("beat_a_counter_that_stops_goes_stale_after_three_intervals", 15,
     [(0, 1, None), (15, 2, None), (30, 2, None), (59, 2, None), (60, 2, None), (90, 2, None)],
     ["fresh", "fresh", "fresh", "fresh", "stale", "stale"]),
    ("beat_the_first_look_counts_as_a_beat", 15,
     [(0, 9, None), (44, 9, None), (45, 9, None)], ["fresh", "fresh", "stale"]),
    ("beat_a_short_interval_is_stale_after_30_seconds", 1,
     [(0, 1, None), (10, 1, None), (29, 1, None), (30, 1, None)], ["fresh", "fresh", "fresh", "stale"]),
    ("beat_a_wall_time_in_the_file_is_never_read", 15,
     [(0, 1, 4102444800.0), (30, 2, 0.0), (60, 2, 4102444800.0), (75, 2, 1.0)], ["fresh", "fresh", "fresh", "stale"]),
    ("beat_a_counter_that_comes_back_is_fresh_again", 15,
     [(0, 1, None), (50, 1, None), (55, 2, None)], ["fresh", "stale", "fresh"]),
]

# The default derived store's key (section 3): whether two opens share a store.
STORE_KEYS = [
    ("store_one_folder_is_one_store", {"real": "/data/runs/exp1", "fs": "local", "host": "gpu-a"},
     {"real": "/data/runs/exp1", "fs": "local", "host": "gpu-a"}, True),
    ("store_two_folders_are_two_stores", {"real": "/data/runs/exp1", "fs": "local", "host": "gpu-a"},
     {"real": "/data/runs/exp2", "fs": "local", "host": "gpu-a"}, False),
    ("store_network_cache_two_machines_are_two_stores", {"real": "/home/u/runs/exp1", "fs": "network", "host": "login-1"},
     {"real": "/home/u/runs/exp1", "fs": "network", "host": "login-2"}, False),
    ("store_unknown_type_two_machines_are_two_stores", {"real": "/home/u/runs/exp1", "fs": "unknown", "host": "a"},
     {"real": "/home/u/runs/exp1", "fs": "unknown", "host": "b"}, False),
    ("store_network_cache_one_machine_is_one_store", {"real": "/home/u/runs/exp1", "fs": "network", "host": "login-1"},
     {"real": "/home/u/runs/exp1", "fs": "network", "host": "login-1"}, True),
]

# A server stopped and started again during a run (section 10). Lines are one attempt's steps 0 to n-1 unless
# "attempts" says otherwise ([attempt, first step, last step + 1], in file order); a part is stored every
# "flush" lines; "kill" says how far the first server got: lines read, parts stored, parts whose offset it wrote.
RESTARTS = [
    ("restart_before_the_first_part", {"lines": 30, "flush": 10, "kill": {"after_lines": 5, "parts": 0, "offset_parts": 0}}),
    ("restart_after_a_part_before_its_offset",
     {"lines": 30, "flush": 10, "kill": {"after_lines": 15, "parts": 1, "offset_parts": 0}}),
    ("restart_after_a_part_and_its_offset",
     {"lines": 30, "flush": 10, "kill": {"after_lines": 25, "parts": 2, "offset_parts": 2}}),
    ("restart_lines_written_while_the_server_was_down",
     {"lines": 50, "flush": 10, "written_while_down": 20, "kill": {"after_lines": 20, "parts": 2, "offset_parts": 1}}),
    ("restart_a_resumed_job_rewrites_steps_under_a_new_attempt",
     {"attempts": [[1, 0, 10], [2, 5, 15]], "flush": 4, "kill": {"after_lines": 12, "parts": 3, "offset_parts": 2}}),
]

END_MESSAGES = [
    ("end_closed", {"end_version": True, "rank_end": CLOSED, "close": "clean"}, {"state": "finished", "ended": CLOSED}),
    ("end_closed_raised",
     {"end_version": True, "rank_end": {"state": "closed", "outcome": "raised", "exception": "RuntimeError"},
      "close": "clean"},
     {"state": "finished", "ended": {"state": "closed", "outcome": "raised", "exception": "RuntimeError"}}),
    ("end_capture_stopped_every_beat_lost",
     {"end_version": True, "beats_said_stop": False, "close": "clean", "rank_end": STOPPED},
     {"state": "capture_stopped", "ended": STOPPED}),
    ("end_capture_stopped_before_any_row",
     {"end_version": True, "rows": 0, "close": "clean",
      "rank_end": {"state": "capture_stopped", "reason": "sink_error", "step": None}},
     {"state": "capture_stopped", "ended": {"state": "capture_stopped", "reason": "sink_error", "step": None}}),
    ("end_declared_but_missing", {"end_version": True, "rank_end": None, "close": "clean"},
     {"state": "reconnecting", "then": "lost", "ended": {"state": "without_notice", "seen": "clean_close_without_end"}}),
    ("end_from_an_older_sender", {"end_version": False, "rank_end": None, "close": "clean"}, {"state": "finished"}),
    ("end_a_dropped_stream_is_no_end", {"end_version": True, "rank_end": None, "close": "error"},
     {"state": "reconnecting", "then": "lost"}),
]


def main():
    cases = [conversion_case(*spec) for spec in CONVERSION]
    cases += lines_cases()
    cases.append(mixed_case())
    cases.append(torn_case())
    cases += names_cases()
    cases += [{"name": n, "section": 6 if n.startswith("killed_job") else 5, "kind": "end_state",
               "about": "what the folder holds, and the beat file by the server's clock", "folder": f, "expect": e}
              for n, f, e in END_STATES]
    cases += [{"name": n, "section": 5, "kind": "beat", "about": "the beat file's freshness at each look",
               "interval_seconds": i, "looks": [{"at": a, "beat": b, **({"written_at": w} if w is not None else {})}
                                                for a, b, w in looks], "expect": e} for n, i, looks, e in BEATS]
    cases += [{"name": n, "section": 3, "kind": "store_key", "about": "whether two opens share a default store",
               "first": a, "second": b, "same": s} for n, a, b, s in STORE_KEYS]
    cases += [{"name": n, "section": 10, "kind": "restart", "about": "a server stopped and started again during a run",
               **spec, "expect": {"each_step_once": True, "lost": []}} for n, spec in RESTARTS]
    cases += [{"name": n, "section": 9, "kind": "end_message", "about": "a rank's end on today's stream",
               "stream": s, "expect": e} for n, s, e in END_MESSAGES]
    cases.append({"name": "first_open_figure", "section": 8, "kind": "figure",
                  "about": "the owner's figure (Nov 27): the page within 5 s, 50 MB of lines a second per core",
                  "page_seconds": 5, "mb_per_second_per_core": 50,
                  "folders": [{"mb": 31000, "seconds_one_core": 620}, {"mb": 7000, "seconds_one_core": 140}]})
    document = {
        "contract": "WAVE_AS_CONTRACT.md",
        "base": BASE,
        "how_to_read": ("Each case is data the folder route must make true. Conversion cases (section 2) hold a line "
                        "and the row today's sender stores for it; the others hold what the page's rules give. "
                        "Values are written as make_cases.py says."),
        "names": {"beat_file": "{stem}.beat.json", "stop_notice": "{stem}.stop.json", "end_marker": "{stem}.end.json",
                  "end_version_key": "end_version", "end_key": "rank_end", "ended_key": "ended",
                  "beat_marker": "tensorquill_beat", "stop_marker": "tensorquill_capture_stopped"},
        "rules": {"silent_floor_seconds": 300, "beat_stale_intervals": 3, "beat_stale_min_seconds": 30,
                  "store_key_hex_digits": 16},
        "cases": cases,
    }
    (HERE / "cases.json").write_text(json.dumps(document, indent=1, ensure_ascii=True, allow_nan=False) + "\n")
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    main()
