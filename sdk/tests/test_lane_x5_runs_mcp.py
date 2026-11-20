"""Lane X5 (wave X): the run listing, the MCP server and the compact file, one test or more for
each row it fixes.

- CLI-13, open part: `runs` reads every record of a rank whose records are in several files, so
  an attempt seen only in the middle of a file orders the attempts as `compare` orders them
  (contract 1), and the `end` cell is taken from the file of the rank's newest attempt.
- DET-17, open part: `manifest.py`, `runs.py` and `mcp_server.py` read `rank=<n>` folder names
  with `evidence._rank_number` (ASCII digits only).
- CLI-14: the MCP server serves the probe's head and attention series.
- DET-13: the compact file's table at a window's first step is made when the full evidence has
  no earlier step that routed tokens either, and a later starvation onset gets a window.

The records are built here, so nothing depends on another lane's test file.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import compare as cmp
from tensorquill import runs as runs_module
from tensorquill.sink import write_end_marker

RUN = "lane-x5"


def rows(attempt: int, steps: range, first_ts: float, *, rank: int = 0, run: str = RUN) -> list[dict[str, Any]]:
    return [{"schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step, "ts": first_ts + at,
             "layers": {}, "grad_norm": 1.0, "grad_norms": {"w": 1.0}, "lr": [0.1]}
            for at, step in enumerate(steps)]


def write(path: Path, records: list[dict[str, Any]], *, tail: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail, encoding="utf-8")
    return path


def only_row(folder: Path) -> dict[str, Any]:
    (row,) = tensorquill.list_runs(folder)
    return row


# --- CLI-13 -----------------------------------------------------------------------------------
#
# One run and rank in two files. In `one.jsonl` attempt B is seen only in the middle: A, then B,
# then C. `two.jsonl` holds B's later rows. Every record: the order is A, B, C (one file orders
# all three), C replaces B from step 25, and the run ends at C's step 34. From the files' first
# and last records only, B is not seen in `one.jsonl`, so the order went by `ts` (A, C, B), B was
# taken as the newest, and the row said 99.

A, B, C = 900, 500, 2


def mid_file_case(folder: Path, first: str = "one.jsonl", second: str = "two.jsonl") -> tuple[Path, Path]:
    one = write(folder / first, rows(A, range(0, 60), 1000.0) + rows(B, range(20, 30), 3000.0)
                + rows(C, range(25, 35), 1500.0))
    two = write(folder / second, rows(B, range(30, 100), 3010.0))
    return one, two


def test_cli13_an_attempt_seen_only_mid_file_orders_the_attempts_as_compare_does(tmp_path):
    folder = tmp_path / "out"
    mid_file_case(folder)
    assert cmp.open_run(RUN, folder)["last_step"] == 34
    assert only_row(folder)["last_step"] == 34


def test_cli13_the_listing_agrees_with_compare_whichever_file_sorts_first(tmp_path):
    for first, second in (("one.jsonl", "two.jsonl"), ("z-one.jsonl", "a-two.jsonl")):
        folder = tmp_path / first
        mid_file_case(folder, first, second)
        assert only_row(folder)["last_step"] == cmp.open_run(RUN, folder)["last_step"] == 34


def test_cli13_a_copy_a_torn_line_and_files_that_disagree_read_as_compare_reads_them(tmp_path):
    # A copy of a file is read once.
    copy = tmp_path / "copy"
    one, _ = mid_file_case(copy)
    (copy / "one.bak.jsonl").write_bytes(one.read_bytes())
    assert only_row(copy)["last_step"] == cmp.open_run(RUN, copy)["last_step"] == 34
    # A torn line in the middle of a file (a killed writer, then a restart) is left out.
    torn = tmp_path / "torn"
    write(torn / "one.jsonl", rows(A, range(0, 60), 1000.0), tail='{"run_id": "lane-x5", "step": 7')
    with (torn / "one.jsonl").open("a") as file:
        file.write("\n" + "".join(json.dumps(r) + "\n" for r in rows(B, range(20, 30), 3000.0) + rows(C, range(25, 35), 1500.0)))
    write(torn / "two.jsonl", rows(B, range(30, 100), 3010.0))
    assert only_row(torn)["last_step"] == cmp.open_run(RUN, torn)["last_step"] == 34
    # Files that disagree on the order (here only through the attempt in the middle): by number,
    # and the row says so, as compare does.
    disagree = tmp_path / "disagree"
    write(disagree / "one.jsonl", rows(900, range(0, 10), 1000.0) + rows(2, range(5, 8), 2000.0)
          + rows(900, range(10, 12), 2100.0))
    write(disagree / "two.jsonl", rows(2, range(15, 20), 3000.0) + rows(900, range(12, 13), 4000.0))
    opened = cmp.open_run(RUN, disagree)
    row = only_row(disagree)
    assert opened["read_from"]["attempt_order"] == "numeric"
    assert row["last_step"] == opened["last_step"] == 12
    assert "disagree" in json.dumps(row)


def test_cli13_the_end_cell_is_the_newest_attempts_file(tmp_path):
    # The first file by name is the old attempt's, never closed; the resume's file closed.
    folder = tmp_path / "out"
    write(folder / "a-old.jsonl", rows(1, range(0, 50), 1000.0))
    new = write(folder / "b-new.jsonl", rows(2, range(20, 40), 2000.0))
    assert write_end_marker(new, run_id=RUN, rank=0, attempt=2)
    row = only_row(folder)
    assert row["last_step"] == 39
    assert row["end"]["state"] == "closed" and row["end"]["closed_ranks"] == [0]

    # The other way round: the old attempt's file closed (a raised session), the newest did not.
    other = tmp_path / "other"
    old = write(other / "a-old.jsonl", rows(1, range(0, 50), 1000.0))
    write(other / "b-new.jsonl", rows(2, range(20, 40), 2000.0))
    assert write_end_marker(old, run_id=RUN, rank=0, attempt=1, exception=RuntimeError())
    end = only_row(other)["end"]
    assert end["state"] == "not_closed" and end["not_closed_ranks"] == [0] and "raised_ranks" not in end
    assert "b-new.jsonl" in end["detail"]

    # A copy beside the newest attempt's file has no marker of its own: the marker of the file
    # the newest attempt wrote still says the rank closed.
    copy = tmp_path / "copy"
    closed = write(copy / f"{RUN}.rank0.jsonl", rows(1, range(0, 30), 1000.0))
    assert write_end_marker(closed, run_id=RUN, rank=0, attempt=1)
    (copy / "x.bak.jsonl").write_bytes(closed.read_bytes())
    assert only_row(copy)["end"]["state"] == "closed"


# --- DET-17 -----------------------------------------------------------------------------------
#
# `rank=<n>` is ASCII digits, as the server writes it (`server/src/storage.rs`) and as the
# evidence readers read it (`evidence._rank_number`). `rank=٣` (Arabic-Indic three) passed
# `isdecimal()` in these modules and was read as rank 3; `²` and `--1` already failed it.

STRAY = ["rank=٣", "rank=²", "rank=--1", "rank=+1", "rank= 1", "rank=x"]


def test_det17_every_module_reads_rank_folders_with_the_evidence_readers_parser(tmp_path):
    from tensorquill import manifest as manifest_module
    from tensorquill import mcp_server
    from tensorquill.evidence import _rank_number

    for name in [*STRAY, "rank=3", "rank=0"]:
        folder = tmp_path / "runs" / "r" / name
        folder.mkdir(parents=True)
        is_rank = _rank_number(name) is not None
        # The manifest of a rank folder is its run folder's; any other folder's is its own.
        (where,) = manifest_module.manifest_candidates(folder)
        assert where == (folder.parent if is_rank else folder) / manifest_module.STORE_MANIFEST, name
        # The MCP server started on a rank folder reads that rank of its run folder.
        kind, path, rank = mcp_server.Source(folder).locate()
        assert (kind, path, rank) == (("store", folder.parent, _rank_number(name)) if is_rank else ("store", folder, None)), name
        # A folder that holds only such a folder is a run folder only for a real rank folder.
        lone = tmp_path / "lone" / name.replace("=", "_") / "runid"
        (lone / name).mkdir(parents=True)
        assert runs_module._is_run_folder(lone) is is_rank, name
        shutil.rmtree(folder)


def test_det17_a_store_with_a_stray_rank_folder_is_not_listed_as_a_run(tmp_path):
    store = tmp_path / "store" / "runs"
    (store / "real" / "rank=0").mkdir(parents=True)
    (store / "stray" / "rank=٣").mkdir(parents=True)
    assert [run["run_id"] for run in runs_module.discover_runs(tmp_path / "store")] == ["real"]


# --- CLI-14 -----------------------------------------------------------------------------------
#
# The probe's head numbers (`probe.head`) and attention statistics (`probe.attention`) are
# served by `get_series` as kinds `head` and `attention`, read from the rank's step records as
# every other series is: the same rank and attempt handling, `max_points`, `start` and `end`.

HEAD, ATTN = "lm_head", "model.layers.0.self_attn"


def probed(attempt: int, steps: range, first_ts: float, *, rank: int = 0, scale: float = 1.0,
           every: int = 5) -> list[dict[str, Any]]:
    out = rows(attempt, steps, first_ts, rank=rank)
    for record in out:
        step = record["step"]
        if step % every:
            continue
        head = {"module": HEAD, "found_by": "output_embeddings", "positions": 64, "nonfinite_positions": 0,
                "vocab": 100, "calls": 1, "input_dtype": "bfloat16", "weight_dtype": "bfloat16",
                "norm": scale * (10.0 + step), "mean": -0.5, "log_z": 4.0, "gap": 0.01, "gap_relative": 0.001}
        if step == 15:
            head = {"module": HEAD, "found_by": "output_embeddings", "not_read": "the head was not called"}
        record["probe"] = {
            "histograms": {},
            "head": head,
            "attention": {ATTN: {"max_logit": [scale * step, 2.0, float("-inf")], "entropy": [1.0, 1.0, 1.0],
                                 "entropy_share": [0.2, 0.4, float("nan")], "masked_rows": 0, "invalid_rows": 0,
                                 "rows": 96, "query_len": 8, "key_len": 8, "map": [[0.5, 0.5], [0.5, 0.5]]}},
        }
    return out


def series(path: Path, **arguments: Any) -> dict[str, Any]:
    from tensorquill import mcp_server

    return mcp_server.get_series(mcp_server.Source(path), arguments)


def series_error(path: Path, **arguments: Any) -> str:
    from tensorquill import mcp_server

    with pytest.raises(mcp_server.ToolError) as error:
        mcp_server.get_series(mcp_server.Source(path), arguments)
    return str(error.value)


def test_cli14_the_head_series_is_served_from_the_probes_that_read_the_head(tmp_path):
    capture = write(tmp_path / "x.jsonl", probed(1, range(0, 50), 1000.0))
    answer = series(capture, name=HEAD)
    assert answer["kind"] == "head"
    # Probes at every fifth step; the one at 15 did not read the head, so it is not a point.
    assert answer["steps"] == [0, 5, 10, 20, 25, 30, 35, 40, 45]
    assert answer["values"]["norm"][:3] == [10.0, 15.0, 20.0]
    assert {"norm", "mean", "log_z", "gap", "gap_relative", "positions"} <= set(answer["values"])
    # Text fields are not series: no column counted as not finite.
    assert not {"module", "found_by", "input_dtype", "weight_dtype"} & set(answer["values"])
    assert answer["over_range"]["norm"]["max"] == 55.0 and "not_finite" not in answer["over_range"]["norm"]
    assert series(capture, name=HEAD, kind="head", stat="gap")["values"] == {"gap": [0.01] * 9}


def test_cli14_the_attention_series_is_the_numbers_the_detectors_judge(tmp_path):
    capture = write(tmp_path / "x.jsonl", probed(1, range(0, 50), 1000.0))
    answer = series(capture, name=ATTN)
    assert answer["kind"] == "attention" and len(answer["steps"]) == 10
    # The mean of the heads' finite entropy shares, the largest finite logit, the heads counted.
    assert answer["values"]["entropy_share"][0] == pytest.approx(0.3)
    assert answer["values"]["max_logit"][:3] == [2.0, 5.0, 10.0]
    assert answer["values"]["heads"] == [2] * 10
    assert "map" not in answer["values"] and "entropy" not in answer["values"]


def test_cli14_probe_series_keep_the_bounds_errors_rank_and_attempt_handling(tmp_path):
    # Thinned to max_points with the stride said, and start and end narrow the range.
    long = write(tmp_path / "long.jsonl", probed(1, range(0, 400), 1000.0, every=1))
    thinned = series(long, name=HEAD, max_points=10)
    assert thinned["points"] <= 10 and thinned["stride"] > 1 and "note" in thinned
    assert series(long, name=ATTN, start=100, end=104)["steps"] == [100, 101, 102, 103, 104]
    # A resume from step 22 rolls back attempt 1's later probes: they are not the run's.
    resumed = write(tmp_path / "resumed.jsonl", probed(1, range(0, 40), 1000.0) + probed(2, range(22, 40), 2000.0, scale=-1.0))
    answer = series(resumed, name=HEAD, stat="norm")
    assert answer["steps"] == [0, 5, 10, 20, 25, 30, 35]
    assert answer["values"]["norm"] == [10.0, 15.0, 20.0, 30.0, -35.0, -40.0, -45.0]
    # rank= reads that rank's own records.
    two = write(tmp_path / "two.jsonl", probed(1, range(0, 20), 1000.0) + probed(1, range(0, 20), 1000.0, rank=1, scale=3.0))
    assert series(two, name=HEAD, rank=1, stat="norm")["values"]["norm"][:2] == [30.0, 45.0]
    assert series(two, name=HEAD, stat="norm")["values"]["norm"][:2] == [10.0, 15.0]
    # The errors other series give.
    assert "is a head here, not a layer" in series_error(two, name=HEAD, kind="layer")
    assert "no head is named 'lm_heads'" in series_error(two, name="lm_heads", kind="head")
    assert f"{HEAD} (head)" in series_error(two, name="lm_heads", kind="head")
    assert "has no `max`" in series_error(two, name=ATTN, stat="max")
    assert "`expert` is for a router's series" in series_error(two, name=ATTN, expert=1)
    assert "rank 4 is not in the capture" in series_error(two, name=HEAD, rank=4)


def test_cli14_a_name_that_is_also_a_layer_stays_the_layer_unless_kind_says_otherwise(tmp_path):
    records = probed(1, range(0, 20), 1000.0)
    for record in records:
        record["layers"] = {ATTN: {"mean": 1.0, "std": 2.0, "abs_max": 3.0, "zero_frac": 0.0}}
    capture = write(tmp_path / "x.jsonl", records)
    layer = series(capture, name=ATTN)
    assert layer["kind"] == "layer" and len(layer["steps"]) == 20  # as before the probe kinds were served
    assert "kind='attention'" in layer["also"]
    assert series(capture, name=ATTN, kind="attention")["kind"] == "attention"


def test_cli14_an_attention_incident_points_at_the_attention_series():
    from tensorquill import mcp_server

    incident = {"kind": "attention_collapse", "step": 40, "modules": [{"module": ATTN, "value": 0.01, "baseline": 0.3}]}
    (call,) = mcp_server._look_next(incident, rank=1)
    assert call["arguments"] == {"name": ATTN, "kind": "attention", "stat": "entropy_share", "rank": 1}
    growth = {"kind": "attention_logit_growth", "step": 40, "modules": [{"module": ATTN, "value": 90.0, "baseline": 9.0}]}
    assert mcp_server._look_next(growth)[0]["arguments"] == {"name": ATTN, "kind": "attention", "stat": "max_logit"}


# --- DET-13 -----------------------------------------------------------------------------------
#
# Hand-built evidence of two routers of four experts over 200 steps, with an imbalance at 100.
# Router 1 routes no token before step 97; router 2 routes from step 0.


def sample(step: int, layer: int, counts: list[int], **flags: Any) -> dict[str, Any]:
    return {"step": step, "layer": layer, "router": f"model.layers.{layer}.mlp.router", "counts": counts,
            "routed_tokens": sum(counts), "diversity": 0.9, "overloaded": [], "dead": [], "starved": [], **flags}


def moe_evidence(*, quiet_until: int = 97, incidents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    samples = []
    for step in range(200):
        samples.append(sample(step, 1, [0, 0, 0, 0] if step < quiet_until else [10 + step % 3, 10, 10, 10]))
        samples.append(sample(step, 2, [12, 9 + step % 2, 10, 10]))
    return {"run_id": RUN, "expert_count": 4, "layer_count": 2, "samples": samples,
            "incidents": incidents if incidents is not None else [{"kind": "moe_imbalance", "step": 100, "layer": 2}]}


def test_det13_a_window_whose_router_routed_nothing_before_it_still_has_its_table():
    from tensorquill.compact import SamplesOmitted, compact_evidence

    full = moe_evidence()
    small = json.loads(json.dumps(compact_evidence(full, window=3)))  # as read back from the file
    assert small["compact"]["kept_steps"] == [[97, 103]]
    # Router 1 routed no token before the window: the full evidence compares step 97 with the
    # last earlier step, which routed none ("n/a" deltas). The compact file gives that table.
    for step in (97, 98, 103):
        assert tensorquill.expert_load_table(small, step, 1) == tensorquill.expert_load_table(full, step, 1)
    assert tensorquill.expert_load_table(small, 97, 1).splitlines()[0] == "**Layer 01 expert load** · step 97 vs. 96"
    # The step it is compared with routed nothing, so it is none of the samples other readers use.
    from tensorquill import mcp_server

    assert [s for s, _ in mcp_server._sample_rows("model.layers.1.mlp.router", small, None)] == list(range(97, 104))
    # Router 2 routed before the window, and that step is left out: still refused, never +0.00.
    with pytest.raises(SamplesOmitted, match="compared with the last earlier step"):
        tensorquill.expert_load_table(small, 97, 2)
    assert tensorquill.expert_load_table(small, 98, 2) == tensorquill.expert_load_table(full, 98, 2)
    # A compact file written before the marker said so refuses as it did.
    old = json.loads(json.dumps(small))
    old["compact"].pop("unrouted_before")
    with pytest.raises(SamplesOmitted, match="compared with"):
        tensorquill.expert_load_table(old, 97, 1)


def test_det13_a_later_starvation_onset_gets_a_window_of_its_own():
    from tensorquill.compact import ONSET_WINDOWS, compact_evidence

    experts = [{"layer": 2, "expert": 1, "step": 40, "window_share": 0.001},
               {"layer": 2, "expert": 2, "step": 40, "window_share": 0.002},
               {"layer": 1, "expert": 3, "step": 150, "window_share": 0.003}]
    starved = {"kind": "starved_experts", "step": 40, "experts": experts}
    full = moe_evidence(quiet_until=0, incidents=[{"kind": "moe_imbalance", "step": 100, "layer": 2}, starved])
    marker = compact_evidence(full, window=2)["compact"]
    assert marker["around"] == [{"kind": "moe_imbalance", "step": 100}, {"kind": "starved_experts", "step": 40},
                                {"kind": "starved_expert_onset", "step": 150}]
    assert marker["kept_steps"] == [[38, 42], [98, 102], [148, 152]]
    assert "starved_onsets_without_window" not in marker

    # Onsets are bounded: past ONSET_WINDOWS later onset steps, the marker counts the rest.
    many = [{"layer": 1, "expert": 1, "step": 40, "window_share": 0.0}] + [
        {"layer": 1, "expert": 2 + n, "step": 50 + 4 * n, "window_share": 0.0} for n in range(ONSET_WINDOWS + 3)]
    full = moe_evidence(quiet_until=0, incidents=[{"kind": "starved_experts", "step": 40, "experts": many}])
    marker = compact_evidence(full, window=0)["compact"]
    onsets = [a["step"] for a in marker["around"] if a["kind"] == "starved_expert_onset"]
    assert onsets == [50 + 4 * n for n in range(ONSET_WINDOWS)]
    assert marker["starved_onsets_without_window"] == {"steps": 3, "first_step": 50 + 4 * ONSET_WINDOWS}
