"""Compact evidence (`tensorquill.compact`): the incidents and a window of per-step samples
around each MoE incident, in a file small enough for an agent to read.

The plan's done test is the first one here: a 10,000-step MoE run gives a file under 1 MB.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

import tensorquill
from tensorquill import compact
from tensorquill.compact import SamplesOmitted, compact_evidence
from tensorquill.evidence import incident_summary, main, run_state_line, watch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


def _size(evidence) -> int:
    """Bytes of the file `write_evidence` writes."""
    return len(json.dumps(evidence, separators=(",", ":"))) + 1


# --- The plan's line, on the detector benchmark's Granite-shaped run -------------------------


@pytest.fixture(scope="module")
def granite_run():
    """10,000 steps of the detector benchmark's synthetic run in Granite's shape (24 routers,
    32 experts, top-8), with two MoE faults planted: from step 3,000 one expert of router 12
    gets a single assignment a step, and from step 6,000 router 5 sends everything to three
    experts. About 20 s: the capture that gives the record shapes, and the detectors over the run."""
    import benchmark_detectors as bd
    from tensorquill.capture import _router_stats

    template = bd.capture_template(blocks=24, experts=32, top_k=8)
    records = [bd.synth_record(template, 1, 0, step) for step in range(10_000)]
    names = list(records[0]["moe"])
    for record in records[3000:]:
        counts = record["moe"][names[12]]
        moved, counts[7] = counts[7] - 1, 1
        counts[counts.index(max(counts))] += moved
        record["moe_stats"][names[12]] = _router_stats(counts)
    for record in records[6000:]:
        total = sum(record["moe"][names[5]])
        collapsed = [total // 2, total // 4, total - total // 2 - total // 4] + [0] * 29
        record["moe"][names[5]] = collapsed
        record["moe_stats"][names[5]] = _router_stats(collapsed)
    return tensorquill.build_evidence(records)


def test_a_10000_step_moe_run_gives_a_compact_file_under_1_mb(granite_run, tmp_path):
    seeded = granite_run
    assert [i["kind"] for i in seeded["incidents"]] == ["moe_imbalance", "starved_experts"]
    assert len(seeded["samples"]) == 10_000 * 24
    assert _size(seeded) > 80_000_000  # the full file: 86.5 MB when this was written

    # Measured when this was written: 194,708 bytes with the default window of 5 (22 steps of
    # 24 routers kept), 21,679 with a window of 0, and 1 MB is passed at a window of 29.
    path = compact.write_compact(seeded, tmp_path / "run.evidence.json")
    assert path == tmp_path / "run.evidence.compact.json" and not (tmp_path / "run.evidence.json").exists()
    assert path.stat().st_size < 1_000_000
    small = json.loads(path.read_text())
    marker = small["compact"]
    assert marker["around"] == [{"kind": "moe_imbalance", "step": 6000}, {"kind": "starved_experts", "step": 3046}]
    assert marker["kept_steps"] == [[3041, 3051], [5995, 6005]]
    assert marker["steps"] == {"total": 10_000, "kept": 22, "omitted": 9_978}
    assert marker["samples"] == {"total": 240_000, "kept": 528, "omitted": 239_472}
    assert marker["moe_flagged"] == {"samples": 4000, "last_step": 9999}  # the collapse never went away
    assert marker["moe_starved"]["last_step"] == 9999
    assert len(small["samples"]) == 528
    # Everything else is the full evidence, key for key and in its order.
    assert [k for k in small if k != "compact"] == list(seeded)
    assert {k: v for k, v in small.items() if k not in ("compact", "samples")} == json.loads(
        json.dumps({k: v for k, v in seeded.items() if k != "samples"})
    )
    # The same text for both files.
    assert incident_summary(small) == incident_summary(seeded)
    assert "First sampled bad step 6000  ·  Layer 06" in incident_summary(small)
    assert "Overloaded: Expert 1 (50%), Expert 2 (25%), Expert 3 (25%)" in incident_summary(small)
    assert run_state_line(small) == run_state_line(seeded)
    # The table at the incident's step is the full evidence's table.
    assert tensorquill.expert_load_table(small, 6000, 6) == tensorquill.expert_load_table(seeded, 6000, 6)

    # The same run with no incident listed keeps no sample at all: a few kilobytes (3,482 bytes
    # for the healthy run, measured apart).
    nothing = compact_evidence({**seeded, "incidents": [], "samples": [
        {**s, "overloaded": [], "dead": []} for s in seeded["samples"]
    ]})
    assert nothing["samples"] == [] and nothing["compact"]["samples"]["omitted"] == 240_000
    assert _size(nothing) < 12_000  # 10,178 bytes with the two measurement blocks of the rules contract, part 6 (8,221 before)


# --- Small runs: the window's edges, and what each reader of samples says ---------------------


def _moe_records(steps, collapse_from=None, every=1, **more):
    """One 4-expert router, balanced, and from `collapse_from` on all on one expert."""
    return [
        {"run_id": "r", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0,
         "moe": {"router": [400, 0, 0, 0] if collapse_from is not None and s >= collapse_from else [100] * 4}, **more}
        for s in range(0, steps, every)
    ]


@pytest.fixture
def collapsed():
    """Steps 0, 2, ..., 198, with the router collapsed from step 100."""
    evidence = tensorquill.build_evidence(_moe_records(200, collapse_from=100, every=2))
    assert [(i["kind"], i["step"]) for i in evidence["incidents"]] == [("moe_imbalance", 100)]
    return evidence


def test_the_window_counts_recorded_steps_on_each_side_of_the_incident(collapsed):
    before = copy.deepcopy(collapsed)
    small = compact_evidence(collapsed, window=3)
    assert collapsed == before  # the full evidence is not touched
    assert [s["step"] for s in small["samples"]] == [94, 96, 98, 100, 102, 104, 106]  # 3 recorded steps a side
    assert small["samples"] == [s for s in collapsed["samples"] if 94 <= s["step"] <= 106]
    marker = small["compact"]
    assert (marker["version"], marker["window"]) == (1, 3)
    assert marker["kept_steps"] == [[94, 106]] and marker["sampled_steps"] == [0, 198]
    assert marker["steps"] == {"total": 100, "kept": 7, "omitted": 93}
    assert marker["samples"] == {"total": 100, "kept": 7, "omitted": 93}
    assert marker["moe_flagged"] == {"samples": 50, "last_step": 198}
    assert list(small).index("compact") == list(small).index("samples") - 1  # read before the samples
    assert incident_summary(small) == incident_summary(collapsed)

    # 0 keeps the incident's own step alone, which is what the summary quotes.
    alone = compact_evidence(collapsed, window=0)
    assert [s["step"] for s in alone["samples"]] == [100] and alone["compact"]["kept_steps"] == [[100, 100]]
    assert incident_summary(alone) == incident_summary(collapsed)
    assert "Overloaded: Expert 1 (100%)" in incident_summary(alone)

    # A window wider than the run leaves nothing out, and says so.
    whole = compact_evidence(collapsed, window=500)
    assert whole["samples"] == collapsed["samples"] and whole["compact"]["samples"]["omitted"] == 0
    assert whole["compact"]["kept_steps"] == [[0, 198]]
    assert "none left out" in compact.describe(whole)

    # Compact evidence stays as it is: what it left out cannot be put back.
    assert compact_evidence(small, window=50) == small
    for bad in (-1, 2.5, "3", True, None):
        with pytest.raises(ValueError, match="recorded steps"):
            compact_evidence(collapsed, window=bad)


def test_windows_stop_at_the_runs_ends_and_merge_when_they_touch(collapsed):
    early = tensorquill.build_evidence(_moe_records(200, collapse_from=2))
    small = compact_evidence(early, window=5)
    assert small["compact"]["kept_steps"] == [[0, 7]]  # cut at the run's first step
    late = tensorquill.build_evidence(_moe_records(200, collapse_from=198))
    assert compact_evidence(late, window=5)["compact"]["kept_steps"] == [[193, 199]]

    # A second MoE incident four recorded steps later: one range. One far away: two.
    near = {**collapsed, "incidents": [*collapsed["incidents"], {"kind": "starved_experts", "step": 108, "experts": []}]}
    marker = compact_evidence(near, window=3)["compact"]
    assert marker["kept_steps"] == [[94, 114]] and marker["steps"]["kept"] == 11
    assert marker["around"] == [{"kind": "moe_imbalance", "step": 100}, {"kind": "starved_experts", "step": 108}]
    far = {**collapsed, "incidents": [*collapsed["incidents"], {"kind": "starved_experts", "step": 20, "experts": []}]}
    assert compact_evidence(far, window=3)["compact"]["kept_steps"] == [[14, 26], [94, 106]]

    # Only MoE incidents get a window: a gradient spike at another step keeps no samples.
    other = {**collapsed, "incidents": [*collapsed["incidents"], {"kind": "grad_spike", "step": 20}]}
    assert compact_evidence(other, window=3)["compact"]["kept_steps"] == [[94, 106]]

    # Flagged samples with no incident naming them (an incident list from elsewhere): the first
    # flagged sample is kept all the same, because the summary quotes it.
    unnamed = {**collapsed, "incidents": []}
    small = compact_evidence(unnamed, window=1)
    assert small["compact"]["around"] == [{"kind": "first_flagged_sample", "step": 100}]
    assert small["compact"]["kept_steps"] == [[98, 102]]
    assert incident_summary(small) == incident_summary(unnamed)


def test_the_readers_of_samples_say_what_a_compact_file_left_out(collapsed, tmp_path):
    small = json.loads(json.dumps(compact_evidence(collapsed, window=3)))  # as read back from the file
    # Inside the window the table is the full evidence's table, also between two recorded steps.
    for step in (96, 100, 101, 106):
        assert tensorquill.expert_load_table(small, step, 1) == tensorquill.expert_load_table(collapsed, step, 1)
    assert tensorquill.expert_load_table(small, 100, 1).splitlines()[0] == "**Layer 01 expert load** · step 100 vs. 98"

    # A step that was left out: not a table of "Missing" rows, and not a KeyError.
    with pytest.raises(SamplesOmitted, match=r"step 50 is not here.*steps 94-106 \(3 recorded steps each side") as error:
        tensorquill.expert_load_table(small, 50, 1)
    assert "leaves out 93 of the run's 100 recorded steps" in str(error.value)
    assert "without --compact" in str(error.value) and not isinstance(error.value, KeyError)
    # The window's first step: the step it is compared with was left out, so its loads are shown
    # with every delta n/a, never +0.00 (DET-13; until Nov 20, 2026 the table was refused).
    first = tensorquill.expert_load_table(small, 94, 1)
    assert first.splitlines()[0] == (
        "**Layer 01 expert load** · step 94 (the step it is compared with was left out of this compact file)"
    )
    assert [line.split("|")[3].strip() for line in first.splitlines()[4:]] == ["n/a"] * 4
    # With a window of 0 that holds for the incident's own step.
    alone = tensorquill.expert_load_table(compact_evidence(collapsed, window=0), 100, 1)
    assert "left out of this compact file" in alone.splitlines()[0] and "+" not in alone
    # A window that starts at the run's first step has nothing left out before it.
    early = tensorquill.build_evidence(_moe_records(200, collapse_from=2))
    assert tensorquill.expert_load_table(compact_evidence(early), 0, 1) == tensorquill.expert_load_table(early, 0, 1)

    # A Rerun recording of the kept windows would pass for the run: refused before anything is written.
    with pytest.raises(SamplesOmitted, match="a Rerun recording needs every step's samples"):
        tensorquill.export_rerun(small, tmp_path / "small.rrd")
    assert not (tmp_path / "small.rrd").exists()
    # Compact evidence that left nothing out is as good as the full one.
    compact.require_all_samples(compact_evidence(collapsed, window=500), "a Rerun recording")
    compact.require_all_samples(collapsed, "a Rerun recording")
    assert compact.describe(collapsed) is None


def test_a_run_with_no_moe_data_comes_back_whole_with_the_marker():
    records = [{"run_id": "r", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0, "scalars": {"loss": 2.0}}
               for s in range(40)]
    records[30]["grad_norm"] = "nan"
    full = tensorquill.build_evidence(records)
    assert [i["kind"] for i in full["incidents"]] == ["non_finite"] and full["samples"] == []
    small = compact_evidence(full)
    assert {k: v for k, v in small.items() if k != "compact"} == full
    marker = small["compact"]
    assert marker["samples"] == {"total": 0, "kept": 0, "omitted": 0} and marker["kept_steps"] == []
    assert marker["sampled_steps"] is None and marker["moe_flagged"] == {"samples": 0, "last_step": None}
    assert incident_summary(small) == incident_summary(full) and run_state_line(small) == run_state_line(full)
    assert "no per-step samples to leave out" in compact.describe(small)
    compact.require_all_samples(small, "a Rerun recording")  # nothing is missing

    # The all-ranks view and a server's report hold no samples either: marked, otherwise as they were.
    view = {"view": "all_ranks", "view_version": 1, "run_id": "r", "incident_groups": [], "run_incidents": [], "ranks": []}
    marked = compact_evidence(view)
    assert list(marked) == [*view, "compact"] and {k: marked[k] for k in view} == view
    assert marked["compact"]["samples"]["total"] == 0


def test_evidence_with_no_incidents_keeps_no_samples_and_reads_the_same():
    full = tensorquill.build_evidence(_moe_records(120))
    assert full["incidents"] == [] and len(full["samples"]) == 120
    small = compact_evidence(full)
    assert small["samples"] == []
    marker = small["compact"]
    assert marker["around"] == [] and marker["kept_steps"] == []
    assert marker["samples"] == {"total": 120, "kept": 0, "omitted": 120}
    assert marker["moe_flagged"] == {"samples": 0, "last_step": None}
    assert incident_summary(small) == incident_summary(full) == "No incidents detected.\n5 of 33 kinds checked, 28 not configured"
    assert run_state_line(small) == run_state_line(full)
    with pytest.raises(SamplesOmitted, match="no step, since no MoE incident is listed"):
        tensorquill.expert_load_table(small, 10, 1)


# --- The command line and the watcher ---------------------------------------------------------


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_the_cli_writes_the_compact_file_instead_of_the_full_one(tmp_path, monkeypatch, capsys):
    source = tmp_path / "run.jsonl"
    _write_jsonl(source, _moe_records(200, collapse_from=100))
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(source)])
    main()
    plain = capsys.readouterr().out
    full_path = tmp_path / "run.evidence.json"
    full_bytes = full_path.read_bytes()
    assert f"wrote evidence -> {full_path}" in plain

    monkeypatch.setattr(sys, "argv", ["tensorquill", str(source), "--compact"])
    main()
    printed = capsys.readouterr().out
    small_path = tmp_path / "run.evidence.compact.json"
    assert f"wrote compact evidence -> {small_path}" in printed
    # The same summary and state line on the terminal, and the full file already there is left alone.
    assert printed.replace("compact evidence", "evidence").replace(str(small_path), str(full_path)) == plain
    assert full_path.read_bytes() == full_bytes
    small = json.loads(small_path.read_text())
    assert small["compact"]["window"] == 5 and small["compact"]["kept_steps"] == [[95, 105]]
    # (What was checked, `kind_checks`, is about 3 KB of either file: a tenth was the bound before it.)
    assert len(small["samples"]) == 11 and small_path.stat().st_size < len(full_bytes) / 5

    # --compact-window implies --compact; -o names the file.
    full_path.unlink()
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(source), "--compact-window", "0", "-o", str(tmp_path / "a.json")])
    main()
    assert json.loads((tmp_path / "a.json").read_text())["compact"]["kept_steps"] == [[100, 100]]
    assert not full_path.exists() and "wrote compact evidence" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(source), "--compact-window", "-1"])
    with pytest.raises(SystemExit):
        main()
    assert "not a number of recorded steps" in capsys.readouterr().err


def test_watch_keeps_the_compact_file_current_and_hands_on_the_full_evidence(tmp_path):
    records = _moe_records(200, collapse_from=100)
    source = tmp_path / "w.jsonl"
    _write_jsonl(source, records[:90])
    seen = []

    def grow(evidence, out):
        seen.append((len(evidence["samples"]), "compact" in evidence, json.loads(out.read_text())["compact"]["kept_steps"]))
        if len(seen) == 1:
            with source.open("a") as file:
                file.writelines(json.dumps(r) + "\n" for r in records[90:])

    out = watch(source, interval=0.0, polls=4, on_update=grow, compact_window=2)
    assert out == tmp_path / "w.evidence.compact.json" and not (tmp_path / "w.evidence.json").exists()
    # No incident yet, so no sample; then the collapse and its window. `on_update` gets everything.
    assert seen == [(90, False, []), (200, False, [[98, 102]])]
    small = json.loads(out.read_text())
    assert small == json.loads(json.dumps(compact_evidence(tensorquill.build_evidence(records, records_file=source),
                                                           window=2)))
    # A named output is used as it is.
    named = watch(source, tmp_path / "named.json", interval=0.0, polls=1, compact_window=2)
    assert named == tmp_path / "named.json" and json.loads(named.read_text()) == small


def test_compact_with_rerun_records_from_the_full_evidence(tmp_path, monkeypatch, capsys):
    """`--compact --rerun`: the file is compact, the recording is of every step."""
    pytest.importorskip("rerun")
    source = tmp_path / "run.jsonl"
    _write_jsonl(source, _moe_records(200, collapse_from=100))
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(source), "--compact", "--rerun", str(tmp_path / "run.rrd")])
    main()
    printed = capsys.readouterr().out
    assert "wrote compact evidence" in printed and "wrote Rerun recording" in printed
    assert (tmp_path / "run.rrd").stat().st_size > 0
    small = json.loads((tmp_path / "run.evidence.compact.json").read_text())
    assert small["compact"]["samples"] == {"total": 200, "kept": 11, "omitted": 189}
    with pytest.raises(SamplesOmitted):  # from the file alone it cannot be made
        tensorquill.export_rerun(small, tmp_path / "small.rrd")
