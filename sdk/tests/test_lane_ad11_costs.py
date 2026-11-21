"""Lane AD11: three measured costs of the evidence readers, made smaller without changing what any
reader gives (`OPEN_BUGS.md`, sections 8 and 9).

1. `watch` with the Python detectors judged the whole run again at every look. It now keeps the
   costly detectors' state between looks (`_Incremental`) and feeds them only the new records, and
   at every look its evidence is the evidence `build_evidence` gives on the same files with the
   same options: after a step is appended, late rows of an old attempt, a new attempt, a threshold
   that changes, two changes in a row, and a file rewritten or shortened under the watcher.
2. The Rerun export built its sample lookup again for every step (quadratic in the steps). It builds
   it once; what the recording logs is the same.
3. The gradient detector kept every total and every parameter's norms where its rule reads a
   trailing window; the activation detector and `update_trends` kept each module's or parameter's
   whole history the same way. Each keeps what its rule reads, and the score beside each ratio
   still reads every value of its window.

Tests marked "same answer" pass on the base too (`586181f`); the others fail there.
"""

from __future__ import annotations

import json
import re
import math
import random

import pytest

from tensorquill import evidence as ev

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "rules" / "cases.json").read_text())["cases"]


# --- 2. The Rerun export ---------------------------------------------------------------------


class _CountingList(list):
    """A list that counts how often it is walked."""

    walks = 0

    def __iter__(self):
        type(self).walks += 1
        return super().__iter__()


def _moe_records(steps: int = 200, collapsed: str = "r1") -> list[dict]:
    """Two routers of eight experts: router `r2` missing on some steps and routing nothing on others,
    so the step its table compares with is not always the step before; `collapsed` collapses at step 120."""
    records = []
    for step in range(steps):
        even = [100 + (step * 7 + e * 3) % 11 for e in range(8)]
        late = [700] + [5] * 7
        moe = {"r1": late if collapsed == "r1" and step >= 120 else even}
        if step % 7 == 3:
            moe["r2"] = [0] * 8  # routed nothing
        elif step % 11 != 5:  # left out on these steps: a missing sample
            moe["r2"] = late if collapsed == "r2" and step >= 120 else list(reversed(even))
        records.append({"run_id": "rr", "step": step, "moe": moe, "moe_top_k": {"r1": 2, "r2": 2}})
    return records


def test_the_export_walks_the_samples_a_fixed_number_of_times_not_once_a_step(tmp_path, monkeypatch):
    pytest.importorskip("rerun")
    evidence = ev.build_evidence(_moe_records(300))
    _CountingList.walks = 0
    evidence["samples"] = _CountingList(evidence["samples"])
    ev.export_rerun(evidence, tmp_path / "run.rrd")
    # On the base every step's table walked them again: 300 steps, more than 300 walks.
    assert _CountingList.walks < 20, _CountingList.walks


def test_each_step_logs_the_table_expert_load_table_gives(tmp_path, monkeypatch):
    """Same answer: the text logged at every step is `expert_load_table`'s, compared with the last
    earlier step that routed tokens, through missing and unrouted samples."""
    rr = pytest.importorskip("rerun")
    for layer, collapsed in ((1, "r1"), (2, "r2")):
        evidence = ev.build_evidence(_moe_records(collapsed=collapsed))
        assert ev.first_incident(evidence)["layer"] == layer
        logged: list[tuple[int, str]] = []
        now = {}
        monkeypatch.setattr(rr, "set_time", lambda name, sequence: now.__setitem__("step", sequence))
        monkeypatch.setattr(rr, "TextDocument", lambda text, **kw: ("text", text))

        def log(path, value, **kw):
            if path == "/incident/expert_load":
                logged.append((now["step"], value[1]))

        monkeypatch.setattr(rr, "log", log)
        ev.export_rerun(evidence, tmp_path / f"run{layer}.rrd")
        monkeypatch.undo()
        steps = sorted({s["step"] for s in evidence["samples"]})
        assert [step for step, _ in logged] == steps
        assert [text for _, text in logged] == [ev.expert_load_table(evidence, step, layer) for step in steps]
        if layer == 2:  # a step compared with an earlier one than the step before, past a gap
            assert any(f"step {step} vs. " in text and f"vs. {step - 1}" not in text for step, text in logged[1:])


# --- 3. Histories bounded to what the rules read ---------------------------------------------


def _grad_records(steps: int, spikes=(), params: int = 3) -> list[dict]:
    rng = random.Random(7)
    records = []
    for step in range(steps):
        norms = {f"p{i}": 1.0 + 0.1 * rng.random() for i in range(params)}
        if step in spikes:
            norms["p0"] *= 40
        records.append({"run_id": "g", "step": step, "grad_norm": math.sqrt(sum(v * v for v in norms.values())),
                        "grad_norms": norms})
    return records


def test_the_gradient_detector_keeps_only_its_window():
    state = ev._GradSpikes("grad_norm")
    state.feed(_grad_records(600))
    assert len(state.totals) == ev.SPIKE_WINDOW + 1 and state.total_count == 600
    assert {len(kept) for kept in state.per_param.values()} == {ev.SPIKE_WINDOW}


def test_the_activation_detector_keeps_only_its_windows():
    state = ev._ActivationAnomalies()
    state.feed([{"step": s, "layers": {"m": {"std": 1.0, "abs_max": 5.0, "zero_frac": 0.0}}} for s in range(400)])
    assert len(state.history["m"]) == ev.ACT_WINDOW + ev.ACT_RECENT and state.rows["m"] == 400


def test_more_spikes_than_are_listed_are_still_counted():
    """Same answer: 30 spikes on the total, the first 20 listed, all 30 counted."""
    spikes = list(range(100, 400, 10))
    found = ev.detect_grad_spikes(_grad_records(450, spikes))
    assert found["spike_count"] == 30 and found["spike_steps"] == spikes[:20]


def _finite(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


@pytest.mark.parametrize("name", [n for n, b in CASES.items() if b["part"] == "score_records"])
def test_each_score_over_records_reads_every_value_of_its_window(name):
    """Same answer: on the `score_records` cases, the score beside each ratio is `robust_score` over
    the values its window holds, taken here from the records themselves, as a whole history would
    hold them: the totals before the step (at most SPIKE_WINDOW), or the module's statistic over its
    ACT_WINDOW records before."""
    records = sorted(CASES[name]["records"], key=lambda r: r["step"])
    found = ev.build_evidence(records, detectors="python")
    expected = CASES[name]["expected"]["incident"]
    incident = next(i for i in found["incidents"] if i["kind"] == expected["kind"])
    step = incident["step"]
    before = [r for r in records if r["step"] < step]
    if incident["kind"] == "grad_spike":
        key = "clip_grad_norm" if any("clip_grad_norm" in r for r in records) else "grad_norm"
        totals = _finite(r.get(key) for r in before if not r.get("step_skipped"))[-ev.SPIKE_WINDOW:]
        assert incident["score"] == ev.robust_score(incident["grad_norm"], totals)["score"]
        assert incident["baseline"] == ev._median(totals)
    else:
        module = incident["modules"][0]
        rows = [r["layers"][module["module"]] for r in before if module["module"] in r.get("layers", {})]
        window = [row[module["stat"]] for row in rows][-ev.ACT_WINDOW:]
        assert module["score"] == ev.robust_score(module["value"], window)["score"]
    assert {k: incident[k] for k in expected if k in incident and k != "class"} == {
        k: v for k, v in expected.items() if k in incident}


def test_update_trends_reads_the_same_window_it_did():
    """Same answer: `update_trends` against a whole-history reading of its rule."""
    rng = random.Random(3)
    records = [{"step": s, "update_rms": {f"w{i}": 1e-3 * (1 + rng.random()) * (0.2 if (s, i) == (300, 1) else 1)
                                          for i in range(4)}} for s in range(400)]
    lowest = {}
    past = {}
    for record in records:
        for name, value in record["update_rms"].items():
            earlier = past.setdefault(name, [])
            if len(earlier) >= ev.SCORE_MIN_HISTORY:
                base = ev._median(earlier[-ev.ACT_WINDOW:])
                if name not in lowest or value / base < lowest[name]["ratio"]:
                    lowest[name] = {"name": name, "ratio": value / base, "step": record["step"],
                                    "baseline": base, "update_rms": value}
            earlier.append(value)
    found = ev.update_trends(records)
    assert found["lowest"] == sorted(lowest.values(), key=lambda p: (p["ratio"], p["name"]))[:ev.MEASUREMENTS_LISTED]
    assert found["lowest"][0]["name"] == "w1" and found["lowest"][0]["step"] == 300


# --- 1. The watcher judges only what is new, and gives what a full judgment gives -------------


def _record(step: int, attempt: int = 1, *, seed: int = 0, nan: bool = False) -> dict:
    """One step of a small MoE run that exercises every state the watcher keeps: two routers of eight
    experts (`r2` collapses from step 130, `r1`'s last expert is starved), per-parameter gradient
    norms with a spike at step 120 and the run clipping, two modules' activations with a spike at
    step 125, and a logged loss. `nan` makes a module's `std` NaN."""
    rng = random.Random(step * 1_000 + attempt * 10 + seed)
    w = 1.0 + 0.05 * rng.random()
    grads = {"w1": w * (30.0 if step == 120 else 1.0), "w2": 0.5 * w * (1 + 0.1 * rng.random())}
    total = math.sqrt(sum(v * v for v in grads.values()))
    layers = {
        "blk.0": {"mean": 0.0, "std": w, "abs_max": 5 * w * (12.0 if step == 125 else 1.0), "zero_frac": 0.0},
        "blk.1": {"mean": 0.0, "std": float("nan") if nan else 2 * w, "abs_max": 9 * w, "zero_frac": 0.1},
    }
    even = [100 + rng.randrange(10) for _ in range(8)]
    r1 = even[:7] + [2]
    r2 = [600] + [10] * 7 if step >= 130 else list(reversed(even))
    return {"run_id": "inc", "rank": 0, "attempt": attempt, "step": step, "ts": 1_000.0 + step,
            "grad_norm": total, "clip_grad_norm": total, "grad_norms": grads, "layers": layers,
            "moe": {"r1": r1, "r2": r2}, "moe_top_k": {"r1": 2, "r2": 2}, "scalars": {"loss": 2.0 - step / 1_000}}


def _text(evidence) -> str:
    """An evidence file's text, as `write_evidence` writes it (and as the base did)."""
    return json.dumps(ev.json_safe(evidence), separators=(",", ":"), allow_nan=False) + "\n"


def _lines(records) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def _replace(path, text: str) -> None:
    """Another file put at `path` by a rename."""
    other = path.with_name("other.jsonl")
    other.write_text(text)
    other.replace(path)


def _append(path, records) -> None:
    with path.open("a") as file:
        file.write(_lines(records))


class _Script:
    """A `trigger` for `watch` that makes one change to the file before each look and, after each
    write, checks the evidence against `build_evidence` on the file as it is then."""

    def __init__(self, path, out, changes, made=None, **options) -> None:
        self.path, self.out, self.changes, self.made, self.options = path, out, list(changes), made, options
        self.looks, self.paths, self.checked = 0, [], 0

    def tick(self) -> None:
        if self.looks < len(self.changes):
            self.changes[self.looks]()
        self.looks += 1

    def look(self, evidence, written, rank=None) -> None:
        full = ev.build_evidence(self.path, **self.options)
        text = _text(full)
        assert _text(evidence) == text, f"look {self.looks}: not what a full judgment gives"
        assert written.read_text() == text, f"look {self.looks}: the file written differs"
        if self.made is not None:
            self.paths.append(self.made[0].path)
        self.checked += 1


@pytest.fixture
def watched(monkeypatch):
    """Every `_Incremental` made while the test runs, in order: the watcher makes its own before its
    first look, so it is the first; the full judgments the tests compare with make the others."""
    made: list = []

    class Kept(ev._Incremental):
        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    monkeypatch.setattr(ev, "_Incremental", Kept)
    return made


def _watch(tmp_path, watched, changes, **options):
    path, out = tmp_path / "run.jsonl", tmp_path / "out.json"
    script = _Script(path, out, changes, watched, **options)
    ev.watch(path, out, interval=0.0, polls=len(changes), trigger=script, **options)
    assert script.checked == len(changes), "a look wrote nothing"
    return script.paths


def test_a_watched_run_through_every_change_gives_what_a_full_judgment_gives(tmp_path, watched):
    path = tmp_path / "run.jsonl"
    changes = [
        lambda: path.write_text(_lines(_record(s) for s in range(140))),
        # (a) steps appended
        lambda: _append(path, [_record(s) for s in range(140, 150)]),
        # (c) a new attempt that begins after the newest step
        lambda: _append(path, [_record(s, 2) for s in range(150, 160)]),
        # (b) late rows of the old attempt, at steps the new one replaced
        lambda: _append(path, [_record(s, 1) for s in range(150, 155)]),
        # (c) a new attempt that replaces steps already judged
        lambda: _append(path, [_record(s, 3) for s in range(145, 150)]),
        # (a) and two changes at once: steps, and late rows of attempt 2
        lambda: _append(path, [_record(s, 3) for s in range(150, 171)] + [_record(160, 2)]),
        # the file cut short under the watcher
        lambda: path.write_text(_lines(_record(s) for s in range(120))),
        lambda: _append(path, [_record(s) for s in range(120, 135)]),
        # the file replaced by another, longer one
        lambda: _replace(path, _lines(_record(s, seed=1) for s in range(160))),
        # the file rewritten in place, longer, with other first bytes
        lambda: path.write_text(_lines(_record(s, seed=2) for s in range(170))),
        # a NaN: `non_finite`, and the samples' text through `json_safe` around it
        lambda: _append(path, [_record(170, nan=True)] + [_record(s) for s in range(171, 180)]),
        lambda: _append(path, [_record(s) for s in range(180, 190)]),
    ]
    paths = _watch(tmp_path, watched, changes)
    assert paths == [
        ("full", "first look"),
        ("incremental", 10),
        ("incremental", 10),
        ("incremental", 0),
        ("full", "records judged before changed"),
        ("incremental", 21),
        ("full", "records judged before changed"),
        ("incremental", 15),
        ("full", "records judged before changed"),
        ("full", "records judged before changed"),
        ("incremental", 10),
        ("incremental", 10),
    ]


def test_late_rows_that_sort_before_the_newest_step_are_judged_again_in_full(tmp_path, watched):
    path = tmp_path / "run.jsonl"
    changes = [
        lambda: path.write_text(_lines([_record(s) for s in range(90)] + [_record(s, 2) for s in range(100, 140)])),
        # rows of attempt 1 for 90..99, which attempt 2 (from 100) did not replace
        lambda: _append(path, [_record(s, 1) for s in range(90, 100)]),
        lambda: _append(path, [_record(s, 2) for s in range(140, 150)]),
    ]
    assert _watch(tmp_path, watched, changes) == [
        ("full", "first look"), ("full", "records judged before changed"), ("incremental", 10)]


def test_a_file_cut_short_is_read_again(tmp_path):
    """On the base the watcher read on from where it was: the evidence kept steps the file no longer had."""
    path, out = tmp_path / "run.jsonl", tmp_path / "out.json"
    seen = []
    changes = [lambda: path.write_text(_lines(_record(s) for s in range(150))),
               lambda: path.write_text(_lines(_record(s) for s in range(40)))]
    script = _Script(path, out, changes)
    script.look = lambda evidence, written, rank=None: seen.append(
        (_text(evidence), _text(ev.build_evidence(path))))
    ev.watch(path, out, interval=0.0, polls=2, trigger=script)
    assert len(seen) == 2 and all(watched == full for watched, full in seen)
    assert json.loads(seen[-1][0])["samples"][-1]["step"] == 39


def test_a_threshold_that_changes_judges_every_record_again():
    records = [_record(s) for s in range(160)]
    memo = ev._Incremental()
    looks = [(100, {}), (120, {"overload_threshold": 0.4}), (130, {"overload_threshold": 0.4}),
             (140, {"overload_threshold": 0.4, "starve_window": 20}), (150, {"overload_threshold": 0.4, "starve_window": 20,
                                                                            "top_k": 1}),
             (160, {"overload_threshold": 0.4, "starve_window": 20, "top_k": 1})]
    paths = []
    for count, options in looks:
        given = ev._build_evidence(records[:count], memo, **options)
        assert _text(given) == _text(ev.build_evidence(records[:count], **options))
        paths.append(memo.path)
    assert paths == [("full", "first look"), ("full", "thresholds or a whole-run fact changed"), ("incremental", 10),
                     ("full", "thresholds or a whole-run fact changed"),
                     ("full", "thresholds or a whole-run fact changed"), ("incremental", 10)]


def test_the_server_s_thresholds_moving_under_the_watcher_judges_every_record_again(tmp_path, monkeypatch, watched):
    """A store whose report is behind, so the detectors run here with the thresholds the server
    applied (`adopted`); the server's thresholds change between two looks. Each look is compared
    with the same looks judged in full (`_INCREMENTAL_WATCH` off), and with `build_evidence` on the
    rank's folder with the thresholds in force, outside the keys the watcher writes itself."""
    from test_lane_ad1_readers import LISTED, _report, _run, _write_report
    from test_watchdog import _write_rank_part

    other = {**LISTED, "overload_threshold": 0.7, "starve_window": 10}

    def play(root, incremental):
        monkeypatch.setattr(ev, "_INCREMENTAL_WATCH", incremental)
        run = _run(root)
        folder = run / "rank=0"
        seen = []

        def at(steps, last, thresholds):
            def change():
                _write_rank_part(run, 0, steps, extra={"grad_norms": {"a": 1.0, "b": 0.5}})
                _write_report(run, _report(0, 1, last) | {"thresholds": {**thresholds, "spike_sustain": ev.SPIKE_SUSTAIN}})
            return change

        class Look:
            changes = [at(range(0, 20), 9, LISTED), at(range(20, 40), 19, other), at(range(40, 60), 29, other)]
            looks = 0

            def tick(self):
                self.changes[self.looks]()
                self.looks += 1

            def look(self, evidence, written, rank=None):
                thresholds = {k: v for k, v in (LISTED if self.looks == 1 else other).items() if v is not None}
                # What the watcher hands `build_evidence`: the rank's records as its folder holds them.
                full = ev.build_evidence(ev.load_records(folder, all_attempts=True), run_manifest=folder,
                                         records_file=folder, **thresholds)
                assert evidence["detection"]["source"] == "python"
                assert evidence["detection"]["thresholds"] == {**ev.THRESHOLD_DEFAULTS, **thresholds}
                outside = {k: v for k, v in evidence.items() if k != "detection"}
                assert _text(outside) == _text(full)
                seen.append((_text(evidence), watched[0].path if incremental else None))

        ev.watch(folder, root / "out.json", interval=0.0, polls=3, trigger=Look(), report_grace=0.0)
        return seen

    incremental = play(tmp_path / "a", True)
    full = play(tmp_path / "b", False)
    assert [text for text, _ in full] == [text for text, _ in incremental] and len(incremental) == 3
    assert [path for _, path in incremental] == [
        ("full", "first look"), ("full", "thresholds or a whole-run fact changed"), ("incremental", 20)]


def test_every_golden_record_set_fed_look_by_look_gives_what_a_full_judgment_gives():
    """Same answer: each record set of `test_golden.py` (the server-detect cases and the Gate 1
    fixtures), fed to one `_Incremental` a prefix at a time, one record at a time and then in
    chunks of random size, gives at every look the text `build_evidence` gives on that prefix."""
    golden = pytest.importorskip("test_golden")
    rng = random.Random(11)
    looks = incremental = 0
    for name in golden.NAMES:
        records = golden.records(name)
        memo = ev._Incremental()
        ends = list(range(1, min(len(records), 6) + 1))
        while ends[-1] < len(records):
            ends.append(min(len(records), ends[-1] + rng.randint(1, max(2, len(records) // 6))))
        for end in ends:
            prefix = records[:end]
            try:
                full = _text(ev.build_evidence(prefix))
            except ValueError as error:  # records of several ranks: refused the same way
                with pytest.raises(ValueError, match=re.escape(str(error))):
                    ev._build_evidence(prefix, memo)
                continue
            given = ev._build_evidence(prefix, memo)
            assert _text(given) == full, (name, end, memo.path)
            assert ev._evidence_text(given, memo.samples_text(given)) == full, (name, end)
            looks += 1
            incremental += memo.path[0] == "incremental"
    assert looks > 300 and incremental > looks // 2, (looks, incremental)


def test_a_look_after_steps_are_appended_feeds_the_states_only_the_new_records(tmp_path, monkeypatch):
    """The cost, counted: on the second look the states are fed the five new records, and the JSON
    text is made for their ten samples and the rest of the file, not for the run's 410 samples."""
    path, out = tmp_path / "run.jsonl", tmp_path / "out.json"
    fed: list[tuple[str, int]] = []
    for state in (ev._MoeSamples, ev._GradSpikes, ev._ActivationAnomalies, ev._NonFinite):
        def counting(self, records, _feed=state.feed, _name=state.__name__):
            records = list(records)
            fed.append((_name, len(records)))
            return _feed(self, records)
        monkeypatch.setattr(state, "feed", counting)
    texts = []
    real_text = ev._json_text
    monkeypatch.setattr(ev, "_json_text", lambda value: texts.append(1) or real_text(value))
    looks = []
    changes = [lambda: path.write_text(_lines(_record(s) for s in range(200))),
               lambda: (fed.clear(), texts.clear(), _append(path, [_record(s) for s in range(200, 205)]))]
    script = _Script(path, out, changes)
    script.look = lambda evidence, written, rank=None: looks.append((list(fed), len(texts)))
    ev.watch(path, out, interval=0.0, polls=2, trigger=script)
    assert looks[1] == ([("_MoeSamples", 5), ("_GradSpikes", 5), ("_ActivationAnomalies", 5), ("_NonFinite", 5)], 11)
    assert looks[0][1] == 401  # 400 samples and the rest, on the first look


def test_a_stored_part_that_changes_or_goes_is_read_again(tmp_path, watched):
    """A server's store under the watcher (no report, so the detectors run here): a part read is
    written again with other rows, then one is removed. Each look is what `build_evidence` gives on
    the records the folder then holds; on the base the part's old rows stayed."""
    from test_watchdog import _write_rank_part

    run = tmp_path / "store" / "runs" / "parts"
    folder = run / "rank=0"
    rows = lambda scale: {"grad_norms": {"a": scale, "b": 0.5 * scale}}

    class Look:
        changes = [
            lambda: [_write_rank_part(run, 0, range(low, low + 20), extra=rows(1.0)) for low in (0, 20, 40)],
            lambda: _write_rank_part(run, 0, range(20, 40), extra=rows(9.0)),  # the part read, written again
            lambda: (folder / "attempt=1" / "part-00000040-00000059.parquet").unlink(),  # a part gone
            lambda: _write_rank_part(run, 0, range(40, 60), extra=rows(1.0)),
        ]
        looks, paths, steps = 0, [], []

        def tick(self):
            self.changes[self.looks]()
            self.looks += 1

        def look(self, evidence, written, rank=None):
            full = ev.build_evidence(ev.load_records(folder, all_attempts=True), run_manifest=folder, records_file=folder)
            assert _text({k: v for k, v in evidence.items() if k != "detection"}) == _text(full)
            self.paths.append(watched[0].path)
            self.steps.append(len(evidence["samples"]))

    look = Look()
    ev.watch(folder, tmp_path / "out.json", interval=0.0, polls=4, trigger=look, server_reports=False)
    assert look.steps == [60, 60, 40, 60]
    assert look.paths == [("full", "first look"), ("full", "records judged before changed"),
                          ("full", "records judged before changed"), ("incremental", 20)]
