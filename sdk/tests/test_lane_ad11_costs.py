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
