"""The attention rules over the probe's records: `attention_collapse`, `attention_uniform` and
`attention_logit_growth`, in Python (`evidence.py`, the reference) and in the server's Rust
detectors, which must give the same incidents after every stretch of a run.

The server half needs `pyarrow` and the server binary, like `test_server_detectors.py`.
"""

from __future__ import annotations

import json
import math

import pytest

import tensorquill
from tensorquill import evidence as ev

HEADS = 4


def heads(value: float, spread: float = 0.02) -> list[float]:
    """Four heads whose mean is `value`."""
    return [value - 1.5 * spread, value - 0.5 * spread, value + 0.5 * spread, value + 1.5 * spread]


def attention(share: float, logit: float, **extra) -> dict:
    return {
        "max_logit": heads(logit, 0.4), "entropy": heads(2.0), "entropy_share": heads(share),
        "masked_rows": 0, "invalid_rows": 0, "map": [[0.1] * 4 for _ in range(4)], **extra,
    }


def record(step: int, run: str = "attn", attempt: int = 1, **values) -> dict:
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step, "ts": float(step), **values}


def probe_run(modules: dict, probes: int, every: int = 10, run: str = "attn", scalars: bool = True,
              extra=None) -> list[dict]:
    """A run with a probe every `every` steps. `modules[name](i)` gives (share, logit) for probe i,
    or None for a probe without that module. `extra(i, probe)` may change a probe in place."""
    out = []
    for step in range(probes * every):
        values: dict = {"grad_norm": 1.0}
        if scalars:
            values["scalars"] = {"train/loss": 2.0 + (step % 7) / 10}
        if (step + 1) % every == 0:
            i = (step + 1) // every - 1
            att = {}
            for name, give in modules.items():
                got = give(i)
                if got is not None:
                    att[name] = attention(*got)
            probe = {"histograms": {name: {"counts": [1, 2], "lo": 0.0, "hi": 1.0} for name in att}, "attention": att}
            if extra is not None:
                extra(i, probe)
            values["probe"] = probe
        out.append(record(step, run=run, **values))
    return out


def steady(share=0.4, logit=10.0):
    return lambda i: (share, logit)


def kinds(records) -> list[tuple[str, int]]:
    return [(i["kind"], i["step"]) for i in ev.detect_attention_anomalies(records)]


# --- each rule, positive and negative ---------------------------------------------------------


def test_steady_attention_is_no_incident():
    assert kinds(probe_run({"a": steady(), "b": steady(0.95, 3.0)}, 12)) == []


def test_collapse_held_on_two_probes_is_dated_at_the_first():
    run = probe_run({"a": lambda i: (0.15 if i >= 5 else 0.4, 10.0), "b": steady()}, 8)
    found = ev.detect_attention_anomalies(run)
    assert [(i["kind"], i["step"], i["confirmed_step"]) for i in found] == [("attention_collapse", 59, 69)]
    hit = found[0]["modules"][0]
    assert hit["module"] == "a" and math.isclose(hit["value"], 0.15) and math.isclose(hit["baseline"], 0.4)
    assert math.isclose(hit["ratio"], 0.375) and (hit["baseline_probes"], hit["heads"]) == (5, HEADS)
    assert (found[0]["affected"], found[0]["affected_probes"], found[0]["probes"]) == (1, 3, 8)


def test_collapse_needs_the_fall_held_and_deep_enough():
    # One probe at 0.15, then back: not held.
    assert kinds(probe_run({"a": lambda i: (0.15 if i == 4 else 0.4, 10.0)}, 10)) == []
    # Held, but only to 0.21 of a 0.4 baseline: 0.525 times, above ATTN_COLLAPSE_RATIO.
    assert kinds(probe_run({"a": lambda i: (0.21 if i >= 4 else 0.4, 10.0)}, 10)) == []
    # Exactly half is enough.
    assert kinds(probe_run({"a": lambda i: (0.2 if i >= 4 else 0.4, 10.0)}, 10)) == [("attention_collapse", 49)]


def test_uniform_attention_where_there_was_structure():
    run = probe_run({"a": lambda i: (0.995 if i >= 3 else 0.35, 10.0 if i < 3 else 1.0), "b": steady()}, 6)
    found = ev.detect_attention_anomalies(run)
    assert [(i["kind"], i["step"]) for i in found] == [("attention_uniform", 39)]
    hit = found[0]["modules"][0]
    assert math.isclose(hit["rise"], 0.645) and "ratio" not in hit and hit["baseline_probes"] == 3


def test_a_layer_near_uniform_from_the_start_is_never_uniform_attention():
    # From 0.95 to 0.995 is a rise of 0.045: below ATTN_UNIFORM_RISE.
    assert kinds(probe_run({"a": lambda i: (0.995 if i >= 3 else 0.95, 5.0)}, 8)) == []
    # A rise of 0.3 that stops short of ATTN_UNIFORM is not uniform either.
    assert kinds(probe_run({"a": lambda i: (0.97 if i >= 3 else 0.6, 5.0)}, 8)) == []
    # A first probe at uniform (a model at initialisation) has no baseline to rise from.
    assert kinds(probe_run({"a": lambda i: (1.0 if i < 2 else 0.6, 5.0)}, 8)) == []


def test_logit_growth():
    run = probe_run({"a": lambda i: (0.4, 25.0 if i >= 6 else 10.0)}, 9)
    found = ev.detect_attention_anomalies(run)
    assert [(i["kind"], i["step"]) for i in found] == [("attention_logit_growth", 69)]
    # The largest head's logit is what is judged: heads(10, 0.4) peaks at 10.6.
    assert math.isclose(found[0]["modules"][0]["baseline"], 10.6)
    assert kinds(probe_run({"a": lambda i: (0.4, 19.0 if i >= 6 else 10.0)}, 9)) == []  # 1.8 times
    assert kinds(probe_run({"a": lambda i: (0.4, 25.0 if i == 6 else 10.0)}, 9)) == []  # one probe


def test_a_lasting_change_stays_out_of_the_baseline_and_keeps_counting():
    found = ev.detect_attention_anomalies(probe_run({"a": lambda i: (0.15 if i >= 3 else 0.4, 10.0)}, 12))
    assert found[0]["affected_probes"] == 9 and found[0]["modules"][0]["baseline_probes"] == 3


def test_too_few_probes_are_listed_as_not_checked():
    not_checked: list = []
    assert ev.detect_attention_anomalies(probe_run({"a": steady()}, 3), not_checked=not_checked) == []
    assert not_checked == [{"check": "attention", "kinds": list(ev.ATTN_KINDS), "reason": not_checked[0]["reason"]}]
    assert "3 probes recorded attention" in not_checked[0]["reason"] and "4 are needed" in not_checked[0]["reason"]
    enough: list = []
    ev.detect_attention_anomalies(probe_run({"a": steady()}, 4), not_checked=enough)
    assert enough == []
    nothing: list = []
    ev.detect_attention_anomalies(probe_run({}, 0) + [record(1, grad_norm=1.0)], not_checked=nothing)
    assert nothing == []  # no probe at all (the probe is opt-in): nothing to say


def test_records_without_probes_change_nothing_in_existing_evidence():
    from test_server_detectors import everything

    records = everything(120)
    evidence = tensorquill.build_evidence(records)
    assert not any(i["kind"] in ev.ATTN_KINDS for i in evidence["incidents"])
    assert all(n["check"] != "attention" for n in evidence.get("not_checked", ()))


def test_odd_probes_are_read_as_python_reads_them():
    def extra(i, probe):
        a = probe["attention"]["a"]
        if i == 1:
            a["entropy_share"] = ["nan", a["entropy_share"][1], a["entropy_share"][2], "nan"]  # masked heads left out
            a["max_logit"][0] = "-inf"
        if i == 2:
            probe["error"] = "RuntimeError: out of memory"  # a failed probe is not a probe, whatever else it holds
        if i == 3:
            probe["attention"] = {}  # no attention seen: not a probe either
        if i == 5:
            a["entropy_share"][:2] = ["nan", "nan"]  # the onset probe: its mean is over the two finite heads
    run = probe_run({"a": lambda i: (0.15 if i >= 5 else 0.4, 10.0)}, 8, extra=extra)
    found = ev.detect_attention_anomalies(run)
    assert [(i["kind"], i["step"], i["probes"]) for i in found] == [("attention_collapse", 59, 6)]
    hit = found[0]["modules"][0]
    assert hit["baseline_probes"] == 3  # probes 0, 1 and 4
    assert math.isclose(hit["value"], 0.17) and hit["heads"] == 2  # (0.16 + 0.18) / 2


def test_a_layer_is_judged_from_its_third_probe():
    # Low from its second probe on: with one probe of baseline that would be a collapse held on
    # probes 1 and 2; with two, the baseline already holds the low value and nothing changed.
    assert kinds(probe_run({"a": lambda i: (0.4 if i == 0 else 0.15, 10.0)}, 8)) == []


def test_a_probe_without_the_module_breaks_its_run():
    gone = lambda i: None if i == 6 else (0.15 if i >= 5 else 0.4, 10.0)
    # Probe 5 starts a run, probe 6 lacks the module, probe 7 starts again and 8 confirms it.
    assert kinds(probe_run({"a": gone, "b": steady()}, 8)) == []
    assert kinds(probe_run({"a": gone, "b": steady()}, 9)) == [("attention_collapse", 79)]


def test_modules_are_listed_in_the_probes_order_and_capped():
    names = [f"model.layers.{n}" for n in range(25)]
    run = probe_run({name: (lambda i: (0.999 if i >= 3 else 0.3, 5.0)) for name in names}, 6)
    found = ev.detect_attention_anomalies(run)[0]
    assert [m["module"] for m in found["modules"]] == names[:ev.ATTN_MODULES] and found["affected"] == 25


def test_build_evidence_lists_attention_after_activations_with_what_was_logged():
    run = probe_run({"a": lambda i: (0.15 if i >= 5 else 0.4, 25.0 if i >= 5 else 10.0)}, 8)
    evidence = tensorquill.build_evidence(run)
    assert [i["kind"] for i in evidence["incidents"]] == ["attention_collapse", "attention_logit_growth"]
    assert evidence["incidents"][0]["scalars"]["train/loss"]["value"] == run[59]["scalars"]["train/loss"]
    summary = tensorquill.incident_summary(evidence)
    assert "Attention entropy collapse at step 59 in a" in summary
    assert "Attention logit growth at step 59 in a" in summary


# --- the same in the server --------------------------------------------------------------------


def _server_tools():
    pytest.importorskip("pyarrow.flight")
    from test_flight_e2e import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    import test_server_detectors as tsd

    return tsd


def parity_run() -> list[dict]:
    """Every rule once among the cases that must not confuse them: a blip, a near-uniform layer,
    a failed probe, NaN heads, a probe that lost a module, and a layer that appears late."""
    def extra(i, probe):
        if i == 4:
            probe["attention"]["blk0"]["entropy_share"][0] = "nan"
            probe["attention"]["blk3"]["max_logit"] = ["-inf"] * HEADS
        if i == 7:
            probe.clear()
            probe["error"] = "RuntimeError: boom"
        if i == 9:
            del probe["attention"]["blk2"]
    return probe_run({
        "blk0": lambda i: (0.995 if i >= 3 else 0.95, 4.0),  # near uniform throughout: nothing
        "blk1": lambda i: (0.999 if 5 <= i <= 8 else 0.35, 12.0),  # uniform at probe 5
        "blk2": lambda i: (0.12 if i in (3, 10, 11, 12) else 0.4, 9.0),  # a blip at 3, collapse at 10
        "blk3": lambda i: (0.5, 30.0 if i >= 8 else 11.0),  # logits grow from probe 8
        "late": lambda i: None if i < 6 else (0.3, 5.0),
    }, 14, extra=extra)


def test_the_parity_fixture_has_every_attention_kind():
    found = {i["kind"]: i for i in ev.detect_attention_anomalies(parity_run())}
    assert {k: (i["step"], [m["module"] for m in i["modules"]]) for k, i in found.items()} == {
        "attention_uniform": (59, ["blk1"]), "attention_collapse": (109, ["blk2"]),
        "attention_logit_growth": (89, ["blk3"]),
    }


def test_the_server_reports_the_same_attention_incidents_after_every_stretch(tmp_path):
    tsd = _server_tools()
    records = parity_run()
    stream = tsd.Stream(tmp_path, *tsd.FAST, run="attn")
    sent = 0
    # Stretches that end on an onset probe (59, 89, 109) before its confirmation, and between.
    for upto in (25, 60, 69, 75, 90, 100, 110, 125, 140):
        document = stream.send(records[sent:upto])
        sent = upto
        expected = [i for i in tensorquill.build_evidence(records[:upto])["incidents"] if i["kind"] in ev.ATTN_KINDS]
        assert tsd.differences(expected, document["incidents"]) == [], f"after {upto} steps"
        expected_nc = [n for n in tensorquill.build_evidence(records[:upto]).get("not_checked", []) if n["check"] == "attention"]
        assert [n for n in document["not_checked"] if n["check"] == "attention"] == expected_nc, f"after {upto} steps"
    assert {i["kind"] for i in document["incidents"]} == set(ev.ATTN_KINDS)
    assert all("scalars" in i for i in document["incidents"])
    assert set(ev.ATTN_KINDS) <= set(document["checked"])
    assert document["thresholds"]["attention_logit_factor"] == ev.ATTN_LOGIT_FACTOR
    # The server's own Parquet gives Python the same answer: the probe keeps the model's order there.
    from_store = tensorquill.build_evidence(stream.store / "runs" / "attn", rank=0)["incidents"]
    assert tsd.differences([i for i in from_store if i["kind"] in ev.ATTN_KINDS], document["incidents"]) == []
    assert stream.server.terminate() == 0


def edge_run(run: str) -> list[dict]:
    """Two cases the long fixture does not hold: a failed probe that still carries attention
    (`f` would collapse there), and a module missing from the probe between two low ones (`g`)."""
    def extra(i, probe):
        if i == 3:
            probe["error"] = "RuntimeError: boom"
            probe["attention"]["f"] = attention(0.12, 9.0)
        if i == 3 or i == 8:
            probe["attention"].pop("g", None)
    return probe_run({
        "f": lambda i: (0.12 if i in (4, 5) else 0.4, 9.0),
        "g": lambda i: (0.12 if i in (7, 9) else 0.4, 9.0),
        "h": steady(),
    }, 11, run=run, extra=extra)


def test_failed_probes_and_broken_runs_are_read_the_same_in_the_server(tmp_path):
    tsd = _server_tools()
    records = edge_run("edges")
    expected = [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in ev.ATTN_KINDS]
    # f collapses at probe 4 (the failed probe 3 is skipped); g's two low probes are not in a row.
    assert [(i["kind"], i["step"], [m["module"] for m in i["modules"]]) for i in expected] == [
        ("attention_collapse", 49, ["f"])]
    assert expected[0]["probes"] == 10
    stream = tsd.Stream(tmp_path, *tsd.FAST, run="edges")
    document = stream.send(records)
    assert tsd.differences(expected, document["incidents"]) == []
    assert stream.server.terminate() == 0


def test_a_restart_between_onset_and_confirmation_changes_nothing(tmp_path):
    tsd = _server_tools()
    records = parity_run()
    stream = tsd.Stream(tmp_path, *tsd.FAST, run="attn")
    stream.send(records[:60])  # blk1's first uniform probe is stored, not yet confirmed
    stream.restart()
    document = stream.send(records[60:])
    expected = [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in ev.ATTN_KINDS]
    assert tsd.differences(expected, document["incidents"]) == [] and document["rebuilds"] == 1
    assert stream.server.terminate() == 0


def test_the_seeded_gpu_probes_give_the_same_incidents_in_the_server(tmp_path):
    """The probes of the recorded SmolLM2 run (block 8 seeded at step 250), if this machine has them."""
    tsd = _server_tools()
    from pathlib import Path

    source = Path.home() / "Downloads" / "runs" / "dense-layer8" / "dense.jsonl"
    if not source.exists():
        pytest.skip("the recorded GPU capture is not on this machine")
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "probe" in r:
            records.append(record(r["step"], run="attn", grad_norm=1.0, probe=r["probe"]))
    expected = [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in ev.ATTN_KINDS]
    assert [(i["kind"], i["step"], i["affected"]) for i in expected] == [("attention_uniform", 299, 23)]
    stream = tsd.Stream(tmp_path, *tsd.FAST, run="attn")
    document = stream.send(records)
    assert tsd.differences(expected, document["incidents"]) == []
    assert stream.server.terminate() == 0


def test_the_baseline_is_the_last_probes_so_a_slow_drift_is_not_a_collapse():
    # 0.8 for five probes, then 0.42 (0.525 times: not a collapse, so it joins the baseline),
    # then 0.3. Against the last five probes (0.42) that is 0.71 times; against the whole run's
    # median (0.61) it would be 0.49 times, a collapse.
    share = lambda i: (0.8 if i < 5 else 0.42 if i < 10 else 0.3, 10.0)
    assert kinds(probe_run({"a": share}, 13)) == []
    # A long steady run is judged against ATTN_WINDOW probes, not all of them.
    found = ev.detect_attention_anomalies(probe_run({"a": lambda i: (0.15 if i >= 9 else 0.4, 10.0)}, 11))
    assert found[0]["modules"][0]["baseline_probes"] == ev.ATTN_WINDOW
