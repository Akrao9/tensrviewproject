"""Every incident kind's class (invariant, anomaly, observation), on every incident and in the
run state: one table per language, kept identical, and observations never make a run worse.

The server half needs `pyarrow` and the server binary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import tensorquill
from tensorquill import evidence as ev
from test_run_state import ACTIVE, comparable, server_state, state, wait_for_state

DETECT_RS = Path(__file__).resolve().parents[2] / "server" / "src" / "detect.rs"

# Pinned here as well as in both tables, so a class changed in both still fails a test. The
# reason for each is in the notes on the decision.
EXPECTED = {
    "invariant": [
        "stalled_weights", "repeated_steps", "loss_scale_trouble", "data_stalled", "epoch_order_repeated",
        "duplicate_batches", "non_finite", "replica_drift", "hardware_errors", "collective_hang", "collective_desync",
        "traincheck_violation",
    ],
    "anomaly": [
        "moe_imbalance", "starved_experts", "grad_spike", "activation_spike", "activation_drift", "dead_activations",
        "attention_collapse", "attention_uniform", "attention_logit_growth", "memory_growth", "input_starvation",
    ],
    "observation": [
        "resume_lr_change", "resume_data_mismatch", "resume_incomplete", "memory_pressure", "allocator_retries",
        "gpu_throttled", "memory_imbalance", "job_stalled", "rank_silent",
    ],
}
# The kinds reported outside a rank's `incidents.json`: the server's status, and Python alone.
STREAM_KINDS = {"job_stalled", "rank_silent", "duplicate_batches"}
PYTHON_ONLY = {"memory_imbalance", "collective_hang", "collective_desync", "traincheck_violation"}


# --- the table ----------------------------------------------------------------------------------


def test_every_kind_has_the_class_its_rule_supports():
    assert ev.INCIDENT_CLASSES == {kind: name for name, kinds in EXPECTED.items() for kind in kinds}
    assert set(ev.INCIDENT_CLASSES.values()) == set(ev.INCIDENT_CLASS_NAMES)


def test_the_table_has_every_kind_reported_and_no_other():
    every = set(ev.SERVER_KINDS) | STREAM_KINDS | PYTHON_ONLY
    assert len(every) == 32
    assert set(ev.INCIDENT_CLASSES) == every


def rust_table() -> list[tuple[str, str]]:
    source = DETECT_RS.read_text(encoding="utf-8")
    found = re.search(r"pub const INCIDENT_CLASSES: \[\(&str, &str\); (\d+)\] = \[(.*?)\];", source, re.S)
    assert found, "INCIDENT_CLASSES is not in server/src/detect.rs in the shape this test reads"
    pairs = re.findall(r'\(\s*"(\w+)",\s*"(\w+)",?\s*\)', found.group(2))
    assert len(pairs) == int(found.group(1))
    return pairs


def test_the_servers_table_is_this_one():
    assert rust_table() == list(ev.INCIDENT_CLASSES.items())


def test_a_kind_this_version_does_not_know_has_no_class():
    assert ev.incident_class("something_new") is None and ev.incident_class(None) is None
    incident = {"kind": "something_new", "step": 3, "class": "observation"}
    ev.with_classes([incident])
    assert incident == {"kind": "something_new", "step": 3, "class": "observation"}  # kept, never trusted


# --- on every incident --------------------------------------------------------------------------


def test_every_incident_carries_its_class_right_after_its_kind():
    sys.path.insert(0, str(Path(__file__).parent))
    from test_server_detectors import everything

    incidents = tensorquill.build_evidence(everything())["incidents"]
    assert len({i["kind"] for i in incidents}) == 10
    for incident in incidents:
        assert incident["class"] == ev.INCIDENT_CLASSES[incident["kind"]]
        assert list(incident)[:2] == ["kind", "class"]


def test_stamping_twice_changes_nothing_and_replaces_a_wrong_class():
    incident = {"kind": "gpu_throttled", "class": "invariant", "step": 4}
    ev.with_classes([incident])
    ev.with_classes([incident])
    assert list(incident.items()) == [("kind", "gpu_throttled"), ("class", "observation"), ("step", 4)]


# --- the run state ------------------------------------------------------------------------------


def test_observations_alone_leave_a_run_healthy_and_are_listed():
    found = ev.run_state(incidents=[{"kind": "gpu_throttled", "step": 5}, {"kind": "resume_lr_change", "step": 9},
                                    {"kind": "gpu_throttled", "step": 7}], steps_judged=True, status=ACTIVE)
    assert found["state"] == "healthy"
    [reason] = found["reasons"]
    assert (reason["state"], reason["code"], reason["kinds"]) == ("healthy", "observations", ["gpu_throttled", "resume_lr_change"])
    line = ev.run_state_line({"run_state": found})
    assert line == ("Run state: healthy (3 observations listed, not counted as failures: gpu_throttled,"
                    " resume_lr_change)")


@pytest.mark.parametrize("kind", EXPECTED["observation"])
def test_no_observation_makes_a_run_unhealthy(kind):
    stream = kind in ("job_stalled", "rank_silent")
    status = {**ACTIVE, "summary": "stalled" if kind == "job_stalled" else "degraded" if stream else "active"}
    found, codes = state(incidents=[{"kind": kind, "step": 3}], status=status)
    assert found == ("unknown" if stream else "healthy")
    assert codes == (["status_summary", "observations"] if stream else ["observations"])


@pytest.mark.parametrize("kind", EXPECTED["invariant"] + EXPECTED["anomaly"])
def test_every_invariant_and_anomaly_makes_a_run_unhealthy(kind):
    assert state(incidents=[{"kind": kind, "step": 3}]) == ("unhealthy", ["incidents"])


def test_observations_never_lift_a_state_either():
    observed = [{"kind": "memory_imbalance", "step": 3}]
    assert state(incidents=observed, status=None) == ("degraded", ["no_status", "observations"])
    assert state(incidents=observed, steps_judged=False) == ("unknown", ["no_steps", "observations"])
    found = ev.run_state(incidents=[*observed, {"kind": "grad_spike", "step": 4}], steps_judged=True, status=ACTIVE)
    assert found["state"] == "unhealthy"
    assert [(r["code"], r["kinds"]) for r in found["reasons"]] == [("incidents", ["grad_spike"]),
                                                                 ("observations", ["memory_imbalance"])]
    assert found["reasons"][0]["detail"] == "1 incident listed: grad_spike"


def test_a_kind_this_version_does_not_know_counts_against_the_run():
    # Even when it calls itself an observation: a newer version's kind is not assumed harmless.
    assert state(incidents=[{"kind": "something_new", "class": "observation"}]) == ("unhealthy", ["incidents"])


def test_a_stall_is_unknown_whatever_its_incidents_say():
    for summary, kind in (("stalled", "job_stalled"), ("degraded", "rank_silent")):
        status = {**ACTIVE, "summary": summary}
        assert state(status=status, incidents=[{"kind": kind}]) == ("unknown", ["status_summary", "observations"])
        assert state(status=status) == ("unknown", ["status_summary"])
        # An invariant beside it (a stuck collective from the Flight Recorder) is unhealthy.
        found = state(status=status, incidents=[{"kind": kind}, {"kind": "collective_hang"}])
        assert found == ("unhealthy", ["incidents", "status_summary", "observations"])


def test_the_class_follows_the_table(monkeypatch):
    monkeypatch.setitem(ev.INCIDENT_CLASSES, "gpu_throttled", "anomaly")
    assert state(incidents=[{"kind": "gpu_throttled"}]) == ("unhealthy", ["incidents"])
    monkeypatch.setitem(ev.INCIDENT_CLASSES, "grad_spike", "observation")
    assert state(incidents=[{"kind": "grad_spike"}]) == ("healthy", ["observations"])


# --- in the CLI ---------------------------------------------------------------------------------


def test_the_summary_names_each_class_once():
    evidence = {"incidents": [{"kind": "gpu_throttled", "step": 5, "reasons": ["hw_thermal"]},
                              {"kind": "non_finite", "step": 6, "layers": ["a"], "gradients": []},
                              {"kind": "something_new", "step": 7}], "samples": [], "layer_count": 0}
    lines = tensorquill.incident_summary(evidence).splitlines()
    assert lines[3] == ("By class: invariant non_finite; observation gpu_throttled; unclassified something_new"
                        " (observations claim no failure and do not make the run unhealthy)")
    assert "By class" not in tensorquill.incident_summary({"incidents": [], "samples": []})


# --- through the server -------------------------------------------------------------------------


def _server():
    pytest.importorskip("pyarrow.flight")
    from test_flight_e2e import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    import test_watch_all_ranks as twa

    return twa


def throttled(twa, rank: int, peak_gib: float | None = None) -> list[dict]:
    """Steady steps; a thermal throttle at step 30 when `rank` is 1, and a memory reading with
    the given peak at step 40 when it is set."""
    from test_evidence_system import GIB, cuda, nvml

    out = twa.steady(60)
    if rank == 1:
        out[30] = {**out[30], "system": {"nvml": nvml(throttle=("hw_thermal",))}}
    if peak_gib is not None:
        out[40] = {**out[40], "system": {"cuda": cuda(peak=int(peak_gib * GIB))}}
    return out


def test_the_server_classes_every_incident_and_an_observation_leaves_both_healthy(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    senders = [run.keep_open(rank, throttled(twa, rank)) for rank in (0, 1)]
    found = wait_for_state(run, "healthy")
    report = json.loads(run.report_file(1).read_text())
    assert [(i["kind"], i["class"]) for i in report["incidents"]] == [("gpu_throttled", "observation")]
    assert found["reasons"] == [{"state": "healthy", "code": "observations", "kinds": ["gpu_throttled"],
                                 "detail": found["reasons"][0]["detail"]}]
    evidence, summary = twa.look(run, tmp_path / "a")
    assert comparable(evidence["run_state"]) == comparable(found)
    [group] = evidence["incident_groups"]
    assert (group["class"], group["incident"]["class"]) == ("observation", "observation")
    assert "By class: observation gpu_throttled" in summary
    # A spike on rank 0: unhealthy, with the observation still listed, in both.
    run.file(0).open("a").write("".join(json.dumps({**r, "rank": 0}) + "\n" for r in twa.steady(120, spike=(100, 9.0))[60:]))
    run.wait_for(0, twa.steady(120)[-1])
    found = wait_for_state(run, "unhealthy")
    assert [(r["code"], r["kinds"]) for r in found["reasons"]] == [("incidents", ["grad_spike"]),
                                                                 ("observations", ["gpu_throttled"])]
    report = json.loads(run.report_file(0).read_text())
    assert [(i["kind"], i["class"]) for i in report["incidents"]] == [("grad_spike", "anomaly")]
    evidence, _ = twa.look(run, tmp_path / "b")
    assert comparable(evidence["run_state"]) == comparable(found)
    for sender in senders:
        sender.close()
    assert run.server.terminate() == 0


def test_a_cross_rank_observation_only_python_sees_leaves_the_states_equal(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS)
    senders = [run.keep_open(rank, throttled(twa, 0, peak_gib=peak)) for rank, peak in ((0, 3.0), (1, 6.0))]
    found = wait_for_state(run, "healthy")
    assert found["reasons"] == []  # the server sees one rank at a time
    evidence, _ = twa.look(run, tmp_path / "a")
    [imbalance] = evidence["run_incidents"]
    assert (imbalance["kind"], imbalance["class"]) == ("memory_imbalance", "observation")
    assert evidence["run_state"]["state"] == found["state"] == "healthy"
    assert [r["code"] for r in evidence["run_state"]["reasons"]] == ["observations"]
    for sender in senders:
        sender.close()
    assert run.server.terminate() == 0


def test_a_stall_is_unknown_in_both_and_its_incident_is_an_observation(tmp_path):
    twa = _server()
    from test_server_reports import QUICK_STATUS

    run = twa.Run(tmp_path, *twa.FAST, *QUICK_STATUS, "--silent-seconds", "1", "--silent-factor", "0")
    senders = [run.keep_open(rank, twa.steady(60)) for rank in (0, 1)]
    try:
        status = twa.wait_for_status(run, lambda s: s["summary"] == "stalled" and s.get("run_state", {}).get("reasons"))
        assert [(i["kind"], i["class"]) for i in status["incidents"]] == [("job_stalled", "observation")]
        found = server_state(run)
        assert comparable(found) == ("unknown", [("observations", ("job_stalled",), (), ()),
                                                 ("status_summary", (), (), ())])
        evidence, _ = twa.look(run, tmp_path / "a")
        assert comparable(evidence["run_state"]) == comparable(found)
        assert evidence["run_incidents"][0]["class"] == "observation"
    finally:
        for sender in senders:
            sender.close(0.0)
    assert run.server.terminate() == 0
