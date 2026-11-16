"""What was checked: `evidence["kind_checks"]` gives every incident kind one of three values.

- `checked`: records carried the kind's input and its rule ran over them.
- `not_configured`: the capture was never set up to provide it. A router-only capture has no
  weight or activation checks, no `scaler=` means no loss-scale check, and a local file that
  no server was watching has no stall check.
- `unavailable`: the input was asked for and could not be judged (a probe that failed, a file
  that could not be read, too few probes, a server's store with no progress status).

Only `unavailable` makes the run `degraded`, and the summary carries the count
("3 of 32 kinds checked"). These tests are the Python side and need nothing but the SDK.
test_server_detect.py holds the Rust report to the same values on every parity case.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import tensorquill
import tensorquill.evidence as ev

from test_capture import train

DETECT_RS = Path(__file__).resolve().parents[2] / "server" / "src" / "detect.rs"
ATTENTION = ("attention_collapse", "attention_uniform", "attention_logit_growth")
WEIGHTS_AND_ACTIVATIONS = ("stalled_weights", "activation_spike", "activation_drift", "dead_activations")
STALLS = ("job_stalled", "rank_silent")
ACTIVE = {"schema_version": 1, "summary": "active", "ranks": [{"rank": 0, "state": "active"}]}


def values(evidence: dict) -> dict[str, str]:
    return {kind: check["value"] for kind, check in evidence["kind_checks"].items()}


def codes(evidence: dict) -> list[str]:
    return [reason["code"] for reason in evidence["run_state"]["reasons"]]


def router_only(steps: int = 60) -> list[dict]:
    """What a router-only capture records: each router's expert counts, and nothing else."""
    return [
        {"run_id": "r", "step": step, "ts": float(step), "moe": {"router": [250 + (step + e) % 3 for e in range(8)]},
         "moe_top_k": {"router": 1}}
        for step in range(steps)
    ]


def probed(steps: int = 60, fail_at: tuple[int, ...] = (), every: int = 10) -> list[dict]:
    """Steady attention, probed every `every` steps; the probes at `fail_at` failed."""
    records = []
    for step in range(steps):
        record: dict = {"run_id": "r", "step": step, "ts": float(step)}
        if step % every == 0:
            record["probe"] = (
                {"error": "RuntimeError: CUDA out of memory"} if step in fail_at
                else {"attention": {"blocks.0.attn": {"entropy_share": [0.6, 0.62], "max_logit": [2.0, 2.1]}}}
            )
        records.append(record)
    return records


# --- the table ------------------------------------------------------------------------------


def test_every_kind_gets_a_value_and_the_servers_kinds_each_have_an_input():
    assert tuple(ev.KIND_INPUTS) == ev.SERVER_KINDS
    assert set(ev.KIND_INPUTS.values()) == set(ev.INPUT_NEEDS)
    checks = tensorquill.build_evidence(router_only())["kind_checks"]
    assert list(checks) == list(ev.INCIDENT_CLASSES) and len(checks) == 32
    for kind, check in checks.items():
        assert check["value"] in ev.KIND_CHECK_VALUES and isinstance(check["needs"], str) and check["needs"]
        assert ("records" in check) == (kind in ev.SERVER_KINDS), kind
        assert ("reason" in check) == (check["value"] == "unavailable"), kind


def rust_pairs(name: str) -> list[tuple[str, str]]:
    source = DETECT_RS.read_text(encoding="utf-8")
    found = re.search(rf"pub const {name}: \[\(&str, &str\); (\d+)\] = \[(.*?)\];", source, re.S)
    assert found, f"{name} is not in server/src/detect.rs in the shape this test reads"
    pairs = re.findall(r'\("([^"]*)",\s*"([^"]*)"\)', found.group(2))
    assert len(pairs) == int(found.group(1))
    return pairs


def test_the_rust_tables_are_these_tables():
    assert rust_pairs("KIND_INPUTS") == list(ev.KIND_INPUTS.items())
    assert rust_pairs("INPUT_NEEDS") == list(ev.INPUT_NEEDS.items())


# --- not configured: the capture was never set up to provide it -------------------------------


def test_a_router_only_capture_reads_healthy_and_lists_what_it_cannot_show():
    evidence = tensorquill.build_evidence(router_only())
    found = values(evidence)
    assert all(found[kind] == "not_configured" for kind in WEIGHTS_AND_ACTIVATIONS), found
    assert all(found[kind] == "not_configured" for kind in ("grad_spike", "non_finite", "loss_scale_trouble", "replica_drift"))
    assert [kind for kind, value in found.items() if value == "checked"] == ["moe_imbalance", "starved_experts", "repeated_steps"]
    assert "unavailable" not in found.values()
    assert evidence["incidents"] == [] and evidence["run_state"] == {"state": "healthy", "reasons": []}
    assert evidence["kind_checks"]["moe_imbalance"] == {"value": "checked", "records": 60, "needs": "moe: a router's expert counts"}
    assert evidence["kind_checks"]["stalled_weights"] == {
        "value": "not_configured", "records": 0, "needs": "weights_unchanged with grad_norms (full capture)",
    }
    # The summary says how much of the list "no incidents" covers.
    assert ev.incident_summary(evidence).splitlines()[:2] == ["No incidents detected.", "3 of 32 kinds checked, 29 not configured"]
    assert ev.run_state_line(evidence) == "Run state: healthy"


def test_a_real_router_only_capture_has_no_weight_or_activation_check_and_full_capture_has(toy_moe):
    sink = tensorquill.MemorySink()
    with tensorquill.init(toy_moe, sink=sink, every=2, routers={"router": 4}):
        train(toy_moe, 6)
    found = values(tensorquill.build_evidence(sink.records))
    assert found["moe_imbalance"] == "checked" and found["repeated_steps"] == "checked"
    assert all(found[kind] == "not_configured" for kind in WEIGHTS_AND_ACTIVATIONS), found
    assert all(found[kind] == "not_configured" for kind in (*STALLS, "loss_scale_trouble", "replica_drift", *ATTENTION))

    full = tensorquill.MemorySink()
    with tensorquill.init(toy_moe, sink=full, every=2, routers={"router": 4}, capture="full", weights=2):
        train(toy_moe, 6)
    found = values(tensorquill.build_evidence(full.records))
    assert all(found[kind] == "checked" for kind in (*WEIGHTS_AND_ACTIVATIONS, "grad_spike", "non_finite")), found


def test_no_scaler_and_no_replica_group_mean_no_loss_scale_and_no_drift_check():
    plain = [{"run_id": "r", "step": s, "ts": float(s), "grad_norm": 1.0} for s in range(30)]
    found = values(tensorquill.build_evidence(plain))
    assert (found["loss_scale_trouble"], found["replica_drift"], found["grad_spike"]) == ("not_configured", "not_configured", "checked")
    scaled = [{**r, "step_skipped": False, "grad_scale": 65536.0, "replica_spread": {"norm.weight": [0.0, 0.0, 0.0]}} for r in plain]
    evidence = tensorquill.build_evidence(scaled)
    assert (values(evidence)["loss_scale_trouble"], values(evidence)["replica_drift"]) == ("checked", "checked")
    assert evidence["kind_checks"]["loss_scale_trouble"]["records"] == 30
    assert evidence["run_state"]["state"] == "healthy"


def test_what_counts_as_carrying_an_input():
    def count(record: dict) -> dict[str, int]:
        return {name: n for name, n in ev.input_counts([{"step": 0, **record}]).items() if n and name != "steps"}

    assert ev.input_counts([]) == dict.fromkeys(ev.INPUT_NEEDS, 0)
    assert count({}) == {}
    assert count({"moe": {"a": [0, 0], "b": [0, 3]}}) == {"moe": 1}
    assert count({"moe": {"a": [0, 0]}}) == {} and count({"moe": {}}) == {}  # nothing routed: nothing to judge
    assert count({"grad_norm": 1.0}) == {"grad_total": 1, "numbers": 1}
    assert count({"clip_grad_norm": "nan"}) == {"grad_total": 1, "numbers": 1}  # recorded, though not finite
    assert count({"grad_norms": {"w": 1.0}}) == {"numbers": 1}
    assert count({"grad_norms": {}, "layers": {}, "replica_spread": {}}) == {}
    # A weight check that found every parameter changed is a check; one without gradients judges nothing.
    assert count({"weights_unchanged": [], "grad_norms": {"w": 1.0}}) == {"weight_check": 1, "numbers": 1}
    assert count({"weights_unchanged": ["w"]}) == {}
    assert count({"step_skipped": False}) == {"loss_scale": 1}
    assert count({"lr": [1e-3]}) == {"lr": 1} and count({"lr": []}) == {} and count({"lr": ["nan"]}) == {}
    assert count({"data": {"batches": 1}}) == {"data": 1} and count({"checkpoint": {"save": {}}}) == {"checkpoint": 1}
    assert count({"layers": {"blocks.0": {"std": 1.0}}}) == {"layers": 1, "numbers": 1}
    assert count({"probe": {"attention": {"a": {}}}}) == {"attention": 1}
    assert count({"probe": {"attention": {}}}) == {} and count({"probe": {"error": "x", "attention": {"a": {}}}}) == {}
    assert count({"replica_spread": {"w": [0.0, 0.0]}}) == {"replica_spread": 1}
    # A reading that could not be taken was still asked for.
    assert count({"system": {"cuda": {"unavailable": "no CUDA"}, "nvml": {"xid": []}}}) == {"cuda": 1, "nvml": 1}
    assert count({"system": {"cuda": None}}) == {}
    assert count({"data_wait": 0.1, "step_time": 0.5}) == {"data_wait": 1}
    assert count({"data_wait": 0.1}) == {} and count({"data_wait": 0.1, "step_time": 0.0}) == {}


# --- unavailable: asked for, and could not be judged ------------------------------------------


def test_a_failed_probe_reads_degraded_and_says_which_kinds_it_leaves_unjudged():
    clean = tensorquill.build_evidence(probed())
    assert all(values(clean)[kind] == "checked" for kind in ATTENTION)
    assert clean["run_state"]["state"] == "healthy" and "not_checked" not in clean
    assert clean["kind_checks"]["attention_collapse"]["records"] == 6

    evidence = tensorquill.build_evidence(probed(fail_at=(30,)))
    entry = {
        "check": "probe", "kinds": list(ATTENTION), "step": 30,
        "reason": "1 of 6 probes failed, the first at step 30: RuntimeError: CUDA out of memory",
    }
    assert evidence["not_checked"] == [entry]
    assert evidence["incidents"] == []
    assert evidence["run_state"]["state"] == "degraded" and codes(evidence) == ["not_checked"]
    for kind in ATTENTION:
        # Five probes did record attention: one that failed is enough, as one failed memory reading is.
        assert evidence["kind_checks"][kind] == {
            "value": "unavailable", "records": 5, "needs": "probe with attention (probe=)",
            "check": "probe", "reason": entry["reason"],
        }
    summary = ev.incident_summary(evidence).splitlines()
    assert summary[1] == "1 of 32 kinds checked, 28 not configured, 3 unavailable (attention_collapse, attention_uniform, attention_logit_growth)"
    assert summary[2] == f"Not checked (probe): {entry['reason']}"
    assert ev.run_state_line(evidence) == "Run state: degraded (not checked: probe)"

    # Every probe failed: nothing recorded attention, and the kinds are still not "not configured".
    every = tensorquill.build_evidence(probed(fail_at=tuple(range(0, 60, 10))))
    assert every["kind_checks"]["attention_uniform"]["records"] == 0
    assert all(values(every)[kind] == "unavailable" for kind in ATTENTION) and every["run_state"]["state"] == "degraded"
    assert every["not_checked"][0]["reason"].startswith("6 of 6 probes failed, the first at step 0")
    # An error that is not text is written as compact JSON, as the server writes it.
    odd = [{"run_id": "r", "step": 0, "ts": 0.0, "probe": {"error": {"code": 3}}}]
    assert tensorquill.build_evidence(odd)["not_checked"][0]["reason"] == '1 of 1 probe failed, the first at step 0: {"code":3}'


def test_too_few_probes_to_judge_are_unavailable_under_their_own_entry():
    evidence = tensorquill.build_evidence(probed(steps=30))  # three probes, four needed
    assert [n["check"] for n in evidence["not_checked"]] == ["attention"]
    check = evidence["kind_checks"]["attention_logit_growth"]
    assert (check["value"], check["check"], check["records"]) == ("unavailable", "attention", 3)
    assert check["reason"] == evidence["not_checked"][0]["reason"]
    assert evidence["run_state"]["state"] == "degraded"
    # Too few and one failed: both entries are listed, and the kinds name the first.
    both = tensorquill.build_evidence(probed(steps=30, fail_at=(20,)))
    assert [n["check"] for n in both["not_checked"]] == ["attention", "probe"]
    assert both["kind_checks"]["attention_collapse"]["check"] == "attention"


def test_a_kind_named_by_a_not_checked_entry_is_unavailable_however_many_records_it_has():
    counts = {**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 9, "lr": 9, "cuda": 4, "nvml": 4}
    not_checked = [
        {"check": "resume_lr", "step": 5, "rank": 0, "reason": "no learning rate"},  # no `kinds`: by its check
        {"check": "nvml", "metric": "xid", "kinds": ["hardware_errors"], "reason": "not supported", "step": 0},
        {"check": "nvml", "metric": "ecc", "kinds": ["hardware_errors"], "reason": "second", "step": 0},
        {"check": "something_new", "reason": "a check this version does not know names no kind"},
        "not an entry",
    ]
    checks = ev.kind_checks(counts, not_checked)
    assert checks["resume_lr_change"] == {"value": "unavailable", "records": 9, "needs": "lr", "check": "resume_lr", "reason": "no learning rate"}
    assert checks["hardware_errors"]["reason"] == "not supported"  # the first entry that names it
    assert checks["gpu_throttled"] == {"value": "checked", "records": 4, "needs": "system.nvml (system=True)"}
    assert checks["memory_growth"]["value"] == "checked" and checks["repeated_steps"]["value"] == "checked"
    assert [kind for kind, check in checks.items() if check["value"] == "unavailable"] == ["resume_lr_change", "hardware_errors"]
    assert sum(check["value"] == "not_configured" for check in checks.values()) == 25 - 2 - 5


# --- the progress status: the rule that changed -------------------------------------------------


def test_a_local_file_no_server_watched_has_no_stall_check_to_miss(tmp_path):
    records = router_only()
    path = tmp_path / "run.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    for source in (records, path):
        evidence = tensorquill.build_evidence(source)
        assert all(evidence["kind_checks"][kind] == {"value": "not_configured", "needs": "the server's progress status (watchdog.json)"}
                   for kind in STALLS)
        assert "no_status" not in codes(evidence) and evidence["run_state"]["state"] == "healthy"


def test_a_status_that_was_named_and_cannot_be_read_is_unavailable_and_degrades(tmp_path):
    evidence = tensorquill.build_evidence(router_only(), watchdog=tmp_path / "watchdog.json")
    for kind in STALLS:
        assert evidence["kind_checks"][kind] == {
            "value": "unavailable", "needs": "the server's progress status (watchdog.json)", "check": "status",
            "reason": "no progress status (the server's watchdog.json): whether the job is still making progress is not checked",
        }
    assert evidence["run_state"]["state"] == "degraded" and codes(evidence) == ["no_status"]
    assert ev.incident_summary(evidence).splitlines()[1] == "3 of 32 kinds checked, 27 not configured, 2 unavailable (job_stalled, rank_silent)"

    read = tensorquill.build_evidence(router_only(), watchdog=ACTIVE)
    assert all(read["kind_checks"][kind] == {"value": "checked", "needs": "the server's progress status (watchdog.json)"} for kind in STALLS)
    assert read["run_state"] == {"state": "healthy", "reasons": []}
    assert ev.incident_summary(read).splitlines()[1] == "5 of 32 kinds checked, 27 not configured"


def test_a_servers_store_without_its_status_is_unavailable_and_degrades(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from tensorquill.flight import records_to_batch

    import pyarrow as pa

    folder = tmp_path / "runs" / "r" / "rank=0"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_batches([records_to_batch(router_only())]), folder / "part-0-59.parquet")
    for source in (folder.parent, folder, folder / "part-0-59.parquet"):
        evidence = tensorquill.build_evidence(source)
        assert all(values(evidence)[kind] == "unavailable" for kind in STALLS), source
        assert evidence["run_state"]["state"] == "degraded" and codes(evidence) == ["no_status"]
    (folder.parent / "watchdog.json").write_text(json.dumps(ACTIVE), encoding="utf-8")
    evidence = tensorquill.build_evidence(folder.parent)
    assert all(values(evidence)[kind] == "checked" for kind in STALLS) and evidence["run_state"]["state"] == "healthy"


def test_run_state_takes_whether_a_status_was_expected():
    usual = ev.run_state(incidents=[], steps_judged=True, status=None)
    assert (usual["state"], [r["code"] for r in usual["reasons"]]) == ("degraded", ["no_status"])
    local = ev.run_state(incidents=[], steps_judged=True, status=None, status_expected=False)
    assert local == {"state": "healthy", "reasons": []}
    # Not expected, and there anyway: it is read as any status is.
    stalled = ev.run_state(incidents=[], steps_judged=True, status={**ACTIVE, "summary": "stalled"}, status_expected=False)
    assert stalled["state"] == "unknown"
    # Evidence without the key (a server's report read by a watcher) expects a status, as before.
    assert ev._per_rank_run_state({"incidents": []}, None, None, steps_judged=True)["state"] == "degraded"
    not_configured = {"incidents": [], "kind_checks": {"job_stalled": {"value": "not_configured"}}}
    assert ev._per_rank_run_state(not_configured, None, None, steps_judged=True)["state"] == "healthy"
    unavailable = {"incidents": [], "kind_checks": {"job_stalled": {"value": "unavailable"}}}
    assert ev._per_rank_run_state(unavailable, None, None, steps_judged=True)["state"] == "degraded"


# --- the kinds read from other files ------------------------------------------------------------


def test_a_file_that_was_given_is_checked_or_unavailable_and_one_that_was_not_is_not_configured(tmp_path):
    from test_traincheck import _summary

    found = values(tensorquill.build_evidence(router_only()))
    assert all(found[kind] == "not_configured" for kind in ("collective_hang", "collective_desync", "traincheck_violation"))
    assert all(found[kind] == "not_configured" for kind in ("duplicate_batches", "memory_imbalance"))

    good = tmp_path / "violations_summary.json"
    good.write_text(json.dumps(_summary([])), encoding="utf-8")
    evidence = tensorquill.build_evidence(router_only(), traincheck=good)
    assert evidence["kind_checks"]["traincheck_violation"] == {"value": "checked", "needs": "TrainCheck results (traincheck=)"}
    assert evidence["run_state"]["state"] == "healthy"

    evidence = tensorquill.build_evidence(router_only(), traincheck=tmp_path / "missing.json", flight_recorder=tmp_path / "no-dumps")
    for kind, check in (("traincheck_violation", "traincheck"), ("collective_hang", "flight_recorder"), ("collective_desync", "flight_recorder")):
        entry = next(n for n in evidence["not_checked"] if n["check"] == check)
        assert evidence["kind_checks"][kind]["value"] == "unavailable"
        assert (evidence["kind_checks"][kind]["check"], evidence["kind_checks"][kind]["reason"]) == (check, entry["reason"])
    assert evidence["run_state"]["state"] == "degraded" and codes(evidence) == ["not_checked"]
    assert ev.checked_counts(evidence) == {
        "kinds": 32, "checked": 3, "not_configured": 26, "unavailable": 3,
        "unavailable_kinds": ["collective_hang", "collective_desync", "traincheck_violation"],
    }


# --- the count ----------------------------------------------------------------------------------


def test_the_count_is_out_of_every_kind_and_evidence_without_the_key_has_no_line():
    assert ev.checked_counts({}) is None and ev.checked_line({"incidents": []}) is None
    assert "kinds checked" not in ev.incident_summary({"incidents": [], "samples": []})
    checks = {"grad_spike": {"value": "checked"}, "non_finite": {"value": "unavailable"}, "not_a_kind": {"value": "checked"}}
    assert ev.checked_counts({"kind_checks": checks}) == {
        "kinds": 32, "checked": 1, "not_configured": 0, "unavailable": 1, "unavailable_kinds": ["non_finite"],
    }
    assert ev.checked_line({"kind_checks": checks}) == "1 of 32 kinds checked, 0 not configured, 1 unavailable (non_finite)"
    # With incidents the line follows them.
    evidence = tensorquill.build_evidence(
        [{"run_id": "r", "step": s, "ts": float(s), "grad_norm": 1.0 if s != 40 else 80.0} for s in range(60)]
    )
    lines = ev.incident_summary(evidence).splitlines()
    assert [i["kind"] for i in evidence["incidents"]] == ["grad_spike"]
    assert "3 of 32 kinds checked, 29 not configured" in lines and lines.index("3 of 32 kinds checked, 29 not configured") > 0


def test_the_kinds_by_value_are_names_with_a_reason_for_each_unavailable_one(tmp_path):
    assert ev.kinds_by_value({"incidents": []}) is None
    evidence = tensorquill.build_evidence(probed(fail_at=(30,)), watchdog=tmp_path / "watchdog.json")
    found = ev.kinds_by_value(evidence)
    assert found["checked"] == ["repeated_steps"] and len(found["not_configured"]) == 26
    assert list(found["unavailable"]) == [*ATTENTION, *STALLS]
    assert found["unavailable"]["attention_uniform"].startswith("1 of 6 probes failed")
    assert found["unavailable"]["rank_silent"].startswith("no progress status")


def test_the_command_line_prints_the_count(tmp_path, capsys, monkeypatch):
    path = tmp_path / "run.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in router_only()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["tensorquill", str(path)])
    ev.main()
    printed = capsys.readouterr().out
    assert "No incidents detected.\n3 of 32 kinds checked, 29 not configured\n" in printed
    assert "Run state: healthy" in printed
    written = json.loads(path.with_suffix(".evidence.json").read_text(encoding="utf-8"))
    assert written["kind_checks"]["job_stalled"]["value"] == "not_configured" and written["run_state"]["state"] == "healthy"
