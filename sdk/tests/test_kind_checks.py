"""What was checked: `evidence["kind_checks"]` gives every incident kind one of three values.

- `checked`: records carried the kind's input and its rule ran over them.
- `not_configured`: the capture was never set up to provide it. A router-only capture has no
  weight or activation checks, no `scaler=` means no loss-scale check, and a local file that
  no server was watching has no stall check.
- `unavailable`: the input was asked for and could not be judged (a probe that failed, a file
  that could not be read, a server's store with no progress status).

Only `unavailable` makes the run `degraded`, and the summary carries the count
("3 of 32 kinds checked"). A kind is `checked` from its first record, so each kind also says
whether its rule has had enough history: `records` against `records_needed`, and
`enough_history`. A rule that is still warming up (too few probes among them) does not
degrade the run, and the summary and the listing by value name it apart from the kinds
whose "nothing found" is a verdict.

These tests are the Python side and need nothing but the SDK. test_server_detect.py holds
the Rust report to the same values on every parity case, and runs `BOUNDARY` through Rust.
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
MOE_NEEDS, STATUS_NEEDS = "moe: a router's expert counts", "the server's progress status (watchdog.json)"
# What each of the server's kinds needs, by hand from its rules (`RECORDS_NEEDED` in evidence.py
# lists them). The Rust unit test `what_each_kind_needs_is_these_numbers_...` has the same list.
BY_HAND = {
    "moe_imbalance": 11, "starved_experts": 50, "grad_spike": 13, "stalled_weights": 3, "repeated_steps": 1,
    "loss_scale_trouble": 10, "resume_lr_change": 1, "resume_data_mismatch": 1, "resume_incomplete": 1,
    "data_stalled": 4, "epoch_order_repeated": 4, "activation_spike": 11, "activation_drift": 60,
    "dead_activations": 60, "attention_collapse": 4, "attention_uniform": 4, "attention_logit_growth": 4,
    "non_finite": 1, "replica_drift": 1, "memory_growth": 6, "memory_pressure": 1, "allocator_retries": 1,
    "hardware_errors": 2, "gpu_throttled": 1, "input_starvation": 21,
}


def checked(records: int, kind: str, needs: str, value: str = "checked") -> dict:
    """A server kind's entry with no reason: its value, the count and what the count is held against."""
    return {"value": value, "records": records, "records_needed": BY_HAND[kind],
            "enough_history": records >= BY_HAND[kind], "needs": needs}


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
        assert ("records" in check) == ("records_needed" in check) == (kind in ev.SERVER_KINDS), kind
        assert ("reason" in check) == (check["value"] == "unavailable"), kind
        # Every kind says whether its rule has had enough history: the count against what the
        # rule needs for a kind counted in records, and "its source was read" for the rest.
        assert isinstance(check["enough_history"], bool), kind
        if kind in ev.SERVER_KINDS:
            assert check["records_needed"] == BY_HAND[kind] >= 1, kind
            assert check["enough_history"] == (check["records"] >= check["records_needed"]), kind
        else:
            assert check["enough_history"] == (check["value"] == "checked"), kind


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


def test_what_each_kind_needs_is_the_same_numbers_here_by_hand_and_in_rust():
    assert ev.RECORDS_NEEDED == BY_HAND and tuple(ev.RECORDS_NEEDED) == ev.SERVER_KINDS
    assert ev.records_needed() == BY_HAND
    # The window a run is judged with, not the default one.
    assert ev.records_needed(7) == {**BY_HAND, "starved_experts": 7}
    source = DETECT_RS.read_text(encoding="utf-8")
    # The table names the kinds in order; its numbers are expressions over the rules' constants,
    # which a Rust unit test holds to the list it has by hand. That list is read here.
    table = re.search(r"pub const RECORDS_NEEDED: \[\(&str, u64\); 25\] = \[(.*?)\n\];", source, re.S)
    assert table, "RECORDS_NEEDED is not in server/src/detect.rs in the shape this test reads"
    assert re.findall(r'^    \("([a-z_]+)",', table.group(1), re.M) == list(BY_HAND)
    by_hand = re.search(r"let by_hand: \[u64; 25\] = \[(.*?)\];", source, re.S)
    assert by_hand, "the Rust unit test's list is not in the shape this test reads"
    assert [int(n) for n in re.findall(r"\d+", by_hand.group(1))] == list(BY_HAND.values())


# --- not configured: the capture was never set up to provide it -------------------------------


def test_a_router_only_capture_reads_healthy_and_lists_what_it_cannot_show():
    evidence = tensorquill.build_evidence(router_only())
    found = values(evidence)
    assert all(found[kind] == "not_configured" for kind in WEIGHTS_AND_ACTIVATIONS), found
    assert all(found[kind] == "not_configured" for kind in ("grad_spike", "non_finite", "loss_scale_trouble", "replica_drift"))
    assert [kind for kind, value in found.items() if value == "checked"] == ["moe_imbalance", "starved_experts", "repeated_steps"]
    assert "unavailable" not in found.values()
    assert evidence["incidents"] == [] and evidence["run_state"] == {"state": "healthy", "reasons": []}
    assert evidence["kind_checks"]["moe_imbalance"] == {
        "value": "checked", "records": 60, "records_needed": 11, "enough_history": True, "needs": MOE_NEEDS,
    }
    assert evidence["kind_checks"]["stalled_weights"] == {
        "value": "not_configured", "records": 0, "records_needed": 3, "enough_history": False,
        "needs": "weights_unchanged with grad_norms (full capture)",
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
            **checked(5, kind, "probe with attention (probe=)", "unavailable"), "check": "probe", "reason": entry["reason"],
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


def test_too_few_probes_are_checked_without_enough_history_and_do_not_degrade_the_run():
    evidence = tensorquill.build_evidence(probed(steps=30))  # three probes, four needed
    assert "not_checked" not in evidence and evidence["incidents"] == []
    for kind in ATTENTION:
        assert evidence["kind_checks"][kind] == {
            "value": "checked", "records": 3, "records_needed": 4, "enough_history": False,
            "needs": "probe with attention (probe=)",
        }
    assert evidence["run_state"] == {"state": "healthy", "reasons": []}
    # "Nothing found" is not said for them: the summary and the listing by value set them apart.
    assert ev.incident_summary(evidence).splitlines() == [
        "No incidents detected.",
        "4 of 32 kinds checked, 3 of them without enough history yet (attention_collapse 3 of 4 records,"
        " attention_uniform 3 of 4, attention_logit_growth 3 of 4), 28 not configured",
    ]
    listed = ev.kinds_by_value(evidence)
    assert listed["checked"] == ["repeated_steps"]
    assert listed["not_enough_history"] == dict.fromkeys(ATTENTION, {"records": 3, "records_needed": 4})
    # The fourth probe is enough.
    enough = tensorquill.build_evidence(probed(steps=31))
    assert all(enough["kind_checks"][kind]["enough_history"] for kind in ATTENTION)
    assert ev.kinds_by_value(enough)["not_enough_history"] == {} and enough["run_state"]["state"] == "healthy"
    # Too few and one failed: the failure is the only entry, and it is what makes the kinds unavailable.
    both = tensorquill.build_evidence(probed(steps=30, fail_at=(20,)))
    assert [n["check"] for n in both["not_checked"]] == ["probe"]
    check = both["kind_checks"]["attention_collapse"]
    assert (check["value"], check["check"], check["records"], check["enough_history"]) == ("unavailable", "probe", 2, False)
    assert both["run_state"]["state"] == "degraded"
    # An unavailable kind is not listed as warming up as well: one reason is enough.
    assert ev.kinds_by_value(both)["not_enough_history"] == {} and ev.checked_counts(both)["not_enough_history"] == 0


def test_a_real_capture_probes_on_every_nth_committed_step_so_the_fourth_probe_is_the_first_that_can_report():
    import torch

    from test_probe import Toy, _train

    torch.manual_seed(0)
    model, sink = Toy(), tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, probe=torch.zeros(2, 8, dtype=torch.long), probe_every=3)
    _train(model, session, 12, torch.randint(0, 32, (2, 8)))
    session.close()
    recorded = sorted(sink.records, key=lambda r: r["step"])
    probes = [n for n, r in enumerate(recorded, 1) if "probe" in r]
    assert probes == [3, 6, 9, 12]  # the 3rd, 6th, ... step the session committed, counting from 1
    assert all(r["probe"]["attention"] for r in recorded if "probe" in r), "the toy model's real attention"

    # As the capture recorded it: three probes are too few, and the run is not degraded for it.
    for steps, seen in ((2, 0), (3, 1), (8, 2), (11, 3), (12, 4)):
        evidence = tensorquill.build_evidence(recorded[:steps])
        for kind in ATTENTION:
            check = evidence["kind_checks"][kind]
            assert (check["records"], check["records_needed"], check["enough_history"]) == (seen, 4, seen >= 4), steps
            assert check["value"] == ("checked" if seen else "not_configured"), steps
        assert "not_checked" not in evidence and evidence["run_state"]["state"] == "healthy", steps

    # On the same schedule, attention that collapses on the third probe: it cannot be reported
    # before the fourth. With `probe_every` 1,000 that is the 4,000th step of a run's first
    # attempt, and the incident is dated at the third probe, the 3,000th.
    def collapsing(steps):
        out = []
        for count, record in enumerate(recorded[:steps], 1):
            if "probe" in record:
                share = 0.6 if count <= 6 else 0.1
                record = {**record, "probe": {"attention": {"blocks.0": {"entropy_share": [share], "max_logit": [2.0]}}}}
            out.append(record)
        return tensorquill.build_evidence(out)

    assert not [i for i in collapsing(11)["incidents"] if i["kind"] in ATTENTION]
    assert collapsing(11)["kind_checks"]["attention_collapse"]["enough_history"] is False
    evidence = collapsing(12)
    found = [i for i in evidence["incidents"] if i["kind"] == "attention_collapse"]
    assert len(found) == 1 and evidence["kind_checks"]["attention_collapse"]["enough_history"] is True
    assert (found[0]["step"], found[0]["confirmed_step"]) == (recorded[8]["step"], recorded[11]["step"])


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
    assert checks["resume_lr_change"] == {
        **checked(9, "resume_lr_change", "lr", "unavailable"), "check": "resume_lr", "reason": "no learning rate",
    }
    assert checks["hardware_errors"]["reason"] == "not supported"  # the first entry that names it
    assert checks["gpu_throttled"] == checked(4, "gpu_throttled", "system.nvml (system=True)")
    # Whether there is enough history is the count against the need, whatever the value.
    assert checks["hardware_errors"]["enough_history"] is True and checks["memory_growth"]["enough_history"] is False
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
        assert all(evidence["kind_checks"][kind] == {"value": "not_configured", "enough_history": False, "needs": STATUS_NEEDS}
                   for kind in STALLS)
        assert "no_status" not in codes(evidence) and evidence["run_state"]["state"] == "healthy"


def test_a_status_that_was_named_and_cannot_be_read_is_unavailable_and_degrades(tmp_path):
    evidence = tensorquill.build_evidence(router_only(), watchdog=tmp_path / "watchdog.json")
    for kind in STALLS:
        assert evidence["kind_checks"][kind] == {
            "value": "unavailable", "enough_history": False, "needs": STATUS_NEEDS, "check": "status",
            "reason": "no progress status (the server's watchdog.json): whether the job is still making progress is not checked",
        }
    assert evidence["run_state"]["state"] == "degraded" and codes(evidence) == ["no_status"]
    assert ev.incident_summary(evidence).splitlines()[1] == "3 of 32 kinds checked, 27 not configured, 2 unavailable (job_stalled, rank_silent)"

    read = tensorquill.build_evidence(router_only(), watchdog=ACTIVE)
    # A status that was read is whole: the server waited out its own thresholds before writing it.
    assert all(read["kind_checks"][kind] == {"value": "checked", "enough_history": True, "needs": STATUS_NEEDS} for kind in STALLS)
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
    assert evidence["kind_checks"]["traincheck_violation"] == {
        "value": "checked", "enough_history": True, "needs": "TrainCheck results (traincheck=)",
    }
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
        "not_enough_history": 0, "not_enough_history_kinds": [], "history_not_stated": 0,
    }


# --- the count ----------------------------------------------------------------------------------


def test_the_count_is_out_of_every_kind_and_evidence_without_the_key_has_no_line():
    assert ev.checked_counts({}) is None and ev.checked_line({"incidents": []}) is None
    assert "kinds checked" not in ev.incident_summary({"incidents": [], "samples": []})
    checks = {"grad_spike": {"value": "checked", "enough_history": True}, "non_finite": {"value": "unavailable"},
              "not_a_kind": {"value": "checked"}}
    assert ev.checked_counts({"kind_checks": checks}) == {
        "kinds": 32, "checked": 1, "not_configured": 0, "unavailable": 1, "unavailable_kinds": ["non_finite"],
        "not_enough_history": 0, "not_enough_history_kinds": [], "history_not_stated": 0,
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
    assert found["not_enough_history"] == {} and "history_not_stated" not in found
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


# --- enough history: each kind's slowest rule fires on exactly the records the table gives ------
#
# One record set for every kind that needs more than one record. Each holds exactly
# `records_needed` records carrying the kind's input, and its slowest rule reports on the last
# of them. Without its first record (one record of history less) the same fault is not
# reported. test_server_detect.py runs the same sets through the Rust detectors.

GIB = 1024 ** 3


def _step(step: int, **values) -> dict:
    return {"schema_version": 1, "run_id": "boundary", "rank": 0, "attempt": 1, "step": step, "ts": float(step), **values}


def _moe(counts: list[int], top_k: int) -> dict:
    return {"moe": {"router": counts}, "moe_top_k": {"router": top_k}}


def _layer(std: float = 1.0, abs_max: float = 2.0, zero_frac: float = 0.0) -> dict:
    return {"layers": {"blocks.0": {"mean": 0.0, "std": std, "abs_max": abs_max, "zero_frac": zero_frac}}}


def _probe(share: float = 0.6, logit: float = 2.0) -> dict:
    return {"probe": {"attention": {"blocks.0.attn": {"entropy_share": [share, share], "max_logit": [logit, logit]}}}}


def _batch(name: str, epoch: int = 0) -> dict:
    return {"data": {"batches": 1, "batch_hash": name * 16, "epoch": epoch, "sampler_epoch": 0, "shuffle": True}}


def _each(values: list[dict]) -> list[dict]:
    return [_step(step, **value) for step, value in enumerate(values)]


def boundary() -> dict[str, tuple[str, list[dict], dict]]:
    """name -> (kind, the records, what the incident must hold: the rule that reported)."""
    wait = [0.0] * 11 + [0.5] * 10  # the median share is 0 over the first 20 steps and 0.25 over the last 20
    return {
        # Top-4 routing: ten even samples, then four experts take nearly everything and none is empty,
        # so only the diversity drop against the router's own history can flag it.
        "moe_imbalance, the diversity drop": (
            "moe_imbalance", _each([_moe([100] * 8, 4)] * 10 + [_moe([195] * 4 + [5] * 4, 4)]), {"overloaded": [1, 2, 3, 4]},
        ),
        # One expert gets a token a step: far under a tenth of an even share over the window, and never empty.
        "starved_experts, a full window": ("starved_experts", _each([_moe([1000] * 7 + [1], 1)] * 50), {}),
        # A small parameter beside a large one: the total does not move, the parameter stays at 50 times its own norm.
        "grad_spike, one parameter": (
            "grad_spike",
            _each([{"grad_norm": 1000.0, "grad_norms": {"big": 1000.0, "small": 1.0}}] * 10
                  + [{"grad_norm": 1001.25, "grad_norms": {"big": 1000.0, "small": 50.0}}] * 3),
            {"trigger": "parameter", "sustained": 3},
        ),
        "stalled_weights": (
            "stalled_weights", _each([{"grad_norm": 1.0, "grad_norms": {"w": 1.0}, "weights_unchanged": ["w"]}] * 3), {"consecutive": 3},
        ),
        "loss_scale_trouble, skips in a row": (
            "loss_scale_trouble", _each([{"step_skipped": True, "grad_scale": 65536.0}] * 10), {"reason": "skips"},
        ),
        "data_stalled, the same batch": ("data_stalled", _each([_batch("a")] * 4), {"reason": "same_batch"}),
        "epoch_order_repeated": (
            "epoch_order_repeated", _each([_batch("a"), _batch("b"), _batch("a", 1), _batch("b", 1)]), {"batches_compared": 2},
        ),
        "activation_spike": ("activation_spike", _each([_layer()] * 10 + [_layer(abs_max=20.0)]), {}),
        # 3.5 times the baseline: under the spike rule's 5, over the drift rule's 3.
        "activation_drift": ("activation_drift", _each([_layer()] * 50 + [_layer(std=3.5, abs_max=7.0)] * 10), {}),
        "dead_activations": ("dead_activations", _each([_layer()] * 50 + [_layer(zero_frac=0.5)] * 10), {}),
        "attention_collapse": ("attention_collapse", _each([_probe()] * 2 + [_probe(share=0.2)] * 2), {}),
        "attention_uniform": ("attention_uniform", _each([_probe()] * 2 + [_probe(share=0.99)] * 2), {}),
        "attention_logit_growth": ("attention_logit_growth", _each([_probe()] * 2 + [_probe(logit=5.0)] * 2), {}),
        "memory_growth": (
            "memory_growth",
            _each([{"system": {"cuda": {"device": 0, "allocated": GIB + n * 100 * 1024 ** 2, "reserved": 4 * GIB,
                                         "total": 16 * GIB, "alloc_retries": 0, "ooms": 0}}} for n in range(6)]),
            {"readings": 6},
        ),
        "hardware_errors, an ECC count that rose": (
            "hardware_errors",
            _each([{"system": {"nvml": {"ecc_uncorrected": count, "xid": [], "throttle": []}}} for count in (3, 4)]),
            {"xid": [], "ecc_uncorrected_at_start": 3},
        ),
        "input_starvation, the rise": (
            "input_starvation", _each([{"data_wait": share, "step_time": 1.0} for share in wait]), {"rule": "rise"},
        ),
    }


BOUNDARY = boundary()


def test_the_boundary_sets_cover_every_kind_that_needs_more_than_one_record():
    assert {kind for kind, _, _ in BOUNDARY.values()} == {kind for kind, needed in BY_HAND.items() if needed > 1}
    assert sorted({kind for kind, needed in BY_HAND.items() if needed == 1}) == sorted([
        "repeated_steps", "resume_lr_change", "resume_data_mismatch", "resume_incomplete", "non_finite", "replica_drift",
        "memory_pressure", "allocator_retries", "gpu_throttled",
    ])


def boundary_findings(name: str, build=tensorquill.build_evidence) -> None:
    """The slowest rule of the kind reports on exactly `records_needed` records and not on one
    fewer, and `enough_history` turns true with that record. `build` makes the evidence."""
    kind, records, holds = BOUNDARY[name]
    evidence = build(records)
    check = evidence["kind_checks"][kind]
    assert (check["value"], check["records"], check["records_needed"]) == ("checked", len(records), BY_HAND[kind]), name
    assert check["records"] == check["records_needed"] and check["enough_history"] is True, name
    found = [i for i in evidence["incidents"] if i["kind"] == kind]
    assert len(found) == 1, (name, [i["kind"] for i in evidence["incidents"]])
    assert {key: found[0][key] for key in holds} == holds, name
    # It is established on the last record, not before.
    last = found[0].get("confirmed_step", found[0].get("through_step", found[0]["step"]))
    if kind not in ("grad_spike", "epoch_order_repeated"):  # these are dated at the start of what lasted
        assert last == records[-1]["step"], name
    for fewer in (records[1:], records[:-1]):  # one record of history less, and the last record not yet written
        evidence = build(fewer)
        check = evidence["kind_checks"][kind]
        assert (check["value"], check["records"], check["enough_history"]) == ("checked", BY_HAND[kind] - 1, False), name
        assert not [i for i in evidence["incidents"] if i["kind"] == kind], name
        # Warming up is not a fault of the monitoring.
        assert "not_checked" not in evidence, name
        assert kind in ev.kinds_by_value(evidence)["not_enough_history"] and kind not in ev.kinds_by_value(evidence)["checked"]
        assert f"{kind} {BY_HAND[kind] - 1} of {BY_HAND[kind]}" in ev.checked_line(evidence), name


@pytest.mark.parametrize("name", BOUNDARY)
def test_a_kinds_slowest_rule_reports_on_exactly_the_records_it_needs(name):
    boundary_findings(name)


def test_a_kind_can_report_through_a_faster_rule_before_it_has_enough_history():
    # A loss scale below 1 is reported on the first record; the run of skips needs ten.
    evidence = tensorquill.build_evidence(_each([{"step_skipped": False, "grad_scale": 0.5}]))
    assert [(i["kind"], i["reason"]) for i in evidence["incidents"]] == [("loss_scale_trouble", "scale")]
    check = evidence["kind_checks"]["loss_scale_trouble"]
    assert (check["records"], check["records_needed"], check["enough_history"]) == (1, 10, False)
    assert evidence["run_state"]["state"] == "unhealthy"
    # The incident stands, and the line and the listing say so beside the count.
    assert ev.checked_line(evidence) == (
        "2 of 32 kinds checked, 1 of them without enough history yet"
        " (loss_scale_trouble 1 of 10 records but reported by a faster rule), 30 not configured"
    )
    assert ev.kinds_by_value(evidence)["not_enough_history"] == {
        "loss_scale_trouble": {"records": 1, "records_needed": 10, "reported": True},
    }
    # The total's gradient spike needs eleven records, the kind thirteen.
    spiked = tensorquill.build_evidence(_each([{"grad_norm": 1.0}] * 10 + [{"grad_norm": 80.0}]))
    assert [i["kind"] for i in spiked["incidents"]] == ["grad_spike"] and "trigger" not in spiked["incidents"][0]
    assert spiked["kind_checks"]["grad_spike"]["enough_history"] is False
    assert "(grad_spike 11 of 13 records but reported by a faster rule)" in ev.checked_line(spiked)


def test_starved_experts_needs_the_window_the_run_was_judged_with():
    records = _each([_moe([1000] * 7 + [1], 1)] * 12)
    usual = tensorquill.build_evidence(records)["kind_checks"]
    assert (usual["starved_experts"]["records_needed"], usual["starved_experts"]["enough_history"]) == (50, False)
    assert usual["moe_imbalance"]["enough_history"] is True  # 12 of 11
    for window, enough in ((12, True), (13, False)):
        evidence = tensorquill.build_evidence(records, starve_window=window)
        check = evidence["kind_checks"]["starved_experts"]
        assert (check["records"], check["records_needed"], check["enough_history"]) == (12, window, enough)
        assert bool([i for i in evidence["incidents"] if i["kind"] == "starved_experts"]) == enough
    assert ev.kind_checks({"moe": 12}, [], starve_window=12)["starved_experts"]["enough_history"] is True


def test_a_file_from_before_the_field_is_never_read_as_enough_history(tmp_path):
    evidence = tensorquill.build_evidence(router_only())
    old = {**evidence, "kind_checks": {
        kind: {key: value for key, value in check.items() if key not in ("records_needed", "enough_history")}
        for kind, check in evidence["kind_checks"].items()
    }}
    assert ev.kind_history(evidence["kind_checks"]["moe_imbalance"]) == "enough"
    assert ev.kind_history(old["kind_checks"]["moe_imbalance"]) == "not_stated"
    assert ev.kind_history(old["kind_checks"]["grad_spike"]) is None and ev.kind_history("no entry") is None
    listed = ev.kinds_by_value(old)
    assert listed["checked"] == [] and listed["not_enough_history"] == {}
    assert listed["history_not_stated"] == ["moe_imbalance", "starved_experts", "repeated_steps"]
    assert ev.checked_counts(old)["history_not_stated"] == 3 and ev.checked_counts(old)["not_enough_history"] == 0
    assert ev.checked_line(old) == (
        "3 of 32 kinds checked, 3 of them with no word on their history (an older version wrote this), 29 not configured"
    )
    # A current file with every checked kind past its warm-up says nothing more than the count.
    assert ev.checked_line(evidence) == "3 of 32 kinds checked, 29 not configured"


# --- The all-ranks view: what was checked for the whole run ---------------------------------
#
# `--rank all`, `all_ranks_evidence` and `get_all_ranks_status` read each rank's report and run
# no detector, so the run's `kind_checks` is made from the ranks' own. The rule these tests hold
# (`run_kind_checks` in evidence.py):
#   - only judged ranks count, and a rank that is not judged is named and is in no value;
#   - a kind's value is its worst judged rank's, so `checked` means every judged rank, and the
#     ranks are listed by value wherever they differ;
#   - history the same way: short on one rank is short for the run, with that rank named;
#   - a judged rank whose report does not say what was checked means no count at all;
#   - `unavailable` on one rank is `unavailable` for the run, and the run is `degraded`.
# The stores are written by hand, with each rank's `kind_checks` as the Python table gives it
# (test_server_detect.py holds the server's to the same values); test_watch_all_ranks.py has
# the same cases through the real server.

HAND_RUN = "handrun"
WEIGHTS = {"weights_unchanged": [], "grad_norms": {"w": 1.0}}
CUDA = {"system": {"cuda": {"allocated": 1 << 30, "reserved": 2 << 30, "peak_allocated": 3 << 30, "total": 16 << 30}}}
PROBE_FAILED = {"check": "probe", "kinds": list(ATTENTION), "reason": "1 of 3 probes failed, the first at step 20: boom"}
FINISHED = "finished"


def steps(count: int, **extra) -> list[dict]:
    """Step records with a gradient norm and a learning rate; `extra` goes on every one (a
    callable is given the step)."""
    return [{"schema_version": 1, "run_id": HAND_RUN, "rank": 0, "attempt": 1, "step": step, "ts": float(step),
             "grad_norm": 1.0, "lr": [1e-3], **{k: v(step) if callable(v) else v for k, v in extra.items()}}
            for step in range(count)]


def hand_report(rank: int, records: list[dict], not_checked: list[dict] = (), incidents: list[dict] = ()) -> dict:
    """A rank's `incidents.json` as the server writes it for these records, by the Python tables."""
    thresholds = {name.lower(): value for name, value in vars(ev).items()
                  if name.isupper() and isinstance(value, (int, float)) and not isinstance(value, bool)}
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": HAND_RUN, "rank": rank, "attempt": 1,
        "last_step": records[-1]["step"], "steps": len(records), "rebuilds": 0, "checked": list(ev.SERVER_KINDS),
        "thresholds": thresholds, "moe_flagged": {"samples": 0, "last_step": None}, "incidents": list(incidents),
        "not_checked": list(not_checked), "memory": ev.latest_memory(records),
        "kind_checks": ev.kind_checks(ev.input_counts(records), not_checked),
    }


def hand_store(tmp_path: Path, ranks: dict[int, list[dict]], *, status: str | None = FINISHED,
               not_checked: dict[int, list[dict]] | None = None, incidents: dict[int, list[dict]] | None = None) -> Path:
    """A run folder as `tensorquill-server` leaves it: for each rank a current report and one
    part (only its name is read), and the run's status unless `status` is None."""
    folder = tmp_path / "store" / "runs" / HAND_RUN
    for rank, records in ranks.items():
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / f"part-0-{records[-1]['step']}.parquet").write_bytes(b"")
        report = hand_report(rank, records, (not_checked or {}).get(rank, ()), (incidents or {}).get(rank, ()))
        (folder / f"rank={rank}" / "incidents.json").write_text(json.dumps(report), encoding="utf-8")
        look = ev.read_server_report(folder / f"rank={rank}")
        assert look["state"] == "current", f"update `hand_report`: read_server_report now says {look['reason']}"
    if status is not None:
        (folder / "watchdog.json").write_text(json.dumps({
            "schema_version": 1, "summary": status, "world_size": len(ranks),
            "ranks": [{"rank": rank, "state": status, "last_step": records[-1]["step"]} for rank, records in ranks.items()],
        }), encoding="utf-8")
    return folder


def change_report(folder: Path, rank: int, change) -> None:
    path = folder / f"rank={rank}" / "incidents.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    change(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def fall_behind(folder: Path, rank: int) -> None:
    """A part the rank's report does not cover: the rank is `behind`, so not judged."""
    [part] = (folder / f"rank={rank}" / "attempt=1").glob("part-*.parquet")
    last = int(part.stem.split("-")[2])
    (part.parent / f"part-{last + 1}-{last + 20}.parquet").write_bytes(b"")


def three_ways(folder: Path, tmp_path: Path, **options) -> tuple[dict, str, dict]:
    """The view as the public look returns it, held equal to the file the watcher writes, with
    its summary and `get_all_ranks_status`'s answer for the same folder."""
    from tensorquill import mcp_server

    ours = json.loads(json.dumps(ev.all_ranks_evidence(folder, **options)))
    out = tmp_path / f"all-{len(list(tmp_path.glob('all-*.json')))}.json"
    assert ev.watch(folder, out, rank="all", polls=1, interval=0.0, **options) == out
    theirs = json.loads(out.read_text(encoding="utf-8"))
    assert theirs.pop("written_at_ms") >= ours.pop("written_at_ms")
    assert theirs == ours, "the watcher's file and the look differ"
    status = mcp_server.get_all_ranks_status(mcp_server.Source(out), {})
    if not options:  # the tool on the folder makes its own look, with no options
        from_folder = mcp_server.get_all_ranks_status(mcp_server.Source(folder), {})
        assert {k: v for k, v in status.items() if k != "source"} == {k: v for k, v in from_folder.items() if k != "source"}
    assert status.get("summary") == ev.incident_summary(ours)
    return ours, ev.incident_summary(ours), status


def count_line(summary: str) -> str:
    [line] = [line for line in summary.splitlines() if " kinds checked" in line]
    return line


def test_a_kind_checked_on_one_rank_only_is_not_checked_for_the_run_and_names_the_ranks(tmp_path):
    folder = hand_store(tmp_path, {0: steps(60, **WEIGHTS), 1: steps(60)})
    evidence, summary, status = three_ways(folder, tmp_path)
    checks = evidence["kind_checks"]
    assert list(checks) == list(ev.INCIDENT_CLASSES)
    assert all(check["value"] in ev.KIND_CHECK_VALUES for check in checks.values())
    # Rank 0 records weights and rank 1 does not: not the best rank's value, and both are named.
    assert checks["stalled_weights"] == {
        "value": "not_configured", "enough_history": False, "records": 0, "records_needed": 3,
        "needs": "weights_unchanged with grad_norms (full capture)", "ranks": {"checked": [0], "not_configured": [1]},
    }
    # What both ranks record is checked, with the fewest records any rank has, and no rank list.
    assert checks["grad_spike"] == checked(60, "grad_spike", "grad_norm or clip_grad_norm")
    assert "ranks" not in checks["moe_imbalance"] and checks["moe_imbalance"]["value"] == "not_configured"
    assert checks["job_stalled"] == {"value": "checked", "enough_history": True, "needs": STATUS_NEEDS}
    counts = ev.checked_counts(evidence)
    assert counts["checked"] + counts["not_configured"] + counts["unavailable"] == counts["kinds"] == 32
    assert (counts["checked"], counts["unavailable"]) == (6, 0)
    assert summary.splitlines()[-1] == count_line(summary) == (
        "6 of 32 kinds checked on both ranks, 26 not configured, 1 of them checked on some ranks only"
        " (stalled_weights on rank 0)"
    )
    # Per rank: each rank's own count of the server's 25 kinds shows who differs.
    per_rank = {entry["rank"]: entry["kind_counts"] for entry in evidence["ranks"]}
    assert (per_rank[0]["kinds"], per_rank[0]["checked"], per_rank[1]["checked"]) == (25, 5, 4)
    assert evidence["run_state"]["state"] == "healthy"  # a kind that is not configured degrades nothing
    kinds = status["kinds"]
    assert "stalled_weights" in kinds["not_configured"] and "stalled_weights" not in kinds["checked"]
    assert kinds["differs_by_rank"] == {"stalled_weights": {"checked": "0", "not_configured": "1"}}
    assert (kinds["judged_ranks"], kinds["ranks_not_counted"]) == ("0-1", "")
    assert [row["kinds"]["checked"] for row in status["per_rank"]] == [5, 4]
    assert "kinds_not_stated" not in status


def test_a_rank_that_is_not_judged_is_named_and_is_in_no_value(tmp_path):
    # Rank 1 records weights, and its report is behind its store: what it says does not count.
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(60, **WEIGHTS)})
    fall_behind(folder, 1)
    evidence, summary, status = three_ways(folder, tmp_path)
    assert (evidence["judged_ranks"], evidence["unknown_ranks"]) == ([0], [1])
    checks = evidence["kind_checks"]
    assert checks["stalled_weights"]["value"] == "not_configured" and "ranks" not in checks["stalled_weights"]
    assert "kind_counts" in evidence["ranks"][0] and "kind_counts" not in evidence["ranks"][1]
    for kind, check in checks.items():
        named = [rank for ranks in (check.get("ranks") or {}).values() for rank in ranks]
        named += [*check.get("not_enough_history_ranks", ()), *check.get("history_not_stated_ranks", ())]
        assert 1 not in named, f"{kind} counts rank 1, which is not judged"
    assert count_line(summary) == (
        "6 of 32 kinds checked on the 1 judged rank of 2 (rank 1 not judged, so not counted), 26 not configured"
    )
    assert (status["kinds"]["judged_ranks"], status["kinds"]["ranks_not_counted"]) == ("0", "1")
    assert evidence["run_state"]["state"] == "unknown"

    # The other way round: the judged rank records weights, so the kind is checked, for that rank.
    other = hand_store(tmp_path / "other", {0: steps(60, **WEIGHTS), 1: steps(60)})
    fall_behind(other, 1)
    evidence, summary, _ = three_ways(other, tmp_path)
    assert evidence["kind_checks"]["stalled_weights"]["value"] == "checked"
    assert count_line(summary).startswith("7 of 32 kinds checked on the 1 judged rank of 2 (rank 1 not judged")


def test_no_judged_rank_means_no_count(tmp_path):
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(60)})
    fall_behind(folder, 0)
    fall_behind(folder, 1)
    evidence, summary, status = three_ways(folder, tmp_path)
    assert "kind_checks" not in evidence and evidence["kind_checks_not_stated"]["code"] == "no_judged_rank"
    assert " kinds checked" not in summary and "kinds" not in status
    assert status["kinds_not_stated"]["code"] == "no_judged_rank"
    assert ev.checked_counts(evidence) is None and ev.all_ranks_kinds(evidence) is None


def test_a_rank_short_of_history_makes_the_kind_short_for_the_run_and_is_named(tmp_path):
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(5)})
    evidence, summary, status = three_ways(folder, tmp_path)
    checks = evidence["kind_checks"]
    assert checks["grad_spike"] == {
        "value": "checked", "enough_history": False, "records": 5, "records_needed": 13,
        "needs": "grad_norm or clip_grad_norm", "not_enough_history_ranks": [1],
    }
    assert ev.kind_history(checks["grad_spike"]) == "not_enough"
    assert ev.kind_history(checks["non_finite"]) == "enough" and checks["non_finite"]["records"] == 5
    assert count_line(summary) == (
        "6 of 32 kinds checked on both ranks, 1 of them without enough history yet (grad_spike 5 of 13 records on"
        " rank 1), 26 not configured"
    )
    kinds = status["kinds"]
    assert kinds["not_enough_history"] == {"grad_spike": {"records": 5, "records_needed": 13, "ranks": "1"}}
    assert "grad_spike" not in kinds["checked"] and evidence["run_state"]["state"] == "healthy"

    # Short on every rank: no rank differs, so none is named.
    both = hand_store(tmp_path / "both", {0: steps(5), 1: steps(7)})
    evidence, summary, status = three_ways(both, tmp_path)
    assert evidence["kind_checks"]["grad_spike"]["not_enough_history_ranks"] == [0, 1]
    assert "(grad_spike 5 of 13 records)" in count_line(summary)
    assert status["kinds"]["not_enough_history"] == {"grad_spike": {"records": 5, "records_needed": 13}}


def test_a_kind_checked_on_some_ranks_only_says_when_those_ranks_are_still_warming_up(tmp_path):
    # Ranks 0 and 1 record weights, rank 0 on two steps only (the rule needs three); rank 2 records none.
    twice = steps(60, weights_unchanged=lambda step: [] if step < 2 else None, grad_norms={"w": 1.0})
    folder = hand_store(tmp_path, {0: twice, 1: steps(60, **WEIGHTS), 2: steps(60)})
    evidence, summary, status = three_ways(folder, tmp_path)
    check = evidence["kind_checks"]["stalled_weights"]
    assert (check["value"], check["ranks"]) == ("not_configured", {"checked": [0, 1], "not_configured": [2]})
    assert check["not_enough_history_ranks"] == [0]
    assert "(stalled_weights on ranks 0-1 without enough history yet on rank 0)" in count_line(summary)
    assert status["kinds"]["differs_by_rank"]["stalled_weights"] == {
        "checked": "0-1", "not_configured": "2", "not_enough_history": "0"}
    # Short on every rank that checks it: no rank among them differs, so none is named twice.
    both = hand_store(tmp_path / "both", {0: twice, 1: steps(60)})
    _, summary, status = three_ways(both, tmp_path)
    assert "(stalled_weights on rank 0 without enough history yet)" in count_line(summary)
    assert status["kinds"]["differs_by_rank"]["stalled_weights"]["not_enough_history"] == "0"

    def strip(report):
        for item in report["kind_checks"].values():
            del item["enough_history"]

    older = hand_store(tmp_path / "older", {0: steps(60, **WEIGHTS), 1: steps(60)})
    change_report(older, 0, strip)
    _, summary, status = three_ways(older, tmp_path)
    assert "(stalled_weights on rank 0 with no word on its history)" in count_line(summary)
    assert status["kinds"]["differs_by_rank"]["stalled_weights"]["history_not_stated"] == "0"


def test_an_incident_on_a_rank_with_enough_history_is_not_called_a_faster_rules(tmp_path):
    spike = {"kind": "grad_spike", "step": 40, "grad_norm": 9.0, "baseline": 1.0, "ratio": 9.0, "parameters": [],
             "spike_count": 1, "spike_steps": [40]}
    early = {**spike, "step": 3, "spike_steps": [3]}
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(5)}, incidents={0: [spike]})
    _, summary, status = three_ways(folder, tmp_path)
    assert "faster rule" not in summary and "reported" not in status["kinds"]["not_enough_history"]["grad_spike"]
    short = hand_store(tmp_path / "short", {0: steps(60), 1: steps(5)}, incidents={1: [early]})
    _, summary, status = three_ways(short, tmp_path)
    assert "grad_spike 5 of 13 records on rank 1 but reported by a faster rule" in count_line(summary)
    assert status["kinds"]["not_enough_history"]["grad_spike"]["reported"] is True
    # A kind the run does not count as checked, reported on the rank that does check it.
    frozen = steps(60, weights_unchanged=["w"], grad_norms={"w": 1.0})
    [stalled] = [i for i in ev.build_evidence(frozen)["incidents"] if i["kind"] == "stalled_weights"]
    partly = hand_store(tmp_path / "partly", {0: frozen, 1: steps(60)}, incidents={0: [stalled]})
    _, summary, _ = three_ways(partly, tmp_path)
    assert "checked on some ranks only (stalled_weights on rank 0, where it is reported)" in count_line(summary)


def test_unavailable_on_one_rank_is_unavailable_for_the_run_and_the_run_is_degraded(tmp_path):
    probed = {"probe": {"attention": {"blk": {"entropy": 1.0, "max_entropy": 2.0, "max_logit": 1.0}}}}
    folder = hand_store(tmp_path, {0: steps(60, **probed), 1: steps(60, **probed)}, not_checked={1: [PROBE_FAILED]})
    evidence, summary, status = three_ways(folder, tmp_path)
    for kind in ATTENTION:
        check = evidence["kind_checks"][kind]
        assert (check["value"], check["check"], check["reason"]) == ("unavailable", "probe", PROBE_FAILED["reason"])
        assert check["ranks"] == {"checked": [0], "unavailable": [1]}
    assert count_line(summary).endswith(
        "3 unavailable (attention_collapse on rank 1, attention_uniform on rank 1, attention_logit_growth on rank 1)")
    assert evidence["run_state"]["state"] == "degraded"
    assert [r["code"] for r in evidence["run_state"]["reasons"] if r["state"] == "degraded"] == ["not_checked"]
    assert status["kinds"]["unavailable"] == dict.fromkeys(ATTENTION, PROBE_FAILED["reason"])
    assert status["kinds"]["differs_by_rank"]["attention_collapse"] == {"checked": "0", "unavailable": "1"}
    # Without the failed probe the same store is healthy and nothing is unavailable.
    clean = hand_store(tmp_path / "clean", {0: steps(60, **probed), 1: steps(60, **probed)})
    evidence, _, _ = three_ways(clean, tmp_path)
    assert ev.checked_counts(evidence)["unavailable"] == 0 and evidence["run_state"]["state"] == "healthy"
    # Unavailable is worse than not configured: a rank that never probed does not soften it.
    mixed = hand_store(tmp_path / "mixed", {0: steps(60), 1: steps(60, **probed)}, not_checked={1: [PROBE_FAILED]})
    evidence, _, _ = three_ways(mixed, tmp_path)
    assert evidence["kind_checks"]["attention_uniform"]["value"] == "unavailable"
    assert evidence["kind_checks"]["attention_uniform"]["ranks"] == {"not_configured": [0], "unavailable": [1]}


def test_a_report_that_does_not_say_what_was_checked_gives_no_count(tmp_path):
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(60)})
    change_report(folder, 1, lambda report: report.pop("kind_checks"))
    assert ev.read_server_report(folder / "rank=1")["state"] == "current"  # an older server's report is still read
    evidence, summary, status = three_ways(folder, tmp_path)
    assert "kind_checks" not in evidence and evidence["judged_ranks"] == [0, 1]
    missing = evidence["kind_checks_not_stated"]
    assert (missing["code"], missing["ranks"]) == ("report_without_kind_checks", [1])
    assert " kinds checked" not in summary
    assert summary.splitlines()[-1].startswith("What was checked is not stated: the report of rank 1 does not say")
    assert "kinds" not in status and status["kinds_not_stated"] == missing
    # The rank that does say keeps its own count; the one that does not has none.
    assert "kind_counts" in evidence["ranks"][0] and "kind_counts" not in evidence["ranks"][1]
    # A report that says it for only some of the server's kinds is not taken either.
    change_report(folder, 0, lambda report: report["kind_checks"].pop("grad_spike"))
    evidence, _, _ = three_ways(folder, tmp_path)
    assert evidence["kind_checks_not_stated"]["ranks"] == [0, 1]


def test_a_report_from_before_enough_history_is_never_read_as_enough(tmp_path):
    def strip(report):
        for check in report["kind_checks"].values():
            del check["enough_history"]

    folder = hand_store(tmp_path, {0: steps(60), 1: steps(60)})
    change_report(folder, 1, strip)
    evidence, summary, status = three_ways(folder, tmp_path)
    checks = evidence["kind_checks"]
    assert "enough_history" not in checks["grad_spike"] and checks["grad_spike"]["history_not_stated_ranks"] == [1]
    assert ev.kind_history(checks["grad_spike"]) == "not_stated"
    counts = ev.checked_counts(evidence)
    assert (counts["checked"], counts["history_not_stated"], counts["not_enough_history"]) == (6, 4, 0)
    assert "4 of them with no word on their history on some rank" in count_line(summary)
    # Only the run's own two (the stall kinds, from the status) are listed as checked.
    assert status["kinds"]["checked"] == list(STALLS)
    assert status["kinds"]["history_not_stated"] == ["grad_spike", "repeated_steps", "resume_lr_change", "non_finite"]
    # A rank that says "not enough" decides it, whatever the silent rank would have said.
    short = hand_store(tmp_path / "short", {0: steps(5), 1: steps(60)})
    change_report(short, 1, strip)
    evidence, _, _ = three_ways(short, tmp_path)
    assert evidence["kind_checks"]["grad_spike"]["enough_history"] is False
    assert evidence["kind_checks"]["grad_spike"]["not_enough_history_ranks"] == [0]


def test_the_stall_kinds_are_the_runs_and_a_store_without_its_status_is_degraded(tmp_path):
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(60)}, status=None)
    evidence, summary, _ = three_ways(folder, tmp_path)
    for kind in STALLS:
        check = evidence["kind_checks"][kind]
        assert (check["value"], check["check"], check["enough_history"]) == ("unavailable", "status", False)
        assert "ranks" not in check
    assert count_line(summary) == "4 of 32 kinds checked on both ranks, 26 not configured, 2 unavailable (job_stalled, rank_silent)"
    assert evidence["run_state"]["state"] == "degraded"
    assert [r["code"] for r in evidence["run_state"]["reasons"] if r["state"] == "degraded"] == ["no_status"]


def test_the_file_kinds_are_the_runs(tmp_path):
    folder = hand_store(tmp_path, {0: steps(60), 1: steps(60)})
    evidence, _, _ = three_ways(folder, tmp_path, traincheck=tmp_path / "no-such-results.json")
    check = evidence["kind_checks"]["traincheck_violation"]
    assert (check["value"], check["check"]) == ("unavailable", "traincheck") and evidence["run_state"]["state"] == "degraded"
    assert evidence["kind_checks"]["collective_hang"]["value"] == "not_configured"


def test_the_memory_comparison_is_checked_only_when_every_judged_rank_is_in_it(tmp_path):
    needs = "system.cuda (system=True) on every judged rank, and at least two ranks"
    both = hand_store(tmp_path / "both", {0: steps(60, **CUDA), 1: steps(60, **CUDA)})
    evidence, _, _ = three_ways(both, tmp_path)
    assert evidence["kind_checks"]["memory_imbalance"] == {"value": "checked", "enough_history": True, "needs": needs}
    one = hand_store(tmp_path / "one", {0: steps(60, **CUDA), 1: steps(60, **CUDA), 2: steps(60)})
    evidence, summary, status = three_ways(one, tmp_path)
    assert evidence["kind_checks"]["memory_imbalance"] == {
        "value": "not_configured", "enough_history": False, "needs": needs, "ranks": {"checked": [0, 1], "not_configured": [2]},
    }
    assert "memory_imbalance on ranks 0-1" in count_line(summary)
    assert status["kinds"]["differs_by_rank"]["memory_imbalance"] == {"checked": "0-1", "not_configured": "2"}
    # One rank has nothing to be compared with.
    alone = hand_store(tmp_path / "alone", {0: steps(60, **CUDA)})
    evidence, summary, _ = three_ways(alone, tmp_path)
    assert evidence["kind_checks"]["memory_imbalance"] == {"value": "not_configured", "enough_history": False, "needs": needs}
    assert evidence["kind_checks"]["memory_pressure"]["value"] == "checked"
    assert count_line(summary).startswith("9 of 32 kinds checked on the only rank, ")
    # A rank whose memory could not be read is unavailable for the comparison, not unconfigured.
    failed = {"check": "cuda", "kinds": ["memory_growth", "memory_pressure", "allocator_retries"], "reason": "no CUDA"}
    broken = hand_store(tmp_path / "broken", {0: steps(60, **CUDA), 1: steps(60)}, not_checked={1: [failed]})
    evidence, _, _ = three_ways(broken, tmp_path)
    check = evidence["kind_checks"]["memory_imbalance"]
    assert (check["value"], check["reason"], check["ranks"]) == ("unavailable", "no CUDA", {"checked": [0], "unavailable": [1]})


def test_duplicate_batches_needs_data_on_every_judged_rank_and_the_servers_status(tmp_path):
    data = {"data": lambda step: {"batch_hash": f"h{step}", "epoch": 0, "batch": step}}
    both = hand_store(tmp_path / "both", {0: steps(60, **data), 1: steps(60, **data)})
    evidence, _, _ = three_ways(both, tmp_path)
    check = evidence["kind_checks"]["duplicate_batches"]
    assert (check["value"], check["enough_history"]) == ("checked", True) and "only when it finds a duplicate" in check["note"]
    # The server makes this comparison and its status carries the result: no status, no verdict.
    silent = hand_store(tmp_path / "silent", {0: steps(60, **data), 1: steps(60, **data)}, status=None)
    evidence, _, _ = three_ways(silent, tmp_path)
    check = evidence["kind_checks"]["duplicate_batches"]
    assert (check["value"], check["check"], check["enough_history"]) == ("unavailable", "status", False)
    one = hand_store(tmp_path / "one", {0: steps(60, **data), 1: steps(60)})
    evidence, _, _ = three_ways(one, tmp_path)
    check = evidence["kind_checks"]["duplicate_batches"]
    assert (check["value"], check["ranks"]) == ("not_configured", {"checked": [0], "not_configured": [1]})
    alone = hand_store(tmp_path / "alone", {0: steps(60, **data)})
    evidence, _, _ = three_ways(alone, tmp_path)
    assert evidence["kind_checks"]["duplicate_batches"]["value"] == "not_configured"


def test_the_count_line_stays_short_with_many_ranks_and_many_kinds(tmp_path):
    full = {**WEIGHTS, **CUDA, "layers": {"blk": {"abs_max": 1.0, "mean": 0.0, "std": 1.0, "zero_frac": 0.0}},
            "step_skipped": False, "replica_spread": {"w": 0.0}, "checkpoint": {"path": "x"}}
    ranks = {rank: steps(60, **(full if rank % 2 else {})) for rank in range(16)}
    evidence, summary, status = three_ways(hand_store(tmp_path, ranks), tmp_path)
    line = count_line(summary)
    assert "checked on some ranks only (stalled_weights on ranks 1, 3, 5, 7, and 4 more, " in line and ", and " in line
    assert line.startswith("6 of 32 kinds checked on all 16 ranks, 26 not configured, ") and len(line) < 700
    assert len(status["kinds"]["differs_by_rank"]) > ev.CHECKED_LINE_KINDS  # the listing has them all


def test_one_ranks_evidence_and_its_count_are_what_they_were(tmp_path):
    # The run-level object is the all-ranks view's alone: one rank of the same store reads as before.
    folder = hand_store(tmp_path, {0: steps(60, **WEIGHTS), 1: steps(60)})
    look = ev.read_server_report(folder / "rank=1")
    one = ev._report_evidence(folder / "rank=1", look, ev.read_watchdog(folder / "watchdog.json"))
    assert one["kind_checks"]["stalled_weights"] == checked(0, "stalled_weights", "weights_unchanged with grad_norms (full capture)", "not_configured")
    assert "ranks" not in json.dumps(one["kind_checks"]) and "kind_checks_not_stated" not in one
    assert ev.checked_line(one) == "6 of 32 kinds checked, 26 not configured"
