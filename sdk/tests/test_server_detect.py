"""`build_evidence(detectors="rust")` against the Python detectors, on the same records.

The Rust path runs `tensorquill-server detect --samples` on the records in hand and takes the
incidents of the server's kinds and the per-step MoE `samples` from its report; everything
else in the evidence is still made in Python. Both paths must give the same evidence:
incidents, `not_checked`, what was checked (`kind_checks`) and the run state compared as the
server parity tests compare them (numbers to 1e-12, a NaN as `null`), the rest exactly. The samples are compared with no
tolerance: every field of every sample must be the same value of the same type
(`sample_differences`). The two places where the paths differ on purpose are pinned further
down.

The five threshold options go to the Rust rules as the server's one `--thresholds` JSON
object. `CASES` stays at the defaults; the options have their own table (`OPTIONS`).

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill.evidence import SERVER_KINDS  # noqa: E402
from tensorquill.server_detect import BINARY_ENV, server_binary, server_report  # noqa: E402

import test_attention  # noqa: E402
import test_data_server  # noqa: E402
import test_evidence_system  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402
from test_server_detectors import differences, edges, everything, moe_edges, one_parameter, record, wobble  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
BAKEOFF = Path(__file__).resolve().parents[2] / "experiments" / "moe-bakeoff"


def gate_fixtures() -> dict[str, list[dict]]:
    """The seven Gate 1 fixtures' records, made by the bake-off's own generator. Its
    `artifacts` folder is not in the repository, so nothing here reads it. Empty when the
    experiment is not in this checkout."""
    if not (BAKEOFF / "generate.py").exists():
        return {}
    sys.path.insert(0, str(BAKEOFF))
    try:
        import generate
    finally:
        sys.path.remove(str(BAKEOFF))
    return {name: fixture.records() for name, fixture in generate.BY_NAME.items()}


GATE_FIXTURES = gate_fixtures()
JUDGED = ("incidents", "not_checked", "kind_checks", "run_state")


@pytest.fixture(autouse=True)
def binary(monkeypatch):
    monkeypatch.setenv(BINARY_ENV, str(BINARY))


def sample_differences(python: list[dict], rust: list[dict], limit: int = 20) -> list[str]:
    """Where the Rust path's MoE samples differ from Python's, by step, router and field.

    No tolerance: a field must hold the same value of the same type, a float the same double.
    Both paths read the same recorded numbers and do the same arithmetic in the same order.
    """
    if len(python) != len(rust):
        return [f"samples: {len(python)} from Python, {len(rust)} from Rust"]
    found = []
    for at, (ours, theirs) in enumerate(zip(python, rust)):
        where = f"samples[{at}] step {ours.get('step')} router {ours.get('router')}"
        for field in dict.fromkeys([*ours, *theirs]):
            if field not in ours or field not in theirs:
                found.append(f"{where}: {field} is only in {'Python' if field in ours else 'Rust'}'s sample")
            elif json.dumps(ours[field], sort_keys=True) != json.dumps(theirs[field], sort_keys=True):
                # As text: 1 is not 1.0, and a float is the same only when every digit is.
                found.append(f"{where}: {field} {ours[field]!r} != {theirs[field]!r}")
    return found[:limit] + ([f"and {len(found) - limit} more sample fields"] if len(found) > limit else [])


def both(records, **options) -> list[str]:
    """Where the Rust path's evidence differs from the Python path's, both given `options`."""
    python = tensorquill.build_evidence(records, **options)
    rust = tensorquill.build_evidence(records, detectors="rust", **options)
    assert "detectors" not in python and rust.pop("detectors") == "rust"
    found = [d for key in JUDGED for d in differences(python.get(key, []), rust.get(key, []), key)]
    found += sample_differences(python["samples"], rust["samples"])

    def rest(evidence):
        return json.dumps({k: v for k, v in evidence.items() if k not in (*JUDGED, "samples")}, sort_keys=True)

    return found + (["the keys made in Python differ"] if rest(python) != rest(rust) else [])


def lr_resumes() -> list[list[dict]]:
    """A run with resumes and restarts, as the stretches it was written in."""
    def warmup(start):
        return lambda step: 1e-3 * min(1.0, (step - start + 1) / 100)

    def attempt(number, steps, lr):
        return [
            record(step, number, grad_norm=wobble(step), scalars={"train/loss": 2.0 - step / 1000 + number / 100},
                   **({} if lr is None else {"lr": [lr(step), 0.1 * lr(step)]}))
            for step in steps
        ]

    return [
        attempt(1, range(150), warmup(0)),
        attempt(2, range(100, 180), warmup(100)),  # warmup starts over at a step the run had
        attempt(3, range(200, 220), lambda step: 5e-4),  # a resume past the stored steps
        attempt(4, range(230, 240), lambda step: 1e-5),
        attempt(5, range(235, 250), lambda step: 1e-5),
        attempt(6, range(220, 245), None),  # no learning rate
        attempt(7, range(120, 130), warmup(0)),
        attempt(8, range(0, 120), warmup(0)),  # a restart from step 0
        attempt(9, range(0, 10), lambda step: 3e-4),
    ]


def growing(stretches) -> list[list[dict]]:
    sent: list[dict] = []
    return [list(sent := sent + stretch) for stretch in stretches]


def cases() -> dict[str, list[dict]]:
    out = {"everything": everything(), "one parameter": one_parameter(), "moe edges": moe_edges()}
    out.update({f"edge {name}": records for name, records in edges().items()})
    stretches = test_data_server.attempts()
    out.update({
        f"data, up to {name}": records
        for (name, _), records in zip(stretches, growing(records for _, records in stretches))
    })
    out.update({f"data edge {name}": records for name, records in test_data_server.edges().items()})
    out["attention"] = test_attention.parity_run()
    out.update({f"attention edge {run}": test_attention.edge_run(run) for run in ("a", "b")})
    out.update({f"resumes, {n + 1} attempts written": records for n, records in enumerate(growing(lr_resumes()))})
    system = test_evidence_system.system_run()
    # Ends inside and at a reading, and at or just past each incident, as the server's stretches do.
    out.update({f"system, {upto} steps": system[:upto] for upto in (10, 11, 112, 115, 190, 230, 260, 300)})
    out.update({f"system edge {name}": records for name, records in test_evidence_system.edges().items()})
    # Checks that could not be made: too few probes, and resumes with nothing to compare.
    out["attention, the first probes"] = test_attention.parity_run()[:30]
    row, tag = test_data_server.row, test_data_server.tag
    out["a resume after a rollback"] = (
        [row(s, 1, tag(s)) for s in range(30)] + [row(s, 2, f"b-{s}") for s in range(10, 15)]
        + [row(s, 3, tag(s)) for s in range(20, 25)]
    )
    unloadable = [row(s, 1, tag(s)) for s in range(10)]
    unloadable[3]["checkpoint"] = {"save": {"components": test_data_server.FULL, "after_step": 30}}
    out["a resume from a checkpoint it cannot have loaded"] = unloadable + [row(s, 2, tag(s)) for s in range(5, 12)]
    out["no run id and no attempt"] = [{"step": s, "ts": float(s), "grad_norm": wobble(s)} for s in range(30)]
    return out


CASES = cases()


def test_the_cases_reach_every_kind_the_server_has():
    seen = {i["kind"] for records in CASES.values() for i in tensorquill.build_evidence(records)["incidents"]}
    assert set(SERVER_KINDS) <= seen, set(SERVER_KINDS) - seen
    # And the checks that could not be made, which the Rust path lists in Python's place.
    listed = {n["check"] for records in CASES.values() for n in tensorquill.build_evidence(records).get("not_checked", [])}
    assert {"resume_lr", "resume_data", "resume_state", "attention", "cuda_memory", "nvml"} <= listed, listed


@pytest.mark.parametrize("name", CASES)
def test_the_rust_path_gives_the_python_paths_evidence(name):
    assert both(CASES[name]) == []


@pytest.mark.parametrize("fixture", sorted(GATE_FIXTURES))
def test_the_gate_1_fixtures_give_the_same_evidence(fixture):
    records = GATE_FIXTURES[fixture]
    assert both(records) == []
    assert tensorquill.build_evidence(records, detectors="rust")["incidents"], "every fixture has a planted failure"


def test_the_report_is_the_document_the_server_writes():
    report = server_report(everything())
    assert report["source"] == "tensorquill-server" and report["rebuilds"] == 0
    assert (report["run_id"], report["rank"], report["steps"], report["last_step"]) == ("detect", 0, 300, 299)
    assert tuple(report["checked"]) == tuple(SERVER_KINDS)
    assert all("class" in incident for incident in report["incidents"])
    assert "samples" not in report, "the server's incidents.json holds no per-step samples"
    # What was checked is a key of its own; the two lists beside it keep their shapes.
    assert list(report["kind_checks"]) == sorted(SERVER_KINDS)
    assert all(isinstance(kind, str) for kind in report["checked"])
    assert all(isinstance(item, dict) and "check" in item and "reason" in item for item in report["not_checked"])


# --- the per-step MoE samples, made in Rust ---------------------------------------------------


def moe_sets() -> dict[str, list[dict]]:
    """The record sets that have MoE samples at all."""
    sets = {**CASES, **{f"fixture {name}": records for name, records in GATE_FIXTURES.items()}}
    return {name: records for name, records in sets.items() if any(r.get("moe") for r in records)}


MOE_SETS = moe_sets()


def test_the_moe_sets_hold_every_form_a_sample_takes():
    assert len(MOE_SETS) >= 7, sorted(MOE_SETS)
    samples = [s for records in MOE_SETS.values() for s in tensorquill.build_evidence(records)["samples"]]
    judged = [s for s in samples if not s.get("missing")]
    assert {s["overload_rule"]["rule"] for s in judged} == {"share", "diversity_drop", "collapse"}
    assert any(s.get("missing") and not any(s["counts"]) for s in samples), "a router a step did not record"
    assert any(s["overloaded"] for s in judged) and any(s["dead"] for s in judged) and any(s["starved"] for s in judged)
    # A window that was judged with nobody starved, and one that could not be judged yet.
    assert any(s.get("starved_shares") == {} for s in judged) and any("starved_shares" not in s for s in judged)
    assert any(len(s["counts"]) != len(t["counts"]) for s in judged for t in judged[:1]), "routers of several sizes"


@pytest.mark.parametrize("name", MOE_SETS)
def test_rust_makes_the_samples_python_makes_field_by_field(name):
    records = MOE_SETS[name]
    python = tensorquill.build_evidence(records)["samples"]
    assert python, "this record set has MoE samples"
    assert sample_differences(python, tensorquill.build_evidence(records, detectors="rust")["samples"]) == []
    # And straight from the report, before `build_evidence` has touched them.
    from tensorquill.evidence import load_records

    report = server_report(load_records(records, all_attempts=True), samples=True)
    assert sample_differences(python, report["samples"]) == []
    plain = server_report(load_records(records, all_attempts=True))
    assert {key: value for key, value in report.items() if key != "samples"} == plain


def test_the_rust_path_runs_no_python_moe_rule(monkeypatch):
    import tensorquill.evidence as evidence

    usual = tensorquill.build_evidence(moe_edges())["samples"]

    def never(*args, **kwargs):
        raise AssertionError("the Python MoE rule ran on the Rust path")

    for name in ("_moe_samples", "_mark_starved", "_jump_overloaded", "effective_overload_threshold", "_dead_experts_significant"):
        monkeypatch.setattr(evidence, name, never)
    rust = tensorquill.build_evidence(moe_edges(), detectors="rust")
    assert sample_differences(usual, rust["samples"]) == [] and len(usual) == 60
    with pytest.raises(AssertionError, match="the Python MoE rule ran"):
        tensorquill.build_evidence(moe_edges())


@pytest.mark.parametrize("change, said", [
    (lambda s: s.update(routed_tokens=s["routed_tokens"] + 1), "routed_tokens"),
    (lambda s: s.update(diversity=s["diversity"] * (1 + 2.3e-16)), "diversity"),  # the next double
    (lambda s: s["counts"].__setitem__(0, s["counts"][0] + 1), "counts"),
    (lambda s: s.update(overloaded=[*s["overloaded"], 2]), "overloaded"),
    (lambda s: s["overload_rule"].update(rule="collapse"), "overload_rule"),
    (lambda s: s.update(starved_shares={}), "starved_shares is only in Rust's sample"),
    (lambda s: s.pop("starved"), "starved is only in Python's sample"),
    (lambda s: s.update(layer=float(s["layer"])), "layer"),  # 1.0 where 1 was
])
def test_one_wrong_field_in_one_sample_is_named(monkeypatch, change, said):
    import tensorquill.server_detect as module

    real = module.server_report

    def tampered(records, **kwargs):
        report = real(records, **kwargs)
        sample = next(s for s in report["samples"] if not s.get("missing") and "starved_shares" not in s and s["step"] > 0)
        change(sample)
        return report

    assert both(moe_edges()) == []
    monkeypatch.setattr(module, "server_report", tampered)
    found = both(moe_edges())
    assert len(found) == 1 and said in found[0] and "step " in found[0] and "router " in found[0], found


def test_a_binary_that_gives_no_samples_or_other_routers_is_refused(monkeypatch, tmp_path):
    import tensorquill.server_detect as module

    real = module.server_report
    drop = {}

    def other(records, **kwargs):
        report = real(records, **kwargs)
        report["samples"] = report["samples"][drop["from"]:]
        return report

    monkeypatch.setattr(module, "server_report", other)
    drop["from"] = 1
    with pytest.raises(RuntimeError, match="MoE samples"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")
    monkeypatch.undo()
    monkeypatch.setenv(BINARY_ENV, str(BINARY))

    # A binary from before `--samples`: it refuses the flag, or (if it ignored it) prints no samples.
    old = tmp_path / "old-server"
    old.write_text("#!/bin/sh\necho \"error: unexpected argument '--samples' found\" >&2\nexit 2\n", encoding="utf-8")
    old.chmod(0o755)
    with pytest.raises(RuntimeError, match="too old to have `detect`, `--thresholds` or `--samples`"):
        server_report(moe_edges(), samples=True, binary=old)
    silent = tmp_path / "silent-server"
    silent.write_text("#!/bin/sh\ncat > /dev/null\necho '{\"incidents\": []}'\n", encoding="utf-8")
    silent.chmod(0o755)
    with pytest.raises(RuntimeError, match="without samples"):
        server_report(moe_edges(), samples=True, binary=silent)
    assert server_report(moe_edges(), binary=silent) == {"incidents": []}


# --- what was checked: one of three values for every kind -------------------------------------

from test_kind_checks import probed, router_only  # noqa: E402

# The parity record sets have no probe run without a failed probe, and no router-only capture.
CHECK_SETS: dict[str, list[dict]] = {
    "router only": router_only(),
    "probes, all good": probed(),
    "probes, one failed": probed(fail_at=(30,)),
    "probes, all failed": probed(fail_at=tuple(range(0, 60, 10))),
    "probes, too few": probed(steps=30),
    "probes, too few and one failed": probed(steps=30, fail_at=(20,)),
    "a probe whose error is not text": [{"run_id": "r", "step": 0, "ts": 0.0, "probe": {"error": {"code": 3, "why": [1, 2]}}}],
}


@pytest.mark.parametrize("name", CHECK_SETS)
def test_both_paths_say_the_same_about_what_was_checked(name):
    assert both(CHECK_SETS[name]) == []
    evidence = tensorquill.build_evidence(CHECK_SETS[name], detectors="rust")
    degraded = "failed" in name or "too few" in name or "error" in name
    assert evidence["run_state"]["state"] == ("degraded" if degraded else "healthy")


def test_a_router_only_capture_and_a_failed_probe_read_the_same_through_rust():
    router = tensorquill.build_evidence(CHECK_SETS["router only"], detectors="rust")
    values = {kind: check["value"] for kind, check in router["kind_checks"].items()}
    assert all(values[kind] == "not_configured" for kind in ("stalled_weights", "activation_spike", "activation_drift", "dead_activations"))
    assert router["run_state"] == {"state": "healthy", "reasons": []}
    from tensorquill.evidence import incident_summary

    assert incident_summary(router).splitlines()[:2] == ["No incidents detected.", "3 of 32 kinds checked, 29 not configured"]
    failed = tensorquill.build_evidence(CHECK_SETS["probes, one failed"], detectors="rust")
    assert failed["run_state"]["state"] == "degraded" and [n["check"] for n in failed["not_checked"]] == ["probe"]
    assert failed["kind_checks"]["attention_collapse"] == {
        "value": "unavailable", "records": 5, "needs": "probe with attention (probe=)", "check": "probe",
        "reason": "1 of 6 probes failed, the first at step 30: RuntimeError: CUDA out of memory",
    }


def test_the_report_gives_pythons_value_for_every_kind_and_every_value_is_reached():
    from tensorquill.evidence import load_records

    sets = {**CASES, **{f"fixture {name}": records for name, records in GATE_FIXTURES.items()}, **CHECK_SETS}
    reached: dict[str, set[str]] = {kind: set() for kind in SERVER_KINDS}
    for name, records in sets.items():
        python = tensorquill.build_evidence(records)["kind_checks"]
        report = server_report(load_records(records, all_attempts=True))["kind_checks"]
        # No tolerance: a value, a count and two strings.
        assert report == {kind: python[kind] for kind in SERVER_KINDS}, name
        for kind, check in report.items():
            reached[kind].add(check["value"])
    assert reached.pop("repeated_steps") == {"checked"}  # every record is its input
    assert all({"checked", "not_configured"} <= found for found in reached.values()), reached
    # The kinds a `not_checked` entry can name.
    can_fail = {
        "resume_lr_change", "resume_data_mismatch", "resume_incomplete", "attention_collapse", "attention_uniform",
        "attention_logit_growth", "memory_growth", "memory_pressure", "allocator_retries", "hardware_errors", "gpu_throttled",
    }
    assert {kind for kind, found in reached.items() if "unavailable" in found} == can_fail


def test_the_rust_path_takes_each_kinds_value_from_the_report(monkeypatch):
    import tensorquill.server_detect as module

    real = module.server_report
    change: dict = {}

    def other(records, **kwargs):
        report = real(records, **kwargs)
        if "drop" in change:
            del report["kind_checks"]
        else:
            report["kind_checks"]["grad_spike"] = change["grad_spike"]
        return report

    made_up = {"value": "unavailable", "records": 7, "needs": "x", "check": "made up", "reason": "because"}
    change["grad_spike"] = made_up
    monkeypatch.setattr(module, "server_report", other)
    evidence = tensorquill.build_evidence(moe_edges(), detectors="rust")
    assert evidence["kind_checks"]["grad_spike"] == made_up
    assert len(evidence["kind_checks"]) == 32 and evidence["kind_checks"]["job_stalled"]["value"] == "not_configured"
    # And the comparison names it: this is what a wrong value in the Rust report looks like.
    found = both(moe_edges())
    assert found and all(d.startswith("kind_checks.grad_spike") for d in found), found
    change["grad_spike"] = {"value": "fine", "records": 7, "needs": "x"}
    with pytest.raises(RuntimeError, match="does not say what was checked"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")
    change["drop"] = True
    with pytest.raises(RuntimeError, match="does not say what was checked"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")


# --- where the two paths differ, on purpose -------------------------------------------------


def test_a_step_written_twice_in_one_attempt_is_reported_with_the_scalars_of_its_second_writing():
    def row(step, salt):
        return record(step, 1, grad_norm=wobble(step), scalars={"train/loss": 2.0 - step / 100 + salt})

    records = [row(s, 0) for s in range(40)] + [row(s, 1) for s in range(30, 50)]
    python = tensorquill.build_evidence(records)["incidents"]
    rust = tensorquill.build_evidence(records, detectors="rust")["incidents"]
    assert [i["kind"] for i in python] == [i["kind"] for i in rust] == ["repeated_steps"]
    # The incident is the same: the server's own store never holds such rows, so only this path finds it.
    assert {k: v for k, v in python[0].items() if k != "scalars"} == {k: v for k, v in rust[0].items() if k != "scalars"}
    assert (python[0]["step"], python[0]["repeated"]) == (30, 10)
    # Python gives what was logged when step 30 was first written, Rust when it was written again.
    assert python[0]["scalars"]["train/loss"]["value"] == pytest.approx(1.7)
    assert rust[0]["scalars"]["train/loss"]["value"] == pytest.approx(2.7)


def test_attempts_out_of_order_in_the_input_are_read_oldest_first_as_a_store_holds_them():
    first = [record(s, 1, grad_norm=wobble(s), lr=[1e-3 * min(1.0, (s + 1) / 100)], scalars={"train/loss": 2.0 - s / 1000}) for s in range(120)]
    resumed = [record(s, 2, grad_norm=wobble(s), lr=[1e-3 * min(1.0, (s - 99) / 100)], scalars={"train/loss": 2.02 - s / 1000}) for s in range(100, 150)]
    rust = tensorquill.build_evidence(resumed + first, detectors="rust")["incidents"]
    assert [i["kind"] for i in rust] == ["resume_lr_change"]
    assert differences(tensorquill.build_evidence(first + resumed)["incidents"], rust) == []


# --- what the Rust path refuses -------------------------------------------------------------


# --- the five options, through `--thresholds` -------------------------------------------------

OPTIONS: list[dict] = (
    [{"overload_threshold": v} for v in (0.0, 0.2, 0.5, 0.9)]
    + [{"drift_tolerance": v} for v in (0.0, 1e-3, 0.5)]
    + [{"starve_window": v} for v in (1, 10, 20)]
    + [{"starve_ratio": v} for v in (0.0, 0.5, 0.9)]
    + [{"top_k": v} for v in (1, 2, 3, 8)]
    + [{"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2}]
)


def option_sets() -> dict[str, list[dict]]:
    """The record sets these options can change: the ones with MoE counts or a replica spread."""
    sets = dict(CASES)
    sets.update({f"fixture {name}": records for name, records in GATE_FIXTURES.items()})
    return {name: records for name, records in sets.items() if any(r.get("moe") or r.get("replica_spread") for r in records)}


OPTION_SETS = option_sets()
_usual: dict[str, list] = {}


def moves_python(name: str, **options) -> bool:
    """Whether the options change the incidents Python reports on this record set."""
    if name not in _usual:
        _usual[name] = tensorquill.build_evidence(OPTION_SETS[name])["incidents"]
    return tensorquill.build_evidence(OPTION_SETS[name], **options)["incidents"] != _usual[name]


def test_the_option_sets_hold_the_data_the_options_judge():
    assert len(OPTION_SETS) >= 8, sorted(OPTION_SETS)
    recorded = {k for records in OPTION_SETS.values() for r in records for k in (r.get("moe_top_k") or {}).values()}
    # Top-1 routers for the share rule, top-k of three or more for the baseline rule, and a
    # recorded value on every row for an override to stand against.
    assert 1 in recorded and any(k >= 3 for k in recorded), recorded
    assert any(r.get("replica_spread") for records in OPTION_SETS.values() for r in records)


@pytest.mark.parametrize("option", OPTIONS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_an_option_reaches_the_rust_rules_and_gives_pythons_result(option):
    moved = []
    for name, records in OPTION_SETS.items():
        assert both(records, **option) == [], name
        if moves_python(name, **option):
            moved.append(name)
    # Without this the test would pass with the option ignored on both sides.
    assert moved, "this value changes no incident on any record set"


# One router of a record set, and a value for it other than the one recorded on every row,
# chosen because the override moves an incident there.
ONE_ROUTER = {
    "everything": ("a", 2),
    "moe edges": ("early", 1),
    "fixture unseen-3": ("layers.11.router", 2),
    "fixture unseen-4": ("layers.08.router", 3),
    "fixture unseen-6": ("layers.03.router", 8),
}


@pytest.mark.parametrize("name", ONE_ROUTER)
def test_a_top_k_for_one_router_stands_against_the_value_recorded_on_every_row(name):
    records = OPTION_SETS[name]
    target, override = ONE_ROUTER[name]
    recorded = {
        router: {r["moe_top_k"][router] for r in records if router in (r.get("moe_top_k") or {})}
        for router in {router for r in records for router in (r.get("moe") or {})}
    }
    assert recorded[target] and override not in recorded[target], "the override must differ from what the rows carry"
    option = {"top_k": {target: override, "not-a-router": 5}}
    assert both(records, **option) == []
    assert moves_python(name, **option), "this override changes no incident, so the test would pass with it ignored"
    evidence = tensorquill.build_evidence(records, detectors="rust", **option)
    assert evidence["top_k"][target] == override and "not-a-router" not in evidence["top_k"]
    # Every other router keeps what its rows recorded.
    others = [router for router in recorded if router != target]
    assert others and all(evidence["top_k"][router] == max(recorded[router], default=1) for router in others)


def test_the_report_lists_the_thresholds_it_applied():
    usual = server_report(everything())["thresholds"]
    assert (usual["overload_threshold"], usual["drift_tolerance"], usual["starve_window"], usual["starve_ratio"]) == (0.35, 1e-6, 50, 0.1)
    assert usual["top_k"] is None
    assert server_report(everything(), thresholds={})["thresholds"] == usual  # every key left out
    asked = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": {"r1": 2}}
    applied = server_report(everything(), thresholds=asked)["thresholds"]
    assert {name: applied[name] for name in asked} == asked
    assert {name: value for name, value in applied.items() if name not in asked} == {
        name: value for name, value in usual.items() if name not in asked
    }
    # The drift incident says which tolerance it was judged by.
    drift = [i for i in server_report(everything(), thresholds={"drift_tolerance": 0.0})["incidents"] if i["kind"] == "replica_drift"]
    assert drift and drift[0]["tolerance"] == 0.0


@pytest.mark.parametrize("option, reason", [
    ({"starve_window": 0}, "starve_window must be at least 1"),
    ({"top_k": 0}, "top_k must be at least 1"),
    ({"top_k": {"r1": 0}}, "top_k must be at least 1"),
    ({"overload_threshold": -0.1}, "overload_threshold must be a finite number of at least 0"),
    ({"drift_tolerance": -1.0}, "drift_tolerance must be"),
    ({"starve_ratio": float("nan")}, "cannot be sent as JSON"),
    ({"overload_threshold": float("inf")}, "cannot be sent as JSON"),
])
def test_values_the_rust_rules_refuse_raise_value_error_with_the_reason(option, reason):
    with pytest.raises(ValueError, match=reason):
        tensorquill.build_evidence(moe_edges(), detectors="rust", **option)


def test_an_unknown_threshold_key_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown field `overload_treshold`"):
        server_report(moe_edges(), thresholds={"overload_treshold": 0.5})
    with pytest.raises(ValueError, match="unknown field `spike_factor`"):  # listed in a report, but not an option
        server_report(moe_edges(), thresholds={"spike_factor": 4.0})


def test_thresholds_are_compared_to_one_part_in_1e12_in_case_json_text_moves_a_last_place():
    # A precaution: no run has shown a value come back changed.
    from tensorquill.evidence import _same_threshold

    assert both(moe_edges(), overload_threshold=0.1 + 0.2, drift_tolerance=1e-6 * 3, starve_ratio=1 / 3) == []
    assert _same_threshold(0.35, 0.35) and _same_threshold(0.35, 0.35000000000000003) and _same_threshold(1, 1.0)
    assert not _same_threshold(0.35, 0.5) and not _same_threshold(0.35, None) and not _same_threshold(None, 0.35)
    assert _same_threshold(None, None) and _same_threshold({"a": 2}, {"a": 2}) and _same_threshold(50, 50)
    assert not _same_threshold(2, None) and not _same_threshold({"a": 2}, {"a": 3}) and not _same_threshold(50, 20)


def test_a_server_started_with_thresholds_applies_them_and_a_reader_asking_for_the_same_finds_it_current(tmp_path):
    from tensorquill.evidence import read_server_report
    from test_server_detectors import FAST, Stream

    asked = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2}
    records = everything()

    def reference(upto):
        return [i for i in tensorquill.build_evidence(records[:upto], **asked)["incidents"] if i["kind"] in SERVER_KINDS]

    assert reference(300) != [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in SERVER_KINDS]
    stream = Stream(tmp_path, *FAST, "--thresholds", json.dumps(asked))
    sent = 0
    for upto in (60, 130, 200):  # each stretch is appended to the detectors the server holds
        document = stream.send(records[sent:upto])
        sent = upto
        assert differences(reference(upto), document["incidents"]) == [], f"after {upto} steps"
    assert {name: document["thresholds"][name] for name in asked} == asked and document["rebuilds"] == 1
    # The server's own incidents.json says what was checked, as the same function makes it.
    usual = tensorquill.build_evidence(records[:200], **asked)["kind_checks"]
    assert document["kind_checks"] == {kind: usual[kind] for kind in SERVER_KINDS}
    stream.restart()  # the new server rebuilds from its store, with the thresholds it was started with
    document = stream.send(records[200:])
    assert differences(reference(300), document["incidents"]) == []
    assert {name: document["thresholds"][name] for name in asked} == asked

    folder = stream.incidents_file.parent
    assert read_server_report(folder, **asked)["state"] == "current"
    look = read_server_report(folder)
    assert look["state"] == "other_thresholds"
    assert "top_k 2 where None was asked for" in look["reason"] and "overload_threshold 0.5 where 0.35" in look["reason"]
    assert read_server_report(folder, **{**asked, "top_k": {"r1": 2}})["state"] == "other_thresholds"
    assert stream.server.terminate() == 0

    # The watcher takes the server's incidents when it asks for the same values, and runs
    # the Python detectors when it asks for others.
    from test_server_reports import look

    evidence, _ = look(stream, tmp_path / "same", **asked)
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("server", "current")
    assert differences(reference(300), [i for i in evidence["incidents"] if i["kind"] in SERVER_KINDS]) == []
    evidence, _ = look(stream, tmp_path / "usual")
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("python", "other_thresholds")
    # (A logged NaN is in these incidents, so they are compared by kind and step.)
    usual = tensorquill.build_evidence(records)["incidents"]
    assert [(i["kind"], i["step"]) for i in evidence["incidents"]] == [(i["kind"], i["step"]) for i in usual]
    assert [(i["kind"], i["step"]) for i in usual] != [(i["kind"], i["step"]) for i in reference(300)]


def test_the_server_refuses_bad_thresholds_before_it_listens_or_reads(tmp_path):
    store = tmp_path / "store"
    for arguments in (
        ["--store", str(store), "--thresholds", '{"bogus": 1}'],
        ["detect", "--thresholds", '{"bogus": 1}'],
        ["--thresholds", '{"bogus": 1}', "detect"],  # the option is taken before the subcommand too
        ["detect", "--thresholds", "[0.5]"],
        ["detect", "--thresholds", "0.5"],
        ["detect", "--thresholds", "null"],
    ):
        done = subprocess.run([str(BINARY), *arguments], input=b"", capture_output=True, timeout=30)
        assert done.returncode == 2, arguments
        assert b"--thresholds" in done.stderr and (b"unknown field `bogus`" in done.stderr or b"expected a JSON object" in done.stderr)
        assert done.stdout == b""
    assert not store.exists()
    # Before the subcommand, a good value is applied, not dropped.
    stream = __import__("tensorquill.server_detect", fromlist=["records_stream"]).records_stream(moe_edges())
    done = subprocess.run([str(BINARY), "--thresholds", '{"starve_window": 7}', "detect"], input=stream, capture_output=True, timeout=30)
    assert done.returncode == 0 and json.loads(done.stdout)["thresholds"]["starve_window"] == 7


def test_an_unknown_engine_and_a_missing_binary_say_so(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="'python' or 'rust'"):
        tensorquill.build_evidence(moe_edges(), detectors="server")
    monkeypatch.delenv(BINARY_ENV)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match=BINARY_ENV):
        tensorquill.build_evidence(moe_edges(), detectors="rust")
    with pytest.raises(RuntimeError, match="is not a file"):
        server_binary(tmp_path / "missing")
    # No records: nothing to judge, and the binary is not needed.
    assert tensorquill.build_evidence([], detectors="rust")["incidents"] == []


def test_the_binary_refuses_two_ranks_and_its_message_is_passed_on():
    rows = [record(s, 1) for s in range(5)] + [{**record(s, 1), "rank": 1} for s in range(5)]
    with pytest.raises(RuntimeError, match="send one rank of one run"):
        server_report(rows)
    with pytest.raises(ValueError, match="pass rank="):
        tensorquill.build_evidence(rows, detectors="rust")
    assert both([r for r in rows if r["rank"] == 1]) == []


def test_a_server_built_from_other_rules_is_refused(monkeypatch):
    import tensorquill.server_detect as module

    real = module.server_report
    change = {}

    def older(records, **kwargs):
        report = real(records, **kwargs)
        report["thresholds"].update(change.get("thresholds", {}))
        report["checked"] = [kind for kind in report["checked"] if kind not in change.get("unchecked", ())]
        return report

    monkeypatch.setattr(module, "server_report", older)
    tensorquill.build_evidence(moe_edges(), detectors="rust")  # unchanged, it is taken
    change["thresholds"] = {"overload_threshold": 0.5}
    with pytest.raises(RuntimeError, match="overload_threshold 0.5, here 0.35"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")
    change.clear()
    change["unchecked"] = ("input_starvation", "attention_collapse")
    with pytest.raises(RuntimeError, match="does not check attention_collapse, input_starvation"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")


# --- the command line -----------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "tensorquill", *args], capture_output=True, text=True, timeout=120)


def test_the_command_line_takes_the_incidents_from_rust(tmp_path):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in everything()), encoding="utf-8")
    plain = run_cli(str(capture), "-o", str(tmp_path / "python.json"))
    rust = run_cli(str(capture), "-o", str(tmp_path / "rust.json"), "--detectors", "rust")
    assert plain.returncode == 0 and rust.returncode == 0, rust.stderr
    written = json.loads((tmp_path / "rust.json").read_text(encoding="utf-8"))
    assert written.pop("detectors") == "rust"
    assert differences(json.loads((tmp_path / "python.json").read_text(encoding="utf-8")), written, "evidence") == []
    # The summary a person reads is the same, line for line, but for the file names and a
    # logged NaN: it is `null` in the server's report, which is strict JSON, and reads
    # "not finite" where Python's own value reads "nan".
    assert "not finite" in rust.stdout and "not finite" not in plain.stdout
    assert plain.stdout.replace("python.json", "X").replace(" nan ", " not finite ") == rust.stdout.replace("rust.json", "X")


def test_the_command_lines_thresholds_go_to_rust(tmp_path):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in everything()), encoding="utf-8")
    options = ("--top-k", "2", "--overload-threshold", "0.5")
    plain = run_cli(str(capture), "-o", str(tmp_path / "python.json"), *options)
    rust = run_cli(str(capture), "-o", str(tmp_path / "rust.json"), "--detectors", "rust", *options)
    assert plain.returncode == 0 and rust.returncode == 0, rust.stderr
    python_evidence = json.loads((tmp_path / "python.json").read_text(encoding="utf-8"))
    written = json.loads((tmp_path / "rust.json").read_text(encoding="utf-8"))
    assert written.pop("detectors") == "rust" and written["overload_threshold"] == 0.5 and set(written["top_k"].values()) == {2}
    assert differences(python_evidence, written, "evidence") == []
    assert python_evidence["incidents"] != tensorquill.build_evidence(everything())["incidents"]


@pytest.mark.parametrize("extra", [("--watch",), ("--rank", "all")])
def test_the_command_line_refuses_what_the_rust_path_cannot_do(tmp_path, extra):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in moe_edges()), encoding="utf-8")
    done = run_cli(str(capture), "--detectors", "rust", *extra)
    assert done.returncode == 2, done.stdout
    assert "rust" in done.stderr.lower()
