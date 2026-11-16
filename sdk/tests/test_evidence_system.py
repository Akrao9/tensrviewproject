"""The GPU memory, NVML and data-wait detectors (`evidence_system.py`): each rule on and
off, the attempt reset, what could not be checked, and the cross-rank memory imbalance.
The fixtures here are also streamed through the server in `test_server_system_detectors.py`."""

from __future__ import annotations

import json

import pytest

import tensorquill
from tensorquill import evidence_system as es
from tensorquill.evidence_system import SYSTEM_KINDS, detect_memory_imbalance, detect_system_incidents, latest_memory

GIB = 1 << 30
MIB = 1 << 20
RUN = "system"


def record(step, attempt=1, run=RUN, **values):
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step, "ts": float(step), **values}


def wobble(step, salt=0):
    """A repeatable value between 0.9 and 1.1."""
    return 0.9 + ((step * 37 + salt * 11) % 21) / 100


def cuda(allocated=2 * GIB, reserved=10 * GIB, total=16 * GIB, retries=0, ooms=0, peak=None):
    return {"device": 0, "allocated": allocated, "reserved": reserved, "peak_allocated": peak or allocated + GIB,
            "total": total, "alloc_retries": retries, "ooms": ooms}


def nvml(ecc=3, throttle=("sw_power_cap",), xid=(), **more):
    return {"temperature_c": 70, "sm_clock_mhz": 1410, "power_w": 300.5, "power_limit_w": 400.0,
            "ecc_uncorrected": ecc, "ecc_corrected": 12, "throttle": list(throttle), "xid": list(xid), **more}


def system_run(steps=300):
    """Every per-rank kind once, at its own step, among what must not set one off: a jump in
    memory that then stays flat, a reserved share just under the line, ECC errors already
    there at the start, a GPU at its power cap, one very slow step at an epoch boundary, a
    metric NVML cannot read. A reading every 10 steps (step 9, 19, ...), a loss on every step."""
    out = []
    allocated = 2 * GIB
    for step in range(steps):
        values = {"scalars": {"train/loss": 2.0 / (1 + step / 50) * wobble(step)}}
        if step > 0:
            took = 0.1 * wobble(step, 1)
            share = 0.1 * wobble(step, 2) if step < 100 else 0.4 * wobble(step, 3)
            if step == 60:
                share = 0.9  # the first batch of an epoch: one slow step is not starvation
            values.update(data_wait=share * took, step_time=took)
        else:
            values.update(data_wait=0.2)  # the first step has no step time
        if step % 10 == 9:
            if step == 49:
                allocated = 4 * GIB  # a jump, then flat
            elif step >= 149:
                allocated += 200 * MIB  # the leak
            elif step % 20 == 9:
                allocated += MIB  # noise: up, then
            else:
                allocated -= MIB  # down
            reserved = int(15.1 * GIB) if step < 229 else int(15.3 * GIB)  # 94.4%, then 95.6%
            reading = {
                "cuda": cuda(allocated=allocated, reserved=reserved, retries=0 if step < 259 else 2),
                "nvml": nvml(
                    ecc=3 if step < 269 else 4,
                    throttle=("sw_power_cap", "hw_thermal") if step == 239 else ("sw_power_cap",),
                    **({"unavailable": {"xid": "NVMLError_NotSupported: Not Supported", "temperature_c": "x"}} if step == 9 else {}),
                ),
            }
            values["system"] = reading
        out.append(record(step, **values))
    return out


def edges():
    """Small runs for the edges the long run does not reach."""
    def run(name, steps, by_step):
        return [record(s, run=name, **by_step.get(s, {})) for s in range(steps)]

    rise = [
        record(s, run="rise", data_wait=(0.05 if s < 40 else 0.27) * 0.1, step_time=0.1) for s in range(80)
    ]
    readings = {s: {"system": {"cuda": cuda(allocated=GIB + s * 100 * MIB)}} for s in range(0, 60, 10)}
    # Three rising readings and 30 steps at a 20% share in attempt 1, then a resume at step 30
    # with three more rising readings and 15 steps at 45%: either would be an incident if it
    # were joined to the attempt before. No trend or window spans two processes.
    resumed = [record(s, run="resume", attempt=1, data_wait=0.02, step_time=0.1, **readings.get(s, {})) for s in range(40)]
    resumed += [record(s, run="resume", attempt=2, **({"data_wait": 0.045, "step_time": 0.1} if s < 45 else {}),
                       **readings.get(s, {})) for s in range(30, 55)]
    unavailable = run("unavailable", 30, {
        9: {"system": {"cuda": {"unavailable": "the model is on cpu, not a CUDA device"},
                       "nvml": {"unavailable": "pynvml is not installed (pip install nvidia-ml-py)"}}},
        19: {"system": {"cuda": {"unavailable": "again"}, "nvml": {"unavailable": "again"}}},
    })
    odd = run("odd", 40, {
        1: {"system": "not an object"},
        2: {"system": {"cuda": [1, 2], "nvml": "x"}},
        3: {"system": {"cuda": {"allocated": 1.5e9, "reserved": True, "total": "16", "alloc_retries": 1.0, "ooms": None}}},
        4: {"system": {"nvml": {"ecc_uncorrected": 2.0, "xid": ["79", 1.0, True, None], "throttle": [1, None, "hw_x"]}}},
        5: {"system": {"nvml": {"ecc_uncorrected": 0, "throttle": "hw_thermal"}}},
        6: {"system": {"nvml": {"ecc_uncorrected": False, "unavailable": {"throttle": 5, "power_w": "x"}}}},
        7: {"system": {"cuda": {"allocated": 3 * GIB, "reserved": 16 * GIB, "total": 0}}},
        8: {"data_wait": float("nan"), "step_time": 1.0},
        9: {"data_wait": -1.0, "step_time": 1.0},
        10: {"data_wait": 1.0, "step_time": 0.0},
        11: {"system": {"nvml": {"throttle": ["hw_power_brake"], "temperature_c": True, "power_w": "300"}}},
        12: {"system": {"nvml": {"ecc_uncorrected": 1}}},
        13: {"system": {"cuda": {"ooms": 1}}},
    })
    # A small model's growth: a large share of its first reading but under 64 MiB, then over it.
    small = [record(s, run="small", system={"cuda": cuda(allocated=100 * MIB + (s // 10) * (12 if s < 60 else 14) * MIB)})
             for s in range(0, 120, 10)]
    return {"rise": rise, "resume": resumed, "unavailable": unavailable, "odd": odd, "small": small}


def by_kind(records):
    return {i["kind"]: i for i in detect_system_incidents(records)}


# --- the long run --------------------------------------------------------------------------


def test_each_kind_once_at_its_step_and_nothing_from_the_decoys():
    found = by_kind(system_run())
    assert list(found) == list(SYSTEM_KINDS)
    growth = found["memory_growth"]
    assert (growth["first_step"], growth["step"], growth["readings"]) == (139, 189, 6)  # the flat reading at 139 starts the trend
    assert growth["bytes_per_step"] == pytest.approx(5 * 200 * MIB / 50) and growth["steps_to_total"] > 0
    assert (found["memory_pressure"]["step"], found["memory_pressure"]["reserved"]) == (229, int(15.3 * GIB))
    assert found["allocator_retries"] == {"kind": "allocator_retries", "step": 259, "alloc_retries": 2, "ooms": 0}
    assert found["hardware_errors"] == {
        "kind": "hardware_errors", "step": 269, "xid": [], "ecc_uncorrected": 4, "ecc_uncorrected_at_start": 3,
    }
    assert found["gpu_throttled"]["step"] == 239 and found["gpu_throttled"]["reasons"] == ["sw_power_cap", "hw_thermal"]
    starving = found["input_starvation"]
    assert starving["rule"] == "share" and 100 < starving["step"] < 120 and starving["median_share"] >= 0.3
    assert starving["baseline_share"] == pytest.approx(0.1, rel=0.1)


def test_build_evidence_lists_them_with_the_logged_loss_and_what_was_not_checked():
    evidence = tensorquill.build_evidence(system_run())
    kinds = [i["kind"] for i in evidence["incidents"]]
    assert kinds == list(SYSTEM_KINDS)
    assert all("train/loss" in i["scalars"] for i in evidence["incidents"])
    assert evidence["not_checked"] == [{
        "check": "nvml", "kinds": ["hardware_errors"], "reason": "NVMLError_NotSupported: Not Supported",
        "step": 9, "metric": "xid",
    }]
    assert evidence["memory"]["step"] == 299 and evidence["memory"]["reserved"] == int(15.3 * GIB)
    summary = tensorquill.incident_summary(evidence)
    for words in ("GPU memory kept growing", "near the device's total", "allocator ran short", "hardware errors",
                  "clocks throttled", "Input pipeline starving", "Not checked (nvml)"):
        assert words in summary, words


def test_nothing_without_readings_and_nothing_on_the_decoys_alone():
    plain = [record(s) for s in range(100)]
    assert detect_system_incidents(plain) == [] and latest_memory(plain) is None
    evidence = tensorquill.build_evidence(plain)
    assert "memory" not in evidence and "not_checked" not in evidence
    # The run up to step 140: the jump, the noise, the slow step, the power cap and the ECC
    # errors present from the start, and no incident.
    assert detect_system_incidents(system_run()[:100]) == []


# --- one rule at a time --------------------------------------------------------------------


def readings(values, key="allocated", **fixed):
    return [record(10 * i + 9, system={"cuda": {**fixed, key: v}}) for i, v in enumerate(values)]


@pytest.mark.parametrize("values, found", [
    ([GIB + k * 100 * MIB for k in range(6)], True),  # six readings, each above the last
    ([GIB + k * 100 * MIB for k in range(5)], False),  # five
    ([GIB + k * 100 * MIB for k in (0, 1, 2, 2, 3, 4)], False),  # one flat reading breaks it
    ([GIB + k * 10 * MIB for k in range(6)], False),  # under 64 MiB in all
    ([100 * MIB + k * 12 * MIB for k in range(6)], False),  # 60% of the first, but 60 MiB: under 64 MiB
    ([100 * MIB + k * 13 * MIB for k in range(6)], True),  # 65 MiB
    ([40 * GIB + k * 100 * MIB for k in range(6)], False),  # under 5% of the first
    ([2 * GIB] + [GIB + k * 100 * MIB for k in range(6)], True),  # a drop starts the trend again
])
def test_memory_growth(values, found):
    hits = by_kind(readings(values, total=80 * GIB))
    assert ("memory_growth" in hits) is found


def test_memory_pressure_at_95_percent_of_the_device():
    assert "memory_pressure" not in by_kind(readings([int(0.9499 * 16 * GIB)], key="reserved", total=16 * GIB))
    hit = by_kind(readings([0, int(0.95 * 16 * GIB) + 1], key="reserved", total=16 * GIB))["memory_pressure"]
    assert hit["step"] == 19 and hit["fraction"] == pytest.approx(0.95)
    assert "memory_pressure" not in by_kind(readings([GIB], key="reserved"))  # no total


@pytest.mark.parametrize("cuda_part, found", [
    ({"alloc_retries": 0, "ooms": 0}, False), ({"alloc_retries": 1}, True), ({"ooms": 1}, True), ({}, False),
])
def test_allocator_retries_or_ooms(cuda_part, found):
    assert ("allocator_retries" in by_kind([record(9, system={"cuda": cuda_part})])) is found


def nvml_run(*parts, attempt=1):
    return [record(10 * i + 9, attempt=attempt, system={"nvml": part}) for i, part in enumerate(parts)]


def test_hardware_errors_from_a_rise_in_uncorrected_ecc_or_any_xid():
    assert "hardware_errors" not in by_kind(nvml_run({"ecc_uncorrected": 5}, {"ecc_uncorrected": 5}))
    assert by_kind(nvml_run({"ecc_uncorrected": 5}, {"ecc_uncorrected": 6}))["hardware_errors"]["step"] == 19
    assert by_kind(nvml_run({"ecc_uncorrected": 0, "xid": [79]}))["hardware_errors"]["xid"] == [79]
    # The baseline is the first reading of the attempt, so a resumed process starts again.
    resumed = nvml_run({"ecc_uncorrected": 1}) + nvml_run({"ecc_uncorrected": 4}, attempt=2)
    assert "hardware_errors" not in by_kind(resumed)
    # A metric that could not be read at first does not become the baseline later.
    assert "hardware_errors" not in by_kind(nvml_run({}, {"ecc_uncorrected": 5}, {"ecc_uncorrected": 5}))


@pytest.mark.parametrize("reasons, found", [
    (["sw_power_cap"], False), (["applications_clocks", "sync_boost"], False), (["sw_thermal"], True),
    (["hw_thermal"], True), (["hw_slowdown"], True), (["hw_power_brake"], True), ([], False),
])
def test_gpu_throttled_ignores_the_power_cap(reasons, found):
    assert ("gpu_throttled" in by_kind(nvml_run({"throttle": reasons}))) is found


def waits(shares, attempt=1, start=1):
    return [record(start + i, attempt=attempt, data_wait=share * 0.2, step_time=0.2) for i, share in enumerate(shares)]


def test_input_starvation_needs_a_sustained_share():
    assert "input_starvation" not in by_kind(waits([0.5] * 19))  # not a full window
    hit = by_kind(waits([0.5] * 20))["input_starvation"]
    assert (hit["step"], hit["rule"], hit["median_share"]) == (20, "share", pytest.approx(0.5))
    spikes = [0.9 if i % 3 == 0 else 0.05 for i in range(200)]  # one slow step in three
    assert "input_starvation" not in by_kind(waits(spikes))
    assert "input_starvation" not in by_kind(waits([0.29] * 100))


def test_input_starvation_by_a_rise_over_the_first_window():
    hit = by_kind(waits([0.05] * 20 + [0.26] * 11))["input_starvation"]
    assert hit["rule"] == "rise" and hit["baseline_share"] == pytest.approx(0.05) and hit["step"] == 31
    assert "input_starvation" not in by_kind(waits([0.05] * 20 + [0.24] * 40))


def test_the_resume_edge_would_be_incidents_if_the_attempts_were_joined():
    joined = [{**r, "attempt": 1} for r in tensorquill.evidence.resolve_attempts(edges()["resume"])]
    assert set(by_kind(joined)) == {"memory_growth", "input_starvation"}
    assert by_kind(edges()["resume"]) == {}


def test_input_starvation_starts_again_on_a_resume():
    resumed = waits([0.5] * 15) + waits([0.5] * 10, attempt=2, start=10)
    assert "input_starvation" not in by_kind(resumed)


def test_unavailable_readings_are_listed_once_per_check():
    notes: list = []
    detect_system_incidents(edges()["unavailable"], notes)
    assert notes == [
        {"check": "cuda_memory", "kinds": ["memory_growth", "memory_pressure", "allocator_retries"],
         "reason": "the model is on cpu, not a CUDA device", "step": 9},
        {"check": "nvml", "kinds": ["hardware_errors", "gpu_throttled"],
         "reason": "pynvml is not installed (pip install nvidia-ml-py)", "step": 9},
    ]


def test_odd_values_are_skipped_not_misread():
    found = by_kind(edges()["odd"])
    assert sorted(found) == ["allocator_retries", "gpu_throttled", "hardware_errors"]
    assert found["gpu_throttled"]["step"] == 11 and found["gpu_throttled"]["temperature_c"] == 1.0
    assert found["hardware_errors"]["step"] == 12  # 0 at step 5 is the baseline; 2.0 and False are not counts
    assert found["allocator_retries"]["step"] == 13


# --- across ranks --------------------------------------------------------------------------


def test_memory_imbalance_names_the_ranks_well_above_the_median():
    even = {r: {"step": 99, "peak_allocated": 20 * GIB + r * 100 * MIB} for r in range(8)}
    assert detect_memory_imbalance(even) is None
    uneven = {**even, 5: {"step": 99, "peak_allocated": 30 * GIB}}
    hit = detect_memory_imbalance(uneven)
    assert hit["kind"] == "memory_imbalance" and hit["ranks"] == [5] and hit["step"] == 99
    assert hit["peak_allocated"]["5"] == 30 * GIB
    # Small models: 25% above the median but under 1 GiB is not reported.
    small = {0: {"peak_allocated": GIB}, 1: {"peak_allocated": GIB}, 2: {"peak_allocated": 2 * GIB - MIB}}
    assert detect_memory_imbalance(small) is None
    assert detect_memory_imbalance({0: {"peak_allocated": 9 * GIB}, 1: None, 2: {}}) is None  # one rank with a reading
    assert "memory imbalance" not in tensorquill.incident_summary({"incidents": [hit]}).lower()
    assert "GPU memory uneven across ranks" in tensorquill.incident_summary({"incidents": [hit]})


def test_the_all_ranks_view_reports_memory_imbalance_from_each_ranks_report(tmp_path):
    from tensorquill import evidence

    reports = {}
    for rank in range(4):
        peak = (30 if rank == 2 else 20) * GIB
        reports[rank] = {"state": "current", "report": {
            "run_id": "r", "rank": rank, "last_step": 99, "incidents": [], "memory": {"step": 99, "peak_allocated": peak},
        }, "stored": (1, 99), "reason": "current"}
    out = evidence._all_ranks_evidence(tmp_path, {}, reports, None, traincheck=None, flight_recorder=None,
                                       flight_recorder_prefix=None)
    assert [i["kind"] for i in out["run_incidents"]] == ["memory_imbalance"]
    assert out["run_incidents"][0]["ranks"] == [2]
    json.dumps(out)


def test_constants_match_the_server():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "server" / "src" / "detect" / "system.rs").read_text()
    for name in ("MEMORY_GROWTH_READINGS", "MEMORY_GROWTH_FRACTION", "MEMORY_RESERVED_FRACTION", "INPUT_WINDOW",
                 "INPUT_SHARE", "INPUT_RISE"):
        assert f"pub const {name}: " in source and f"= {getattr(es, name)};" in source, name
    assert "pub const MEMORY_GROWTH_BYTES: i64 = 64 * 1024 * 1024;" in source and es.MEMORY_GROWTH_BYTES == 64 * 1024 * 1024
    assert all(f'"{reason}"' in source for reason in es.THROTTLE_REPORTED)
