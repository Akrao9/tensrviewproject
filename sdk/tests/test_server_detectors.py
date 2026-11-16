"""The server's Rust detectors against the Python reference, through the real binary.

Records are streamed to `tensorquill-server`, which runs its detectors over every part it
stores and writes `runs/<run_id>/rank=<rank>/incidents.json`. After each stretch of the run
that file must hold what `build_evidence` gives on the same records, for the kinds the
server has (`PORTED` below): steps, layers, experts and names exactly, numbers to 1e-12.
`job_stalled` and `rank_silent` are in the run's `watchdog.json`; `test_flight_e2e.py` and
`test_two_ranks_e2e.py` compare those with Python.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill.flight import send_file  # noqa: E402

from test_flight_e2e import BINARY, Server  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

ATTEMPT_KINDS = {"repeated_steps", "resume_lr_change"}  # these need more than one attempt
# Read from the `data` and `checkpoint` fields; their own fixtures are in test_data_server.py.
DATA_KINDS = {"resume_data_mismatch", "resume_incomplete", "data_stalled", "epoch_order_repeated"}
PORTED = DATA_KINDS | {
    "moe_imbalance", "starved_experts", "grad_spike", "stalled_weights", "loss_scale_trouble", "activation_spike",
    "activation_drift", "dead_activations", "non_finite", "replica_drift",
} | ATTEMPT_KINDS
RUN = "detect"
FAST = ("--flush-steps", "50", "--flush-seconds", "0.2")


def differences(python, rust, path="incidents"):
    """Where two JSON values differ. A NaN or Inf in Python's evidence is `null` in the
    server's file, which is strict JSON."""
    # Read back from Parquet, Python has the strings the capture writes for them.
    if (isinstance(python, float) and not math.isfinite(python)) or (isinstance(python, str) and python in ("nan", "inf", "-inf")):
        return [] if rust is None else [f"{path}: {python!r} != {rust!r}"]
    if isinstance(python, bool) or isinstance(rust, bool):
        return [] if python is rust else [f"{path}: {python!r} != {rust!r}"]
    if isinstance(python, (int, float)) and isinstance(rust, (int, float)):
        same = python == rust if isinstance(python, int) and isinstance(rust, int) else math.isclose(
            python, rust, rel_tol=1e-12, abs_tol=0.0
        )
        return [] if same else [f"{path}: {python!r} != {rust!r}"]
    if isinstance(python, dict) and isinstance(rust, dict):
        out = [f"{path}: keys differ by {sorted(set(python) ^ set(rust))}"] if set(python) != set(rust) else []
        return out + [d for key in set(python) & set(rust) for d in differences(python[key], rust[key], f"{path}.{key}")]
    if isinstance(python, (list, tuple)) and isinstance(rust, list):
        if len(python) != len(rust):
            return [f"{path}: {len(python)} items != {len(rust)}"]
        return [d for i, (a, b) in enumerate(zip(python, rust)) for d in differences(a, b, f"{path}[{i}]")]
    return [] if python == rust else [f"{path}: {python!r} != {rust!r}"]


def reference(records):
    """What Python reports for the kinds the server has."""
    return [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in PORTED]


def resumes_not_checked(records):
    """The resumes Python could not compare, which the server lists the same way."""
    return [n for n in tensorquill.build_evidence(records).get("not_checked", []) if n["check"] == "resume_lr"]


def flagged(records):
    """The MoE samples Python flags: how many, and the last step."""
    steps = [
        s["step"] for s in tensorquill.build_evidence(records)["samples"]
        if not s.get("missing") and s["routed_tokens"] > 0 and (s["overloaded"] or s["dead"])
    ]
    return {"samples": len(steps), "last_step": max(steps, default=None)}


class Stream:
    """One rank's `.jsonl` file, sent to a server a stretch at a time."""

    def __init__(self, tmp_path: Path, *server_args: str, run: str = RUN, server: Server | None = None) -> None:
        self.store = tmp_path / "store"
        self.run = run
        self.file = tmp_path / f"{run}.rank0.jsonl"
        self.file.write_text("")
        self.server_args = server_args
        self.server = server or Server(self.store, 0, *server_args)
        self.sent: list[dict] = []

    def restart(self) -> None:
        assert self.server.terminate() == 0
        self.server = Server(self.store, 0, *self.server_args)

    @property
    def incidents_file(self) -> Path:
        return self.store / "runs" / self.run / "rank=0" / "incidents.json"

    def send(self, records: list[dict]) -> dict:
        """Append `records`, send them, and return the server's document once it covers them."""
        with self.file.open("a", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record) + "\n")
        self.sent.extend(records)
        assert send_file(self.file, self.server.endpoint, self.run, timeout=30.0)
        deadline = time.monotonic() + 10
        while True:
            # The file is rewritten after the part is acknowledged, so it may lag a moment.
            try:
                document = json.loads(self.incidents_file.read_text(encoding="utf-8"))
                if document["last_step"] == records[-1]["step"] and document["attempt"] == records[-1].get("attempt", 0):
                    return document
            except (FileNotFoundError, ValueError):
                pass
            assert time.monotonic() < deadline, "the server did not report on the rows it stored"
            time.sleep(0.05)


def record(step: int, attempt: int = 1, run: str = RUN, **values) -> dict:
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step, "ts": float(step), **values}


def wobble(step: int, salt: int = 0) -> float:
    """A repeatable value between 0.9 and 1.1."""
    return 0.9 + ((step * 37 + salt * 11) % 21) / 100


def everything(steps: int = 300) -> list[dict]:
    """A run in which each kind that one attempt can show happens once, among the things
    that must not confuse it: a router that is missing on some steps, one that appears late,
    a run that starts clipping after 20 steps, loss-scaler skips, NaN values, a module with
    no statistics, weight checks that find nothing and a replica check within tolerance."""
    out = []
    for step in range(steps):
        # Router a: top-8 of 32. Experts 9-15 take nearly every token from step 60 to 199.
        a = [1024] + [231] * 31
        if 60 <= step < 200:
            a = [1024] + [8] * 31
            for expert in range(9, 16):
                a[expert - 1] = 1000
        a[1] += 8192 - sum(a)
        moe = {"a": a}
        stats = {}
        # Router b: top-1 of 8. Expert 8 gets one token in 2,101 throughout: starved.
        if step % 17 != 5:
            moe["b"] = [300 + (step * (e + 3)) % 5 for e in range(7)] + [1]
            if step % 3 == 0:
                stats["b"] = {"diversity": 0.875 + (step % 5) / 1000, "entropy": 1.9}
        # Router c appears at step 100; from step 250 one expert takes everything.
        if step >= 100:
            moe["c"] = [1000, 0, 0, 0] if step >= 250 else [250 + (step % 3), 250, 249, 251]
        grad = wobble(step)
        norms = {"w1": 0.6 * wobble(step, 1), "w2": 0.4 * wobble(step, 2)}
        if 147 <= step <= 150:  # w2 builds up to the spike at step 150
            norms["w2"] *= 1.5 ** (step - 146) * 3
        # Step 25 is five steps after clipping began. Python reads the clipped runs' norm
        # from `clip_grad_norm` alone, so it has five values of history there and says
        # nothing; a stream that kept the 20 steps before clipping would call it a spike.
        if step in (25, 150, 151, 220):
            grad *= 8
            norms["w1"] *= 9 if step == 220 else 1
        values = {
            "moe": moe,
            "moe_top_k": {"a": 8, "b": 1, "c": 1},
            "grad_norm": grad,
            "grad_norms": norms,
            "scalars": {"train/loss": 2.0 - step / 1000, "sometimes_nan": float("nan") if step % 7 == 0 else 1.0},
            "layers": {
                "blk0": {"abs_max": (21.0 if step == 130 else 3.0) * wobble(step, 3), "mean": 0.01, "std": (4.0 if step >= 180 else 1.0) * wobble(step, 4),
                         "zero_frac": 0.0},
                "blk1": {"abs_max": (40.0 if step == 130 else 3.0) * wobble(step, 5), "mean": 0.0, "std": wobble(step, 6),
                         "zero_frac": 0.5 if step >= 210 else 0.01},
                "blk2": {"abs_max": 1.0, "mean": 0.0, "std": None, "zero_frac": 0.0},
            },
        }
        if stats:
            values["moe_stats"] = stats
        if step >= 20:
            values["clip_grad_norm"] = grad * 1.25  # read before clipping, so the spike shows here
        if step == 40:
            values["grad_norm"] = values["clip_grad_norm"] = float("nan")
        if step == 90:  # a skipped step with a huge but finite norm must not count as a spike
            values.update(step_skipped=True, grad_norm=grad * 9, clip_grad_norm=grad * 9)
        if step == 95:
            values.update(step_skipped=True, grad_norm=float("inf"), clip_grad_norm=float("inf"))
        if step == 40:  # the capture writes a NaN statistic as this string
            values["layers"]["blk2"]["mean"] = "nan"
        # A loss scaler from step 60: twelve skipped steps in a row from 262, the tenth at 271.
        if step >= 60 and "step_skipped" not in values:
            values.update(step_skipped=262 <= step < 274, grad_scale=65536.0 / (2 ** max(0, min(step, 273) - 261)))
        # A weight check every five steps. w1 is unchanged on two checks, then changes, then
        # is unchanged on two more: never three in a row. w2 stops changing at step 230, and
        # the check at 235 has no gradient for it, which neither counts nor starts it over.
        if step % 5 == 0:
            values["weights_unchanged"] = (["w1"] if step in (50, 55, 65, 70) else []) + (["w2", "frozen"] if step >= 230 else [])
            values["optimizer_ownership"] = {"state": "known", "unheld": ["w2", "frozen"], "unheld_count": 2}
            if step == 235:
                del values["grad_norms"]["w2"]
        # A replica check every 25 steps: inside the tolerance until step 175, where one
        # parameter is apart (an older record, with two values) and one is not finite.
        if step % 25 == 0:
            values["replica_spread"] = {"w1": [0.0, 0.0, 0.0], "w2": [5e-7, 0.0, 1e-9]}
            if step == 175:
                values["replica_spread"] = {"w1": [2e-6, 0.0], "w2": [0.0, float("nan"), 0.0], "w3": [1e-7, 1e-7, 1e-7]}
        out.append(record(step, **values))
    return out


def test_the_fixture_has_every_kind_the_server_reports():
    found = reference(everything())
    assert {i["kind"] for i in found} == PORTED - ATTEMPT_KINDS - DATA_KINDS
    by_kind = {i["kind"]: i for i in found}
    assert (by_kind["stalled_weights"]["step"], by_kind["stalled_weights"]["parameters"]) == (245, ["w2"])
    assert by_kind["stalled_weights"]["optimizer"] == {"state": "known", "not_held": ["w2"], "not_held_count": 1, "held_count": 0}
    assert (by_kind["loss_scale_trouble"]["step"], by_kind["loss_scale_trouble"]["reason"]) == (271, "skips")
    assert by_kind["non_finite"] == {**by_kind["non_finite"], "step": 40, "layers": ["blk2"],
                                     "gradients": ["<total>", "<total before clipping>"]}
    assert by_kind["replica_drift"]["step"] == 175 and set(by_kind["replica_drift"]["parameters"]) == {"w1", "w2"}
    assert (by_kind["moe_imbalance"]["step"], by_kind["moe_imbalance"]["overloaded"]) == (60, list(range(9, 16)))
    assert by_kind["grad_spike"]["step"] == 150 and by_kind["grad_spike"]["spike_steps"] == [150, 151, 220]
    assert by_kind["grad_spike"]["parameters"][0]["name"] == "w2"
    assert by_kind["grad_spike"]["parameters"][0]["rising_since"] < 150
    assert by_kind["activation_spike"]["step"] == 130
    # blk0 shows it first but blk1 has the larger ratio: the first block in the model leads.
    spike_modules = by_kind["activation_spike"]["modules"]
    assert [m["module"] for m in spike_modules] == ["blk0", "blk1"] and spike_modules[0]["ratio"] < spike_modules[1]["ratio"]
    starved = {(e["layer"], e["expert"]) for e in by_kind["starved_experts"]["experts"]}
    assert (2, 8) in starved and (1, 20) in starved  # router b's expert 8, and router a's after its collapse
    assert math.isnan(by_kind["grad_spike"]["scalars"]["sometimes_nan"]["value"]) is False  # 150 % 7 != 0
    assert by_kind["moe_imbalance"]["scalars"]["train/loss"]["median_before"] is not None


def test_the_server_reports_what_python_reports_after_every_stretch(tmp_path):
    records = everything()
    stream = Stream(tmp_path, *FAST)
    sent = 0
    kinds_seen = []
    # Stretches that end inside, at and between the 50-step parts and each incident.
    for upto in (7, 19, 21, 41, 45, 61, 100, 131, 150, 152, 176, 215, 246, 251, 272, 300):
        document = stream.send(records[sent:upto])
        sent = upto
        expected = reference(records[:upto])
        assert differences(expected, document["incidents"]) == [], f"after {upto} steps"
        # The collapse lasts 140 steps, nearly three windows: it must stay flagged to its
        # end at step 199 and no further, as it does in Python's samples.
        assert document["moe_flagged"] == flagged(records[:upto]), f"after {upto} steps"
        assert document["steps"] == upto and document["source"] == "tensorquill-server"
        kinds_seen.append(len(expected))
    assert kinds_seen[0] == 0 and kinds_seen[-1] == len(PORTED - ATTEMPT_KINDS - DATA_KINDS)  # nothing early, everything by the end
    # One rebuild, for the first part after the server started; every later part was added
    # to the state that was kept.
    assert document["rebuilds"] == 1
    assert set(document["checked"]) == PORTED
    # What the server cannot check is listed the way Python's evidence lists it.
    assert [(n["check"], n["kinds"]) for n in document["not_checked"]] == [
        ("flight_recorder", ["collective_hang", "collective_desync"]), ("traincheck", ["traincheck_violation"])]
    assert document["thresholds"]["diversity_drop"] == 0.6 and document["thresholds"]["drift_tolerance"] == 1e-6
    # The server's own Parquet gives Python the same answer.
    from_store = [i for i in tensorquill.build_evidence(stream.store / "runs" / RUN, rank=0)["incidents"] if i["kind"] in PORTED]
    assert differences(from_store, document["incidents"]) == []
    assert stream.server.terminate() == 0


def test_a_restart_in_the_middle_of_a_collapse_changes_nothing(tmp_path):
    # At step 120 router a has been collapsed since step 60, so its baseline is 60 steps
    # old and held; the restarted server has to rebuild that from the store, not from the
    # last window.
    records = everything()
    stream = Stream(tmp_path, *FAST)
    before = stream.send(records[:120])
    stream.restart()
    after = stream.send(records[120:260])
    assert differences(reference(records[:260]), after["incidents"]) == []
    assert after["steps"] == 260 and before["steps"] == 120
    assert (before["rebuilds"], after["rebuilds"]) == (1, 1)  # one per server, the second from 120 stored steps
    # The collapse is still the one found at step 60, and the recovery at 200 was seen:
    # a baseline rebuilt from the last window alone would have lost both.
    moe = next(i for i in after["incidents"] if i["kind"] == "moe_imbalance")
    assert moe["step"] == 60
    assert after["moe_flagged"] == flagged(records[:260]) and before["moe_flagged"] == flagged(records[:120])
    assert after["moe_flagged"]["samples"] >= 140
    assert stream.server.terminate() == 0


def test_a_resume_takes_back_the_incidents_of_the_steps_it_rolled_back(tmp_path):
    def run(steps, attempt, spikes=()):
        return [
            record(step, attempt, grad_norm=wobble(step) * (9 if step in spikes else 1),
                   moe={"r": [200, 210, 190, 200]}, moe_top_k={"r": 1})
            for step in steps
        ]

    stream = Stream(tmp_path, *FAST)
    first = run(range(150), 1, spikes={120})
    document = stream.send(first)
    assert [i["step"] for i in document["incidents"] if i["kind"] == "grad_spike"] == [120]

    # The job resumed from the checkpoint at step 100; this time nothing spiked.
    second = run(range(100, 180), 2)
    document = stream.send(second)
    assert document["incidents"] == [] and reference(first + second) == []
    assert document["steps"] == 180 and document["attempt"] == 2
    assert document["rebuilds"] == 2  # the rollback cannot be undone in place

    # A third attempt from step 170 that does spike, at step 175.
    third = run(range(170, 200), 3, spikes={175})
    document = stream.send(third)
    assert differences(reference(first + second + third), document["incidents"]) == []
    assert [i["step"] for i in document["incidents"]] == [175]
    assert document["rebuilds"] == 3
    assert stream.server.terminate() == 0


def one_parameter(steps: int = 400) -> list[dict]:
    """What only a parameter's own history shows. Gradients: `small` holds about 0.2% of the
    total norm and is 100 times its size from step 250, which moves the total 1.02 times; before
    that it is 40 times its size for one step and for two steps in a row, which is ordinary
    (the two are still inside its window at step 250, and must not be in its median);
    `a_tiny` is 60 times its size from the same step, so the two are listed by ratio, not by
    name; `other` is high for exactly three steps much later, after the incident is already made.
    Weights, checked every 10 steps: `norm_a` and `norm_b` never change (low-precision weights
    near 1.0), `frozen` trains and then stops at step 200, `late` stops at step 300,
    `flicker` is unchanged on two checks at a time, never three, and `a_idle` never changes but
    has no gradient until step 200, so it stalls on the same check as `frozen` and is listed
    after it although it comes first in the record."""
    out = []
    for step in range(steps):
        norms = {"a_idle": 0.3 * wobble(step, 9) if step >= 200 else 0.0, "a_tiny": 0.01 * wobble(step, 10),
                 # No two of `small`'s norms are equal, and of the two that the excursion at steps 230
                 # and 231 would push out of its window one is low and one high, so letting the
                 # excursion into the median changes the median at step 250.
                 "big": 10.0 * wobble(step, 1), "small": 0.02 * wobble(step, 27) * (1 + step / 9973), "other": 0.5 * wobble(step, 3),
                 "norm_a": 0.1 * wobble(step, 4), "norm_b": 0.1 * wobble(step, 5), "frozen": 1.0 * wobble(step, 6),
                 "late": 1.0 * wobble(step, 7), "flicker": 1.0 * wobble(step, 8)}
        if step == 60 or step in (230, 231) or step >= 250:
            norms["small"] *= 40.0 if step < 250 else 100.0
        if step >= 250:
            norms["a_tiny"] *= 60.0
        if 330 <= step < 333:
            norms["other"] *= 30.0
        values = {"grad_norms": norms, "grad_norm": math.sqrt(sum(v * v for v in norms.values()))}
        if step % 4 != 2:  # logged on most steps: not on step 250, where the spike is dated, but on 252, which makes it
            values["scalars"] = {"train/loss": 3.0 - step / 1000}
        if step % 10 == 9:
            unchanged = ["a_idle", "norm_a", "norm_b"]
            if step >= 200:
                unchanged.append("frozen")
            if step >= 300:
                unchanged.append("late")
            if step % 30 != 29:
                unchanged.append("flicker")
            if step > 9:  # the first check has nothing to compare with
                values["weights_unchanged"] = unchanged
            values["optimizer_ownership"] = {"state": "known", "unheld": ["late"], "unheld_count": 1}
            values["weight_dtypes"] = {"bfloat16": 10}
        out.append(record(step, **values))
    return out


def test_the_one_parameter_fixture_shows_what_the_totals_cannot():
    records = one_parameter()
    by_kind = {i["kind"]: i for i in reference(records)}
    assert set(by_kind) == {"grad_spike", "stalled_weights"}
    spike = by_kind["grad_spike"]
    assert (spike["step"], spike["trigger"], spike["sustained"]) == (250, "parameter", 3)
    assert [p["name"] for p in spike["parameters"]] == ["small", "a_tiny"] and spike["parameters"][0]["rising_since"] == 250
    assert 80 < spike["parameters"][0]["ratio"] < 125 and 48 < spike["parameters"][1]["ratio"] < 75
    assert spike["ratio"] < 1.3 and spike["spike_count"] == 0
    # The logged values are those of the step the incident is dated at, not of the step that made it.
    assert "scalars" not in spike  # nothing was logged on step 250
    # Not before it has lasted three recorded steps, and never for the one- and two-step excursions.
    assert [i["kind"] for i in reference(records[:252])] == ["stalled_weights"]
    assert [i["step"] for i in reference(records[:253]) if i["kind"] == "grad_spike"] == [250]
    stalled = by_kind["stalled_weights"]
    assert (stalled["step"], stalled["parameters"], stalled["stalled"]) == (39, ["norm_a", "norm_b"], 2)
    assert stalled["weight_dtypes"] == {"bfloat16": 10}
    assert [(o["step"], o["parameters"], o["trained_before"]) for o in stalled["onsets"]] == [
        (39, ["norm_a", "norm_b"], 0), (229, ["frozen", "a_idle"], 1), (329, ["late"], 1)]
    assert stalled["onsets"][2]["optimizer"] == {"state": "known", "not_held": ["late"], "not_held_count": 1, "held_count": 0}
    assert stalled["onsets"][1]["optimizer"]["not_held_count"] == 0
    assert (stalled["stalled_total"], stalled["onsets_omitted"]) == (5, 0)  # `flicker` never makes three in a row


def test_one_parameters_spike_and_each_stall_onset_match_python_after_every_stretch(tmp_path):
    records = one_parameter()
    stream = Stream(tmp_path, *FAST)
    sent = 0
    # Stretches that end inside each excursion, on the step that establishes the incident and the
    # one before it, between two onsets, and after the later three-step excursion.
    for upto in (40, 61, 231, 233, 240, 251, 252, 253, 300, 331, 334, 400):
        document = stream.send(records[sent:upto])
        sent = upto
        assert differences(reference(records[:upto]), document["incidents"]) == [], f"after {upto} steps"
    assert document["thresholds"]["spike_sustain"] == 3
    assert document["rebuilds"] == 1
    assert stream.server.terminate() == 0


def test_a_restart_inside_an_excursion_and_between_onsets_changes_nothing(tmp_path):
    # The server stops with `small` two steps into its excursion and one stall onset reported: the
    # replay has to rebuild the run of high steps, the median it is measured against (which the
    # excursion's own norms stay out of) and which parameters were already listed.
    records = one_parameter()
    stream = Stream(tmp_path, *FAST)
    before = stream.send(records[:252])
    assert [i["kind"] for i in before["incidents"]] == ["stalled_weights"]
    stream.restart()
    after = stream.send(records[252:400])
    assert differences(reference(records), after["incidents"]) == []
    assert [i["step"] for i in after["incidents"] if i["kind"] == "grad_spike"] == [250]
    assert stream.server.terminate() == 0


def test_a_parameter_is_judged_from_its_eleventh_recorded_step(tmp_path):
    # Ten norms are the history the rule asks for, so an excursion from step 10 is dated at step 10.
    records = [
        record(step, grad_norms={"big": 10.0 * wobble(step), "small": 0.02 * wobble(step, 2) * (100.0 if step >= 10 else 1.0)},
               grad_norm=10.0 * wobble(step))
        for step in range(20)
    ]
    spike = next(i for i in reference(records) if i["kind"] == "grad_spike")
    assert (spike["step"], spike["trigger"], spike["baseline"]) == (10, "parameter", pytest.approx(10.0, rel=0.11))
    stream = Stream(tmp_path, *FAST)
    for upto in (12, 13, 20):
        document = stream.send(records[len(stream.sent):upto])
        assert differences(reference(records[:upto]), document["incidents"]) == [], f"after {upto} steps"
    assert stream.server.terminate() == 0


def test_excursions_that_last_to_the_same_record_from_different_steps_give_the_earliest(tmp_path):
    # `a` is high from step 30 but has no gradient on step 31; `b` is high from step 31. Both have
    # lasted three recorded steps at step 33: the incident is step 30 and names `a` alone.
    records = []
    for step in range(40):
        norms = {"big": 10.0 * wobble(step), "b": 0.02 * wobble(step, 1) * (90.0 if step >= 31 else 1.0),
                 "a": 0.02 * wobble(step, 2) * (50.0 if step >= 30 else 1.0)}
        if step == 31:
            del norms["a"]
        records.append(record(step, grad_norms=norms, grad_norm=10.0 * wobble(step), scalars={"loss": float(step)}))
    spike = next(i for i in reference(records) if i["kind"] == "grad_spike")
    assert (spike["step"], [p["name"] for p in spike["parameters"]]) == (30, ["a"])
    # Dated at step 30 and made at step 33: the logged values are step 30's.
    assert spike["scalars"]["loss"] == {"value": 30.0, "median_before": 14.5}
    assert reference(records[:33]) == []
    stream = Stream(tmp_path, *FAST)
    for upto in (33, 34, 40):
        document = stream.send(records[len(stream.sent):upto])
        assert differences(reference(records[:upto]), document["incidents"]) == [], f"after {upto} steps"
    assert stream.server.terminate() == 0


def test_more_onsets_than_an_incident_lists_are_counted(tmp_path):
    # 25 parameters stop one after another, one per weight check: 20 onsets are listed, 5 counted.
    names = [f"w{i:02}" for i in range(25)]
    records = []
    for step in range(300):
        values = {"grad_norms": {name: 1.0 for name in names}, "grad_norm": 5.0}
        if step % 5 == 4 and step > 4:
            values["weights_unchanged"] = [name for i, name in enumerate(names) if step >= 100 + 5 * i]
        records.append(record(step, **values))
    stalled = next(i for i in reference(records) if i["kind"] == "stalled_weights")
    assert (len(stalled["onsets"]), stalled["onsets_omitted"], stalled["stalled_total"]) == (20, 5, 25)
    assert [o["parameters"] for o in stalled["onsets"][:2]] == [["w00"], ["w01"]]
    assert all(o["trained_before"] == 1 and o["optimizer"]["state"] == "unknown" for o in stalled["onsets"])
    stream = Stream(tmp_path, *FAST)
    for upto in (120, 215, 300):
        document = stream.send(records[len(stream.sent):upto])
        assert differences(reference(records[:upto]), document["incidents"]) == [], f"after {upto} steps"
    assert stream.server.terminate() == 0


def moe_edges(steps: int = 30) -> list[dict]:
    """Two MoE rules the long run does not reach: a top-k router that collapses before it
    has a baseline (attributed by share), and dead experts with no expert overloaded."""
    out = []
    for step in range(steps):
        early = [500] * 4 + [0] * 12 if 3 <= step < 6 else [125] * 16  # top-4 of 16
        quiet = [300, 301, 299, 300, 300, 300, 0, 0] if step >= 12 else [225] * 8  # top-1 of 8
        out.append(record(step, moe={"early": early, "quiet": quiet}, moe_top_k={"early": 4, "quiet": 1}))
    return out


def test_moe_rules_before_a_baseline_and_without_an_overloaded_expert(tmp_path):
    records = moe_edges()
    samples = {(s["step"], s["router"]): s for s in tensorquill.build_evidence(records)["samples"]}
    assert samples[(3, "early")]["overload_rule"]["rule"] == "collapse" and samples[(3, "early")]["overloaded"] == [1, 2, 3, 4]
    assert samples[(12, "quiet")]["dead"] == [7, 8] and samples[(12, "quiet")]["overloaded"] == []
    stream = Stream(tmp_path, *FAST)
    sent = 0
    for upto in (3, 4, 10, 13, 30):
        document = stream.send(records[sent:upto])
        sent = upto
        assert differences(reference(records[:upto]), document["incidents"]) == [], f"after {upto} steps"
        assert document["moe_flagged"] == flagged(records[:upto]), f"after {upto} steps"
    assert document["moe_flagged"] == {"samples": 3 + 18, "last_step": 29}
    assert stream.server.terminate() == 0


def test_the_gate_1_fixtures_give_the_same_incidents(tmp_path):
    # The seven bake-off fixtures (four MoE shapes, a gradient spike, an activation
    # outlier), each planted with one failure and made through the SDK's own pipeline.
    import sys

    bakeoff = Path(__file__).resolve().parents[2] / "experiments" / "moe-bakeoff"
    if not (bakeoff / "generate.py").exists():
        pytest.skip("the bake-off experiment is not in this checkout")
    sys.path.insert(0, str(bakeoff))
    try:
        import generate
    finally:
        sys.path.remove(str(bakeoff))
    server = Server(tmp_path / "store", 0, *FAST)
    for name, fixture in generate.BY_NAME.items():
        records = fixture.records()
        document = Stream(tmp_path, run=records[0]["run_id"], server=server).send(records)
        expected = reference(records)
        assert len(expected) == 1, name  # each fixture has its one planted failure
        assert differences(expected, document["incidents"]) == [], name
        assert document["moe_flagged"] == flagged(records), name
    assert server.terminate() == 0


def test_resumes_and_restarts_are_compared_with_the_attempts_before_them(tmp_path):
    def warmup(start):
        return lambda step: 1e-3 * min(1.0, (step - start + 1) / 100)

    def attempt(number, steps, lr):
        return [
            record(step, number, grad_norm=wobble(step), scalars={"train/loss": 2.0 - step / 1000 + number / 100},
                   **({} if lr is None else {"lr": [lr(step), 0.1 * lr(step)]}))
            for step in steps
        ]

    stream = Stream(tmp_path, *FAST)
    stretches = [
        ("the first attempt", attempt(1, range(150), warmup(0))),
        # Resumed at 100, which attempt 1 had passed: compared at the same step. Warmup began again.
        ("warmup starts over at a step the run had", attempt(2, range(100, 180), warmup(100))),
        # Resumed at 200, past everything stored: the history is continued along its slope.
        # A new value that is not the run's first is not reported, only listed with its numbers.
        ("a resume past the stored steps", attempt(3, range(200, 220), lambda step: 5e-4)),
        # The same, but the value is the run's first learning rate: that is reported.
        ("warmup starts over past the stored steps", attempt(4, range(230, 240), lambda step: 1e-5)),
        ("a resume that continues the schedule", attempt(5, range(235, 250), lambda step: 1e-5)),
        ("a resume with no learning rate", attempt(6, range(220, 245), None)),
        # The numbering starts over: the steps it shares with earlier attempts were recorded twice.
        # Resumed at 120 with what attempt 1 had there. Attempt 2 rolled that back and had
        # another value at 120, and it is attempt 2 the resume is compared with.
        ("a resume that matches only what was rolled back", attempt(7, range(120, 130), warmup(0))),
        ("a restart from step 0", attempt(8, range(0, 30), warmup(0))),
        ("the restart goes on", attempt(8, range(30, 60), warmup(0))),
        ("the restart passes the first resume", attempt(8, range(60, 120), warmup(0))),
        ("a restart from before the run began is not a resume", attempt(9, range(0, 10), lambda step: 3e-4)),
        # It goes on over steps that already count as repeated: they are not counted again.
        ("the second restart goes on", attempt(9, range(10, 20), lambda step: 3e-4)),
    ]
    rebuilds, documents = [], {}
    for name, records in stretches:
        document = stream.send(records)
        expected = reference(stream.sent)
        assert differences(expected, document["incidents"]) == [], name
        assert differences(resumes_not_checked(stream.sent), [n for n in document["not_checked"] if n["check"] == "resume_lr"]) == [], name
        rebuilds.append(document["rebuilds"])
        documents[name] = document
    # Every new attempt is read against the store; an attempt that goes on is not.
    assert rebuilds == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9, 9]
    by_kind = {i["kind"]: i for i in document["incidents"]}
    assert set(by_kind) == ATTEMPT_KINDS
    assert by_kind["repeated_steps"]["repeated"] == 120 and by_kind["repeated_steps"]["steps"] == list(range(20))
    assert [(c["step"], c["basis"], c["previous_attempt"], c["equals_first_lr"]) for c in by_kind["resume_lr_change"]["changes"]] == [
        (100, "same_step", 1, True), (120, "same_step", 2, False), (230, "extrapolated", 3, True)]
    # The restart rolled step 100 back, so the incident that names it had nothing logged
    # there; once the restart wrote step 100 itself, the incident shows what it logged; and
    # the last restart rolled that back again.
    def logged_at_the_resume(name):
        [incident] = [i for i in documents[name]["incidents"] if i["kind"] == "resume_lr_change"]
        return incident.get("scalars", {}).get("train/loss", {}).get("value")

    assert logged_at_the_resume("the restart goes on") is None
    assert logged_at_the_resume("the restart passes the first resume") == pytest.approx(2.0 - 0.1 + 0.08)
    assert logged_at_the_resume("the second restart goes on") is None
    reasons = [n["reason"] for n in document["not_checked"] if n["check"] == "resume_lr"]
    assert len(reasons) == 2 and "no learning rate recorded in the resumed attempt" in reasons
    # The server's own Parquet gives Python the same answer.
    from_store = tensorquill.build_evidence(stream.store / "runs" / RUN, rank=0)
    assert differences([i for i in from_store["incidents"] if i["kind"] in PORTED], document["incidents"]) == []
    assert stream.server.terminate() == 0


def edges() -> dict[str, list[dict]]:
    """Short runs, each for one rule the long run does not reach."""
    def run(name, steps, **by_step):
        out = []
        for step in range(steps):
            values = {"grad_norm": 1.0, "grad_norms": {"w1": 0.6, "w2": 0.4}, "layers": {"blk": {"std": 1.0, "abs_max": 2.0, "zero_frac": 0.0}}}
            for key, give in by_step.items():
                value = give(step)
                if value is not None:
                    values[key] = value
            out.append(record(step, run=name, **values))
        return out

    nan, inf = float("nan"), float("inf")
    return {
        # A parameter's own norm is named, and then the total is not.
        "bad-parameter": run("bad-parameter", 8, grad_norm=lambda s: nan if s == 5 else 1.0,
                             grad_norms=lambda s: {"w1": 0.6, "w2": -inf if s == 5 else 0.4}),
        # On a skipped step only the activations are read, and Inf there is still reported.
        "skipped": run("skipped", 8, step_skipped=lambda s: s in (2, 6), grad_norm=lambda s: inf if s in (2, 6) else 1.0,
                       grad_norms=lambda s: {"w1": inf if s == 2 else 0.6},
                       layers=lambda s: {"blk": {"std": 1.0, "abs_max": "inf" if s == 6 else 2.0, "zero_frac": 0.0, "mean": 0.0}}),
        # A loss scale under 1 before any run of skips; a NaN scale is not a low one.
        "low-scale": run("low-scale", 8, step_skipped=lambda s: s == 1, grad_scale=lambda s: nan if s == 2 else (0.5 if s >= 4 else 4.0)),
        # Stalled weights when the capture could not tell who holds the parameters, on checks
        # that are not on every step; a step with a non-finite total is left out, not a reset.
        "unknown-owner": run("unknown-owner", 12, weights_unchanged=lambda s: ["w1"] if s % 2 == 0 else None,
                             grad_norm=lambda s: nan if s == 4 else 1.0,
                             optimizer_ownership=lambda s: {"state": "unknown", "reason": "optimizer hooks are off (auto_step=False)"}),
        "no-owner-recorded": run("no-owner-recorded", 6, weights_unchanged=lambda s: ["w1", "w2"]),
        # A gradient of zero is not a stalled parameter.
        "zero-gradient": run("zero-gradient", 6, weights_unchanged=lambda s: ["w1"], grad_norms=lambda s: {"w1": 0.0, "w2": 0.4}),
        "apart": run("apart", 6, replica_spread=lambda s: {"w1": [0.0, 0.0, 3e-6 if s == 3 else 0.0]} if s % 3 == 0 else None),
    }


def test_rules_the_long_run_does_not_reach(tmp_path):
    runs = edges()
    found = {name: reference(records) for name, records in runs.items()}
    summary = {name: [(i["kind"], i["step"]) for i in incidents] for name, incidents in found.items()}
    assert summary == {
        "bad-parameter": [("non_finite", 5)], "skipped": [("non_finite", 6)], "low-scale": [("loss_scale_trouble", 4)],
        "unknown-owner": [("stalled_weights", 6), ("non_finite", 4)], "no-owner-recorded": [("stalled_weights", 2)],
        "zero-gradient": [], "apart": [("replica_drift", 3)],
    }
    assert found["bad-parameter"][0]["gradients"] == ["w2"] and found["skipped"][0]["layers"] == ["blk"]
    assert found["unknown-owner"][0]["optimizer"]["state"] == "unknown"
    server = Server(tmp_path / "store", 0, *FAST)
    for name, records in runs.items():
        document = Stream(tmp_path, run=name, server=server).send(records)
        assert differences(found[name], document["incidents"]) == [], name
    assert server.terminate() == 0


def test_no_detect_writes_no_incidents_file(tmp_path):
    stream = Stream(tmp_path, *FAST, "--no-detect")
    with stream.file.open("a", encoding="utf-8") as out:
        for item in everything(120):
            out.write(json.dumps(item) + "\n")
    assert send_file(stream.file, stream.server.endpoint, RUN, timeout=30.0)
    assert stream.server.terminate() == 0
    assert list((stream.store / "runs" / RUN / "rank=0").glob("attempt=*/part-*.parquet"))
    assert not stream.incidents_file.exists()


def _unpadded_layers_run(blocks=12, first=2, steps=60):
    """`model.layers.0` to `model.layers.11`, whose alphabetical order (0, 1, 10, 11, 2, ...) is not the model's.
    From step 40 a fault starting in block `first` shows in every later block, larger with depth."""
    out = []
    for step in range(steps):
        layers = {}
        for b in range(blocks):
            amax = 10.0 * (6 + b if step == 40 and b >= first else 1)
            layers[f"model.layers.{b}"] = {"mean": 0.0, "std": 1.0, "abs_max": amax, "zero_frac": 0.0}
        out.append(record(step, layers=layers))
    return out


def test_a_spike_leads_with_the_first_block_in_the_model_not_the_first_by_name(tmp_path):
    records = _unpadded_layers_run()
    expected = next(i for i in reference(records) if i["kind"] == "activation_spike")
    assert [m["module"] for m in expected["modules"]][:2] == ["model.layers.2", "model.layers.11"]
    stream = Stream(tmp_path, *FAST)
    try:
        document = stream.send(records)
    finally:
        stream.server.terminate()
    # What the server judged from the stream it was sent.
    served = next(i for i in document["incidents"] if i["kind"] == "activation_spike")
    assert [m["module"] for m in served["modules"]] == [m["module"] for m in expected["modules"]]
    assert differences([expected], [served]) == []
    # And what Python makes of the Parquet parts the server stored.
    stored = tensorquill.load_records(stream.store / "runs" / RUN / "rank=0")
    assert list(stored[0]["layers"]) == [f"model.layers.{b}" for b in range(12)]
    from_parts = next(i for i in tensorquill.build_evidence(stored)["incidents"] if i["kind"] == "activation_spike")
    assert [m["module"] for m in from_parts["modules"]][0] == "model.layers.2"
