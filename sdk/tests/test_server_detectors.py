"""The server's Rust detectors against the Python reference, through the real binary.

Records are streamed to `tensorquill-server`, which runs its detectors over every part it
stores and writes `runs/<run_id>/rank=<rank>/incidents.json`. After each stretch of the run
that file must hold what `build_evidence` gives on the same records, for the kinds the
server has (`PORTED` below): steps, layers, experts and names exactly, numbers to 1e-12.

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

PORTED = {"moe_imbalance", "starved_experts", "grad_spike", "activation_spike", "activation_drift", "dead_activations"}
RUN = "detect"
FAST = ("--flush-steps", "50", "--flush-seconds", "0.2")


def differences(python, rust, path="incidents"):
    """Where two JSON values differ. A NaN in Python's evidence is `null` in the server's
    file, which is strict JSON."""
    if isinstance(python, float) and math.isnan(python):
        return [] if rust is None else [f"{path}: nan != {rust!r}"]
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


def flagged(records):
    """The MoE samples Python flags: how many, and the last step."""
    steps = [
        s["step"] for s in tensorquill.build_evidence(records)["samples"]
        if not s.get("missing") and s["routed_tokens"] > 0 and (s["overloaded"] or s["dead"])
    ]
    return {"samples": len(steps), "last_step": max(steps, default=None)}


class Stream:
    """One rank's `.jsonl` file, sent to a server a stretch at a time."""

    def __init__(self, tmp_path: Path, *server_args: str) -> None:
        self.store = tmp_path / "store"
        self.file = tmp_path / f"{RUN}.rank0.jsonl"
        self.file.write_text("")
        self.server_args = server_args
        self.server = Server(self.store, 0, *server_args)
        self.sent: list[dict] = []

    def restart(self) -> None:
        assert self.server.terminate() == 0
        self.server = Server(self.store, 0, *self.server_args)

    @property
    def incidents_file(self) -> Path:
        return self.store / "runs" / RUN / "rank=0" / "incidents.json"

    def send(self, records: list[dict]) -> dict:
        """Append `records`, send them, and return the server's document once it covers them."""
        with self.file.open("a", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record) + "\n")
        self.sent.extend(records)
        assert send_file(self.file, self.server.endpoint, RUN, timeout=30.0)
        deadline = time.monotonic() + 10
        while True:
            # The file is rewritten after the part is acknowledged, so it may lag a moment.
            try:
                document = json.loads(self.incidents_file.read_text(encoding="utf-8"))
                if document["last_step"] == records[-1]["step"] and document["attempt"] == records[-1]["attempt"]:
                    return document
            except (FileNotFoundError, ValueError):
                pass
            assert time.monotonic() < deadline, "the server did not report on the rows it stored"
            time.sleep(0.05)


def record(step: int, attempt: int = 1, **values) -> dict:
    return {"schema_version": 1, "run_id": RUN, "rank": 0, "attempt": attempt, "step": step, "ts": float(step), **values}


def wobble(step: int, salt: int = 0) -> float:
    """A repeatable value between 0.9 and 1.1."""
    return 0.9 + ((step * 37 + salt * 11) % 21) / 100


def everything(steps: int = 300) -> list[dict]:
    """A run in which each kind the server has happens once, among the things that must not
    confuse it: a router that is missing on some steps, one that appears late, a run that
    starts clipping after 20 steps, a loss-scaler skip, NaN values and a module with no
    statistics."""
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
        if step in (150, 151, 220):
            grad *= 8
            norms["w1"] *= 9 if step == 220 else 1
        values = {
            "moe": moe,
            "moe_top_k": {"a": 8, "b": 1, "c": 1},
            "grad_norm": grad,
            "grad_norms": norms,
            "scalars": {"train/loss": 2.0 - step / 1000, "sometimes_nan": float("nan") if step % 7 == 0 else 1.0},
            "layers": {
                "blk0": {"abs_max": 3.0 * wobble(step, 3), "mean": 0.01, "std": (4.0 if step >= 180 else 1.0) * wobble(step, 4),
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
        if step == 90:
            values.update(step_skipped=True, grad_norm=float("inf"), clip_grad_norm=float("inf"))
        out.append(record(step, **values))
    return out


def test_the_fixture_has_every_kind_the_server_reports():
    found = reference(everything())
    assert {i["kind"] for i in found} == PORTED
    by_kind = {i["kind"]: i for i in found}
    assert (by_kind["moe_imbalance"]["step"], by_kind["moe_imbalance"]["overloaded"]) == (60, list(range(9, 16)))
    assert by_kind["grad_spike"]["step"] == 150 and by_kind["grad_spike"]["spike_steps"] == [150, 151, 220]
    assert by_kind["grad_spike"]["parameters"][0]["name"] == "w2"
    assert by_kind["grad_spike"]["parameters"][0]["rising_since"] < 150
    assert by_kind["activation_spike"]["step"] == 130
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
    for upto in (7, 19, 21, 45, 61, 100, 131, 150, 152, 215, 251, 300):
        document = stream.send(records[sent:upto])
        sent = upto
        expected = reference(records[:upto])
        assert differences(expected, document["incidents"]) == [], f"after {upto} steps"
        # The collapse lasts 140 steps, nearly three windows: it must stay flagged to its
        # end at step 199 and no further, as it does in Python's samples.
        assert document["moe_flagged"] == flagged(records[:upto]), f"after {upto} steps"
        assert document["steps"] == upto and document["source"] == "tensorquill-server"
        kinds_seen.append(len(expected))
    assert kinds_seen[0] == 0 and kinds_seen[-1] == len(PORTED)  # nothing early, everything by the end
    # One rebuild, for the first part after the server started; every later part was added
    # to the state that was kept.
    assert document["rebuilds"] == 1
    assert set(document["checked"]) == PORTED and "stalled_weights" in document["not_checked"]
    assert document["thresholds"]["diversity_drop"] == 0.6
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


def test_no_detect_writes_no_incidents_file(tmp_path):
    stream = Stream(tmp_path, *FAST, "--no-detect")
    with stream.file.open("a", encoding="utf-8") as out:
        for item in everything(120):
            out.write(json.dumps(item) + "\n")
    assert send_file(stream.file, stream.server.endpoint, RUN, timeout=30.0)
    assert stream.server.terminate() == 0
    assert list((stream.store / "runs" / RUN / "rank=0").glob("attempt=*/part-*.parquet"))
    assert not stream.incidents_file.exists()
