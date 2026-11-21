"""Writes `cases.json`, the shared cases of `WAVE_AF_CONTRACT.md`, section 1 and the one case of section 2.

    TENSORQUILL_SERVER_BIN=server/target/release/tensorquill-server \
        PYTHONPATH=sdk/src python sdk/tests/fixtures/wave_af/make_cases.py

Run once on the base of wave AF (`76a3b5d`), with a server binary built from it, before anything on
the page is built. Each case of section 1 is a one-record `.jsonl` file whose record is valid but
for one identity field, and has, for each boundary (`python` and `rust`: `build_evidence` on the
file with that engine; `cli_python` and `cli_rust`: `python -m tensorquill` on it; `detect`: the
server binary's offline `detect` given the row, where an Arrow column can hold the value):

- `expected`, written from the page's rule by `expected_for` below, never from what the base does;
- `today`, what the base answers, computed here and written as a literal: a record of the base,
  never recomputed by a test.

A case is a `pin` when `today` already satisfies `expected` at every boundary, and `new` otherwise.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tensorquill import divergence as dv
from tensorquill import evidence as ev
from tensorquill import flight

HERE = Path(__file__).resolve().parent
BIN = os.environ["TENSORQUILL_SERVER_BIN"]
GOOD = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0}
ABSENT = "<absent>"
VALUES: dict[str, Any] = {"absent": ABSENT, "null": None, "true": True, "whole_float": 1.0, "float": 1.5,
                          "text": "1", "negative": -3}
FIELDS = ("schema_version", "rank", "attempt", "step")


def record(field: str, value: Any) -> dict[str, Any]:
    made = dict(GOOD)
    if value == ABSENT and isinstance(value, str):
        del made[field]
    else:
        made[field] = value
    return made


def valid(field: str, value: Any) -> bool:
    """The page's table, "What is valid"."""
    if field == "run_id":
        return True  # left as it is
    if isinstance(value, str) and value == ABSENT:
        return field != "step"
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return value in (0, 1) if field == "schema_version" else value >= 0


def expected_for(field: str, value: Any, boundaries: list[str]) -> dict[str, Any]:
    if valid(field, value):
        return {b: {"reads": True} for b in boundaries}
    out: dict[str, Any] = {}
    for b in boundaries:
        if b in ("python", "rust"):
            out[b] = {"raises": "MalformedRecordError", "field": field}
        elif b.startswith("cli"):
            out[b] = {"exit": 2, "traceback": False, "names": field}
        else:
            out[b] = {"exit": 1, "names": field}
    return out


def api(path: Path, engine: str) -> dict[str, Any]:
    try:
        ev.build_evidence(path, detectors=engine)
    except BaseException as error:  # noqa: BLE001  whatever the base does is the record
        return {"raises": type(error).__name__, "text": str(error)[:120]}
    return {"reads": True}


def cli(path: Path, engine: str, work: Path) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, "-m", "tensorquill", str(path), "--detectors", engine, "--output", str(work / "out.json")],
        capture_output=True, text=True, timeout=120)
    if done.returncode == 0:
        return {"reads": True}
    last = (done.stderr.strip().splitlines() or [""])[-1]
    return {"exit": done.returncode, "traceback": "Traceback" in done.stderr, "last_line": last[:120]}


def detect(field: str, value: Any) -> dict[str, Any] | None:
    """The offline `detect` given the row directly; None where the column cannot hold the value."""
    import pyarrow as pa

    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        return None
    schema = flight.flight_schema()
    row = flight.record_to_row(GOOD)
    row[field] = value
    try:
        batch = pa.RecordBatch.from_pylist([row], schema=schema)
    except Exception:  # noqa: BLE001  Arrow will not build it, so no stream can carry it
        return None
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        writer.write_batch(batch)
    done = subprocess.run([BIN, "detect"], input=out.getvalue(), capture_output=True, timeout=60)
    if done.returncode:
        return {"exit": done.returncode, "last_line": done.stderr.decode().strip().splitlines()[-1][:160]}
    report = json.loads(done.stdout)
    return {"reads": True, "steps": report.get("steps"), "rank": report.get("rank")}


def satisfied(today: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("reads"):
        return today.get("reads") is True
    if "raises" in expected:
        return today.get("raises") == expected["raises"]
    return (today.get("exit") == expected["exit"] and not today.get("traceback", False)
            and expected["names"] in today.get("last_line", ""))


def main() -> None:
    cases: dict[str, dict[str, Any]] = {}
    named = {f"{f}_{n}": (f, v) for f in FIELDS for n, v in VALUES.items()}
    named["schema_version_99"] = ("schema_version", 99)
    named.update({"run_id_absent": ("run_id", ABSENT), "run_id_null": ("run_id", None), "run_id_number": ("run_id", 7)})
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        for name, (field, value) in named.items():
            made = record(field, value)
            path = work / f"{name}.jsonl"
            path.write_text(json.dumps(made) + "\n")
            today = {"python": api(path, "python"), "rust": api(path, "rust"),
                     "cli_python": cli(path, "python", work), "cli_rust": cli(path, "rust", work)}
            if field != "run_id" and not (isinstance(value, str) and value == ABSENT):
                found = detect(field, value)
                if found is not None:
                    today["detect"] = found
            expected = expected_for(field, value, list(today))
            pin = all(satisfied(today[b], expected[b]) for b in today)
            cases[name] = {"section": 1, "status": "pin" if pin else "new", "field": field,
                           "given": {"record": made}, "expected": expected, "today": today}

    # Section 2, EV-01: two runs that differ in nothing but the two provenance keys.
    def run(last_loss: float) -> list[dict[str, Any]]:
        return [{"schema_version": 1, "run_id": "a", "rank": 0, "attempt": 0, "step": step,
                 "scalars": {"loss": loss}, "layers_not_read": {"x": "why"}, "moe_source": {"router": "counted"}}
                for step, loss in ((0, 1.0), (1, last_loss))]

    given = {"records_a": run(1.0), "records_b": run(2.0)}
    result = dv.first_divergence(given["records_a"], given["records_b"])
    listed = sorted(k for k in ("layers_not_read", "moe_source")
                    if k in ((result.get("left_out") or {}).get("not_compared") or ()))
    assert listed, "the base no longer lists them: EV-01 is not reproduced, and the case is not new"
    cases["divergence_provenance_keys_are_not_series"] = {
        "section": 2, "status": "new", "given": given,
        "expected": {"not_compared_has": []}, "today": {"not_compared_has": listed}}

    base = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    document = {"contract": "WAVE_AF_CONTRACT.md", "base": base, "cases": cases}
    (HERE / "cases.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(len(cases), "cases:", sum(c["status"] == "pin" for c in cases.values()), "pins")


if __name__ == "__main__":
    main()
