"""Lane V7 (Nov 19, 2026): the server's detectors and their offline modes against Python, one
test per row of `OPEN_BUGS.md` the lane fixed (RS10, RS11, RS12, RS16, L7, C16's Rust reader).

The Rust half of each is also tested in `server/src/detect/lane_v7_tests.rs`; here the binary
is run as the SDK runs it. Needs `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`, or
a build in `server/target`); skipped without them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill.server_detect import server_report, server_run_report  # noqa: E402

from test_flight_e2e import BINARY  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "c16-batch-hash-versions.json"
C16_REASON = "the batch hashes before and after the resume are of different kinds"


def rec(step: int, **fields) -> dict:
    return {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": step, "ts": float(step), **fields}


def both(records: list[dict]) -> dict[str, dict]:
    return {engine: tensorquill.build_evidence(records, detectors=engine, server_binary=str(BINARY))
            for engine in ("python", "rust")}


def incident(evidence: dict, kind: str) -> dict:
    found = [i for i in evidence["incidents"] if i["kind"] == kind]
    assert len(found) == 1, evidence["incidents"]
    return found[0]


def test_rs12_a_repeated_step_shows_its_first_writing_in_both_engines():
    # T6's rust06.py: step 5 written three times in one attempt, at 2.0, 7.0 and 9.0.
    rows = ([rec(s, scalars={"train/loss": 2.0}) for s in range(10)]
            + [rec(5, scalars={"train/loss": 7.0}), rec(5, scalars={"train/loss": 9.0})]
            + [rec(s, scalars={"train/loss": 2.0}) for s in range(10, 20)])
    found = {engine: incident(e, "repeated_steps") for engine, e in both(rows).items()}
    assert found["python"]["scalars"] == found["rust"]["scalars"] == {
        "train/loss": {"value": 2.0, "median_before": 2.0}}
    # The audit's: losses 1 then 9 on the first step.
    rows = [rec(0, scalars={"train/loss": 1.0}), rec(0, scalars={"train/loss": 9.0}),
            rec(1, scalars={"train/loss": 3.0})]
    found = {engine: incident(e, "repeated_steps") for engine, e in both(rows).items()}
    assert found["python"]["scalars"] == found["rust"]["scalars"] == {
        "train/loss": {"value": 1.0, "median_before": None}}


def test_l7_memory_growth_at_one_repeated_step_has_no_rate_in_either_engine():
    gib = 1 << 30
    rows = [rec(7, system={"cuda": {"allocated": gib + k * 100 * (1 << 20), "total": 16 * gib}}) for k in range(6)]
    found = {engine: incident(e, "memory_growth") for engine, e in both(rows).items()}
    for engine, growth in found.items():
        assert growth["bytes_per_step"] is None and growth["steps_to_total"] is None, (engine, growth)


def test_rs11_only_a_checkpoint_object_counts_in_both_engines():
    rows = [rec(0, checkpoint={}), rec(1, checkpoint="{ broken"), rec(2, checkpoint=[1]), rec(3, checkpoint=None),
            rec(4)]
    counted = {engine: e["kind_checks"]["resume_incomplete"]["records"] for engine, e in both(rows).items()}
    assert counted == {"python": 1, "rust": 1}


def stream(table) -> bytes:
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def detect(table, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(BINARY), "detect", *flags], input=stream(table), capture_output=True, timeout=120)


def keys_table(**columns):
    import pyarrow as pa

    base = {"run_id": pa.array(["r", "r"]), "rank": pa.array([0, 0], pa.int32()),
            "attempt": pa.array([1, 1], pa.int64()), "step": pa.array([0, 1], pa.int64())}
    base.update(columns)
    return pa.table({name: array for name, array in base.items() if array is not None})


def test_rs11_text_that_does_not_parse_is_no_checkpoint():
    import pyarrow as pa

    table = keys_table(checkpoint_json=pa.array(["{ broken", "{}"]))
    done = detect(table)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["kind_checks"]["resume_incomplete"]["records"] == 1


def test_rs10_the_offline_modes_refuse_a_null_or_mistyped_key():
    import pyarrow as pa

    for field, array in [
        ("run_id", pa.array(["r", None])),
        ("rank", pa.array([0, None], pa.int32())),
        ("attempt", pa.array([1, None], pa.int64())),
        ("attempt", pa.array(["1", "1"])),
        ("step", pa.array([0, None], pa.int64())),
    ]:
        table = keys_table(**{field: array})
        for flags in ((), ("--run",)):
            done = detect(table, *flags)
            assert done.returncode != 0 and field in done.stderr.decode(), (field, flags, done.stderr)
    # No attempt column at all: a sender before attempts, attempt 0.
    done = detect(keys_table(attempt=None))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["attempt"] == 0


def test_rs16_the_largest_step_is_judged():
    largest = 2**63 - 1
    report = server_report([rec(0), rec(largest)], binary=str(BINARY))
    assert (report["steps"], report["last_step"]) == (2, largest)


def test_c16_the_rust_reader_compares_only_hashes_of_one_kind():
    fixture = json.loads(FIXTURE.read_text())
    assert len(fixture["cases"]) == 11
    for case in fixture["cases"]:
        name, kind = case["name"], case["detector"]
        if kind == "duplicate_batches":
            # A kind across ranks: `build_evidence` judges one rank, so the run's Rust reader
            # (`detect --run`, what the all-ranks view and the server's status use) is asked.
            document = server_run_report(case["records"], binary=str(BINARY))
            incidents, not_checked = document["incidents"], []
            # `compared`: both ranks are listed as compared, or neither is.
            assert document["ranks_compared"] == ([0, 1] if case["compared"] else []), name
        else:
            evidence = tensorquill.build_evidence(case["records"], detectors="rust", server_binary=str(BINARY))
            incidents, not_checked = evidence["incidents"], evidence.get("not_checked", [])  # left out when empty
        found = [i for i in incidents if i["kind"] == kind]
        if case["incident"] is None:
            assert found == [], name
        else:
            assert len(found) == 1, (name, incidents)
            for key, value in case["incident"].items():
                got = found[0]["changes"][0][key] if key == "finding" else found[0][key]
                assert got == value, (name, key, found[0])
        if kind == "resume_data_mismatch":
            skipped = [i for i in not_checked if i["check"] == "resume_data"]
            if case["compared"]:
                assert skipped == [], name
            else:
                assert [i["reason"] for i in skipped] == [C16_REASON], name
