"""`duplicate_batches` across ranks from the Rust rule, without a server: `detect --run`.

`tensorquill-server detect` judges one rank. With `--run` it takes the rows of every rank of
one run, grouped by their `rank` column, judges each rank as before and compares the ranks'
batch hashes as the server compares them for a run's status. Python's
`detect_duplicate_batches(resolve_attempts(records))` is the reference: on the same records
the Rust rule must give the same incident, or none where Python gives none.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import subprocess

import pytest

pytest.importorskip("pyarrow")

from tensorquill.data_checks import DUPLICATE_WINDOW, detect_duplicate_batches  # noqa: E402
from tensorquill.evidence import resolve_attempts  # noqa: E402
from tensorquill.server_detect import (  # noqa: E402
    BINARY_ENV, records_stream, run_stream, server_duplicate_batches, server_report, server_run_report,
)

from test_data_progress import drew, hx, rec  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402
from test_server_detect import CASES  # noqa: E402
from test_server_detectors import everything  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


@pytest.fixture(autouse=True)
def binary(monkeypatch):
    monkeypatch.setenv(BINARY_ENV, str(BINARY))


def python(records) -> dict | None:
    """The reference, as an incident carries its class everywhere in the evidence."""
    found = detect_duplicate_batches(resolve_attempts(list(records)))
    return None if found is None else {**found, "class": "invariant"}


def ranked(rank, attempt, steps, drawn, share=None, run="dup"):
    return [{"schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": s, "ts": float(s),
             "data": {"batches": 1, "batch_hash": hx(drawn(s)), **({} if share is None else {"data_rank": share})}}
            for s in steps]


def four_ranks() -> list[dict]:
    """The run of `test_data_server.py`'s status test: rank 1 draws rank 0's batches at 10-19
    and resumes at 15 with its own; rank 2 is rank 0's tensor-parallel peer; rank 3 draws
    rank 1's at 30-34."""
    return [
        *ranked(0, 1, range(40), lambda s: f"r0-{s}", share=0),
        *ranked(1, 1, range(30), lambda s: f"r0-{s}" if 10 <= s < 20 else f"r1-{s}"),
        *ranked(1, 2, range(15, 40), lambda s: f"r1-{s}"),
        *ranked(2, 1, range(40), lambda s: f"r0-{s}", share=0),
        *ranked(3, 1, range(40), lambda s: f"r1-{s}" if 30 <= s < 35 else f"r3-{s}"),
    ]


def past_the_window() -> list[dict]:
    """More hashed steps than a rank's window holds. The ranks share batches at 100-109, which
    has left both windows, and at 4900-4904, which has not."""
    steps = range(DUPLICATE_WINDOW + 1000)

    def shared(s):
        return 100 <= s < 110 or 4900 <= s < 4905

    return [*ranked(0, 1, steps, lambda s: f"r0-{s}"), *ranked(1, 1, steps, lambda s: f"r0-{s}" if shared(s) else f"r1-{s}")]


def uneven() -> list[dict]:
    """A rank that stopped early: its window is its own last steps (0-999), and the other
    rank's is 904-4999. They shared a batch every hundred steps from 50; only step 950 is in both."""
    return [*ranked(0, 1, range(5000), lambda s: f"r0-{s}"),
            *ranked(1, 1, range(1000), lambda s: f"r0-{s}" if s % 100 == 50 else f"r1-{s}")]


# The cases of `test_data_progress.py`: its one finding, and its four without one.
PROGRESS = {
    "three ranks, one a tensor-parallel peer": [
        record for s in range(20) for record in (
            rec(s, rank=0, data=drew(f"r0-{s}")),
            rec(s, rank=1, data=drew(f"r0-{s}" if 5 <= s < 9 else f"r1-{s}")),
            rec(s, rank=2, data=drew(f"r0-{s}", data_rank=0)),
        )
    ],
    "different batches": [rec(s, rank=r, data=drew(f"r{r}-{s}")) for s in range(5) for r in (0, 1)],
    "one share, two ranks": [rec(s, rank=r, data=drew(s, data_rank=0)) for s in range(5) for r in (0, 1)],
    "one rank": [rec(s, rank=0, data=drew(s)) for s in range(5)],
    "nothing hashed": [rec(s, rank=r, data={"batches": 1}) for s in range(5) for r in (0, 1)],
}
RUNS = {
    **PROGRESS,
    "four ranks with a resume": four_ranks(),
    "past the window": past_the_window(),
    "a rank that stopped early": uneven(),
    "no batch drawn on the shared steps": [
        rec(s, rank=r, data={"batches": 0 if s == 3 else 1, "batch_hash": hx(f"same-{s}")}) for s in range(6) for r in (0, 1)
    ],
    "a data rank of another rank's number": [
        *[rec(s, rank=0, data=drew(f"a-{s}")) for s in range(6)],
        *[rec(s, rank=1, data=drew(f"a-{s}", data_rank=0)) for s in range(6)],  # declared rank 0's share: not a duplicate
        *[rec(s, rank=2, data=drew(f"a-{s}", data_rank=7)) for s in range(6)],  # another share: one
    ],
}


def second_rank(records: list[dict], shared) -> list[dict]:
    """`records` as rank 0 and again as rank 1, which keeps rank 0's batch on the steps
    `shared` picks and draws its own elsewhere."""
    out = [{**r, "rank": 0} for r in records]
    for r in records:
        data = r.get("data")
        if isinstance(data, dict) and isinstance(data.get("batch_hash"), str) and not shared(int(r["step"])):
            data = {**data, "batch_hash": hx(f"rank1-{r['step']}-{r.get('attempt', 0)}")}
        out.append({**r, "rank": 1, **({"data": data} if data is not None else {})})
    return out


# Every parity record set as two ranks: the second shares every third step's batch with the first.
PARITY = {name: second_rank(records, lambda step: step % 3 == 0) for name, records in CASES.items()}


def test_the_reference_finds_duplicates_in_these_runs_and_none_where_none_are():
    found = {name: python(records) for name, records in RUNS.items()}
    assert found["three ranks, one a tensor-parallel peer"] == {
        "kind": "duplicate_batches", "class": "invariant", "step": 5, "ranks": [0, 1, 2], "duplicated_steps": 4,
        "ranks_compared": [0, 1, 2]}
    assert found["four ranks with a resume"] == {
        "kind": "duplicate_batches", "class": "invariant", "step": 10, "ranks": [0, 1, 2], "duplicated_steps": 10,
        "ranks_compared": [0, 1, 2, 3]}
    assert (found["past the window"]["step"], found["past the window"]["duplicated_steps"]) == (4900, 5)
    assert DUPLICATE_WINDOW == 4096
    assert (found["a rank that stopped early"]["step"], found["a rank that stopped early"]["duplicated_steps"]) == (950, 1)
    assert found["no batch drawn on the shared steps"]["duplicated_steps"] == 5  # step 3 drew none
    assert found["a data rank of another rank's number"]["ranks"] == [0, 1, 2]
    for name in ("different batches", "one share, two ranks", "one rank", "nothing hashed"):
        assert found[name] is None, name
    # The parity record sets hold duplicates wherever they hold batch hashes at all.
    assert sum(python(records) is not None for records in PARITY.values()) >= 10


@pytest.mark.parametrize("name", RUNS)
def test_the_rust_rule_gives_pythons_duplicate_batches(name):
    assert server_duplicate_batches(RUNS[name]) == python(RUNS[name])


@pytest.mark.parametrize("name", PARITY)
def test_the_parity_record_sets_as_two_ranks_give_pythons_duplicate_batches(name):
    assert server_duplicate_batches(PARITY[name]) == python(PARITY[name])


def test_the_order_the_ranks_arrive_in_does_not_matter():
    records = four_ranks()
    expected = python(records)
    by_step = sorted(records, key=lambda r: (r["attempt"], r["step"], r["rank"]))  # the ranks row by row
    backwards = sorted(records, key=lambda r: (-r["rank"], r["attempt"]))
    assert server_duplicate_batches(by_step) == server_duplicate_batches(backwards) == expected
    # Attempts out of order are put oldest first, as `records_stream` does for one rank.
    assert server_duplicate_batches(list(reversed(records))) is not None


def test_the_runs_document_holds_each_ranks_report_and_says_what_was_compared():
    ranks = {0: everything(120), 1: [{**r, "rank": 1} for r in everything(200)], 5: [{**r, "rank": 5} for r in everything(60)]}
    document = server_run_report([r for records in ranks.values() for r in records])
    assert document["source"] == "tensorquill-server" and document["schema_version"] == 1
    assert document["run_id"] == ranks[0][0]["run_id"] and document["ranks"] == [0, 1, 5]
    # Each rank's report is the one `detect` gives for that rank alone, key for key.
    assert document["reports"] == [server_report(records) for records in ranks.values()]
    assert [r["steps"] for r in document["reports"]] == [120, 200, 60]
    # These records hold no batch hashes: nothing was found, and nothing could be compared.
    assert document["duplicate_batches"] is None and document["incidents"] == [] and document["ranks_compared"] == []

    document = server_run_report(four_ranks())
    assert document["ranks"] == document["ranks_compared"] == [0, 1, 2, 3]
    assert document["incidents"] == [python(four_ranks())]
    assert document["duplicate_batches"] == {k: v for k, v in python(four_ranks()).items() if k not in ("kind", "class")}
    clean = server_run_report(RUNS["different batches"])
    assert clean["duplicate_batches"] is None and clean["ranks_compared"] == [0, 1]  # compared, and clean
    alone = server_run_report(RUNS["one rank"])
    assert alone["duplicate_batches"] is None and alone["ranks_compared"] == [0]  # nothing to compare with

    # The thresholds go to every rank's detectors, as for one rank.
    options = {"overload_threshold": 0.5, "top_k": 2}
    document = server_run_report([r for records in ranks.values() for r in records], thresholds=options)
    assert document["reports"] == [server_report(records, thresholds=options) for records in ranks.values()]
    assert document["reports"][0]["thresholds"]["top_k"] == 2


def test_the_binary_takes_a_stream_a_rank_and_only_with_run(tmp_path):
    records = four_ranks()
    files = []
    for rank in range(4):
        path = tmp_path / f"rank{rank}.arrow"
        path.write_bytes(records_stream([r for r in records if r["rank"] == rank]))
        files.append(str(path))
    done = subprocess.run([str(BINARY), "detect", "--run", *files], capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr
    document = json.loads(done.stdout)
    assert document == server_run_report(records) and document["incidents"] == [python(records)]
    # One stream holding every rank, on standard input, is the same.
    piped = subprocess.run([str(BINARY), "detect", "--run"], input=run_stream(records), capture_output=True, timeout=60)
    assert json.loads(piped.stdout) == document
    # Without --run `detect` reads one rank, as before: several inputs and two ranks are refused.
    several = subprocess.run([str(BINARY), "detect", *files], capture_output=True, timeout=60)
    assert several.returncode == 1 and b"give --run" in several.stderr and several.stdout == b""
    two = subprocess.run([str(BINARY), "detect"], input=run_stream(records), capture_output=True, timeout=60)
    assert two.returncode == 1 and b"send one rank of one run" in two.stderr
    one = subprocess.run([str(BINARY), "detect", files[1]], capture_output=True, timeout=60)
    assert json.loads(one.stdout) == server_report([r for r in records if r["rank"] == 1]) == document["reports"][1]


def test_what_cannot_be_judged_is_refused_with_the_reason():
    with pytest.raises(RuntimeError, match="send the ranks of one run"):
        server_run_report([*ranked(0, 1, range(5), str), *ranked(1, 1, range(5), str, run="other")])
    with pytest.raises(ValueError, match="no records"):
        server_run_report([])
    assert server_duplicate_batches([]) is None
    with pytest.raises(ValueError, match="unknown field `bogus`"):
        server_run_report(four_ranks(), thresholds={"bogus": 1})


def test_a_step_written_twice_in_one_attempt_is_compared_by_its_last_writing():
    # The one place the two differ, as for the scalars of such a step: Python compares both
    # writings of the step with the other ranks, Rust the last one. A server's store never
    # holds a step twice within an attempt.
    zero = ranked(0, 1, range(6), lambda s: f"r0-{s}")
    twice = [*ranked(1, 1, range(6), lambda s: "r0-3" if s == 3 else f"r1-{s}"), *ranked(1, 1, [3], lambda s: "r1-3")]
    assert python([*zero, *twice])["step"] == 3
    assert server_duplicate_batches([*zero, *twice]) is None
    again = [*ranked(1, 1, range(6), lambda s: f"r1-{s}"), *ranked(1, 1, [3], lambda s: "r0-3")]
    assert python([*zero, *again])["step"] == 3 and server_duplicate_batches([*zero, *again]) == python([*zero, *again])
