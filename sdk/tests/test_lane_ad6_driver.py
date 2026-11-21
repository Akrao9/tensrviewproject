"""Lane AD6: the replica driver (`examples/replica_check.py`) under torchrun, when a rank ends badly.

From CI on `e531c36` (Nov 21, 2026): `test_lane_z4_driver.py::test_trial_merge_server_mode[after-verdict-2]`
failed once with no `verdict.json`. It did not recur on this Mac (lane AD6's report has the counts).
One path that ends that way was found by reading the driver: every rank but 0 left `main()` right
after its `done` file, while rank 0 still had to wait for the reports and judge (in server mode
for as long as the server takes to acknowledge and report). A rank that then ends with a nonzero
status, at its own exit for instance, makes torchrun stop every rank with SIGTERM, rank 0 among
them, before rank 0 wrote its verdict. Now every rank but 0 stays until rank 0 has judged; each
rank says on stderr each stage it reaches; and a rank that raises leaves `failed.rank<R>.json`.

Both tests run the driver through a small wrapper (written here) that changes only timing and
how a rank ends, never the driver's code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SDK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK / "examples"))

import replica_proof as proof  # noqa: E402
from test_lane_z4_driver import READER, _free_port, _passed  # noqa: E402

_WRAPPER = """
import os, sys, time
sys.path.insert(0, {examples!r})
sys.argv[0] = {driver!r}
import replica_check

rank, case = os.environ["RANK"], os.environ["AD6_CASE"]
if case == "late-judge" and rank == "0":
    # Rank 0 is slow to judge, as it is in server mode while the server catches up.
    wait = replica_check._wait_for_reports
    def late(*args, **kwargs):
        time.sleep(5)
        return wait(*args, **kwargs)
    replica_check._wait_for_reports = late
if case == "raise" and rank == "1":
    real, calls = replica_check.proof.reset_and_plant, [0]
    def failing(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:  # the first call is before training: this is in step 2
            raise RuntimeError("planted failure on rank 1")
        return real(*args, **kwargs)
    replica_check.proof.reset_and_plant = failing
code = replica_check.main()
if case == "late-judge" and rank == "1":
    os._exit(1)  # main() returned 0, then the process ends nonzero (a crash at exit, say)
sys.exit(code)
"""


def _run(tmp_path: Path, case: str) -> tuple[subprocess.CompletedProcess, Path]:
    wrapper = tmp_path / "driver.py"
    wrapper.write_text(textwrap.dedent(_WRAPPER.format(examples=str(SDK / "examples"),
                                                       driver=str(SDK / "examples" / "replica_check.py"))))
    env = {**os.environ, "PYTHONPATH": str(SDK / "src"), "AD6_CASE": case}
    env.pop("TENSORQUILL_ENDPOINT", None)
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2", "--master-addr", "127.0.0.1",
         "--master-port", str(_free_port()), str(wrapper), "--cpu", "--out", str(out), "--steps", "16",
         "--step-seconds", "0.2", "--hold-rank", "1"],
        capture_output=True, text=True, timeout=300, env=env, cwd=SDK,
    )
    return done, out


@pytest.mark.skipif(READER is None, reason=f"needs the reader {proof.DEFAULT_READER}")
def test_a_rank_that_ends_nonzero_after_its_done_file_does_not_stop_rank_0_before_the_verdict(tmp_path):
    """Fail before: rank 1 left `main()` at once and exited 1, torchrun stopped rank 0 (SIGTERM)
    while it waited to judge, and there was no `verdict.json` and no PASS line."""
    done, out = _run(tmp_path, "late-judge")
    said = done.stdout + done.stderr
    verdicts = [json.loads(p.read_text()) for p in out.glob("*/verdict.json")]
    assert len(verdicts) == 1, f"{len(verdicts)} verdict files, exit {done.returncode}\n{said}"
    assert _passed(verdicts[0]), said
    assert "PASS files on 2 ranks" in done.stdout, said
    # Rank 1 really ended with 1: the run's status says so, as torchrun reports it.
    assert done.returncode != 0, said
    # Each stage is said on stderr, by rank.
    for line in ("replica_check rank 0: done files of ranks [0, 1] of 2; judging",
                 "replica_check rank 0: verdict written (exit 0)",
                 "replica_check rank 1: done file written",
                 "replica_check rank 1: rank 0's verdict: verdict"):
        assert line in done.stderr, said
    assert not list(out.glob("*/failed.rank*.json")), said


def test_a_rank_that_raises_leaves_a_file_that_says_where_and_why(tmp_path):
    """Fail before: the rank's traceback went to stderr only, and nothing in the run's folder said
    which rank failed, at which stage, or why."""
    done, out = _run(tmp_path, "raise")
    said = done.stdout + done.stderr
    assert done.returncode != 0 and not list(out.glob("*/verdict.json")), said
    [failed] = list(out.glob("*/failed.rank1.json"))
    note = json.loads(failed.read_text())
    assert note["rank"] == 1 and note["stage"].startswith("training 16 steps in run "), note
    assert note["error"] == "RuntimeError: planted failure on rank 1", note
    assert "planted failure on rank 1" in note["traceback"] and "reset_and_plant" not in note["error"], note
    assert "replica_check rank 1: failed at stage 'training 16 steps" in done.stderr, said
