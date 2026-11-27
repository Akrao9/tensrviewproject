"""Lane AR-7, item G (the coordinator's addition, after lane AR-4's section 12.5): the Python readers of
a status take its new count of missing ranks.

Since lane AR-4 the server's status lists at most the lowest 1,000 missing ranks in `missing_ranks`
and counts the others in `missing_ranks_omitted`, a key written only when some are left out. The
server's own readers carry the count (its incident, its run state). Before this item, `evidence.py`
read the list alone, so with no world size and over 1,000 missing only the listed 1,000 counted as
unknown, a rank past the first 1,000 that connected was no news to the watcher, and the incident's
text said nothing of the others.

AR-4's proposals, each held here:
1. `_progress_incidents` carries `missing_ranks_omitted` into its incident, as the Rust incident does;
2. the run-state reasons and `_ranks_without_folder` count the omitted ranks as the server's
   `run_state` does: every rank not seen below the larger of the world size and the highest
   connected rank plus one, at most `MAX_EXPECTED_RANKS`;
3. `_status_stamp` and `_grace_alert_key` change when the count does;
4. the incident's text says "and N more".

A status with no such key reads exactly as before (the first tests, which pass on the base too).
The last test runs the real server and holds Python's run state from its status to the server's.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import pytest

from tensorquill import evidence
from tensorquill.evidence import MAX_EXPECTED_RANKS, run_state

NOW = 1_700_000_000_000


def _rank(rank: int, state: str = "active", step: int = 10) -> dict:
    return {"rank": rank, "state": state, "last_step": step, "attempt": 1, "severity": None if state == "active"
            else "warning", "seconds_since_last_step": 1.0 if state == "active" else 400.0}


def _status(missing: list[int], omitted: int | None = None, world: int | None = None, ranks=None,
            summary: str = "degraded") -> dict:
    status = {"summary": summary, "written_at_ms": NOW, "silent_floor_seconds": 300, "critical_seconds": 1800,
              "world_size": world, "ranks": ranks if ranks is not None else [_rank(0), _rank(1, "silent")],
              "missing_ranks": missing}
    if omitted is not None:
        status["missing_ranks_omitted"] = omitted
    return status


def _unknown(status: dict, judged=(0, 1)) -> list[int]:
    return _unknown_reason(status, judged).get("ranks", [])


def _unknown_reason(status: dict, judged=(0, 1)) -> dict:
    found = run_state(incidents=[], steps_judged=True, status=status, judged_ranks=list(judged))
    return next((r for r in found["reasons"] if r["code"] == "ranks_unknown"), {})


# --- A status with no such key reads exactly as before ---------------------------------------------


def test_with_no_count_the_progress_incident_is_as_before():
    (incident,) = evidence._progress_incidents(_status([2, 3], world=4))
    assert "missing_ranks_omitted" not in incident
    assert incident["missing_ranks"] == [2, 3] and incident["kind"] == "rank_silent"


def test_with_no_count_the_run_state_and_the_folders_are_as_before():
    assert _unknown(_status([2, 3], world=4)) == [2, 3]
    assert _unknown(_status(list(range(2, 1002)))) == list(range(2, 1002))
    found = evidence._ranks_without_folder({0: Path("a"), 1: Path("b")}, _status([2, 3], world=4))
    listed = "the server's status lists this rank as never connected, and nothing is stored for it"
    assert found == {2: listed, 3: listed}
    found = evidence._ranks_without_folder({0: Path("a")}, _status([], world=3, ranks=[_rank(0)]))
    assert found == {1: "the job's world_size is 3 and nothing is stored for this rank",
                     2: "the job's world_size is 3 and nothing is stored for this rank"}


def test_with_no_count_the_stamps_are_as_before():
    status = _status([2, 3], world=4)
    stamp = evidence._status_stamp(status, None)
    assert stamp[1] == (2, 3) and len(stamp) == 7
    key = evidence._grace_alert_key(status, None, None)
    assert key[1] == (2, 3) and len(key) == 5


def test_with_no_count_the_text_is_as_before():
    (incident,) = evidence._progress_incidents(_status(list(range(2, 20))))
    text = evidence._describe_stream(incident)
    assert "ranks 2, 3, 4, 5, 6, 7, 8, 9 never connected (if only some" in text
    assert "more never connected" not in text


# --- The count, carried ----------------------------------------------------------------------------


def test_the_progress_incident_carries_the_count():
    for summary in ("degraded", "stalled"):
        (incident,) = evidence._progress_incidents(_status(list(range(2, 1002)), 500, summary=summary))
        assert incident["missing_ranks"] == list(range(2, 1002))
        assert incident["missing_ranks_omitted"] == 500


def test_the_run_state_counts_the_ranks_the_status_counts_and_does_not_list():
    """No world size, ranks 0 and 1501 connected: the status lists 1 to 1000 and counts 1001 to 1500."""
    ranks = [_rank(0), _rank(1501)]
    status = _status(list(range(1, 1001)), 500, ranks=ranks)
    # Since lane AS-1 the reason lists the lowest 1,000 of the 1,500 and counts the others, as the server's.
    reason = _unknown_reason(status, judged=(0, 1501))
    assert (reason["ranks"], reason["ranks_omitted"]) == (list(range(1, 1001)), 500)
    assert reason["detail"].startswith("1500 ranks without a current report"), reason["detail"]


def test_the_counted_ranks_stop_at_the_limit():
    ranks = [_rank(0), _rank(MAX_EXPECTED_RANKS - 1)]
    status = _status(list(range(1, 1001)), MAX_EXPECTED_RANKS - 1002, ranks=ranks)
    reason = _unknown_reason(status, judged=(0, MAX_EXPECTED_RANKS - 1))
    assert (reason["ranks"], reason["ranks_omitted"]) == (list(range(1, 1001)), MAX_EXPECTED_RANKS - 1002)
    # A rank over the limit names nothing below it, as the server's `rank_taken`.
    over = _status(list(range(1, 1001)), 7, ranks=[_rank(0), _rank(MAX_EXPECTED_RANKS + 5)])
    assert _unknown(over, judged=(0,)) == [*range(1, 1001)]


def test_the_folders_name_the_counted_ranks_too():
    status = _status(list(range(1, 1001)), 500, ranks=[_rank(0), _rank(1501)])
    found = evidence._ranks_without_folder({0: Path("a"), 1501: Path("b")}, status)
    assert sorted(found) == list(range(1, 1501))
    assert "counts this rank as never connected" in found[1200]
    assert "lists this rank as never connected" in found[5]


def test_the_stamps_change_when_the_count_does():
    one, two = (_status(list(range(2, 1002)), n) for n in (500, 499))
    assert evidence._status_stamp(one, None) != evidence._status_stamp(two, None)
    assert evidence._grace_alert_key(one, None, None) != evidence._grace_alert_key(two, None, None)


def test_the_text_says_how_many_more():
    (incident,) = evidence._progress_incidents(_status(list(range(2, 1002)), 500))
    text = evidence._describe_stream(incident)
    assert "ranks 2, 3, 4, 5, 6, 7, 8, 9 and 1492 more never connected" in text, text


# --- Both engines, on a live server ----------------------------------------------------------------


pytest.importorskip("pyarrow.flight")
from test_flight_e2e import BINARY, Server  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


@needs_server
def test_the_python_run_state_counts_what_the_servers_counts(tmp_path, monkeypatch):
    """Rank 1500 alone, with no world size: the server lists 1,000 of the 1,500 ranks below it and
    counts the rest. Its run state (Rust, in `watchdog.json`) and Python's, from the same status,
    name the same unknown ranks: each lists the lowest 1,000 and counts the others (lane AS-1)."""
    import torch

    import tensorquill

    for name in list(os.environ):
        if name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") or name.startswith(("TENSORQUILL_", "SLURM_", "TORCHELASTIC_")):
            if name != "TENSORQUILL_SERVER_BIN":
                monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "1500")
    store = tmp_path / "store"
    server = Server(store, 0, "--silent-seconds", "1", "--start-seconds", "1", "--watchdog-seconds", "0.2")
    status_file = store / "runs" / "many" / "watchdog.json"
    try:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = tensorquill.init(model, run_id="many", every=1, endpoint=server.endpoint)
        status = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            model(torch.randn(3, 2)).pow(2).mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            session.flush()
            try:
                status = json.loads(status_file.read_text())
            except (FileNotFoundError, ValueError):
                status = None
            unknown = next((r for r in (status or {}).get("run_state", {}).get("reasons", ())
                            if r.get("code") == "ranks_unknown"), None)
            # The count is there, and the server's report of rank 1500 is current (it is judged).
            if status and status.get("missing_ranks_omitted") and unknown \
                    and len(unknown["ranks"]) + unknown.get("ranks_omitted", 0) == 1500:
                break
            time.sleep(0.2)
        assert status and status.get("missing_ranks_omitted") == 500, status
        assert status["missing_ranks"] == list(range(1000))
        rust = next(r for r in status["run_state"]["reasons"] if r["code"] == "ranks_unknown")
        # Python's own engine over the store (it reads the same `watchdog.json`), and its run state
        # from this status alone with the connected rank judged.
        whole = tensorquill.build_evidence(store / "runs" / "many")
        python_whole = next((r for r in whole["run_state"]["reasons"] if r["code"] == "ranks_unknown"), {})
        python = _unknown_reason(status, judged=[r["rank"] for r in status["ranks"]])
        session.close()
    finally:
        server.terminate()
    # Before this item Python counted the listed 1,000 only. Since lane AS-1 both engines list the
    # lowest 1,000 of the 1,500 and count the other 500, Python from the status alone and in its
    # engine over the store.
    assert (python["ranks"], python["ranks_omitted"]) == (list(range(1000)), 500), python.get("detail")
    assert python["detail"].startswith("1500 ranks without a current report"), python["detail"]
    assert rust["ranks"] == python["ranks"] and rust["ranks_omitted"] == 500, rust
    assert (python_whole["ranks"], python_whole.get("ranks_omitted")) == (python["ranks"], 500), python_whole.get("detail")
