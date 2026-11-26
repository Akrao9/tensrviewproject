"""Wave AO, lane AO-9: BUG-28, the verifier half.

`experiments/moe-bakeoff/verify_rerun.py` reads a fixture's Rerun recording back and checks it
against the counts. Since DET-14 the SDK's export writes an expert-load table at every step of
the layer, the first one included: its heading says there is no earlier step to compare with,
every delta is `n/a`, and the rows are sorted by load (`expert_load_table`). The verifier must
take that recording, the one the SDK makes from `generate.py`'s `seeded` fixture, and still check
the first table's rows against the counts.

Only `seeded` is used here: the held-out fixtures are never used to choose a rule. These need
Rerun and NumPy (the SDK's `rerun` extra) and skip without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BAKEOFF = Path(__file__).resolve().parent.parent.parent / "experiments" / "moe-bakeoff"
pytestmark = pytest.mark.skipif(not (BAKEOFF / "verify_rerun.py").is_file(), reason="no bake-off in this checkout")


@pytest.fixture(scope="module")
def bakeoff():
    pytest.importorskip("numpy")
    pytest.importorskip("rerun")
    sys.path.insert(0, str(BAKEOFF))
    try:
        import generate
        import verify_rerun
    finally:
        sys.path.remove(str(BAKEOFF))
    return generate, verify_rerun


@pytest.fixture(scope="module")
def seeded(bakeoff, tmp_path_factory) -> Path:
    generate, _ = bakeoff
    folder = tmp_path_factory.mktemp("bakeoff") / "seeded"
    generate.export_fixture(generate.BY_NAME["seeded"], folder)
    return folder


def test_the_verifier_takes_the_recording_the_sdk_makes_of_seeded(bakeoff, seeded, capsys):
    _, verify_rerun = bakeoff
    verify_rerun.verify(seeded)
    said = capsys.readouterr().out
    assert "and 4 layer 31 expert load tables" in said, said  # steps 1750, 1800, 1850 and 1900


def _with_first_table(verify_rerun, monkeypatch, change):
    """The recording as read, with the first expert-load table's text changed by `change`."""
    read = verify_rerun.read_samples

    def changed(store, path, component):
        samples = read(store, path, component)
        if path == "/incident/expert_load":
            first = min(samples)
            samples = {**samples, first: change(samples[first])}
        return samples

    monkeypatch.setattr(verify_rerun, "read_samples", changed)


def test_the_first_table_is_checked_against_the_counts(bakeoff, seeded, monkeypatch):
    """Two of the first table's rows swapped: the rows no longer follow the counts' order."""
    _, verify_rerun = bakeoff

    def swapped(text):
        lines = text.splitlines()
        lines[4], lines[5] = lines[5], lines[4]
        return "\n".join(lines)

    _with_first_table(verify_rerun, monkeypatch, swapped)
    with pytest.raises(ValueError, match="rows differ from the counts at step 1750"):
        verify_rerun.verify(seeded)


def test_the_first_tables_heading_says_there_is_no_earlier_step(bakeoff, seeded, monkeypatch):
    _, verify_rerun = bakeoff
    _with_first_table(verify_rerun, monkeypatch, lambda text: text.replace("(no earlier step to compare with)", "vs. 1700"))
    with pytest.raises(ValueError, match="heading differs at step 1750"):
        verify_rerun.verify(seeded)


def test_the_first_steps_rows_have_no_delta_and_follow_the_load(bakeoff):
    """Recomputed from the counts alone: no step to compare with, `n/a` deltas, the largest load first."""
    _, verify_rerun = bakeoff
    sample = {"counts": [1, 5, 0, 4], "routed_tokens": 10, "dead": [], "overloaded": [2], "starved": []}
    later = {"counts": [2, 4, 0, 4], "routed_tokens": 10, "dead": [3], "overloaded": [], "starved": []}
    by_key = {(10, 1): sample, (20, 1): later}
    previous, rows = verify_rerun.expected_expert_rows(by_key, 10, 1)
    assert previous is None
    assert rows == [["#002", "50.00%", "n/a", "Overloaded"], ["#004", "40.00%", "n/a", "Active"],
                    ["#001", "10.00%", "n/a", "Active"], ["#003", "0.00%", "n/a", "Idle"]]
    previous, rows = verify_rerun.expected_expert_rows(by_key, 20, 1)  # pinned: a later step, as before
    assert previous == 10 and rows[0] == ["#001", "20.00%", "+10.00", "Active"]
