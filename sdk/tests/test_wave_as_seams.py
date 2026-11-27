"""Wave AS, the merge's seams: what the lanes left for the merge, each outside the lane that made
it untrue or incomplete.

- Lane AS-7 made `init` take a sink whose `path` is not path-like as a sink with no records file
  (`sink.records_file_of`). `flight.attach` still read the sink's `path` itself, so with
  `endpoint=` such a sink raised a `TypeError` out of `init`, where a sink with no file is refused
  by name with a `ValueError`. `attach` now reads the file through `records_file_of` too.
- Lane AS-6 bounded a scalar-only record's `seq` at Int64 in `identity.check`.
  `WAVE_AF_CONTRACT.md` section 12 still said `seq` is not bounded there, and the README's identity
  table, its streaming section and `WAVE_AL_CONTRACT.md` section 4.2 named no upper bound.
- Lane AS-1 made Python's `ranks_unknown` list the lowest 1,000 ranks and count the others. The
  README still said the Python run state names every rank, up to 100,000.
- Lane AS-3 changed what `examples/gpu_system_check.py` holds the data wait and the NVML power
  to. The row of `GPU_RUN_SHEET.md` still said the data wait equals the injected sleep.

The pages are read as text, and each page statement stands beside the code's own number or refusal.
The `attach` tests need no `pyarrow`: the refusal comes before anything imports it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import evidence, flight
from tensorquill.identity import BOUNDS, MalformedRecordError, check
from tensorquill.sink import MemorySink

SDK = Path(__file__).resolve().parents[1]
ROOT = SDK.parent
README = (SDK / "README.md").read_text(encoding="utf-8")
NOWHERE = "grpc://127.0.0.1:9"


class _PathSink(MemorySink):
    """A caller's sink that keeps its records in memory and carries a `path` of any kind."""

    def __init__(self, path):
        super().__init__()
        self.path = path


class _BadFspath:
    def __fspath__(self):
        raise RuntimeError("no path here")


NOT_PATHS = [pytest.param(42, id="int"), pytest.param(object(), id="object"), pytest.param(b"r.rank0.jsonl", id="bytes"),
             pytest.param(_BadFspath(), id="fspath-raises"), pytest.param("", id="empty-str")]


# --- Lane AS-7's seam: the sender's side of a sink with no records file ------------------------------


@pytest.mark.parametrize("path", NOT_PATHS)
def test_attach_refuses_a_sink_whose_path_is_not_path_like_by_name(path):
    """As a sink with no `path` is refused: a `ValueError` that names the sink's type, never a
    `TypeError` from `pathlib` and never a sender started on such a path."""
    with pytest.raises(ValueError, match="the sink must be a JsonlSink .*not _PathSink"):
        flight.attach(_PathSink(path), "as-seams", 0, NOWHERE, heartbeat_seconds=None)


def test_attach_still_refuses_a_sink_with_no_path_and_passes_one_with_no_endpoint(monkeypatch):
    monkeypatch.delenv(flight.ENDPOINT_ENV, raising=False)
    with pytest.raises(ValueError, match="the sink must be a JsonlSink .*not MemorySink"):
        flight.attach(MemorySink(), "as-seams", 0, NOWHERE, heartbeat_seconds=None)
    sink = _PathSink(42)
    assert flight.attach(sink, "as-seams", 0, None) is sink  # no endpoint: returned unchanged, as before


def test_init_with_an_endpoint_refuses_such_a_sink_by_name_and_the_next_init_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(flight.ENDPOINT_ENV, raising=False)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="the sink must be a JsonlSink .*not _PathSink"):
            tensorquill.init(model, sink=_PathSink(42), endpoint=NOWHERE, run_id="as-seams", every=1,
                             heartbeat_seconds=None, stack_seconds=None)
        kept = _PathSink(42)
        session = tensorquill.init(model, sink=kept, run_id="as-seams", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        session.close()
    assert [record["step"] for record in kept.records] == [0]


# --- Lane AS-6's seam: the pages on a `seq` past Int64 ------------------------------------------------


def test_the_pages_say_identity_check_bounds_seq_at_int64_as_it_does():
    scalar = {"schema_version": 2, "record": "scalars", "rank": 0, "attempt": 1, "step": 3, "scalars": {"a": 1.0}}
    check({**scalar, "seq": BOUNDS["seq"][0]})
    with pytest.raises(MalformedRecordError) as refused:
        check({**scalar, "seq": BOUNDS["seq"][0] + 1})
    reason = "a scalar-only record needs a seq, an integer of 0 or more that Int64 holds"
    assert reason in str(refused.value)
    contract = (ROOT / "WAVE_AF_CONTRACT.md").read_text(encoding="utf-8")
    assert "`seq` is not bounded by `identity.check`" not in contract
    assert ("a scalar-only record's `seq` is bounded by `identity.check` at Int64 too, with the reason "
            f'"{reason}"') in contract
    assert "- `seq` is an integer of 0 or more that Int64 holds, counted per session" in (
        ROOT / "WAVE_AL_CONTRACT.md").read_text(encoding="utf-8")
    for words in ('`"record": "scalars"` and a `seq` that is an integer of 0 or more that Int64 holds |',
                  "or without a `seq` that is an integer of 0 or more that Int64 holds, is not valid",
                  "So is one whose `seq` is past Int64, the column's type; a reader of the local file refuses the "
                  "whole file on such a line, as on a negative step (`identity.check`, Nov 27, 2026)."):
        assert words in README, words


# --- Lane AS-1's seam: the README on Python's `ranks_unknown` ------------------------------------------


def test_the_readme_says_python_lists_the_lowest_ranks_and_counts_the_others():
    assert evidence.LISTED_RANKS == 1_000
    assert "its run state names every rank the status counts as never connected" not in README
    assert ("and its run state's `ranks_unknown` has the server's shape (Nov 27, 2026): the lowest 1,000 of its ranks "
            "listed, the others counted in `ranks_omitted`, and the whole count in its `detail`, over every rank the "
            "status counts as never connected, up to 100,000.") in README


# --- Lane AS-3's seam: the run sheet's row for the system check --------------------------------------


def test_the_run_sheet_says_what_the_system_check_holds_the_data_wait_and_power_to():
    row = [line for line in (ROOT / "GPU_RUN_SHEET.md").read_text(encoding="utf-8").splitlines()
           if line.startswith("| system-check |")]
    assert len(row) == 1
    assert "data wait equals a known injected sleep" not in row[0]
    assert ("TensorQuill's data wait is at most 15 ms over a known injected sleep plus the loader's own delay, "
            "measured with capture off in the same run, on 98% of steps, and equals the script's own timer") in row[0]
    assert "power as medians over a steady window" in row[0]
    script = (SDK / "examples" / "gpu_system_check.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--wait-slack-ms", type=float, default=15.0' in script and "WITHIN_SLACK = 0.98" in script
