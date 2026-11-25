"""Wave AL step 2, lane AL-1': the Rust paths are never sent a scalar-only record (`WAVE_AL_CONTRACT.md`
section 4.6, its `server_detect` line; Q4 a).

Scalar-only records are set apart, after `identity.check` and before any engine, in `server_detect._stream`
(so `records_stream`, `run_stream`, `server_report` and `server_run_report`), `FollowedDetect` and
`stream_order`, and before `server_duplicate_batches` slims its records. The detectors read exactly what the
training records alone give.

What a scalar-only record is, is `identity.is_scalar_record`'s one rule. Tests that need pyarrow or the server
binary skip one by one without them; none needs git history."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensorquill import identity
from tensorquill import server_detect as sd

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text(encoding="utf-8"))["cases"]
SECTION_4 = {c["name"]: c for c in CASES if c["section"] == 4}
WITH_SCALARS = [name for name, c in SECTION_4.items() if any(r.get("record") == "scalars" for r in c["records"])]
# And training records too: a case of scalar-only records alone has nothing to judge (tested apart).
MIXED = [name for name in WITH_SCALARS if any(r.get("record") != "scalars" for r in SECTION_4[name]["records"])]


def _records(name: str = "scalar_record_keys") -> list[dict]:
    return copy.deepcopy(SECTION_4[name]["records"])


def _training(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("record") != "scalars"]


def test_the_cases_hold_scalar_only_records_among_training_records():
    assert "scalar_record_keys" in WITH_SCALARS and len(WITH_SCALARS) >= 5
    records = _records()
    assert len(_training(records)) == len(records) - 1


@pytest.mark.parametrize("name", WITH_SCALARS)
def test_stream_order_never_holds_a_scalar_only_record(name):
    records = _records(name)
    assert sd.stream_order(records) == sd.stream_order(_training(records))


@pytest.mark.parametrize("name", WITH_SCALARS)
def test_the_stream_the_binary_reads_is_the_training_records_alone(name):
    pytest.importorskip("pyarrow")
    records = _records(name)
    alone = sd.records_stream(_training(records))
    assert sd.records_stream(records) == alone
    assert sd.run_stream(records) == sd.run_stream(_training(records))
    import pyarrow as pa

    table = pa.ipc.open_stream(alone).read_all()
    assert set(table.column("seq").to_pylist()) <= {None}
    assert table.column("step").to_pylist() == [r["step"] for r in sd.stream_order(_training(records))]


def test_every_record_is_checked_before_the_scalar_only_ones_are_set_apart():
    pytest.importorskip("pyarrow")
    records = _records()
    place = next(i for i, r in enumerate(records) if r.get("record") == "scalars")
    records[place]["rank"] = -1
    with pytest.raises(identity.MalformedRecordError) as raised:
        sd.records_stream(records)
    assert str(raised.value) == (f"rank is -1 at record {place} of the records passed: "
                                 "tensorquill reads an integer of 0 or more")


class _Child:
    """`FollowedDetect`'s child, stood in for: what each look sends, and a report of no incidents."""

    def __init__(self, monkeypatch) -> None:
        self.sent: list[list[dict]] = []
        child = self

        def start(detect, thresholds):
            detect._process, detect._thresholds = object(), thresholds

        def ask(detect, records):
            sd._stream(records)  # what the real child is sent, checked and set apart as there
            child.sent.append([dict(r) for r in records])
            return {"incidents": [], "kind_checks": {}, "samples": [{"step": r["step"]} for r in records],
                    "follow": {"rows": len(records), "samples_from": len(detect.samples)}}

        monkeypatch.setattr(sd.FollowedDetect, "_start", start)
        monkeypatch.setattr(sd.FollowedDetect, "_ask", ask)
        monkeypatch.setattr(sd.FollowedDetect, "close", lambda detect: setattr(detect, "_process", None))


def test_a_watcher_never_sends_a_scalar_only_record_and_an_append_stays_an_append(monkeypatch):
    pytest.importorskip("pyarrow")
    child = _Child(monkeypatch)
    records = _records()
    training = _training(records)
    detect = sd.FollowedDetect()
    detect.report(records)
    assert child.sent == [training]
    assert detect.last["mode"] == "full" and detect._top == training[-1]["step"]
    scalar = next(r for r in records if r.get("record") == "scalars")
    late = {**scalar, "seq": 1, "step": 2, "ts": 101.0}  # a scalar-only record below the last step read
    step_6 = {**training[-1], "step": 6, "ts": 6.0}
    detect.report([*records, late, step_6])
    assert detect.last == {"mode": "append", "records_sent": 1, "rows_read": 1}
    assert child.sent[-1] == [step_6]
    detect.report([*records, late, step_6, {**late, "seq": 2}])
    assert detect.last["mode"] == "same" and len(child.sent) == 2
    assert [s["step"] for s in detect.samples] == [*[r["step"] for r in training], 6]


def test_a_watcher_given_only_scalar_only_records_has_nothing_to_judge(monkeypatch):
    _Child(monkeypatch)
    only = [r for r in _records() if r.get("record") == "scalars"]
    with pytest.raises(ValueError, match="no records to judge"):
        sd.FollowedDetect().report(only)


def test_the_cross_rank_comparison_never_sends_a_scalar_only_record(monkeypatch):
    sent: list = []

    def run_report(records, **_):
        sent.append(list(records))
        return {"incidents": []}

    monkeypatch.setattr(sd, "server_run_report", run_report)
    records = _records()
    assert sd.server_duplicate_batches(records) is None
    assert [r["step"] for r in sent[0]] == [r["step"] for r in _training(records)]
    sent.clear()
    assert sd.server_duplicate_batches([r for r in records if r.get("record") == "scalars"]) is None
    assert sent == []


def test_a_report_of_only_scalar_only_records_has_nothing_to_judge():
    assert "scalar_seq_watermark_across_restart" not in MIXED
    for only in ([r for r in _records() if r.get("record") == "scalars"], _records("scalar_seq_watermark_across_restart")):
        for report in (sd.server_report, sd.server_run_report):
            with pytest.raises(ValueError, match="no records to judge: every record given is a scalar-only record"):
                report(only, binary="/nonexistent/tensorquill-server")


# --- With the server binary: what the detectors report ------------------------------------------


def _binary() -> str | None:
    try:
        return sd.find_binary()
    except Exception:  # noqa: BLE001
        return None


needs_binary = pytest.mark.skipif(_binary() is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


@needs_binary
@pytest.mark.parametrize("name", MIXED)
def test_the_rust_detectors_report_what_the_training_records_alone_give(name):
    pytest.importorskip("pyarrow")
    records = _records(name)
    with_scalars = sd.server_report(records, samples=True)
    alone = sd.server_report(_training(records), samples=True)
    assert with_scalars == alone
    assert "repeated_steps" not in [i["kind"] for i in with_scalars["incidents"]]
    assert sd.server_run_report(records) == sd.server_run_report(_training(records))
