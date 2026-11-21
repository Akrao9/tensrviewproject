"""Seams of wave AE: what no lane could run in its own tree.

1. Both engines, through `build_evidence`, on every case of sections 1, 2 and 4 of
   `WAVE_AE_CONTRACT.md`: the Python detectors (lane AE6) and the server's (lane AE7) were each
   built from the cases alone. They must give the same evidence, and each case's `expected`.
2. A record as the capture writes it (lane AE5) read by both engines: `clip_calls` and
   `step_skipped_part` travel in `extra_json` and reach the server's rules.
3. The late-row counts (section 6): a real server's status (lane AE1) read by the evidence readers
   (lane AE6), on the 60-row part with every fifth row three places late.

All need the built server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import evidence as ev  # noqa: E402
from tensorquill.flight import record_to_row, row_to_record  # noqa: E402

from test_flight_e2e import BINARY  # noqa: E402
from test_wave_ad_seams import _live, _out_of_order  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_ae" / "cases.json").read_text())["cases"]
ENGINE_CASES = sorted(name for name, body in CASES.items() if body["section"] in (1, 2, 4))
JUDGED = ("incidents", "not_checked", "kind_checks", "run_state", "skipped_updates")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and not isinstance(a, (str, bool)) and not isinstance(b, (str, bool)) \
            and abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


def _rows(body: dict) -> list[dict]:
    """A case's records as one run's, each with a run ID, a rank and an attempt, as a capture has."""
    return [{"schema_version": 1, "run_id": "seam", "rank": 0, "attempt": 1, **record} for record in body["given"]["records"]]


def _incident(evidence: dict, kind: str) -> dict | None:
    return next((i for i in evidence["incidents"] if i["kind"] == kind), None)


def _answer(body: dict, evidence: dict) -> dict:
    """The case's `expected` keys, read from evidence either engine built."""
    expected, out = body["expected"], {}
    if "update_spike_step" in expected:
        found = _incident(evidence, "update_spike")
        out["update_spike_step"] = None if found is None else found["step"]
    if "clip_groups_entry" in expected:
        entries = [e for e in evidence.get("not_checked", []) if e.get("check") == "clip_groups"]
        want = expected["clip_groups_entry"]
        out["clip_groups_entry"] = None if not entries else ({k: entries[0].get(k) for k in want} if want else entries[0])
    if "update_spike_check" in expected:
        out["update_spike_check"] = evidence["kind_checks"]["update_spike"]["value"]
    if "stalled_weights_step" in expected:
        found = _incident(evidence, "stalled_weights")
        out["stalled_weights_step"] = None if found is None else found["step"]
    if "skipped_updates" in expected:
        out["skipped_updates"] = evidence.get("skipped_updates")
    if "loss_scale_trouble_step" in expected:
        found = _incident(evidence, "loss_scale_trouble")
        out["loss_scale_trouble_step"] = None if found is None else found["step"]
        for key in ("reason", "consecutive_skips"):
            if key in expected:
                out[key] = None if found is None else found[key]
        if "incident_partial_steps" in expected:
            out["incident_partial_steps"] = None if found is None else found.get("partial_steps")
        if found is not None and "counters_at_end" in expected:  # the incident carries the run's counts
            want = expected["counters_at_end"]
            assert found["recorded_steps"] == want["recorded_steps"] and found["skipped_steps"] == want["skipped_steps"]
            assert found.get("partial_steps", 0) == want["partial_steps"]
    return out


@pytest.mark.parametrize("name", ENGINE_CASES)
def test_both_engines_give_a_case_its_expected_evidence_and_the_same_evidence(name):
    body = CASES[name]
    options = {"update_window": body["given"]["window"]} if "window" in body["given"] else {}
    python = ev.build_evidence(_rows(body), detectors="python", **options)
    rust = ev.build_evidence(_rows(body), detectors="rust", **options)
    expected = {k: v for k, v in body["expected"].items() if k != "counters_at_end"}
    for engine, evidence in (("python", python), ("rust", rust)):
        assert _answer(body, evidence) == expected, f"{engine}: {_answer(body, evidence)} is not {expected}"
    for key in JUDGED:
        assert _same(python.get(key), rust.get(key)), f"{key}:\npython {python.get(key)!r}\nrust   {rust.get(key)!r}"[:2500]


def test_the_two_record_keys_travel_as_the_capture_writes_them():
    """`clip_calls` and `step_skipped_part` are no typed column: they go through `extra_json` and come
    back as written, which is how the server's rules read them."""
    record = {"schema_version": 1, "run_id": "seam", "rank": 0, "attempt": 1, "step": 3, "step_skipped": True,
              "step_skipped_by": "scaler", "step_skipped_part": {"skipped": 1, "of": 2}, "clip_calls": 2, "clip_grad_norm": 16.0}
    row = record_to_row(record)
    extra = json.loads(row["extra_json"])
    assert extra["clip_calls"] == 2 and extra["step_skipped_part"] == {"skipped": 1, "of": 2}
    back = row_to_record(row)
    assert back["clip_calls"] == 2 and back["step_skipped_part"] == {"skipped": 1, "of": 2}


def test_the_late_row_counts_of_a_real_server_reach_the_evidence_readers(tmp_path):
    """Section 6, joined: the server takes the 60-row part with every fifth row three places late,
    stores 48, and its status says 12 rows not newer and 12 distinct late steps confirmed. The
    evidence built from that rank's report carries both as written, with the page's words."""
    _, stored, folder = _live(tmp_path, _out_of_order())
    assert len(stored) == 48
    status = json.loads((folder.parent / "watchdog.json").read_text())
    entry = next(r for r in status["ranks"] if r["rank"] == 0)
    assert (entry.get("rows_not_newer"), entry.get("late_steps_confirmed")) == (12, 12), entry
    evidence = ev._report_evidence(folder, ev.read_server_report(folder), ev.read_watchdog(folder.parent / "watchdog.json"))
    block = (evidence.get("watchdog") or {}).get("rows_not_newer")
    assert isinstance(block, dict) and (block.get("rows"), block.get("late_steps_confirmed")) == (12, 12), evidence.get("watchdog")
    assert "lower bound" in block["note"] and "not a number of steps lost" in block["note"], block["note"]
