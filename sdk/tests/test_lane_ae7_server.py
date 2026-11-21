"""Lane AE7: the Rust server's side of sections 1, 2 and 4 of `WAVE_AE_CONTRACT.md`, through the
built binary.

Every case of those sections in `fixtures/wave_ae/cases.json` is run through
`tensorquill-server detect` (`server_report`) and through `build_evidence(..., detectors="rust")`,
and must give its `expected`. A report does not carry `loss_scale_trouble`'s counters when nothing
was raised; those are held in `server/src/detect/lane_ae7_tests.rs` (`counters_at_end`), and here
only `recorded_steps` (the kind's `records` in `kind_checks`) and, on an incident, the counts it
carries.

Then one case of each section is sent to a real server a part at a time, with the pair that is not
compared, the attempt boundary or the partial step on a part boundary, with and without a restart
of the server between the parts. The live report must be the offline report on the rows the server
stored, and both must give the case's `expected`.

Needs `pyarrow` (the `flight` extra) and the server binary (`TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import evidence as ev  # noqa: E402
from tensorquill.server_detect import server_report  # noqa: E402

from test_flight_e2e import BINARY, stored_records  # noqa: E402
from test_server_detectors import FAST, Stream, differences, record  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_ae" / "cases.json").read_text())
CASES = [(name, body) for name, body in DOCUMENT["cases"].items() if body["section"] in (1, 2, 4)]
REASON = ("clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter, "
          "so {n} comparison(s) of update_spike were not made")


def _options(given: dict) -> dict | None:
    return {"update_window": given["window"]} if "window" in given else None


def _incident(document: dict, kind: str) -> dict | None:
    return next((i for i in document["incidents"] if i.get("kind") == kind), None)


def answer(document: dict, expected: dict) -> dict:
    """What a server report (or evidence built from one) answers, under the keys the case uses.
    `counters_at_end` is held to what the document shows of it: see the module's docstring."""
    out: dict = {}
    scale = _incident(document, "loss_scale_trouble")
    for key, want in expected.items():
        if key == "update_spike_step":
            found = _incident(document, "update_spike")
            out[key] = None if found is None else found["step"]
        elif key == "stalled_weights_step":
            found = _incident(document, "stalled_weights")
            out[key] = None if found is None else found["step"]
        elif key == "loss_scale_trouble_step":
            out[key] = None if scale is None else scale["step"]
        elif key in ("reason", "consecutive_skips"):
            out[key] = None if scale is None else scale[key]
        elif key == "incident_partial_steps":
            out[key] = None if scale is None else scale.get("partial_steps")
        elif key == "clip_groups_entry":
            entries = [e for e in document.get("not_checked", []) if e.get("check") == "clip_groups"]
            out[key] = None if not entries else {k: entries[0].get(k) for k in (want or entries[0])}
        elif key == "skipped_updates":
            out[key] = document.get("skipped_updates")
        elif key == "counters_at_end":
            shown = {"recorded_steps": document["kind_checks"]["loss_scale_trouble"]["records"]}
            if scale is not None:
                shown.update(skipped_steps=scale["skipped_steps"], partial_steps=scale.get("partial_steps", 0))
            out[key] = {**{k: want[k] for k in want if k not in shown}, **shown}
        else:
            raise AssertionError(f"a key no case of sections 1, 2 and 4 uses: {key}")
    return out


@pytest.mark.parametrize("name,body", CASES, ids=[n for n, _ in CASES])
def test_the_server_gives_each_case_its_expected(name, body):
    given = body["given"]
    report = server_report(given["records"], thresholds=_options(given))
    assert answer(report, body["expected"]) == body["expected"]


@pytest.mark.parametrize("name,body", CASES, ids=[n for n, _ in CASES])
def test_evidence_from_the_rust_detectors_gives_each_case_its_expected(name, body):
    given = body["given"]
    options = {"update_window": given["window"]} if "window" in given else {}
    evidence = ev.build_evidence(given["records"], detectors="rust", **options)
    assert answer(evidence, body["expected"]) == body["expected"]
    if body["expected"].get("clip_groups_entry"):
        # The entry names update_spike, which stays checked: every other pair was judged. And as any
        # `not_checked` entry does, it makes the run read degraded (this page's reading).
        assert evidence["kind_checks"]["update_spike"]["value"] == "checked"
        entry = next(e for e in evidence["not_checked"] if e["check"] == "clip_groups")
        assert entry["reason"] == REASON.format(n=entry["comparisons"])
        degraded = [r for r in evidence["run_state"]["reasons"] if r["state"] == "degraded"]
        assert any("clip_groups" in str(r) for r in degraded), evidence["run_state"]
        if body["expected"]["update_spike_step"] is None:
            assert evidence["run_state"]["state"] == "degraded"


# --- the live server, a part at a time -----------------------------------------------------------

RUN = "ae7"
BLOCKS = ("incidents", "not_checked", "kind_checks", "skipped_updates")


def _sendable(given: dict) -> list[dict]:
    """A case's records as a capture writes them, attempt 1 unless the case names one."""
    return [record(r["step"], r.get("attempt", 1), run=RUN,
                   **{k: v for k, v in r.items() if k not in ("step", "attempt", "rank")})
            for r in given["records"]]


def _blocks(document: dict) -> dict:
    return {name: document.get(name) for name in BLOCKS}


# One case of each section, cut where the rule turns: the later record of the pair that is not
# compared, the first row of the resumed attempt, the partial step (and the row after it).
LIVE = [
    ("after_a_pair_that_is_not_compared_the_next_is", (5,)),
    ("two_pairs_that_are_not_compared", (5, 6)),
    ("a_stall_does_not_carry_across_a_resume", (3,)),
    ("a_stall_does_not_carry_across_a_resume", (3, 5)),
    ("five_full_skips_one_partial_step_five_full_skips", (5,)),
    ("five_full_skips_one_partial_step_five_full_skips", (6,)),
    ("skipped_updates_counts_a_partial_step_both_reported_apart", (5,)),
]


@pytest.mark.parametrize("restart", [False, True], ids=["one_server", "restarted_at_each_cut"])
@pytest.mark.parametrize("name,cuts", LIVE, ids=[f"{n}-at-{'-'.join(map(str, c))}" for n, c in LIVE])
def test_the_live_report_is_offline_detection_on_the_stored_rows(tmp_path, name, cuts, restart):
    body = DOCUMENT["cases"][name]
    given = body["given"]
    options = _options(given)
    args = (*FAST, *(("--thresholds", json.dumps(options)) if options else ()))
    rows = _sendable(given)
    stream = Stream(tmp_path, *args, run=RUN)
    try:
        start = 0
        for cut in (*cuts, len(rows)):
            if restart and start:
                stream.restart()
            live = stream.send(rows[start:cut])
            start = cut
        stored = stored_records(stream.store, RUN)
        assert [r["step"] for r in stored] == [r["step"] for r in rows]
        offline = server_report(stored, thresholds=options)
        assert differences(_blocks(offline), _blocks(live)) == []
        assert answer(live, body["expected"]) == body["expected"]
        if body["expected"].get("clip_groups_entry"):
            _assert_degraded_by_clip_groups(stream.store)
    finally:
        stream.server.terminate()


def _assert_degraded_by_clip_groups(store: Path) -> None:
    """The live run state lists the `clip_groups` check among what was not checked."""
    status_file = store / "runs" / RUN / "watchdog.json"
    deadline = time.monotonic() + 10
    while True:
        try:
            reasons = json.loads(status_file.read_text())["run_state"]["reasons"]
            checks = [c for r in reasons if r.get("code") == "not_checked" for c in r["checks"]]
            if "clip_groups" in checks:
                assert any(r["state"] == "degraded" and r["code"] == "not_checked" for r in reasons)
                return
        except (FileNotFoundError, ValueError, KeyError):
            pass
        assert time.monotonic() < deadline, "the live run state does not name clip_groups"
        time.sleep(0.05)
