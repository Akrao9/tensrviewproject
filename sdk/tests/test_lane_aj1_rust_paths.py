"""Lane AJ-1, A3 and A5's Rust half (WAVE_AJ_CONTRACT.md): the shared cases through every engine path,
the three Rust paths among them. These are the raw report (`server_report`), the SDK in Rust mode
(`build_evidence(detectors="rust")`), and a live server fed by `send_file`, read from the rank's
`incidents.json`. Each case must give what is written in its `expect`, and the same as it gives for
its `clean_records`.

Fail-before: on 212205d the three Rust paths report nothing for a NaN beside `true`, a string, or
"NaN". A path a case lists under `needs` waits for another lane (AJ-2, `evidence.py`) and is marked
xfail here, not strict. The integrated gate runs this file with `--runxfail`, where every path must
pass."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import evidence  # noqa: E402
from tensorquill.flight import send_file  # noqa: E402
from tensorquill.server_detect import find_binary, server_report  # noqa: E402

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_aj" / "cases.json").read_text())["cases"]
BINARY = find_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
ENGINE_PATHS = ("python", "rust_raw", "rust_sdk", "live_server")
COMPARED = ("incidents", "not_checked", "kind_checks", "run_state", *evidence.MEASUREMENT_BLOCKS)
LIVE_TIMEOUT = 30.0


def _params():
    out = []
    for case in CASES:
        for path in ENGINE_PATHS:
            if path not in case["paths"]:
                continue
            marks = []
            if path in case.get("needs", {}):
                marks.append(pytest.mark.xfail(reason=case["needs"][path], strict=False))
            out.append(pytest.param(case, path, marks=marks, id=f"{case['name']}-{path}"))
    return out


def _text(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _part(doc: dict) -> dict:
    return {key: doc.get(key) for key in COMPARED}


def _check(doc: dict, expect: dict, path: str) -> None:
    incidents = doc.get("incidents") or []
    if "update_spike" in expect:  # an explicit case (no clean control): the update_spike incidents alone
        got = [[i["step"], round(i["grad_norm"], 9), round(i["grad_norm_before"], 9)]
               for i in incidents if i.get("kind") == "update_spike"]
        assert got == [[step, round(now, 9), round(then, 9)] for step, now, then in expect["update_spike"]], incidents
        return
    for block, value in expect.get("blocks", {}).items():  # exact, null for an absent block
        assert doc.get(block) == value, (block, doc.get(block))
    if "incidents" not in expect:
        return
    assert [i["kind"] for i in incidents] == [w["kind"] for w in expect["incidents"]], incidents
    for got, want in zip(incidents, expect["incidents"]):
        for key, value in want.items():
            if key == "parameters":
                assert sorted(got.get("parameters") or {}) == value, got
            else:
                assert got.get(key) == value, (key, got)
    routing = next((n for n in doc.get("not_checked") or [] if n.get("check") == "moe_routing"), None)
    assert (routing["routers"] if routing else None) == expect["moe_routing_not_checked"], doc.get("not_checked")
    if path in ("python", "rust_sdk"):
        assert (doc.get("run_state") or {}).get("state") == expect["run_state"], doc.get("run_state")
    for block, listed in (("weight_jumps", "largest"), ("update_trends", "lowest")):
        if block not in expect:
            continue
        got = doc.get(block)
        assert isinstance(got, dict), (block, got)
        for key, value in expect[block].items():
            if key == listed:
                for got_entry, want_entry in zip(got[listed], value):
                    assert {k: got_entry.get(k) for k in want_entry} == want_entry, (block, got)
            else:
                assert got.get(key) == value, (block, key, got)


# --- the live server -------------------------------------------------------------------------------


class _Live:
    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.store = store
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0.5",
             "--flush-seconds", "0.3"], stdout=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()  # "tensorquill-server listening on 127.0.0.1:PORT"
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def report(self, records: list[dict], run_id: str, directory: Path) -> dict:
        """The rank's `incidents.json` once it covers every record sent: its run and rank are this
        run's, its `last_step` is the last step sent and its `steps` the number of records. A read
        that is only unchanged is not enough (a stale report can be read twice)."""
        records = [{**r, "run_id": run_id} for r in records]
        path = directory / f"{run_id}.rank0.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert send_file(path, self.endpoint, run_id, 0, timeout=LIVE_TIMEOUT) is True
        last = max(r["step"] for r in records)
        deadline, doc = time.monotonic() + LIVE_TIMEOUT, None
        while time.monotonic() < deadline:
            for found in (self.store / "runs" / run_id).rglob("incidents.json"):
                try:
                    doc = json.loads(found.read_text())
                except ValueError:
                    continue
                if (doc.get("run_id") == run_id and doc.get("rank") == 0 and doc.get("last_step") == last
                        and doc.get("steps") == len(records)):
                    return doc
            time.sleep(0.1)
        raise AssertionError(f"no incidents.json of {run_id} covering step {last} within {LIVE_TIMEOUT} s: {doc}")

    def close(self) -> None:
        self.process.send_signal(signal.SIGTERM)
        self.process.wait(20)
        self.process.stdout.close()


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    server = _Live(tmp_path_factory.mktemp("aj1-store"))
    yield server
    server.close()


def _run(path: str, records: list[dict], live, tmp_path: Path, run_id: str, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or {}
    if path == "python":
        return evidence.build_evidence(records, detectors="python", **thresholds)
    if path == "rust_raw":
        return server_report(records, thresholds=thresholds or None)
    if path == "rust_sdk":
        return evidence.build_evidence(records, detectors="rust", **thresholds)
    assert not thresholds, "the live server judges with its own thresholds"
    return live.report(records, run_id, tmp_path)


@pytest.mark.parametrize(("case", "path"), _params())
def test_every_path_reports_what_the_case_expects_and_what_its_clean_records_give(case, path, request, tmp_path):
    server = request.getfixturevalue("live") if path == "live_server" else None
    thresholds = case.get("thresholds")
    got = _run(path, case["records"], server, tmp_path, case["name"], thresholds)
    _check(got, case["expect"], path)
    if case["clean_records"] is not None:
        clean = _run(path, case["clean_records"], server, tmp_path, case["name"] + "-clean", thresholds)
        assert _text(_part(got)) == _text(_part(clean))
