"""Wave AF-W: `--watch` with the Rust detectors judges only what was appended.

`server_detect.FollowedDetect` keeps one `tensorquill-server detect --samples --follow` child per
watcher: the first look sends every record, a plain append sends only the new ones, and anything else
starts a new child that is sent every record. The tests here:

1. The mechanism: on every golden record set, cut where the rest is a plain append, the prefix and
   then the rest give what one `detect --samples` over the whole gives (report and samples).
2. The real command (`python -m tensorquill <input> --watch`, Rust detectors) after each change, for a
   growing `.jsonl` file and for a store folder of Parquet parts: its evidence file against a fresh
   `build_evidence(..., detectors="rust")` in a fresh process, key for key; and, read from the
   command's own process (`TENSORQUILL_FOLLOW_LOG`), what the binary read at each look: an append
   reads only the appended rows, and every other change is a full recomputation.

No test asserts a timing.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import server_detect as sd

SRC = Path(__file__).parents[1] / "src"


def _binary_or_skip() -> str:
    binary = sd.find_binary()
    if binary is None:
        pytest.skip("no tensorquill-server binary")
    pytest.importorskip("pyarrow")
    if sd.binary_protocol(binary) != sd.PROTOCOL:
        pytest.skip(f"{binary} speaks another protocol than {sd.PROTOCOL}")
    return binary


def _text(value) -> str:
    return json.dumps(value, sort_keys=True, default=repr)


# ---- 1. The mechanism, on every golden record set ------------------------------------------------


def _golden_sets() -> dict[str, list[dict]]:
    import test_golden

    return {name: test_golden.records(name) for name in test_golden.NAMES}


def _routers(records: list[dict]) -> dict[str, int]:
    """Each router the records name, with its most experts."""
    found: dict[str, int] = {}
    for record in records:
        for name, counts in (record.get("moe") or {}).items():
            if isinstance(counts, list):
                found[name] = max(found.get(name, 0), len(counts))
    return found


def _cuts(records: list[dict]) -> list[int]:
    """Points where the rest is a plain append, by the follower's and the binary's rule: the newest
    attempt of the prefix, steps above every step before, in order, and no router new or grown. A
    few of them, early, middle and late."""
    found = []
    for cut in range(1, len(records)):
        prefix, rest = records[:cut], records[cut:]
        newest = sd._whole(sd.stream_order(prefix)[-1].get("attempt", 0))
        top = max(sd._whole(r["step"]) for r in prefix)
        ok = _routers(records) == _routers(prefix)
        for record in rest if ok else ():
            if sd._whole(record.get("attempt", 0)) != newest or sd._whole(record["step"]) <= top:
                ok = False
                break
            top = sd._whole(record["step"])
        if ok:
            found.append(cut)
    if len(found) <= 5:
        return found
    return sorted({found[0], found[len(found) // 3], found[len(found) // 2], found[-2], found[-1]})


@pytest.fixture(scope="module")
def golden_sets():
    _binary_or_skip()
    pytest.importorskip("pyarrow.flight")
    return _golden_sets()


def test_a_prefix_and_then_its_append_give_what_the_whole_gives_on_every_golden_set(golden_sets):
    appended = 0
    for name, records in golden_sets.items():
        whole = sd.server_report(records, samples=True)
        for cut in _cuts(records):
            follow = sd.FollowedDetect()
            try:
                follow.report(records[:cut])
                got = follow.report(records)
                assert follow.last["mode"] == "append", (name, cut, follow.last)
                assert follow.last["rows_read"] == len(records) - cut, (name, cut)
                appended += 1
            finally:
                follow.close()
            assert _text({**got, "samples": None}) == _text({**whole, "samples": None}), (name, cut)
            assert _text(got["samples"]) == _text(whole["samples"]), (name, cut)
    assert appended >= 20, f"only {appended} appends were exercised"


def test_an_append_that_is_not_plain_is_judged_again_in_full(golden_sets):
    """A late row (a step not above the last), a row of another attempt, and thresholds changed: each
    is a full recomputation, by the counter, and gives what the whole gives."""
    records = next(r for r in golden_sets.values() if len(r) >= 30 and len({x.get("attempt", 0) for x in r}) == 1)
    late = [*records, {**records[5]}]
    resumed = [*records, {**records[-1], "attempt": int(records[-1].get("attempt", 0)) + 1, "step": records[3]["step"]}]
    for changed, why in ((late, "not above the last step read"), (resumed, "where the newest attempt read is")):
        follow = sd.FollowedDetect()
        try:
            follow.report(records)
            got = follow.report(changed)
            assert follow.last["mode"] == "full" and why in follow.last["why"], follow.last
            assert follow.last["records_sent"] == len(changed) == follow.last["rows_read"]
        finally:
            follow.close()
        whole = sd.server_report(changed, samples=True)
        assert _text(got) == _text(whole)
    follow = sd.FollowedDetect()
    try:
        follow.report(records)
        got = follow.report(records, thresholds={"update_window": 50})
        assert follow.last["mode"] == "full" and follow.last["why"] == "the thresholds changed"
    finally:
        follow.close()
    assert _text(got) == _text(sd.server_report(records, thresholds={"update_window": 50}, samples=True))


def test_the_binary_refuses_an_append_it_cannot_take_and_the_follower_starts_again(golden_sets):
    """The binary checks each append itself: rows of a router not seen before change every earlier
    sample, so it refuses them, and the follower judges every record again."""
    records = next(r for r in golden_sets.values() if len(r) >= 30 and any(x.get("moe") for x in r))
    last = records[-1]
    grown = {**last, "step": int(last["step"]) + 1,
             "moe": {**last["moe"], "a.router.never.seen": [1, 2, 3, 4]}}
    follow = sd.FollowedDetect()
    try:
        follow.report(records)
        got = follow.report([*records, grown])
        assert follow.last["mode"] == "full" and "the binary refused the append" in follow.last["why"], follow.last
    finally:
        follow.close()
    assert _text(got) == _text(sd.server_report([*records, grown], samples=True))


# ---- 2. The real command, after each change ---------------------------------------------------------

RUN = "w"


def records(steps, *, attempt: int = 0, spike: int | None = None, run: str = RUN) -> list[dict]:
    """Step records of one rank with what the Rust detectors judge: two routers, gradient norms, layer
    statistics and a loss; a gradient spike at `spike`."""
    out = []
    for step in steps:
        norm = 1.0 + 0.001 * (step % 7) + (40.0 if step == spike else 0.0)
        out.append({
            "schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step, "ts": 1000.0 + step,
            "grad_norm": norm, "grad_norms": {"w": norm / 2, "b": norm / 3},
            "scalars": {"train/loss": 1.0 / (step + 1)},
            "layers": {"blocks.0": {"abs_max": 1.0 + 0.01 * (step % 5), "mean": 0.0, "std": 0.5, "zero_frac": 0.0}},
            "moe": {"r0": [5 + step % 3, 4, 6, 5], "r1": [9, 1 + step % 2, 3, 3]}, "moe_top_k": {"r0": 1, "r1": 1},
        })
    return out


def write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as file:
        file.write("".join(json.dumps(r) + "\n" for r in rows))


def write_part(run_folder: Path, rows: list[dict], *, attempt: int = 0, name: str | None = None) -> Path:
    """One Parquet part as the server stores it: `rank=0/attempt=<n>/part-<first>-<last>.parquet`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tensorquill.flight import _row, flight_schema

    folder = run_folder / "rank=0" / f"attempt={attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in rows]
    path = folder / (name or f"part-{min(steps):08d}-{max(steps):08d}.parquet")
    table = pa.Table.from_pylist([_row(r) for r in rows], schema=flight_schema())
    pq.write_table(table, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    return path


def _env(log: Path) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(SRC), sd.LOOKS_ENV: str(log)}


class Watching:
    """`python -m tensorquill <input> --watch --detectors rust`, the command itself, in its own
    process: its `updated` lines are counted, and its follow log read."""

    def __init__(self, tmp: Path, source: Path, *args: str) -> None:
        self.out, self.log = tmp / "watched.json", tmp / f"looks-{time.monotonic_ns()}.jsonl"
        command = [sys.executable, "-m", "tensorquill", str(source), "--watch", "--detectors", "rust",
                   "--interval", "0.05", "-o", str(self.out), *args]
        # Ctrl-C as at a terminal: a child of a background job would inherit SIGINT ignored, and then
        # the command never sees it.
        self.process = subprocess.Popen(command, cwd=tmp, env=_env(self.log), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True,
                                        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        self.lines: queue.Queue[str] = queue.Queue()
        self.updates = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.put(line)

    def wait(self, updates: int) -> None:
        """Until the command has written `updates` times in all."""
        deadline = time.monotonic() + 120
        while self.updates < updates:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                line = ""
            if line.startswith("updated "):
                self.updates += 1
            if time.monotonic() > deadline or self.process.poll() is not None and self.lines.empty():
                raise AssertionError(f"the watcher wrote {self.updates} times, not {updates}: "
                                     f"{self.process.stderr.read() if self.process.poll() is not None else ''}")

    def looks(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def stop(self) -> None:
        self.process.send_signal(signal.SIGINT)
        try:
            code = self.process.wait(60)
        except subprocess.TimeoutExpired:
            self.process.kill()  # never left running, whatever the test found
            self.process.wait()
            raise
        assert code == 0, self.process.stderr.read()


def _children_left(binary: str) -> list[str]:
    listed = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True).stdout.splitlines()
    return [line for line in listed if line.startswith(binary) and "--follow" in line]


def fresh(tmp: Path, source: Path, **thresholds) -> dict:
    """A full recomputation over `source` with the Rust detectors, in a fresh process, as written."""
    out = tmp / f"fresh-{time.monotonic_ns()}.json"
    code = ("import json, sys\nfrom tensorquill import evidence as ev\n"
            "ev.write_evidence(ev.build_evidence(sys.argv[1], detectors='rust', **json.loads(sys.argv[3])), sys.argv[2])\n")
    done = subprocess.run([sys.executable, "-c", code, str(source), str(out), json.dumps(thresholds)],
                          env={**os.environ, "PYTHONPATH": str(SRC)}, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return {"text": out.read_text(), "evidence": json.loads(out.read_text())}


def assert_parity(watching: Watching, reference: dict, *, store: bool) -> None:
    """The watcher's file is the fresh recomputation key for key. Only a store's watcher writes
    `detection` (what it found when it looked for the server's report), which a read has not."""
    watched = json.loads(watching.out.read_text())
    if store:
        assert "detection" in watched and "detection" not in reference["evidence"]
        watched.pop("detection")
    else:
        assert watching.out.read_text() == reference["text"], "the watcher's file is not the fresh file byte for byte"
    assert sorted(watched) == sorted(reference["evidence"])
    for key in watched:
        assert _text(watched[key]) == _text(reference["evidence"][key]), key


@pytest.fixture
def binary():
    found = _binary_or_skip()
    pytest.importorskip("pyarrow.parquet")
    yield found
    assert _children_left(found) == [], "a detect --follow child outlived its watcher"


# Each change: what to do to the input, and what the next look must be ("append", or "full").
JSONL_CHANGES = {
    "append": [("append", lambda p: write_jsonl(p, records(range(40, 55)), "a"))],
    "late_rows": [("full", lambda p: write_jsonl(p, records(range(10, 12)), "a"))],
    "resume_below": [("full", lambda p: write_jsonl(p, records(range(30, 50), attempt=1), "a"))],
    "truncation": [("full", lambda p: write_jsonl(p, records(range(30))))],
    "replaced": [("full", lambda p: (write_jsonl(p.with_suffix(".new"), records(range(45), spike=20)),
                                      os.replace(p.with_suffix(".new"), p)))],
    "two_in_succession": [("append", lambda p: write_jsonl(p, records(range(40, 50)), "a")),
                          ("append", lambda p: write_jsonl(p, records(range(50, 70), spike=60), "a"))],
}


@pytest.mark.parametrize("change", sorted(JSONL_CHANGES))
def test_the_watch_command_on_a_jsonl_file(binary, tmp_path, change):
    path = tmp_path / f"{RUN}.rank0.jsonl"
    write_jsonl(path, records(range(40), spike=25))
    watching = Watching(tmp_path, path)
    try:
        watching.wait(1)
        assert_parity(watching, fresh(tmp_path, path), store=False)
        assert watching.looks()[-1]["mode"] == "full"
        for look, (expected, make) in enumerate(JSONL_CHANGES[change], 2):
            before = len(path.read_text().splitlines())
            make(path)
            watching.wait(look)
            last = watching.looks()[-1]
            assert last["mode"] == expected, last
            if expected == "append":
                # The binary read only the appended rows: no record judged before is judged again.
                assert last["records_sent"] == last["rows_read"] == len(path.read_text().splitlines()) - before
            else:
                assert last["records_sent"] == last["rows_read"] == len(path.read_text().splitlines()), last
            assert_parity(watching, fresh(tmp_path, path), store=False)
    finally:
        watching.stop()


def test_a_threshold_changed_between_two_watch_runs(binary, tmp_path):
    path = tmp_path / f"{RUN}.rank0.jsonl"
    write_jsonl(path, records(range(60), spike=40))
    for args, thresholds in (((), {}), (("--update-window", "20"), {"update_window": 20})):
        watching = Watching(tmp_path, path, *args)
        try:
            watching.wait(1)
            assert [look["mode"] for look in watching.looks()] == ["full"]
            assert_parity(watching, fresh(tmp_path, path, **thresholds), store=False)
            write_jsonl(path, records(range(60 + len(args) * 5, 65 + len(args) * 5)), "a")
            watching.wait(2)
            assert watching.looks()[-1]["mode"] == "append"
            assert_parity(watching, fresh(tmp_path, path, **thresholds), store=False)
        finally:
            watching.stop()


STORE_CHANGES = {
    "append": [("append", lambda f: write_part(f, records(range(40, 60))))],
    "late_rows": [("full", lambda f: write_part(f, records(range(10, 15)), name="part-00000010-00000014.parquet"))],
    "resume_below": [("full", lambda f: write_part(f, records(range(30, 50), attempt=1), attempt=1))],
    "truncation": [("full", lambda f: (f / "rank=0" / "attempt=0" / "part-00000020-00000039.parquet").unlink())],
    "replaced": [("full", lambda f: write_part(f, records(range(20, 40), spike=30), name="part-00000020-00000039.parquet"))],
    "two_in_succession": [("append", lambda f: write_part(f, records(range(40, 50)))),
                          ("append", lambda f: write_part(f, records(range(50, 70), spike=60)))],
}


@pytest.mark.parametrize("change", sorted(STORE_CHANGES))
def test_the_watch_command_on_a_store_folder(binary, tmp_path, change):
    folder = tmp_path / "runs" / RUN
    write_part(folder, records(range(20), spike=12))
    write_part(folder, records(range(20, 40)))
    watching = Watching(tmp_path, folder)
    try:
        watching.wait(1)
        assert_parity(watching, fresh(tmp_path, folder), store=True)
        assert watching.looks()[-1]["mode"] == "full"
        for look, (expected, make) in enumerate(STORE_CHANGES[change], 2):
            before = len(ev.load_records(folder, all_attempts=True))
            make(folder)
            watching.wait(look)
            last = watching.looks()[-1]
            now = len(ev.load_records(folder, all_attempts=True))
            assert last["mode"] == expected, last
            if expected == "append":
                assert last["records_sent"] == last["rows_read"] == now - before
            else:
                assert last["records_sent"] == last["rows_read"] == now, last
            assert_parity(watching, fresh(tmp_path, folder), store=True)
    finally:
        watching.stop()


# ---- A threshold change while one watcher runs (the author's follow-up) ----------------------------


def _serve(store: Path, rows: list[dict], name: str, *extra: str) -> dict:
    """Send `rows` to a real tensorquill-server on `store` (started with `extra`), wait for its report
    on rank 0 to be current, stop it, and give the report."""
    from test_flight_e2e import Server
    from tensorquill.flight import send_file

    server = Server(store, 0, "--flush-steps", "50", "--flush-seconds", "0.2", *extra)
    try:
        path = store.parent / f"{name}.rank0.jsonl"
        write_jsonl(path, rows)
        assert send_file(path, server.endpoint, RUN, 0, timeout=30.0)
        folder = store / "runs" / RUN / "rank=0"
        deadline = time.monotonic() + 30
        while ev.read_server_report(folder)["state"] != "current":
            assert time.monotonic() < deadline, "the server did not report on rank 0"
            time.sleep(0.05)
    finally:
        server.terminate()
    return json.loads((store / "runs" / RUN / "rank=0" / "incidents.json").read_text())


def test_a_threshold_change_while_one_watcher_runs_is_judged_again_in_full(binary, tmp_path):
    """The watcher takes the thresholds a rank's server applied from its report (`thresholds`), and judges
    the stored parts itself while that report is behind them. A server restarted with `--thresholds`
    writes its new ones there. Here the report stays behind (as when that server stopped after storing
    and before reporting), so the one watcher process judges the parts itself throughout: first under
    the defaults, then, the report's thresholds changed with no new record, under the new ones (a full
    look, by the counter, and the evidence a fresh read under them gives), then an append
    incrementally again. The report is the server's own (rows 0 to 19), and the new `thresholds`
    block the one a server started with `--update-window` 20 writes."""
    store = tmp_path / "store"
    report = _serve(store, records(range(20), spike=12), "first")
    other = _serve(tmp_path / "other", records(range(5)), "other", "--thresholds", json.dumps({"update_window": 20}))
    assert other["thresholds"]["update_window"] == 20 and report["thresholds"]["update_window"] == 100
    folder = store / "runs" / RUN
    (folder / "watchdog.json").unlink(missing_ok=True)  # the stopped server's status: no part of this test
    write_part(folder, records(range(20, 40)))  # stored after the report, which is now behind
    assert ev.read_server_report(folder / "rank=0")["state"] == "behind"
    watching = Watching(tmp_path, folder, "--report-grace", "0")
    try:
        watching.wait(1)
        assert [look["mode"] for look in watching.looks()] == ["full"]
        assert_parity(watching, fresh(tmp_path, folder), store=True)
        # The rank's report now lists the thresholds a server started with --update-window 20 applied.
        changed = folder / "rank=0" / "incidents.json.new"
        changed.write_text(json.dumps({**report, "thresholds": other["thresholds"]}))
        os.replace(changed, folder / "rank=0" / "incidents.json")
        watching.wait(2)
        last = watching.looks()[-1]
        assert last["mode"] == "full" and last["why"] == "the thresholds changed", last
        assert last["records_sent"] == last["rows_read"] == 40
        assert_parity(watching, fresh(tmp_path, folder, update_window=20), store=True)
        # The same process then takes a plain append incrementally, under the new thresholds.
        write_part(folder, records(range(40, 60), spike=50))
        watching.wait(3)
        last = watching.looks()[-1]
        assert (last["mode"], last["records_sent"], last["rows_read"]) == ("append", 20, 20), last
        assert_parity(watching, fresh(tmp_path, folder, update_window=20), store=True)
        assert watching.process.poll() is None, "the watcher was not the same process throughout"
    finally:
        watching.stop()


# ---- Protocol 4: the pair and a mismatch, through the command (the author's follow-up 2) ------------


def test_this_trees_binary_and_sdk_are_a_pair_of_protocol_4(binary):
    done = subprocess.run([binary, "protocol"], capture_output=True, text=True, timeout=30)
    assert (done.returncode, done.stdout, done.stderr) == (0, "4\n", "") and sd.PROTOCOL == 4
    assert sd.rust_ready(binary) == binary


def _speaks_3(tmp: Path) -> Path:
    """A binary of protocol 3, the one before `detect --follow`: it answers `protocol` with 3, and would
    fail anything else (it is never run for more)."""
    path = tmp / "old" / "tensorquill-server"
    path.parent.mkdir(parents=True)
    path.write_text('#!/bin/sh\nif [ "$1" = protocol ]; then echo 3; exit 0; fi\necho "ran $*" >&2; exit 3\n')
    path.chmod(0o755)
    return path


def _command(tmp: Path, binary: Path, *args: str) -> subprocess.CompletedProcess:
    path = tmp / f"{RUN}.rank0.jsonl"
    if not path.exists():
        write_jsonl(path, records(range(30)))
    env = {**os.environ, "PYTHONPATH": str(SRC), sd.BINARY_ENV: str(binary)}
    return subprocess.run([sys.executable, "-m", "tensorquill", str(path), *args, "--interval", "0.05",
                           "-o", str(tmp / "out.json")], cwd=tmp, env=env, capture_output=True, text=True, timeout=120)


MISMATCH = "speaks protocol 3 and this tensorquill needs protocol 4: install a tensorquill and a tensorquill-server that match"


@pytest.mark.parametrize("args", [("--watch",), ("--watch", "--detectors", "rust")], ids=["watch", "watch_rust"])
def test_a_watch_with_a_binary_of_protocol_3_is_refused_before_any_look(tmp_path, args):
    done = _command(tmp_path, _speaks_3(tmp_path), *args)
    assert done.returncode == 2 and "Traceback" not in done.stderr
    last = done.stderr.strip().splitlines()[-1]
    # argparse names the program "__main__.py" up to Python 3.13 and "python -m tensorquill" from 3.14.
    assert last.startswith(("__main__.py: error: --watch needs the Rust detectors: ",
                            "python -m tensorquill: error: --watch needs the Rust detectors: ")) and MISMATCH in last, last
    assert last.endswith("To watch with the Python detectors, add --detectors python")
    assert "ran " not in done.stderr and not (tmp_path / "out.json").exists()


def test_one_read_asking_for_rust_with_a_binary_of_protocol_3_is_refused(tmp_path):
    done = _command(tmp_path, _speaks_3(tmp_path), "--detectors", "rust")
    assert done.returncode == 2 and MISMATCH in done.stderr.strip().splitlines()[-1]
    assert "ran " not in done.stderr and not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("args", [(), ("--detectors", "auto")], ids=["default", "auto"])
def test_one_read_with_auto_and_a_binary_of_protocol_3_uses_the_python_detectors_and_says_so(tmp_path, args):
    done = _command(tmp_path, _speaks_3(tmp_path), *args)
    assert done.returncode == 0, done.stderr
    warned = [line for line in done.stderr.splitlines() if line.startswith("tensorquill: warning: ")]
    assert len(warned) == 1 and "using the Python detectors, not the Rust ones" in warned[0] and MISMATCH in warned[0]
    assert "ran " not in done.stderr
    written = json.loads((tmp_path / "out.json").read_text())
    assert "detectors" not in written  # the Python detectors' evidence: no `"detectors": "rust"`


# ---- The detector child's life (the author's follow-up 2) -----------------------------------------------


def _followers(binary: str) -> list[tuple[int, int]]:
    """(pid, parent pid) of each `detect --follow` process of `binary`, from the process table."""
    listed = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in listed.splitlines():
        pid, ppid, command = line.split(None, 2)
        if command.startswith(binary) and "--follow" in command:
            found.append((int(pid), int(ppid)))
    return found


def _gone(pid: int, within: float = 30.0) -> bool:
    """Whether process `pid` is gone, waiting up to `within` seconds (a bound, not a timing asserted)."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode != 0:
            return True
        time.sleep(0.05)
    return False


def _watching_with_child(tmp: Path, binary: str) -> tuple["Watching", int, Path]:
    """The command watching a `.jsonl` file with the Rust detectors, its follow child running, and the
    command's own temporary folder (TMPDIR), where the child's standard error file would be."""
    path = tmp / f"{RUN}.rank0.jsonl"
    write_jsonl(path, records(range(40)))
    scratch = tmp / "tmpdir"
    scratch.mkdir()
    os.environ["TMPDIR"], before = str(scratch), os.environ.get("TMPDIR")
    try:
        watching = Watching(tmp, path)
    finally:
        os.environ.pop("TMPDIR") if before is None else os.environ.__setitem__("TMPDIR", before)
    watching.wait(1)
    children = [pid for pid, ppid in _followers(binary) if ppid == watching.process.pid]
    assert len(children) == 1, "the watcher has no detect --follow child"
    return watching, children[0], scratch


def test_the_child_ends_with_a_watch_that_ends_by_itself(binary, tmp_path):
    """`watch` with `polls`: when the last look is done, so is the child."""
    path = tmp_path / f"{RUN}.rank0.jsonl"
    write_jsonl(path, records(range(40)))
    seen: list[int] = []

    def grow(evidence, written):
        seen.extend(pid for pid, ppid in _followers(binary) if ppid == os.getpid())
        if len(seen) == 1:
            write_jsonl(path, records(range(40, 45)), "a")

    ev.watch(path, tmp_path / "out.json", interval=0.0, polls=3, on_update=grow, detectors="rust")
    assert len(seen) == 2 and len(set(seen)) == 1, "one child, the same at both looks"
    assert _gone(seen[0]) and not [pid for pid, _ in _followers(binary) if pid == seen[0]]


def test_one_read_by_the_command_starts_no_follow_child(binary, tmp_path):
    """The command line that ends by itself, one read with the Rust detectors: `detect --samples`, never
    `--follow`, and nothing left behind."""
    before = {pid for pid, _ in _followers(binary)}
    done = _command(tmp_path, Path(binary), "--detectors", "rust")
    assert done.returncode == 0, done.stderr
    assert {pid for pid, _ in _followers(binary)} <= before


@pytest.mark.parametrize("how", ["sigint", "sigterm", "sigkill"])
def test_the_child_ends_when_the_command_is_stopped(binary, tmp_path, how):
    """Ctrl-C: the watcher closes the child on its way out. SIGTERM and SIGKILL end the command without
    it: the child then reads the end of its input and ends by itself."""
    watching, child, scratch = _watching_with_child(tmp_path, binary)
    if how == "sigint":
        watching.stop()
    else:
        watching.process.send_signal(signal.SIGTERM if how == "sigterm" else signal.SIGKILL)
        assert watching.process.wait(30) != 0
    assert _gone(child), f"the detect --follow child outlived the command ({how})"
    assert list(scratch.iterdir()) == [], "a temporary file was left"


def test_the_child_ends_when_a_look_raises(binary, tmp_path):
    """A record that is not valid, appended mid-watch, ends the command (exit 2, section 1); the child
    ends with it."""
    watching, child, scratch = _watching_with_child(tmp_path, binary)
    with (tmp_path / f"{RUN}.rank0.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({**records(range(40, 41))[0], "step": -1}) + "\n")
    assert watching.process.wait(60) == 2
    assert "tensorquill: error: step is -1" in watching.process.stderr.read()
    assert _gone(child) and list(scratch.iterdir()) == []
