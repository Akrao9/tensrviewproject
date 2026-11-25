"""Section 5's matrix across a real protocol 4 build (WAVE_AL_CONTRACT.md, section 5; README "An older
server or an older SDK" and "A downgrade is not supported").

    git clone <this repository> <old> && git -C <old> checkout 85a477a      # the last protocol 4 commit
    cargo build --release --manifest-path <old>/server/Cargo.toml --bin tensorquill-server
    cargo build --release --manifest-path server/Cargo.toml --bin tensorquill-server
    .venv/bin/python experiments/protocol-v5/cross_version.py --old-checkout <old> \\
        --old-binary <old>/server/target/release/tensorquill-server \\
        --new-binary server/target/release/tensorquill-server --out experiments/protocol-v5/results/cross_version.json

"The old SDK" is `<old>/sdk/src`, "the new SDK" this tree's `sdk/src`. Every SDK call runs in a child
process of this script with that SDK first on `sys.path` (`--child`), so the two never share a process;
every server is the real release binary, on a real store in a temporary folder. Nothing is timed.

Rows (each records what happened, with the exact messages, and what section 5 and the README say):
1. The new sender to the old server: refused, nothing acknowledged, the lines kept, `OlderServer` said
   once; fresh, and resumed from a saved offset (the stream's first batch is the empty one that asks
   for the store). Then the old server is replaced by the new one on the same store and port, and the
   same sender finishes.
2. The old sender to the new server: stored as sent (protocol 4's layout), judged from `layers_json`,
   kept across a restart; then the new sender continues the rank, and the new `load_records` reads
   both layouts in step order.
3. The new SDK with the old binary, and the old SDK with the new binary: `detectors="rust"`,
   `detectors="auto"`, and the command line's one read and `--watch`.
4. The old `load_records` and `runs` on a store of protocol 5 parts.
5. The old server restarted on a store that holds protocol 5 parts (the downgrade, Q9 a).

Step 2's scalar-only records (section 4), added for the run on the joined tree:
6. The new sender, with scalar-only records in its file, to the old server; then the new server in its
   place: where the rows with a `seq` go, the offset, and the new `load_records` with and without
   `scalars=True`.
7. The old readers on that store, which holds `scalars-*` objects beside protocol 5 parts.
8. The old `runs` on that store; 8b, the old `runs` and `load_records` on a new `.jsonl` holding a
   scalar-only record (section 5's "such a file").
9. The old sender to the new server, and the new `load_records` with and without `scalars=True` on what
   it stored: (a) a file of training records; (b) a protocol 5 capture's file, training steps 0 to 9, a
   scalar-only record at step 500, training steps 10 to 29 (the owner's rule of Nov 25: under metadata 1
   a version 2 row is dropped and counted). Then, measured and not judged, the same file sent by the
   new `send_file` to the same server: (c) as the old sender left its offset file, and (d) from byte 0,
   the offset file removed.
10. A batch whose `min_step` and `max_step` are null: the new sender resumed with only scalar-only
   lines left to send, to the old server, then to the new; and the old sender's own batch with no steps
   (its resumed stream's first) to the old server, as the control.
11. The downgrade with `scalars-*` objects: the old server restarted on row 6's store, given later training
   steps of that rank by the old sender. If it read the object holding step 500 as a part, it would drop
   them as not newer.

The records: one rank, layers in a model's order, an `activation_spike` at step 35 and one at 85. The
detectors give one incident whose `affected_steps` counts the spikes they read, so it says which
parts' layers a server read: the old sender writes steps 0 to 59, the new one 60 to 119.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORWARD = [f"model.layers.{i}" for i in (0, 1, 2, 10, 11, 3)]
SPIKES = (35, 85)


# --- the records --------------------------------------------------------------------------------


def layer(std: float = 1.0, abs_max: float = 3.0) -> dict:
    return {"mean": 0.0, "std": std, "abs_max": abs_max, "zero_frac": 0.0}


def records(run_id: str, steps: range) -> list[dict]:
    out = []
    for s in steps:
        layers = {n: layer(std=1.0 + 0.01 * (s % 3), abs_max=3.0 * (1 + 0.01 * (s % 5))) for n in FORWARD}
        if s in SPIKES:
            layers["model.layers.10"] = layer(abs_max=60.0)
            layers["model.layers.2"] = layer(abs_max=60.0)
        out.append({"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 1, "step": s, "ts": float(s),
                    "grad_norm": 1.0, "layers": layers})
    return out


def scalar(run_id: str, step: int, seq: int, value: float) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "attempt": 1, "step": step,
            "seq": seq, "ts": 1000.0 + seq, "scalars": {"eval/loss": value}}


def with_scalars(run_id: str, steps: range, at: dict[int, list[tuple[int, int, float]]]) -> list[dict]:
    """`records(run_id, steps)` with scalar-only records after the training record of each step in
    `at`: `{after_step: [(step, seq, value), ...]}`."""
    out = []
    for record in records(run_id, steps):
        out.append(record)
        out.extend(scalar(run_id, s, q, v) for s, q, v in at.get(record["step"], []))
    return out


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --- the child: one SDK call in a process of its own ----------------------------------------------


def child(task: str, sdk: str, args: dict) -> None:
    sys.path.insert(0, sdk)
    import warnings

    import tensorquill
    from tensorquill import evidence, flight

    assert Path(tensorquill.__file__).resolve().is_relative_to(Path(sdk).resolve()), tensorquill.__file__
    out: dict = {}
    if task == "send_file":
        out["confirmed"] = flight.send_file(args["path"], args["endpoint"], args["run_id"], 0, timeout=args["timeout"])
        out["offset"] = flight.read_offset(flight.offset_path(args["path"]))
    elif task == "sender_swap":
        sender = flight.FlightSender(args["path"], args["endpoint"], args["run_id"], 0, backoff_cap=args["backoff_cap"])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and type(sender.last_error).__name__ != "OlderServer":
            time.sleep(0.1)
        time.sleep(1.0)  # one more retry or two, to show the message is said once
        print(json.dumps({"phase": "refused", "last_error_type": type(sender.last_error).__name__,
                          "last_error": str(sender.last_error), "unacked_bytes": sender.unacked_bytes(),
                          "offset_file": flight.read_offset(flight.offset_path(args["path"]))}), flush=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not Path(args["go"]).exists():
            time.sleep(0.05)
        out["confirmed"] = sender.close(timeout=args["timeout"])
        out["last_error"] = None if sender.last_error is None else f"{type(sender.last_error).__name__}: {sender.last_error}"
        out["offset"] = flight.read_offset(flight.offset_path(args["path"]))
    elif task == "load_records":
        try:
            options = {k: args[k] for k in ("scalars", "all_attempts") if k in args}  # the old SDK has no scalars=
            got = evidence.load_records(args["source"], **options)
            want = [json.loads(line) for line in Path(args["file"]).read_text().splitlines()] if args.get("file") else None
            training = [r for r in got if r.get("record") != "scalars"]
            out["records"] = len(got)
            out["scalar_only"] = len(got) - len(training)
            out["steps_in_order"] = [r["step"] for r in training] == sorted(r["step"] for r in training)
            out["equal_to_file"] = None if want is None else got == want
            if want is not None:
                kept = [r for r in want if r.get("record") != "scalars"]
                out["training_equal_to_file_s"] = training == kept
                out["scalar_only_equal_to_file_s"] = (sorted((r for r in got if r.get("record") == "scalars"), key=lambda r: r["seq"])
                                                      == sorted((r for r in want if r.get("record") == "scalars"), key=lambda r: r["seq"]))
        except Exception as error:  # noqa: BLE001
            out["raised"] = f"{type(error).__name__}: {error}"
    elif task == "binding":
        try:
            out["binding"] = flight.read_binding(flight.offset_path(args["path"]))
        except Exception as error:  # noqa: BLE001
            out["raised"] = f"{type(error).__name__}: {error}"
    elif task == "send_count":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sender = flight.FlightSender(args["path"], args["endpoint"], args["run_id"], 0)
            out["confirmed"] = sender.close(timeout=args["timeout"])
        out["skipped_lines"] = getattr(sender, "skipped_lines", None)
        out["last_error"] = None if sender.last_error is None else f"{type(sender.last_error).__name__}: {sender.last_error}"
        out["offset"] = flight.read_offset(flight.offset_path(args["path"]))
        out["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    elif task == "evidence":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                built = evidence.build_evidence(args["file"], detectors=args["detectors"])
                out["detectors"] = built.get("detectors")
                out["incidents"] = [(i["kind"], i.get("step"), i.get("affected_steps")) for i in built["incidents"]]
            except Exception as error:  # noqa: BLE001
                out["raised"] = f"{type(error).__name__}: {error}"
        out["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    print(json.dumps(out, default=str), flush=True)


# --- the parent -----------------------------------------------------------------------------------


class Context:
    def __init__(self, args: argparse.Namespace, tmp: Path) -> None:
        self.python = args.python
        self.old_sdk = str((args.old_checkout / "sdk" / "src").resolve())
        self.new_sdk = str((ROOT / "sdk" / "src").resolve())
        self.old_binary = str(args.old_binary.resolve())
        self.new_binary = str(args.new_binary.resolve())
        self.tmp = tmp
        self.hide = [(str(tmp), "<tmp>"), (str(tmp.resolve()), "<tmp>"), (str(args.old_checkout.resolve()), "<old>"),
                     (str(ROOT), "<new>"), (self.new_binary, "<new-binary>"), (self.old_binary, "<old-binary>"),
                     (str(Path.home()), "~")]

    def clean(self, value):
        """Every path in a message, as a name: no local path is written to the results."""
        if isinstance(value, str):
            for path, name in self.hide:
                value = value.replace(path, name)
            return value
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, tuple):
            return [self.clean(v) for v in value]
        if isinstance(value, dict):
            return {k: self.clean(v) for k, v in value.items()}
        return value

    def env(self, sdk: str, binary: str | None = None) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_ENDPOINT")}
        env["PYTHONPATH"] = sdk
        if binary:
            env["TENSORQUILL_SERVER_BIN"] = binary
        else:
            env.pop("TENSORQUILL_SERVER_BIN", None)
        return env

    def child(self, task: str, sdk: str, binary: str | None = None, seconds: float = 180, **args) -> dict:
        done = subprocess.run([self.python, __file__, "--child", task, "--sdk", sdk, "--args", json.dumps(args)],
                              capture_output=True, text=True, timeout=seconds, env=self.env(sdk, binary), cwd=self.tmp)
        lines = done.stdout.strip().splitlines()
        try:
            out = json.loads(lines[-1])
        except (IndexError, ValueError):
            out = {"child_failed": done.returncode, "stdout": done.stdout[-1500:]}
        out["stderr"] = [line for line in done.stderr.splitlines() if line.strip()]
        return out

    def cli(self, sdk: str, binary: str, argv: list[str], cwd: Path, timeout: float = 60) -> dict:
        before = set(p.name for p in cwd.iterdir())
        try:
            done = subprocess.run([self.python, "-m", "tensorquill", *argv], capture_output=True, text=True,
                                  timeout=timeout, env=self.env(sdk, binary), cwd=cwd)
            out = {"exit": done.returncode, "stderr": [line for line in done.stderr.splitlines() if line.strip()]}
        except subprocess.TimeoutExpired as expired:
            out = {"exit": None, "timed_out_after_seconds": timeout,
                   "stderr": [line for line in (expired.stderr or b"").decode().splitlines() if line.strip()]}
        out["files_written"] = sorted(set(p.name for p in cwd.iterdir()) - before)
        return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, binary: str, store: Path, port: int) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen(
            [binary, "--listen", f"127.0.0.1:{port}", "--store", str(store), "--watchdog-seconds", "0.5",
             "--flush-seconds", "0.3"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.ready = self.process.stdout.readline().strip()
        self.endpoint = f"grpc://127.0.0.1:{port}"

    def close(self) -> list[str]:
        self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return [line for line in err.splitlines() if line.strip()]


def parts(store: Path, run_id: str) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for path in sorted((store / "runs" / run_id).rglob("part-*.parquet")):
        names = pq.read_schema(path).names
        steps = pq.read_table(path, columns=["step"]).column(0).to_pylist()
        rows = pq.read_table(path, columns=["schema_version"]).column(0).to_pylist()
        out.append({"part": path.name, "layout": "protocol 5" if "layers" in names else "protocol 4",
                    "steps": [min(steps), max(steps)], "row_versions": sorted(set(rows)), "rows": len(steps),
                    **({"step_list": steps} if len(steps) <= 64 else {})})
    return out


def scalar_objects(store: Path, run_id: str) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for path in sorted((store / "runs" / run_id).rglob("scalars-*.parquet")):
        seq = pq.read_table(path, columns=["seq"]).column(0).to_pylist()
        steps = pq.read_table(path, columns=["step"]).column(0).to_pylist()
        out.append({"object": str(path.relative_to(store / "runs" / run_id)), "rows": len(seq), "seq": [min(seq), max(seq)],
                    "steps": steps})
    return out


def watchdog_rank0(store: Path, run_id: str):
    try:
        document = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
    except (OSError, ValueError) as error:
        return f"unreadable: {error}"
    return next((r for r in document.get("ranks", []) if r.get("rank") == 0), None)


def runs_json(ctx: "Context", sdk: str, target: Path) -> dict:
    done = subprocess.run([ctx.python, "-m", "tensorquill", "runs", str(target), "--json"], capture_output=True, text=True,
                          timeout=120, env=ctx.env(sdk), cwd=ctx.tmp)
    try:
        listed = json.loads(done.stdout)
    except ValueError:
        listed = done.stdout[-1500:]
    return {"exit": done.returncode, "listed": listed, "stderr": [line for line in done.stderr.splitlines() if line.strip()]}


def swap(ctx: "Context", run_id: str, path: Path, store: Path, port: int, last_step: int) -> dict:
    """The new sender started against the old server on `store`, refused; then the new server in its
    place on the same store and port, and the same sender finishes (row 1's method)."""
    entry: dict = {}
    lines_before = path.read_text().count("\n")
    parts_before = parts(store, run_id)
    old = Server(ctx.old_binary, store, port)
    go = ctx.tmp / f"go-{run_id}"
    proc = subprocess.Popen([ctx.python, __file__, "--child", "sender_swap", "--sdk", ctx.new_sdk, "--args",
                             json.dumps({"path": str(path), "endpoint": old.endpoint, "run_id": run_id,
                                         "backoff_cap": 2.0, "go": str(go), "timeout": 90})],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ctx.env(ctx.new_sdk), cwd=ctx.tmp)
    refused = json.loads(proc.stdout.readline())
    entry["old_server"] = {"ready": old.ready, "sender": refused, "parts_before": parts_before,
                           "parts_after": parts(store, run_id), "scalar_objects_after": scalar_objects(store, run_id),
                           "lines_in_file": path.read_text().count("\n"), "lines_before": lines_before}
    entry["old_server"]["stderr"] = old.close()
    new = Server(ctx.new_binary, store, port)
    go.write_text("go")
    stdout, stderr = proc.communicate(timeout=180)
    entry["new_server"] = {"ready": new.ready, "sender": json.loads(stdout.strip().splitlines()[-1])}
    stderr_lines = [line for line in stderr.splitlines() if line.strip()]
    entry["sender_stderr"] = stderr_lines
    entry["older_message_said"] = says(stderr_lines, "speaks an older protocol")
    entry["new_server"]["report"] = report_through(store, run_id, last_step)
    entry["new_server"]["stderr"] = new.close()
    entry["new_server"]["parts"] = parts(store, run_id)
    entry["new_server"]["scalar_objects"] = scalar_objects(store, run_id)
    try:
        entry["new_server"]["watchdog_rank0"] = next(
            (r for r in json.loads((store / "runs" / run_id / "watchdog.json").read_text()).get("ranks", [])
             if r.get("rank") == 0), None)
    except (OSError, ValueError) as error:
        entry["new_server"]["watchdog_rank0"] = f"unreadable: {error}"
    return entry


def report_through(store: Path, run_id: str, last_step: int, seconds: float = 30) -> dict | None:
    path = store / "runs" / run_id / "rank=0" / "incidents.json"
    deadline = time.monotonic() + seconds
    report = None
    while time.monotonic() < deadline:
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            report = None
        if report and report.get("last_step") == last_step:
            break
        time.sleep(0.1)
    if report is None:
        return None
    return {"last_step": report.get("last_step"),
            "incidents": [(i["kind"], i.get("step"), i.get("affected_steps")) for i in report.get("incidents", [])],
            "activation_spike": (report.get("kind_checks", {}).get("activation_spike") or {}).get("value")}


def says(lines: list[str], text: str) -> int:
    return sum(text in line for line in lines)


# --- the rows -------------------------------------------------------------------------------------


def row_1(ctx: Context) -> dict:
    """The new sender to the old server, fresh and resumed; then the new server in its place."""
    out = {}
    for case in ("fresh", "resumed"):
        run_id = f"xv-{case}"
        store, port = ctx.tmp / f"store-1-{case}", free_port()
        path = ctx.tmp / f"{run_id}.rank0.jsonl"
        entry: dict = {}
        if case == "resumed":  # a saved offset, bound to this store: the new sender sent 20 lines to the new server
            append(path, records(run_id, range(0, 20)))
            server = Server(ctx.new_binary, store, port)
            entry["before"] = ctx.child("send_file", ctx.new_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id,
                                        timeout=30)
            server.close()
            append(path, records(run_id, range(20, 40)))
        else:
            append(path, records(run_id, range(0, 40)))
        lines_before = path.read_text().count("\n")
        parts_before = parts(store, run_id)
        old = Server(ctx.old_binary, store, port)
        go = ctx.tmp / f"go-{case}"
        proc = subprocess.Popen([ctx.python, __file__, "--child", "sender_swap", "--sdk", ctx.new_sdk, "--args",
                                 json.dumps({"path": str(path), "endpoint": old.endpoint, "run_id": run_id,
                                             "backoff_cap": 2.0, "go": str(go), "timeout": 90})],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ctx.env(ctx.new_sdk), cwd=ctx.tmp)
        refused = json.loads(proc.stdout.readline())
        entry["old_server"] = {"ready": old.ready, "sender": refused, "parts_after": parts(store, run_id),
                               "parts_before": parts_before, "lines_in_file": path.read_text().count("\n"),
                               "lines_before": lines_before}
        entry["old_server"]["stderr"] = old.close()
        new = Server(ctx.new_binary, store, port)  # the same store and port
        go.write_text("go")
        stdout, stderr = proc.communicate(timeout=180)
        entry["new_server"] = {"ready": new.ready, "sender": json.loads(stdout.strip().splitlines()[-1])}
        stderr_lines = [line for line in stderr.splitlines() if line.strip()]
        entry["sender_stderr"] = stderr_lines
        entry["older_message_said"] = says(stderr_lines, "speaks an older protocol")
        entry["new_server"]["report"] = report_through(store, run_id, 39)
        entry["new_server"]["stderr"] = new.close()
        entry["new_server"]["parts"] = parts(store, run_id)
        entry["new_server"]["load_records"] = ctx.child("load_records", ctx.new_sdk, source=str(store / "runs" / run_id),
                                                        file=str(path))
        out[case] = entry
    return out


def row_2(ctx: Context) -> dict:
    """The old sender to the new server, a restart, then the new sender on the same rank."""
    run_id, store, port = "xv-mixed", ctx.tmp / "store-2", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    append(path, records(run_id, range(0, 60)))
    out: dict = {}
    server = Server(ctx.new_binary, store, port)
    out["old_sender"] = ctx.child("send_file", ctx.old_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id, timeout=30)
    out["old_sender"]["report"] = report_through(store, run_id, 59)
    out["old_sender"]["parts"] = parts(store, run_id)
    out["old_sender"]["server_stderr"] = server.close()
    server = Server(ctx.new_binary, store, port)  # the restart
    append(path, records(run_id, range(60, 120)))
    out["new_sender_after_restart"] = ctx.child("send_file", ctx.new_sdk, path=str(path), endpoint=server.endpoint,
                                                run_id=run_id, timeout=30)
    out["new_sender_after_restart"]["report"] = report_through(store, run_id, 119)
    out["new_sender_after_restart"]["server_stderr"] = server.close()
    out["parts"] = parts(store, run_id)
    out["new_load_records"] = ctx.child("load_records", ctx.new_sdk, source=str(store / "runs" / run_id), file=str(path))
    return out


def row_3(ctx: Context) -> dict:
    """Each SDK with the other's binary (and with its own, as the control)."""
    path = ctx.tmp / "xv-mixed.rank0.jsonl"
    out = {}
    for name, sdk, binary in (("new_sdk_old_binary", ctx.new_sdk, ctx.old_binary),
                              ("old_sdk_new_binary", ctx.old_sdk, ctx.new_binary),
                              ("new_sdk_new_binary", ctx.new_sdk, ctx.new_binary),
                              ("old_sdk_old_binary", ctx.old_sdk, ctx.old_binary)):
        entry = {}
        for detectors in ("rust", "auto"):
            entry[f"build_evidence_{detectors}"] = ctx.child("evidence", sdk, binary, file=str(path), detectors=detectors)
        if name in ("new_sdk_old_binary", "old_sdk_new_binary"):
            for label, argv in (("cli_rust", ["--detectors", "rust"]), ("cli_auto", []), ("cli_watch", ["--watch"])):
                cwd = ctx.tmp / f"cli-{name}-{label}"
                cwd.mkdir()
                (cwd / path.name).write_text(path.read_text())
                entry[label] = ctx.cli(sdk, binary, [path.name, *argv], cwd, timeout=60)
        out[name] = entry
    return out


def row_4(ctx: Context) -> dict:
    """The old readers on a store of protocol 5 parts (row 1's fresh store, written by the new sender)."""
    store = ctx.tmp / "store-1-fresh"
    out = {"parts": parts(store, "xv-fresh"),
           "old_load_records": ctx.child("load_records", ctx.old_sdk, source=str(store / "runs" / "xv-fresh"))}
    done = subprocess.run([ctx.python, "-m", "tensorquill", "runs", str(store), "--json"], capture_output=True, text=True,
                          timeout=120, env=ctx.env(ctx.old_sdk), cwd=ctx.tmp)
    try:
        out["old_runs"] = json.loads(done.stdout)
    except ValueError:
        out["old_runs"] = done.stdout[-1500:]
    out["old_runs_exit"] = done.returncode
    out["old_runs_stderr"] = [line for line in done.stderr.splitlines() if line.strip()]
    cwd = ctx.tmp / "cli-old-one-read"
    cwd.mkdir()
    out["old_one_read"] = ctx.cli(ctx.old_sdk, ctx.old_binary, [str(store / "runs" / "xv-fresh"), "--detectors", "python"], cwd)
    done = subprocess.run([ctx.python, "-m", "tensorquill", "runs", str(store), "--json"], capture_output=True, text=True,
                          timeout=120, env=ctx.env(ctx.new_sdk), cwd=ctx.tmp)
    try:
        out["new_runs_control"] = json.loads(done.stdout)
    except ValueError:
        out["new_runs_control"] = done.stdout[-1500:]
    return out


def row_5(ctx: Context) -> dict:
    """The old server restarted on row 2's store (protocol 4 parts, then protocol 5 parts), given more of
    the rank by the old sender: the rank is rebuilt from every stored part."""
    run_id, store, port = "xv-mixed", ctx.tmp / "store-2", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    out: dict = {"parts_before": parts(store, run_id)}
    server = Server(ctx.old_binary, store, port)
    out["ready"] = server.ready
    append(path, records(run_id, range(120, 140)))
    out["old_sender"] = ctx.child("send_file", ctx.old_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id, timeout=30)
    out["report"] = report_through(store, run_id, 139)
    time.sleep(1.0)
    status = store / "runs" / run_id / "watchdog.json"
    try:
        document = json.loads(status.read_text())
        out["watchdog_rank0"] = next((r for r in document.get("ranks", []) if r.get("rank") == 0), None)
    except (OSError, ValueError) as error:
        out["watchdog_rank0"] = f"unreadable: {error}"
    out["server_stderr"] = server.close()
    out["parts_after"] = parts(store, run_id)
    return out


def row_6(ctx: Context) -> dict:
    """The new sender with scalar-only records to the old server, then the new server in its place."""
    run_id, store, port = "xv-scal", ctx.tmp / "store-6", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    append(path, with_scalars(run_id, range(0, 40), {9: [(500, 0, 1.5)], 19: [(19, 1, 1.4)], 39: [(39, 2, 1.3)]}))
    out = swap(ctx, run_id, path, store, port, 39)
    source = str(store / "runs" / run_id)
    out["new_load_records"] = ctx.child("load_records", ctx.new_sdk, source=source, file=str(path))
    out["new_load_records_scalars"] = ctx.child("load_records", ctx.new_sdk, source=source, file=str(path), scalars=True)
    return out


def row_7(ctx: Context) -> dict:
    """The old readers on row 6's store: `scalars-*` objects beside protocol 5 parts."""
    store, run_id = ctx.tmp / "store-6", "xv-scal"
    objects = sorted((store / "runs" / run_id).rglob("scalars-*.parquet"))
    out = {"parts": parts(store, run_id), "scalar_objects": scalar_objects(store, run_id),
           "old_load_records": ctx.child("load_records", ctx.old_sdk, source=str(store / "runs" / run_id)),
           "old_load_records_of_a_scalars_object": (ctx.child("load_records", ctx.old_sdk, source=str(objects[0]))
                                                    if objects else "no scalars-* object")}
    cwd = ctx.tmp / "cli-old-one-read-6"
    cwd.mkdir()
    out["old_one_read"] = ctx.cli(ctx.old_sdk, ctx.old_binary, [str(store / "runs" / run_id), "--detectors", "python"], cwd)
    return out


def row_8(ctx: Context) -> dict:
    """The old `runs` on row 6's store (with the new `runs` as the control); 8b, on a folder holding
    row 6's `.jsonl` (with scalar-only records), and the old `load_records` on that file."""
    store = ctx.tmp / "store-6"
    out = {"old_runs": runs_json(ctx, ctx.old_sdk, store), "new_runs_control": runs_json(ctx, ctx.new_sdk, store)}
    folder = ctx.tmp / "folder-8b"
    folder.mkdir()
    (folder / "xv-scal.rank0.jsonl").write_text((ctx.tmp / "xv-scal.rank0.jsonl").read_text())
    out["8b"] = {"old_runs": runs_json(ctx, ctx.old_sdk, folder), "new_runs_control": runs_json(ctx, ctx.new_sdk, folder),
                 "old_load_records": ctx.child("load_records", ctx.old_sdk, source=str(folder / "xv-scal.rank0.jsonl")),
                 "new_load_records": ctx.child("load_records", ctx.new_sdk, source=str(folder / "xv-scal.rank0.jsonl"),
                                               file=str(folder / "xv-scal.rank0.jsonl"))}
    return out


def row_9(ctx: Context) -> dict:
    """The old sender to the new server; the new `load_records` with and without `scalars=True` on the
    store. (a) a file of training records; (b) a protocol 5 capture's file with a scalar-only record at
    step 500 between training steps 9 and 10; then (c) and (d), the new `send_file` on (b)'s file."""
    out = {}
    run_id, store, port = "xv-old-a", ctx.tmp / "store-9-a_training_only", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    append(path, records(run_id, range(0, 30)))
    server = Server(ctx.new_binary, store, port)
    entry = {"lines": 30, "old_sender": ctx.child("send_count", ctx.old_sdk, path=str(path), endpoint=server.endpoint,
                                                 run_id=run_id, timeout=30)}
    entry["report"] = report_through(store, run_id, 29)
    entry["server_stderr"] = server.close()
    entry["parts"] = parts(store, run_id)
    entry["scalar_objects"] = scalar_objects(store, run_id)
    source = str(store / "runs" / run_id)
    entry["new_load_records"] = ctx.child("load_records", ctx.new_sdk, source=source, file=str(path))
    entry["new_load_records_scalars"] = ctx.child("load_records", ctx.new_sdk, source=source, file=str(path), scalars=True)
    out["a_training_only"] = entry

    run_id, store, port = "xv-old-b", ctx.tmp / "store-9-b_with_scalar_only", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    append(path, with_scalars(run_id, range(0, 30), {9: [(500, 0, 1.5)]}))
    source = str(store / "runs" / run_id)
    server = Server(ctx.new_binary, store, port)

    def look(entry: dict) -> None:
        time.sleep(1.5)  # the status is written every --watchdog-seconds (0.5): two writes, not a timing
        entry["report"] = report_through(store, run_id, 29, seconds=10)
        entry["watchdog_rank0"] = watchdog_rank0(store, run_id)
        entry["parts"] = parts(store, run_id)
        entry["scalar_objects"] = scalar_objects(store, run_id)
        entry["new_load_records"] = ctx.child("load_records", ctx.new_sdk, source=source, file=str(path))
        entry["new_load_records_scalars"] = ctx.child("load_records", ctx.new_sdk, source=source, file=str(path), scalars=True)

    try:
        entry = {"lines": path.read_text().count("\n"), "file_bytes": path.stat().st_size,
                 "old_sender": ctx.child("send_count", ctx.old_sdk, path=str(path), endpoint=server.endpoint,
                                         run_id=run_id, timeout=30)}
        look(entry)
        entry["local_file_scalars"] = ctx.child("load_records", ctx.new_sdk, source=str(path), file=str(path), scalars=True)
        out["b_with_scalar_only"] = entry
        c = {"binding_before": ctx.child("binding", ctx.new_sdk, path=str(path)),
             "send_file": ctx.child("send_file", ctx.new_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id,
                                    timeout=30)}
        look(c)
        c["binding_after"] = ctx.child("binding", ctx.new_sdk, path=str(path))
        out["c_new_send_file_as_left"] = c
        offset = path.with_suffix(".offset")
        d = {"offset_file_removed": offset.exists()}
        offset.unlink(missing_ok=True)
        d["send_file"] = ctx.child("send_file", ctx.new_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id,
                                   timeout=30)
        look(d)
        d["binding_after"] = ctx.child("binding", ctx.new_sdk, path=str(path))
        out["d_new_send_file_from_byte_0"] = d
    finally:
        out["b_server_stderr"] = server.close()
    return out


def row_10(ctx: Context) -> dict:
    """A batch whose `min_step` and `max_step` are null. (a) The new sender's saved offset is bound to a
    new server's store holding steps 0 to 19; only scalar-only lines are appended, so its resumed stream
    is the batch with no steps that asks for the store, then a batch of scalar-only rows alone: to the old
    server, then the new in its place. (b) The control: the old sender resumed against the old server,
    whose stream opens with its own batch with no steps (metadata 1)."""
    run_id, store, port = "xv-null", ctx.tmp / "store-10", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    append(path, records(run_id, range(0, 20)))
    server = Server(ctx.new_binary, store, port)
    out: dict = {"a": {"before": ctx.child("send_file", ctx.new_sdk, path=str(path), endpoint=server.endpoint,
                                           run_id=run_id, timeout=30)}}
    out["a"]["before"]["report"] = report_through(store, run_id, 19)
    server.close()
    append(path, [scalar(run_id, 19, 0, 1.4), scalar(run_id, 500, 1, 1.5)])
    out["a"].update(swap(ctx, run_id, path, store, port, 19))
    run_id, store, port = "xv-null-old", ctx.tmp / "store-10-old", free_port()
    path = ctx.tmp / f"{run_id}.rank0.jsonl"
    append(path, records(run_id, range(0, 20)))
    server = Server(ctx.old_binary, store, port)
    out["b"] = {"first": ctx.child("send_count", ctx.old_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id, timeout=30)}
    append(path, records(run_id, range(20, 40)))
    out["b"]["resumed"] = ctx.child("send_count", ctx.old_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id, timeout=30)
    out["b"]["report"] = report_through(store, run_id, 39)
    out["b"]["server_stderr"] = server.close()
    out["b"]["parts"] = parts(store, run_id)
    out["b"]["file_bytes"] = path.stat().st_size
    return out


def row_11(ctx: Context) -> dict:
    """The old server restarted on row 6's store (a protocol 5 part of steps 0 to 39 and a scalars-*
    object holding steps 500, 19 and 39); the old sender sends steps 40 to 59 of the same rank from a file
    of its own."""
    run_id, store, port = "xv-scal", ctx.tmp / "store-6", free_port()
    folder = ctx.tmp / "folder-11"
    folder.mkdir()
    path = folder / f"{run_id}.rank0.jsonl"
    append(path, records(run_id, range(40, 60)))
    out: dict = {"parts_before": parts(store, run_id), "scalar_objects_before": scalar_objects(store, run_id)}
    server = Server(ctx.old_binary, store, port)
    out["ready"] = server.ready
    out["old_sender"] = ctx.child("send_count", ctx.old_sdk, path=str(path), endpoint=server.endpoint, run_id=run_id, timeout=30)
    out["report"] = report_through(store, run_id, 59)
    out["server_stderr"] = server.close()
    out["parts_after"] = parts(store, run_id)
    out["scalar_objects_after"] = scalar_objects(store, run_id)
    return out


# --- checks against section 5 and the README -------------------------------------------------------


def checks(r: dict) -> list[dict]:
    """Each claim of section 5's matrix or the README, and whether the run shows it. A False is a finding."""
    out = []

    def claim(row: str, text: str, holds: bool, seen: str) -> None:
        out.append({"row": row, "claim": text, "holds": bool(holds), "seen": seen})

    for case in ("fresh", "resumed"):
        e = r["row_1"][case]
        o, n = e["old_server"], e["new_server"]
        saved = e["before"]["offset"] if case == "resumed" else 0
        claim(f"1 {case}", "the old server refuses the new sender's stream and acknowledges nothing",
              o["parts_after"] == o["parts_before"] and o["sender"]["offset_file"] == saved,
              f"parts before {len(o['parts_before'])}, after {len(o['parts_after'])}; saved offset {saved}, "
              f"offset while refused {o['sender']['offset_file']}; unacked {o['sender']['unacked_bytes']} bytes")
        if case == "resumed":
            claim("1 resumed", "the stream's first batch is the empty one that asks for the store, and that is what is refused",
                  any("received 1 batches, 0 rows" in line and "schema_version 2 is not supported" in line for line in o["stderr"]),
                  " | ".join(o["stderr"]))
        claim(f"1 {case}", "last_error is OlderServer", o["sender"]["last_error_type"] == "OlderServer",
              o["sender"]["last_error_type"])
        claim(f"1 {case}", "the lines stay in the sender's file", o["lines_in_file"] == o["lines_before"] == 40,
              f"{o['lines_in_file']} lines")
        claim(f"1 {case}", "the older-server message is said once", e["older_message_said"] == 1,
              f"said {e['older_message_said']} time(s)")
        claim(f"1 {case}", "with the new server in its place, the same sender finishes and every line is stored",
              n["sender"].get("confirmed") is True and n["load_records"].get("equal_to_file") is True,
              f"confirmed {n['sender'].get('confirmed')}; load_records {n['load_records'].get('records')} records, "
              f"equal to the file {n['load_records'].get('equal_to_file')}")
    e = r["row_2"]
    first = e["old_sender"]
    claim("2", "the old sender's batches are stored as sent, in protocol 4's layout",
          first.get("confirmed") is True and {p["layout"] for p in first["parts"]} == {"protocol 4"}, str(first["parts"]))
    claim("2", "the new server's detectors judge the layers from layers_json",
          bool(first["report"]) and ("activation_spike", 35, 1) in [tuple(i) for i in first["report"]["incidents"]],
          str(first["report"]))
    after = e["new_sender_after_restart"]
    claim("2", "after a restart, and the new sender continuing the rank, both layouts are judged",
          bool(after["report"]) and ("activation_spike", 35, 2) in [tuple(i) for i in after["report"]["incidents"]],
          str(after["report"]))
    claim("2", "the store holds parts of both layouts", {p["layout"] for p in e["parts"]} == {"protocol 4", "protocol 5"},
          str(e["parts"]))
    claim("2", "the new load_records reads them all, in step order",
          e["new_load_records"].get("records") == 120 and e["new_load_records"].get("steps_in_order") is True
          and e["new_load_records"].get("equal_to_file") is True, str({k: v for k, v in e["new_load_records"].items() if k != "stderr"}))
    for name, needs, speaks in (("new_sdk_old_binary", 5, 4), ("old_sdk_new_binary", 4, 5)):
        e = r["row_3"][name]
        message = f"speaks protocol {speaks} and this tensorquill needs protocol {needs}"
        claim(f"3 {name}", 'detectors="rust" raises, with the protocol message',
              message in (e["build_evidence_rust"].get("raised") or ""), e["build_evidence_rust"].get("raised") or "did not raise")
        auto = e["build_evidence_auto"]
        # The evidence names its detectors only when they are Rust's ("detectors": "rust"); Python's has none.
        claim(f"3 {name}", 'detectors="auto" falls back to Python, with the protocol message',
              "raised" not in auto and auto.get("detectors") is None and any(message in w for w in auto.get("warnings", [])),
              f"detectors {auto.get('detectors')}; warnings {auto.get('warnings')}")
        claim(f"3 {name}", "one read with --detectors rust stops with exit 2 and the message, no evidence file",
              e["cli_rust"]["exit"] == 2 and any(message in s for s in e["cli_rust"]["stderr"]) and not e["cli_rust"]["files_written"],
              str(e["cli_rust"]))
        claim(f"3 {name}", "one read with auto uses Python, writes the evidence, warns once with the message",
              e["cli_auto"]["exit"] == 0 and says(e["cli_auto"]["stderr"], message) == 1 and bool(e["cli_auto"]["files_written"]),
              str(e["cli_auto"]))
        claim(f"3 {name}", "--watch stops before the first look with exit 2 and the message, no evidence file",
              e["cli_watch"]["exit"] == 2 and any(message in s for s in e["cli_watch"]["stderr"]) and not e["cli_watch"]["files_written"],
              str(e["cli_watch"]))
    e = r["row_4"]
    raised = e["old_load_records"].get("raised") or ""
    claim("4", "the old load_records refuses a store of protocol 5 parts, naming the part and the row",
          "part-" in raised and "row" in raised, raised or "did not raise")
    listed = [row for row in (e["old_runs"] if isinstance(e["old_runs"], list) else []) if isinstance(row, dict)]
    run = next((row for row in listed if row.get("run_id") == "xv-fresh"), None)
    claim("4", "the old runs lists the run with the protocol 5 server's own report (corrected Nov 24)",
          run is not None and run.get("run_state") == "unhealthy" and run.get("last_step") == 39
          and (run.get("first_incident") or {}).get("kind") == "activation_spike", json.dumps(listed)[:900])
    read = e["old_one_read"]
    claim("4", "the old one read of the store's run refuses it, naming the part and the row",
          read["exit"] not in (0, None) and any("part-" in line and "row" in line for line in read["stderr"]), str(read))
    e = r["row_5"]
    claim("5", "the old server reads the protocol 5 parts without a word (nothing on its stderr about them)",
          not any("layers" in s or "schema" in s or "part-" in s for s in e["server_stderr"]), str(e["server_stderr"]))
    claim("5", "their steps have no layers to it: the spike in a protocol 5 part is not judged, the one in a protocol 4 part is",
          bool(e["report"]) and ("activation_spike", 35, 1) in [tuple(i) for i in e["report"]["incidents"]],
          str(e["report"]))
    out.extend(checks_step_2(r))
    return out


def checks_step_2(r: dict) -> list[dict]:
    out = []

    def claim(row: str, text: str, holds: bool, seen: str) -> None:
        out.append({"row": row, "claim": text, "holds": bool(holds), "seen": seen})

    def listed(entry: dict, run_id: str) -> dict | None:
        rows = entry["listed"] if isinstance(entry["listed"], list) else []
        return next((x for x in rows if isinstance(x, dict) and x.get("run_id") == run_id), None)

    e = r["row_6"]
    o, n = e["old_server"], e["new_server"]
    claim("6", "the old server refuses the new sender's stream on its metadata version, before any row: nothing stored",
          not o["parts_after"] and not o["scalar_objects_after"]
          and any("schema_version 2 is not supported" in line for line in o["stderr"]), " | ".join(o["stderr"]))
    claim("6", "nothing acknowledged, the lines kept, OlderServer said once",
          o["sender"]["offset_file"] == 0 and o["lines_in_file"] == o["lines_before"] == 43
          and o["sender"]["last_error_type"] == "OlderServer" and e["older_message_said"] == 1,
          f"offset {o['sender']['offset_file']}; {o['lines_in_file']} lines; {o['sender']['last_error_type']}; "
          f"said {e['older_message_said']} time(s)")
    claim("6", "with the new server in its place: confirmed; the training rows alone in part-* files (steps 0 to 39)",
          n["sender"].get("confirmed") is True and n["parts"] and all(p["layout"] == "protocol 5" for p in n["parts"])
          and min(p["steps"][0] for p in n["parts"]) == 0 and max(p["steps"][1] for p in n["parts"]) == 39,
          f"confirmed {n['sender'].get('confirmed')}; parts {n['parts']}")
    claim("6", "the rows with a seq only in scalars-* objects, each once (seq 0 to 2)",
          sum(x["rows"] for x in n["scalar_objects"]) == 3
          and sorted(q for x in n["scalar_objects"] for q in range(x["seq"][0], x["seq"][1] + 1)) == [0, 1, 2],
          str(n["scalar_objects"]))
    claim("6", "a scalar-only row never moves the rank's last step (the report's last step is 39, not 500)",
          bool(n["report"]) and n["report"]["last_step"] == 39, str(n["report"]))
    d, s_ = e["new_load_records"], e["new_load_records_scalars"]
    claim("6", "the new load_records gives the training records alone by default",
          d.get("scalar_only") == 0 and d.get("training_equal_to_file_s") is True, str({k: v for k, v in d.items() if k != "stderr"}))
    claim("6", "with scalars=True it also reads the scalars-* objects: every scalar-only record of the file",
          s_.get("scalar_only") == 3 and s_.get("training_equal_to_file_s") is True and s_.get("scalar_only_equal_to_file_s") is True,
          str({k: v for k, v in s_.items() if k != "stderr"}))
    e = r["row_7"]
    raised = e["old_load_records"].get("raised") or ""
    claim("7", "the old load_records refuses the store, naming the part and the row (protocol 5 parts)",
          "part-" in raised and "row" in raised, raised or "did not raise")
    read = e["old_one_read"]
    claim("7", "the old one read of the run refuses it, naming the part and the row",
          read["exit"] not in (0, None) and any("part-" in line and "row" in line for line in read["stderr"]), str(read))
    e = r["row_8"]
    old, new = listed(e["old_runs"], "xv-scal"), listed(e["new_runs_control"], "xv-scal")
    keys = ("run_state", "last_step", "incidents", "first_incident")
    claim("8", "the old runs lists the store's run from the protocol 5 server's report, as the new runs does (last step 39)",
          old is not None and new is not None and all(old.get(k) == new.get(k) for k in keys) and old.get("last_step") == 39,
          f"old {json.dumps({k: (old or {}).get(k) for k in keys})}; new {json.dumps({k: (new or {}).get(k) for k in keys})}")
    b = e["8b"]
    raised = b["old_load_records"].get("raised") or ""
    claim("8b", "the old load_records refuses a new .jsonl holding a scalar-only record, naming the line",
          "schema_version" in raised and "line" in raised, raised or "did not raise")
    old = listed(b["old_runs"], "xv-scal")
    claim("8b", "the old runs lists such a file as a run with no evidence",
          old is not None and not old.get("incidents") and old.get("run_state") is None, json.dumps(old)[:900])
    for case in ("a_training_only", "b_with_scalar_only"):
        e = r["row_9"][case]
        d, s_ = e["new_load_records"], e["new_load_records_scalars"]
        if case == "a_training_only":
            claim("9 a", "the old sender's store: confirmed, protocol 4 parts, no scalars-* object",
                  e["old_sender"].get("confirmed") is True and {p["layout"] for p in e["parts"]} == {"protocol 4"}
                  and not e["scalar_objects"], f"{e['old_sender']}; {e['parts']}; {e['scalar_objects']}")
            claim("9 a", "the new load_records reads it as the file, with and without scalars=True",
                  d.get("equal_to_file") is True and s_.get("equal_to_file") is True,
                  f"default {d.get('equal_to_file')}, scalars=True {s_.get('equal_to_file')}")
    e = r["row_9"]["b_with_scalar_only"]
    w = e["watchdog_rank0"] if isinstance(e["watchdog_rank0"], dict) else {}
    steps = sorted(x for p in e["parts"] for x in p.get("step_list", []))
    d, s_, f = e["new_load_records"], e["new_load_records_scalars"], e["local_file_scalars"]
    stderr = r["row_9"]["b_server_stderr"]
    claim("9 b", "the protocol 4 sender given a protocol 5 capture's file is acknowledged to the end of the file",
          e["old_sender"].get("confirmed") is True and e["old_sender"].get("offset") == e["file_bytes"],
          f"{json.dumps({k: v for k, v in e['old_sender'].items() if k != 'stderr'})}; file {e['file_bytes']} bytes")
    claim("9 b", "the parts hold exactly steps 0 to 29 and no row at step 500; no scalars-* object",
          steps == list(range(30)) and not e["scalar_objects"], f"parts {e['parts']}; objects {e['scalar_objects']}")
    claim("9 b", "last_step 29 in incidents.json and in the rank's entry of watchdog.json",
          bool(e["report"]) and e["report"]["last_step"] == 29 and w.get("last_step") == 29,
          f"incidents.json {e['report']}; watchdog.json last_step {w.get('last_step')}")
    claim("9 b", "the rank's entry has dropped_rows 1 and dropped_row_keys {unsupported_schema_version: 1}, and no rows_not_newer",
          w.get("dropped_rows") == 1 and w.get("dropped_row_keys") == {"unsupported_schema_version": 1}
          and not w.get("rows_not_newer"), json.dumps(w))
    claim("9 b", 'the server\'s stderr says "dropped rows that cannot be stored: 1 of a schema_version this server does not read"',
          says(stderr, "dropped rows that cannot be stored: 1 of a schema_version this server does not read") == 1,
          " | ".join(line for line in stderr if "dropped" in line) or "no such line")
    claim("9 b", "the protocol 5 load_records gives 30 training records by default, and the same 30 with scalars=True",
          d.get("records") == 30 and d.get("scalar_only") == 0 and d.get("training_equal_to_file_s") is True
          and s_.get("records") == 30 and s_.get("scalar_only") == 0 and s_.get("training_equal_to_file_s") is True,
          f"default {json.dumps({k: v for k, v in d.items() if k != 'stderr'})}; "
          f"scalars=True {json.dumps({k: v for k, v in s_.items() if k != 'stderr'})}")
    claim("9 b", "the local .jsonl still holds the scalar-only record, read by the protocol 5 load_records(path, scalars=True)",
          f.get("scalar_only") == 1 and f.get("scalar_only_equal_to_file_s") is True and f.get("equal_to_file") is True,
          json.dumps({k: v for k, v in f.items() if k != "stderr"}))
    e = r["row_10"]["a"]
    o, n = e["old_server"], e["new_server"]
    claim("10 a", "the old server refuses the resumed stream's first batch, the one with no steps (null min_step and "
          "max_step), on its metadata version: nothing acknowledged",
          any("received 1 batches, 0 rows" in line and "schema_version 2 is not supported" in line for line in o["stderr"])
          and o["sender"]["offset_file"] == e["before"]["offset"] and o["parts_after"] == o["parts_before"]
          and not o["scalar_objects_after"], f"{' | '.join(o['stderr'])}; offset {o['sender']['offset_file']}")
    claim("10 a", "the new server in its place takes the batch of scalar-only rows alone: confirmed, one scalars-* object "
          "of 2 rows, no new part, the last step still 19",
          n["sender"].get("confirmed") is True and sum(x["rows"] for x in n["scalar_objects"]) == 2
          and n["parts"] == o["parts_before"] and bool(n["report"]) and n["report"]["last_step"] == 19
          and isinstance(n["watchdog_rank0"], dict) and n["watchdog_rank0"].get("last_step") != 500,
          f"confirmed {n['sender'].get('confirmed')}; objects {n['scalar_objects']}; parts {n['parts']}; report {n['report']}; "
          f"watchdog.json rank 0 {n['watchdog_rank0']}")
    b = r["row_10"]["b"]
    claim("10 b", "the old server takes the old sender's batch with no steps (metadata 1, null min_step and max_step): "
          "the resumed stream is confirmed and every byte acknowledged",
          b["resumed"].get("confirmed") is True and b["resumed"].get("offset") == b["file_bytes"]
          and bool(b["report"]) and b["report"]["last_step"] == 39,
          f"{json.dumps({k: v for k, v in b['resumed'].items() if k != 'stderr'})}; file {b['file_bytes']} bytes; report {b['report']}")
    e = r["row_11"]
    claim("11", "the old server restarted on a store with scalars-* objects does not list them: the rank's later "
          "training steps (40 to 59) are stored and judged, not dropped as below step 500",
          e["old_sender"].get("confirmed") is True and bool(e["report"]) and e["report"]["last_step"] == 59
          and any(p["steps"] == [40, 59] for p in e["parts_after"]),
          f"sender {json.dumps({k: v for k, v in e['old_sender'].items() if k != 'stderr'})}; report {e['report']}; "
          f"parts {e['parts_after']}")
    claim("11", "it says nothing about them, and leaves them as they were",
          not any("scalars" in line or "seq" in line for line in e["server_stderr"])
          and e["scalar_objects_after"] == e["scalar_objects_before"],
          f"{e['server_stderr']}; objects {e['scalar_objects_after']}")
    return out


def measured(r: dict) -> dict:
    """Rows 9 c and 9 d: what happened, not judged (the README's clause "to get them to the server, send
    the file with an SDK of the capture's version")."""
    out = {}
    for name, key in (("9 c", "c_new_send_file_as_left"), ("9 d", "d_new_send_file_from_byte_0")):
        e = r["row_9"].get(key)
        if not e:
            out[name] = "not reached"
            continue
        w = e["watchdog_rank0"] if isinstance(e["watchdog_rank0"], dict) else {}
        out[name] = {
            "send_file_returned": e["send_file"].get("confirmed"),
            "offset_before": (e.get("binding_before") or {}).get("binding", {}).get("offset") if key.startswith("c") else
            ("offset file removed" if e.get("offset_file_removed") else "no offset file"),
            "offset_after": e["send_file"].get("offset"),
            "sender_stderr": e["send_file"].get("stderr"),
            "scalar_objects": e["scalar_objects"],
            "step_500_value_on_the_server": any(500 in x["steps"] for x in e["scalar_objects"]),
            "parts": [p["part"] for p in e["parts"]],
            "rows_not_newer": w.get("rows_not_newer"),
            "scalars_not_newer": w.get("scalars_not_newer"),
            "dropped_rows": w.get("dropped_rows"),
            "last_step_incidents": (e["report"] or {}).get("last_step"),
            "last_step_watchdog": w.get("last_step"),
            "load_records_scalars_true": {k: v for k, v in e["new_load_records_scalars"].items() if k != "stderr"},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    parser.add_argument("--sdk")
    parser.add_argument("--args")
    parser.add_argument("--old-checkout", type=Path)
    parser.add_argument("--old-binary", type=Path)
    parser.add_argument("--new-binary", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--new-binary-commit", help="the commit the new side's server was built at, recorded as given")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.sdk, json.loads(args.args))
        return
    for binary, want in ((args.old_binary, "4"), (args.new_binary, "5")):
        got = subprocess.run([str(binary), "protocol"], capture_output=True, text=True).stdout.strip()
        if got != want:
            raise SystemExit(f"{binary} speaks protocol {got!r}, not {want}")
    commit = subprocess.run(["git", "-C", str(args.old_checkout), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    new_commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="tq-xv-") as tmp:
        ctx = Context(args, Path(tmp))
        result: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "old_commit": commit, "new_commit": new_commit,
                        "old_protocol": 4, "new_protocol": 5,
                        "old_binary": "the old checkout's own release build (server/target/release/tensorquill-server)",
                        "new_binary": {"name": args.new_binary.name, "built_at": args.new_binary_commit,
                                       "sha256": hashlib.sha256(args.new_binary.read_bytes()).hexdigest()}}
        for name, row in (("row_1", row_1), ("row_2", row_2), ("row_3", row_3), ("row_4", row_4), ("row_5", row_5),
                          ("row_6", row_6), ("row_7", row_7), ("row_8", row_8), ("row_9", row_9), ("row_10", row_10),
                          ("row_11", row_11)):
            started = time.monotonic()
            result[name] = row(ctx)
            print(f"{name} done in {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
        result["measured"] = measured(result)
        try:
            result["checks"] = checks(result)
        except Exception as error:  # noqa: BLE001  a check that cannot be read is said, and the rows are kept
            result["checks"] = [{"row": "-", "claim": "the checks ran", "holds": False, "seen": f"{type(error).__name__}: {error}"}]
        result = ctx.clean(result)
    text = json.dumps(result, indent=1, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    for name, seen in result["measured"].items():
        print(f"measured row {name}: {json.dumps(seen)[:900]}")
    for check in result["checks"]:
        print(f"{'holds   ' if check['holds'] else 'FINDING '} row {check['row']}: {check['claim']} -- {check['seen'][:300]}")


if __name__ == "__main__":
    main()
