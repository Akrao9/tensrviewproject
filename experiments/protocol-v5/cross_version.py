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

The records: one rank, layers in a model's order, an `activation_spike` at step 35 and one at 85. The
detectors give one incident whose `affected_steps` counts the spikes they read, so it says which
parts' layers a server read: the old sender writes steps 0 to 59, the new one 60 to 119.
"""

from __future__ import annotations

import argparse
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
            got = evidence.load_records(args["source"])
            want = [json.loads(line) for line in Path(args["file"]).read_text().splitlines()] if args.get("file") else None
            out["records"] = len(got)
            out["steps_in_order"] = [r["step"] for r in got] == sorted(r["step"] for r in got)
            out["equal_to_file"] = None if want is None else got == want
        except Exception as error:  # noqa: BLE001
            out["raised"] = f"{type(error).__name__}: {error}"
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
                     (str(ROOT), "<new>"), (str(Path.home()), "~")]

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
                    "steps": [min(steps), max(steps)], "row_versions": sorted(set(rows))})
    return out


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
    claim("4", "the old runs lists it as a run with no evidence",
          run is not None and not run.get("incidents") and run.get("run_state") is None, json.dumps(listed)[:900])
    read = e["old_one_read"]
    claim("4", "the old one read of the store's run refuses it, naming the part and the row",
          read["exit"] not in (0, None) and any("part-" in line and "row" in line for line in read["stderr"]), str(read))
    e = r["row_5"]
    claim("5", "the old server reads the protocol 5 parts without a word (nothing on its stderr about them)",
          not any("layers" in s or "schema" in s or "part-" in s for s in e["server_stderr"]), str(e["server_stderr"]))
    claim("5", "their steps have no layers to it: the spike in a protocol 5 part is not judged, the one in a protocol 4 part is",
          bool(e["report"]) and ("activation_spike", 35, 1) in [tuple(i) for i in e["report"]["incidents"]],
          str(e["report"]))
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
                        "old_protocol": 4, "new_protocol": 5}
        for name, row in (("row_1", row_1), ("row_2", row_2), ("row_3", row_3), ("row_4", row_4), ("row_5", row_5)):
            started = time.monotonic()
            result[name] = row(ctx)
            print(f"{name} done in {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
        try:
            result["checks"] = checks(result)
        except Exception as error:  # noqa: BLE001  a check that cannot be read is said, and the rows are kept
            result["checks"] = [{"row": "-", "claim": "the checks ran", "holds": False, "seen": f"{type(error).__name__}: {error}"}]
        result = ctx.clean(result)
    text = json.dumps(result, indent=1, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    for check in result["checks"]:
        print(f"{'holds   ' if check['holds'] else 'FINDING '} row {check['row']}: {check['claim']} -- {check['seen'][:300]}")


if __name__ == "__main__":
    main()
