#!/usr/bin/env bash
# Clean-venv smoke test for a built tensorquill wheel, with an outbound-network audit hook.
#
#   packaging-smoke.sh <path/to/tensorquill-*.whl> <path/to/examples/toy_moe.py>
#
# Optional environment: PYTHON (interpreter that creates the venv, default python3),
# TORCH_INDEX_URL (for example https://download.pytorch.org/whl/cpu, to install a CPU torch first),
# SERVER_WHEEL (a built tensorquill-server wheel to install with it; without one, see below).
#
# What it proves: the wheel alone (plus torch) imports, `python -m tensorquill --help` runs, and
# toy_moe.py runs to completion from a directory with no source tree beside it, the evidence file it
# prints is the one `tensorquill runs` and `tensorquill-mcp` pair with its capture, with no Python-level
# socket connect or name lookup to anything but loopback. The hook is shown to fire first, so a
# clean result is not vacuous. It does not see native-code networking or child processes.
set -euo pipefail

WHEEL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
EXAMPLE="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d)"
echo "scratch directory: $WORK"

"$PYTHON" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  "$VPY" -m pip install -q --disable-pip-version-check torch --index-url "$TORCH_INDEX_URL"
fi
if [ -n "${SERVER_WHEEL:-}" ]; then
  # A built tensorquill-server wheel: pip resolves the wheel's whole requirement list, as a user's install does.
  "$VPY" -m pip install -q --disable-pip-version-check "$SERVER_WHEEL" "$WHEEL"
else
  # tensorquill-server is not on PyPI before the first release, so pip cannot resolve the wheel's own
  # requirement list. Until it is: every other requirement the wheel declares (torch, if it is not there
  # yet), then the wheel alone. requirements.py fails if the wheel does not require tensorquill-server.
  # NOT covered this way: that `pip install tensorquill` resolves, and the binary beside the interpreter.
  "$PYTHON" "$(dirname "$0")/requirements.py" --wheel "$WHEEL" > "$WORK/requirements.txt"
  "$VPY" -m pip install -q --disable-pip-version-check -r "$WORK/requirements.txt"
  "$VPY" -m pip install -q --disable-pip-version-check --no-deps "$WHEEL"
fi

# The hook goes in after pip has finished, so pip itself is not blocked.
mkdir "$WORK/hook" "$WORK/proj"
cat > "$WORK/hook/sitecustomize.py" <<'PYHOOK'
"""Outbound-network audit hook for the packaging smoke test (not part of the repo).

Loaded by Python at startup through PYTHONPATH. Raises on any socket connect / send / name
lookup whose destination is not loopback or a Unix-domain path, and appends every decision to
the file named by NETAUDIT_LOG, so a violation is on record even if the code under test
swallows the exception. It sees Python-level sockets only, not native code or child processes
(spawned processes are logged, not followed).
"""
import atexit
import ipaddress
import os
import sys

_LOG = os.environ.get("NETAUDIT_LOG")
_counts = {"allowed": 0, "violations": 0, "spawns": 0}


def _log(line: str) -> None:
    if _LOG:
        with open(_LOG, "a") as fh:
            fh.write(line + "\n")


def _host_ok(host) -> bool:
    if host in (None, "", b"", "localhost", b"localhost"):
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # a real hostname that would need DNS
    return ip.is_loopback or ip.is_unspecified


def _addr_ok(address) -> bool:
    if isinstance(address, (str, bytes)):
        return True  # AF_UNIX path
    if isinstance(address, tuple) and address:
        return _host_ok(address[0])
    return False


def _hook(event: str, args) -> None:
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        address = args[1]
        ok = _addr_ok(address)
        what = f"{event} {address!r}"
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex"):
        host = args[0]
        ok = _host_ok(host)
        what = f"{event} {host!r}"
    elif event in ("socket.gethostbyaddr", "socket.getnameinfo"):
        ok = _addr_ok(args[0]) if isinstance(args[0], tuple) else _host_ok(args[0])
        what = f"{event} {args[0]!r}"
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"):
        _counts["spawns"] += 1
        _log(f"SPAWN {event} {str(args)[:200]}")
        return
    else:
        return
    if ok:
        _counts["allowed"] += 1
        _log(f"ALLOW {what}")
        return
    _counts["violations"] += 1
    _log(f"VIOLATION {what}")
    raise RuntimeError(f"NETAUDIT blocked outbound network access: {what}")


def _at_exit() -> None:
    mod = sys.modules.get("tensorquill")
    _log(f"EXIT pid={os.getpid()} tensorquill_file={getattr(mod, '__file__', None)} "
         f"allowed={_counts['allowed']} violations={_counts['violations']} spawns={_counts['spawns']}")


if _LOG:
    sys.addaudithook(_hook)
    atexit.register(_at_exit)
    _log(f"INSTALLED pid={os.getpid()} argv={sys.argv!r} exe={sys.executable}")
PYHOOK
export NETAUDIT_LOG="$WORK/netaudit.log"
export PYTHONPATH="$WORK/hook"

echo "== 1. the hook fires on a deliberate non-loopback connect (TEST-NET-1, RFC 5737) =="
"$VPY" - <<'PYLIVE'
import socket
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("192.0.2.1", 80))
except RuntimeError as exc:
    assert "NETAUDIT" in str(exc), exc
    print("hook fired:", exc)
else:
    raise SystemExit("hook did NOT fire: the audit proof would be vacuous")
PYLIVE
: > "$NETAUDIT_LOG"   # forget the deliberate violation

echo "== 2. the installed wheel, not a source tree, is what runs =="
cp "$EXAMPLE" "$WORK/proj/toy_moe.py"   # alone: its ../src lookup finds nothing
cd "$WORK/proj"
"$VPY" -c "import tensorquill; p = tensorquill.__file__; print(p); assert 'site-packages' in p, p"
"$VPY" -m tensorquill --help > /dev/null
"$WORK/venv/bin/tensorquill" --help > /dev/null
"$WORK/venv/bin/tensorquill-mcp" --help > /dev/null
echo "== 3. toy_moe.py to completion =="
"$VPY" toy_moe.py | tee "$WORK/toy_moe.out"
test -s tensorquill-out/toy-moe.rank0.jsonl
EVIDENCE="tensorquill-out/toy-moe.rank0.evidence.json"   # the name the readers pair with the capture
grep -qxF "evidence -> $EVIDENCE" "$WORK/toy_moe.out"
test -s "$EVIDENCE"

echo "== 3b. the evidence it printed, read back through the installed CLI and MCP server =="
# `runs` must find the evidence beside the capture: a missing one is listed with its state unknown.
"$WORK/venv/bin/tensorquill" runs tensorquill-out --json > "$WORK/runs.json"
# The MCP server on the printed path must read the capture beside it for a router's series.
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_series","arguments":{"name":"router"}}}' \
  | "$WORK/venv/bin/tensorquill-mcp" "$EVIDENCE" > "$WORK/mcp.out"
"$VPY" - "$WORK/runs.json" "$WORK/mcp.out" "$EVIDENCE" <<'PYREAD'
import json, sys
from pathlib import Path
runs, mcp, evidence = sys.argv[1:]
(row,) = json.loads(Path(runs).read_text())
assert row["run_id"] == "toy-moe", row
assert row["evidence"]["state"] != "missing", row["evidence"]
assert row["incidents"], row
replies = {r.get("id"): r for r in map(json.loads, Path(mcp).read_text().splitlines())}
result = replies[2]["result"]
assert result["isError"] is False, result
series = json.loads(result["content"][0]["text"])
capture = str(Path(evidence).resolve().with_name("toy-moe.rank0.jsonl"))
assert series["records"] is not None and Path(series["records"]).resolve() == Path(capture), series["records"]
print("runs:", row["run_id"], row["evidence"]["state"], "| MCP get_series read", series["records"])
PYREAD

echo "== 4. audit log =="
cat "$NETAUDIT_LOG"
grep -q '^INSTALLED' "$NETAUDIT_LOG"      # the hook was live in the runs above
if grep -q '^VIOLATION' "$NETAUDIT_LOG"; then echo "outbound network attempt recorded"; exit 1; fi
# Untested on Linux: a harmless child process there is a warning, not a failure.
if grep -q '^SPAWN' "$NETAUDIT_LOG"; then echo "WARNING: a child process was spawned; the hook cannot see inside it"; fi
echo "OK: wheel smoke test passed with zero outbound connections"
