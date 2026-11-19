#!/bin/bash
# One measurement of `tensorquill-server` under simulated ranks: starts a fresh release
# server on an empty store, samples its memory, CPU and open files once a second, drives it
# with the `live_streams` example, stops it and prints a summary.
#
#     cargo build --release && cargo build --release --examples
#     examples/simulated_ranks.sh <label> <source store> <ranks> [live_streams options...]
#
#     examples/simulated_ranks.sh r64   $SOURCE 64   --step-ms 65
#     SERVER_ARGS="--quota-mib 4096" examples/simulated_ranks.sh r1024 $SOURCE 1024 --step-ms 65
#
# <source store> holds the run `bench` in 50-step parts (sdk/examples/benchmark_detectors.py
# writes it). Environment: OUT (where results and the server's store go; default a new
# temporary directory), SERVER_ARGS (more server options), PORT (default 50071), RESTARTS
# (after the run, stop the server, start another on the same store and send 3 batches a rank
# from the step after the stored ones, this many times), REPORTS=1 (also time until each
# rank's incidents.json covers a batch; the client then reads those files, which costs it CPU),
# KEEP=1 (keep the server's store). Run from the `server` directory. Nothing here changes a
# system setting: the open-file limit is whatever the calling shell has.
#
# Results go to $OUT/<label>, which must be new or empty: one that holds anything, such as an
# earlier run with the same label, is refused, never deleted. If a step fails (the server does
# not start, the load client or the summary exits nonzero), the server and the sampler started
# here are stopped, the logs and the store are kept, and the script exits with that step's status.
set -u
label=${1:?label}; src=${2:?source store}; ranks=${3:?ranks}; shift 3
bin=${BIN_DIR:-target/release}
out=${OUT:-$(mktemp -d)}/$label; store=$out/store
port=${PORT:-50071}; restarts=${RESTARTS:-0}
for built in tensorquill-server examples/live_streams; do
  [ -x "$bin/$built" ] || { echo "$bin/$built is not built: cargo build --release && cargo build --release --examples" >&2; exit 1; }
done
if [ -e "$out" ] && { [ ! -d "$out" ] || [ -n "$(ls -A "$out")" ]; }; then
  echo "$out is not empty (an earlier run with the label $label?); nothing was deleted. Choose another label or OUT." >&2
  exit 2
fi
mkdir -p "$store" || exit

server=; timer=; sampler=
cleanup() { # on every exit: stop what this script started and is still running
  local status=$? pid i
  trap - EXIT INT TERM
  for pid in $sampler $server $timer; do kill -TERM "$pid" 2>/dev/null; done
  for ((i = 0; i < 50; i++)); do
    alive=; for pid in $sampler $server $timer; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ -z "$alive" ] && break; sleep 0.1
  done
  for pid in $sampler $server $timer; do kill -KILL "$pid" 2>/dev/null; done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
fail() { # $1 = exit status, $2 = what failed
  echo "simulated_ranks.sh: $2 (status $1); logs kept in $out" >&2
  exit "$1"
}

if nc -z 127.0.0.1 "$port" 2>/dev/null; then
  fail 1 "something already listens on port $port; set PORT to a free one"
fi

start_server() { # $1 = tag
  /usr/bin/time -l "$bin/tensorquill-server" --listen 127.0.0.1:"$port" --store "$store" --flush-steps 50 \
    ${SERVER_ARGS:-} > "$out/server-$1.log" 2> "$out/server-$1.time" &
  timer=$!
  until nc -z 127.0.0.1 "$port" 2>/dev/null; do
    kill -0 $timer 2>/dev/null || fail 1 "the server did not start: $(cat "$out/server-$1.time")"
    sleep 0.1
  done
  server=$(pgrep -P $timer)
  [ -n "$server" ] || fail 1 "the server's process was not found under $timer"
  ( n=0
    while kill -0 "$server" 2>/dev/null; do
      echo "$(date +%s) $(ps -o rss=,time= -p "$server")" >> "$out/samples-$1.txt"
      if (( n % 10 == 5 )); then echo "$(date +%s) $(lsof -n -P -p "$server" 2>/dev/null | tail -n +2 | wc -l)" >> "$out/files-$1.txt"; fi
      n=$((n + 1)); sleep 1
    done ) &
  sampler=$!
}
stop_server() {
  local i
  kill -TERM "$server"; wait $timer 2>/dev/null; wait $sampler 2>/dev/null
  server=; timer=; sampler=
  for ((i = 0; i < 300; i++)); do nc -z 127.0.0.1 "$port" 2>/dev/null || return 0; sleep 0.1; done
  fail 1 "port $port is still open 30 s after the server was stopped"
}
drive() { # $1 = tag, then live_streams options; returns the client's exit status
  local tag=$1 status; shift
  { echo "before: $(uptime | sed 's/.*load/load/'); $(sysctl -n vm.swapusage)"
    echo "start $(date +%s)"
    /usr/bin/time -l "$bin/examples/live_streams" "$src" bench http://127.0.0.1:"$port" "$@" 2> "$out/client-$tag.time"
    status=$?
    echo "end $(date +%s)"
    echo "after: $(uptime | sed 's/.*load/load/'); $(sysctl -n vm.swapusage)"; } > "$out/client-$tag.txt" 2>&1
  return $status
}

reports=(); [ "${REPORTS:-0}" = 1 ] && reports=(--store "$store")
echo "open-file limit of this shell: $(ulimit -n); server options: --flush-steps 50 ${SERVER_ARGS:-}" > "$out/setup.txt"
start_server first
drive first --ranks "$ranks" ${reports[@]+"${reports[@]}"} "$@" || fail $? "the load client (first) failed"
stop_server
steps=$(( $(ls "$src"/runs/bench/rank=0/attempt=*/ | wc -l) * 50 ))
for ((i = 1; i <= restarts; i++)); do
  start_server restart$i
  drive restart$i --ranks "$ranks" --first-step $steps --batches 3 || fail $? "the load client (restart$i) failed"
  stop_server
  steps=$((steps + 150))
done

python3 - "$out" "$ranks" <<'PY' || fail $? "the summary failed"
import collections, glob, json, os, statistics, sys
out, ranks = sys.argv[1], int(sys.argv[2])
store = f"{out}/store"
print(open(f"{out}/setup.txt").read().strip())
def time_file(path):
    lines = open(path).read().splitlines()
    real = next(l for l in lines if " real " in l and l.strip().endswith("sys")).split()
    rss = int(next(l for l in lines if "maximum resident set size" in l).split()[0])
    other = [l for l in lines if l.strip() and not l.startswith(" ")]
    return float(real[0]), float(real[2]) + float(real[4]), rss, other
for client in sorted(glob.glob(f"{out}/client-*.txt")):
    tag = os.path.basename(client)[7:-4]
    text = open(client).read().splitlines()
    print(f"== {tag}")
    for line in text:
        if not line.startswith("{") and not line.startswith(("start ", "end ")):
            print("  " + line)
    start = next((int(l.split()[1]) for l in text if l.startswith("start ")), None)
    end = next((int(l.split()[1]) for l in text if l.startswith("end ")), None)
    result = next((json.loads(l) for l in text if l.startswith("{")), None)
    real, cpu, rss, _ = time_file(f"{out}/client-{tag}.time")
    print(f"  client: {real:.1f} s, cpu {cpu:.1f} s ({100 * cpu / real:.0f}% of a core), peak RSS {rss / 2**20:.0f} MiB")
    real, cpu, rss, log = time_file(f"{out}/server-{tag}.time")
    print(f"  server: alive {real:.1f} s, cpu {cpu:.1f} s, peak RSS {rss / 2**20:.0f} MiB ({rss / 2**20 / ranks:.2f} MiB a rank)")
    samples = [l.split() for l in open(f"{out}/samples-{tag}.txt") if len(l.split()) == 3]
    def cpu_seconds(text):
        parts = text.replace(".", ":").split(":")  # [h:]m:s.cc
        whole = [int(p) for p in parts[:-1]]
        return sum(v * 60 ** i for i, v in enumerate(reversed(whole))) + int(parts[-1]) / 100
    during = [(int(t), int(rss), cpu_seconds(c)) for t, rss, c in samples if start <= int(t) <= end]
    if result and len(during) > 4:
        # The stream itself: from the first batch being due to the last acknowledgement.
        seconds = result["seconds"]
        running = [s for s in during if s[0] >= end - seconds]
        half = [s[1] for s in running[len(running) // 2:]]
        used = running[-1][2] - running[0][2]
        span = running[-1][0] - running[0][0]
        print(f"  server while streaming ({span} s sampled): cpu {used:.1f} s = {100 * used / span:.0f}% of a core; "
              f"RSS in the second half median {statistics.median(half) / 1024:.0f} MiB "
              f"({statistics.median(half) / 1024 / ranks:.2f} MiB a rank), least {min(half) / 1024:.0f}, most {max(half) / 1024:.0f}; "
              f"before the first batch {during[0][1] / 1024:.0f} MiB")
    files = [int(l.split()[1]) for l in open(f"{out}/files-{tag}.txt")] if os.path.exists(f"{out}/files-{tag}.txt") else []
    if files:
        print(f"  server open files (lsof, every 10 s): most {max(files)}, least {min(files)}")
    for line in log:
        print("  server stderr: " + line[:300])
    stdout = [l for l in open(f"{out}/server-{tag}.log").read().splitlines() if "listening" not in l]
    for line in stdout[:10]:
        print("  server stdout: " + line[:300])
last = collections.Counter(); kinds = collections.Counter(); reported = 0
for path in glob.glob(f"{store}/runs/bench/rank=*/incidents.json"):
    document = json.load(open(path)); reported += 1; last[document.get("last_step")] += 1
    for incident in document.get("incidents", []):
        kinds[incident.get("kind")] += 1
parts = glob.glob(f"{store}/runs/bench/rank=*/attempt=*/*.parquet")
size = sum(os.path.getsize(p) for p in parts)
whole = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(store) for f in fs)
print(f"== store: {len(parts)} parts in {len(glob.glob(f'{store}/runs/bench/rank=*'))} rank folders, {size / 2**20:.0f} MiB of parts, "
      f"{whole / 2**20:.0f} MiB in all; incidents.json for {reported} ranks, last_step {dict(last)}, incidents {dict(kinds)}")
PY
[ "${KEEP:-0}" = 1 ] || rm -rf "$store"
echo "results in $out"
