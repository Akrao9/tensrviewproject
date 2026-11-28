"""The cases of `WAVE_AN_CONTRACT.md` (the read API and the catalog of step 2 of the local-first 0.1.0,
with the parts moved to 0.2.0 kept as written: detector checkpoints and compaction):
`fixtures/wave_an/cases.json`.

What runs here, on the base (`main` at `1dddcce`) and on every lane's tree:

- The document: every case is named on the page, rests on rows of its decision table, names a rule
  and its paths, and is listed under its release (`0.1.0` or `0.2.0`); every row of the table is in one
  of the page's states (decided, answered by the owner, moved to 0.2.0, open).
- The rules: each case's `expect` is what the page's rules give for its `input`, with the rules
  written plainly below (`RULES`), whatever the case's release. These are ordinary tests: they keep the
  cases and the page one thing. They are not product code, and nothing in the SDK or the server calls
  them.
- Targets: the product paths that can already be named, each a strict expected failure
  (`xfail(strict=True, raises=AssertionError)`) until the lane in its reason lands it and removes the
  mark; a 0.2.0 target says so in its reason and stays marked through 0.1.0. Each fails on an
  assertion about what the product gives (a server that does not start its read API or does not
  print what the page states, a read that returns other rows), never on an import or a connection
  error. On the base every server target fails because the server refuses `--http-listen` (or
  `--snapshot-every-parts`) with exit 2.
- Pins: `load_records` on a compacted store whose inputs are all kept reads each step once (D7); and
  today's `evidence.scalar_series` gives the resolved points of the two scalar series cases (D13), the
  reader the read API must agree with.

The server paths need `TENSORQUILL_SERVER_BIN` (or the binary `conftest.py` names) and are skipped
without it; anything that writes Parquet needs pyarrow and is skipped without it, test by test. CI's
plain jobs have neither, so there only the document and the rules run; CI's server job has both, so
nothing here is skipped there for want of them.

The page is kept outside this repository. The tests that read it take it through `conftest.contract_page`,
inside the test and after the checks that need no page, so in a checkout with no page (CI's among them) those
tests are skipped and every other test here runs.
"""

from __future__ import annotations

import fnmatch
import http.client
import ipaddress
import json
import os
import re
import selectors
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

import conftest

ROOT = Path(__file__).parents[2]
DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "wave_an" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
BY_NAME = {case["name"]: case for case in CASES}

LANE_API = "lane AN-2 (http.rs, read.rs), in 0.1.0: the read API of contract sections 2 and 3"
LANE_LISTEN = "lane AN-2 (http.rs) with main.rs's owner at the seam, in 0.1.0: the listener of contract section 2.1"
LANE_ALLOW = "lane AN-2 (http.rs), in 0.1.0: the Host and Origin allowlist and framing of contract section 6.2.1"
LANE_SCALARS = "lane AN-2 (read.rs), in 0.1.0: the scalar series of contract section 5"
LANE_RETIRED = "0.2.0, lanes AN-4 and AN-2 (manifest.rs, http.rs): the retired listing of contract section 9"
LANE_SNAPSHOT = "0.2.0, lanes AN-3 and AN-5 (detect.rs, snapshot.rs, ingest.rs): the restart of contract section 7.4"
LANE_FILES = "0.2.0, lanes AN-4 and AN-2 (storage.rs, manifest.rs, http.rs): the manifest-first listing of contract sections 2.2 and 8.4"
LANE_READER = "0.2.0, lane AN-6 (evidence.py): load_records manifest-first, contract section 8.8"
RUN = "an-run"
ATTEMPT = 1732400000000
NOW_MS = 1_800_000_000_000  # the rules' clock for ages given in hours; the server's own clock in a target
HOUR_MS = 3_600_000


# --- the rules of the page, written plainly ------------------------------------------------------


def catalog_start(rows: list[dict]) -> list[dict]:
    """Section 1.5's table, in its order."""
    out = []
    for row in rows:
        if not row["file"]:
            out.append({"action": "build", "why": "missing"})
        elif not row["user_version_ok"]:
            out.append({"action": "build", "why": "version"})
        elif not row["quick_check_ok"]:
            out.append({"action": "build", "why": "damaged"})
        elif not row["store_id_ok"]:
            out.append({"action": "build", "why": "store_replaced"})
        elif not row["clean_stop"]:
            out.append({"action": "resync", "why": "unclean_stop"})
        else:
            out.append({"action": "open", "why": None})
    return out


def catalog_events(given: dict) -> dict:
    """Section 1.4 and 1.5: an event is sent only after its object is stored; a crash loses events not
    committed; the resync takes them from the store."""
    stored = [key for key in given["puts"]]
    committed = [key for key in given["committed"] if key in stored]
    sent_before_put = [key for key in given["committed"] if key not in stored]
    return {
        "after_crash": committed,
        "ahead_of_store": sent_before_put,
        "behind_store": [key for key in stored if key not in committed],
        "after_resync": stored,
    }


def _finite(value) -> bool:
    return not isinstance(value, str)


def series(given: dict) -> dict:
    """Section 3.1: buckets by position that keep their lowest and highest finite value and their first
    non-finite one, with the series' first and last points."""
    points = _points(given["series"])
    limit = given["points"]
    n = len(points)
    if n <= limit:
        return {"points": points, "downsampled": False, "of": n, "buckets": None}
    buckets = (limit - 2) // 3
    kept = {0, n - 1}
    for i in range(buckets):
        lo, hi = i * n // buckets, (i + 1) * n // buckets
        low = high = odd = None
        for at in range(lo, hi):
            value = points[at][1]
            if not _finite(value):
                if odd is None:
                    odd = at
                continue
            if low is None or value < points[low][1]:
                low = at
            if high is None or value > points[high][1]:
                high = at
        kept.update(at for at in (low, high, odd) if at is not None)
    return {"points": [points[at] for at in sorted(kept)], "downsampled": True, "of": n, "buckets": buckets}


def _points(spec: dict) -> list[list]:
    """A series given as a constant with some steps set (`{"steps", "value", "set"}`), or as points."""
    if "points" in spec:
        return [list(p) for p in spec["points"]]
    values = {int(step): value for step, value in spec.get("set", {}).items()}
    return [[step, values.get(step, spec["value"])] for step in range(spec["steps"])]


def _rows(spec: list[dict]) -> list[dict]:
    """Rows given per attempt as `{"attempt", "steps": [first, last], "columns": {name: rule}}`; a
    column's rule is `"step/10"` or a constant."""
    rows = []
    for block in spec:
        first, last = block["steps"]
        for step in range(first, last + 1):
            row = {"run_id": RUN, "rank": 0, "attempt": block["attempt"], "step": step}
            for name, rule in block.get("columns", {}).items():
                row[name] = step / 10 if rule == "step/10" else rule
            rows.append(row)
    return rows


def _first_steps(rows: list[dict], order: list[int]) -> list[tuple[int, int]]:
    firsts: dict[int, int] = {}
    for row in rows:
        firsts[row["attempt"]] = min(firsts.get(row["attempt"], row["step"]), row["step"])
    return [(attempt, firsts[attempt]) for attempt in order if attempt in firsts]


def _cutoffs(order: list[tuple[int, int]]) -> dict[int, int | None]:
    """`resolve` in `detect.rs`: each attempt's rows at or past the lowest first step of a later
    attempt were rolled back."""
    cut: int | None = None
    out: dict[int, int | None] = {}
    for attempt, first in reversed(order):
        out[attempt] = cut
        cut = first if cut is None else min(cut, first)
    return out


def _resolved(rows: list[dict], order: list[int]) -> list[dict]:
    cutoffs = _cutoffs(_first_steps(rows, order))
    kept = [row for row in rows if cutoffs[row["attempt"]] is None or row["step"] < cutoffs[row["attempt"]]]
    return sorted(kept, key=lambda row: row["step"])


IDENTITY = ("run_id", "rank", "attempt", "step")


def scan(given: dict) -> dict:
    """Section 3.2: every stored row in [from_step, to_step), named columns only, in pages."""
    rows = _rows(given["rows"])
    order = given.get("order") or sorted({row["attempt"] for row in rows})
    known = {name for row in rows for name in row if name not in IDENTITY}
    unknown = [name for name in given["columns"] if name not in known]
    if unknown:
        return {"error": {"status": 400, "code": "unknown_column", "column": unknown[0]}}
    if given.get("attempts", "resolved") == "resolved":
        picked = _resolved(rows, order)
    else:
        picked = sorted(rows, key=lambda row: (order.index(row["attempt"]), row["step"]))
    low, high = given.get("from_step"), given.get("to_step")
    picked = [row for row in picked if (low is None or row["step"] >= low) and (high is None or row["step"] < high)]
    shaped = [{**{key: row[key] for key in IDENTITY}, **{name: row.get(name) for name in given["columns"]}}
              for row in picked]
    size = given["page_size"]
    pages = [shaped[at:at + size] for at in range(0, len(shaped), size)] or [[]]
    return {"pages": [{"rows": page, "next": index < len(pages) - 1} for index, page in enumerate(pages)]}


def _view(rows: list[dict], order: list[int], boundary: int) -> list[list[int]]:
    return [[attempt, first] for attempt, first in _first_steps(rows, order) if first <= boundary]


def scan_view(given: dict) -> dict:
    """Section 3.2's view: a token holds its boundary and the attempts that start at or below it."""
    before = _rows(given["before"])
    first_page = _resolved(before, given["order_before"])[: given["page_size"]]
    boundary = first_page[-1]["step"]
    view = _view(before, given["order_before"], boundary)
    out = {"first_page_steps": [row["step"] for row in first_page], "boundary": boundary, "after": []}
    for change in given["changes"]:
        after = before + _rows(change["added"])
        order = change["order"]
        if _view(after, order, boundary) != view:
            out["after"].append({"status": 409, "code": "view_changed"})
            continue
        rest = [row for row in _resolved(after, order) if row["step"] > boundary][: given["page_size"]]
        out["after"].append({"status": 200, "steps": [[row["attempt"], row["step"]] for row in rest]})
    return out


def stored_cutoff(given: dict) -> dict:
    """Section 4: stored rows only; the cutoff and what is held are named."""
    last = max(high for _, high in given["parts"])
    return {"returned_last_step": last, "stored_through": {"attempt": given["attempt"], "step": last},
            "held": given["held"]}


def scalar_series(given: dict) -> dict:
    """Section 5 (`WAVE_AL_CONTRACT.md` section 4.6): the training records' values and the scalar-only
    records' points by step, the highest `seq` winning at one step, with the training records'
    cutoffs."""
    name = given["name"]
    order = given["order"]
    training = given["training"]
    firsts: dict[int, int] = {}
    for record in training:
        firsts[record["attempt"]] = min(firsts.get(record["attempt"], record["step"]), record["step"])
    cutoffs = _cutoffs([(a, firsts[a]) for a in order if a in firsts])
    by_attempt: dict[int, dict[int, object]] = {a: {} for a in order}
    for record in training:
        if name in record.get("scalars", {}):
            by_attempt[record["attempt"]][record["step"]] = record["scalars"][name]
    best: dict[tuple[int, int], int] = {}
    for record in given["scalar_only"]:
        if name not in record["scalars"]:
            continue
        key = (record["attempt"], record["step"])
        if key not in best or record["seq"] > best[key]:
            best[key] = record["seq"]
            by_attempt[record["attempt"]][record["step"]] = record["scalars"][name]
    if given.get("attempts", "resolved") == "all":
        return {"points": [[a, step, by_attempt[a][step]] for a in order for step in sorted(by_attempt[a])]}
    points = []
    for a in order:
        cut = cutoffs.get(a)
        points += [[step, value] for step, value in by_attempt[a].items() if cut is None or step < cut]
    return {"points": sorted(points)}


def compact_key(attempt: int, first: int, last: int, gen: int) -> str:
    return f"attempt={attempt}/compact-{first:08}-{last:08}-{gen:010}.parquet"


def compact_scalars_key(attempt: int, first_seq: int, last_seq: int, gen: int) -> str:
    return f"attempt={attempt}/compact-scalars-{first_seq:08}-{last_seq:08}-{gen:010}.parquet"


def manifest_key(number: int) -> str:
    return f"manifests/{number:010}.json"


def snapshot_key(attempt: int, through_step: int) -> str:
    return f"checkpoints/state-{attempt}-{through_step:08}.json.zst"


def part_key(attempt: int, first: int, last: int) -> str:
    return f"attempt={attempt}/part-{first:08}-{last:08}.parquet"


def scalars_key(attempt: int, first_seq: int, last_seq: int) -> str:
    return f"attempt={attempt}/scalars-{first_seq:08}-{last_seq:08}.parquet"


def names(given: list[dict]) -> list[str]:
    """Section 8.2's names, relative to the rank folder."""
    make = {"compact": lambda n: compact_key(n["attempt"], n["first_step"], n["last_step"], n["gen"]),
            "compact-scalars": lambda n: compact_scalars_key(n["attempt"], n["first_seq"], n["last_seq"], n["gen"]),
            "manifest": lambda n: manifest_key(n["number"]),
            "snapshot": lambda n: snapshot_key(n["attempt"], n["through_step"])}
    return [make[item["kind"]](item) for item in given]


def _parts(spec: dict) -> list[dict]:
    """Parts given as runs: `[{"first", "last", "every", "layout"}]`, each `every` steps a part."""
    parts = []
    for block in spec["runs"]:
        for first in range(block["first"], block["last"] + 1, block["every"]):
            last = min(first + block["every"] - 1, block["last"])
            parts.append({"key": part_key(spec["attempt"], first, last), "first_step": first, "last_step": last,
                          "rows": last - first + 1, "bytes": spec.get("part_bytes", 1000), "layout": block["layout"]})
    return parts


def compaction(given: dict) -> dict:
    """Sections 8.1 and 8.3: groups of consecutive parts of one layout from above the coverage, closed
    at `rows_per_file` rows or a layout change; the new manifest, whole."""
    attempt = given["attempt"]
    parts = [p for p in _parts(given) if given.get("through") is None or p["first_step"] > given["through"]]
    groups, group = [], []
    for part in parts:
        if group and part["layout"] != group[0]["layout"]:
            groups.append(group)
            group = []
        group.append(part)
        if sum(p["rows"] for p in group) >= given["rows_per_file"]:
            groups.append(group)
            group = []
    number = given["number"]
    files = []
    for index, chosen in enumerate(groups):
        first, last = chosen[0]["first_step"], chosen[-1]["last_step"]
        files.append({"key": compact_key(attempt, first, last, number), "first_step": first, "last_step": last,
                      "rows": sum(p["rows"] for p in chosen), "bytes": given["compact_bytes"][index],
                      "layout": chosen[0]["layout"], "listed_since": number, "listed_since_ms": given["written_at_ms"],
                      "inputs": [p["key"] for p in chosen]})
    manifest = {"format": "tensorquill-manifest", "v": 1, "run_id": given["run_id"], "rank": given["rank"],
                "number": number, "previous": number - 1 if number > 1 else None,
                "written_at_ms": given["written_at_ms"],
                "attempts": [{"attempt": attempt,
                              "steps": {"through": files[-1]["last_step"] if files else given.get("through"),
                                        "files": files},
                              "scalars": {"through_seq": None, "files": []}}]}
    return {"key": manifest_key(number), "manifest": manifest, "tail": [p["key"] for p in group]}


_PART = re.compile(r"attempt=(-?\d+)/part-(-?\d+)-(-?\d+)\.parquet$")
_SCALARS = re.compile(r"attempt=(-?\d+)/scalars-(\d+)-(\d+)\.parquet$")


def _newest(manifests: dict, unreadable: list[int]) -> dict | None:
    numbers = sorted((int(n) for n in manifests if int(n) not in unreadable), reverse=True)
    return manifests[str(numbers[0])] if numbers else None


def replay_plan(given: dict) -> dict:
    """Section 8.4: per attempt in order, the listed compacted files (or their inputs when a file is
    missing), then the parts that are no listed file's input, by first step."""
    listing = set(given["listing"])
    manifest = _newest(given["manifests"], given.get("unreadable", []))
    read: list[str] = []
    for attempt in given["order"]:
        entries: list[tuple[int, int, str]] = []
        inputs: set[str] = set()
        entry = next((a for a in (manifest or {}).get("attempts", []) if a["attempt"] == attempt), None)
        for item in (entry or {}).get("steps", {}).get("files", []):
            inputs.update(item["inputs"])
            if item["key"] in listing:
                entries.append((item["first_step"], 0, item["key"]))
                continue
            missing = [key for key in item["inputs"] if key not in listing]
            if missing:
                return {"error": {"missing": [item["key"], *missing]}}
            for key in item["inputs"]:
                entries.append((int(_PART.search(key).group(2)), 1, key))
        for key in sorted(listing):
            found = _PART.search(key)
            if found and int(found.group(1)) == attempt and key not in inputs:
                entries.append((int(found.group(2)), 1, key))
        read += [key for _, _, key in sorted(entries)]
    return {"read": read}


def manifest_race(given: dict) -> dict:
    """Section 8.5: one writer wins a number; the other reads it and stops when it covers its inputs,
    or plans again on top of it."""
    newest = given["start"]
    planned = newest + 1  # every writer planned on the same newest manifest, so the same `gen`
    files = {key: list(inputs) for key, inputs in given["start_files"].items()}
    written = []
    unreferenced = []
    for writer in given["writers"]:  # in the order they reach the manifest's create
        name = writer["file"].replace("{gen}", f"{planned:010}")
        covered = {key for inputs in files.values() for key in inputs}
        if set(writer["inputs"]) <= covered:
            if name not in files:  # the same group's file is the winner's own (section 8.5, a reuse)
                unreferenced.append(name)
            continue
        newest += 1
        files[name] = writer["inputs"]
        written.append(newest)
    return {"written": written, "newest": newest, "newest_lists": sorted(files), "unreferenced": unreferenced}


def orphan(given: dict) -> dict:
    """Section 8.5: a compacted file no manifest lists is never read; the next compaction of the same
    group reuses it when its inputs are the same, and takes the next free name otherwise."""
    planned = compact_key(given["attempt"], given["first_step"], given["last_step"], given["number"])
    if planned != given["orphan"]:
        return {"replay_reads_it": False, "action": "write", "key": planned}
    if given["orphan_inputs"] == given["planned_inputs"]:
        return {"replay_reads_it": False, "action": "reuse", "key": planned}
    return {"replay_reads_it": False, "action": "new_name",
            "key": compact_key(given["attempt"], given["first_step"], given["last_step"], given["number"] + 1)}


def watermarks(given: dict) -> dict:
    """Sections 5 and 8.7: the highest of the names and the manifest's coverage."""
    steps = [int(m.group(3)) for key in given["names"] if (m := _PART.search(key))]
    seqs = [int(m.group(3)) for key in given["names"] if (m := _SCALARS.search(key))]
    through, through_seq = given.get("through"), given.get("through_seq")
    today_step = max(steps) if steps else None
    today_seq = max(seqs) if seqs else None
    step = max([s for s in (today_step, through) if s is not None], default=None)
    seq = max([s for s in (today_seq, through_seq) if s is not None], default=None)
    return {"step": step, "seq": seq, "today_step": today_step, "today_seq": today_seq}


def snapshot_usable(rows: list[dict]) -> list[dict]:
    """Section 7.4, step 3, in its order, and the tail when the snapshot is used."""
    out = []
    for row in rows:
        snap, now = row["snapshot"], row["store"]
        attempts = [a["attempt"] for a in snap["attempts"]]
        parts = {int(a): [tuple(p) for p in ps] for a, ps in now["parts"].items()}
        if snap["v"] != now["state_version"]:
            out.append({"use": False, "why": "state_version"})
        elif snap["thresholds"] != now["thresholds"]:
            out.append({"use": False, "why": "thresholds"})
        elif now["order"][: len(attempts)] != attempts:
            out.append({"use": False, "why": "attempt_order"})
        elif len(now["order"]) > len(attempts):
            out.append({"use": False, "why": "new_attempt"})
        elif any(sum(1 for first, last in parts.get(a["attempt"], []) if last <= a["through_step"]) != a["parts"]
                 for a in snap["attempts"]):
            out.append({"use": False, "why": "parts_below"})
        elif any(first > a["through_step"] for a in snap["attempts"][:-1] for first, _ in parts.get(a["attempt"], [])):
            out.append({"use": False, "why": "not_a_continuation"})
        else:
            newest = snap["attempts"][-1]
            tail = [list(p) for p in sorted(parts.get(newest["attempt"], [])) if p[0] > newest["through_step"]]
            out.append({"use": True, "why": None, "tail": tail})
    return out


def retired(given: dict) -> dict:
    """Section 9: the decided parts and manifests, and the proposed snapshots and unreferenced
    compacted files, each once its clock is older than the grace."""
    grace = given["grace_hours"] * HOUR_MS
    age = {key: NOW_MS - hours * HOUR_MS for key, hours in given["listing"].items()}
    manifests = _manifests_at(given["manifests"])
    newest_number = max(manifests)
    newest = manifests[newest_number]
    out = []
    listed = set()
    for entry in newest["attempts"]:
        for family in ("steps", "scalars"):
            for item in entry[family]["files"]:
                listed.add(item["key"])
                if item["listed_since_ms"] <= NOW_MS - grace:
                    out += [[key, "input"] for key in item["inputs"] if key in age]
    numbers = sorted(manifests)
    for low, high in zip(numbers, numbers[1:]):
        if manifests[high]["written_at_ms"] <= NOW_MS - grace and manifest_key(low) in age:
            out.append([manifest_key(low), "superseded_manifest"])
    order = given["order"]
    snaps = sorted((key for key in age if key.startswith("checkpoints/state-")),
                   key=lambda key: (order.index(int(key.split("-")[1])), int(key.split("-")[2].split(".")[0])))
    if snaps and age[snaps[-1]] <= NOW_MS - grace:
        out += [[key, "superseded_snapshot"] for key in snaps[:-1]]
    for key in age:
        if "/compact-" in key and key not in listed and age[key] <= NOW_MS - grace:
            out.append([key, "unreferenced_compact"])
    return {"retired": sorted(out)}


def _manifests_at(spec: dict) -> dict[int, dict]:
    """Manifests given with ages in hours (`written_age_hours`, `listed_since_age_hours`), made whole."""
    out = {}
    for number, doc in spec.items():
        made = json.loads(json.dumps(doc))
        made["written_at_ms"] = NOW_MS - made.pop("written_age_hours") * HOUR_MS
        for entry in made["attempts"]:
            for family in ("steps", "scalars"):
                for item in entry[family]["files"]:
                    item["listed_since_ms"] = NOW_MS - item.pop("listed_since_age_hours") * HOUR_MS
        out[int(number)] = made
    return out


def globs(given: dict) -> dict:
    """Section 8.8: what an outside reader reads with a glob over the attempt folder's names."""
    out = {}
    for pattern in given["patterns"]:
        counts: dict[int, int] = {}
        scalar_rows = False
        for key, spans in given["listing"].items():
            if not fnmatch.fnmatch(key.rsplit("/", 1)[-1], pattern):
                continue
            if spans is None:
                scalar_rows = True
                continue
            for step in range(spans[0], spans[1] + 1):
                counts[step] = counts.get(step, 0) + 1
        ranges: dict[str, list[list[int]]] = {}
        for step in sorted(counts):
            runs = ranges.setdefault(str(counts[step]), [])
            if runs and runs[-1][1] == step - 1:
                runs[-1][1] = step
            else:
                runs.append([step, step])
        out[pattern] = {"times_read": ranges, "scalar_rows": scalar_rows}
    return out


def quota(given: dict) -> list[int]:
    """Section 10: min(max(quota, streams x share), ceiling), in MiB."""
    return [min(max(given["quota_mib"], streams * given["share_mib"]), given["ceiling_mib"])
            for streams in given["streams"]]


def api_errors(given: dict) -> list[dict]:
    """Section 2.5, 2.2 (Q35) and 6.2.2 for the requests given, against a store of one run with one rank: with
    no token set (0.1.0), or with one (0.2.0)."""
    out = []
    for request in given["requests"]:
        path, token = request["path"], request.get("token")
        query = dict(part.split("=", 1) for part in path.partition("?")[2].split("&") if part)
        route = path.partition("?")[0]
        if route == "/healthz":
            out.append({"status": 200, "code": None})
        elif given["token_set"] and token != "right":
            out.append({"status": 401, "code": "unauthenticated"})
        elif route.startswith("/v1/runs/") and route.split("/")[3] != given["run_id"]:
            out.append({"status": 404, "code": "run_not_found"})
        elif "/ranks/" in route and not 0 <= int(route.split("/")[5]) < MAX_RANKS:
            out.append({"status": 400, "code": "bad_parameter"})
        elif "/ranks/" in route and int(route.split("/")[5]) not in given["ranks"]:
            out.append({"status": 404, "code": "rank_not_found"})
        elif "limit" in query and not 1 <= int(query["limit"]) <= 1000:
            out.append({"status": 400, "code": "bad_parameter"})
        elif "page_size" in query and not 1 <= int(query["page_size"]) <= 10000:
            out.append({"status": 400, "code": "bad_parameter"})
        elif "columns" in query and any(c not in given["columns"] for c in query["columns"].split(",")):
            out.append({"status": 400, "code": "unknown_column"})
        else:
            out.append({"status": 200, "code": None})
    return out


MAX_RANKS = 100_000  # watchdog::MAX_RANKS on the base
DEFAULT_UI = "127.0.0.1:50052"  # D16, D17
FREE_PORT_HINT = "--http-listen 127.0.0.1:0"  # D20
ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]")  # Q23
FRAME_HEADERS = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'"}  # D22, Q25


def catalog_place(given: dict) -> list[dict]:
    """Section 1.2 (Q2's answer, D24, Q27): a named file, or the state folder on a local disk, or the per-user
    temporary folder, or off and said. A type that cannot be read counts as a network filesystem."""
    name = f"catalog-{given['store_id']}-v{given['schema']}.sqlite3"
    out = []
    for row in given["rows"]:
        flag = row.get("flag")
        if flag == "off":
            out.append({"catalog": None, "said": "flag_off"})
        elif flag:
            if flag.startswith(given["store_root"].rstrip("/") + "/"):
                out.append({"refused": "under_store_root"})
            elif row["flag_fs"] != "local":
                out.append({"refused": "network_filesystem"})
            else:
                out.append({"catalog": flag})
        elif row["state_fs"] == "local":
            out.append({"catalog": f"{given['state']}/tensorquill/{name}"})
        elif row["tmp_fs"] == "local" and row.get("tmp_usable", True):
            out.append({"catalog": f"{given['tmp']}/tensorquill-{given['uid']}/{name}"})
        else:
            out.append({"catalog": None, "said": "no_local_folder"})
    return out


def _address(args: list[str]) -> str | None:
    if "--http-listen" in args:
        return args[args.index("--http-listen") + 1]
    return DEFAULT_UI if "--ui" in args else None


def http_listen(rows: list[dict]) -> list[dict]:
    """Section 2.1: which address the read API takes from `--http-listen` and `--ui` (D17), loopback only in
    0.1.0 (Q26), and the stop when the address is in use (D20)."""
    out = []
    for row in rows:
        address = _address(row["args"])
        if row.get("taken"):
            out.append({"stops": True, "exit": 1, "names": [address, FREE_PORT_HINT]})
        elif address is None:
            out.append({"http": None, "page": False})
        elif not ipaddress.ip_address(address.rsplit(":", 1)[0].strip("[]")).is_loopback:
            out.append({"refused": "loopback_only"})
        else:
            out.append({"http": address, "page": "--ui" in row["args"]})
    return out


def start_lines(rows: list[dict]) -> list[list[str]]:
    """Section 2.1 (D18, D19, Q33): the lines after Flight's on standard output."""
    out = []
    for row in rows:
        host, port = row["bound"].rsplit(":", 1)
        lines = [f"tensorquill-server http on {row['bound']}", f"http://{row['bound']}"]
        if any(row["env"].get(key) for key in ("SSH_CONNECTION", "SSH_TTY")):
            lines.append(f"on your own machine: ssh -N -L {port}:{host}:{port} <user>@<server>")
        out.append(lines)
    return out


def relative_urls(given: dict) -> dict:
    """Section 2.2 (D21): what a browser asks for, from each page URL, for each path the page names."""
    return {base: [urllib.parse.urljoin(base, path) for path in given["paths"]] for base in given["bases"]}


def _host_name(value: str | None) -> str | None:
    """A `Host` header's name, lowercased, with an IPv6 address in brackets; None for one that does not parse."""
    if value is None:
        return None
    text = value.strip().lower()
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return None
        name, rest = text[: end + 1], text[end + 1:]
    else:
        name, colon, port = text.partition(":")
        rest = colon + port
    if rest and not (rest.startswith(":") and rest[1:].isdigit()):
        return None
    if not name:
        return None
    return name  # a trailing dot parses, and names no allowed host


def _origin_name(value: str) -> str | None:
    if value == "null":
        return None
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname


def allowlist(given: dict) -> list[dict]:
    """Section 6.2.1 (D23, Q23, Q24): `/healthz` unchecked; then the `Host` (none, or one that does not parse, is a bad
    request, as RFC 9112 section 3.2 has it), then the `Origin` when there is one."""
    names = set(ALLOWED_HOSTS) | {name.lower() for name in given["allow_host"]}
    out = []
    for request in given["requests"]:
        if request["path"] == "/healthz":
            out.append({"status": 200, "code": None})
        elif _host_name(request.get("host")) is None:
            out.append({"status": 400, "code": "bad_request"})
        elif _host_name(request.get("host")) not in names:
            out.append({"status": 403, "code": "host_not_allowed"})
        elif "origin" in request and _origin_name(request["origin"]) not in names:
            out.append({"status": 403, "code": "origin_not_allowed"})
        else:
            out.append({"status": 200, "code": None})
    return out


def frame_headers(given: dict) -> list[dict]:
    """Section 6.2.1 (D22, Q25): the two headers on every answer."""
    return [dict(FRAME_HEADERS) for _ in given["requests"]]


def runs_pages(given: dict) -> list[list[str]]:
    """Section 2.2: `/v1/runs`, newest status first, then run ID, `limit` a page."""
    runs = sorted(given["runs"], key=lambda run: (-run["status_written_ms"], run["run_id"]))
    ids = [run["run_id"] for run in runs]
    return [ids[at:at + given["limit"]] for at in range(0, len(ids), given["limit"])]


def _each(rule):
    """A rule over a list of scenarios, each given and expected in order."""
    return lambda given: [rule(one) for one in given] if isinstance(given, list) else rule(given)


RULES = {
    "catalog_start": catalog_start, "catalog_events": catalog_events, "series": series, "scan": _each(scan),
    "scan_view": scan_view, "stored_cutoff": stored_cutoff, "scalar_series": _each(scalar_series),
    "names": names, "compaction": compaction, "replay_plan": replay_plan, "manifest_race": _each(manifest_race),
    "orphan": _each(orphan), "watermarks": _each(watermarks), "snapshot_usable": snapshot_usable,
    "retired": retired, "globs": globs, "quota": quota, "api_errors": api_errors, "runs_pages": runs_pages,
    "catalog_place": catalog_place, "http_listen": http_listen, "start_lines": start_lines,
    "relative_urls": relative_urls, "allowlist": allowlist, "frame_headers": frame_headers,
}


# --- the document and the rules ------------------------------------------------------------------


def _page() -> str:
    """The contract page's text, or the calling test skipped where no page is at hand (conftest.contract_page)."""
    return conftest.contract_page(DOCUMENT["contract"])


def _decision_rows() -> set[str]:
    return set(re.findall(r"^\| (D\d+|Q\d+) \|", _page(), flags=re.M))


def test_the_document_and_the_page():
    assert DOCUMENT["contract"] == "WAVE_AN_CONTRACT.md"
    assert DOCUMENT["base"] == "1dddcce262373c1b3a552b53dff6ec639a8c2cb7"
    names_seen = [case["name"] for case in CASES]
    assert len(names_seen) == len(set(names_seen))
    for case in CASES:
        assert case["section"] in range(1, 20), case["name"]
        assert case["rests_on"], case["name"]
        assert case["paths"] and set(case["paths"]) <= {"rule", "http", "python", "server"}, case["name"]
        assert case["rule"] is None or case["rule"] in RULES, case["name"]
        assert case["release"] in ("0.1.0", "0.2.0"), case["name"]
        assert case["why"], case["name"]
    # From here on the page is read: a checkout with no page skips the rest.
    page = _page()
    assert DOCUMENT["base"][:7] in page
    rows = _decision_rows()
    assert {f"D{n}" for n in range(1, 6)} <= rows
    for case in CASES:
        assert f"`{case['name']}`" in page, case["name"]
        assert set(case["rests_on"]) <= rows, case["name"]


def _stands() -> dict[str, str]:
    """Each decision row's third cell, by its number."""
    out = {}
    for line in _page().splitlines():
        found = re.match(r"^\| (D\d+|Q\d+) \|", line)
        if found:
            out[found.group(1)] = line.split(" | ")[2]
    return out


def test_each_decision_row_is_in_one_state():
    """The page's states: a D row is decided; a Q row is answered by the owner (with its date and its ledger
    row), moved to 0.2.0, or open. Of the 22 questions first put on Nov 24, the records answer or move every
    one, so none is open."""
    stands = _stands()
    for row, cell in stands.items():
        if row.startswith("D"):
            assert cell.startswith(("**Decided", "Decided by")), row
            continue
        states = [state for state in ("**Answered by the owner**", "**Moved to 0.2.0**", "**Open**") if cell.startswith(state)]
        assert len(states) == 1, row
        if states[0] == "**Answered by the owner**":
            assert re.search(r"\(Nov 2\d, ", cell) and "row " in cell, row
        if int(row[1:]) <= 22:
            assert states[0] != "**Open**", row


def _listed(label: str) -> set[str]:
    line = next(line for line in _page().splitlines() if line.startswith(label))
    return set(re.findall(r"`([a-z0-9_]+)`", line))


def test_the_cases_are_listed_by_release():
    """"The trim to 0.1.0" lists every case under its release, as `cases.json` marks it."""
    assert _listed("**In 0.1.0:**") == {c["name"] for c in CASES if c["release"] == "0.1.0"}
    assert _listed("**In 0.2.0, kept as written:**") == {c["name"] for c in CASES if c["release"] == "0.2.0"}


def test_the_decided_lines_are_kept():
    """The five lines the plan records as decided on Nov 24 are on the page as decided, not reopened."""
    for row in ("D1", "D2", "D3", "D4", "D5"):
        line = next(line for line in _page().splitlines() if line.startswith(f"| {row} |"))
        assert "**Decided (Nov 24)**, not reopened here" in line, row


@pytest.mark.parametrize("case", [c for c in CASES if c["rule"]], ids=lambda c: c["name"])
def test_each_case_is_what_the_page_gives(case):
    assert RULES[case["rule"]](case["input"]) == case["expect"]


def test_the_downsampled_series_keeps_every_spike_and_stays_bounded():
    """Section 3.1's two promises, beyond the cases: at most `points` points, and every spike kept, on
    series with one spike or one non-finite value at the start of every third bucket."""
    for n, limit in ((1000, 100), (10_007, 1000), (6, 5), (200, 7), (4_000, 11)):
        buckets = (limit - 2) // 3
        marked = {i * n // buckets: (100.0 + i if i % 2 else "nan") for i in range(0, buckets, 3)}
        spec = {"steps": n, "value": 1.0, "set": {str(step): value for step, value in marked.items()}}
        got = series({"series": spec, "points": limit})
        assert got["downsampled"] and len(got["points"]) <= limit, (n, limit)
        kept = {step for step, _ in got["points"]}
        assert set(marked) <= kept, (n, limit, sorted(set(marked) - kept))
    short = series({"series": {"steps": 5, "value": 1.0}, "points": 5})
    assert not short["downsampled"] and len(short["points"]) == 5


# --- targets: the product paths ------------------------------------------------------------------


def _binary() -> str:
    path = os.environ.get("TENSORQUILL_SERVER_BIN")
    if not path or not Path(path).is_file():
        pytest.skip("needs the server binary (TENSORQUILL_SERVER_BIN)")
    return path


def _pyarrow():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    return pa, pq


def _records(steps: range, attempt: int = ATTEMPT, grad=lambda s: 1.0, run_id: str = RUN) -> list[dict]:
    return [{"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": attempt, "step": s, "ts": 1.0 + s,
             "grad_norm": grad(s), "weight_norm": 10.0 + s / 1000, "lr": [1e-3], "scalars": {"loss": 2.0 - s / 1000}}
            for s in steps]


def _write_parts(rank_folder: Path, records: list[dict], every: int) -> list[Path]:
    """Protocol 5 parts as the server names them, from the SDK's own rows."""
    pa, pq = _pyarrow()
    from tensorquill import flight

    written = []
    for at in range(0, len(records), every):
        chunk = records[at:at + every]
        folder = rank_folder / f"attempt={chunk[0]['attempt']}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"part-{chunk[0]['step']:08}-{chunk[-1]['step']:08}.parquet"
        pq.write_table(pa.Table.from_batches([flight.records_to_batch(chunk)]), path, compression="zstd")
        written.append(path)
    return written


class _Server:
    """`tensorquill-server` on a store with the read API asked for. A server that does not print its
    HTTP line, which is every server before lane AN-2, fails the test on an assertion."""

    def __init__(self, store: Path, *extra: str, token: str | None = None, http: bool = True,
                 env: dict[str, str] | None = None, lines: int | None = None) -> None:
        environment = {k: v for k, v in os.environ.items()
                       if k not in ("TENSORQUILL_TOKEN", "SSH_CONNECTION", "SSH_TTY")}
        if token:
            environment["TENSORQUILL_TOKEN"] = token
        environment.update(env or {})
        args = [_binary(), "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0", *extra]
        if http:
            args[3:3] = ["--http-listen", "127.0.0.1:0"]
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        env=environment)
        self.flight = self.http = None
        self.lines = lines = self._lines(lines or (2 if http else 1))
        for line in lines:
            if line.startswith("tensorquill-server listening on "):
                self.flight = "grpc://127.0.0.1:" + line.rsplit(":", 1)[1].strip()
            if line.startswith("tensorquill-server http on "):
                self.http = "http://127.0.0.1:" + line.rsplit(":", 1)[1].strip()
                self.http_address = line.removeprefix("tensorquill-server http on ").strip()
        if self.flight is None or (http and self.http is None):
            err = self.close()
            raise AssertionError(f"the server did not start as asked ({' '.join(args[1:])}): "
                                 f"exit {self.process.returncode}, stderr: {err[-600:]}")

    def _lines(self, wanted: int, deadline: float = 20.0) -> list[str]:
        lines: list[str] = []
        with selectors.DefaultSelector() as chosen:
            chosen.register(self.process.stdout, selectors.EVENT_READ)
            end = time.monotonic() + deadline
            while len(lines) < wanted and time.monotonic() < end:
                if not chosen.select(timeout=max(0.0, end - time.monotonic())):
                    break
                line = self.process.stdout.readline()
                if not line:
                    break
                lines.append(line.strip())
        return lines

    def get(self, path: str, token: str | None = None) -> tuple[int, object]:
        request = urllib.request.Request(self.http + path)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body, status = response.read().decode(), response.status
        except urllib.error.HTTPError as error:
            body, status = error.read().decode(), error.code
        try:
            return status, json.loads(body)
        except ValueError:
            return status, body

    def raw(self, path: str, host: str | None, origin: str | None = None) -> tuple[int, dict, object]:
        """One GET with the `Host` and `Origin` given (no `Host` at all for None), as a browser or a
        rebinding page would send them: the status, the headers (names in lower case, as HTTP/1.1 compares
        them without regard to case) and the body."""
        port = int(self.http.rsplit(":", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
            if host is not None:
                connection.putheader("Host", host)
            if origin is not None:
                connection.putheader("Origin", origin)
            connection.endheaders()
            response = connection.getresponse()
            status, text = response.status, response.read().decode()
            headers = {name.lower(): value for name, value in response.getheaders()}
        finally:
            connection.close()
        try:
            return status, headers, json.loads(text)
        except ValueError:
            return status, headers, text

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(10)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


@pytest.fixture
def server_factory():
    started: list[_Server] = []

    def start(*args, **kwargs) -> _Server:
        server = _Server(*args, **kwargs)
        started.append(server)
        return server

    yield start
    for server in started:
        if server.process.poll() is None or not server.process.stdout.closed:
            server.close()


def _rank_folder(store: Path, run_id: str = RUN) -> Path:
    return store / "runs" / run_id / "rank=0"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_API)
def test_target_api_runs_page(tmp_path, server_factory):
    case = BY_NAME["api_runs_page"]
    _pyarrow()
    store = tmp_path / "store"
    for run in case["input"]["runs"]:
        folder = store / "runs" / run["run_id"]
        _write_parts(folder / "rank=0", _records(range(10), run_id=run["run_id"]), 10)
        (folder / "run.json").write_text(json.dumps({"run_id": run["run_id"], "project": run["project"]}))
        (folder / "watchdog.json").write_text(json.dumps(
            {"run_id": run["run_id"], "written_at_ms": run["status_written_ms"], "state": "finished"}))
    server = server_factory(store)
    pages, token = [], None
    for _ in range(len(case["expect"]) + 1):
        status, body = server.get(f"/v1/runs?limit={case['input']['limit']}" + (f"&page_token={token}" if token else ""))
        assert status == 200, body
        pages.append([run["run_id"] for run in body["runs"]])
        token = body.get("next_page_token")
        if not token:
            break
    assert pages == case["expect"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_API)
def test_target_api_errors(tmp_path, server_factory):
    case = BY_NAME["api_errors"]
    given = case["input"]
    store = tmp_path / "store"
    _write_parts(_rank_folder(store, given["run_id"]), _records(range(10), run_id=given["run_id"]), 10)
    server = server_factory(store)
    got = []
    for request in given["requests"]:
        status, body = server.get(request["path"])
        got.append({"status": status, "code": body["error"]["code"] if isinstance(body, dict) and "error" in body else None})
    assert got == case["expect"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_API)
@pytest.mark.parametrize("name", ["series_keeps_spikes", "series_short_is_whole"])
def test_target_series(tmp_path, server_factory, name):
    case = BY_NAME[name]
    points = _points(case["input"]["series"])
    values = {step: value for step, value in points}
    store = tmp_path / "store"
    _write_parts(_rank_folder(store), _records(range(len(points)), grad=lambda s: float(values[s])), 50)
    server = server_factory(store)
    status, body = server.get(f"/v1/runs/{RUN}/ranks/0/series?kind=grad_norm&points={case['input']['points']}")
    assert status == 200, body
    got = {key: body.get(key) for key in ("points", "downsampled", "of", "buckets")}
    assert got == case["expect"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_API)
def test_target_scan_pages_by_step_range(tmp_path, server_factory):
    case = BY_NAME["scan_pages_by_step_range"]
    given = case["input"]
    store = tmp_path / "store"
    rows = _rows(given["rows"])
    records = _records(range(len(rows)), grad=lambda s: s / 10)
    _write_parts(_rank_folder(store), records, 10)
    server = server_factory(store)
    query = (f"/v1/runs/{RUN}/ranks/0/rows?columns={','.join(given['columns'])}&from_step={given['from_step']}"
             f"&to_step={given['to_step']}&page_size={given['page_size']}")
    pages, token = [], None
    for _ in range(len(case["expect"]["pages"]) + 1):
        status, body = server.get(query + (f"&page_token={token}" if token else ""))
        assert status == 200, body
        token = body.get("next_page_token")
        pages.append({"rows": body["rows"], "next": bool(token)})
        if not token:
            break
    assert pages == case["expect"]["pages"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_ALLOW)
def test_target_host_allowlist(tmp_path, server_factory):
    """Section 6.2.1: each request of case `host_allowlist` sent with its own `Host` and `Origin`, to a server
    told `--allow-host` for the case's extra name."""
    case = BY_NAME["host_allowlist"]
    extra = [arg for name in case["input"]["allow_host"] for arg in ("--allow-host", name)]
    server = server_factory(tmp_path / "store", *extra)
    got = []
    for request in case["input"]["requests"]:
        status, _, body = server.raw(request["path"], request.get("host"), request.get("origin"))
        got.append({"status": status, "code": body["error"]["code"] if isinstance(body, dict) and "error" in body else None})
    assert got == case["expect"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_ALLOW)
def test_target_framing_refused(tmp_path, server_factory):
    """Section 6.2.1 (D22): the two headers on every answer, a refusal and an error included."""
    case = BY_NAME["framing_refused"]
    server = server_factory(tmp_path / "store")
    got = []
    for request in case["input"]["requests"]:
        _, headers, _ = server.raw(request["path"], request.get("host", server.http_address))
        got.append({name: headers.get(name.lower()) for name in FRAME_HEADERS})
    assert got == case["expect"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_LISTEN)
def test_target_start_lines(tmp_path, server_factory):
    """Section 2.1 (D18, D19): after Flight's line, the address, the URL, and the tunnel hint when the server
    is started from an SSH session."""
    ssh = {"SSH_CONNECTION": "10.0.0.2 52000 10.0.0.9 22"}
    server = server_factory(tmp_path / "store", env=ssh, lines=4)
    assert server.lines[0].startswith("tensorquill-server listening on "), server.lines
    assert server.lines[1:] == start_lines([{"bound": server.http_address, "env": ssh}])[0]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_LISTEN)
def test_target_port_taken(tmp_path):
    """Section 2.1 (D20): an address in use stops the server, exit 1, with a line naming the address and
    `--http-listen 127.0.0.1:0`. On the base the server refuses `--http-listen` itself (exit 2), and its message
    names neither."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        address = f"127.0.0.1:{taken.getsockname()[1]}"
        args = [_binary(), "--listen", "127.0.0.1:0", "--store", str(tmp_path / "store"), "--watchdog-seconds", "0",
                "--http-listen", address]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env={k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"})
        try:
            out, err = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=30)
            raise AssertionError(f"the server went on with {address} in use: {out[-300:]}")
    expect = http_listen([{"args": ["--http-listen", address], "taken": True}])[0]
    assert process.returncode == expect["exit"], (process.returncode, err[-300:])
    line = next((line for line in err.splitlines() if all(name in line for name in expect["names"])), None)
    assert line is not None, err[-600:]


def _scalar_store(root: Path, given: dict) -> Path:
    """A store holding a scenario of a scalar series case: the training records as `part-*` objects and the
    scalar-only records as `scalars-*` objects (`WAVE_AL_CONTRACT.md` section 4.5), each attempt in its folder,
    with the rank's attempt order; rows made by the SDK's own `flight.record_to_row`."""
    pa, pq = _pyarrow()
    from tensorquill import flight

    schema = flight.flight_schema()
    rank = _rank_folder(root)
    training: dict[int, list[dict]] = {}
    for item in given["training"]:
        training.setdefault(item["attempt"], []).append(
            {"schema_version": 1, "run_id": RUN, "rank": 0, "attempt": item["attempt"], "step": item["step"],
             "ts": 1.0 + item["step"], "grad_norm": 1.0, "scalars": item["scalars"]})
    scalar_only: dict[int, list[dict]] = {}
    for item in given["scalar_only"]:
        scalar_only.setdefault(item["attempt"], []).append(
            {"schema_version": 2, "record": "scalars", "run_id": RUN, "rank": 0, "attempt": item["attempt"],
             "step": item["step"], "seq": item["seq"], "ts": 100.0 + item["seq"], "scalars": item["scalars"]})
    for attempt, records in training.items():
        folder = rank / f"attempt={attempt}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([flight.record_to_row(r) for r in records], schema=schema),
                       folder / f"part-{records[0]['step']:08}-{records[-1]['step']:08}.parquet")
    for attempt, records in scalar_only.items():
        records.sort(key=lambda record: record["seq"])
        folder = rank / f"attempt={attempt}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([flight.record_to_row(r) for r in records], schema=schema),
                       folder / f"scalars-{records[0]['seq']:08}-{records[-1]['seq']:08}.parquet")
    (rank / "attempts.json").write_text(json.dumps({"schema_version": 1, "run_id": RUN, "rank": 0, "attempts": [
        {"attempt": attempt, "by": "claim"} for attempt in given["order"]]}))
    return root / "runs" / RUN


def _resolved_scenario(name: str) -> tuple[dict, list]:
    """The scenario of a scalar series case read with the default `attempts=resolved`, and its points."""
    case = BY_NAME[name]
    if isinstance(case["input"], list):
        index = next(i for i, given in enumerate(case["input"]) if given.get("attempts", "resolved") == "resolved")
        return case["input"][index], case["expect"][index]["points"]
    return case["input"], case["expect"]["points"]


@pytest.mark.parametrize("name", ["scalar_series_merge", "scalar_series_rolled_back"])
def test_pin_scalar_series_python_reader(tmp_path, name):
    """D13 on the base: today's `evidence.scalar_series` on a store of `part-*` and `scalars-*` objects gives the
    case's resolved points, the series the read API must give too."""
    from tensorquill import evidence

    given, points = _resolved_scenario(name)
    run = _scalar_store(tmp_path, given)
    assert evidence.scalar_series(run)[given["name"]] == points


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_SCALARS)
@pytest.mark.parametrize("name", ["scalar_series_merge", "scalar_series_rolled_back"])
def test_target_scalar_series_over_http(tmp_path, server_factory, name):
    given, points = _resolved_scenario(name)
    _scalar_store(tmp_path / "store", given)
    server = server_factory(tmp_path / "store")
    status, body = server.get(f"/v1/runs/{RUN}/ranks/0/series?kind=scalar&name={given['name']}")
    assert status == 200, body
    assert (body["points"], body["downsampled"]) == (points, False)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_RETIRED)
def test_target_retired_listing(tmp_path, server_factory):
    case = BY_NAME["retired_listing"]
    given = case["input"]
    rank = _rank_folder(tmp_path / "store")
    now = time.time() * 1000
    for key, hours in given["listing"].items():
        path = rank / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        stamp = (now - hours * HOUR_MS) / 1000
        os.utime(path, (stamp, stamp))
    for number, doc in _manifests_at(given["manifests"]).items():
        shift = now - NOW_MS
        doc["written_at_ms"] += shift
        for entry in doc["attempts"]:
            for family in ("steps", "scalars"):
                for item in entry[family]["files"]:
                    item["listed_since_ms"] += shift
        (rank / manifest_key(number)).write_text(json.dumps(doc))
    server = server_factory(tmp_path / "store")
    status, body = server.get(f"/v1/retired?run_id={RUN}&grace_hours={given['grace_hours']}")
    assert status == 200, body
    prefix = f"runs/{RUN}/rank=0/"
    got = sorted([item["key"].removeprefix(prefix), item["reason"]] for item in body["retired"])
    assert got == case["expect"]["retired"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_FILES)
def test_target_files_in_replay_order(tmp_path, server_factory):
    """Section 2.2's `/v1/runs/{id}/files`: the objects a reader reads, each once, in the order the
    manifest-first replay reads them (section 8.4), so the names and the order are seen from outside.
    The objects are empty: neither the listing nor the manifest is read for rows."""
    case = BY_NAME["replay_manifest_first_order"]
    given = case["input"]
    rank = _rank_folder(tmp_path / "store")
    for key in given["listing"]:
        path = rank / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    for number, doc in given["manifests"].items():
        (rank / manifest_key(int(number))).write_text(json.dumps(doc) if int(number) not in given["unreadable"] else "{")
    (rank / "attempts.json").write_text(json.dumps({"schema_version": 1, "run_id": RUN, "rank": 0, "attempts": [
        {"attempt": attempt, "by": "claim"} for attempt in given["order"]]}))
    server = server_factory(tmp_path / "store")
    status, body = server.get(f"/v1/runs/{RUN}/files?rank=0")
    assert status == 200, body
    prefix = f"runs/{RUN}/rank=0/"
    assert [item["key"].removeprefix(prefix) for item in body["files"]] == case["expect"]["read"]


def _put(endpoint: str, batches: list, start: int) -> int:
    """One DoPut stream, an acknowledgement awaited per batch, as the SDK's sender does."""
    flight = pytest.importorskip("pyarrow.flight")
    import pyarrow as pa

    client = flight.FlightClient(endpoint)
    descriptor = flight.FlightDescriptor.for_path("runs", RUN, "rank=0")
    writer, reader = client.do_put(descriptor, batches[0].schema)
    end = start
    for batch in batches:
        steps = batch.column("step").to_pylist()
        metadata = {"schema_version": 2, "run_id": RUN, "rank": 0, "min_step": min(steps), "max_step": max(steps),
                    "start_byte_offset": end, "end_byte_offset": end + 1000}
        end += 1000
        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
        assert reader.read() is not None, "no acknowledgement"
    writer.done_writing()
    writer.close()
    client.close()
    return end


def _report_reaches(rank: Path, last_step: int, deadline: float = 60.0) -> bytes:
    path = rank / "incidents.json"
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if path.is_file():
            data = path.read_bytes()
            try:
                reached = json.loads(data).get("last_step")
                if isinstance(reached, int) and reached >= last_step:
                    return data
            except ValueError:
                pass
        time.sleep(0.1)
    raise AssertionError(f"the report never reached step {last_step}")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_SNAPSHOT)
def test_target_snapshot_restart_equals_replay(tmp_path, server_factory):
    """Section 7.4's requirement: a report after a restart from a snapshot equals the report after a full
    replay of the same parts, byte for byte. `--snapshot-every-parts` is the test setting section 13.3
    names; 0 turns snapshots off."""
    _pyarrow()
    from tensorquill import flight

    records = _records(range(350), grad=lambda s: 40.0 if s == 330 else 1.0 + (s % 7) / 10)
    batches = [flight.records_to_batch(records[at:at + 50]) for at in range(0, 350, 50)]
    first = tmp_path / "with-snapshots"
    server = server_factory(first, "--flush-steps", "50", "--snapshot-every-parts", "1", http=False)
    offset = _put(server.flight, batches[:6], 0)
    _report_reaches(_rank_folder(first), 299)
    server.close()
    snapshots = sorted((_rank_folder(first) / "checkpoints").glob("state-*.json.zst"))
    assert snapshots, "no snapshot was written"
    second = tmp_path / "replayed"
    shutil.copytree(first, second, ignore=shutil.ignore_patterns("checkpoints"))
    reports = []
    for store, every in ((first, "1"), (second, "0")):
        restarted = server_factory(store, "--flush-steps", "50", "--snapshot-every-parts", every, http=False)
        _put(restarted.flight, batches[6:], offset)
        reports.append(_report_reaches(_rank_folder(store), 349))
        restarted.close()
    assert reports[0] == reports[1]


def _compacted_store(root: Path, delete_inputs: bool) -> tuple[Path, list[int]]:
    """A rank of 250 steps in 50-step parts, steps 0 to 199 compacted into one file listed by manifest 1
    (section 8.3), its inputs kept or deleted as an operator would after the retired listing."""
    pa, pq = _pyarrow()
    from tensorquill import flight

    records = _records(range(250))
    rank = _rank_folder(root)
    parts = _write_parts(rank, records, 50)
    compacted = rank / compact_key(ATTEMPT, 0, 199, 1)
    pq.write_table(pa.Table.from_batches([flight.records_to_batch(records[:200])]), compacted, compression="zstd")
    plan = compaction({"run_id": RUN, "rank": 0, "attempt": ATTEMPT, "runs": [
        {"first": 0, "last": 249, "every": 50, "layout": 5}], "rows_per_file": 200, "number": 1,
        "written_at_ms": NOW_MS, "compact_bytes": [compacted.stat().st_size]})
    assert plan["manifest"]["attempts"][0]["steps"]["files"][0]["key"] == compacted.relative_to(rank).as_posix()
    (rank / "manifests").mkdir()
    (rank / plan["key"]).write_text(json.dumps(plan["manifest"]))
    if delete_inputs:
        for part in parts[:4]:
            part.unlink()
    return root / "runs" / RUN, list(range(250))


def test_pin_load_records_reads_each_step_once(tmp_path):
    """D7: with every input kept, today's `load_records` reads `part-*` only, so each step once."""
    from tensorquill import evidence

    folder, steps = _compacted_store(tmp_path, delete_inputs=False)
    assert [r["step"] for r in evidence.load_records(folder)] == steps


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_READER)
def test_target_load_records_after_operator_delete(tmp_path):
    from tensorquill import evidence

    folder, steps = _compacted_store(tmp_path, delete_inputs=True)
    assert [r["step"] for r in evidence.load_records(folder)] == steps
