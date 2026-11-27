"""Both detector engines on the same generated runs (`record_generator.py`), malformed values included.

Each case is written as the SDK writes a run, one `.jsonl` file per rank, and read by the routes real
use takes. Three comparisons, each on the fields the two engines are already held to agree on by the
hand-written shared cases:

- `sdk`: `build_evidence(file)` against `build_evidence(file, detectors="rust")`, as
  `test_server_detect.both` compares them: `incidents`, `not_checked`, `kind_checks` and `run_state`
  by `differences` (numbers to 1e-12, a NaN as null), the MoE `samples` exactly
  (`sample_differences`), and every key made in Python exactly.
- `raw`: the raw Rust report (`server_report`, what `incidents.json` holds) against the Python
  evidence of the same file: the incidents, the four measurement blocks with their absence
  (`MEASUREMENT_BLOCKS`, the author's A9 decision) and `moe_source` (A8).
- `run`: for a case of several ranks, `server_run_report` against Python's cross-rank readings:
  `duplicate_batches` (`detect_duplicate_batches` over `resolve_attempts`, as `test_run_detect.py`)
  and the `replica` object's `steps` and `replica_drift` (`compare_replicas`); `legacy` as a set of
  entries, its order not being one the engines are held to.

An outcome is a report or a refusal. Both engines refusing the same input is agreement, whatever the
two messages say. One refusing where the other reports is a disagreement.

A seed that disagrees is a finding: it is not fixed here. It is shrunk (`record_generator.shrink`),
its smallest input held below as a strict expected failure, and its seed listed in `KNOWN`, which
marks it in the sweep as an expected failure too. `KNOWN` holds for `GENERATOR_VERSION` alone.

The ordinary suite runs `SEEDS` seeds. `TENSORQUILL_GENERATED_SEEDS=<n>` runs seeds 0 to n-1, and
`TENSORQUILL_GENERATED_SEEDS=<a>-<b>` seeds a to b-1: a hunt, in which a seed not in `KNOWN` may fail.

Needs `pyarrow` and the server binary (`TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill import evidence  # noqa: E402
from tensorquill.data_checks import detect_duplicate_batches  # noqa: E402
from tensorquill.evidence import MEASUREMENT_BLOCKS, SERVER_KINDS, resolve_attempts  # noqa: E402
from tensorquill.server_detect import find_binary, server_report, server_run_report  # noqa: E402

import record_generator as gen  # noqa: E402
from test_server_detect import JUDGED, sample_differences  # noqa: E402
from test_server_detectors import differences  # noqa: E402

BINARY = find_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

SEEDS = 40
SEEDS_ENV = "TENSORQUILL_GENERATED_SEEDS"


def _seeds() -> list[int]:
    given = os.environ.get(SEEDS_ENV, "").strip()
    if not given:
        return list(range(SEEDS))
    if "-" in given:
        low, high = (int(part) for part in given.split("-", 1))
        return list(range(low, high))
    return list(range(int(given)))


# --- outcomes ------------------------------------------------------------------------------------


class Refused:
    """What an engine raised in place of a report."""

    def __init__(self, error: BaseException) -> None:
        self.kind = type(error).__name__
        self.message = str(error).splitlines()[0][:300] if str(error) else ""

    def __repr__(self) -> str:
        return f"refused ({self.kind}: {self.message})"


def outcome(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception as error:  # noqa: BLE001  a refusal is an outcome to compare
        return Refused(error)


def _refusals(python: Any, rust: Any, route: str) -> list[str] | None:
    """The difference when one side refused, [] when both did, None when neither did."""
    if isinstance(python, Refused) and isinstance(rust, Refused):
        return []
    if isinstance(python, Refused) or isinstance(rust, Refused):
        who = "python" if isinstance(python, Refused) else "rust"
        return [f"{route}: refused by {who} alone: python {python!r} != rust {rust!r}"]
    return None


# --- the three comparisons -----------------------------------------------------------------------


def _rest(document: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in document.items() if k not in (*JUDGED, "samples", "detectors")},
                      sort_keys=True, default=repr)


def sdk_differences(path: Path) -> list[str]:
    python = outcome(lambda: tensorquill.build_evidence(path))
    rust = outcome(lambda: tensorquill.build_evidence(path, detectors="rust", server_binary=BINARY))
    refused = _refusals(python, rust, "sdk")
    if refused is not None:
        return refused
    found = [f"sdk: {d}" for key in JUDGED for d in differences(python.get(key, []), rust.get(key, []), key)]
    found += [f"sdk: {d}" for d in sample_differences(python["samples"], rust["samples"])]
    if _rest(python) != _rest(rust):
        found.append("sdk: the keys made in Python differ")
    return found


def raw_differences(path: Path) -> list[str]:
    python = outcome(lambda: tensorquill.build_evidence(path))
    rust = outcome(lambda: server_report(tensorquill.load_records(path, all_attempts=True), binary=BINARY))
    refused = _refusals(python, rust, "raw")
    if refused is not None:
        return refused
    ours = [i for i in python["incidents"] if i.get("kind") in SERVER_KINDS]
    found = [f"raw: {d}" for d in differences(ours, rust["incidents"], "incidents")]
    for block in (*MEASUREMENT_BLOCKS, "moe_source"):
        found += [f"raw: {d}" for d in differences(python.get(block), rust.get(block), block)]
    return found


def _ranks_records(paths: dict[int, Path]) -> list[dict[str, Any]]:
    return [r for path in paths.values() for r in tensorquill.load_records(path, all_attempts=True)]


def _legacy(entries: Any) -> Any:
    return sorted(entries, key=lambda e: json.dumps(e, sort_keys=True)) if isinstance(entries, list) else entries


def run_differences(paths: dict[int, Path]) -> list[str]:
    records = outcome(lambda: _ranks_records(paths))
    if isinstance(records, Refused):
        return []  # the per-rank comparisons hold the refusal
    rust = outcome(lambda: server_run_report(records, binary=BINARY))

    def python_reading() -> dict[str, Any]:
        found = detect_duplicate_batches(resolve_attempts(list(records)))
        return {"duplicate_batches": None if found is None else {**found, "class": "invariant"},
                "replica": evidence.compare_replicas(records)}

    python = outcome(python_reading)
    refused = _refusals(python, rust, "run")
    if refused is not None:
        return refused
    found = []
    duplicate = [i for i in rust["incidents"] if i.get("kind") == "duplicate_batches"]
    found += [f"run: {d}" for d in differences(python["duplicate_batches"], duplicate[0] if duplicate else None,
                                               "duplicate_batches")]
    theirs = rust.get("replica") or {}
    ours = python["replica"]
    for key in ("steps", "replica_drift"):
        found += [f"run: {d}" for d in differences(ours.get(key, []), theirs.get(key, []), f"replica.{key}")]
    found += [f"run: {d}" for d in differences(_legacy(ours.get("legacy")), _legacy(theirs.get("legacy")),
                                               "replica.legacy")]
    return found


def case_differences(case: gen.Case, folder: Path) -> list[str]:
    paths = gen.write(case, folder)
    found: list[str] = []
    for rank, path in paths.items():
        found += [f"rank {rank} {d}" for d in sdk_differences(path) + raw_differences(path)]
    if len(paths) > 1:
        found += run_differences(paths)
    return found


def records_differences(records: list[dict[str, Any]], folder: Path) -> list[str]:
    """The differences for one rank's records as a file, `records` written as the generator writes
    them: what a shrunk input is held to."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "held.rank0.jsonl"
    path.write_text("".join(gen.line(r) for r in records), encoding="utf-8")
    return sdk_differences(path) + raw_differences(path)


def signature(found: list[str]) -> str:
    """What a disagreement is, for keeping the same one while shrinking: its route and the path of
    the first field that differs, without the values."""
    if not found:
        return ""
    first = found[0].split(" ", 2)[2] if found[0].startswith("rank ") else found[0]
    route, rest = first.split(": ", 1)
    if rest.startswith("refused by "):
        return f"{route}:{rest.split(':', 1)[0]}"
    # The pattern is taken out of the f-string: a backslash inside one is a syntax error before
    # Python 3.12, and the package supports 3.10.
    path = re.sub(r"\[[^\]]*\]", "", rest.split(":", 1)[0]).strip()
    return f"{route}:{path}"


# --- the generator itself ------------------------------------------------------------------------


def _digest(seed: int) -> str:
    return hashlib.sha256(json.dumps(gen.generate(seed).files(), sort_keys=True).encode()).hexdigest()[:16]


def test_a_seed_gives_the_same_lines_every_time():
    assert gen.generate(7).files() == gen.generate(7).files()
    assert gen.generate(7).files() != gen.generate(8).files()


# Pinned: a change to how cases are drawn fails here, and `KNOWN` has to be made again with
# `GENERATOR_VERSION` raised.
PINNED = {0: "9db38a59c7941a80", 1: "0742af055087b23d", 2: "a4b7f9b6dd47051d"}


def test_the_generator_still_draws_what_the_known_seeds_were_found_with():
    assert gen.GENERATOR_VERSION == 1
    assert {seed: _digest(seed) for seed in PINNED} == PINNED


def test_every_seed_of_a_wide_range_makes_a_case_whose_lines_parse():
    # Two malformations can meet on one record (an identity field made text, then a step compared): a
    # case must still be made. Seed 1818 raised here before `_placed`.
    for seed in range(1600, 2100):
        case = gen.generate(seed)
        assert len(gen.read("".join(case.files().values()))) == len(case.records), seed


def test_the_lines_are_what_a_reader_parses_and_the_sink_spelling_is_used():
    case = gen.generate(0, base="non_finite", malformations=["non_finite_foreign"])
    text = "".join(case.files().values())
    records = gen.read(text)
    assert len(records) == len(case.records)
    assert '"nan"' in text or '"inf"' in text or '"-inf"' in text  # the planted fault, as the sink writes it
    assert any(token in text for token in ("NaN", "Infinity"))  # the malformation, as another writer does


def test_the_default_seeds_reach_most_kinds_and_are_mostly_judged(tmp_path):
    reached: set[str] = set()
    judged = 0
    for seed in range(SEEDS):
        case = gen.generate(seed)
        for path in gen.write(case, tmp_path / str(seed)).values():
            found = outcome(lambda: tensorquill.build_evidence(path))
            if isinstance(found, Refused):
                continue
            judged += 1
            reached |= {i["kind"] for i in found["incidents"]}
    files = sum(len(gen.generate(seed).ranks) for seed in range(SEEDS))
    assert judged >= 0.8 * files, (judged, files)
    assert len(reached & set(SERVER_KINDS)) >= 12, sorted(reached)


def test_shrinking_keeps_what_still_holds_and_drops_the_rest():
    case = gen.generate(3, base="healthy", malformations=[])
    records = gen.read("".join(case.files().values()))
    records[17]["grad_norm"] = "bad"

    def still(candidate):
        return any(r.get("grad_norm") == "bad" for r in candidate)

    smallest = gen.shrink(records, still)
    assert len(smallest) == 1 and smallest[0]["grad_norm"] == "bad"
    assert set(smallest[0]) == {"schema_version", "run_id", "rank", "attempt", "step", "grad_norm"}


def test_a_corpus_is_written_with_its_index(tmp_path):
    index = json.loads(gen.write_corpus(tmp_path, range(3)).read_text())
    assert index["generator_version"] == gen.GENERATOR_VERSION
    assert [c["seed"] for c in index["cases"]] == [0, 1, 2]
    for entry in index["cases"]:
        for name in entry["files"].values():
            assert (tmp_path / name).is_file()


# --- the sweep -----------------------------------------------------------------------------------

# Seeds of the default range that disagree, by the finding of `HELD` that holds their smallest input.
KNOWN: dict[int, str] = {
    13: "AQ5-D5: a router count above 2**53 rounds the MoE sample's shares differently",
}


def _sweep_params():
    return [pytest.param(seed, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=KNOWN[seed]))
            if seed in KNOWN else seed for seed in _seeds()]


@pytest.mark.parametrize("seed", _sweep_params())
def test_both_engines_agree_on_a_generated_run(seed, tmp_path):
    case = gen.generate(seed)
    found = case_differences(case, tmp_path)
    assert found == [], f"{case.describe()}\n" + "\n".join(found[:20])


# --- the disagreements found, each at its smallest ------------------------------------------------

def _record(step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "aq5", "rank": 0, "attempt": 1, "step": step, **fields}


def _stall(grad_norm: Any) -> list[dict[str, Any]]:
    middle = {"grad_norm": grad_norm}
    return [_record(35, grad_norms={"p0.weight": 0.49}, weights_unchanged=["p0.weight"]),
            _record(36, **middle, grad_norms={"p0.weight": 0.48}, weights_unchanged=["p0.weight"]),
            _record(37, grad_norms={"p0.weight": 0.51}, weights_unchanged=["p0.weight"])]


def _written_twice() -> list[dict[str, Any]]:
    starved = {"r0": [256, 128, 127, 1]}
    last = _record(48, moe=starved, scalars={"loss": 2.0})
    return [_record(step, moe=starved) for step in range(48)] + [last, dict(last)]


# Each disagreement found, at the smallest input shrinking reached (or a shorter one made from it by
# hand), with the seed it was found at. Each must still disagree: a strict expected failure.
HELD: dict[str, tuple[str, list[dict[str, Any]]]] = {
    # Seeds 93, 340, 731, 750, 1490.
    "AQ5-D2 system with an oversized integer": (
        "Python reads the system block (memory_growth, memory_pressure, allocator_retries checked); Rust "
        "loses the whole system_json of the row to one integer too large for a float",
        [_record(39, system={"cuda": {"ooms": 10**400}})],
    ),
    # Seeds 366, 729, 1352, 1413.
    "AQ5-D2 data with an oversized integer": (
        "the same for data_json: data_stalled, epoch_order_repeated and resume_data_mismatch",
        [_record(37, data={"sampler_epoch": 10**400})],
    ),
    # Seeds 482, 760, 1424; at 1371 Rust misses an attention_collapse incident Python reports.
    "AQ5-D2 probe with an oversized integer": (
        "the same for probe_json: the three attention kinds",
        [_record(0, probe={"attention": {"blk1": {"max_logit": [3.943363681, -(10**400), 3.956723686]}}})],
    ),
    "AQ5-D2 checkpoint with an oversized integer": (
        "the same for checkpoint_json: resume_incomplete (found by hand from D2, not by a seed)",
        [_record(0, checkpoint={"save": {"after_step": 10**400}})],
    ),
    # Seed 1863, two records.
    "AQ5-D4 a MoE incident at a step written twice": (
        "Rust's scalar context of the moe_imbalance incident takes the step's first writing as a value "
        "before it (median_before 2.4); Python's has none before it (None), and so has repeated_steps' "
        "in both engines",
        [_record(26, scalars={"loss": 2.4}), _record(26, scalars={"loss": 2.4}, moe={"r0": [1, 1, 10_000, 1]})],
    ),
    # Seed 41, starved_experts: the same with 50 records.
    "AQ5-D4 a MoE incident at a step written twice, starved_experts": (
        "as above, for the starved_experts incident (median_before 2.0 in Rust, None in Python)",
        _written_twice(),
    ),
    # Seeds 13, 210, 357, 376, 413, 542, 563, 592, 653, 674, 753, 1032, 1281, 1379.
    "AQ5-D5 a router count above 2**53": (
        "the MoE sample's diversity and starved_shares differ in the last digits: Python divides the "
        "integers exactly, Rust divides their floats. Fifty records, so that starved_shares, a division "
        "alone, differs as well as diversity, which also goes through log and exp, whose last bits may "
        "differ by platform",
        [_record(step, moe={"r1": [68, 64, 70, 66, 61, 65, 64, 62]}) for step in range(49)]
        + [_record(49, moe={"r1": [68, 64, 2**53 + 1, 66, 61, 65, 64, 62]})],
    ),
    # Seed 4082, two records.
    "AQ5-D7 a scalars entry the column refuses, in an incident's context": (
        "Python's scalar context of the repeated_steps incident shows the refused value (-10**400, or a "
        "text); Rust's shows no scalars, the whole map being in extra_json (lane W3's rule, WAVE_AJ_CONTRACT.md "
        "section 3, listed there as open)",
        [_record(115, scalars={"train/loss": -(10**400)}), _record(115)],
    ),
    "AQ5-D7 a scalars entry the column refuses, a text": (
        "as above, for a text value",
        [_record(115, scalars={"train/loss": "text"}), _record(115)],
    ),
}


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=AssertionError,
                                                                             reason=HELD[name][0]))
                                  for name in sorted(HELD)])
def test_a_held_disagreement_still_disagrees(name, tmp_path):
    reason, records = HELD[name]
    found = records_differences(records, tmp_path)
    assert found == [], f"{reason}:\n" + "\n".join(found[:10])


@pytest.mark.parametrize("seed", sorted(KNOWN))
def test_a_known_seed_still_disagrees_as_its_finding_says(seed, tmp_path):
    # The sweep marks these seeds as expected failures; here, what they give is held to the finding named.
    found = case_differences(gen.generate(seed), tmp_path)
    finding = KNOWN[seed].split(":", 1)[0]
    expected = {"AQ5-D5": "sample"}[finding]
    assert found and all(expected in line for line in found), found[:5]


# --- the disagreements settled since, each held at its smallest input as an agreement ------------

# Lane AR-2 of wave AR: AQ5-D1, AQ5-D3 and AQ5-D6, settled in the Python engine. Each input is the one
# `HELD` held as a disagreement, with the seeds it was found at.
SETTLED_AR2: dict[str, tuple[str, list[dict[str, Any]]]] = {
    # Seeds 23, 294, 1096, 1368 (a string that is not the sink's spelling, or a bool) and 266 (10**400).
    "AQ5-D1 update_rms with no held entry, a string": (
        "a refused update_rms map is an absent key in both engines (the author: Python brought to Rust)",
        [_record(47, grad_norm=1.048992069, update_rms={"p0.weight": "nan "})],
    ),
    "AQ5-D1 update_rms with no held entry, an oversized integer": (
        "as above, for 10**400",
        [_record(112, grad_norm=1.036979024, update_rms={"p0.weight": 10**400})],
    ),
    # Seed 69.
    "AQ5-D3 a grad_norm of the wrong type, a text": (
        "a top-level grad_norm that is not a number is absent in Python too: both report the stall",
        _stall("NAN"),
    ),
    "AQ5-D3 a grad_norm of the wrong type, a bool": (
        "as above, for true",
        _stall(True),
    ),
    # Seed 572.
    "AQ5-D6 a step above int64": (
        "identity.check is bounded at Int64: both engines refuse the record by name (MalformedRecordError)",
        [_record(9223372036854775808, grad_norm=1.0)],
    ),
    "AQ5-D6 a rank above int32": (
        "identity.check is bounded at Int32 for rank: both engines refuse the record by name",
        [_record(3, rank=2**31, grad_norm=1.0)],
    ),
}


@pytest.mark.parametrize("name", sorted(SETTLED_AR2))
def test_a_disagreement_settled_by_lane_ar2_now_agrees(name, tmp_path):
    reason, records = SETTLED_AR2[name]
    found = records_differences(records, tmp_path)
    assert found == [], f"{reason}:\n" + "\n".join(found[:10])
