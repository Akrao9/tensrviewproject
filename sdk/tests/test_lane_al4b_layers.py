"""Lane AL-4′ (wave AL, step 2): what the Python engine reads of `layers` and `moe_stats`.

`WAVE_AL_CONTRACT.md` section 1.5, as both engines read it:

- An entry of `layers` that is not an object, and a `layers` value that is not an object, are
  absent to every detector: read, not raised on (cases `layers_entry_not_object`,
  `layers_value_not_object`).
- Row S5, the owner's approved change of detector behaviour (Nov 24): a row whose only `layers`
  entries are not objects holds no layer, as Rust reads it (case `layers_only_entries_not_object`).
  That is the input count's side; the `layers_not_read` coverage's side is tested here on its own,
  since no case record says `layers_not_read`.
- A router's `diversity` that is not a finite number (a bool, a string, `null`, a non-finite
  value, an entry that is not an object) is taken from the counts, as Rust's `Number` reads it, so
  the per-step samples match `detect --samples` (case `moe_stats_diversity_refused`).

The samples comparison needs the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the
interpreter) and pyarrow, and skips without them; everything else needs only the SDK.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorquill import evidence

HERE = Path(__file__).parent
DOCUMENT = json.loads((HERE / "fixtures" / "wave_al" / "cases.json").read_text())
CASES = {c["name"]: c for c in DOCUMENT["cases"]}
LAYER_CASES = ["layers_entry_not_object", "layers_value_not_object", "layers_only_entries_not_object"]


def _brief(incident: dict) -> dict:
    out = {"kind": incident["kind"], "step": incident.get("step")}
    for key in ("layers", "modules", "gradients", "routers", "router", "parameters", "experts"):
        if key in incident:
            value = incident[key]
            out[key] = sorted(value) if isinstance(value, dict) else value
    return out


@pytest.mark.parametrize("name", LAYER_CASES)
def test_the_python_engine_reports_what_the_case_expects(name):
    case = CASES[name]
    want = case["expect"]["python"]
    built = evidence.build_evidence(copy.deepcopy(case["records"]), detectors="python")
    assert [_brief(i) for i in built["incidents"]] == want["incidents"]
    assert {k: (built["kind_checks"].get(k) or {}).get("value") for k in want["kind_checks"]} == want["kind_checks"]
    assert built["run_state"]["state"] == want["run_state"]


@pytest.mark.parametrize("name", ["layers_entry_not_object", "layers_value_not_object"])
def test_a_layers_entry_or_value_that_is_not_an_object_reads_as_the_clean_records(name):
    """Section 1.5: absent, so the evidence is the clean records' (`clean_records`), whole."""
    case = CASES[name]
    built = evidence.build_evidence(copy.deepcopy(case["records"]), detectors="python")
    clean = evidence.build_evidence(copy.deepcopy(case["clean_records"]), detectors="python")
    assert built["incidents"] == clean["incidents"]
    assert built["kind_checks"] == clean["kind_checks"]


def test_a_row_whose_only_entries_are_not_objects_counts_no_layer():
    """Row S5, the input counts (`evidence.py` near line 3354): `layers` and `numbers` count only a
    row that holds an entry that is an object."""
    rows = [{"step": 0, "layers": {"a": 5, "b": [1]}},
            {"step": 1, "layers": {"a": 5, "b": {"mean": 0.0, "std": 1.0, "abs_max": 3.0, "zero_frac": 0.0}}},
            {"step": 2, "layers": [1, 2]},
            {"step": 3, "layers": {}}]
    counts = evidence.input_counts(rows)
    assert (counts["steps"], counts["layers"], counts["numbers"]) == (4, 1, 1)


def test_a_row_whose_only_entries_are_not_objects_is_no_layer_read_for_the_coverage():
    """Row S5, the `layers_not_read` coverage (`evidence.py` near line 1013): a capture that read no
    layer as an object has coverage "none", and the activation kinds are listed."""
    said = {"model.layers.9": "a module the capture could not read"}
    only = [{"step": s, "layers": {"model.layers.0": 5}, "layers_not_read": said} for s in range(3)]
    found = evidence.layers_not_read_not_checked(only)
    assert found["coverage"] == "none" and found["kinds"] == list(evidence.LAYERS_NOT_READ_KINDS)
    some = [*only, {"step": 3, "layers": {"model.layers.0": {"mean": 0.0, "std": 1.0, "abs_max": 3.0, "zero_frac": 0.0}}}]
    found = evidence.layers_not_read_not_checked(some)
    assert found["coverage"] == "partial" and "kinds" not in found


def test_non_finite_reads_the_object_entries_beside_one_that_is_not():
    records = copy.deepcopy(CASES["layers_entry_not_object"]["records"])
    records[4]["layers"]["model.layers.3"]["abs_max"] = "nan"
    built = evidence.build_evidence(records, detectors="python")
    [hit] = [i for i in built["incidents"] if i["kind"] == "non_finite"]
    assert (hit["step"], hit["layers"]) == (4, ["model.layers.3"])


def _router_records(diversity_of: dict[int, object], moe_stats_of: dict[int, object] | None = None) -> list[dict]:
    """A top-1 router of four experts over six steps, uneven from step 3, its `diversity` set per step."""
    router = "model.layers.0.block_sparse_moe.router"
    out = []
    for step in range(6):
        counts = [10, 10, 10, 10] if step < 3 else [25, 5, 5, 5]
        stats = {"routed_tokens": 40, "entropy": 1.0, "dead_experts": 0, "dead_frac": 0.0}
        if step in diversity_of:
            stats["diversity"] = diversity_of[step]
        moe_stats = (moe_stats_of or {}).get(step, {router: stats})
        out.append({"schema_version": 1, "run_id": "div", "rank": 0, "attempt": 1, "step": step, "ts": float(step),
                    "moe": {router: counts}, "moe_top_k": {router: 1}, "moe_stats": moe_stats, "grad_norm": 1.0})
    return out


def _from_counts(counts: list[int]) -> float:
    shares = [c / sum(counts) for c in counts if c > 0]
    return math.exp(-math.fsum(s * math.log(s) for s in shares)) / len(counts)


@pytest.mark.parametrize("value", [True, False, "nan", "inf", "0.5", None, float("nan"), float("inf"), [0.5], {"v": 0.5}])
def test_a_diversity_that_is_not_a_finite_number_is_taken_from_the_counts(value):
    records = _router_records({3: value, 4: 0.25})
    samples = evidence.build_evidence(records, detectors="python")["samples"]
    by_step = {s["step"]: s["diversity"] for s in samples}
    assert by_step[3] == pytest.approx(_from_counts([25, 5, 5, 5]))
    assert by_step[4] == 0.25  # a finite number recorded is taken as it is
    assert by_step[5] == pytest.approx(_from_counts([25, 5, 5, 5]))  # none recorded


def test_a_moe_stats_entry_or_value_that_is_not_an_object_is_taken_from_the_counts():
    router = "model.layers.0.block_sparse_moe.router"
    records = _router_records({}, {3: {router: 5}, 4: [1, 2], 5: "text"})
    samples = evidence.build_evidence(records, detectors="python")["samples"]
    assert [s["diversity"] for s in samples if s["step"] >= 3] == pytest.approx([_from_counts([25, 5, 5, 5])] * 3)


def _binary():
    if importlib.util.find_spec("pyarrow") is None:
        return None
    from tensorquill.server_detect import find_binary

    return find_binary()


@pytest.mark.parametrize("name", ["moe_stats_diversity_refused", "moe_stats_real_fields", "moe_stats_float_in_int_field"])
def test_the_per_step_samples_match_detect_samples(name):
    """Case `moe_stats_diversity_refused` (section 1.5): `build_evidence`'s samples against the
    binary's `detect --samples`, through the SDK's Rust path. Two cases that already agree are
    pinned beside it."""
    binary = _binary()
    if binary is None:
        pytest.skip("tensorquill-server is not built (TENSORQUILL_SERVER_BIN), or pyarrow is not installed")
    records = CASES[name]["records"]
    python = evidence.build_evidence(copy.deepcopy(records), detectors="python")["samples"]
    rust = evidence.build_evidence(copy.deepcopy(records), detectors="rust", server_binary=binary)["samples"]
    assert len(python) == len(rust)
    for ours, theirs in zip(python, rust):
        assert {k: v for k, v in ours.items() if k != "diversity"} == {k: v for k, v in theirs.items() if k != "diversity"}
        assert ours["diversity"] == pytest.approx(theirs["diversity"], rel=1e-12, abs=0.0), ours["step"]


def test_the_refused_diversity_case_is_one_where_the_two_readings_differ():
    """The comparison above would pass for a wrong reason if `float(true)` gave the counts' value:
    it does not at the step the incident is reported."""
    records = CASES["moe_stats_diversity_refused"]["records"]
    router = "model.layers.0.block_sparse_moe.router"
    at = next(r for r in records if r["step"] == 14)
    assert at["moe_stats"][router]["diversity"] is True
    assert _from_counts(at["moe"][router]) != pytest.approx(1.0)
