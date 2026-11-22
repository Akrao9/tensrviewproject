"""Lane C of wave AF: the same observation in every view (WAVE_AF_CONTRACT.md, section 2).

A key that says what was not observed, or how an observation was made, reaches every view that
shows the check it qualifies, and in the all-ranks view keeps the ranks it came from.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tensorquill.divergence import divergence_summary, first_divergence

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases.json").read_text())["cases"]


def _steps(count: int, *, run_id: str = "a", **said) -> list[dict]:
    return [{"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 0, "step": step,
             "scalars": {"loss": 1.0 / (step + 1)}, "grad_norm": 1.0 + 0.01 * step, **said} for step in range(count)]


# ---- EV-01: `first_divergence` ------------------------------------------------------------


def test_divergence_provenance_keys_are_not_series():
    case = CASES["divergence_provenance_keys_are_not_series"]
    result = first_divergence(case["given"]["records_a"], case["given"]["records_b"])
    assert result["left_out"]["not_compared"] == case["expected"]["not_compared_has"]
    assert not any(name.startswith(("layers_not_read", "moe_source")) for name in result["series"])
    # The two runs say the same of both: nothing differs, so no key of its own.
    assert "provenance_differs" not in result


def test_where_the_runs_differ_in_how_they_observed_the_answer_says_so():
    a = _steps(40, layers_not_read={"blocks.1": "a fused kernel"}, moe_source={"r": {"form": "scores", "point": "requested"}})
    b = _steps(40, run_id="b", moe_source={"r": {"form": "indices", "point": "dispatched"}})
    result = first_divergence(a, b)
    assert result["left_out"]["not_compared"] == []
    assert result["provenance_differs"] == {
        "layers_not_read": {"a": {"blocks.1": "a fused kernel"}, "b": None},
        "moe_source": {"a": {"r": {"form": "scores", "point": "requested"}},
                       "b": {"r": {"form": "indices", "point": "dispatched"}}},
    }
    assert "The runs differ in which modules the capture could not read and in how the routers' counts were read" \
        in divergence_summary(result)
    json.dumps(result, allow_nan=False)


def test_a_key_said_once_on_the_first_record_is_read_as_said_for_the_run():
    """The capture says each once, on the first record after it knew: a run that says it on step 0
    and one that repeats it on every record observed the same way."""
    once = _steps(40)
    once[0]["layers_not_read"] = {"blocks.1": "why"}
    every = _steps(40, run_id="b", layers_not_read={"blocks.1": "why"})
    assert "provenance_differs" not in first_divergence(once, every)
    other = copy.deepcopy(every)
    other[5]["layers_not_read"] = {"blocks.2": "another"}
    assert first_divergence(once, other)["provenance_differs"] == {
        "layers_not_read": {"a": {"blocks.1": "why"}, "b": {"blocks.1": "why", "blocks.2": "another"}}}


def test_compare_passes_the_difference_on(tmp_path):
    from tensorquill.compare import compare_runs

    for name, records in (("a", _steps(40, moe_source={"r": {"form": "scores", "point": "requested"}})),
                          ("b", _steps(40, run_id="b", moe_source={"r": {"form": "map", "point": "unknown"}}))):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    result = compare_runs(tmp_path / "a.jsonl", tmp_path / "b.jsonl", source=tmp_path)
    part = result["divergence"]
    assert part["left_out"]["not_compared"] == []
    assert set(part["provenance_differs"]) == {"moe_source"}
