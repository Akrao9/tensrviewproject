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


# ---- EV-02, EV-03, EV-04: the all-ranks views -----------------------------------------------

import time  # noqa: E402

import pytest  # noqa: E402

from tensorquill import evidence as ev  # noqa: E402
from tensorquill import mcp_server  # noqa: E402
from tensorquill.compare import open_run  # noqa: E402

MEASURED = ("gradient_trend", "attention_trends", "update_trends", "weight_jumps")
ACTIVATION = ("activation_spike", "activation_drift", "dead_activations")


def rank_records(rank: int, steps: int = 40, starting: bool = False) -> list[dict]:
    """Rank 1 alone says what it could not read and how it read its router, and measures what the
    four measurements read; rank 0 is plain. `starting`: a router collapsed since step 0, on a run
    too short for it to be reported (`starting_state`)."""
    out = []
    for step in range(steps):
        r = {"schema_version": 1, "run_id": "evrun", "rank": rank, "attempt": 0, "step": step,
             "grad_norm": 1.0 + 0.01 * step, "scalars": {"loss": 1.0 / (step + 1)},
             "layers": {"blocks.0": {"abs_max": 1.0 + 0.001 * step, "mean": 0.1, "std": 0.2, "zero_frac": 0.0}}}
        if rank == 1:
            r["probe"] = {"attention": {"attn.0": {"entropy_share": [0.5, 0.6], "max_logit": [3.0 + 0.05 * step, 2.0]}}}
            r["update_rms"] = {"w": 1e-3 * (1 + 0.01 * step)}
            if step % 5 == 0:
                r["weight_norms"] = {"w": 10.0 + step * 0.1}
            r["moe"] = {"r": [40, 0, 0, 0] if starting else [10, 10, 10, 10]}
            r["moe_top_k"] = {"r": 1}
            if step == 0:
                r["layers_not_read"] = {"blocks.1": "a fused kernel"}
                r["grads_not_read"] = {"emb.weight": "a sparse gradient"}
                r["moe_source"] = {"r": {"form": "scores", "point": "requested"}}
        out.append(r)
    return out


def write_ranks(folder: Path, starting: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for rank in (0, 1):
        rows = rank_records(rank, steps=5 if starting else 40, starting=starting)
        (folder / f"evrun.rank{rank}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def assert_kept_with_ranks(view: dict, *, starting: bool, own: dict, zero: dict, summary: bool = True) -> None:
    """What each rank's own evidence (`own` rank 1's, `zero` rank 0's) says reaches the all-ranks
    view, said to be that rank's. Rank 0 has a `gradient_trend` of its own."""
    checks = view["kind_checks"]
    for kind in ACTIVATION:
        assert checks[kind]["modules_not_judged"] == ["blocks.1"], kind
        assert checks[kind]["modules_not_judged_by_rank"] == {"1": ["blocks.1"]}, kind
    if starting:
        assert checks["moe_imbalance"]["starting_state"] == [{"router": "r", "samples": 5}]
        assert checks["moe_imbalance"]["starting_state_by_rank"] == {"1": [{"router": "r", "samples": 5}]}
        assert checks["moe_imbalance"]["enough_history"] is False
    assert view["moe_source_by_rank"] == [{"rank": 1, "moe_source": own["moe_source"]}]
    for name in MEASURED:
        wanted = [{"rank": n, name: e[name]} for n, e in ((0, zero), (1, own)) if name in e]
        assert view.get(f"{name}_by_rank") == (wanted or None), name
    assert "gradient_trend" not in own or {e["rank"] for e in view["gradient_trend_by_rank"]} == {0, 1}
    said = {(item["check"], tuple(ranks)) for item, ranks in ev.rank_not_checked(view)}
    assert said == {("gradients", (1,)), ("layers", (1,))}
    if summary:  # the command line's summary of the view (`--rank all`)
        text = ev.incident_summary(view)
        assert "Not checked on rank 1 (gradients)" in text and "Not checked on rank 1 (layers)" in text


@pytest.mark.parametrize("starting", [False, True], ids=["long", "starting_state"])
def test_the_all_ranks_view_of_local_files_keeps_each_ranks_keys(tmp_path, starting):
    """`combine_rank_evidence` (a run of local files, through `compare`): EV-02, EV-03, EV-04."""
    write_ranks(tmp_path, starting)
    own = ev.build_evidence(rank_records(1, steps=5 if starting else 40, starting=starting))
    zero = ev.build_evidence(rank_records(0, steps=5 if starting else 40, starting=starting))
    view = open_run("evrun", source=tmp_path)["evidence"]
    assert view["not_checked_by_rank"] == [{"rank": 1, "not_checked": own["not_checked"]}]
    # `compare` reads this view and has no summary line of `not_checked` for any run, one rank's either.
    assert_kept_with_ranks(view, starting=starting, own=own, zero=zero, summary=False)


def test_a_view_whose_ranks_have_none_of_the_keys_has_none_of_the_new_keys(tmp_path):
    for rank in (0, 1):
        (tmp_path / f"evrun.rank{rank}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rank_records(0, steps=5)).replace('"rank": 0', f'"rank": {rank}'))
    view = open_run("evrun", source=tmp_path)["evidence"]
    assert not [key for key in view if key.endswith("_by_rank")]
    assert not any(key.endswith("_by_rank") for check in view["kind_checks"].values() for key in check)
    assert "Not checked on" not in ev.incident_summary(view)


def test_the_run_entry_keeps_the_fewest_samples_and_each_ranks_list():
    starting = {"value": "checked", "enough_history": False}
    checks = {
        0: {kind: {"value": "checked", "enough_history": True} for kind in ev.SERVER_KINDS},
        1: {kind: {"value": "checked", "enough_history": True} for kind in ev.SERVER_KINDS},
    }
    checks[0]["moe_imbalance"] = {**starting, "starting_state": [{"router": "r", "samples": 7}]}
    checks[1]["moe_imbalance"] = {**starting, "starting_state": [{"router": "r", "samples": 4}, {"router": "s", "samples": 4}]}
    others = ev.other_kind_checks(status=None, status_expected=False, not_checked=[], flight_recorder=False,
                                  traincheck=False)
    run = ev.run_kind_checks(checks, others, memory={}, status=None)["moe_imbalance"]
    assert run["starting_state"] == [{"router": "r", "samples": 4}, {"router": "s", "samples": 4}]
    assert run["starting_state_by_rank"] == {"0": [{"router": "r", "samples": 7}],
                                             "1": [{"router": "r", "samples": 4}, {"router": "s", "samples": 4}]}
    assert run["enough_history"] is False
    checks[0]["moe_imbalance"] = dict(checks[1]["moe_imbalance"])
    same = ev.run_kind_checks(checks, others, memory={}, status=None)["moe_imbalance"]
    assert "starting_state_by_rank" not in same and same["starting_state"] == checks[1]["moe_imbalance"]["starting_state"]


def _store(tmp_path: Path, starting: bool) -> Path:
    from test_flight_e2e import BINARY, Server

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    from tensorquill.flight import send_file

    store = tmp_path / "store"
    server = Server(store, 0, "--flush-steps", "50", "--flush-seconds", "0.2")
    try:
        for rank in (0, 1):
            path = tmp_path / f"send.rank{rank}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rank_records(rank, 5 if starting else 40, starting)))
            assert send_file(path, server.endpoint, "evrun", rank, timeout=30.0)
        folder = store / "runs" / "evrun"
        deadline = time.monotonic() + 20
        while not all(ev.read_server_report(folder / f"rank={n}")["state"] == "current" for n in (0, 1)):
            assert time.monotonic() < deadline, "the server did not report on both ranks"
            time.sleep(0.05)
    finally:
        server.terminate()
    return folder


@pytest.mark.parametrize("starting", [False, True], ids=["long", "starting_state"])
def test_the_all_ranks_view_of_a_store_and_the_mcp_tools_keep_each_ranks_keys(tmp_path, starting):
    """`_all_ranks_evidence` (a store's reports), the command line's summary of it, and the MCP
    tools on the run folder: EV-02 (already on each rank's entry), EV-03, EV-04."""
    pytest.importorskip("pyarrow.flight")
    folder = _store(tmp_path, starting)
    report = json.loads((folder / "rank=1" / "incidents.json").read_text())
    zero = json.loads((folder / "rank=0" / "incidents.json").read_text())
    view = ev.all_ranks_evidence(folder)
    assert_kept_with_ranks(view, starting=starting, own=report, zero=zero)
    source = mcp_server.Source(folder)
    for answer in (mcp_server.list_incidents(source, {}), mcp_server.get_all_ranks_status(source, {})):
        assert {(i["check"], i["ranks"]) for i in answer["not_checked"]} == {("gradients", "1"), ("layers", "1")}
        assert answer["moe_source_by_rank"] == view["moe_source_by_rank"]
        for name in MEASURED:
            assert answer.get(f"{name}_by_rank") == view.get(f"{name}_by_rank"), name
        for kind in ACTIVATION:
            assert answer["kind_checks"][kind]["modules_not_judged_by_rank"] == {"1": ["blocks.1"]}
        if starting:
            assert answer["kind_checks"]["moe_imbalance"]["starting_state_by_rank"] == {"1": [{"router": "r", "samples": 5}]}
    # One rank asked for: that rank's entries, and none of a rank not asked for.
    zero = mcp_server.list_incidents(source, {"rank": 0})
    assert "moe_source_by_rank" not in zero
    assert mcp_server.list_incidents(source, {"rank": 1})["moe_source_by_rank"] == view["moe_source_by_rank"]
