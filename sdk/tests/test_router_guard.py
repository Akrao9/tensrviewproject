"""A router whose output's first integer tensor cannot be expert indices is not counted.

By default the capture takes the first integer tensor in a router's output as the chosen expert
indices. nanoMoE's `Router` returns `(used_capacity, weights, mask)`, and `used_capacity` is an
integer tensor of per-expert token counts: read as indices it gives wrong loads and no error.
The shapes show it (eight numbers for hundreds of tokens), so the hook leaves that router
uncounted and the session says so once.

A refused router leaves no entry in a record: not in `moe`, `moe_top_k` or `moe_stats`. All-zero
counts would read as a router that routed nothing. The first record written after the refusal
says why under `moe_refused`, and the evidence turns that into a `not_checked` entry, so the
MoE kinds read `unavailable`: no MoE incident is then not "MoE checked".
"""

from __future__ import annotations

import json
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.capture import Router
from tensorquill.evidence import MOE_KINDS, input_counts

EXPERTS, TOP_K = 8, 2


class CountsFirstRouter(nn.Module):
    """As nanoMoE's: per-expert counts, then weights, then a mask. No indices in the output."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w_g = nn.Linear(width, EXPERTS, bias=False)

    def forward(self, x):
        logits = self.w_g(x)
        chosen = logits.topk(TOP_K, dim=-1).indices
        mask = torch.nn.functional.one_hot(chosen, EXPERTS).sum(-2)  # [B, T, EXPERTS]
        used_capacity = mask.sum(dim=(0, 1))  # [EXPERTS], integer counts
        return used_capacity, logits.softmax(-1) * mask, mask.bool()


class IndexRouter(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.w_g = nn.Linear(width, EXPERTS, bias=False)

    def forward(self, x):
        return self.w_g(x).topk(TOP_K, dim=-1).indices  # [B, T, TOP_K]


class Model(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 16)
        self.router = router
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        hidden = self.embed(x)
        self.router(hidden)
        return self.head(hidden).pow(2).mean()


def _run(tmp_path, router, routers, steps=4, compile_backend=None):
    torch.manual_seed(0)
    model = Model(router)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "guard.jsonl"
    session = tensorquill.init(model, run_id="guard", sink=tensorquill.JsonlSink(path), routers=routers, every=steps)
    stepped = torch.compile(model, backend=compile_backend) if compile_backend else model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        stepped(torch.randn(6, 20, 4)).backward()  # 120 tokens
        optimizer.step()
    session.close()
    return [json.loads(line) for line in path.read_text().splitlines()]


def _routed(records, name="router"):
    return [r["moe_stats"][name]["routed_tokens"] for r in records if name in r.get("moe_stats", {})]


def _no_entry(records, name="router"):
    """The router is in no record's `moe`, `moe_top_k` or `moe_stats`."""
    return not any(name in r.get(key, {}) for r in records for key in ("moe", "moe_top_k", "moe_stats"))


def _moe_incidents(evidence):
    return [i for i in evidence["incidents"] if i["kind"] in MOE_KINDS]


def test_a_router_that_returns_counts_first_is_not_counted_and_is_said_once(tmp_path):
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'") as caught:
        records = _run(tmp_path, CountsFirstRouter(16), {"router": Router(EXPERTS, TOP_K)})
    said = [w for w in caught if "not counting the router" in str(w.message)]
    assert len(said) == 1  # once, not at every step
    assert "shape (8,)" in str(said[0].message) and "120 tokens" in str(said[0].message)
    # Nothing was counted, not even zeros: the only router leaves no MoE keys at all.
    assert len(records) == 4 and _no_entry(records)
    assert not any(key in r for r in records for key in ("moe", "moe_top_k", "moe_stats"))
    # Said once in the records too, on the first record written after the refusal.
    assert [r["step"] for r in records if "moe_refused" in r] == [0]
    assert "shape (8,)" in records[0]["moe_refused"]["router"]
    evidence = tensorquill.build_evidence(records)
    assert not _moe_incidents(evidence) and evidence["samples"] == []
    assert input_counts(records)["moe"] == 0
    # Not "checked" (nothing found) and not "not_configured" (nothing asked): unavailable, with why.
    [entry] = [e for e in evidence["not_checked"] if e["check"] == "router"]
    assert entry["routers"] == ["router"] and entry["kinds"] == list(MOE_KINDS) and entry["step"] == 0
    for kind in MOE_KINDS:
        check = evidence["kind_checks"][kind]
        assert check["value"] == "unavailable" and check["check"] == "router"
        assert "router router is not counted" in check["reason"]


class TwoRouters(nn.Module):
    """A refused router beside one that is counted."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 16)
        self.refused = CountsFirstRouter(16)
        self.counted = IndexRouter(16)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        hidden = self.embed(x)
        self.refused(hidden)
        self.counted(hidden)
        return self.head(hidden).pow(2).mean()


def _two_routers(tmp_path, steps=8, every=4, routers=None, endpoint=None):
    torch.manual_seed(0)
    model = TwoRouters()
    routers = routers or {"refused": Router(EXPERTS, TOP_K), "counted": Router(EXPERTS, TOP_K)}
    if endpoint is None:
        path = tmp_path / "two.jsonl"
        out = {"sink": tensorquill.JsonlSink(path)}
    else:  # streamed: the local file is written where the run is, in tensorquill-out/
        path = tmp_path / "tensorquill-out" / "two.rank0.jsonl"
        out = {"endpoint": endpoint}
    with pytest.warns(RuntimeWarning, match="not counting the router 'refused'") as caught:
        session = tensorquill.init(model, run_id="two", routers=routers, every=every, **out)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(steps):
            optimizer.zero_grad()
            model(torch.randn(6, 20, 4)).backward()
            optimizer.step()
        session.close()
    assert len([w for w in caught if "not counting the router" in str(w.message)]) == 1
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_refused_router_beside_a_counted_one_leaves_no_entry_and_no_history(tmp_path):
    records = _two_routers(tmp_path)
    assert len(records) == 8 and _no_entry(records, "refused")
    # The counted router is counted as before, on every step.
    assert _routed(records, "counted") == [120 * TOP_K] * 8
    assert all(list(r["moe"]) == ["counted"] and r["moe_top_k"] == {"counted": TOP_K} for r in records)
    # The refusal is said once, though two blocks were written after it.
    assert [r["step"] for r in records if "moe_refused" in r] == [0]
    assert list(records[0]["moe_refused"]) == ["refused"]

    evidence = tensorquill.build_evidence(records)
    assert not _moe_incidents(evidence)
    # MoE history is the counted router's alone: the refused one adds no record and no sample.
    without = [{k: v for k, v in r.items() if k != "moe_refused"} for r in records]
    assert input_counts(records)["moe"] == input_counts(without)["moe"] == 8
    assert evidence["router_names"] == ["counted"]
    assert {s["router"] for s in evidence["samples"]} == {"counted"} and len(evidence["samples"]) == 8
    assert all(s["layer"] == 1 and not s.get("missing") for s in evidence["samples"])
    # One router was not counted, so the MoE kinds are not `checked`, however many records the other had.
    for kind in MOE_KINDS:
        check = evidence["kind_checks"][kind]
        assert check["value"] == "unavailable" and check["records"] == 8
        assert check["reason"].startswith("router refused is not counted: its output's first integer tensor")
    # The same records without the refusal note read as an MoE run that was checked.
    assert tensorquill.build_evidence(without)["kind_checks"]["moe_imbalance"]["value"] == "checked"


def test_the_same_router_is_counted_through_its_inner_logits_or_an_indices_function(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        inner = _run(tmp_path / "a", CountsFirstRouter(16), {"router.w_g": Router(EXPERTS, TOP_K)})
        given = _run(tmp_path / "b", CountsFirstRouter(16),
                     {"router": Router(EXPERTS, TOP_K, indices=lambda out: out[2].nonzero()[:, -1])})
    assert set(_routed(inner, "router.w_g")) == {120 * TOP_K}
    assert set(_routed(given)) == {120 * TOP_K}


def test_a_router_that_returns_indices_is_counted_as_before(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        records = _run(tmp_path, IndexRouter(16), {"router": Router(EXPERTS, TOP_K)})
    assert set(_routed(records)) == {120 * TOP_K}


def test_under_torch_compile_the_guard_says_the_same_and_counts_nothing(tmp_path):
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'"):
        records = _run(tmp_path, CountsFirstRouter(16), {"router": Router(EXPERTS, TOP_K)}, compile_backend="eager")
    assert len(records) == 4 and _no_entry(records)
    assert [r["step"] for r in records if "moe_refused" in r] == [0]


def test_a_model_with_no_refused_router_writes_no_refusal(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        records = _run(tmp_path, IndexRouter(16), {"router": Router(EXPERTS, TOP_K)})
    assert not any("moe_refused" in r for r in records)
    evidence = tensorquill.build_evidence(records)
    assert "not_checked" not in evidence and evidence["kind_checks"]["moe_imbalance"]["value"] == "checked"


# --- Through the Rust server -------------------------------------------------------------------


def _binary():
    from test_flight_e2e import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    pytest.importorskip("pyarrow")
    return BINARY


@pytest.mark.parametrize("both", [False, True], ids=["one-refused", "all-refused"])
def test_the_server_stores_a_refused_router_as_no_entry_and_says_why(tmp_path, monkeypatch, both):
    """Streamed over Flight: records with a router missing from `moe`, or with no `moe` at all,
    are accepted and stored as written. The server sees no zeros, raises no MoE incident, and
    its report says the MoE kinds were not checked, as the Python evidence does."""
    _binary()
    from test_flight_e2e import Server, stored_records

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    server = Server(tmp_path / "store")
    try:
        routers = {"refused": Router(EXPERTS, TOP_K)} if both else None
        records = _two_routers(tmp_path, endpoint=server.endpoint, routers=routers)
    finally:
        assert server.terminate() == 0
    stored = stored_records(tmp_path / "store", "two")
    assert stored == records and len(stored) == 8
    assert _no_entry(stored, "refused") and [r["step"] for r in stored if "moe_refused" in r] == [0]
    if both:
        assert not any("moe" in r for r in stored)
    else:
        assert _routed(stored, "counted") == [120 * TOP_K] * 8
    python = tensorquill.build_evidence(stored)
    rust = tensorquill.build_evidence(stored, detectors="rust")
    assert rust["detectors"] == "rust"
    for evidence in (python, rust):
        assert not _moe_incidents(evidence)
        assert {s["router"] for s in evidence["samples"]} == (set() if both else {"counted"})
        assert all(evidence["kind_checks"][kind]["value"] == "unavailable" for kind in MOE_KINDS)
    assert [e for e in rust["not_checked"] if e["check"] == "router"] == [
        e for e in python["not_checked"] if e["check"] == "router"
    ]
    assert rust["kind_checks"] == python["kind_checks"] and rust["samples"] == python["samples"]
