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
from tensorquill.evidence import MOE_KINDS, incident_summary, input_counts

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


# --- A router given an `indices=` function ----------------------------------------------------

DEFAULT_FIX = (
    "To count it, pass routers={'router': tensorquill.Router(8, 2, indices=...)} with a function that returns "
    "the chosen expert indices from its output, or name the module inside it whose output is the router's logits"
)
INDICES_FIX = (
    "To count it, change its indices= function to return the chosen expert indices, integers from 0 to 7, one "
    "or 2 for each token, or drop indices= and name the module inside it whose output is the router's logits"
)


def _refused_by_indices(tmp_path, indices, steps=4, every=None, compile_backend=None, endpoint=None):
    """Run `IndexRouter` with the given `indices=` function; the router must be refused, said once
    as a warning, never raised into training. Returns the records and the warning."""
    torch.manual_seed(0)
    model = Model(IndexRouter(16))
    if endpoint is None:
        path = tmp_path / "given.jsonl"
        out = {"sink": tensorquill.JsonlSink(path)}
    else:  # streamed: the local file is written where the run is, in tensorquill-out/
        path = tmp_path / "tensorquill-out" / "given.rank0.jsonl"
        out = {"endpoint": endpoint}
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'") as caught:
        session = tensorquill.init(model, run_id="given", routers={"router": Router(EXPERTS, TOP_K, indices=indices)},
                                   every=every or steps, **out)
        stepped = torch.compile(model, backend=compile_backend) if compile_backend else model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(steps):
            optimizer.zero_grad()
            stepped(torch.randn(6, 20, 4)).backward()
            optimizer.step()
        session.close()
    said = [str(w.message) for w in caught if "not counting the router" in str(w.message)]
    assert len(said) == 1
    assert INDICES_FIX in said[0] and "pass routers=" not in said[0]
    return [json.loads(line) for line in path.read_text().splitlines()], said[0]


def _one_refusal(records, step=0):
    assert [r["step"] for r in records if "moe_refused" in r] == [step]
    [why] = [r["moe_refused"]["router"] for r in records if "moe_refused" in r]
    [entry] = [e for e in tensorquill.build_evidence(records)["not_checked"] if e["check"] == "router"]
    assert entry["reason"] == f"router router is not counted: {why}" and entry["step"] == step
    return why


def test_an_indices_function_that_returns_the_wrong_shape_is_refused(tmp_path):
    records, said = _refused_by_indices(tmp_path, lambda out: out[..., 0].sum(0))  # [20] for 120 tokens
    assert len(records) == 4 and _no_entry(records)
    why = _one_refusal(records)
    assert why.startswith("its indices= function returned shape (20,), which is not expert indices for the 120 tokens")
    assert why.endswith(INDICES_FIX) and why in said


@pytest.mark.parametrize(
    ("indices", "start"),
    [
        (lambda out: out.float(), "its indices= function returned a float32 tensor, not integer expert indices"),
        (lambda out: out > 3, "its indices= function returned a bool tensor"),
        (lambda out: out.tolist(), "its indices= function returned list, not a tensor"),
        # On another device than the counts, `==` would raise in the hook: refused instead.
        (lambda out: out.to("meta"), "its indices= function returned a tensor on meta, not on cpu where the counts are"),
    ],
    ids=["float", "bool", "list", "device"],
)
def test_an_indices_function_that_returns_no_integer_tensor_is_refused(tmp_path, indices, start):
    records, _ = _refused_by_indices(tmp_path, indices)
    assert _no_entry(records) and _one_refusal(records).startswith(start)


@pytest.mark.parametrize("compile_backend", [None, "eager"], ids=["eager", "compiled"])
def test_an_indices_function_that_raises_is_refused_and_training_goes_on(tmp_path, compile_backend):
    def broken(out):
        raise KeyError("indices")  # as one written for a router that returns a dict

    records, said = _refused_by_indices(tmp_path, broken, compile_backend=compile_backend)
    assert len(records) == 4 and _no_entry(records)  # every step trained and was written
    why = _one_refusal(records)
    assert why.startswith("its indices= function raised ") and why.endswith(INDICES_FIX)


def test_an_indices_function_whose_values_leave_the_experts_is_refused_from_that_block(tmp_path):
    """Values are judged after the counts' copy to the CPU that `flush` makes anyway, so the
    first block (steps 0-3, in range) is counted and the router is refused from the block in
    which a value fell outside 0..7: a router counted and then refused."""
    calls = []

    def drifting(out):
        calls.append(1)
        return out + EXPERTS if len(calls) == 6 else out  # step 5 only, all its indices out

    records, said = _refused_by_indices(tmp_path, drifting, steps=8, every=4)
    assert _routed(records[:4]) == [120 * TOP_K] * 4
    assert _no_entry(records[4:]) and len(records) == 8
    why = _one_refusal(records, step=4)
    assert why.startswith("its indices= function returned 240 values outside 0 to 7 at step 5, which match no expert")
    evidence = tensorquill.build_evidence(records)
    # Counted steps are judged; the refused ones are a router missing from the record, as for
    # any router a record lacks, never zeros.
    assert [s["step"] for s in evidence["samples"] if not s.get("missing")] == [0, 1, 2, 3]
    assert [s["step"] for s in evidence["samples"] if s.get("missing")] == [4, 5, 6, 7]
    for kind in MOE_KINDS:
        check = evidence["kind_checks"][kind]
        assert check["value"] == "unavailable" and check["records"] == 4 and check["reason"].endswith(INDICES_FIX)


def test_the_reason_names_the_router_and_both_fixes_in_every_place_it_is_printed(tmp_path):
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'") as caught:
        records = _run(tmp_path, CountsFirstRouter(16), {"router": Router(EXPERTS, TOP_K)})
    why = (
        "its output's first integer tensor has shape (8,), which is not expert indices for the 120 tokens it "
        f"was called on. {DEFAULT_FIX}"
    )
    [said] = [str(w.message) for w in caught if "not counting the router" in str(w.message)]
    assert said == f"TensorQuill is not counting the router 'router': {why}. Its expert loads are not recorded."
    assert records[0]["moe_refused"] == {"router": why}
    reason = f"router router is not counted: {why}"
    evidence = tensorquill.build_evidence(records)
    [entry] = [e for e in evidence["not_checked"] if e["check"] == "router"]
    assert entry["reason"] == reason
    assert all(evidence["kind_checks"][kind]["reason"] == reason for kind in MOE_KINDS)
    assert f"Not checked (router): {reason}" in incident_summary(evidence).splitlines()


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
    # The server prints the capture's reason as written: the router, why, and both fixes.
    [entry] = [e for e in rust["not_checked"] if e["check"] == "router"]
    assert entry["reason"] == (
        "router refused is not counted: its output's first integer tensor has shape (8,), which is not expert "
        f"indices for the 120 tokens it was called on. {DEFAULT_FIX.replace(repr('router'), repr('refused'))}"
    )


def test_the_server_reads_an_indices_router_counted_and_then_refused_as_python_does(tmp_path, monkeypatch):
    """A state the shape guard never made: a router counted in the first block and refused from
    the second, when its `indices=` function returned values outside the experts."""
    _binary()
    from test_flight_e2e import Server, stored_records

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    calls = []

    def drifting(out):
        calls.append(1)
        return out + EXPERTS if len(calls) == 6 else out

    server = Server(tmp_path / "store")
    try:
        records, _ = _refused_by_indices(tmp_path, drifting, steps=8, every=4, endpoint=server.endpoint)
    finally:
        assert server.terminate() == 0
    stored = stored_records(tmp_path / "store", "given")
    assert stored == records and len(stored) == 8
    assert _routed(stored[:4]) == [120 * TOP_K] * 4 and _no_entry(stored[4:])
    python = tensorquill.build_evidence(stored)
    rust = tensorquill.build_evidence(stored, detectors="rust")
    assert rust["detectors"] == "rust"
    [entry] = [e for e in rust["not_checked"] if e["check"] == "router"]
    assert entry == next(e for e in python["not_checked"] if e["check"] == "router")
    assert entry["step"] == 4 and entry["reason"].endswith(INDICES_FIX)
    assert rust["kind_checks"] == python["kind_checks"] and rust["samples"] == python["samples"]
    assert all(rust["kind_checks"][kind]["value"] == "unavailable" for kind in MOE_KINDS)


# --- What the counting could not read: refused, never raised into training ---------------------
#
# A default router (or a named inner module) whose output the counting cannot read used to raise
# into the training loop: an integer tensor or logits on another device than the counts (a model
# split across GPUs), a `top_k` above the number of experts, uint32 indices, float8 logits, a
# sparse tensor or a DTensor. Logits of another width than `n_experts` raised nothing and wrote
# zeros that read as a router that routed no token. Each is now refused as the shape guard
# refuses, from types, devices and shapes alone, so the same under `torch.compile`. On CPU,
# `meta` stands in for another device; `mps` is a real one where the machine has it.

DEVICE_FIX = (
    "To count it, pass routers={NAME: tensorquill.Router(8, 2, indices=...)} with a function that returns the "
    "chosen expert indices from its output on cpu, where the counts are"
)
COMPILED = pytest.mark.parametrize("compile_backend", [None, "eager"], ids=["eager", "compiled"])


class Emits(nn.Module):
    """A router whose output is `make(logits)`, so a test can give it any output."""

    def __init__(self, make) -> None:
        super().__init__()
        self.w_g = nn.Linear(16, EXPERTS, bias=False)
        self.make = make

    def forward(self, x):
        return self.make(self.w_g(x))


class Inner(nn.Module):
    """A router with an inner module, `inner`, whose output is `make(logits)`; the router itself
    returns indices on the CPU."""

    def __init__(self, make) -> None:
        super().__init__()
        self.inner = Emits(make)
        self.gate = nn.Linear(16, EXPERTS, bias=False)

    def forward(self, x):
        self.inner(x)
        return self.gate(x).topk(TOP_K, dim=-1).indices


def _refused(tmp_path, router, name, spec, compile_backend=None, steps=4):
    """Run, expecting `name` refused and said once, never raised: every step trains and is written.
    Returns the reason as the record gives it."""
    with pytest.warns(RuntimeWarning, match=f"not counting the router {name!r}") as caught:
        records = _run(tmp_path, router, {name: spec}, steps=steps, compile_backend=compile_backend)
    said = [str(w.message) for w in caught if "not counting the router" in str(w.message)]
    assert len(said) == 1 and len(records) == steps and _no_entry(records, name)
    assert [r["step"] for r in records if "moe_refused" in r] == [0]
    why = records[0]["moe_refused"][name]
    assert said[0] == f"TensorQuill is not counting the router {name!r}: {why}. Its expert loads are not recorded."
    [entry] = [e for e in tensorquill.build_evidence(records)["not_checked"] if e["check"] == "router"]
    assert entry["reason"] == f"router {name} is not counted: {why}" and entry["routers"] == [name]
    return why


ELSEWHERE = ["meta"] + (["mps"] if torch.backends.mps.is_available() else [])


@COMPILED
@pytest.mark.parametrize("device", ELSEWHERE)
@pytest.mark.parametrize(
    ("make", "what"),
    [
        (lambda logits: logits.topk(TOP_K, dim=-1).indices, "its output's first integer tensor"),
        (lambda logits: logits, "its output's first floating-point tensor"),
    ],
    ids=["indices", "logits"],
)
def test_a_router_whose_output_is_on_another_device_is_refused_not_raised(tmp_path, make, what, device,
                                                                           compile_backend):
    """It raised into training before (`==` across devices). The fix it names is the one that
    holds: an inner module's logits would be on that device too, so only `indices=` is offered."""
    on = str(torch.empty(0, device=device).device)
    why = _refused(tmp_path, Emits(lambda logits: make(logits).to(device)), "router", Router(EXPERTS, TOP_K),
                   compile_backend)
    assert why == f"{what} is on {on}, not on cpu where the counts are. {DEVICE_FIX.replace('NAME', repr('router'))}"


@COMPILED
def test_a_named_inner_module_on_another_device_is_refused_the_same_way(tmp_path, compile_backend):
    why = _refused(tmp_path, Inner(lambda logits: logits.to("meta")), "router.inner", Router(EXPERTS, TOP_K),
                   compile_backend)
    assert why == (
        "its output's first floating-point tensor is on meta, not on cpu where the counts are. "
        + DEVICE_FIX.replace("NAME", repr("router.inner"))
    )


@COMPILED
@pytest.mark.parametrize(
    ("make", "shape"),
    [
        (lambda logits: torch.cat([logits, logits], dim=-1), "(6, 20, 16)"),  # 16 wide for 8 experts
        (lambda logits: (logits @ torch.ones(EXPERTS, 16), logits), "(6, 20, 16)"),  # (hidden, logits), as a block
        (lambda logits: logits.sum(), "()"),
    ],
    ids=["wider", "hidden-first", "scalar"],
)
def test_logits_whose_last_dimension_is_not_the_experts_are_refused_not_counted_as_zeros(
    tmp_path, make, shape, compile_backend
):
    """Before, nothing was counted and every record carried eight zeros for the router."""
    why = _refused(tmp_path, Emits(make), "router", Router(EXPERTS, TOP_K), compile_backend)
    assert why == (
        f"its output has no integer tensor and its first floating-point tensor has shape {shape}, which is not "
        f"logits for its 8 experts. {DEFAULT_FIX}"
    )


@COMPILED
def test_a_top_k_above_the_experts_is_refused_with_its_own_fix(tmp_path, compile_backend):
    why = _refused(tmp_path, Emits(lambda logits: logits), "router", Router(EXPERTS, 10), compile_backend)
    assert why == (
        "its top_k is 10 but it has 8 experts. To count it, pass routers={'router': tensorquill.Router(n_experts, "
        "top_k)} with its number of experts and a top_k of at most that number"
    )


@COMPILED
@pytest.mark.parametrize(
    ("make", "start"),
    [
        (lambda logits: logits.topk(TOP_K, dim=-1).indices.to(torch.uint32),
         "its output's first integer tensor is uint32, which TensorQuill cannot count"),
        (lambda logits: logits.to(torch.float8_e4m3fn),
         "its output's first floating-point tensor is float8_e4m3fn, which TensorQuill cannot count"),
        (lambda logits: logits.topk(TOP_K, dim=-1).indices.reshape(-1, TOP_K).to_sparse(),
         "its output's first integer tensor is a sparse_coo tensor, not a dense one"),
    ],
    ids=["uint32", "float8", "sparse"],
)
def test_a_tensor_the_counting_cannot_read_is_refused_not_raised(tmp_path, make, start, compile_backend):
    why = _refused(tmp_path, Emits(make), "router", Router(EXPERTS, TOP_K), compile_backend)
    assert why == f"{start}. {DEFAULT_FIX}"


@COMPILED
def test_an_indices_function_that_returns_uint32_is_refused_not_raised(tmp_path, compile_backend):
    records, said = _refused_by_indices(tmp_path, lambda out: out.to(torch.uint32), compile_backend=compile_backend)
    assert _no_entry(records)
    assert _one_refusal(records) == (
        f"the tensor its indices= function returned is uint32, which TensorQuill cannot count. {INDICES_FIX}"
    )


def test_a_dtensor_is_refused_not_raised(tmp_path):
    """Tensor parallelism can hand a router's output out as a DTensor; `==` with the plain expert
    numbers raised. One process, gloo, so it runs on CPU."""
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or dist.is_initialized():
        pytest.skip("needs torch.distributed and no process group yet")
    from torch.distributed.tensor import DTensor, Replicate, init_device_mesh

    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        mesh = init_device_mesh("cpu", (1,))
        for compile_backend in (None, "eager"):
            torch._dynamo.reset()
            router = Emits(lambda logits: DTensor.from_local(logits, mesh, [Replicate()]))
            why = _refused(tmp_path / str(compile_backend), router, "router", Router(EXPERTS, TOP_K), compile_backend)
            assert why == f"its output's first floating-point tensor is a DTensor, not a plain tensor. {DEFAULT_FIX}"
    finally:
        dist.destroy_process_group()


def test_in_eager_mode_anything_else_the_counting_raises_on_refuses_the_router(tmp_path):
    """The checks are what keep a compiled hook from raising; in eager mode whatever they do not
    foresee is caught as well. A nested tensor of indices is one: reading its shape raises."""
    def nested(logits):
        return torch.nested.nested_tensor([logits[0].topk(TOP_K, dim=-1).indices, logits[1, :5].argsort(dim=-1)])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")
        why = _refused(tmp_path, Emits(nested), "router", Router(EXPERTS, TOP_K))
    assert why.startswith("counting its output raised RuntimeError: ") and why.endswith(DEFAULT_FIX)


@COMPILED
@pytest.mark.parametrize(
    ("make", "spec", "routed"),
    [
        (lambda logits: logits.topk(TOP_K, dim=-1).indices.int(), Router(EXPERTS, TOP_K), 120 * TOP_K),
        (lambda logits: logits.topk(TOP_K, dim=-1).indices.to(torch.uint8), Router(EXPERTS, TOP_K), 120 * TOP_K),
        (lambda logits: logits.half(), Router(EXPERTS, TOP_K), 120 * TOP_K),
        (lambda logits: (logits.softmax(-1), logits), Router(EXPERTS, TOP_K), 120 * TOP_K),
        # An integer router's top_k is only written down: above the experts it is counted as before.
        (lambda logits: logits.topk(TOP_K, dim=-1).indices, Router(EXPERTS, 12), 120 * TOP_K),
        # No tensor at all: nothing counted and nothing refused, as before (zeros).
        (lambda logits: None, Router(EXPERTS, TOP_K), 0),
    ],
    ids=["int32", "uint8", "float16", "weights-then-logits", "int-top_k-above", "no-tensor"],
)
def test_what_was_counted_before_is_counted_as_before(tmp_path, make, spec, routed, compile_backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        records = _run(tmp_path, Emits(make), {"router": spec}, compile_backend=compile_backend)
    assert _routed(records) == [routed] * 4 and not any("moe_refused" in r for r in records)


def test_the_server_reads_a_router_refused_for_its_device_as_python_does(tmp_path):
    binary = _binary()
    with pytest.warns(RuntimeWarning, match="not counting the router 'router'"):
        records = _run(tmp_path, Emits(lambda logits: logits.to("meta")), {"router": Router(EXPERTS, TOP_K)})
    python = tensorquill.build_evidence(records)
    rust = tensorquill.build_evidence(records, detectors="rust", server_binary=binary)
    assert rust["detectors"] == "rust"
    [entry] = [e for e in rust["not_checked"] if e["check"] == "router"]
    assert entry == next(e for e in python["not_checked"] if e["check"] == "router")
    assert entry["reason"].endswith(DEVICE_FIX.replace("NAME", repr("router")))
    assert rust["kind_checks"] == python["kind_checks"]
