"""Capture paths that raised into training or dropped data, found in the plan's sweep after the
outside review (lane T2, Nov 18, 2026).

1. The cross-GPU drift check (`Session.record_replica_spread`) read each replicated parameter's
   fingerprints right before an `all_gather`, and the fingerprints raised on a parameter on
   `meta` or a sparse one: that rank left the collective the others had entered. Now every rank
   gathers the same rows, a parameter some replica cannot read is flagged in its own column and
   left out of `replica_spread` on every rank, and a warning names it once. A NaN weight that was
   read still reads as drift.
2. A trainable parameter left on `meta` that came first in `named_parameters` (one registered on
   the root module) put every capture buffer on `meta`, and the flush raised.
3. Router counting with a tensor subclass that refuses a plain tensor beside it (as DTensor does)
   still raises under `torch.compile`. It was reproduced by scripts kept in the session notes and is
   not fixed here: which subclasses to stop counting is the author's choice (see the plan).
4. `_decode` wrote `weights_unchanged` only when the first parameter's check was known, so one
   unread first parameter dropped the list for every other parameter.

Distributed tests fake the `all_gather` in one process, or run two gloo ranks with a file store,
so no port is taken.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import distributed as D
from tensorquill.capture import _buffers_device
from tensorquill.evidence import input_counts, kind_checks

META_WHY = "it is on meta, which holds no values"


class Holder(nn.Module):
    """A module whose one parameter is on `meta` (as a model built on `meta` and not fully
    materialized leaves it), unused by the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.extra = nn.Parameter(torch.empty(3, device="meta"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _replica_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 2), Holder())  # `1.extra` comes after the real parameters


def _fake_two_ranks(monkeypatch, other=None):
    """Fake a two-rank `all_gather`: rank 0 is this process; rank 1 gathers `other(table)`'s
    table, or this rank's own."""
    import torch.distributed as dist

    gathers = []

    def gather(out, table, group=None):
        gathers.append(table.clone())
        out[0].copy_(table)
        out[1].copy_(table if other is None else other(table))

    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    monkeypatch.setattr(dist, "all_gather", gather)
    return gathers


def _train(model, session, steps, gather=False):
    for _ in range(steps):
        model(torch.ones(2, 4)).sum().backward()
        if gather:
            # Since lane Z1 (SDK-22) the scheduled check gathers nothing: the gather these tests
            # are about is the explicit call's, a collective every rank makes on the same step.
            session.record_replica_spread()
        session.step()


def _replica_warnings(caught):
    return [str(w.message) for w in caught if "across replicas" in str(w.message)]


# ---- 1. the fingerprints gathered for the drift check ---------------------------------------


def test_the_gathered_table_reads_every_dense_parameter_as_before_and_flags_the_rest():
    a, b = torch.randn(5, dtype=torch.float64), torch.arange(6.0).reshape(2, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sparse CSR is in beta, nested tensors a prototype
        nested = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
        params =[a, torch.empty(3, device="meta"), torch.eye(3).to_sparse(), torch.eye(3).to_sparse_csr(), nested, b]
    table, reasons = D._fingerprint_table(params, projection=True)
    assert table.shape == (6, D.FINGERPRINT_WIDTH + 1) and table.dtype == torch.float64
    # The rows read are the public fingerprints, each seeded by its place in the list as before.
    before = D.parameter_fingerprints([a, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), b])
    assert torch.equal(table[[0, 5], :3], before[[0, 5]])
    assert table[:, 3].tolist() == [0.0, 1.0, 3.0, 4.0, 2.0, 0.0]
    assert torch.all(table[1:5, :3] == 0)
    assert reasons == [
        None, META_WHY, "it is a sparse_coo tensor, not a dense one", "it is a sparse_csr tensor, not a dense one",
        "it is a nested tensor", None,
    ]
    assert D._fingerprint_table(params, projection=False)[0].shape == (6, 3)
    # A NaN in a dense tensor is read: its row is NaN and flagged as read.
    nan = torch.tensor([1.0, math.nan])
    row = D._fingerprint_table([nan], projection=True)[0][0]
    assert math.isnan(row[0]) and row[3] == 0.0


def test_a_parameter_whose_reading_raises_is_flagged_and_the_others_keep_their_values(monkeypatch):
    params = [torch.randn(4), torch.randn(3), torch.randn(2)]
    want = D.parameter_fingerprints(params)
    real = D._projection_f64

    def projection(p, seed):
        if seed == 1:
            raise RuntimeError("no memory for the float64 chunk")
        return real(p, seed)

    monkeypatch.setattr(D, "_projection_f64", projection)
    table, reasons = D._fingerprint_table(params, projection=True)
    assert torch.equal(table[[0, 2], :3], want[[0, 2]])
    assert table[1].tolist() == [0.0, 0.0, 0.0, float(D._RAISED)]
    assert reasons[1] == "reading it raised RuntimeError: no memory for the float64 chunk"
    assert D._unread_reason(D._RAISED) == "reading it raised an error"


def test_a_meta_parameter_is_gathered_left_out_of_the_record_and_named_once(monkeypatch):
    model = _replica_model()
    gathers = _fake_two_ranks(monkeypatch)
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, auto_step=False, every=2, replica_group=object(), replica_every=1)
        _train(model, session, 4, gather=True)
        session.close()
    # Every check gathered every parameter's row, the one on `meta` included.
    assert [g.shape for g in gathers] == [(3, 4)] * 4
    assert [set(r["replica_spread"]) for r in sink.records] == [{"0.weight", "0.bias"}] * 4
    assert sink.records[0]["replica_spread"]["0.weight"] == [0.0, 0.0, 0.0]
    said = _replica_warnings(caught)
    assert said == [
        f"TensorQuill is not comparing the parameter '1.extra' across replicas: {META_WHY}. It is left out of "
        "replica_spread, so the cross-GPU drift check does not cover it."
    ]
    assert tensorquill.detect_replica_drift(sink.records) is None


def test_a_parameter_only_another_replica_cannot_read_is_left_out_here_too(monkeypatch):
    """Ranks decide from the gathered flags, which every rank sees, so all leave out the same
    rows whatever each could read itself."""
    model = nn.Linear(4, 2)

    def peer(table):
        theirs = table.clone()
        theirs[1] = 0.0
        theirs[1, -1] = 1.0  # `bias` is on meta on the other rank
        return theirs

    _fake_two_ranks(monkeypatch, peer)
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, auto_step=False, every=2, replica_group=object(), replica_every=1)
        _train(model, session, 2, gather=True)
        session.close()
    assert [set(r["replica_spread"]) for r in sink.records] == [{"weight"}] * 2
    assert _replica_warnings(caught) == [
        f"TensorQuill is not comparing the parameter 'bias' across replicas: on another replica in the group, "
        f"{META_WHY}. It is left out of replica_spread, so the cross-GPU drift check does not cover it."
    ]


def test_a_nan_weight_that_was_read_still_reads_as_drift_beside_one_left_out(monkeypatch):
    model, other = _replica_model(), _replica_model()
    with torch.no_grad():
        other[0].weight[0, 0] = math.nan
    _fake_two_ranks(monkeypatch, lambda table: D._fingerprint_table(other.parameters(), projection=True)[0])
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, sink=sink, auto_step=False, replica_group=object(), replica_every=1)
        _train(model, session, 1, gather=True)
        session.close()
    spread = sink.records[0]["replica_spread"]
    assert set(spread) == {"0.weight", "0.bias"}
    assert all(math.isnan(v) for v in spread["0.weight"]) and spread["0.bias"] == [0.0, 0.0, 0.0]
    hit = tensorquill.detect_replica_drift(sink.records)
    assert set(hit["parameters"]) == {"0.weight"} and hit["parameters"]["0.weight"]["non_finite"] is True


def test_with_every_replicated_parameter_left_out_the_drift_check_is_never_called_checked(monkeypatch):
    model = _replica_model()
    _fake_two_ranks(monkeypatch)
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, sink=sink, auto_step=False, replica_group=object(), replica_every=1,
                                   replica_params=["1.extra"])
        _train(model, session, 2)
        session.close()
    assert not any("replica_spread" in r for r in sink.records)  # never an empty map, never zeros
    assert kind_checks(input_counts(sink.records), [])["replica_drift"]["value"] == "not_configured"


def test_a_spread_passed_in_is_recorded_whole_as_before():
    model = nn.Linear(4, 2)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, auto_step=False)
    session.record_replica_spread(torch.tensor([[0.25, 0.0], [0.0, 0.5]], dtype=torch.float64))
    session.step()
    session.close()
    assert sink.records[0]["replica_spread"] == {"weight": [0.25, 0.0], "bias": [0.0, 0.5]}


def _two_rank_worker(rank, store_path, out_dir):
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=2, store=dist.FileStore(store_path, 2),
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 2), Holder())
        if rank == 1:  # only rank 0 left its parameter on `meta`
            model[1].extra = nn.Parameter(torch.ones(3))
        path = Path(out_dir) / f"rank{rank}.jsonl"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = tensorquill.init(model, run_id="r1", sink=tensorquill.JsonlSink(path), every=2,
                                       replica_group=dist.group.WORLD, replica_every=1)
            optimizer = torch.optim.SGD(model[0].parameters(), lr=0.1)
            for _ in range(3):
                model(torch.ones(2, 4)).sum().backward()
                session.record_replica_spread()  # the explicit gather, on every rank (lane Z1)
                optimizer.step()
                optimizer.zero_grad()
            session.close()
        (Path(out_dir) / f"warned{rank}.json").write_text(json.dumps(_replica_warnings(caught)))
    finally:
        dist.destroy_process_group()


def test_on_two_ranks_one_meta_parameter_on_one_rank_neither_hangs_nor_raises(tmp_path):
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("needs torch.distributed")
    import torch.multiprocessing as mp

    context = mp.start_processes(_two_rank_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2,
                                 join=False, start_method="spawn")
    for _ in range(240):  # a rank stuck in the collective would never finish: fail, do not hang
        if context.join(timeout=0.5):
            break
    else:
        for process in context.processes:
            process.terminate()
        pytest.fail("a rank did not finish: it was left waiting in the all_gather")
    for rank in (0, 1):
        records = [json.loads(line) for line in (tmp_path / f"rank{rank}.jsonl").read_text().splitlines()]
        assert [set(r["replica_spread"]) for r in records] == [{"0.weight", "0.bias"}] * 3
        assert all(v == [0.0, 0.0, 0.0] for r in records for v in r["replica_spread"].values())
        # Each rank's own observation (lane Z1): rank 0 lists its parameter on `meta` as unread.
        observed = [r["replica"] for r in records]
        assert len(observed) == 3 and all(o["group"] == [0, 1] for o in observed)
        if rank == 0:
            assert all(o["unread"] == {"1.extra": META_WHY} and "1.extra" not in o["params"] for o in observed)
        else:
            assert all(o["unread"] == {} and set(o["params"]) == {"0.weight", "0.bias", "1.extra"} for o in observed)
        warned = json.loads((tmp_path / f"warned{rank}.json").read_text())
        assert len(warned) == 1 and "'1.extra'" in warned[0]
        assert (META_WHY if rank == 0 else f"on another replica in the group, {META_WHY}") in warned[0]


# ---- 2. where the buffers are kept --------------------------------------------------------


class RootMeta(nn.Module):
    """A parameter registered on the root module comes first in `named_parameters`; this one is
    left on `meta`."""

    def __init__(self) -> None:
        super().__init__()
        self.extra = nn.Parameter(torch.empty(4, device="meta"))
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def test_a_root_parameter_on_meta_puts_no_buffer_on_meta_and_the_flush_reads_the_run():
    torch.manual_seed(0)
    model = RootMeta()
    assert next(iter(dict(model.named_parameters()))) == "extra"
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2, weights=1)
    assert session.device == torch.device("cpu")
    buffers = [session._act, session._grad, session._ring_act, session._ring_grad, session._clip]
    assert {b.device.type for b in buffers} == {"cpu"}
    optimizer = torch.optim.SGD(model.blocks.parameters(), lr=0.0)  # weights stay as they are
    for _ in range(3):
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2]
    assert set(records[0]["layers"]) == {"blocks.0", "blocks.1"}
    assert set(records[0]["grad_norms"]) == {"blocks.0.weight", "blocks.0.bias", "blocks.1.weight", "blocks.1.bias"}
    assert math.isnan(records[0]["weight_norms"]["extra"])  # not read, and no norm made up
    # Item 4 end to end: the first parameter was not read, the others were, and none moved.
    assert "weights_unchanged" not in records[0]  # nothing to compare with yet
    assert [sorted(r["weights_unchanged"]) for r in records[1:]] == [
        ["blocks.0.bias", "blocks.0.weight", "blocks.1.bias", "blocks.1.weight"]] * 2


def test_the_buffers_go_where_the_first_parameter_with_values_is():
    meta, cpu = torch.empty(2, device="meta"), torch.ones(2)
    assert _buffers_device([]) == torch.device("cpu")
    assert _buffers_device([meta, meta]) == torch.device("cpu")
    assert _buffers_device([meta, cpu]) == torch.device("cpu")
    if torch.backends.mps.is_available():
        mps = torch.ones(2, device="mps")
        assert _buffers_device([meta, mps, cpu]) == mps.device
        assert _buffers_device([cpu, mps]) == torch.device("cpu")  # the first, as before


# ---- 4. weights_unchanged with an unread first parameter -----------------------------------


def _decode_one_check(rows):
    model = nn.Linear(2, 1)  # parameters "weight", then "bias"
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), auto_step=False, layers=())
    try:
        floats = torch.tensor(rows, dtype=torch.float32).reshape(-1)
        return session._decode([7], [None], floats, None, None, None, [False], [True], [True], [0])[0]
    finally:
        session.close()


def test_an_unread_first_parameter_leaves_the_others_in_weights_unchanged():
    # gradient norm, seen, weight norm, unchanged (1 = unchanged, 0 = changed, -1 = not known)
    record = _decode_one_check([[1.0, 1.0, math.nan, -1.0], [1.0, 1.0, 0.5, 1.0]])
    assert record["weights_unchanged"] == ["bias"]
    assert _decode_one_check([[1.0, 1.0, 2.0, -1.0], [1.0, 1.0, 0.5, 0.0]])["weights_unchanged"] == []


def test_a_check_that_knew_no_parameter_still_writes_no_list():
    record = _decode_one_check([[1.0, 1.0, 2.0, -1.0], [1.0, 1.0, 0.5, -1.0]])
    assert "weights_unchanged" not in record and set(record["weight_norms"]) == {"weight", "bias"}
    assert _decode_one_check([[1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 0.5, -1.0]])["weights_unchanged"] == ["weight"]
