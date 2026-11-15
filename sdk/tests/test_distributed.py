import pytest
import torch

import tensorquill


def test_identical_replicas_have_zero_spread():
    p0 = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16), torch.tensor([0.5, -0.5])]
    p1 = [t.clone() for t in p0]
    fp = torch.stack([tensorquill.parameter_fingerprints(p0), tensorquill.parameter_fingerprints(p1)])
    spread = tensorquill.spread_from_fingerprints(fp)
    assert spread.shape == (2, 3)
    assert torch.all(spread == 0.0)


def test_weight_drift_and_sign_flip_are_caught():
    # Tensor 0: small drift on rank 1. Tensor 1: sign flip ([1, -1] vs [1, 1]) with identical norm.
    r0 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, -1.0])]
    r1 = [torch.tensor([1.0, 2.0, 3.001]), torch.tensor([1.0, 1.0])]
    fp = torch.stack([tensorquill.parameter_fingerprints(r0), tensorquill.parameter_fingerprints(r1)])
    spread = tensorquill.spread_from_fingerprints(fp)

    # Tensor 0 drifts in both norm and sum
    assert spread[0, 0].item() > 1e-4
    assert spread[0, 1].item() > 1e-4
    # Tensor 1 has identical L2 norm (spread == 0) but a large sum spread (2 / sqrt(2))
    assert spread[1, 0].item() == pytest.approx(0.0, abs=1e-12)
    assert spread[1, 1].item() == pytest.approx(2.0 / (2.0**0.5), rel=1e-6)


def test_replica_spread_requires_explicit_group():
    with pytest.raises(ValueError, match="replication process group"):
        tensorquill.replica_spread([torch.ones(2)], None)
    with pytest.raises(ValueError, match="no replicated parameters"):
        tensorquill.parameter_fingerprints([])


def test_replica_params_compares_only_replicated_weights(monkeypatch):
    """Tensor parallelism: each rank holds a different shard of the linear weight, and only
    the norm weight is replicated. Comparing every weight would report false drift."""
    import torch.distributed as dist
    from torch import nn

    class Block(nn.Module):
        def __init__(self, shard):
            super().__init__()
            self.norm = nn.LayerNorm(4)
            self.proj = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.proj.weight.fill_(shard)  # a different shard on each rank
            self.proj.weight.tensor_model_parallel = True  # as Megatron marks sharded weights

    rank0, rank1 = Block(1.0), Block(2.0)
    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)

    def gather(out, fp, group=None):
        out[0].copy_(fp)
        names = [n for n, _ in rank0.named_parameters() if n.startswith("norm")]
        out[1].copy_(tensorquill.parameter_fingerprints(p for n, p in rank1.named_parameters() if n in names))

    monkeypatch.setattr(dist, "all_gather", gather)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(
        rank0, sink=sink, auto_step=False, replica_group=object(), replica_every=1,
        replica_params=lambda name, p: not getattr(p, "tensor_model_parallel", False),
    )
    session.step()
    session.close()
    assert set(sink.records[0]["replica_spread"]) == {"norm.weight", "norm.bias"}
    assert tensorquill.detect_replica_drift(sink.records) is None


def test_replica_params_accepts_names_or_parameters_and_rejects_unknown_names():
    from torch import nn

    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    by_name = tensorquill.init(model, sink=tensorquill.MemorySink(), replica_params=["1.weight"])
    assert by_name.replica_names == ["1.weight"]
    by_name.close()
    by_param = tensorquill.init(model, sink=tensorquill.MemorySink(), replica_params=list(model[1].parameters()))
    assert by_param.replica_names == ["1.weight", "1.bias"]
    by_param.close()
    with pytest.raises(ValueError, match="no trainable parameter"):
        tensorquill.init(model, sink=tensorquill.MemorySink(), replica_params=["2.weight"])


def _spread(a, b):
    fp = torch.stack([tensorquill.parameter_fingerprints([a]), tensorquill.parameter_fingerprints([b])])
    return tensorquill.spread_from_fingerprints(fp)[0]


def test_swapping_two_elements_is_caught_by_the_projection_only():
    # A swap keeps the norm and the sum, which is all the older fingerprint had.
    a = torch.arange(1.0, 9.0)
    b = a.clone()
    b[[1, 5]] = b[[5, 1]]
    spread = _spread(a, b)
    assert spread[0].item() == pytest.approx(0.0, abs=1e-12)
    assert spread[1].item() == pytest.approx(0.0, abs=1e-12)
    assert spread[2].item() > 1e-3
    old = tensorquill.spread_from_fingerprints(
        torch.stack([tensorquill.parameter_fingerprints([a], projection=False),
                     tensorquill.parameter_fingerprints([b], projection=False)])
    )
    assert old.shape == (1, 2) and torch.all(old == 0.0)
    records = [{"step": 0, "replica_spread": {"w": spread.tolist()}}]
    assert tensorquill.detect_replica_drift(records)["parameters"]["w"]["projection_spread"] > 1e-3
    assert tensorquill.detect_replica_drift([{"step": 0, "replica_spread": {"w": old[0].tolist()}}]) is None


def test_projection_is_deterministic_and_leaves_the_random_streams_alone():
    t = torch.randn(10_000, dtype=torch.float64)
    torch.manual_seed(7)
    before = torch.get_rng_state()
    first = tensorquill.parameter_fingerprints([t])
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(first, tensorquill.parameter_fingerprints([t.clone()]))
    # Chunking must not change the value: the weights depend only on the element index.
    from tensorquill import distributed

    small = distributed.PROJECTION_CHUNK
    distributed.PROJECTION_CHUNK = 1000
    try:
        chunked = tensorquill.parameter_fingerprints([t])
    finally:
        distributed.PROJECTION_CHUNK = small
    assert chunked[0, 2].item() == pytest.approx(first[0, 2].item(), rel=1e-12)


def test_records_with_two_spread_values_still_read():
    assert tensorquill.detect_replica_drift([{"step": 0, "replica_spread": {"w": [0.0, 0.0]}}]) is None
    hit = tensorquill.detect_replica_drift([{"step": 0, "replica_spread": {"w": [0.0, 0.5]}}])
    assert hit["parameters"]["w"] == {"norm_spread": 0.0, "sum_spread": 0.5}


@pytest.mark.parametrize("projection, width", [(True, 3), (False, 2)])
def test_session_records_the_projection_unless_switched_off(monkeypatch, projection, width):
    import torch.distributed as dist
    from torch import nn

    other = nn.Linear(3, 2)
    model = nn.Linear(3, 2)
    other.load_state_dict(model.state_dict())

    def gather(out, fp, group=None):
        out[0].copy_(fp)
        out[1].copy_(tensorquill.parameter_fingerprints(other.parameters(), projection=fp.shape[-1] == 3))

    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    monkeypatch.setattr(dist, "all_gather", gather)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(
        model, sink=sink, auto_step=False, replica_group=object(), replica_every=1,
        replica_projection=projection,
    )
    session.step()
    session.close()
    assert {len(v) for v in sink.records[0]["replica_spread"].values()} == {width}
