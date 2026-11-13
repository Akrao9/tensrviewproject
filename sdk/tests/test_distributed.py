import pytest
import torch

import tensorlens


def test_identical_replicas_have_zero_spread():
    p0 = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16), torch.tensor([0.5, -0.5])]
    p1 = [t.clone() for t in p0]
    fp = torch.stack([tensorlens.parameter_fingerprints(p0), tensorlens.parameter_fingerprints(p1)])
    spread = tensorlens.spread_from_fingerprints(fp)
    assert spread.shape == (2, 2)
    assert torch.all(spread == 0.0)


def test_weight_drift_and_sign_flip_are_caught():
    # Tensor 0: small drift on rank 1. Tensor 1: sign flip ([1, -1] vs [1, 1]) with identical norm.
    r0 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, -1.0])]
    r1 = [torch.tensor([1.0, 2.0, 3.001]), torch.tensor([1.0, 1.0])]
    fp = torch.stack([tensorlens.parameter_fingerprints(r0), tensorlens.parameter_fingerprints(r1)])
    spread = tensorlens.spread_from_fingerprints(fp)

    # Tensor 0 drifts in both norm and sum
    assert spread[0, 0].item() > 1e-4
    assert spread[0, 1].item() > 1e-4
    # Tensor 1 has identical L2 norm (spread == 0) but a large sum spread (2 / sqrt(2))
    assert spread[1, 0].item() == pytest.approx(0.0, abs=1e-12)
    assert spread[1, 1].item() == pytest.approx(2.0 / (2.0**0.5), rel=1e-6)


def test_replica_spread_requires_explicit_group():
    with pytest.raises(ValueError, match="replication process group"):
        tensorlens.replica_spread([torch.ones(2)], None)
    with pytest.raises(ValueError, match="no replicated parameters"):
        tensorlens.parameter_fingerprints([])
