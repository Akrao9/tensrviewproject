"""Lane X4 (wave X): data.py and probe.py.

Each test reproduces the open part of a row of OPEN_BUGS.md and failed on the base commit (727b71f).
"""

from __future__ import annotations

import pytest
import torch

import tensorquill
from tensorquill import data as D
from tensorquill import data_checks as dc
from tensorquill import evidence as ev

# --- C16: a batch of integers and bools ----------------------------------------------------------


@pytest.mark.parametrize("left, right", [
    ([1, True], [1, 1]),  # the open part: a list that starts with an int and holds a bool
    ([0, False], [0, 0]),
    ([5, 6, True], [5, 6, 1]),
    ([1, True], [True, 1]),  # order still counts
])
def test_c16_a_list_of_ints_holding_a_bool_does_not_hash_as_its_integers(left, right):
    assert D.batch_hash(left) != D.batch_hash(right)


def test_c16_a_mixed_batch_hashes_the_same_as_a_list_a_tuple_or_tensors():
    as_list = D.batch_hash([1, True])
    assert as_list == D.batch_hash((1, True)) == D.batch_hash([torch.tensor(1), torch.tensor(True)])


def test_c16_a_step_with_a_mixed_batch_is_of_its_own_kind():
    # The writer before this fix wrote `[1, True]` as `v2:` and the hash of `[1, 1]`: a hash of the
    # new encoding is of another kind, so neither reader compares it with that one.
    old = D.batch_hash([1, 1])
    new = D.batch_hash([1, True])
    assert new.startswith(D.MIXED_PREFIX) and D.MIXED_PREFIX == "v3:"
    assert dc.hash_kind(new) == "v3" and dc.hash_kind(old) == "v2"
    # One mixed batch makes the step's hash of that kind.
    assert D.batch_hash([0, 1], [2, True]).startswith("v3:")
    assert D.batch_hash([True, 1]).startswith("v3:")


@pytest.mark.parametrize("batches, expected", [
    # Written by data.py at the base commit: batches whose encoding the fix leaves as it was.
    (([0, 1, 2, 3],), "v2:cd09ce7a08834a9e"),
    (([True, False],), "v2:f8738f6061b54869"),
    ((torch.tensor([True, False]),), "v2:f8738f6061b54869"),
    ((["a", 1],), "v2:cc40eb8dfc65b276"),
    (([1.0, 2],), "v2:a77d2ab4eee16922"),
    (([1, 1],), "v2:bb153dc27acf9bbf"),
    (([0, 1], [2, 3]), "v2:aebe7742b6b158e3"),
])
def test_c16_every_other_batch_keeps_its_v2_hash(batches, expected):
    assert D.batch_hash(*batches) == expected


def _drawn(attempt, steps, hashes):
    return [{"schema_version": 1, "run_id": "x4", "rank": 0, "attempt": attempt, "step": step,
             "ts": float(step), "data": {"batches": 1, "batch_hash": digest}} for step, digest in zip(steps, hashes)]


def test_c16_a_resume_across_the_fix_is_not_compared_and_one_after_it_is():
    before, after = D.batch_hash([1, 1]), D.batch_hash([1, True])  # [1, True] by the old writer, and by this one
    head = [D.batch_hash([i, i + 1]) for i in (10, 12, 14)]
    # Attempt 0 draws [1, True] at step 3, attempt 1 resumes at step 3 and draws it again (this writer).
    for at_step_3, compared in ((before, False), (after, True)):
        records = _drawn(0, range(4), [*head, at_step_3]) + _drawn(1, [3], [after])
        evidence = tensorquill.build_evidence(records, detectors="python")
        assert not [i for i in evidence["incidents"] if i["kind"] == "resume_data_mismatch"]
        skipped = [i for i in evidence.get("not_checked", ()) if i.get("check") == "resume_data"]
        assert bool(skipped) == (not compared), compared
        if skipped:
            assert skipped[0]["reason"] == dc.KINDS_DIFFER


def test_c16_two_ranks_drawing_the_same_mixed_batches_are_a_duplicate():
    hashes = [D.batch_hash([i, True]) for i in range(4)]
    records = [{**r, "rank": rank} for rank in (0, 1) for r in _drawn(0, range(4), hashes)]
    found = dc.detect_duplicate_batches(ev.resolve_attempts(records))
    assert found is not None and found["ranks"] == [0, 1]


# --- C22: a mean that overflows float32 in the probe's head numbers ------------------------------


def _numbers(hidden, weight, bias=None):
    from tensorquill import probe as P

    sums = P._head_logit_sums(hidden, weight, bias)
    return P._head_numbers([{**sums, "input_dtype": "torch.float32", "weight_dtype": "torch.float32"}])


def test_c22_logits_whose_float32_sum_overflows_are_read():
    # Eight logits of 3e38 each: finite in FP32 and BF16, their sum (2.4e39) is past float32's range.
    numbers = _numbers(torch.ones(1, 1), torch.full((8, 1), 3e38))
    assert numbers["positions"] == 1 and "overflow_positions" not in numbers and "not_read" not in numbers
    assert numbers["mean"] == pytest.approx(3e38, rel=1e-6)
    assert numbers["norm"] == pytest.approx(3e38 * 8 ** 0.5, rel=1e-6)
    assert numbers["log_z"] == pytest.approx(3e38, rel=1e-6)


def test_c22_a_centred_sum_that_overflows_float32_is_read():
    # A mean of 0 from +-3e38: the shift is 0, and the centred logits' float32 sum overflows.
    weight = torch.tensor([[3e38]] * 4 + [[-3e38]] * 4)
    numbers = _numbers(torch.ones(1, 1), weight)
    assert numbers["positions"] == 1 and "overflow_positions" not in numbers
    assert numbers["mean"] == 0.0
    assert numbers["norm"] == pytest.approx(3e38 * 8 ** 0.5, rel=1e-6)


def test_c22_the_overflowing_mean_is_read_across_chunks(monkeypatch):
    from tensorquill import probe as P

    monkeypatch.setattr(P, "CHUNK_ELEMENTS", 3)  # one position: three logits a chunk, three chunks
    weight = torch.tensor([[3e38], [2e38], [3e38], [3e38], [1e38], [3e38], [3e38], [2e38], [3e38]])
    numbers = _numbers(torch.ones(1, 1), weight)
    assert numbers["positions"] == 1 and "overflow_positions" not in numbers
    assert numbers["mean"] == pytest.approx(float(weight.double().mean()), rel=1e-6)
    spread = float(weight.double().std(unbiased=False))
    assert numbers["gap_relative"] is not None and numbers["gap"] / numbers["gap_relative"] == pytest.approx(spread, rel=1e-5)


def test_c22_ordinary_logits_give_the_same_numbers_as_before():
    torch.manual_seed(0)
    hidden, weight, bias = torch.randn(5, 7, 16), torch.randn(50, 16), torch.randn(50)
    numbers = _numbers(hidden, weight, bias)
    z = torch.nn.functional.linear(hidden.reshape(-1, 16), weight, bias).double()
    assert numbers["positions"] == 35
    assert numbers["mean"] == pytest.approx(float(z.mean()), rel=1e-6, abs=1e-9)
    assert numbers["norm"] == pytest.approx(float(z.norm(dim=-1).mean()), rel=1e-6)
    assert numbers["log_z"] == pytest.approx(float(torch.logsumexp(z, -1).mean()), rel=1e-6)


class _Head(torch.nn.Module):
    def __init__(self, logit: float, vocab: int = 32) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, vocab)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(logit)  # every logit of every position is `logit`

    def forward(self, ids):
        return self.head(self.embed(ids))


def _probe_records(model):
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, probe=torch.arange(6).reshape(2, 3), probe_every=1,
                               probe_head="head")
    opt = torch.optim.SGD(model.embed.parameters(), lr=0.01)
    for _ in range(2):
        model.embed(torch.arange(4)).pow(2).mean().backward()  # the head is left as it is
        opt.step()
        opt.zero_grad()
    session.close()
    return [r["probe"]["head"] for r in sink.records if "probe" in r]


def test_c22_a_head_whose_float32_mean_overflows_is_read_in_the_record():
    heads = _probe_records(_Head(3e38))  # 32 logits of 3e38: a float32 sum of 9.6e39
    assert heads and all(h["positions"] == 6 and "overflow_positions" not in h for h in heads)
    assert heads[-1]["mean"] == pytest.approx(3e38, rel=1e-6)


def test_c22_positions_that_still_overflow_are_counted_in_the_record():
    # 3.4e38 is finite in FP32 and past BF16's range: the BF16 pass's logits are Inf.
    heads = _probe_records(_Head(3.4e38))
    assert heads and all(h["positions"] == 0 and h["overflow_positions"] == 6 for h in heads)
    assert all(h["nonfinite_positions"] == 0 and "overflow" in h["not_read"] for h in heads)


def test_c22_a_position_with_a_nan_leaves_the_others_and_an_overflowing_mean_read():
    hidden = torch.tensor([[1.0], [float("nan")], [2.0]])
    weight = torch.full((8, 1), 1.5e38)  # position 2: eight logits of 3e38, a float32 sum past its range
    numbers = _numbers(hidden, weight)
    assert numbers["positions"] == 2 and numbers["nonfinite_positions"] == 1 and "overflow_positions" not in numbers
    assert numbers["mean"] == pytest.approx((1.5e38 + 3e38) / 2, rel=1e-6)


def test_c16_the_step_record_of_a_mixed_batch_is_v3_and_differs_from_its_integers():
    def recorded(batch):
        sink = tensorquill.MemorySink()
        model = torch.nn.Linear(2, 1)
        session = tensorquill.init(model, sink=sink)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        session.data(batch)
        model(torch.ones(1, 2)).sum().backward()
        opt.step()
        session.close()
        return [r["data"]["batch_hash"] for r in sink.records if "batch_hash" in (r.get("data") or {})]

    mixed, ints = recorded([1, True]), recorded([1, 1])
    assert mixed == [D.batch_hash([1, True])] and ints == [D.batch_hash([1, 1])]
    assert mixed[0].startswith("v3:") and ints[0].startswith("v2:")
