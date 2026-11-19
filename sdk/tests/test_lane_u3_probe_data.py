"""Lane U3: the probe and the loader wrapper leave the user's training as they found it.

C02: the probe's SDPA interposition reaches only the thread running the probe, its statistics
cannot raise into the forward, overlapping probes leave the original function, and the probe
leaves the deterministic-algorithms setting alone. C17: `watch_loader` leaves the batch
sampler its own class (it pickles), records only its own passes, and stops once the session
is closed. C18: a composite config's sub-configs, Python's `random` and NumPy's global state
are as they were after a probe, and a failure to note a batch is not raised under `-W error`.
"""

from __future__ import annotations

import collections
import copy
import io
import pickle
import random
import threading
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, TensorDataset

import tensorquill
from tensorquill import data as tq_data
from tensorquill import probe as tq_probe
from tensorquill.data import WatchedLoader

CPU = [torch.device("cpu")]
TIMEOUT = 20


# --- C02: the SDPA interposition -----------------------------------------------------------------


class Attn(nn.Module):
    def __init__(self, shape=(2, 2, 3, 4)):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        q = x.reshape(self.shape)
        return F.scaled_dot_product_attention(q, q, q).reshape(x.shape)


class Blocks(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _open_probe_in_thread(model, state, *, body=None):
    """Start a probe on another thread whose function waits until released. Returns
    (entered, release, thread, out); `out["result"]` is the probe's result once joined."""
    entered, release, out = threading.Event(), threading.Event(), {}

    def fn(m):
        if body is not None:
            body(m)
        entered.set()
        release.wait(TIMEOUT)

    def run():
        try:
            out["result"] = tq_probe.run_probe(model, fn, state, CPU)
        except BaseException as error:  # noqa: BLE001
            out["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert entered.wait(TIMEOUT)
    return release, thread, out


def test_another_threads_attention_call_is_untouched_while_a_probe_is_open():
    original = F.scaled_dot_product_attention
    q5 = torch.randn(1, 1, 2, 3, 4)  # a valid SDPA input the probe's statistics cannot read
    q4 = torch.randn(2, 2, 3, 4)
    expected5, expected4 = original(q5, q5, q5), original(q4, q4, q4)
    state = tq_probe.ProbeState()
    release, thread, out = _open_probe_in_thread(nn.Linear(2, 2), state)
    try:
        got5 = F.scaled_dot_product_attention(q5, q5, q5)  # raised ValueError before
        got4 = F.scaled_dot_product_attention(q4, q4, q4)
    finally:
        release.set()
        thread.join(TIMEOUT)
    assert torch.equal(got5, expected5) and torch.equal(got4, expected4)
    assert "error" not in out
    # This thread's calls are not the probe's: nothing of them is in its record.
    assert out["result"]["attention"] == {}
    assert F.scaled_dot_product_attention is original


def test_another_threads_forward_does_not_reach_the_probes_block_hooks():
    model = Blocks(nn.Linear(4, 4), nn.Linear(4, 4))
    state = tq_probe.ProbeState()
    tq_probe.install_probe_hooks(list(model.blocks.named_children()), state)
    release, thread, out = _open_probe_in_thread(model, state)
    try:
        model(torch.randn(3, 4))  # the training thread's forward while the probe is open
    finally:
        release.set()
        thread.join(TIMEOUT)
    assert out["result"]["histograms"] == {}


@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), "no keys"])
def test_statistics_that_fail_are_that_entrys_error_and_the_rest_of_the_record_stays(shape):
    class NoKeys(nn.Module):
        def forward(self, x):
            q = x.reshape(2, 2, 3, 4)
            k = q[:, :, :0]
            return F.scaled_dot_product_attention(q, k, k).reshape(x.shape) + x

    bad = NoKeys() if shape == "no keys" else Attn(shape)
    model = Blocks(nn.Linear(48, 48) if shape == "no keys" else nn.Linear(24, 24), bad, Attn((1, 2, -1, 4)))
    state = tq_probe.ProbeState()
    tq_probe.install_probe_hooks(list(model.blocks.named_children()), state)
    width = 48 if shape == "no keys" else 24
    result = tq_probe.run_probe(model, lambda m: m(torch.randn(1, width)), state, CPU)
    assert "error" in result["attention"]["1"]
    assert set(result["attention"]["1"]) == {"error"}
    assert "max_logit" in result["attention"]["2"]  # the next attention call is still read
    assert set(result["histograms"]) == {"0", "1", "2"}


@pytest.mark.parametrize("order", ["first opened closes first", "last opened closes first"])
def test_overlapping_probes_leave_the_original_function(order):
    original = F.scaled_dot_product_attention
    model = Blocks(Attn())
    a_state, b_state = tq_probe.ProbeState(), tq_probe.ProbeState()
    release_a, thread_a, out_a = _open_probe_in_thread(model, a_state, body=lambda m: m(torch.randn(1, 48)))
    release_b, thread_b, out_b = _open_probe_in_thread(nn.Linear(2, 2), b_state)
    first, second = (
        ((release_a, thread_a), (release_b, thread_b)) if order == "first opened closes first"
        else ((release_b, thread_b), (release_a, thread_a))
    )
    for release, thread in (first, second):
        release.set()
        thread.join(TIMEOUT)
    assert F.scaled_dot_product_attention is original
    assert torch._dynamo.eval_frame._stance.stance == "default"  # not left on force_eager
    assert list(out_a["result"]["attention"]) == ["attention"]
    assert out_b["result"]["attention"] == {}
    q = torch.randn(1, 1, 3, 4)
    assert torch.equal(F.scaled_dot_product_attention(q, q, q), original(q, q, q))


def test_a_wrapper_put_over_the_probes_during_a_probe_does_not_make_a_later_probe_recurse():
    original = F.scaled_dot_product_attention
    model = Blocks(Attn())
    try:
        def wrap_over(m):
            inner = F.scaled_dot_product_attention  # another tool wraps whatever is there

            def other_tool(*args, **kwargs):
                return inner(*args, **kwargs)

            F.scaled_dot_product_attention = other_tool

        tq_probe.run_probe(model, wrap_over, tq_probe.ProbeState(), CPU)
        other_tool = F.scaled_dot_product_attention
        assert other_tool is not original  # the other tool's wrapper stays
        result = tq_probe.run_probe(model, lambda m: m(torch.randn(1, 48)), tq_probe.ProbeState(), CPU)
        assert list(result["attention"]) == ["attention"]  # read once, not twice, no RecursionError
        assert F.scaled_dot_product_attention is other_tool
        q = torch.randn(1, 1, 3, 4)
        assert torch.equal(F.scaled_dot_product_attention(q, q, q), original(q, q, q))
    finally:
        F.scaled_dot_product_attention = original


def test_the_probe_leaves_the_deterministic_setting_alone():
    model = Blocks(nn.Linear(48, 48), Attn())
    state = tq_probe.ProbeState()
    tq_probe.install_probe_hooks(list(model.blocks.named_children()), state)
    seen = []

    def fn(m):
        seen.append(torch.are_deterministic_algorithms_enabled())
        m(torch.randn(1, 48))

    before = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        result = tq_probe.run_probe(model, fn, state, CPU)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)
    assert seen == [True]
    assert set(result["histograms"]) == {"0", "1"} and "max_logit" in result["attention"]["1"]


def _histc_reference(tensor: torch.Tensor, bins: int) -> dict:
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        return tq_probe._histogram(tensor, bins)
    finally:
        torch.use_deterministic_algorithms(before)


def _histogram_cases():
    g = torch.Generator().manual_seed(7)
    edges = torch.linspace(-3.0, 5.0, 65)
    return {
        "normal": torch.randn(10_000, generator=g),
        "heavy tails": torch.randn(5_000, generator=g) ** 5,
        "on the bin edges": torch.cat([edges, edges[1:-1], edges.flip(0)]),
        "next to the bin edges": torch.cat([edges.nextafter(edges + 1), edges.nextafter(edges - 1)]),
        "the top value many times": torch.cat([torch.full((100,), 9.0), torch.rand(50, generator=g)]),
        "one value": torch.full((33,), 1.25),
        "integers": torch.randint(-20, 20, (4_000,), generator=g).float(),
        "a tiny span": 1.0 + torch.rand(1_000, generator=g) * 1e-6,
        "large values": torch.randn(2_000, generator=g) * 1e30,
        "with non-finite": torch.tensor([1.0, float("nan"), 2.0, float("inf"), -float("inf"), 3.0]),
        "bf16": torch.randn(3_000, generator=g).to(torch.bfloat16),
        "all non-finite": torch.tensor([float("nan")] * 4),
    }


@pytest.mark.parametrize("case", list(_histogram_cases()))
def test_with_deterministic_algorithms_on_the_histogram_is_histcs(case, monkeypatch):
    tensor = _histogram_cases()[case]
    expected = _histc_reference(tensor, tq_probe.HIST_BINS)
    monkeypatch.setattr(tq_probe, "CHUNK_ELEMENTS", 1000)  # several chunks for the larger cases
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        got = tq_probe._histogram(tensor, tq_probe.HIST_BINS)
    finally:
        torch.use_deterministic_algorithms(before)
    assert got == expected


# --- C17: the loader wrapper ---------------------------------------------------------------------


class FakeSession:
    """What `WatchedLoader` asks of a session: `data(values, **fields)` and `_closed`."""

    def __init__(self):
        self.noted: list = []
        self._closed = False

    def data(self, values, **fields):
        if self._closed:
            raise RuntimeError("the session is closed")
        self.noted.append(values)


DATA40 = TensorDataset(torch.arange(40))  # each sample is its own index


def _indices(batch) -> list:
    return batch[0].tolist()


def _loader(**kwargs) -> DataLoader:
    kwargs.setdefault("batch_size", 2)
    return DataLoader(DATA40, **kwargs)


def _pickles(obj) -> bool:
    try:
        pickle.dumps(obj)
        torch.save(obj, io.BytesIO())
        return True
    except Exception:  # noqa: BLE001
        return False


def test_a_watched_loader_pickles_before_during_and_after_and_keeps_its_sampler(tmp_path):
    model = nn.Linear(1, 1)
    session = tensorquill.init(model, run_id="u3", sink=tensorquill.JsonlSink(tmp_path / "r.jsonl"),
                               heartbeat_seconds=None, stack_seconds=None)
    loader = _loader(shuffle=True)
    sampler, kind = loader.batch_sampler, type(loader.batch_sampler)
    assert _pickles(loader) and _pickles(sampler)
    watched = session.watch_loader(loader)
    assert type(loader.batch_sampler) is kind is BatchSampler
    assert _pickles(loader) and _pickles(loader.batch_sampler)
    it = iter(watched)
    next(it)
    assert type(loader.batch_sampler) is kind and _pickles(loader) and _pickles(loader.batch_sampler)
    assert type(copy.deepcopy(loader).batch_sampler) is kind
    for _ in it:
        pass
    session.close()
    assert loader.batch_sampler is sampler and type(sampler) is kind
    assert _pickles(loader) and _pickles(sampler)


def test_while_a_pass_is_being_opened_the_sampler_pickles_and_copies_as_its_own_class():
    sampler = _loader(shuffle=True).batch_sampler
    assert tq_data._swap_in(sampler)
    try:
        assert type(sampler) is not BatchSampler and isinstance(sampler, BatchSampler)
        for clone in (pickle.loads(pickle.dumps(sampler)), copy.deepcopy(sampler), copy.copy(sampler)):
            assert type(clone) is BatchSampler and clone.batch_size == 2 and len(list(clone)) == 20
        buffer = io.BytesIO()
        torch.save(sampler, buffer)
        buffer.seek(0)
        assert type(torch.load(buffer, weights_only=False)) is BatchSampler
    finally:
        tq_data._swap_out(sampler)
    assert type(sampler) is BatchSampler


def test_iterating_the_unwrapped_loader_keeps_nothing_and_does_not_shift_the_wrapper():
    session = FakeSession()
    loader = _loader(shuffle=True)
    watched = WatchedLoader(session, loader)
    for _ in loader:  # the original loader, after wrapping: 20 index batches
        pass
    held = [v for obj in (watched, loader.batch_sampler) for v in vars(obj).values()
            if isinstance(v, collections.deque)]
    assert sum(len(q) for q in held) == 0
    it = iter(watched)
    first = next(it)
    drawn = [_indices(first)]
    for _ in loader:  # a whole unwrapped pass in the middle of a watched one
        pass
    drawn += [_indices(b) for b in it]
    assert session.noted == drawn


@pytest.mark.parametrize("workers", [0, 2])
def test_two_iterators_of_one_watched_loader_each_note_their_own_batches(workers):
    session = FakeSession()
    watched = WatchedLoader(session, _loader(shuffle=True, num_workers=workers))
    a, b = iter(watched), iter(watched)
    drawn = []
    for _ in range(5):
        drawn.append(_indices(next(a)))
        drawn.append(_indices(next(b)))
    assert session.noted == drawn


def test_a_second_wrap_does_not_take_the_first_wrappers_batches():
    loader = _loader(shuffle=True)
    s1, s2 = FakeSession(), FakeSession()
    w1 = WatchedLoader(s1, loader)
    w2 = WatchedLoader(s2, loader)
    drawn1 = [_indices(b) for b in w1]
    drawn2 = [_indices(b) for b in w2]
    assert s1.noted == drawn1 and s2.noted == drawn2
    assert type(loader.batch_sampler) is BatchSampler


@pytest.mark.parametrize("workers", [0, 2])
def test_an_early_exit_then_a_new_pass_notes_the_new_passes_batches(workers):
    session = FakeSession()
    watched = WatchedLoader(session, _loader(shuffle=True, num_workers=workers))
    drawn = []
    for i, batch in enumerate(watched):
        drawn.append(_indices(batch))
        if i == 1:
            break
    drawn += [_indices(b) for b in watched]
    assert session.noted == drawn


def test_spawn_workers_with_a_watched_loader():
    session = FakeSession()
    loader = _loader(shuffle=True, num_workers=2, multiprocessing_context="spawn", persistent_workers=True)
    watched = WatchedLoader(session, loader)
    drawn = []
    for _ in range(2):
        drawn += [_indices(b) for b in watched]
    assert session.noted == drawn and len(drawn) == 40
    assert type(loader.batch_sampler) is BatchSampler and _pickles(loader.batch_sampler)


class GeneratorLoader(DataLoader):
    """A loader whose `__iter__` is a generator that runs one batch ahead, as Accelerate's
    `DataLoaderShard` does: the batch sampler is iterated only at the first `next()`."""

    def __iter__(self):
        inner = super().__iter__()
        try:
            current = next(inner)
        except StopIteration:
            return
        for following in inner:
            yield current
            current = following
        yield current


def test_a_loader_whose_iter_is_a_generator_is_noted():
    session = FakeSession()
    loader = GeneratorLoader(DATA40, batch_size=4, shuffle=True)
    watched = WatchedLoader(session, loader)
    drawn = [_indices(b) for b in watched]
    assert session.noted == drawn and len(drawn) == 10


class DrawAllFirst(DataLoader):
    """A loader that draws every index batch before it yields the first batch."""

    def __iter__(self):
        batches = list(self.batch_sampler)
        for indices in batches:
            yield self.collate_fn([self.dataset[i] for i in indices])


def test_indices_drawn_far_ahead_are_noted_as_unknown_not_shifted(monkeypatch):
    monkeypatch.setattr(tq_data, "_FIFO_LIMIT", 3, raising=False)
    session = FakeSession()
    watched = WatchedLoader(session, DrawAllFirst(DATA40, batch_size=4, shuffle=True))
    batches = [b for b in watched]
    assert len(batches) == 10 and session.noted == [None] * 10


def test_a_watched_loader_after_the_session_closed_passes_batches_through(tmp_path):
    model = nn.Linear(1, 1)
    session = tensorquill.init(model, run_id="u3", sink=tensorquill.JsonlSink(tmp_path / "r.jsonl"),
                               heartbeat_seconds=None, stack_seconds=None)
    loader = _loader()
    watched = session.watch_loader(loader)
    session.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batches = [_indices(b) for b in watched]  # raised RuntimeWarning before
    assert len(batches) == 20 and type(loader.batch_sampler) is BatchSampler


# --- C18 ------------------------------------------------------------------------------------------


def test_a_failure_to_note_a_batch_is_not_raised_under_warnings_as_errors(capsys):
    session = FakeSession()
    watched = WatchedLoader(session, _loader(), ids=lambda batch: 1 / 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batches = [b for b in watched]
    assert len(batches) == 20
    assert "could not note a batch" in capsys.readouterr().err


class HFLike(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = Attn()

    def forward(self, x):
        return self.attn(x)


def _llava(top, text, vision):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlavaConfig()
    config._attn_implementation = {"": top, "text_config": text, "vision_config": vision}
    return config


def _settings(config):
    return (config._attn_implementation, config.text_config._attn_implementation,
            config.vision_config._attn_implementation)


@pytest.mark.parametrize("before", [
    ("flash_attention_2", "flash_attention_2", "sdpa"),
    ("flash_attention_2", "flash_attention_2", "eager"),
    ("sdpa", "flash_attention_2", "eager"),
])
def test_a_composite_configs_parts_are_as_they_were_after_a_probe(before):
    config = _llava(*before)
    assert _settings(config) == before
    model = HFLike(config)
    seen = []
    result = tq_probe.run_probe(model, lambda m: (seen.append(_settings(config)), m(torch.randn(1, 48))),
                                tq_probe.ProbeState(), CPU)
    assert _settings(config) == before
    assert seen == [tuple("sdpa" if v.startswith("flash") else v for v in before)]
    assert result["attention_note"] == "FlashAttention switched to SDPA for the probe"


def test_a_probe_leaves_pythons_and_numpys_random_state_as_it_found_them():
    def draws():
        return random.random(), float(np.random.rand()), float(torch.rand(()))

    def fn(m):
        random.random()
        np.random.rand(3)
        torch.rand(5)
        m(torch.randn(1, 48))

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = draws()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    tq_probe.run_probe(Blocks(Attn()), fn, tq_probe.ProbeState(), CPU)
    assert draws() == expected
