"""Dataloader wait time (`datawait.py`): what the timer counts and what it must not, through
the explicit wrapper, `watch_loaders`, and the Accelerate, Hugging Face and Lightning adapters.

Each slow dataset sleeps `FETCH` seconds per batch; each device move sleeps `MOVE`. A
`data_wait` near `FETCH` and far below `FETCH + MOVE` shows the batch fetch is timed and the
move to the device is not (a blocking move waits behind the GPU's queued work)."""

from __future__ import annotations

import threading
import time

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter

import tensorquill
from tensorquill import datawait

FETCH = 0.02
MOVE = 0.06


def _vector(i):
    return torch.randn(4)


def _pair(i):
    return {"x": torch.randn(4), "labels": torch.randn(())}


@pytest.fixture(autouse=True)
def _no_session_left_watching():
    yield
    for session in list(datawait._watching):
        datawait.unwatch(session)


class Slow(Dataset):
    def __init__(self, n=16, item=_vector):
        self.n, self.item = n, item

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(FETCH)
        return self.item(i)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _waits(records):
    return [r.get("data_wait") for r in records]


def _assert_fetch_timed(records, batches, fewest=None):
    """Every fetch counted once and no move: the waits add up to `batches` fetches (at least
    `fewest`, when the loader's fetch ahead may fall outside the last recorded step). A loader
    that fetches one batch ahead (Accelerate, Lightning) moves a fetch to the step before."""
    waits = [r["data_wait"] for r in records]
    fewest = batches if fewest is None else fewest
    assert fewest * FETCH * 0.9 <= sum(waits) < batches * (FETCH + MOVE / 2), waits
    times = [r.get("step_time") for r in records]
    assert times[0] is None and all(t >= w for t, w in zip(times[1:], waits[1:]))


def test_off_by_default_and_the_dataloader_is_left_alone():
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, run_id="dw")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for x in DataLoader(torch.randn(8, 4), batch_size=2):
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    assert "__next__" in vars(_BaseDataLoaderIter) and vars(_BaseDataLoaderIter)["__next__"] is not datawait._timed_next
    assert _waits(sink.records) == [None] * 4 and not any("step_time" in r for r in sink.records)


def test_watch_loaders_times_the_fetch_not_the_move_and_restores_on_close():
    original = vars(_BaseDataLoaderIter)["__next__"]
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, run_id="dw", data_wait=True)
    assert vars(_BaseDataLoaderIter)["__next__"] is datawait._timed_next
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for x in DataLoader(Slow(8), batch_size=1):
        time.sleep(MOVE)  # the move to the device, after the loader handed the batch over
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    assert vars(_BaseDataLoaderIter)["__next__"] is original
    _assert_fetch_timed(sink.records, 8)


def test_only_the_sessions_thread_and_the_train_phase_count():
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=8, run_id="dw", data_wait=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # A loader drained on another thread, such as a background prefetcher, is not this thread waiting.
    other = threading.Thread(target=lambda: list(DataLoader(Slow(4), batch_size=1)))
    other.start()
    other.join()
    with session.phase("eval"):
        list(DataLoader(Slow(4), batch_size=1))  # an evaluation loader
    batches_in, spent = iter(DataLoader(torch.randn(2, 4), batch_size=1)), []
    for _ in range(2):
        start = time.perf_counter()
        x = next(batches_in)
        spent.append(time.perf_counter() - start)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    # Each step's wait is inside this loop's own next(); the eight slow fetches above, if they
    # counted, would add at least 8 * FETCH to the first step's.
    assert all(w <= s for w, s in zip(_waits(sink.records), spent)), (_waits(sink.records), spent)


def test_time_data_wraps_any_iterable_and_is_not_counted_twice():
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, run_id="dw", data_wait=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(Slow(4), batch_size=1)
    timed = session.time_data(loader)
    assert len(timed) == 4 and timed.dataset is loader.dataset
    batches_in, spent = iter(timed), []  # a DataLoader inside the wrapper, with watch_loaders on too
    for _ in range(4):
        start = time.perf_counter()
        x = next(batches_in)
        spent.append(time.perf_counter() - start)
        time.sleep(MOVE)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.flush()
    deadline = time.monotonic() + 10
    while len(sink.records) < 4:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    # Each fetch once: counted by both timers, a wait would be about twice the time this loop
    # itself spent in that next().
    assert all(w <= s for w, s in zip(_waits(sink.records), spent)), (_waits(sink.records), spent)

    def plain():  # not a DataLoader at all
        for _ in range(4):
            time.sleep(FETCH)
            yield torch.randn(1, 4)

    for x in session.time_data(plain()):
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    _assert_fetch_timed(sink.records, 8)


def test_two_sessions_and_a_closed_one():
    a, b = _model(), _model()
    sink_a, sink_b = tensorquill.MemorySink(), tensorquill.MemorySink()
    session_a = tensorquill.init(a, sink=sink_a, every=2, run_id="a", data_wait=True)
    session_b = tensorquill.init(b, sink=sink_b, every=2, run_id="b", data_wait=True)
    session_b.close()
    assert vars(_BaseDataLoaderIter)["__next__"] is datawait._timed_next  # a still watches
    optimizer = torch.optim.SGD(a.parameters(), lr=0.1)
    for x in DataLoader(Slow(2), batch_size=1):
        a(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session_a.close()
    assert vars(_BaseDataLoaderIter)["__next__"] is not datawait._timed_next
    assert all(w >= FETCH * 0.9 for w in _waits(sink_a.records))


def test_with_loader_workers():
    """A worker fetches ahead (two batches by default) while the loop runs the step before, so
    a batch can be waiting when the loop asks for it and that `data_wait` is near zero: the loop
    did not wait. What one worker cannot do is hand over batch `k` sooner than `k` fetches after
    the loader started, so the waits make up whatever the loop's own time does not cover."""
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=8, run_id="dw", data_wait=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = 8
    started = time.perf_counter()
    batches_in = iter(DataLoader(Slow(batches), batch_size=1, num_workers=1))
    asked, handed = [], []  # this loop's own stopwatch around each next()
    for _ in range(batches):
        asked.append(time.perf_counter())
        x = next(batches_in)
        handed.append(time.perf_counter())
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    waits = _waits(sink.records)
    assert len(waits) == batches
    # Each wait is inside the loop's own next(), and is measured: a timer that is not running
    # reads zero, while a real next() through a worker's queue always takes some time.
    assert all(0 < w <= h - a for w, a, h in zip(waits, asked, handed)), waits
    # The worker fetches one batch at a time, each at least FETCH, so the last batch is handed
    # over no sooner than `batches * FETCH` after `started`. Until then the loop was either in
    # next(), counted in the waits, or busy outside it: building the iterator and its steps.
    busy = (asked[0] - started) + sum(a - h for h, a in zip(handed, asked[1:]))
    assert sum(waits) >= batches * FETCH * 0.9 - busy, (waits, busy)


# --- adapters ------------------------------------------------------------------------------


def test_accelerate_times_the_shards_fetch_not_send_to_device(monkeypatch):
    pytest.importorskip("accelerate")
    import accelerate.data_loader
    from accelerate import Accelerator
    from tensorquill.accelerator import watch_accelerator

    original = accelerate.data_loader.send_to_device

    def slow_move(*args, **kwargs):
        time.sleep(MOVE)
        return original(*args, **kwargs)

    monkeypatch.setattr(accelerate.data_loader, "send_to_device", slow_move)
    accelerator = Accelerator(cpu=True)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, DataLoader(Slow(6), batch_size=1))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=3, run_id="acc")
    watch_accelerator(session)
    for x in loader:
        accelerator.backward(model(x).sum())
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    _assert_fetch_timed(sink.records, 6)


def test_hugging_face_trainer_records_the_wait(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    import accelerate.data_loader
    from tensorquill.huggingface import TensorQuillCallback

    original = accelerate.data_loader.send_to_device

    def slow_move(*args, **kwargs):
        time.sleep(MOVE)
        return original(*args, **kwargs)

    monkeypatch.setattr(accelerate.data_loader, "send_to_device", slow_move)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _model()

        def forward(self, x, labels=None):
            return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

    sink = tensorquill.MemorySink()
    callback = TensorQuillCallback(sink=sink, capture="full")
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=1, report_to="none",
        save_strategy="no", disable_tqdm=True, use_cpu=True, gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
    )
    data = Slow(16, item=_pair)
    transformers.Trainer(model=Model(), args=args, train_dataset=data, callbacks=[callback]).train()
    assert vars(_BaseDataLoaderIter)["__next__"] is not datawait._timed_next
    _assert_fetch_timed(sink.records, 8)  # four steps of two microbatches


def test_hugging_face_data_wait_can_be_turned_off(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")  # the huggingface extra needs both
    from tensorquill.huggingface import TensorQuillCallback

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _model()

        def forward(self, x, labels=None):
            return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

    sink = tensorquill.MemorySink()
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), max_steps=2, per_device_train_batch_size=1, report_to="none",
        save_strategy="no", disable_tqdm=True, use_cpu=True,
    )
    data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(4)]
    callback = TensorQuillCallback(sink=sink, capture="full", data_wait=False)
    transformers.Trainer(model=Model(), args=args, train_dataset=data, callbacks=[callback]).train()
    assert _waits(sink.records) == [None, None]


def test_lightning_times_the_fetch_not_the_transfer(tmp_path, monkeypatch):
    L = pytest.importorskip("lightning.pytorch")
    from tensorquill.lightning import TensorQuillCallback

    monkeypatch.chdir(tmp_path)

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = _model()

        def training_step(self, batch, batch_idx):
            return self.net(batch).pow(2).mean()

        def validation_step(self, batch, batch_idx):
            self.net(batch)

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            time.sleep(MOVE)
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

        def train_dataloader(self):
            return DataLoader(Slow(16), batch_size=1)

        def val_dataloader(self):
            return DataLoader(Slow(2), batch_size=1)

    sink = tensorquill.MemorySink()
    trainer = L.Trainer(
        default_root_dir=tmp_path, max_steps=4, callbacks=[TensorQuillCallback(sink=sink, capture="full")],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, logger=False,
        accelerator="cpu", devices=1, val_check_interval=2, num_sanity_val_steps=2,
    )
    trainer.fit(Module())
    assert [r["step"] for r in sink.records] == [0, 1, 2, 3]
    # Validation (sanity check and every 2 steps) fetches its own slow batches, in the eval
    # phase; Lightning fetches one training batch ahead, so a fifth fetch lands in the last step
    # on macOS, and after it (not recorded) on Linux CI: four or five fetches, never a move.
    _assert_fetch_timed(sink.records, 5, fewest=4)


def test_another_wrapper_on_top_is_never_broken():
    """A library that wraps `__next__` after the timer (a data-progress hook, say) keeps working
    after the session closes, and a later session still times its fetches."""
    original = vars(_BaseDataLoaderIter)["__next__"]
    model = _model()
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), run_id="a", data_wait=True)
    below = vars(_BaseDataLoaderIter)["__next__"]
    seen = []

    def theirs(self):
        seen.append(1)
        return below(self)

    _BaseDataLoaderIter.__next__ = theirs
    try:
        session.close()
        assert vars(_BaseDataLoaderIter)["__next__"] is theirs  # theirs is not taken out
        assert len(list(DataLoader(torch.randn(3, 4), batch_size=1))) == 3 and len(seen) == 4
        sink = tensorquill.MemorySink()
        again = tensorquill.init(model, sink=sink, every=2, run_id="b", data_wait=True)
        assert vars(_BaseDataLoaderIter)["__next__"] is theirs
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for x in DataLoader(Slow(2), batch_size=1):
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        again.close()
        assert all(w >= FETCH * 0.9 for w in _waits(sink.records)), _waits(sink.records)
    finally:
        _BaseDataLoaderIter.__next__ = original  # theirs removed, back to what it first saw
    # With theirs gone and the timer under it gone too, a new session installs the timer again.
    sink = tensorquill.MemorySink()
    third = tensorquill.init(model, sink=sink, every=1, run_id="c", data_wait=True)
    assert vars(_BaseDataLoaderIter)["__next__"] is datawait._timed_next
    third.close()
    assert vars(_BaseDataLoaderIter)["__next__"] is original
