"""Forward passes that are not training leave no trace in any step's record.

The capture hooks fire on every forward, and their buffers are emptied only when an optimizer
step is committed, so an evaluation between two steps used to be added to the next step's
record: its tokens to the router counts, its largest activation to `abs_max`, and its mean,
std and zero fraction to a layer's statistics when it came after the step's last training
forward. A forward is now left out when the hooked module is in eval mode and gradients are
off (how Lightning, the Hugging Face `Trainer` and most loops evaluate), or under
`session.phase("eval")` when either one is (`capture._evaluating`).

Each test runs the same training twice, with and without the evaluation, and the records
must be equal: whatever leaked would show as a difference.
"""

from __future__ import annotations

import contextlib
import json
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.capture import Router

EXPERTS, TOP_K = 4, 2
VOLATILE = {"attempt", "ts", "system", "wall_time", "step_time", "host_ms", "timing"}
COMPILED = pytest.mark.parametrize("compile_backend", [None, "eager"], ids=["eager", "compiled"])


class Net(nn.Module):
    """A router, a body and a layer that runs only in evaluation (`evalhead`), so a layer that
    no training forward reaches would show up in a record if an evaluation leaked."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(8, 16)
        self.router = nn.Linear(16, EXPERTS)
        self.body = nn.Linear(16, 16)
        self.evalhead = nn.Linear(16, 16)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.embed(x)
        self.router(h)
        h = torch.relu(self.body(h))
        if not self.training:
            h = self.evalhead(h)
        return self.head(h).pow(2).mean()


LAYERS = ("router", "body", "evalhead")


def _train(tmp_path, name, evaluate=None, where="between", steps=4, compile_backend=None, accumulate=1):
    """Train `steps` steps; `evaluate(session, run)` runs at `where`: "between" steps 1 and 2,
    "before" the first step, or "inside" step 2, after its backward and before its optimizer
    step. Returns the records without the keys that differ from run to run."""
    torch.manual_seed(0)
    model = Net()
    path = tmp_path / f"{name}.jsonl"
    session = tensorquill.init(
        model, run_id=name, sink=tensorquill.JsonlSink(path), routers={"router": Router(EXPERTS, TOP_K)},
        layers=lambda layer, _module: layer in LAYERS, every=2,
    )
    run = torch.compile(model, backend=compile_backend) if compile_backend else model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.Generator().manual_seed(1)  # the evaluation draws from its own generator
    if evaluate is not None and where == "before":
        evaluate(session, run)
    for step in range(steps):
        if evaluate is not None and where == "between" and step == 2:
            evaluate(session, run)
        optimizer.zero_grad()
        for _ in range(accumulate):
            run(torch.randn(2, 5, 8, generator=data)).backward()  # 10 tokens a forward
        if evaluate is not None and where == "inside" and step == 2:
            evaluate(session, run)
        optimizer.step()
    session.close()
    return [{k: v for k, v in json.loads(line).items() if k not in VOLATILE} for line in path.read_text().splitlines()]


def _eval(model_mode="eval", grad="no_grad", phase=False, batches=3, scale=100.0):
    """An evaluation: `batches` forwards of large inputs (so a leaked `abs_max` shows)."""

    def evaluate(session, run):
        model = run._orig_mod if hasattr(run, "_orig_mod") else run
        with contextlib.ExitStack() as stack:
            if phase:
                stack.enter_context(session.phase("eval"))
            if grad == "no_grad":
                stack.enter_context(torch.no_grad())
            elif grad == "inference_mode":
                stack.enter_context(torch.inference_mode())
            if model_mode == "eval":
                model.eval()
            for _ in range(batches):
                run(torch.randn(2, 5, 8) * scale)
            model.train()

    return evaluate


def _routed(records, name="router"):
    return [r["moe_stats"][name]["routed_tokens"] for r in records]


@COMPILED
@pytest.mark.parametrize(
    "evaluate",
    [_eval(phase=True), _eval(), _eval(grad="inference_mode")],
    ids=["phase-eval-no_grad", "eval-no_grad", "eval-inference_mode"],
)
def test_an_evaluation_between_steps_is_in_no_record(tmp_path, evaluate, compile_backend):
    """The reproduction: three evaluation batches between steps 1 and 2 made step 2's router
    counts 40 tokens instead of 10, and its `abs_max` that of the evaluation's inputs."""
    without = _train(tmp_path, "without", compile_backend=compile_backend)
    with_eval = _train(tmp_path, "with", evaluate, compile_backend=compile_backend)
    assert _routed(without) == [10 * TOP_K] * 4
    assert with_eval == [{**r, "run_id": "with"} for r in without]


@COMPILED
@pytest.mark.parametrize("where", ["inside", "before"])
def test_an_evaluation_inside_a_step_or_before_the_first_is_in_no_record(tmp_path, where, compile_backend):
    """Inside a step, after its last training forward, the evaluation's mean, std and zero
    fraction would have replaced the step's (they are the last forward's); before the first
    step (Lightning's sanity check) it would have gone into step 0. A layer that runs only in
    evaluation, `evalhead`, would have had an entry."""
    without = _train(tmp_path, "without", compile_backend=compile_backend)
    with_eval = _train(tmp_path, "with", _eval(), where=where, compile_backend=compile_backend)
    assert all("evalhead" not in r["layers"] for r in with_eval)
    assert with_eval == [{**r, "run_id": "with"} for r in without]


@COMPILED
def test_gradient_accumulation_still_adds_its_microbatches(tmp_path, compile_backend):
    records = _train(tmp_path, "accumulated", _eval(), accumulate=3, compile_backend=compile_backend)
    assert _routed(records) == [3 * 10 * TOP_K] * 4


@COMPILED
@pytest.mark.parametrize(
    "evaluate",
    [_eval(model_mode="train", grad="no_grad", phase=True), _eval(model_mode="eval", grad=None, phase=True)],
    ids=["train-mode-no_grad", "eval-mode-with-grad"],
)
def test_under_the_eval_phase_one_signal_is_enough(tmp_path, evaluate, compile_backend):
    without = _train(tmp_path, "without", compile_backend=compile_backend)
    with_eval = _train(tmp_path, "with", evaluate, compile_backend=compile_backend)
    assert with_eval == [{**r, "run_id": "with"} for r in without]


@pytest.mark.parametrize(
    ("evaluate", "why"),
    [
        (_eval(model_mode="train", grad=None, phase=True), "the phase alone: a phase left behind must not drop a step"),
        (_eval(model_mode="eval", grad=None), "eval mode alone: training can keep modules in eval mode"),
        (_eval(model_mode="train", grad="no_grad"), "no gradients alone: reentrant checkpointing trains so"),
    ],
    ids=["phase-only", "eval-mode-only", "no_grad-only"],
)
def test_one_signal_alone_without_the_phase_still_counts(tmp_path, evaluate, why):
    """The rule's edges, as decided: each of these is counted, as before. An evaluation written
    this way still reaches the next step's record (said in the README)."""
    records = _train(tmp_path, "one", evaluate)
    assert _routed(records) == [10 * TOP_K] * 2 + [40 * TOP_K, 10 * TOP_K], why


def test_a_hooked_module_kept_in_eval_mode_during_training_is_still_recorded(tmp_path):
    torch.manual_seed(0)
    model = Net()
    model.router.eval()  # frozen, say: still in every training forward
    session = tensorquill.init(model, sink=(sink := tensorquill.MemorySink()), routers={"router": Router(EXPERTS, TOP_K)},
                               layers=lambda layer, _module: layer in LAYERS, every=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(2, 5, 8)).backward()
        optimizer.step()
    session.close()
    assert _routed(sink.records) == [10 * TOP_K] * 2 and all("router" in r["layers"] for r in sink.records)


def test_an_evaluation_batch_cannot_refuse_a_router(tmp_path):
    """Nothing in an evaluation forward is read, so an output the guard would refuse there
    (here per-expert counts first, in eval mode only) neither refuses nor warns."""

    class Shifty(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Linear(16, EXPERTS)

        def forward(self, x):
            logits = self.w(x)
            return logits if self.training else (logits.argmax(-1).flatten().bincount(minlength=EXPERTS), logits)

    torch.manual_seed(0)
    model = Net()
    model.router = Shifty()
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        session = tensorquill.init(model, sink=sink, routers={"router": Router(EXPERTS, TOP_K)}, every=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(3):
            if step == 1:
                _eval()(session, model)
            optimizer.zero_grad()
            model(torch.randn(2, 5, 8)).backward()
            optimizer.step()
        session.close()
    assert not any("moe_refused" in r for r in sink.records) and _routed(sink.records) == [10 * TOP_K] * 3


# --- The adapters that run an evaluation ---------------------------------------------------------


def test_lightning_validation_and_its_sanity_check_are_in_no_record(tmp_path, monkeypatch):
    L = pytest.importorskip("lightning.pytorch")
    from tensorquill.lightning import TensorQuillCallback

    class Module(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = Net()
            self.validated = 0

        def training_step(self, batch, batch_idx):
            return self.net(batch)

        def validation_step(self, batch, batch_idx):
            self.validated += 1
            return self.net(batch * 100)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(torch.randn(64, 5, 8, generator=torch.Generator().manual_seed(0)), batch_size=2)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(torch.randn(6, 5, 8, generator=torch.Generator().manual_seed(1)), batch_size=2)

    monkeypatch.chdir(tmp_path)
    sink = tensorquill.MemorySink()
    module = Module()
    trainer = L.Trainer(
        default_root_dir=tmp_path, max_steps=6, val_check_interval=2, num_sanity_val_steps=2, logger=False,
        callbacks=[TensorQuillCallback(sink=sink, routers={"net.router": Router(EXPERTS, TOP_K)},
                                       layers=lambda name, _m: name in ("net.body", "net.evalhead"))],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, accelerator="cpu", devices=1,
    )
    trainer.fit(module)
    assert module.validated >= 2 + 3 * 2  # the sanity check and at least two validations ran
    records = sink.records
    assert [r["step"] for r in records] == list(range(6))
    assert _routed(records, "net.router") == [10 * TOP_K] * 6  # 2 × 5 tokens a training batch, nothing else
    assert all(list(r["layers"]) == ["net.body"] for r in records)  # `evalhead` runs only in validation


def test_the_hugging_face_trainer_evaluation_is_in_no_record(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from tensorquill.huggingface import TensorQuillCallback

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = Net()

        def forward(self, x, labels=None):
            return {"loss": self.net(x if self.training else x * 100)}

    data = torch.Generator().manual_seed(0)
    train = [{"x": torch.randn(5, 8, generator=data), "labels": torch.zeros(())} for _ in range(16)]
    held_out = [{"x": torch.randn(5, 8, generator=data), "labels": torch.zeros(())} for _ in range(6)]
    sink = tensorquill.MemorySink()
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=2, per_device_eval_batch_size=2,
        eval_strategy="steps", eval_steps=2, report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True,
        learning_rate=0.01,
    )
    callback = TensorQuillCallback(sink=sink, routers={"net.router": Router(EXPERTS, TOP_K)},
                                   layers=lambda name, _m: name in ("net.body", "net.evalhead"))
    trainer = transformers.Trainer(model=Model(), args=args, train_dataset=train, eval_dataset=held_out,
                                   callbacks=[callback])
    trainer.train()
    assert sum("eval_loss" in entry for entry in trainer.state.log_history) == 2
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    assert _routed(records, "net.router") == [10 * TOP_K] * 4
    assert all(list(r["layers"]) == ["net.body"] for r in records)


def test_an_accelerate_loop_that_evaluates_with_no_grad_leaves_it_out(tmp_path):
    pytest.importorskip("accelerate")
    from accelerate import Accelerator
    from tensorquill.accelerator import watch_accelerator

    torch.manual_seed(0)
    model = Net()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, routers={"router": Router(EXPERTS, TOP_K)}, every=1)
    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    watch = watch_accelerator(session, data_wait=False)
    for step in range(4):
        loss = model(torch.randn(2, 5, 8))
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if step == 1:  # as Accelerate's examples evaluate: eval mode and no gradients, no phase
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    model(torch.randn(2, 5, 8) * 100)
            model.train()
    watch.close()
    session.close()
    assert _routed(sink.records) == [10 * TOP_K] * 4
