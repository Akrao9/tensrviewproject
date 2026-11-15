import pytest
import torch
from torch import nn

import tensorquill


class Beat:
    """Stands in for the heartbeat thread: records the phases it is told."""

    def __init__(self):
        self.phases = []
        self.counter = None

    def set_phase(self, phase):
        self.phases.append(phase)


def _session(beat=None):
    sink = tensorquill.MemorySink()
    if beat is not None:
        sink.heartbeat = beat
    return tensorquill.init(nn.Linear(2, 2), sink=sink)


def test_phase_marks_a_stretch_and_goes_back_even_when_the_block_raises():
    beat = Beat()
    session = _session(beat)
    with session.phase("eval"):
        assert session._phase == "eval"
        with session.phase("checkpoint"):
            pass
        assert session._phase == "eval"
    with pytest.raises(RuntimeError):
        with session.phase("checkpoint"):
            raise RuntimeError("the save failed")
    assert session._phase == "train"
    assert beat.phases == ["eval", "checkpoint", "eval", "train", "checkpoint", "train"]
    session.close()


def test_phase_does_nothing_without_a_server():
    session = _session()  # no endpoint: no heartbeat, nothing to tell
    with session.phase("eval"):
        pass
    session.set_phase("checkpoint")
    assert session._phase == "checkpoint"
    session.close()


def test_the_huggingface_callback_marks_evaluation_and_checkpoints():
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from tensorquill.huggingface import TensorQuillCallback

    beat = Beat()
    callback = TensorQuillCallback()
    callback.session = _session(beat)
    flow = SimpleNamespace(should_evaluate=False, should_save=False)
    callback.on_step_end(None, None, flow)
    assert beat.phases == []  # an ordinary step is not a pause
    flow.should_evaluate = flow.should_save = True  # the flow callback asks for both at this step
    callback.on_step_end(None, None, flow)
    callback.on_evaluate(None, None, flow)  # evaluation done, the save is still to come
    flow.should_evaluate = False
    callback.on_save(None, None, flow)
    assert beat.phases == ["eval", "checkpoint", "train"]
    flow.should_evaluate, flow.should_save = False, True
    callback.on_epoch_end(None, None, flow)
    assert beat.phases[-1] == "checkpoint"
    callback.session.close()


def test_the_lightning_callback_marks_validation():
    L = pytest.importorskip("lightning")
    from tensorquill.lightning import TensorQuillCallback

    class Model(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def training_step(self, batch, index):
            return self.net(batch[0]).pow(2).mean()

        def validation_step(self, batch, index):
            self.net(batch[0]).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    beat = Beat()
    sink = tensorquill.MemorySink()
    sink.heartbeat = beat
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(16, 4)), batch_size=4)
    trainer = L.Trainer(max_epochs=2, callbacks=[TensorQuillCallback(sink=sink)], logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                        accelerator="cpu", num_sanity_val_steps=0)
    trainer.fit(Model(), data, data)
    assert beat.phases == ["eval", "train", "eval", "train"]


def test_a_committed_step_ends_a_pause_and_ticks_count_as_work():
    beat = Beat()
    sink = tensorquill.MemorySink()
    sink.heartbeat = beat
    model = nn.Linear(2, 2)
    session = tensorquill.init(model, sink=sink, every=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # The counter the heartbeat reports moves with steps and with ticks.
    assert beat.counter() == 0
    session.tick()
    session.tick()
    assert beat.counter() == 2
    session.set_phase("eval")  # a callback announced an evaluation and never announced its end
    model(torch.randn(1, 2)).sum().backward()
    optimizer.step()
    assert session._phase == "train" and beat.phases == ["eval", "train"]
    assert beat.counter() == 3
    session.close()


def test_a_real_trainer_announces_each_evaluation_and_save(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from tensorquill.huggingface import TensorQuillCallback

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def forward(self, x, labels=None):
            return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

    torch.manual_seed(0)
    data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(32)]
    beat = Beat()
    sink = tensorquill.MemorySink()
    sink.heartbeat = beat
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path), max_steps=8, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        report_to="none", disable_tqdm=True, use_cpu=True, learning_rate=0.01,
        eval_strategy="steps", eval_steps=3, save_strategy="steps", save_steps=6, logging_steps=100,
    )
    trainer = transformers.Trainer(model=Regressor(), args=args, train_dataset=data, eval_dataset=data[:8],
                                   callbacks=[TensorQuillCallback(sink=sink)])
    trainer.train()
    # Evaluation after step 3, evaluation after step 6 followed by the save, evaluation after step 8's
    # end is not scheduled. Whatever the exact order, every pause is entered, ended by a step or by
    # its own end event, and the run ends in training.
    assert "eval" in beat.phases and "checkpoint" in beat.phases
    assert beat.phases[-1] == "train"
    assert all(a != b for a, b in zip(beat.phases, beat.phases[1:]))  # only changes are sent
    print("PHASES", beat.phases)
