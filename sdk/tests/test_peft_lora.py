"""A LoRA model made by the real `peft` library, trained for a few steps on CPU.

The other tests of clipping and frozen parameters use a hand-made model whose first parameter is
frozen. Here the model is a tiny Llama built from a config (nothing is downloaded) with a real
adapter from `peft.get_peft_model`, so its first parameter (the embedding) is frozen because
`peft` froze it, and its trainable parameters are `peft`'s own `lora_A` and `lora_B` weights.

What is held here is what the README promises: the parameters recorded are those trainable at
`init`, and one unfrozen later is recorded from that step with a warning; a `clip_grad_norm_` call
counts when any parameter given is a trainable one; a healthy run raises nothing; the probe reads
only a plain `nn.Linear` head and refuses any other without raising; and a checkpoint that holds
only the adapter names nothing as missing. How an adapter checkpoint's tensors should be compared
with the model is not decided, so nothing is asserted about whether they are.

Each test takes `transformers` and `peft` with `pytest.importorskip` inside the test, so the file
collects everywhere and skips test by test where either is absent.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import evidence as ev

VOCAB = 32
TARGETS = ["q_proj", "v_proj"]
QUIET = dict(heartbeat_seconds=None, stack_seconds=None)


def _libraries(monkeypatch):
    """`transformers` and `peft`, or a skip. Nothing may be fetched from the Hub."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    return transformers, peft


def _lora_model(transformers, peft, seed: int = 0, **lora):
    config = transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=2, vocab_size=VOCAB)
    torch.manual_seed(seed)
    options = dict(r=4, lora_alpha=8, target_modules=TARGETS, lora_dropout=0.0)
    options.update(lora)
    return peft.get_peft_model(transformers.LlamaForCausalLM(config), peft.LoraConfig(**options))


def _trainable(model) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _ids(i: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (4, 8), generator=torch.Generator().manual_seed(i))


def _train(model, optimizer, steps: int, *, first: int = 0, clip=None, before_step=None) -> list[float]:
    """`steps` optimizer steps; returns what each `clip(model)` call returned."""
    returned = []
    for i in range(first, first + steps):
        if before_step is not None:
            before_step(i)
        ids = _ids(i)
        model(input_ids=ids, labels=ids).loss.backward()
        if clip is not None:
            returned.append(float(clip(model)))
        optimizer.step()
        optimizer.zero_grad()
    return returned


def _adamw(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)


# ---- which parameters the record names ---------------------------------------------------------


def test_the_record_names_the_adapter_weights_and_no_frozen_base_weight(monkeypatch):
    transformers, peft = _libraries(monkeypatch)
    model = _lora_model(transformers, peft)
    first_name, first = next(iter(model.named_parameters()))
    assert not first.requires_grad, f"{first_name} should be frozen by peft"
    trainable = set(_trainable(model))
    assert trainable and all(".lora_A." in n or ".lora_B." in n for n in trainable)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, run_id="peft-names", capture="full", weights=2, updates=True,
                               **QUIET)
    _train(model, _adamw(model), 4)
    session.close()
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    for record in records:
        assert set(record["grad_norms"]) == trainable
        assert set(record["update_rms"]) == trainable
    checked = [r for r in records if "weight_norms" in r]
    assert checked, "weights=2 should check the weights on two of four steps"
    for record in checked:
        assert set(record["weight_norms"]) == trainable
        # The one optimizer holds every trainable parameter.
        assert record["optimizer_ownership"]["state"] == "known"
        assert record["optimizer_ownership"]["unheld_count"] == 0


def test_a_base_weight_unfrozen_later_is_recorded_from_that_step_with_a_warning(monkeypatch):
    transformers, peft = _libraries(monkeypatch)
    model = _lora_model(transformers, peft)
    norm_name = next(n for n, _ in model.named_parameters() if n.endswith("model.norm.weight"))
    norm = model.get_parameter(norm_name)
    assert not norm.requires_grad
    optimizer = _adamw(model)
    sink = tensorquill.MemorySink()

    def unfreeze(i):
        if i == 4:
            norm.requires_grad_(True)
            optimizer.add_param_group({"params": [norm]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, run_id="peft-unfreeze", capture="full", **QUIET)
        _train(model, optimizer, 8, before_step=unfreeze)
        session.close()
    named = [norm_name in r["grad_norms"] for r in sink.records]
    assert named == [False] * 4 + [True] * 4
    assert sum(norm_name in str(w.message) for w in caught) == 1
    assert ev.build_evidence(sink.records)["incidents"] == []


# ---- clipping ----------------------------------------------------------------------------------


@pytest.mark.parametrize("given", ["trainable", "all"])
def test_the_clip_norm_is_recorded_when_the_first_parameter_is_frozen(monkeypatch, given):
    """`given="all"` is what the Hugging Face `Trainer` passes: `model.parameters()`, frozen first."""
    transformers, peft = _libraries(monkeypatch)
    model = _lora_model(transformers, peft)

    def clip(m):
        params = [p for p in m.parameters() if p.requires_grad] if given == "trainable" else m.parameters()
        return torch.nn.utils.clip_grad_norm_(params, 1.0)

    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, run_id="peft-clip", **QUIET)
    returned = _train(model, _adamw(model), 4, clip=clip)
    session.close()
    assert [r.get("clip_grad_norm") for r in sink.records] == pytest.approx(returned, rel=1e-6)
    assert all("clip_calls" not in r for r in sink.records)


# ---- a healthy run -----------------------------------------------------------------------------


def _healthy(model, steps: int = 14) -> list[dict]:
    sink = tensorquill.MemorySink()
    probe = _ids(1000)[:2]
    session = tensorquill.init(model, sink=sink, run_id="peft-healthy", capture="full", weights=2, updates=True,
                               probe=probe, probe_every=2, **QUIET)
    _train(model, _adamw(model), steps, clip=lambda m: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    session.close()
    return sink.records


def test_a_healthy_lora_run_raises_nothing(monkeypatch):
    """Fourteen steps: enough for `grad_spike`'s history, not for the activation or update rules,
    which are judged on what there is. Which kinds had enough history is in the lane's report;
    here only that each kind below was judged and that nothing was raised."""
    transformers, peft = _libraries(monkeypatch)
    records = _healthy(_lora_model(transformers, peft))
    assert len(records) == 14
    evidence = ev.build_evidence(records, detectors="python")
    assert evidence["incidents"] == []
    checks = evidence["kind_checks"]
    for kind in ("grad_spike", "non_finite", "repeated_steps", "stalled_weights", "activation_spike",
                 "attention_collapse"):
        assert checks[kind]["value"] == "checked", kind
    assert checks["grad_spike"]["enough_history"] is True
    assert evidence["run_state"]["state"] == "healthy"


def test_both_engines_raise_nothing_on_a_healthy_lora_run(monkeypatch):
    transformers, peft = _libraries(monkeypatch)
    pytest.importorskip("pyarrow")
    binary = os.environ.get("TENSORQUILL_SERVER_BIN")
    if not binary or not Path(binary).exists():
        pytest.skip("no tensorquill-server binary named by TENSORQUILL_SERVER_BIN")
    records = _healthy(_lora_model(transformers, peft))
    python = ev.build_evidence(records, detectors="python")
    rust = ev.build_evidence(records, detectors="rust", server_binary=binary)
    assert rust["detectors"] == "rust"
    assert python["incidents"] == [] and rust["incidents"] == []


# ---- the probe's head --------------------------------------------------------------------------


def test_the_probe_reads_a_plain_head_through_peft(monkeypatch):
    """With the adapter on the attention only, `get_output_embeddings()` is still the base model's
    plain `nn.Linear`, and the probe reads it on every probe."""
    transformers, peft = _libraries(monkeypatch)
    model = _lora_model(transformers, peft)
    assert type(model.get_output_embeddings()) is torch.nn.Linear
    evidence = ev.build_evidence(_healthy(model, steps=6))
    head = evidence["head_trends"]
    assert head["probes"] == 3 and head["read"] == 3
    assert head["module"].endswith("lm_head")
    assert "not_read" not in head


@pytest.mark.parametrize("wrapped", ["modules_to_save", "lora_on_head"])
def test_the_probe_refuses_a_head_peft_has_wrapped_and_training_goes_on(monkeypatch, wrapped):
    """`modules_to_save=["lm_head"]` puts the head in peft's `ModulesToSaveWrapper`; an adapter on
    `lm_head` makes it peft's own `Linear` layer, which is not `torch.nn.Linear`. Either is refused,
    not guessed, and the refusal raises nothing. The reason's words are not held here."""
    transformers, peft = _libraries(monkeypatch)
    lora = {"modules_to_save": ["lm_head"]} if wrapped == "modules_to_save" else {"target_modules": TARGETS + ["lm_head"]}
    model = _lora_model(transformers, peft, **lora)
    assert type(model.get_output_embeddings()) is not torch.nn.Linear
    records = _healthy(model, steps=6)
    assert [r["step"] for r in records] == list(range(6))
    probed = [r["probe"] for r in records if "probe" in r]
    assert len(probed) == 3
    assert all("not_read" in p["head"] for p in probed)
    evidence = ev.build_evidence(records)
    assert evidence["head_trends"]["read"] == 0
    assert evidence["head_trends"]["not_read"]["probes"] == 3
    assert evidence["incidents"] == []


# ---- a checkpoint that holds only the adapter ---------------------------------------------------


def _findings(evidence) -> list[dict]:
    return [f for i in evidence["incidents"] if i["kind"] == "resume_incomplete"
            for c in i["changes"] for f in c["findings"]]


def test_an_adapter_only_checkpoint_saved_and_loaded_names_nothing_missing(monkeypatch, tmp_path):
    """`save_pretrained` writes `adapter_model.safetensors` and `adapter_config.json` only; a second
    attempt builds the base model again, loads the adapter with `PeftModel.from_pretrained` and
    resumes. Nothing of the model may be named missing or of another shape: the frozen base
    weights are not in the folder on purpose, and every adapter tensor is there.

    The optimizer, the random state and the data position are not saved here, and the evidence
    says so; that is true of this resume and is not held. Whether the folder's adapter tensors are
    compared with the model is not decided, and is not held either."""
    transformers, peft = _libraries(monkeypatch)
    model = _lora_model(transformers, peft)
    folder = tmp_path / "adapter"
    first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
    session = tensorquill.init(model, sink=first, run_id="peft-adapter", **QUIET)
    _train(model, _adamw(model), 4)
    model.save_pretrained(folder)
    assert (folder / "adapter_model.safetensors").exists()
    assert not any(p.name.startswith("model") for p in folder.iterdir())
    session.checkpoint_files(folder, event="save")
    session.close()
    [save] = [r["checkpoint"]["save"] for r in first.records if "save" in r.get("checkpoint", {})]
    assert "model" in save["components"]

    time.sleep(0.01)  # the second attempt starts later
    torch.manual_seed(1)  # a base built again, as a new process would, with other adapter weights
    base = transformers.LlamaForCausalLM(model.base_model.model.config)
    resumed = peft.PeftModel.from_pretrained(base, folder, is_trainable=True)
    assert set(_trainable(resumed)) == set(_trainable(model))
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(resumed.get_parameter(name), p.detach()), name
    session = tensorquill.init(resumed, sink=second, run_id="peft-adapter", start_step=4, **QUIET)
    session.checkpoint_files(folder, event="load")
    _train(resumed, _adamw(resumed), 2, first=4)
    session.close()
    records = first.records + second.records
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4, 5]
    [load] = [r["checkpoint"]["load"] for r in second.records if "load" in r.get("checkpoint", {})]
    assert "model" in load["components"]
    evidence = ev.build_evidence(records, detectors="python")
    model_findings = [f for f in _findings(evidence) if f.get("component") == "model"]
    assert not any(f["finding"] in ("missing_tensors", "shape_differs") for f in model_findings)
    assert evidence["checkpoint_tensors"]["resumes"] == 1
