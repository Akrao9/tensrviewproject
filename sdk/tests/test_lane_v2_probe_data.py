"""Lane V2 (wave V): probe.py, data.py, huggingface.py and system.py.

Each test reproduces a row of OPEN_BUGS.md and failed on the base commit (66b52d7).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from tensorquill import data as D

FIXTURE = Path(__file__).parent / "fixtures" / "c16-batch-hash-versions.json"


# --- C16: batch hashes -------------------------------------------------------------------------


@pytest.mark.parametrize("left, right", [
    (["a\x00b", "c"], ["a", "b\x00c"]),  # the audit's pair: the old writer joined text with NUL
    ([b"\xff"], [b"\xfe"]),  # bytes that are not UTF-8: the old writer decoded them with replacement
    (["1"], [1]),  # text and a number: one item's type is part of its bytes
    ([1, "a"], ["1", "a"]),
    ([True], [1]),
    ([1.0], [1]),
    (["ab", "c"], ["a", "bc"]),
    ([b"a"], ["a"]),
    ([2**63], [str(2**63)]),  # beyond int64: typed items, not an unknown batch
])
def test_c16_different_batches_hash_apart(left, right):
    assert D.batch_hash(left) != D.batch_hash(right)


def test_c16_a_batch_of_int64_and_its_typed_twin_hash_apart():
    # [1] goes the int64 way; a NumPy-free stand-in for an index the int64 path refuses.
    class Index(int):
        pass

    assert D.batch_hash([1, 2]) == D.batch_hash(torch.tensor([1, 2]))
    assert D.batch_hash([1, 2]) == D.batch_hash([Index(1), Index(2)])
    assert D.batch_hash([1, 2]) != D.batch_hash([1.0, 2.0])


def test_c16_an_index_beyond_int64_still_hashes():
    assert D.batch_hash([2**64, 1]) is not None


def test_c16_text_with_a_lone_surrogate_hashes():
    assert D.batch_hash(["\ud800"]) not in (None, D.batch_hash(["\ud801"]))


def test_c16_hashes_carry_their_version():
    value = D.batch_hash([0, 1, 2])
    assert value.startswith(D.HASH_PREFIX) and D.HASH_PREFIX == "v2:"
    digest = value[len(D.HASH_PREFIX):]
    assert len(digest) == 16 and int(digest, 16) >= 0


def _legacy(*batches: list) -> str:
    """The writer before C16 (data.py at 66b52d7), for integer and text batches."""
    h = hashlib.blake2b(digest_size=8)
    for values in batches:
        if all(isinstance(v, int) for v in values):
            raw = b"".join(v.to_bytes(8, "little", signed=True) for v in values)
        else:
            raw = "\x00".join(map(str, values)).encode()
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def test_c16_the_shared_fixture_holds_both_writers_hashes():
    doc = json.loads(FIXTURE.read_text())
    assert doc["batches"]
    for name, entry in doc["batches"].items():
        assert entry["legacy"] == _legacy(entry["ids"]), name
        assert entry["v2"] == D.batch_hash(entry["ids"]), name
        assert ":" not in entry["legacy"] and entry["v2"].split(":", 1)[0] == "v2"
    hashes = {name: e for name, e in doc["batches"].items()}
    for case in doc["cases"]:
        for record in case["records"]:
            value = record["data"]["batch_hash"]
            assert any(value in (e["legacy"], e["v2"]) for e in hashes.values()), case["name"]
        kinds = {r["data"]["batch_hash"].partition(":")[0] if ":" in r["data"]["batch_hash"] else ""
                 for r in case["records"]}
        assert (len(kinds) == 1) == case["compared"], case["name"]


# --- C21 (SDK-20): checkpoint manifests ----------------------------------------------------------


class _WithExtraState(torch.nn.Module):
    """A model whose state dict holds a non-tensor `_extra_state`, as Transformer Engine's do."""

    def __init__(self, resampler: bool = False) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)
        if resampler:
            self.resampler = torch.nn.Linear(2, 2)
        self.calls = 0

    def get_extra_state(self):
        return {"calls": self.calls}

    def set_extra_state(self, state):
        self.calls = state["calls"]


@pytest.mark.parametrize("resampler", [False, True])
def test_c21_a_model_with_extra_state_is_the_model_component(tmp_path, resampler):
    model = _WithExtraState(resampler)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    state = torch.load(path, weights_only=False)
    _WithExtraState(resampler).load_state_dict(state["model"])  # a real checkpoint of the real module
    manifest = D.checkpoint_manifest(state)
    components = manifest["components"]
    assert components["model"]["keys"] == ["model"]
    assert components["model"]["tensors"] == (4 if resampler else 2)
    assert "data" not in components and "other" not in manifest


def test_c21_a_parameter_named_like_a_sampler_or_rng_is_not_that_component():
    tensors = {"resampler.weight": torch.zeros(1), "rng_proj.bias": torch.zeros(1), "_extra_state": b"x"}
    manifest = D.checkpoint_manifest({"model": tensors, "rng_state": torch.zeros(4, dtype=torch.uint8)})
    assert set(manifest["components"]) == {"model", "rng"}
    assert manifest["components"]["rng"]["keys"] == ["rng_state"]


# --- C22 (SDK-11, SDK-17): large finite numbers in the probe ----------------------------------


def _numbers(hidden, weight, bias=None):
    from tensorquill import probe as P

    sums = P._head_logit_sums(hidden, weight, bias)
    return P._head_numbers([{**sums, "input_dtype": "torch.float32", "weight_dtype": "torch.float32"}])


def test_c22_finite_logits_of_1e20_are_read_not_called_nonfinite():
    numbers = _numbers(torch.ones(1, 4), torch.full((1000, 4), 2.5e19))  # every logit 1e20
    assert numbers["positions"] == 1 and numbers["nonfinite_positions"] == 0 and "not_read" not in numbers
    assert numbers["norm"] == pytest.approx(1e20 * 1000 ** 0.5, rel=1e-5)
    assert numbers["mean"] == pytest.approx(1e20, rel=1e-5)


def test_c22_logits_beyond_the_sums_range_are_not_said_to_hold_nan_or_inf():
    numbers = _numbers(torch.ones(1, 1), torch.tensor([[3.39e38], [3.39e38]]))  # finite FP32, past BF16's max
    assert numbers["nonfinite_positions"] == 0 and numbers["positions"] == 0
    assert numbers["overflow_positions"] == 1
    assert "NaN" not in numbers["not_read"] and "overflow" in numbers["not_read"]


def test_c22_no_positions_is_not_said_to_hold_nan_or_inf():
    numbers = _numbers(torch.zeros(1, 0, 4), torch.randn(10, 4))
    assert numbers["positions"] == 0 and numbers["nonfinite_positions"] == 0
    assert numbers["not_read"] == "the head saw no positions"


def test_c22_genuinely_nonfinite_logits_are_still_counted():
    hidden = torch.randn(3, 4)
    hidden[1, 0] = float("inf")
    numbers = _numbers(hidden, torch.randn(8, 4))
    assert numbers["positions"] == 2 and numbers["nonfinite_positions"] == 1 and "overflow_positions" not in numbers


@pytest.fixture(params=[False, True], ids=["histc", "deterministic"])
def deterministic(request):
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(request.param)
    yield request.param
    torch.use_deterministic_algorithms(before)


@pytest.mark.parametrize("span", [1e36, 1e37, 1e38, 6e38])
def test_c22_histogram_counts_every_finite_value_at_any_span(deterministic, span):
    from tensorquill import probe as P

    values = (torch.linspace(-0.5, 0.5, 1000, dtype=torch.float64) * span).float()  # float32 linspace overflows
    assert values.isfinite().all()
    values[-1] = float("nan")
    out = P._histogram(values, 64)
    assert sum(out["counts"]) == 999 and out["nonfinite"] == 1
    assert (out["lo"], out["hi"]) == (float(values[0]), float(values[-2]))
    # The same as the histogram of the values scaled down by a power of two, where nothing overflows.
    assert out["counts"] == P._histogram(values * 2.0 ** -80, 64)["counts"]


def test_c22_histogram_of_two_values_near_the_float32_limits():
    from tensorquill import probe as P

    out = P._histogram(torch.tensor([-3e38, 3e38]), 8)
    assert out["counts"] == [1, 0, 0, 0, 0, 0, 0, 1] and out["nonfinite"] == 0


def test_c22_histogram_of_one_huge_value_counts_it():
    from tensorquill import probe as P

    for determinism in (False, True):
        before = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(determinism)
        try:
            out = P._histogram(torch.full((5,), 3e38), 8)
        finally:
            torch.use_deterministic_algorithms(before)
        assert sum(out["counts"]) == 5, determinism


# --- SDK-25: blocks that return a dict --------------------------------------------------------


def test_sdk25_a_block_returning_a_dict_gets_its_histogram():
    from tensorquill import probe as P

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return {"mask": x > 0, "hidden_states": self.lin(x)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)["hidden_states"]
            return x

    model = Model()
    state = P.ProbeState()
    state.head_off = True
    P.install_probe_hooks(list(model.layers.named_children()), state)
    result = P.run_probe(model, lambda m: m(torch.randn(2, 4)), state, [])
    assert set(result["histograms"]) == {"0", "1"}
    assert sum(result["histograms"]["0"]["counts"]) == 8


# --- SDK-16: a module that wraps the Hugging Face model ---------------------------------------


class _HFInner(torch.nn.Module):
    def __init__(self, attention="flash_attention_2"):
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(_attn_implementation=attention)
        self.body = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 6)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, x):
        return self.lm_head(self.body(x))


class _Wrapper(torch.nn.Module):
    """A LightningModule or a DDP of a Hugging Face model: no `config`, no `get_output_embeddings`."""

    def __init__(self, *inner):
        super().__init__()
        self.model = inner[0]
        if len(inner) > 1:
            self.teacher = inner[1]

    def forward(self, x):
        return self.model(x)


def test_sdk16_the_head_of_a_wrapped_model_is_found():
    from tensorquill import probe as P

    assert P.find_head(_Wrapper(_HFInner()), None) == ("model.lm_head", "get_output_embeddings", None)


def test_sdk16_two_wrapped_models_with_different_heads_read_none():
    from tensorquill import probe as P

    name, found_by, reason = P.find_head(_Wrapper(_HFInner(), _HFInner()), None)
    assert (name, found_by) == (None, None) and "model" in reason and "teacher" in reason


def test_sdk16_flash_attention_of_a_wrapped_model_is_switched_for_the_probe_and_back():
    from tensorquill import probe as P

    wrapper = _Wrapper(_HFInner(), _HFInner("eager"))
    with P._sdpa_instead_of_flash(wrapper) as switched:
        during = (wrapper.model.config._attn_implementation, wrapper.teacher.config._attn_implementation)
    assert switched and during == ("sdpa", "eager")
    assert wrapper.model.config._attn_implementation == "flash_attention_2"
    assert wrapper.teacher.config._attn_implementation == "eager"


def test_sdk16_a_config_inside_a_layer_is_not_taken_for_the_model_s():
    from types import SimpleNamespace

    from tensorquill import probe as P

    inner = _HFInner()
    inner.body.config = SimpleNamespace(_attn_implementation="flash_attention_3")  # an attention layer's own
    wrapper = _Wrapper(inner)
    with P._sdpa_instead_of_flash(wrapper):
        assert inner.body.config._attn_implementation == "flash_attention_3"
        assert inner.config._attn_implementation == "sdpa"


# --- SDK-26: NVML with every metric failed, a checkpoint folder given as a Path ---------------


def test_sdk26_a_folder_given_as_a_path_is_the_checkpoint_folder(tmp_path):
    from types import SimpleNamespace

    pytest.importorskip("transformers")
    from tensorquill.huggingface import _checkpoint_folder

    args, state = SimpleNamespace(output_dir=str(tmp_path / "out")), SimpleNamespace(global_step=3)
    assert _checkpoint_folder(args, state, tmp_path) == tmp_path
    assert _checkpoint_folder(args, state, str(tmp_path)) == tmp_path


def test_sdk26_nvml_with_every_metric_failed_adds_nothing_after_its_note(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_system import CUDA0, FakeNvml

    from tensorquill.system import Nvml

    fake = FakeNvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: type("P", (), {"uuid": "1234"})())
    for name in ("temperature", "clock", "power", "ecc", "throttle", "register"):
        fake.fail[name] = FakeNvml.NVMLError(3)
    monkeypatch.setattr(FakeNvml, "nvmlDeviceGetEnforcedPowerLimit", lambda self, handle: (_ for _ in ()).throw(
        FakeNvml.NVMLError(3)))
    reader = Nvml(CUDA0)
    try:
        first = reader.read()
        assert set(first) == {"unavailable"} and len(first["unavailable"]) == 8
        assert reader.read() is None and reader.read() is None
    finally:
        reader.close()
