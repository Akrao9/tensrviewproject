"""Wave AK, lane AK-2: a checkpoint folder's tensors, read from its `.safetensors` headers (B14).

`session.checkpoint_files(folder, event="load")` used to record `{"state": "unavailable", "why":
"folder"}` whatever the folder held, so a Hugging Face `Trainer` or Accelerate resume could never
report `missing_tensors` or `shape_differs`. Now the model's weights are found in the folder
(`model.safetensors.index.json` and its shards, or the `.safetensors` files themselves) and from
each file only the 8-byte little-endian header length N and the N-byte JSON header are read: no
tensor byte. The listing is compared with the model by the rules a state dict already goes
through (CHECKPOINT_TENSOR_CONTRACT.md, section 2), so both engines then read the load as they
read any other.

A header that cannot be trusted is refused, never raised: the load then reads as a folder does
today, `unavailable`, `folder`, with what was refused and in which file. A folder with only
`.bin` or `.pt` weights is not opened at all. The `safetensors` package writes the files here and
cross-checks the reading; the SDK does not import it."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import evidence as ev
from tensorquill.data import files_manifest

safetensors_torch = pytest.importorskip("safetensors.torch")
from safetensors import safe_open  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FOLDER = {"state": "unavailable", "why": "folder"}


class Net(nn.Module):
    """B14's shape: an LSTM and a head."""

    def __init__(self, hidden=4):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.head(self.lstm(x)[0])


def own_state(model):
    return {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}


def save(folder, tensors, name="model.safetensors"):
    folder.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(folder / name), metadata={"format": "pt"})
    return folder / name


def header_length(path):
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def raw_file(path, header: bytes, *, length=None, data=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header) if length is None else length) + header + data)
    return path


def header_of(tensors):
    """A valid header for `{name: shape}`, F32, offsets laid end to end."""
    out, at = {"__metadata__": {"format": "pt"}}, 0
    for name, shape in tensors.items():
        size = 4
        for n in shape:
            size *= n
        out[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [at, at + size]}
        at += size
    return json.dumps(out).encode()


def expected_listing(path):
    """What the `safetensors` package itself reads from the file: each tensor's name and shape."""
    with safe_open(str(path), framework="pt") as f:
        return {name: list(f.get_slice(name).get_shape()) for name in f.keys()}


def listing(folder):
    from tensorquill.data import safetensors_listing

    return safetensors_listing(folder)


def coverage(folder, model):
    from tensorquill.data import folder_tensor_coverage

    return folder_tensor_coverage(folder, model)


def refused(folder, model=None):
    """The refusal's `(why, refused, file)`; the listing and the comparison agree on it."""
    found = listing(folder)
    assert found["state"] == "unavailable", found
    assert coverage(folder, model if model is not None else Net()) == found
    return found["why"], found.get("refused"), found.get("file")


def load_record(tmp_path, model, folder, *, event="load"):
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, run_id="ak2", auto_step=False, heartbeat_seconds=None, stack_seconds=None,
                               sink=sink)
    session.checkpoint_files(folder, event=event)
    session.step()
    if event == "save":
        session.step()
    session.close()
    [manifest] = [r["checkpoint"][event] for r in sink.records if "checkpoint" in r]
    return manifest


# --- reading the header, and only the header ---------------------------------------------------


def test_the_listing_is_what_the_safetensors_package_reads(tmp_path):
    model = Net()
    path = save(tmp_path / "ckpt", own_state(model))
    found = listing(tmp_path / "ckpt")
    assert found["state"] == "listed"
    assert found["entries"] == {"model.safetensors": expected_listing(path)}
    assert set(found["entries"]["model.safetensors"]) == set(model.state_dict())  # `__metadata__` is no tensor


@pytest.mark.parametrize("how", ["truncated", "replaced", "header_only"])
def test_no_tensor_byte_is_needed(tmp_path, how):
    """The data region cut off, or overwritten with other bytes: the listing does not change."""
    model = Net()
    path = save(tmp_path / "ckpt", own_state(model))
    before = expected_listing(path)
    raw = path.read_bytes()
    n = header_length(path)
    data = raw[8 + n:]
    assert len(data) > 0
    if how == "truncated":
        path.write_bytes(raw[:8 + n + len(data) // 3])
    elif how == "replaced":
        path.write_bytes(raw[:8 + n] + bytes(255 - b for b in data))
    else:
        path.write_bytes(raw[:8 + n])
    assert listing(tmp_path / "ckpt")["entries"] == {"model.safetensors": before}
    assert coverage(tmp_path / "ckpt", model)["state"] == "compared"


def test_a_header_padded_with_spaces_is_read(tmp_path):
    """The format pads the header with spaces to an 8-byte boundary; the writer above does too."""
    raw_file(tmp_path / "ckpt" / "model.safetensors", header_of({"head.weight": [2, 4]}) + b"      ")
    assert listing(tmp_path / "ckpt")["entries"] == {"model.safetensors": {"head.weight": [2, 4]}}


def test_a_sharded_folder_with_its_index_is_one_entry(tmp_path):
    model = Net()
    state = own_state(model)
    names = sorted(state)
    folder = tmp_path / "ckpt"
    shards = {"model-00001-of-00002.safetensors": names[:3], "model-00002-of-00002.safetensors": names[3:]}
    for shard, held in shards.items():
        save(folder, {k: state[k] for k in held}, shard)
    index = {"metadata": {"total_size": 1}, "weight_map": {k: s for s, held in shards.items() for k in held}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    found = listing(folder)
    assert found["entries"] == {"model.safetensors.index.json": {k: list(v.shape) for k, v in state.items()}}
    tensors = coverage(folder, model)
    assert (tensors["state"], tensors["entry"], tensors["by"], tensors["matched"]) == (
        "compared", "model.safetensors.index.json", "name", 6)


def test_a_sharded_folder_written_by_transformers(tmp_path):
    """The layout `save_pretrained` writes with `max_shard_size`: `model-0000k-of-0000n.safetensors`
    and `model.safetensors.index.json`, read in transformers' modeling_utils."""
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=4, vocab_size=128)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    folder = tmp_path / "checkpoint-4"
    model.save_pretrained(folder, max_shard_size="20KB")
    shards = sorted(p.name for p in folder.glob("model-*.safetensors"))
    assert len(shards) > 2 and (folder / "model.safetensors.index.json").exists()
    found = listing(folder)
    expected = {}
    for shard in shards:
        expected.update(expected_listing(folder / shard))
    assert found["entries"] == {"model.safetensors.index.json": expected}
    tensors = coverage(folder, model)
    assert tensors["state"] == "compared" and tensors["by"] == "name"
    assert "missing" not in tensors and "shape_differs" not in tensors
    assert tensors["matched"] == tensors["checkpoint"] == len(expected)


def test_a_tied_weight_written_once_by_transformers_is_tied_not_missing(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_embd=16, n_layer=2, n_head=2, vocab_size=64, n_positions=16)
    model = transformers.GPT2LMHeadModel(config)
    assert model.lm_head.weight is model.transformer.wte.weight
    model.save_pretrained(tmp_path / "ckpt")
    assert "lm_head.weight" not in expected_listing(tmp_path / "ckpt" / "model.safetensors")
    tensors = coverage(tmp_path / "ckpt", model)
    assert tensors["state"] == "compared" and "missing" not in tensors
    assert tensors["tied"] == {"count": 1, "names": ["lm_head.weight"]}


def test_an_accelerate_folder_of_two_models(tmp_path):
    """Accelerate's `save_state` writes `model.safetensors` for the first prepared model and
    `model_1.safetensors` for the second (checkpointing.py, `weights_name.replace(".", f"_{i}.")`):
    two models, never one. The session's model is found among them by the same rules as an entry
    of a state dict: the conventional name first, then the names."""
    accelerate = pytest.importorskip("accelerate")
    first, second = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1)), Net()
    accelerator = accelerate.Accelerator(cpu=True)
    first, second = accelerator.prepare(first, second)
    folder = tmp_path / "ckpt"
    accelerator.save_state(str(folder))
    assert {"model.safetensors", "model_1.safetensors"} <= {p.name for p in folder.iterdir()}
    found = listing(folder)
    assert set(found["entries"]) == {"model.safetensors", "model_1.safetensors"}
    one = coverage(folder, first)
    assert (one["state"], one["entry"], one["by"]) == ("compared", "model.safetensors", "name")
    two = coverage(folder, second)
    assert (two["state"], two["entry"], two["by"]) == ("compared", "model_1.safetensors", "names")
    assert "missing" not in one and "missing" not in two


def test_names_saved_under_a_ddp_wrapper_are_compared_without_it(tmp_path):
    """Accelerate saves a prepared model without unwrapping it, so under DDP the names start with
    `module.`; the comparison already removes it on both sides (contract, section 2)."""
    model = Net()
    save(tmp_path / "ckpt", {f"module.{k}": v for k, v in own_state(model).items()})
    tensors = coverage(tmp_path / "ckpt", model)
    assert tensors["state"] == "compared" and tensors["matched"] == 6 and "unexpected" not in tensors


def test_two_unindexed_shards_are_ambiguous_not_missing(tmp_path):
    """Shards whose index is gone: each holds part of the model and neither is named as the model's,
    so nothing is called missing."""
    model = Net()
    state = own_state(model)
    names = sorted(state)
    save(tmp_path / "ckpt", {k: state[k] for k in names[:3]}, "model-00001-of-00002.safetensors")
    save(tmp_path / "ckpt", {k: state[k] for k in names[3:]}, "model-00002-of-00002.safetensors")
    tensors = coverage(tmp_path / "ckpt", model)
    assert tensors["state"] == "unavailable" and tensors["why"] == "ambiguous"


# --- what is compared ---------------------------------------------------------------------------


def test_a_tensor_the_checkpoint_lacks_is_missing(tmp_path):
    model = Net()
    state = own_state(model)
    del state["lstm.weight_hh_l0"], state["lstm.bias_hh_l0"]
    save(tmp_path / "ckpt", state)
    tensors = coverage(tmp_path / "ckpt", model)
    assert tensors == {"state": "compared", "entry": "model.safetensors", "by": "name", "model": 6, "trainable": 6,
                       "checkpoint": 4, "matched": 4,
                       "missing": {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}}


def test_a_shape_that_differs(tmp_path):
    model = Net()
    state = own_state(model)
    state["head.weight"] = torch.zeros(3, 4)
    save(tmp_path / "ckpt", state)
    tensors = coverage(tmp_path / "ckpt", model)
    assert tensors["shape_differs"] == {"count": 1, "names": ["head.weight"],
                                        "shapes": {"head.weight": {"model": [2, 4], "checkpoint": [3, 4]}}}


def test_a_checkpoint_of_another_model_is_not_identified(tmp_path):
    save(tmp_path / "ckpt", {"encoder.layer.0.weight": torch.zeros(2)})
    assert coverage(tmp_path / "ckpt", Net()) == {"state": "unavailable", "why": "not_identified"}


# --- refused, never raised ----------------------------------------------------------------------


def test_a_folder_with_only_pickled_weights_is_not_opened(tmp_path, monkeypatch):
    folder = tmp_path / "ckpt"
    folder.mkdir()
    for name in ("pytorch_model.bin", "optimizer.pt", "model_states.pt", "rng_state.pth"):
        (folder / name).write_bytes(b"\x80\x04 not a pickle anyone should load")
    import builtins
    import io

    opened = []
    for owner, name in ((os, "open"), (builtins, "open"), (io, "open")):
        real = getattr(owner, name)
        monkeypatch.setattr(owner, name, lambda path, *a, _real=real, **k: (opened.append(path), _real(path, *a, **k))[1])
    assert listing(folder) == FOLDER
    assert coverage(folder, Net()) == FOLDER
    assert opened == []


def test_a_header_over_the_formats_cap(tmp_path):
    raw_file(tmp_path / "ckpt" / "model.safetensors", b"{}", length=100_000_001)
    assert refused(tmp_path / "ckpt") == ("folder", "header_too_large", "model.safetensors")


def test_a_header_longer_than_the_file(tmp_path):
    raw_file(tmp_path / "ckpt" / "model.safetensors", b'{"a": 1}', length=4096)
    assert refused(tmp_path / "ckpt") == ("folder", "header_past_end", "model.safetensors")


@pytest.mark.parametrize("size", [0, 5])
def test_a_file_too_short_for_its_length(tmp_path, size):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "model.safetensors").write_bytes(b"\x01" * size)
    assert refused(tmp_path / "ckpt") == ("folder", "short_file", "model.safetensors")


@pytest.mark.parametrize("header,why", [
    (b'{"a": "\xff\xfe"}', "not_utf8"),
    (b'{"head.weight": {"dtype": "F32", "shape": [2, 4]', "not_json"),
    (b"[1, 2]", "not_an_object"),
    (b'"model"', "not_an_object"),
])
def test_a_header_that_is_not_a_json_object(tmp_path, header, why):
    raw_file(tmp_path / "ckpt" / "model.safetensors", header)
    assert refused(tmp_path / "ckpt") == ("folder", why, "model.safetensors")


@pytest.mark.parametrize("entry", [
    {"shape": [2, 4], "data_offsets": [0, 32]},
    {"dtype": "F32", "data_offsets": [0, 32]},
    {"dtype": "", "shape": [2, 4], "data_offsets": [0, 32]},
    {"dtype": 7, "shape": [2, 4], "data_offsets": [0, 32]},
    {"dtype": "F32", "shape": "2x4", "data_offsets": [0, 32]},
    {"dtype": "F32", "shape": [2, -4], "data_offsets": [0, 32]},
    {"dtype": "F32", "shape": [2, 4.0], "data_offsets": [0, 32]},
    {"dtype": "F32", "shape": [True, 4], "data_offsets": [0, 32]},
    [2, 4],
])
def test_a_tensor_entry_without_a_usable_dtype_or_shape(tmp_path, entry):
    header = json.dumps({"__metadata__": {"format": "pt"}, "lstm.weight_hh_l0": {
        "dtype": "F32", "shape": [16, 4], "data_offsets": [0, 256]}, "head.weight": entry}).encode()
    raw_file(tmp_path / "ckpt" / "model.safetensors", header)
    found = listing(tmp_path / "ckpt")
    assert (found["why"], found["refused"], found["file"], found["tensor"]) == (
        "folder", "bad_entry", "model.safetensors", "head.weight")


def test_an_index_naming_a_shard_that_is_absent(tmp_path):
    model = Net()
    state = own_state(model)
    save(tmp_path / "ckpt", {"head.weight": state["head.weight"]}, "model-00001-of-00002.safetensors")
    index = {"weight_map": {"head.weight": "model-00001-of-00002.safetensors",
                            "head.bias": "model-00002-of-00002.safetensors"}}
    (tmp_path / "ckpt" / "model.safetensors.index.json").write_text(json.dumps(index))
    assert refused(tmp_path / "ckpt") == ("folder", "shard_absent", "model-00002-of-00002.safetensors")


@pytest.mark.parametrize("index", [
    "{not json",
    json.dumps([1]),
    json.dumps({"metadata": {}}),
    json.dumps({"weight_map": {"head.weight": 3}}),
    json.dumps({"weight_map": {"head.weight": "../elsewhere/model.safetensors"}}),
    json.dumps({"weight_map": {"head.weight": "/abs/model.safetensors"}}),
    json.dumps({"weight_map": {"head.weight": "pytorch_model-00001-of-00001.bin"}}),  # a pickle is never opened
])
def test_an_index_that_cannot_be_read(tmp_path, index):
    save(tmp_path / "ckpt", {"head.weight": torch.zeros(2, 4)}, "model-00001-of-00001.safetensors")
    (tmp_path / "ckpt" / "pytorch_model-00001-of-00001.bin").write_bytes(b"\x80\x04" + b"\x00" * 64)
    (tmp_path / "ckpt" / "model.safetensors.index.json").write_text(index)
    why, reason, file = refused(tmp_path / "ckpt")
    assert (why, file) == ("folder", "model.safetensors.index.json")
    assert reason in ("bad_index", "not_json", "not_an_object")


def test_the_same_tensor_in_two_shards(tmp_path):
    folder = tmp_path / "ckpt"
    save(folder, {"head.weight": torch.zeros(2, 4)}, "model-00001-of-00002.safetensors")
    save(folder, {"head.weight": torch.zeros(2, 4), "head.bias": torch.zeros(2)}, "model-00002-of-00002.safetensors")
    index = {"weight_map": {"head.weight": "model-00001-of-00002.safetensors",
                            "head.bias": "model-00002-of-00002.safetensors"}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    found = listing(folder)
    assert (found["why"], found["refused"], found["tensor"]) == ("folder", "duplicate_tensor", "head.weight")


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs file permissions that bind")
def test_an_unreadable_file(tmp_path):
    path = save(tmp_path / "ckpt", {"head.weight": torch.zeros(2, 4)})
    path.chmod(0)
    try:
        assert refused(tmp_path / "ckpt") == ("folder", "unreadable", "model.safetensors")
    finally:
        path.chmod(0o644)


def test_a_directory_named_like_weights_is_unreadable(tmp_path):
    (tmp_path / "ckpt" / "model.safetensors").mkdir(parents=True)
    assert refused(tmp_path / "ckpt") == ("folder", "unreadable", "model.safetensors")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes here")
def test_a_named_pipe_is_refused_and_never_waited_on(tmp_path):
    """Opening a pipe for reading waits for a writer: that would stop the training thread."""
    (tmp_path / "ckpt").mkdir()
    pipe = tmp_path / "ckpt" / "model.safetensors"
    os.mkfifo(pipe)
    result = []
    worker = threading.Thread(target=lambda: result.append(listing(tmp_path / "ckpt")), daemon=True)
    worker.start()
    worker.join(10)
    if worker.is_alive():  # let the blocked open finish so the test process can end
        with open(pipe, "wb"):
            pass
        worker.join(10)
        pytest.fail("the listing waited on a named pipe")
    [found] = result
    assert (found["why"], found["refused"], found["file"]) == ("folder", "unreadable", "model.safetensors")


def test_too_many_files(tmp_path, monkeypatch):
    import tensorquill.data as data

    monkeypatch.setattr(data, "FOLDER_FILES_READ", 2)
    for i in range(3):
        save(tmp_path / "ckpt", {"head.weight": torch.zeros(2, 4)}, f"model_{i}.safetensors")
    found = listing(tmp_path / "ckpt")
    assert (found["why"], found["refused"]) == ("folder", "too_many_files")


def test_too_many_header_bytes_across_the_folder(tmp_path, monkeypatch):
    import tensorquill.data as data

    first = save(tmp_path / "ckpt", {"head.weight": torch.zeros(2, 4)}, "model.safetensors")
    save(tmp_path / "ckpt", {"head.bias": torch.zeros(2)}, "model_1.safetensors")
    monkeypatch.setattr(data, "FOLDER_HEADER_BYTES", header_length(first) + 8)
    found = listing(tmp_path / "ckpt")
    assert (found["why"], found["refused"]) == ("folder", "too_many_header_bytes")


def test_a_header_with_no_tensor(tmp_path):
    raw_file(tmp_path / "ckpt" / "model.safetensors", b'{"__metadata__": {"format": "pt"}}')
    found = listing(tmp_path / "ckpt")
    assert (found["why"], found["refused"]) == ("folder", "no_tensors")


# --- through a session ----------------------------------------------------------------------------


def test_a_folder_load_records_the_comparison(tmp_path):
    model = Net()
    state = own_state(model)
    del state["lstm.bias_hh_l0"]
    folder = save(tmp_path / "checkpoint-4", state).parent
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        (folder / name).write_bytes(b"")
    manifest = load_record(tmp_path, model, folder)
    assert manifest["components"] == files_manifest(p.name for p in folder.iterdir())["components"]
    assert manifest["tensors"]["state"] == "compared"
    assert manifest["tensors"]["missing"] == {"count": 1, "names": ["lstm.bias_hh_l0"]}


def test_a_folder_save_carries_no_listing(tmp_path):
    """Contract decision 4: a save is not compared, and `checkpoint()` lists no tensors at a save."""
    folder = save(tmp_path / "checkpoint-4", own_state(Net())).parent
    manifest = load_record(tmp_path, Net(), folder, event="save")
    assert {k: v for k, v in manifest.items() if k != "after_step"} == files_manifest(["model.safetensors"])


def test_a_refused_folder_load_reads_as_a_folder_does_today(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "model.safetensors").write_bytes(b"")
    manifest = load_record(tmp_path, Net(), tmp_path / "ckpt")
    assert manifest["tensors"] == {**FOLDER, "refused": "short_file", "file": "model.safetensors"}


def test_a_comparison_that_raises_is_recorded_as_error(tmp_path, monkeypatch):
    import tensorquill.data as data

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(data, "_read_model", boom)
    folder = save(tmp_path / "ckpt", own_state(Net())).parent
    manifest = load_record(tmp_path, Net(), folder)
    assert manifest["tensors"] == {"state": "unavailable", "why": "error"}


def _resume(tmp_path, model, folder):
    """Four steps and a save, then a second attempt that loads `folder` at step 4 and takes two."""
    first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
    common = dict(run_id="ak2-resume", auto_step=False, heartbeat_seconds=None, stack_seconds=None)
    session = tensorquill.init(model, sink=first, **common)
    for _ in range(4):
        session.step()
    session.checkpoint_files(folder, event="save")
    session.close()
    time.sleep(0.01)  # the second attempt starts later
    session = tensorquill.init(model, sink=second, start_step=4, **common)
    session.checkpoint_files(folder, event="load")
    session.step()
    session.step()
    session.close()
    return first.records + second.records


def _resume_finding(evidence):
    [hit] = [i for i in evidence["incidents"] if i["kind"] == "resume_incomplete"]
    return hit["changes"][0]["findings"]


def _server_binary():
    chosen = os.environ.get("TENSORQUILL_SERVER_BIN")  # conftest names this checkout's build
    if not chosen or not Path(chosen).exists():
        pytest.skip("no tensorquill-server built under this checkout's server/target")
    return Path(chosen).resolve()


@pytest.mark.parametrize("change", ["missing", "shape"])
def test_both_engines_say_the_same_of_a_folder_resume(tmp_path, change):
    model = Net()
    state = own_state(model)
    if change == "missing":
        del state["lstm.weight_hh_l0"], state["lstm.bias_hh_l0"]
        finding = {"component": "model", "finding": "missing_tensors", "count": 2,
                   "tensors": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}
    else:
        state["head.weight"] = torch.zeros(3, 4)
        finding = {"component": "model", "finding": "shape_differs", "count": 1, "tensors": ["head.weight"]}
    folder = save(tmp_path / "checkpoint-4", state).parent
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        (folder / name).write_bytes(b"")
    records = _resume(tmp_path, model, folder)
    python = ev.build_evidence(records, detectors="python")
    assert _resume_finding(python)[-1] == finding
    block = python["checkpoint_tensors"]
    assert block["resumes"] == 1 and block["entries"][0]["state"] == "compared"
    assert block["entries"][0]["entry"] == "model.safetensors"
    binary = _server_binary()
    assert binary.is_relative_to(REPO.resolve()), f"{binary} is not this checkout's server"
    rust = ev.build_evidence(records, detectors="rust")
    assert _resume_finding(rust) == _resume_finding(python)
    assert rust["checkpoint_tensors"] == block


def test_both_engines_read_a_refused_folder_as_a_folder(tmp_path):
    model = Net()
    folder = tmp_path / "checkpoint-4"
    raw_file(folder / "model.safetensors", b"[1]")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        (folder / name).write_bytes(b"")
    records = _resume(tmp_path, model, folder)
    python = ev.build_evidence(records, detectors="python")
    entry = python["checkpoint_tensors"]["entries"][0]
    assert (entry["state"], entry["why"]) == ("unavailable", "folder")
    assert not any(f.get("finding") in ("missing_tensors", "shape_differs")
                   for i in python["incidents"] if i["kind"] == "resume_incomplete"
                   for c in i["changes"] for f in c["findings"])
    _server_binary()
    rust = ev.build_evidence(records, detectors="rust")
    assert rust["checkpoint_tensors"] == python["checkpoint_tensors"]


def test_a_trainer_resume_from_a_folder_missing_a_tensor(tmp_path):
    """The Hugging Face `Trainer` itself: train, save `checkpoint-2`, take a tensor out of its
    `model.safetensors`, resume. The Trainer loads it with `strict=False` and goes on; the callback
    records the folder and the evidence names the tensor."""
    transformers = pytest.importorskip("transformers")
    from tensorquill.huggingface import TensorQuillCallback

    config = transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
    dropped = "model.layers.0.mlp.down_proj.weight"

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            ids = torch.randint(0, 32, (8,), generator=torch.Generator().manual_seed(i))
            return {"input_ids": ids, "labels": ids}

    def train(steps, sink, resume=None):
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / "out"), max_steps=steps, save_steps=2, per_device_train_batch_size=8,
            report_to=[], use_cpu=True, logging_steps=1, disable_tqdm=True)
        torch.manual_seed(0)
        trainer = transformers.Trainer(model=transformers.LlamaForCausalLM(config), args=args, train_dataset=Data(),
                                       callbacks=[TensorQuillCallback(sink=sink, run_id="ak2-trainer")])
        trainer.train(resume_from_checkpoint=resume)

    first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
    train(2, first)
    folder = tmp_path / "out" / "checkpoint-2"
    weights = folder / "model.safetensors"
    state = {name: list(shape) for name, shape in expected_listing(weights).items()}
    assert dropped in state
    with safe_open(str(weights), framework="pt") as f:
        kept = {k: f.get_tensor(k) for k in f.keys() if k != dropped}
    safetensors_torch.save_file(kept, str(weights), metadata={"format": "pt"})
    time.sleep(0.01)
    train(4, second, resume=str(folder))
    [load] = [r["checkpoint"]["load"] for r in second.records if "load" in r.get("checkpoint", {})]
    assert load["tensors"]["missing"] == {"count": 1, "names": [dropped]}
    evidence = ev.build_evidence(first.records + second.records, detectors="python")
    assert _resume_finding(evidence)[-1] == {"component": "model", "finding": "missing_tensors", "count": 1,
                                             "tensors": [dropped]}


# --- nothing raised into training ----------------------------------------------------------------


def test_nothing_is_raised_under_python_w_error(tmp_path):
    """Good, refused and unreadable folders, a save and a load each, in a fresh interpreter with
    every warning an error."""
    good = save(tmp_path / "good", own_state(Net())).parent
    raw_file(tmp_path / "bad" / "model.safetensors", b"[1]")
    (tmp_path / "pickled").mkdir()
    (tmp_path / "pickled" / "pytorch_model.bin").write_bytes(b"")
    script = f"""
import torch, tensorquill
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 4)
        self.head = nn.Linear(4, 2)
sink = tensorquill.MemorySink()
session = tensorquill.init(Net(), run_id="w-error", auto_step=False, heartbeat_seconds=None, stack_seconds=None,
                           sink=sink)
for folder in {[str(good), str(tmp_path / "bad"), str(tmp_path / "pickled"), str(tmp_path / "absent")]!r}:
    session.checkpoint_files(folder, event="load")
    session.step()
    session.checkpoint_files(folder, event="save")
    session.step()
session.close()
loads = [r["checkpoint"]["load"]["tensors"] for r in sink.records if "load" in r.get("checkpoint", {{}})]
assert [t["state"] for t in loads] == ["compared", "unavailable", "unavailable"], loads
print("ok")
"""
    env = {**os.environ, "PYTHONPATH": str(REPO / "sdk" / "src")}
    done = subprocess.run([sys.executable, "-W", "error", "-c", script], capture_output=True, text=True, env=env,
                          timeout=300)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-4000:]


# --- a model whose saved names are not its own --------------------------------------------------


def test_a_trainer_resume_of_a_moe_model_names_the_experts_it_did_not_restore(tmp_path):
    """Transformers 5 holds a Mixtral's experts fused (`mlp.experts.gate_up_proj`) and
    `save_pretrained` writes them back in the original per-expert format
    (`block_sparse_moe.experts.0.w1.weight`; `revert_weight_conversion` in modeling_utils.py). The
    `Trainer` resumes with `model.load_state_dict(state_dict, False)` on that file
    (`_load_from_checkpoint`), so the fused experts and the router keep their fresh values: the
    folder reads `compared` with them `missing`, and that is what happened. A pin of a true finding,
    not a matching rule: `from_pretrained` would have converted the names, and a folder loaded that
    way, recorded by hand, would read the same."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "MixtralForCausalLM"):
        pytest.skip("no Mixtral in this transformers")
    from tensorquill.huggingface import TensorQuillCallback

    config = transformers.MixtralConfig(hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
                                        num_local_experts=4)
    seen = {}

    class Experts(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            seen.setdefault("begin", []).append(model.model.layers[0].mlp.experts.gate_up_proj.detach().clone())

        def on_train_end(self, args, state, control, model=None, **kwargs):
            seen.setdefault("end", []).append(model.model.layers[0].mlp.experts.gate_up_proj.detach().clone())

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            ids = torch.randint(0, 32, (8,), generator=torch.Generator().manual_seed(i))
            return {"input_ids": ids, "labels": ids}

    def train(steps, sink, seed, resume=None):
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / "out"), max_steps=steps, save_steps=2, per_device_train_batch_size=8,
            report_to=[], use_cpu=True, logging_steps=1, disable_tqdm=True)
        torch.manual_seed(seed)
        trainer = transformers.Trainer(model=transformers.MixtralForCausalLM(config), args=args, train_dataset=Data(),
                                       callbacks=[TensorQuillCallback(sink=sink, run_id="ak2-moe"), Experts()])
        trainer.train(resume_from_checkpoint=resume)

    first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
    train(2, first, seed=0)
    folder = tmp_path / "out" / "checkpoint-2"
    names = expected_listing(folder / "model.safetensors")
    assert "model.layers.0.mlp.experts.gate_up_proj" not in names  # written in the original format
    time.sleep(0.01)
    train(4, second, seed=1, resume=str(folder))
    assert not torch.equal(seen["begin"][1], seen["end"][0])  # the resumed run did not get the saved experts
    [load] = [r["checkpoint"]["load"] for r in second.records if "load" in r.get("checkpoint", {})]
    tensors = load["tensors"]
    assert (tensors["state"], tensors["entry"], tensors["by"]) == ("compared", "model.safetensors", "name")
    assert {"model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"} <= set(
        tensors["missing"]["names"])
    assert any(".experts.0." in name for name in tensors["unexpected"]["names"])
    evidence = ev.build_evidence(first.records + second.records, detectors="python")
    finding = _resume_finding(evidence)[-1]
    assert finding["finding"] == "missing_tensors" and "model.layers.0.mlp.experts.gate_up_proj" in finding["tensors"]
