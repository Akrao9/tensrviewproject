"""Wave AF, lane A: the checkpoint comparison's three audit findings (OPEN_BUGS.md section 11).

SDK-02, a child module held twice by one parent; SDK-03, a module without
`_non_persistent_buffers_set`; SDK-04, names that become one when a wrapper or the one prefix is
removed. A model read wrongly or a mapping that is not one-to-one reads as unavailable: never as
clean, never as a false mismatch. The healthy reading and B14's named tensors are held beside them."""

from __future__ import annotations

import torch
from torch import nn

from tensorquill.data import _read_model, load_manifest


def zeros(shape):
    return torch.zeros(*shape)


def build(names, shape=(2,)):
    """A module tree with one trainable parameter under each dotted name."""
    root = nn.Module()
    for name in names:
        *path, leaf = name.split(".")
        module = root
        for part in path:
            if part not in module._modules:
                module.add_module(part, nn.Module())
            module = module._modules[part]
        module.register_parameter(leaf, nn.Parameter(torch.zeros(*shape)))
    return root


def own_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class SharedChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(2, 2, bias=False)
        self.proj_b = self.proj_a


# SDK-02 ------------------------------------------------------------------------------------------

def test_a_child_held_twice_is_read_under_both_names_as_state_dict_names_it():
    model = SharedChild()
    assert [row[0] for row in _read_model(model)] == list(model.state_dict())


def test_a_child_held_twice_compares_clean_against_its_own_state_dict():
    model = SharedChild()
    tensors = load_manifest({"model": own_state(model)}, model)["tensors"]
    assert tensors == {"state": "compared", "entry": "model", "by": "name", "model": 2, "trainable": 2,
                       "checkpoint": 2, "matched": 2}


def test_a_child_held_twice_and_saved_once_is_tied_not_missing():
    model = SharedChild()
    tensors = load_manifest({"model": {"proj_a.weight": torch.zeros(2, 2)}}, model)["tensors"]
    assert tensors["state"] == "compared" and "missing" not in tensors and "unexpected" not in tensors
    assert tensors["tied"]["names"] == ["proj_b.weight"]


def test_a_module_that_holds_itself_still_ends():
    model = nn.Linear(2, 2)
    model._modules["me"] = model
    tensors = load_manifest({"model": {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}}, model)["tensors"]
    assert tensors["state"] == "compared" and tensors["model"] == 2


# SDK-03 ------------------------------------------------------------------------------------------

def test_a_module_with_buffers_and_no_buffer_set_is_not_readable():
    state = {"model": own_state(nn.BatchNorm1d(2))}
    model = nn.BatchNorm1d(2)
    del model.__dict__["_non_persistent_buffers_set"]
    assert load_manifest(state, model)["tensors"] == {"state": "unavailable", "why": "model_not_readable"}


def test_a_module_with_no_buffer_and_no_buffer_set_is_still_read():
    model = nn.Linear(2, 2)
    del model.__dict__["_non_persistent_buffers_set"]
    tensors = load_manifest({"model": own_state(nn.Linear(2, 2))}, model)["tensors"]
    assert tensors["state"] == "compared" and tensors["model"] == 2 and "unexpected" not in tensors


def test_a_non_persistent_buffer_is_still_left_out():
    model = nn.Linear(2, 2)
    model.register_buffer("scratch", torch.zeros(3), persistent=False)
    model.register_buffer("kept", torch.zeros(3))
    tensors = load_manifest({"model": own_state(model)}, model)["tensors"]
    assert tensors["state"] == "compared" and tensors["model"] == 3 and tensors["matched"] == 3
    assert not any(key in tensors for key in ("missing", "missing_buffers", "unexpected"))


# SDK-04 ------------------------------------------------------------------------------------------

class WeightAndModuleWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.module = nn.Linear(2, 2, bias=False)


def test_a_model_whose_names_become_one_without_a_wrapper_is_unavailable():
    model = WeightAndModuleWeight()
    tensors = load_manifest({"model": own_state(model)}, model)["tensors"]
    assert tensors["state"] == "unavailable" and tensors["why"] == "ambiguous"
    assert tensors["collides"] == "model"


def test_an_entry_whose_names_become_one_without_a_wrapper_is_unavailable_by_key_names_and_loader():
    model = build(["w", "b"])
    entry = {"w": torch.zeros(2), "module.w": torch.zeros(3), "b": torch.zeros(2)}
    for state, kwargs in (({"model": entry}, {}), ({"weird": entry}, {}), ({"model": entry}, {"model_state": entry})):
        tensors = load_manifest(state, model, **kwargs)["tensors"]
        assert tensors["state"] == "unavailable" and tensors["why"] == "ambiguous", (state.keys(), kwargs)
        assert tensors["collides"] == "checkpoint"


def test_a_prefix_removed_from_the_model_that_makes_two_names_one_is_unavailable():
    """Seen on the unfixed code: `compared` with a false `missing` that is `w` or `net.w` by the
    hash seed. Only the outcome is asserted, which does not depend on the seed."""
    model = build(["net.w", "w", "net.b", "net.c"])
    tensors = load_manifest({k: torch.zeros(2) for k in ("w", "b", "c")}, model)["tensors"]
    assert tensors["state"] == "unavailable" and tensors["why"] == "ambiguous"
    assert tensors["collides"] == "model"


def test_a_prefix_removed_from_the_checkpoint_that_makes_two_names_one_is_unavailable():
    """Seen on the unfixed code: `compared`, `checkpoint: 4, matched: 3`, nothing listed."""
    model = build(["w", "b", "c"])
    state = {"net.w": torch.zeros(2), "w": torch.zeros(3), "net.b": torch.zeros(2), "net.c": torch.zeros(2)}
    tensors = load_manifest(state, model)["tensors"]
    assert tensors["state"] == "unavailable" and tensors["why"] == "ambiguous"
    assert tensors["collides"] == "checkpoint"


def test_a_collision_leaves_components_model_as_today():
    model = WeightAndModuleWeight()
    state = {"model": own_state(model)}
    manifest = load_manifest(state, model)
    from tensorquill.data import checkpoint_manifest
    assert manifest["components"] == checkpoint_manifest(state)["components"]


def test_a_colliding_cache_beside_a_clean_model_entry_still_compares_the_model():
    model = build(["a.weight", "b.weight"])
    state = {"model": own_state(model), "cache": {"a.weight": torch.zeros(2), "module.a.weight": torch.zeros(2)}}
    tensors = load_manifest(state, model)["tensors"]
    assert tensors == {"state": "compared", "entry": "model", "by": "name", "model": 2, "trainable": 2,
                       "checkpoint": 2, "matched": 2}


# Kept behaviour ----------------------------------------------------------------------------------

def test_a_healthy_checkpoint_still_reads_healthy():
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
    tensors = load_manifest({"model": own_state(model), "step": 3}, model)["tensors"]
    assert tensors["state"] == "compared" and tensors["matched"] == tensors["model"] == tensors["checkpoint"]
    assert not any(key in tensors for key in ("missing", "missing_frozen", "missing_buffers", "unexpected",
                                              "shape_differs", "tied"))


def test_wrapped_and_prefixed_saves_still_compare():
    model = build(["net.lstm.weight", "net.head.weight"])
    plain = own_state(model)
    for saved in ({f"module.{k}": v for k, v in plain.items()}, {f"_orig_mod.{k}": v for k, v in plain.items()},
                  {k[len("net."):]: v for k, v in plain.items()}):
        tensors = load_manifest({"model": saved}, model)["tensors"]
        assert tensors["state"] == "compared" and tensors["matched"] == 2 and "missing" not in tensors, saved.keys()


def test_b14s_two_lstm_tensors_are_still_named_missing():
    model = nn.Module()
    model.lstm = nn.LSTM(4, 4)
    model.head = nn.Linear(4, 1)
    saved = {k: v for k, v in own_state(model).items() if k not in ("lstm.weight_hh_l0", "lstm.bias_hh_l0")}
    tensors = load_manifest({"model": saved}, model)["tensors"]
    assert tensors["state"] == "compared"
    assert tensors["missing"] == {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}
