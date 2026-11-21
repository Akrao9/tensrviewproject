"""Writes cases.json, the cases of CHECKPOINT_TENSOR_CONTRACT.md. Run from the repository's root
with the SDK on the path:

    PYTHONPATH=sdk/src .venv/bin/python sdk/tests/fixtures/checkpoint/make_cases.py

Two groups. `coverage` cases are the SDK's: a model and what a checkpoint holds, and the
`tensors` object the load's manifest must carry. A model's tensor is `[name, shape, kind]`, kind
"p" a trainable parameter, "f" a frozen parameter, "b" a buffer; a checkpoint's tensor is its
shape. `records` cases are the readers': step records in, and what `resume_incomplete` and the
evidence's `checkpoint_tensors` must say, alike in the Python detectors and the Rust server.

Every case states what the code answers today (`today`), computed here and checked by
`sdk/tests/test_checkpoint_tensor_contract.py`: a `pin` is a case whose expectation is already
today's answer; a `new` case differs from it, so it cannot pass before its rule is built.
"""

from __future__ import annotations

import json

import torch

from tensorquill.data import checkpoint_manifest, files_manifest
from tensorquill.data_checks import detect_resume_state

# --- the models ----------------------------------------------------------------------------

LSTM = [
    ["embed.weight", [65, 16], "p"],
    ["lstm.weight_ih_l0", [128, 16], "p"],
    ["lstm.weight_hh_l0", [128, 32], "p"],
    ["lstm.bias_ih_l0", [128], "p"],
    ["lstm.bias_hh_l0", [128], "p"],
    ["head.weight", [65, 32], "p"],
    ["head.bias", [65], "p"],
]
PLAIN = [["alpha", [4], "p"], ["beta", [4], "p"]]
ADAPTED = [
    ["base.embed.weight", [65, 16], "f"],
    ["base.proj.weight", [32, 16], "f"],
    ["base.proj.bias", [32], "f"],
    ["lora.down.weight", [4, 16], "p"],
    ["lora.up.weight", [32, 4], "p"],
]
NORMED = [
    ["conv.weight", [8, 3, 3, 3], "p"],
    ["bn.weight", [8], "p"],
    ["bn.bias", [8], "p"],
    ["bn.running_mean", [8], "b"],
    ["bn.running_var", [8], "b"],
    ["bn.num_batches_tracked", [], "b"],
]
DEEP = [[f"blocks.{i:02d}.weight", [4, 4], "p"] for i in range(60)] + [["head.weight", [2, 4], "p"]]


def held(model, without=(), prefix=""):
    """The model's tensors as a checkpoint entry holds them, leaving `without` out."""
    return {prefix + name: shape for name, shape, _ in model if name not in without}


def names(model, only):
    return sorted(name for name, _, kind in model if name in only or kind == only)


CACHE = {"layers.0.k": [2, 2], "layers.0.v": [2, 2]}
OTHER_MODEL = {f"encoder.layer.{i}.{part}": [4, 4] for i in range(4) for part in ("w", "b")}
OPTIMIZER = "an optimizer's state"  # stands for {"state": {...}, "param_groups": [...]}


def compared(model, entry, checkpoint, matched, **lists):
    out = {"state": "compared", "entry": entry, "model": len(model),
           "trainable": sum(kind == "p" for _, _, kind in model), "checkpoint": checkpoint, "matched": matched}
    for key, found in lists.items():
        if key == "prefix":
            out[key] = found
        elif key == "shapes":
            out["shape_differs"]["shapes"] = found
        else:
            out[key] = {"count": len(found), "names": sorted(found)[:50]}
    return out


def unavailable(why, **more):
    return {"state": "unavailable", "why": why, **more}


# --- what the code answers today -----------------------------------------------------------


def _tensors(entry):
    return {name: torch.zeros(shape) for name, shape in entry.items()}


def _state(checkpoint):
    """A real state dict for a case's checkpoint: tensors for shapes, an optimizer's state."""
    state = {}
    for key, value in checkpoint.items():
        state[key] = ({"state": {0: {"step": torch.tensor(9.0)}}, "param_groups": [{"lr": 1e-3}]}
                      if value == OPTIMIZER else _tensors(value))
    return state


def coverage_today(given):
    if "folder" in given:
        manifest = files_manifest(given["folder"])
    elif "bare" in given:
        manifest = checkpoint_manifest(_tensors(given["bare"]))
    else:
        manifest = checkpoint_manifest(_state(given["checkpoint"]))
    return {"model_component": "model" in manifest["components"], "tensors": "tensors" in manifest}


def coverage(status, says, given, tensors, model_component, **more):
    expected = {"model_component": model_component, **({"tensors": tensors} if tensors is not None else {"tensors_key": False})}
    case = {"group": "coverage", "status": status, "says": says, "given": given, "expected": {**expected, **more}}
    if "raises" not in given:
        case["today"] = coverage_today(given)
    return case


# --- records -------------------------------------------------------------------------------

FIRST, SECOND = 1000, 2000
COMPONENTS = {
    "model": {"tensors": 7, "keys_hash": "c0ffee", "keys": ["model"]},
    "optimizer": {"entries": 7, "param_groups": 1, "step": 10, "keys": ["optimizer"]},
    "rng": {"keys": ["rng"]},
    "data": {"keys": ["sampler"]},
}
FILES = {
    "model": {"files": ["model.safetensors"]},
    "optimizer": {"files": ["optimizer.pt"]},
    "rng": {"files": ["rng_state.pth"]},
    "data": {"files": ["trainer_state.json"]},
}


def without(components, *left_out):
    return {name: body for name, body in components.items() if name not in left_out}


def resumed(load=None, *, save=None):
    """Ten steps, then a second attempt that resumes at step 10 for five more. `load` is the
    manifest the resumed attempt recorded; `save` one the first attempt recorded after step 9."""
    rows = [{"step": s, "rank": 0, "attempt": FIRST, "lr": [1e-3], "grad_norm": 1.0} for s in range(10)]
    if save is not None:
        rows[9]["checkpoint"] = {"save": {**save, "after_step": 9}}
    rows += [{"step": s, "rank": 0, "attempt": SECOND, "lr": [1e-3], "grad_norm": 1.0} for s in range(10, 15)]
    if load is not None:
        rows[10]["checkpoint"] = {"load": load}
    return rows


def records_today(rows):
    found = detect_resume_state([dict(r) for r in rows])[1]
    return {"fires": found is not None, **({"findings": found["changes"][0]["findings"]} if found else {})}


def entry(tensors):
    """The evidence's entry for the resume at step 10: the `tensors` object as the readers pass it
    on (numbers, and each list's count and names)."""
    out = {"step": 10, "rank": 0, "attempt": SECOND}
    for key in ("state", "why", "entry", "model", "trainable", "checkpoint", "matched"):
        if key in tensors:
            out[key] = tensors[key]
    for key in ("missing", "missing_frozen", "missing_buffers", "unexpected", "shape_differs"):
        if key in tensors:
            out[key] = {"count": tensors[key]["count"], "names": tensors[key]["names"]}
    return out


def records(says, rows, fires, findings=(), block=None, **more):
    today = records_today(rows)
    expected = {"fires": fires, **({"findings": list(findings)} if fires else {}),
                "checkpoint_tensors": None if block is None else {"resumes": len(block), "entries": block}, **more}
    same = today["fires"] == fires and today.get("findings", []) == list(findings) and block is None
    return {"group": "records", "status": "pin" if same else "new", "says": says, "records": rows,
            "expected": expected, "today": today}


COMPLETE = compared(LSTM, "model", 7, 7)
TWO_MISSING = compared(LSTM, "model", 5, 5, missing=["lstm.weight_hh_l0", "lstm.bias_hh_l0"])
MANY_MISSING = compared(DEEP, "model", 1, 1, missing=[n for n, _, _ in DEEP if n != "head.weight"])
FROZEN_MISSING = compared(ADAPTED, "adapter", 2, 2, missing_frozen=names(ADAPTED, "f"))
BUFFERS_MISSING = compared(NORMED, "model", 3, 3, missing_buffers=names(NORMED, "b"))
UNEXPECTED = compared(LSTM, "model", 9, 7, unexpected=["old_head.weight", "old_head.bias"])
SHAPE = compared(LSTM, "model", 7, 7, shape_differs=["head.weight"],
                 shapes={"head.weight": {"model": [65, 32], "checkpoint": [10, 32]}})
MISSING_FINDING = {"component": "model", "finding": "missing_tensors", "count": 2,
                   "tensors": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}

cases = {
    # ---- coverage: what the SDK writes at a load ------------------------------------------
    "every_tensor_is_in_the_checkpoint": coverage(
        "new", "A checkpoint that holds every tensor of the model, under the key `model`.",
        {"model": LSTM, "checkpoint": {"model": held(LSTM), "optimizer": OPTIMIZER}}, COMPLETE, True),
    "two_lstm_tensors_left_out": coverage(
        "new", "TrainCheck's B14 as reproduced: the checkpointing function left the LSTM's hidden-to-hidden"
        " weight and bias out. Both are trainable, so both are named as missing.",
        {"model": LSTM, "checkpoint": {"model": held(LSTM, ("lstm.weight_hh_l0", "lstm.bias_hh_l0")),
                                       "optimizer": OPTIMIZER}}, TWO_MISSING, True),
    "saved_from_a_ddp_wrapped_model": coverage(
        "new", "The checkpoint's names carry DDP's `module.`. The wrapper's prefix is removed before names are"
        " compared, so nothing reads as missing.",
        {"model": LSTM, "checkpoint": {"model": held(LSTM, prefix="module.")}}, COMPLETE, True),
    "saved_from_a_compiled_model": coverage(
        "new", "The checkpoint's names carry `torch.compile`'s `_orig_mod.`, removed the same way.",
        {"model": LSTM, "checkpoint": {"model": held(LSTM, prefix="_orig_mod.")}}, COMPLETE, True),
    "the_session_watches_an_inner_module": coverage(
        "new", "The checkpoint was saved from an outer module that holds the watched model as `net`, as a"
        " LightningModule does. One leading prefix is tried when fewer than half the names match; the one"
        " removed is recorded.",
        {"model": LSTM, "checkpoint": {"state_dict": held(LSTM, prefix="net.")}},
        compared(LSTM, "state_dict", 7, 7, prefix={"side": "checkpoint", "removed": "net."}), True),
    "the_checkpoint_was_saved_from_an_inner_module": coverage(
        "new", "The other way round: the watched model holds the saved module as `net`.",
        {"model": [["net." + n, s, k] for n, s, k in LSTM], "checkpoint": {"model": held(LSTM)}},
        compared(LSTM, "model", 7, 7, prefix={"side": "model", "removed": "net."}), True),
    "a_bare_state_dict": coverage(
        "new", "`torch.save(model.state_dict())`: the tensors sit directly in the checkpoint, under no key."
        " Today no entry is taken for the model; compared with the model, the checkpoint itself is.",
        {"model": LSTM, "bare": held(LSTM)}, compared(LSTM, None, 7, 7), True),
    "unusual_names_under_a_plain_key": coverage(
        "new", "L23's first limit: a model whose parameters have no dotted or PyTorch name, saved under a"
        " key that names no model. Today it is passed over; its names are the model's, so it is the model's.",
        {"model": PLAIN, "checkpoint": {"state": held(PLAIN), "optimizer": OPTIMIZER}},
        compared(PLAIN, "state", 2, 2), True),
    "a_cache_with_parameter_like_names": coverage(
        "new", "L23's second limit: a cache under another name with dotted keys, and no model in the"
        " checkpoint. Today it is taken for the model; it shares no name with the model, so no entry is.",
        {"model": LSTM, "checkpoint": {"features": CACHE, "optimizer": OPTIMIZER}},
        unavailable("not_identified"), False),
    "a_cache_beside_the_model": coverage(
        "new", "The same cache beside the model's state: the entry with the model's names is compared.",
        {"model": LSTM, "checkpoint": {"model": held(LSTM), "features": CACHE}}, COMPLETE, True),
    "an_ema_copy_beside_the_model": coverage(
        "new", "Two entries hold every name of the model. A tie goes to the entry the name rule already"
        " takes for the model, then to the first in the checkpoint's order.",
        {"model": LSTM, "checkpoint": {"ema": held(LSTM), "model": held(LSTM)}}, COMPLETE, True),
    "a_checkpoint_of_another_model": coverage(
        "new", "An entry named `model` that shares no name with the model. It is not this model's state.",
        {"model": LSTM, "checkpoint": {"model": OTHER_MODEL}}, unavailable("not_identified"), False),
    "names_that_mostly_differ": coverage(
        "new", "One of the entry's eight names is the model's. Under half: the names follow another scheme"
        " or another model, so nothing is named as missing. The closest entry is recorded.",
        {"model": LSTM, "checkpoint": {"model": {**dict(list(OTHER_MODEL.items())[:7]), "head.bias": [65]}}},
        unavailable("not_identified", closest={"entry": "model", "checkpoint": 8, "matched": 1}), False),
    "exactly_half_the_names_are_the_models": coverage(
        "new", "Two of the entry's four names are the model's: half is enough for the entry to be compared.",
        {"model": LSTM, "checkpoint": {"model": {"head.weight": [65, 32], "head.bias": [65],
                                                 "old.weight": [4], "old.bias": [4]}}},
        compared(LSTM, "model", 4, 2, missing=[n for n, _, _ in LSTM if not n.startswith("head.")],
                 unexpected=["old.weight", "old.bias"]), True),
    "an_adapter_only_checkpoint": coverage(
        "new", "Only the trainable adapter was saved. The frozen base is not in the checkpoint; a frozen"
        " parameter is counted and named apart, and is not a missing tensor.",
        {"model": ADAPTED, "checkpoint": {"adapter": held(ADAPTED, names(ADAPTED, "f"))}}, FROZEN_MISSING, True),
    "buffers_left_out": coverage(
        "new", "The parameters are all there and BatchNorm's three buffers are not. Buffers are counted"
        " and named apart.",
        {"model": NORMED, "checkpoint": {"model": held(NORMED, names(NORMED, "b"))}}, BUFFERS_MISSING, True),
    "tensors_the_model_does_not_have": coverage(
        "new", "The checkpoint holds two tensors more than the model. They are counted and named as"
        " unexpected.",
        {"model": LSTM, "checkpoint": {"model": {**held(LSTM), "old_head.weight": [10, 32], "old_head.bias": [10]}}},
        UNEXPECTED, True),
    "a_shape_that_differs": coverage(
        "new", "A name both hold, with another shape in the checkpoint. It is matched by name and listed"
        " with both shapes.",
        {"model": LSTM, "checkpoint": {"model": {**held(LSTM), "head.weight": [10, 32]}}}, SHAPE, True),
    "more_missing_than_are_named": coverage(
        "new", "Sixty trainable tensors are missing and only the head was saved. The count is 60 and the"
        " first 50 names, sorted, are listed.",
        {"model": DEEP, "checkpoint": {"model": {"head.weight": [2, 4]}}}, MANY_MISSING, True),
    "a_checkpoint_folder": coverage(
        "new", "`session.checkpoint_files`: the components are read from file names and no file is opened,"
        " so no tensor name is known.",
        {"model": LSTM, "folder": ["model.safetensors", "optimizer.pt", "rng_state.pth", "trainer_state.json"]},
        unavailable("folder"), True),
    "the_comparison_raises": coverage(
        "new", "Reading the model's names raises. The load is recorded as it is today, the reason is"
        " `error`, and nothing is raised into training.",
        {"model": "a model whose state_dict() raises", "checkpoint": {"model": held(LSTM)}, "raises": True},
        unavailable("error"), True),
    "a_save_is_not_compared": coverage(
        "pin", "`event=\"save\"`: the manifest of a save carries no `tensors` key.",
        {"model": LSTM, "checkpoint": {"model": held(LSTM, ("lstm.weight_hh_l0",))}, "event": "save"},
        None, True),

    # ---- records: what both readers say ----------------------------------------------------
    "a_resume_with_every_tensor_verified": records(
        "The load compared the checkpoint with the model and every tensor is there. Nothing is raised, and"
        " the evidence says the comparison was made.",
        resumed({"components": COMPONENTS, "tensors": COMPLETE}), False, block=[entry(COMPLETE)]),
    "a_resume_with_two_tensors_verified_missing": records(
        "B14: two trainable tensors are not in the checkpoint. `resume_incomplete` names each.",
        resumed({"components": COMPONENTS, "tensors": TWO_MISSING}), True, [MISSING_FINDING],
        block=[entry(TWO_MISSING)],
        words="model 2 trainable tensors not in the checkpoint: lstm.bias_hh_l0, lstm.weight_hh_l0",
        words_absent="could not be identified"),
    "missing_tensors_beside_a_missing_component": records(
        "The checkpoint has no optimizer state and two tensors are missing. Today's findings come first,"
        " in today's order, then the tensors.",
        resumed({"components": without(COMPONENTS, "optimizer"), "tensors": TWO_MISSING}), True,
        [{"component": "optimizer", "finding": "missing"}, MISSING_FINDING], block=[entry(TWO_MISSING)]),
    "a_resume_with_more_missing_than_named": records(
        "Sixty are missing and the record names fifty. The finding carries the count and the names it was"
        " given; the sentence names five and says how many more.",
        resumed({"components": COMPONENTS, "tensors": MANY_MISSING}), True,
        [{"component": "model", "finding": "missing_tensors", "count": 60, "tensors": MANY_MISSING["missing"]["names"]}],
        block=[entry(MANY_MISSING)], words="and 55 more"),
    "a_resume_with_only_frozen_tensors_missing": records(
        "An adapter-only checkpoint. Nothing is raised; the evidence names the frozen parameters.",
        resumed({"components": COMPONENTS, "tensors": FROZEN_MISSING}), False, block=[entry(FROZEN_MISSING)]),
    "a_resume_with_only_buffers_missing": records(
        "Nothing is raised; the evidence names the buffers.",
        resumed({"components": COMPONENTS, "tensors": BUFFERS_MISSING}), False, block=[entry(BUFFERS_MISSING)]),
    "a_resume_with_only_unexpected_tensors": records(
        "Nothing is raised; the evidence names the tensors the model does not have.",
        resumed({"components": COMPONENTS, "tensors": UNEXPECTED}), False, block=[entry(UNEXPECTED)]),
    "a_resume_with_a_shape_that_differs": records(
        "A tensor the checkpoint holds with another shape. `resume_incomplete` names it.",
        resumed({"components": COMPONENTS, "tensors": SHAPE}), True,
        [{"component": "model", "finding": "shape_differs", "count": 1, "tensors": ["head.weight"]}],
        block=[entry(SHAPE)], words="model 1 tensor with another shape in the checkpoint: head.weight"),
    "a_resume_whose_model_was_not_identified": records(
        "No entry of the checkpoint shares enough names with the model. The finding is today's"
        " `not_identified`, once, and the evidence says why no tensor was compared.",
        resumed({"components": without(COMPONENTS, "model"), "tensors": unavailable("not_identified")}), True,
        [{"component": "model", "finding": "not_identified"}], block=[entry(unavailable("not_identified"))],
        words="model state could not be identified", words_absent="not in the checkpoint"),
    "a_resume_from_a_folder": records(
        "A Hugging Face or Accelerate checkpoint folder with every component's file. Nothing is raised."
        " The evidence says tensor coverage is unavailable, so the silence is not read as every tensor present.",
        resumed({"components": FILES, "tensors": unavailable("folder")}), False, block=[entry(unavailable("folder"))]),
    "a_resume_from_a_folder_with_no_optimizer_file": records(
        "The folder's own finding is unchanged, and tensor coverage is unavailable beside it.",
        resumed({"components": without(FILES, "optimizer"), "tensors": unavailable("folder")}), True,
        [{"component": "optimizer", "finding": "missing"}], block=[entry(unavailable("folder"))]),
    "a_resume_whose_comparison_raised": records(
        "The SDK could not read the model's names. Nothing is raised from it.",
        resumed({"components": COMPONENTS, "tensors": unavailable("error")}), False, block=[entry(unavailable("error"))]),
    "a_resume_recorded_before_this_check": records(
        "A load manifest with no `tensors` key, as every capture written before this page. Nothing is said"
        " about tensors: no finding, and no `checkpoint_tensors` in the evidence.",
        resumed({"components": COMPONENTS}), False),
    "a_resume_judged_from_a_save": records(
        "No load was recorded, so the resume is judged from the save after step 9, as today. A `tensors`"
        " object on a save is not read: tensors are judged from a load only.",
        resumed(save={"components": without(COMPONENTS, "rng"), "tensors": TWO_MISSING}), True,
        [{"component": "rng", "finding": "missing"}]),
    "a_tensors_entry_that_is_not_an_object": records(
        "`tensors` is a string. It is ignored: no finding, no entry.",
        resumed({"components": COMPONENTS, "tensors": "compared"}), False),
    "a_missing_list_with_no_count": records(
        "`missing` has names and a count of zero. A list is read only with a positive whole count, so there"
        " is no finding; the entry is written without the list.",
        resumed({"components": COMPONENTS, "tensors": {**COMPLETE, "missing": {"count": 0, "names": ["head.bias"]}}}),
        False, block=[entry(COMPLETE)]),
}

json.dump({"contract": "CHECKPOINT_TENSOR_CONTRACT.md", "names_listed": 50, "cases": cases},
          open("sdk/tests/fixtures/checkpoint/cases.json", "w"), indent=1, allow_nan=False)
for group in ("coverage", "records"):
    mine = [c for c in cases.values() if c["group"] == group]
    print(group, len(mine), "cases:", sum(c["status"] == "pin" for c in mine), "pins")
