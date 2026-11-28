"""The router registry that `init(model, routers="auto")` reads (lane SR2).

A small table from a Hugging Face model class to where its MoE routers are, how many experts each
has and how many it chooses for a token. `resolve` is called by `capture._plan_session` before any hook
is registered, and only when `routers="auto"` is given; a run that does not pass it never reads
this module.

Rules, each one there because the opposite would count wrong experts with no refusal:

- A model's class matches an entry exactly, by its module and qualified name (`qualified`), or it
  is not in the registry. A subclass, a wrapper (DDP, a PEFT model, a LightningModule) or a class of
  the same name in another module is not matched. Nothing is scanned or guessed.
- Each entry lists one or more layouts: a router module's path under the decoder, the exact class it
  must have, and how its output holds the experts chosen. The layout whose first router is present
  with its exact class is the one used; every other router the configuration places must then be
  there with that class too, or it is refused by name.
- `n_experts` and `top_k` are read from the model's configuration (`model.config`), under the names
  the entry gives, and so is which decoder layers have a router. A value absent or not a positive
  integer refuses the entry's routers by name. A router whose weight rows are not `n_experts`, or
  whose module (or, for a plain `nn.Linear` gate, the block that owns it) holds a `top_k` other than
  the configuration's, is refused by name.
- Two forms. `INDICES`: the router module returns the chosen expert ids, read as the first integer
  tensor of its output; an output with none is refused at its first training forward, never counted
  as the top-k of its logits. `LOGITS`: a plain `nn.Linear` gate whose owner (an exact class too) takes
  the top-k of a softmax of its logits, or of the logits, so the top-k TensorQuill takes of the same
  logits is the model's choice (up to the order of exactly equal logits, which neither top-k defines).

This module imports nothing from transformers, at its top or anywhere: it reads class names,
module paths, configuration attributes and weight shapes only.

What each entry was checked on (each layout's `checked` says which). "Read" is the source at a
transformers tag, in a clone of the transformers repository: 4.46.0 (the lowest version the
`huggingface` extra allows), the last patch release of every minor version from 4.46 to 5.23, and the
clone's main branch at 5.24.0.dev0 (commit 3a134bc of Nov 14, 2026), whose six model files are the
same as 5.23.0's. "Run" is the tiny models of `tests/test_lane_sr2_routers_auto.py` (random weights,
three training steps on CPU, the counts compared with what each model routed), on a Mac with Python
3.12 and PyTorch 2.15.0, Nov 27, 2026: with the installed transformers 5.23.0, and with the source of
4.46.0, 4.51.3, 4.57.6, 5.0.0, 5.6.2, 5.12.1, 5.13.1 and 5.20.2 taken from the clone at the tag and
imported as it is, with its check of dependency versions turned off (and, from 5.0 to 5.13, the
`httpx` package it imports and the environment lacks replaced by a stand-in), since no other
version was installed. The six model files were not changed. PyTorch 2.6 ran no transformers model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

INDICES = "indices"
LOGITS = "logits"

LINEAR = "torch.nn.modules.linear.Linear"


def qualified(cls: type) -> str:
    """A class's module and qualified name, the only thing the registry matches a class on."""
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', getattr(cls, '__name__', ''))}"


@dataclass(frozen=True)
class Layout:
    """Where one version range of a model family keeps its routers.

    `path` is under the decoder (the base model), with `{i}` the layer index. `router` is the exact
    class the module at that path must have. `owner`, for the `LOGITS` form only, is the exact class of
    the module that holds the gate and makes the choice from its logits. `checked` says on what."""

    path: str
    router: str
    form: str
    checked: str
    owner: str | None = None


@dataclass(frozen=True)
class Entry:
    """One model family: its matched classes, each with the path of its decoder ("model." under a
    causal-LM head, "" for the base model), the configuration names of `n_experts` and `top_k`, the
    rule for which decoder layers hold a router (and the configuration names it reads), and its
    layouts, newest first."""

    family: str
    heads: Mapping[str, str]
    n_experts: str
    top_k: str
    layers: Callable[[Any], list[int]]
    layers_from: str
    layouts: Sequence[Layout]


@dataclass
class Resolution:
    """What `resolve` found: the model's class, the entry and layout matched (None when none), each
    router to count with `(n_experts, top_k, form)`, and each one refused with why (a reason with no
    fix: the capture adds how to count it by hand). A class not in the registry, or an entry whose
    routers cannot even be named, is refused under the class's qualified name."""

    model_class: str
    entry: Entry | None = None
    layout: Layout | None = None
    n_experts: int | None = None
    top_k: int | None = None
    routers: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)


class _Absent(Exception):
    """A configuration value the entry needs, absent or not usable: its text is the reason."""


def _int(config: Any, key: str, least: int) -> int:
    value = getattr(config, key, None)
    if value is None:
        raise _Absent(f"the model's configuration has no {key}")
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise _Absent(f"the model's configuration has {key}={value!r}, not an integer of {least} or more")
    return int(value)


def _every_layer(config: Any) -> list[int]:
    return list(range(_int(config, "num_hidden_layers", 1)))


def _qwen_layers(config: Any) -> list[int]:
    """Qwen2-MoE and Qwen3-MoE (`Qwen2MoeDecoderLayer`, `Qwen3MoeDecoderLayer`, the same rule at every tag
    read): a layer has a router unless it is in `mlp_only_layers`, and only when `num_experts` is above 0
    and `(layer + 1) % decoder_sparse_step == 0`."""
    n = _int(config, "num_hidden_layers", 1)
    step = _int(config, "decoder_sparse_step", 1)
    dense = getattr(config, "mlp_only_layers", None)
    if dense is None:
        raise _Absent("the model's configuration has no mlp_only_layers")
    if not isinstance(dense, (list, tuple)) or not all(isinstance(i, int) and not isinstance(i, bool) for i in dense):
        raise _Absent(f"the model's configuration has mlp_only_layers={dense!r}, not a list of layer numbers")
    experts = _int(config, "num_experts", 0)
    return [i for i in range(n) if i not in dense and experts > 0 and (i + 1) % step == 0]


def _deepseek_layers(config: Any) -> list[int]:
    """DeepSeek-V3 (`DeepseekV3DecoderLayer`, every tag read): the layers from `first_k_dense_replace` on."""
    n = _int(config, "num_hidden_layers", 1)
    return list(range(_int(config, "first_k_dense_replace", 0), n))


_READ_4 = "read at 4.46.0 and the last patch of 4.46 to 4.57, run on 4.46.0, 4.51.3 and 4.57.6"
_READ_5 = ("read at the last patch of 5.0 to 5.23 and at 5.24.0.dev0, run on 5.0.0, 5.6.2, 5.12.1, 5.13.1, 5.20.2 "
           "and 5.23.0")


def _entry(family: str, module: str, prefix: str, n_experts: str, top_k: str, layers: Callable[[Any], list[int]],
           layers_from: str, layouts: Sequence[Layout]) -> Entry:
    base = f"transformers.models.{module}.modeling_{module}.{prefix}"
    return Entry(family=family, heads={f"{base}ForCausalLM": "model.", f"{base}Model": ""}, n_experts=n_experts,
                 top_k=top_k, layers=layers, layers_from=layers_from, layouts=tuple(layouts))


def _at(module: str, name: str) -> str:
    return f"transformers.models.{module}.modeling_{module}.{name}"


_QWEN_LAYERS = "num_hidden_layers, mlp_only_layers, decoder_sparse_step and num_experts"

ENTRIES: tuple[Entry, ...] = (
    _entry("Mixtral", "mixtral", "Mixtral", "num_local_experts", "num_experts_per_tok", _every_layer,
           "num_hidden_layers", [
               Layout("layers.{i}.mlp.gate", _at("mixtral", "MixtralTopKRouter"), INDICES,
                      f"5.0 and later, returns (logits or probabilities, weights, indices), {_READ_5}"),
               Layout("layers.{i}.block_sparse_moe.gate", LINEAR, LOGITS,
                      f"4.46 to 4.57, MixtralSparseMoeBlock takes the top-k of the gate's softmax, {_READ_4}",
                      owner=_at("mixtral", "MixtralSparseMoeBlock")),
           ]),
    _entry("OLMoE", "olmoe", "Olmoe", "num_experts", "num_experts_per_tok", _every_layer, "num_hidden_layers", [
        Layout("layers.{i}.mlp.gate", _at("olmoe", "OlmoeTopKRouter"), INDICES,
               f"5.0 and later, returns (logits or probabilities, weights, indices), {_READ_5}"),
        Layout("layers.{i}.mlp.gate", LINEAR, LOGITS,
               f"4.46 to 4.57, OlmoeSparseMoeBlock takes the top-k of the gate's softmax, {_READ_4}",
               owner=_at("olmoe", "OlmoeSparseMoeBlock")),
    ]),
    _entry("Qwen2-MoE", "qwen2_moe", "Qwen2Moe", "num_experts", "num_experts_per_tok", _qwen_layers, _QWEN_LAYERS, [
        Layout("layers.{i}.mlp.gate", _at("qwen2_moe", "Qwen2MoeTopKRouter"), INDICES,
               f"5.0 and later, returns (logits or probabilities, weights, indices), {_READ_5}"),
        Layout("layers.{i}.mlp.gate", LINEAR, LOGITS,
               f"4.46 to 4.57, Qwen2MoeSparseMoeBlock takes the top-k of the gate's softmax, {_READ_4}",
               owner=_at("qwen2_moe", "Qwen2MoeSparseMoeBlock")),
    ]),
    _entry("Qwen3-MoE", "qwen3_moe", "Qwen3Moe", "num_experts", "num_experts_per_tok", _qwen_layers, _QWEN_LAYERS, [
        Layout("layers.{i}.mlp.gate", _at("qwen3_moe", "Qwen3MoeTopKRouter"), INDICES,
               f"5.0 and later, returns (logits or probabilities, weights, indices), {_READ_5}"),
        Layout("layers.{i}.mlp.gate", LINEAR, LOGITS,
               "4.51 to 4.57 (the family is absent before 4.51), Qwen3MoeSparseMoeBlock takes the top-k of the "
               "gate's softmax, read at the last patch of 4.51 to 4.57, run on 4.51.3 and 4.57.6",
               owner=_at("qwen3_moe", "Qwen3MoeSparseMoeBlock")),
    ]),
    _entry("DeepSeek-V3", "deepseek_v3", "DeepseekV3", "n_routed_experts", "num_experts_per_tok", _deepseek_layers,
           "num_hidden_layers and first_k_dense_replace", [
               # One class name, three behaviours: 4.51 to 4.57 return (indices, weights); 5.0 to 5.12 return the
               # logits only and choose in DeepseekV3MoE (sigmoid, correction bias, group-limited top-k), which
               # the INDICES form refuses at the first forward; 5.13 and later return (logits, weights, indices).
               # The family is absent before 4.51.
               Layout("layers.{i}.mlp.gate", _at("deepseek_v3", "DeepseekV3TopkRouter"), INDICES,
                      "4.51 to 4.57 and 5.13 and later, returns the chosen indices, and 5.0 to 5.12 return logits "
                      "only and are refused, read at the last patch of 4.51 to 5.23 and at 5.24.0.dev0, counted on "
                      "4.51.3, 4.57.6, 5.13.1, 5.20.2 and 5.23.0 and refused on 5.0.0, 5.6.2 and 5.12.1"),
           ]),
    _entry("Granite-MoE", "granitemoe", "GraniteMoe", "num_local_experts", "num_experts_per_tok", _every_layer,
           "num_hidden_layers", [
               Layout("layers.{i}.block_sparse_moe.router", _at("granitemoe", "GraniteMoeTopKRouter"), INDICES,
                      "5.13 and later, returns (indices, weights, logits) to 5.20 and (logits, weights, indices) "
                      "from 5.21, read at the last patch of 5.13 to 5.23 and at 5.24.0.dev0, run on 5.13.1, 5.20.2 "
                      "and 5.23.0"),
               # Before 5.13 the router's output holds a sort order of the assignments as its first integer
               # tensor, not expert ids, so its inner linear layer is read instead.
               Layout("layers.{i}.block_sparse_moe.router.layer", LINEAR, LOGITS,
                      "4.46 to 5.12, GraniteMoeTopKGating takes the top-k of this layer's logits, read at 4.46.0 "
                      "and the last patch of 4.46 to 5.12, run on 4.46.0, 4.51.3, 4.57.6, 5.0.0, 5.6.2 and 5.12.1",
                      owner=_at("granitemoe", "GraniteMoeTopKGating")),
           ]),
)

REGISTRY: dict[str, Entry] = {head: entry for entry in ENTRIES for head in entry.heads}


def lookup(cls: type) -> Entry | None:
    """The entry of a model class, matched exactly by `qualified`; None when it has none."""
    return REGISTRY.get(qualified(cls))


def _looked_for(entry: Entry, prefix: str, first: int) -> str:
    return " or ".join(
        f"a {layout.router.rsplit('.', 1)[-1]} at {prefix + layout.path.format(i=first)}" for layout in entry.layouts
    )


def _rows(module: Any) -> int | None:
    shape = getattr(getattr(module, "weight", None), "shape", None)
    return int(shape[0]) if shape is not None and len(shape) == 2 else None


def _problem(name: str, module: Any, owner: Any, layout: Layout, entry: Entry, n_experts: int, top_k: int) -> str | None:
    """Why one router of the chosen layout does not fit the model in hand; None when it does."""
    if module is None:
        return (f"there is no module {name}, where its configuration ({entry.layers_from}) puts a "
                f"{layout.router.rsplit('.', 1)[-1]}")
    if qualified(type(module)) != layout.router:
        return f"{name} is a {qualified(type(module))}, not the {layout.router} the entry reads"
    if layout.owner is not None and qualified(type(owner)) != layout.owner:
        return f"{name} is held by a {qualified(type(owner))}, not the {layout.owner} whose routing the entry read"
    rows = _rows(module)
    if rows != n_experts:
        return f"the weight of {name} has {rows} rows, not the {n_experts} experts of config.{entry.n_experts}"
    held = getattr(module if layout.owner is None else owner, "top_k", None)
    if isinstance(held, int) and not isinstance(held, bool) and held != top_k:
        return f"the top_k {name} is routed with is {held}, not config.{entry.top_k}={top_k}"
    return None


def resolve(model: Any) -> Resolution:
    """The routers `routers="auto"` counts in `model`, and those it refuses, from the registry and the
    model's configuration. Reads class names, module paths, configuration attributes and weight
    shapes; calls nothing in the model and changes nothing."""
    cls = qualified(type(model))
    found = Resolution(model_class=cls)
    entry = REGISTRY.get(cls)
    if entry is None:
        found.refused[cls] = (
            f"routers=\"auto\" found no router: the model's class {cls} is not in TensorQuill's router registry, "
            f"which matches a class exactly by its module and name and has entries for "
            f"{', '.join(e.family for e in ENTRIES)}"
        )
        return found
    found.entry = entry
    matched = f"routers=\"auto\" matched {cls} ({entry.family})"
    prefix = entry.heads[cls]
    config = getattr(model, "config", None)
    try:
        if config is None:
            raise _Absent("the model has no config")
        layers = entry.layers(config)
    except _Absent as absent:
        found.refused[cls] = f"{matched} but cannot name its routers: {absent} (read for {entry.layers_from})"
        return found
    if not layers:
        found.refused[cls] = f"{matched} but its configuration puts a router in no layer ({entry.layers_from})"
        return found
    modules = dict(model.named_modules())
    # The layout whose router in the first layer that has one is there, with the layout's exact class.
    layout = next(
        (lay for lay in entry.layouts
         if prefix + lay.path.format(i=layers[0]) in modules
         and qualified(type(modules[prefix + lay.path.format(i=layers[0])])) == lay.router),
        None,
    )
    if layout is None:
        looked = _looked_for(entry, prefix, layers[0])
        for i in layers:
            found.refused[prefix + entry.layouts[0].path.format(i=i)] = (
                f"{matched} and found no router where the entry puts one: it looked for {looked}"
            )
        return found
    found.layout = layout
    names = [prefix + layout.path.format(i=i) for i in layers]
    try:
        n_experts = _int(config, entry.n_experts, 1)
        top_k = _int(config, entry.top_k, 1)
        if top_k > n_experts:
            raise _Absent(f"the model's configuration has {entry.top_k}={top_k} above {entry.n_experts}={n_experts}")
    except _Absent as absent:
        for name in names:
            found.refused[name] = f"{matched} but {absent}"
        return found
    found.n_experts, found.top_k = n_experts, top_k
    for name in names:
        why = _problem(name, modules.get(name), modules.get(name.rpartition(".")[0]), layout, entry, n_experts, top_k)
        if why is None:
            found.routers[name] = (n_experts, top_k, layout.form)
        else:
            found.refused[name] = f"{matched} but {why}"
    if layout.router != LINEAR:
        # A router of the entry's class in a layer the configuration gives none is refused by name, not
        # left out unsaid.
        for name, module in modules.items():
            if name not in found.routers and name not in found.refused and qualified(type(module)) == layout.router:
                found.refused[name] = (
                    f"{matched} but {name} is in a layer its configuration gives no router ({entry.layers_from})"
                )
    return found


def chosen_line(found: Resolution) -> str:
    """The one line `init` logs for what `resolve` chose."""
    entry, layout = found.entry, found.layout
    assert entry is not None and layout is not None
    return (
        f"routers=\"auto\": {found.model_class} matched the registry entry for {entry.family} "
        f"({layout.form} form, a {layout.router.rsplit('.', 1)[-1]} at {layout.path}, {layout.checked}). "
        f"Counting {len(found.routers)} router(s) with n_experts={found.n_experts} (config.{entry.n_experts}) "
        f"and top_k={found.top_k} (config.{entry.top_k}): {', '.join(found.routers) or 'none'}"
    )
