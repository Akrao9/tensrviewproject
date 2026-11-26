"""What TensorQuill records for one model split by layer over two devices in one process. The
multi-GPU test plan's T4.

The model's decoder blocks are split in halves: the first half, the embedding, the final norm
and the head on the first device of `--devices`, the second half of the blocks on the second (the
head stays with the embedding, so a tied head is never split from it). The session's buffers are
then on the first device (its first trainable parameter's), and every block of the second half
has its statistics copied across. Arms (`--arm`):

- `device-map`: Accelerate's `dispatch_model` with that map (`AlignDevicesHook` on each block
  moves its inputs; on one device the hooks are forced, `force_hooks=True`, as Accelerate adds
  none otherwise);
- `manual`: no Accelerate: the second half moved with `.to()` and a forward pre-hook on each of
  its blocks that moves its inputs there, the output brought back after the last;
- `moe`: an MoE model (default Granite 3.0 1B-A400M) split by `device_map`. A first session
  names its routers with `tensorquill.Router(experts, top_k)`; every router on the second device
  must be refused for the session (`moe_refused`) with the `indices=` fix named, and the routers
  on the first counted. A second session takes the fix, `indices=` returning the router's
  indices moved to the counts' device: every router must then be counted, and its counts must
  sum to tokens x top_k on every step;
- `offload`: Accelerate's offload of the second half (to the CPU from a GPU; to disk when the
  main device is the CPU, since Accelerate does not offload to the CPU then): the offloaded
  parameters are on `meta`, and each one's `weight_norms` must be `"nan"`, never a number.

The dense arms (`device-map`, `manual`) plant, as `fsdp_check.py` does, a gradient spike (at
`--grad-step`, one parameter of the last block on the second device, its gradient multiplied by
`--grad-factor` in place after the backward) and an activation spike (from `--act-step` on, the
first block on the second device's output times `--act-factor`), and fail unless:

- every record has a `layers` entry for every block, and no record says `layers_not_read`;
- the activation detector finds the spike at `--act-step`, first in the planted block;
- `grad_norm` equals the script's own total over both devices (each gradient's norm read where
  it is) on every step, within `--tolerance`; the `grad_spike` incident is at `--grad-step` and
  lists the planted parameter;
- `weight_norms` of every parameter on the second device equal the script's own after the step,
  on every weight check;
- with two CUDA devices, timed steps under `torch.cuda.set_sync_debug_mode("warn")`: the same
  `--timed-steps` steps with capture off and then on, each step's synchronising calls counted
  (`warnings.simplefilter("always")`, so a call site that warns again is counted again). The
  rule: on no step that is not a flush step (the `--every`-th of the session) may capture on
  count more than capture off did on the same step of its run; flush steps are reported, not
  judged. The first `--timed-warmup` steps of each run are left out.

NOT PROVEN (exit 3), with every other check still made: both devices are the same (no
cross-device copy), the sync count cannot be read (not CUDA), no router on another device than
the counts (`moe`), the offload went to disk (`offload` on the CPU), or the script's own total
gradient norm at `--grad-step` is not the detector's factor over its median before.
`verdict.json` lists, under `not_exercised`, what the run could not show.

    python examples/split_model_check.py --out runs/split-device-map --arm device-map
    python examples/split_model_check.py --out runs/split-manual --arm manual
    python examples/split_model_check.py --out runs/split-moe --arm moe
    python examples/split_model_check.py --out runs/split-offload --arm offload
    python examples/split_model_check.py --toy --devices cpu,cpu --arm device-map --out runs/split-cpu

`--model` names the model (default SmolLM2-360M for the dense arms, Granite for `moe`), loaded in
float32 from the Hugging Face cache or the network, with WikiText-2 batches as
`dense_outlier_check.py` loads them; `--toy` uses a tiny random Llama (GraniteMoe for `moe`) on
random tokens. Writes `<arm>.jsonl` (and `timed.jsonl`, `fixed.jsonl`) and `verdict.json` under
`--out`. Exit status: 0 PASS, 1 FAIL, 2 bad arguments or a missing requirement, 3 NOT PROVEN.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

import tensorquill

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_PROVEN = 0, 1, 2, 3
ARMS = ("device-map", "manual", "moe", "offload")
DENSE_MODEL = "HuggingFaceTB/SmolLM2-360M"
MOE_MODEL = "ibm-granite/granite-3.0-1b-a400m-base"
SYNC_WARNING = "synchronizing"  # "called a synchronizing CUDA operation" (torch/csrc/cuda/...: sync debug mode)


# ---- the pieces -------------------------------------------------------------------------------------


def split_map(model: nn.Module, first: str, second: str, offload_to: str | None = None) -> tuple[dict[str, str], list[str]]:
    """A `device_map` for a Hugging Face decoder (`model.model.layers`): every block of the second half
    on `second` (or `offload_to`), everything else on `first`. Returns it and the second half's names."""
    inner = model.model
    blocks = len(inner.layers)
    device_map: dict[str, str] = {}
    for name, _ in model.named_children():
        if name != "model":
            device_map[name] = first
    for name, _ in inner.named_children():
        if name != "layers":
            device_map[f"model.{name}"] = first
    second_half = []
    for i in range(blocks):
        if i < blocks // 2:
            device_map[f"model.layers.{i}"] = first
        else:
            device_map[f"model.layers.{i}"] = offload_to or second
            second_half.append(f"model.layers.{i}")
    return device_map, second_half


def sync_verdict(off: list[int], on: list[int], flush: list[bool]) -> dict[str, Any]:
    """The synchronisation rule: on every step that is not a flush step, capture on counts no more
    synchronising calls than capture off did on the same step of its run. `added` maps each step that
    broke it to how many more; `flush_extra` is what the flush steps added, reported only."""
    added = {i: on[i] - off[i] for i in range(min(len(off), len(on))) if not flush[i] and on[i] > off[i]}
    flush_extra = sum(on[i] - off[i] for i in range(min(len(off), len(on))) if flush[i])
    return {"passed": not added, "added": added, "flush_extra": flush_extra, "off": off, "on": on, "flush": flush}


def _indexed(device: torch.device) -> torch.device:
    """`cuda` and `mps` as `cuda:0` and `mps:0`, the device a tensor put there reports, so that comparing a
    parameter's device with the one asked for is not wrong by its missing index."""
    if device.type in ("cuda", "mps") and device.index is None:
        return torch.device(device.type, 0)
    return device


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value


def place_manually(model: nn.Module, first: torch.device, second: torch.device) -> list[str]:
    """The second half of the blocks moved with `.to(second)`, as a user without Accelerate would: a
    pre-hook on each moves its inputs there, and the last one's output comes back to `first`."""
    model.to(first)
    layers = model.model.layers
    second_half = []
    for i in range(len(layers) // 2, len(layers)):
        layers[i].to(second)
        layers[i].register_forward_pre_hook(lambda _m, a, k, d=second: (_move(a, d), _move(k, d)), with_kwargs=True)
        second_half.append(f"model.layers.{i}")
    layers[len(layers) - 1].register_forward_hook(lambda _m, _a, out, d=first: _move(out, d))
    return second_half


def build_model(args, arm: str):
    """The model in float32, on the CPU, and its batches (token ids on the CPU)."""
    from transformers import AutoModelForCausalLM

    batches = args.steps + args.timed_steps * 2
    if args.toy:
        torch.manual_seed(0)
        if arm == "moe":
            from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

            config = GraniteMoeConfig(vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                                      num_experts_per_tok=2, tie_word_embeddings=True)
            model = GraniteMoeForCausalLM(config)
        else:
            from transformers import LlamaConfig, LlamaForCausalLM

            config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                 num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
                                 tie_word_embeddings=True)
            model = LlamaForCausalLM(config)
        tokens = torch.randint(0, 128, (batches, 2, 32), generator=torch.Generator().manual_seed(1))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dense_outlier_check import load_batches

        name = args.model or (MOE_MODEL if arm == "moe" else DENSE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa")
        tokens = load_batches(name, args.batch, args.seq_len, batches, "cpu")
    model.config.use_cache = False
    return model, tokens


@torch.no_grad()
def output_devices(model: nn.Module, x: torch.Tensor, modules: dict[str, nn.Module], pick=None) -> dict[str, str]:
    """The device of each module's output (`pick(output)`, else the first tensor in it) in one forward."""
    seen: dict[str, str] = {}
    handles = []
    for name, module in modules.items():
        def hook(_m, _a, output, name=name):
            value = pick(output) if pick is not None else output
            while isinstance(value, (tuple, list)) and value:
                value = value[0]
            if isinstance(value, torch.Tensor):
                seen[name] = str(value.device)
        handles.append(module.register_forward_hook(hook))
    try:
        model(input_ids=x)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def _layer_names(model: nn.Module) -> dict[int, str]:
    """TensorQuill's name for each block, by the block's object."""
    return {id(module): name for name, module in tensorquill.capture._select_layers(model, "blocks")}


# ---- the run --------------------------------------------------------------------------------------


class Checks:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.not_proven: list[str] = []
        self.not_exercised: list[str] = []

    def check(self, ok: bool, text: str) -> bool:
        self.lines.append(f"{'PASS' if ok else 'FAIL'}: {text}")
        print(self.lines[-1], flush=True)
        if not ok:
            self.failures.append(text)
        return ok

    def unproven(self, what: str, text: str) -> None:
        self.lines.append(f"NOT PROVEN: {text}")
        print(self.lines[-1], flush=True)
        self.not_proven.append(text)
        if what not in self.not_exercised:
            self.not_exercised.append(what)

    def report(self, text: str) -> None:
        self.lines.append(f"REPORTED: {text}")
        print(self.lines[-1], flush=True)


def train(model, tokens, first: torch.device, steps: range, session=None, *, optimizer, plant=None, own=None,
          sync=False) -> list[int]:
    """Steps of training on `tokens[i]` for each i in `steps`. With `own`, the script's numbers are kept
    (outside any counted window); with `sync`, each step's synchronising calls are counted and returned."""
    counts = []
    for i in steps:
        x = tokens[i].to(first)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            if plant is not None:
                plant(i, "forward")
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            if plant is not None:
                plant(i, "backward")
            if own is not None:
                own(i, "grads")
            optimizer.step()
            if own is not None:
                own(i, "weights")
            optimizer.zero_grad()
        if sync:
            counts.append(sum(1 for w in seen if SYNC_WARNING in str(w.message)))
    return counts


def run(args) -> int:
    first, second = (torch.device(d) for d in args.devices)
    c = Checks()
    out: Path = args.out
    arm = args.arm
    try:
        model, tokens = build_model(args, arm)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"split_model_check.py: cannot load the model or WikiText-2 ({type(error).__name__}: {error}); install "
              "transformers, accelerate, huggingface_hub and pyarrow and give the machine network access or a filled "
              "Hugging Face cache", file=sys.stderr)
        return EXIT_USAGE
    out.mkdir(parents=True)
    document: dict[str, Any] = {"check": "split_model", "arm": arm, "devices": [str(first), str(second)],
                                "torch": torch.__version__, "toy": bool(args.toy)}
    # ---- placement
    offload_to = None
    if arm == "manual":
        second_half = place_manually(model, first, second)
    else:
        from accelerate import dispatch_model

        if arm == "offload":
            offload_to = "cpu" if first.type == "cuda" else "disk"
        device_map, second_half = split_map(model, str(first), str(second), offload_to)
        model = dispatch_model(model, device_map, main_device=str(first), force_hooks=True,
                               offload_dir=str(out / "offload") if offload_to == "disk" else None)
    document["second_half"] = second_half
    document["offload_to"] = offload_to
    document["accelerate_hooks"] = sum(1 for m in model.modules() if hasattr(m, "_hf_hook"))
    blocks = model.model.layers
    names = _layer_names(model)
    block_names = [names.get(id(b)) for b in blocks]
    params = dict(model.named_parameters())
    on_second = [n for n, p in params.items() if p.device == second and any(n.startswith(b + ".") for b in second_half)]
    print(f"{torch.__version__} | {arm} | {'toy' if args.toy else args.model or 'default model'} | devices {first}, {second} | "
          f"second half {second_half} | Accelerate hooks {document['accelerate_hooks']}", flush=True)
    # Where each block's output really is, from one forward before any session: a device_map's "cpu"
    # entry, beside a GPU or MPS main device, runs on the main device (Accelerate's execution device).
    where = output_devices(model, tokens[0].to(first), {name: block for name, block in zip(block_names, blocks)})
    document["block_devices"] = where
    if first == second:
        c.unproven("cross-device copy", f"both halves are on {first}: no block's statistics are copied between devices")
    elif str(second) not in where.values():
        c.unproven("cross-device copy", f"no block's output is on {second} ({sorted(set(where.values()))}): the split "
                                        "put nothing there")
    if arm == "moe":
        return finish(out, document, c, moe_arm(args, model, tokens, first, second, second_half, c, document))
    if arm == "offload":
        return finish(out, document, c, offload_arm(args, model, tokens, first, second_half, c, document))

    # ---- the plants, before the session's hooks
    act_index, grad_index = len(blocks) // 2, len(blocks) - 1
    act_name, grad_block = block_names[act_index], block_names[grad_index]
    document["act_block"] = act_name
    flag = {"on": False}
    def act_plant(_module, _inputs, output):
        if not flag["on"]:
            return output
        if isinstance(output, tuple):  # a decoder layer of an older Transformers returns (hidden, ...)
            return (output[0] * args.act_factor, *output[1:])
        return output * args.act_factor

    blocks[act_index].register_forward_hook(act_plant)
    candidates = [(p.numel(), n) for n, p in params.items() if n.startswith(grad_block + ".") and p.dim() == 2]
    planted = max(candidates)[1]
    document["grad_plant"] = {"parameter": planted, "factor": args.grad_factor, "device": str(params[planted].device)}

    def plant(step: int, phase: str) -> None:
        if phase == "forward":
            flag["on"] = step >= args.act_step
        elif step == args.grad_step and params[planted].grad is not None:
            params[planted].grad.mul_(args.grad_factor)

    own_grad: list[dict[str, float]] = []
    own_weight: list[dict[str, float]] = []

    @torch.no_grad()
    def own(_step: int, phase: str) -> None:
        if phase == "grads":
            # Read on the CPU in float64: the script's own numbers, whatever the device (MPS has no float64).
            own_grad.append({n: float(p.grad.detach().cpu().double().norm()) for n, p in params.items()
                             if p.grad is not None})
        else:
            own_weight.append({n: float(p.detach().cpu().double().norm()) for n, p in params.items()
                               if p.device.type != "meta"})

    jsonl = out / f"{arm}.jsonl"
    session = tensorquill.init(model, run_id=f"split-{arm}", sink=tensorquill.JsonlSink(jsonl), capture="full",
                               every=args.every, weights=args.every, heartbeat_seconds=None, stack_seconds=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    try:
        train(model, tokens, first, range(args.steps), session, optimizer=optimizer, plant=plant, own=own)
    finally:
        session.close()
    flag["on"] = False
    records = [r for r in tensorquill.load_records(jsonl) if r.get("record") != "scalars"]

    # ---- the checks
    c.check([r["step"] for r in records] == list(range(args.steps)), f"{len(records)} records, steps 0 to {args.steps - 1}")
    missing = sorted({b for r in records for b in block_names if b not in r.get("layers", {})})
    c.check(None not in block_names and not missing, f"every block has a layers entry on every record (missing: {missing})")
    not_read = [r["layers_not_read"] for r in records if "layers_not_read" in r]
    c.check(not not_read, f"no layers_not_read ({not_read[:1]})")
    evidence = tensorquill.build_evidence(records)
    act = next((i for i in evidence["incidents"] if i["kind"] == "activation_spike"), None)
    found = None if act is None else {"step": act["step"], "first": act["modules"][0]["module"]}
    c.check(found == {"step": args.act_step, "first": act_name},
            f"activation spike at step {args.act_step}, first in {act_name} on {second} (found {found})")
    worst, problems = 0.0, []
    for record, mine in zip(records, own_grad):
        want = math.sqrt(sum(v * v for v in mine.values()))
        got = record.get("grad_norm")
        if not isinstance(got, (int, float)):
            problems.append(f"step {record['step']}: grad_norm {got!r}")
            continue
        worst = max(worst, abs(got - want) / max(want, 1e-12))
    c.check(len(own_grad) == args.steps and not problems and worst < args.tolerance,
            f"grad_norm equals the script's own total over both devices on {args.steps - len(problems)} of {args.steps} "
            f"steps, largest relative difference {worst:.2e} (tolerance {args.tolerance:g})")
    totals = [math.sqrt(sum(v * v for v in mine.values())) for mine in own_grad]
    before = totals[max(0, args.grad_step - 50):args.grad_step]
    from tensorquill.evidence import SPIKE_FACTOR

    ratio = totals[args.grad_step] / statistics.median(before) if before and statistics.median(before) > 0 else None
    document["grad_plant"]["whole_ratio"] = ratio
    if ratio is None or ratio < SPIKE_FACTOR:
        c.unproven("gradient spike", f"the script's own total at step {args.grad_step} is x{ratio} its median, under "
                                     f"the detector's x{SPIKE_FACTOR:g}: no spike to find")
    spike = next((i for i in evidence["incidents"] if i["kind"] == "grad_spike"), None)
    listed = [] if spike is None else [p["name"] for p in spike.get("parameters", [])]
    text = (f"grad_spike at step {args.grad_step} listing {planted} on {params[planted].device} "
            f"(found {None if spike is None else spike['step']}, lists it: {planted in listed})")
    if ratio is not None and ratio >= SPIKE_FACTOR:
        c.check(spike is not None and spike["step"] == args.grad_step and planted in listed, text)
    else:
        c.report(f"not judged, the plant made no spike: {text}")
    checks, wworst, wproblems = 0, 0.0, []
    for record, mine in zip(records, own_weight):
        if "weight_norms" not in record:
            continue
        checks += 1
        for name in on_second:
            got = record["weight_norms"].get(name)
            if not isinstance(got, (int, float)):
                wproblems.append(f"step {record['step']}: {name} {got!r}")
                continue
            wworst = max(wworst, abs(got - mine[name]) / max(mine[name], 1e-12))
    if not on_second:
        c.unproven("parameters on the second device", f"no parameter of the second half is on {second}: "
                                                         "its weight norms cannot be compared")
    c.check(checks > 0 and not wproblems and wworst < args.tolerance,
            f"weight_norms of the {len(on_second)} parameters on {second} equal the script's own on {checks} weight checks, "
            f"largest relative difference {wworst:.2e} ({wproblems[:2]})")

    # ---- the synchronising calls
    why = None if first.type == second.type == "cuda" else f"the devices are {first.type} and {second.type}, not CUDA"
    if why is not None:
        c.unproven("synchronisation count", f"torch.cuda.set_sync_debug_mode cannot be read here: {why}")
    else:
        sync = timed(args, model, tokens, first, out)
        document["sync"] = sync
        c.check(sync["passed"], f"under set_sync_debug_mode('warn'), capture on added no synchronising call on a step "
                                f"that is not a flush step ({sync['added'] or 'none added'}; flush steps added "
                                f"{sync['flush_extra']}, reported only)")
    return finish(out, document, c)


def timed(args, model, tokens, first: torch.device, out: Path) -> dict[str, Any]:
    warm = args.timed_warmup
    start = args.steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        off = train(model, tokens, first, range(start, start + args.timed_steps), optimizer=optimizer, sync=True)
        session = tensorquill.init(model, run_id="split-timed", sink=tensorquill.JsonlSink(out / "timed.jsonl"),
                                   capture="full", every=args.every, heartbeat_seconds=None, stack_seconds=None)
        try:
            on = train(model, tokens, first, range(start + args.timed_steps, start + 2 * args.timed_steps), session,
                       optimizer=optimizer, sync=True)
        finally:
            torch.cuda.set_sync_debug_mode(0)
            session.close()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    flush = [(i + 1) % args.every == 0 for i in range(args.timed_steps)]
    verdict = sync_verdict(off[warm:], on[warm:], flush[warm:])
    verdict["warmup_left_out"] = warm
    return verdict


def moe_arm(args, model, tokens, first, second, second_half, c: Checks, document) -> None:
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeTopKRouter

    routers = {name: m for name, m in model.named_modules() if isinstance(m, GraniteMoeTopKRouter)}
    # The routers whose chosen indices are on another device than the counts, which are on the session's
    # device, the first trainable parameter's (`first`), seen in one forward before any session.
    where = output_devices(model, tokens[0].to(first), routers, pick=lambda out: out[2])
    document["moe"] = {"router_devices": where}
    on_second = sorted(n for n, d in where.items() if d != str(first))
    tokens_per_step = tokens.shape[1] * tokens.shape[2]
    document["moe"].update(routers=sorted(routers), on_second=on_second)
    if not on_second:
        c.unproven("router on another device", f"no router is on another device than the counts ({first})")
    half = args.steps // 2
    results = {}
    for label, fixed in (("plain", False), ("fixed", True)):
        spec = {
            name: tensorquill.Router(m.num_experts, m.top_k, indices=(lambda out, d=first: out[2].to(d)) if fixed else None)
            for name, m in routers.items()
        }
        jsonl = args.out / f"{'moe' if not fixed else 'fixed'}.jsonl"
        session = tensorquill.init(model, run_id=f"split-moe-{label}", sink=tensorquill.JsonlSink(jsonl), capture="full",
                                   every=args.every, routers=spec, heartbeat_seconds=None, stack_seconds=None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        steps = range(0, half) if not fixed else range(half, args.steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # the refusal's own warning; its record is what is read
            try:
                train(model, tokens, first, steps, session, optimizer=optimizer)
            finally:
                session.close()
        results[label] = [r for r in tensorquill.load_records(jsonl) if r.get("record") != "scalars"]
    plain, fixed = results["plain"], results["fixed"]
    refused = (plain[0].get("moe_refused") or {}) if plain else {}
    document["moe"]["refused_without_fix"] = sorted(refused)
    c.check(sorted(refused) == on_second,
            f"without indices=, exactly the routers off {first} are refused ({len(refused)} refused, {len(on_second)} there)")
    c.check(all("indices=" in why for why in refused.values()),
            f"each refusal names the indices= fix ({next(iter(refused.values()), 'none refused')[:160]})")
    counted = sorted({n for r in plain for n in r.get("moe", {})})
    c.check(counted == sorted(set(routers) - set(on_second)), f"without indices=, the routers on {first} are counted ({len(counted)})")
    off_by = []
    for record in fixed:
        for name, spec in routers.items():
            row = record.get("moe", {}).get(name)
            if row is None or sum(row) != tokens_per_step * spec.top_k:
                off_by.append(f"step {record['step']}: {name} {None if row is None else sum(row)}")
    c.check(bool(fixed) and not off_by and not fixed[0].get("moe_refused"),
            f"with the fix every router is counted and its counts sum to tokens x top_k ({tokens_per_step} x "
            f"{next(iter(routers.values())).top_k}) on every step of {len(fixed)} ({off_by[:3]})")


def offload_arm(args, model, tokens, first, second_half, c: Checks, document) -> None:
    offloaded = sorted(n for n, p in model.named_parameters() if p.device.type == "meta")
    document["offloaded"] = offloaded
    if not offloaded:
        c.unproven("offload", "no parameter was offloaded (none is on meta)")
    if document["offload_to"] == "disk":
        c.unproven("offload to the CPU", f"the main device is {first}, so Accelerate offloaded to disk, not to the CPU "
                                         "(the same meta parameters and per-forward loading)")
    jsonl = args.out / "offload.jsonl"
    session = tensorquill.init(model, run_id="split-offload", sink=tensorquill.JsonlSink(jsonl), capture="full",
                               every=args.every, weights=args.every, heartbeat_seconds=None, stack_seconds=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    try:
        train(model, tokens, first, range(args.steps), session, optimizer=optimizer)
    finally:
        session.close()
    records = [r for r in tensorquill.load_records(jsonl) if r.get("record") != "scalars"]
    c.check([r["step"] for r in records] == list(range(args.steps)), f"{len(records)} records, steps 0 to {args.steps - 1}")
    names = _layer_names(model)
    block_names = [names.get(id(b)) for b in model.model.layers]
    missing = sorted({b for r in records for b in block_names if b not in r.get("layers", {})})
    c.check(None not in block_names and not missing, f"every block has a layers entry on every record (missing: {missing})")
    checks = [r for r in records if "weight_norms" in r]
    wrong = sorted({n for r in checks for n in offloaded if r["weight_norms"].get(n) != "nan"})
    c.check(bool(checks) and bool(offloaded) and not wrong,
            f"weight_norms is \"nan\" for every offloaded parameter on {len(checks)} weight checks ({wrong[:3]})")
    read = sorted({n for r in checks for n, v in r["weight_norms"].items() if n not in offloaded and not isinstance(v, (int, float))})
    c.check(not read, f"every parameter not offloaded has a number in weight_norms ({read[:3]})")
    with_grad = sorted({n for r in records for n in r.get("grad_norms", {}) if n in offloaded})
    named = sorted({n for r in records for n in (r.get("grads_not_read") or {}) if n in offloaded})
    document["offloaded_gradients"] = {"in_grad_norms": with_grad, "in_grads_not_read": named}
    c.report(f"offloaded parameters with a gradient norm: {len(with_grad)}; named in grads_not_read: {len(named)} of "
             f"{len(offloaded)} (under Accelerate's offload their gradients go to the temporary weights loaded for the "
             "forward, so the parameters the model holds get none)")


def finish(out: Path, document: dict, c: Checks, _: Any = None) -> int:
    status = "FAIL" if c.failures else "NOT PROVEN" if c.not_proven else "PASS"
    code = {"FAIL": EXIT_FAIL, "NOT PROVEN": EXIT_NOT_PROVEN, "PASS": EXIT_PASS}[status]
    document.update(status=status, exit=code, passed=status == "PASS", failures=c.failures, not_proven=c.not_proven,
                    not_exercised=c.not_exercised, lines=c.lines)
    (out / "verdict.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"\n{status}: {out / 'verdict.json'}", flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="new directory for the run")
    p.add_argument("--arm", choices=ARMS, default="device-map")
    p.add_argument("--model", help=f"default {DENSE_MODEL} ({MOE_MODEL} for --arm moe)")
    p.add_argument("--toy", action="store_true", help="a tiny random model on random tokens")
    p.add_argument("--devices", default="cuda:0,cuda:1", help="the two devices, first,second")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--every", type=int, default=10, help="steps between transfers, and between weight checks")
    p.add_argument("--grad-step", type=int, default=25)
    p.add_argument("--grad-factor", type=float, default=1000.0,
                   help="one parameter among hundreds carries a small share of the total norm")
    p.add_argument("--act-step", type=int, default=40)
    p.add_argument("--act-factor", type=float, default=50.0)
    p.add_argument("--timed-steps", type=int, default=30)
    p.add_argument("--timed-warmup", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args(argv)

    def usage(text: str) -> int:
        p.print_usage(sys.stderr)
        print(f"split_model_check.py: error: {text}", file=sys.stderr)
        return EXIT_USAGE

    args.devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(args.devices) != 2:
        return usage("--devices takes two devices, first,second (the same one twice for a dry run)")
    if not 10 <= args.grad_step < args.act_step < args.steps:
        return usage("--grad-step must be at least 10, before --act-step, which is before --steps")
    if args.timed_steps <= args.timed_warmup:
        return usage("--timed-steps must be more than --timed-warmup")
    try:
        devices = [_indexed(torch.device(d)) for d in args.devices]
    except RuntimeError as error:
        return usage(f"--devices: {error}")
    args.devices = [str(d) for d in devices]
    cuda = [d for d in devices if d.type == "cuda"]
    if cuda and torch.cuda.device_count() <= max(d.index or 0 for d in cuda):
        return usage(f"--devices {','.join(args.devices)} needs CUDA devices this machine does not have "
                     f"({torch.cuda.device_count()} found); use --toy --devices cpu,cpu to check the script")
    if args.out.exists():
        return usage(f"{args.out} exists; choose a new --out")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
