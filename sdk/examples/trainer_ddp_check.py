"""Acceptance check for the Hugging Face `Trainer` callback on more than one process (DDP).

`trainer_check.py`'s run under `torchrun`: the same small dense model, the same outlier planted in
one middle block from `--seed-step` on every rank, `torch_compile=True`, gradient clipping and, on
CUDA, fp16 with Accelerate's `GradScaler`. Each rank writes its own capture through
`tensorquill.huggingface.TensorQuillCallback`; a second callback keeps, on every rank, the
`Trainer`'s own `global_step` at each `on_step_end` and whether Accelerate skipped that optimizer
step (`AcceleratedOptimizer.step_was_skipped`, the flag `Trainer` itself reads before stepping its
scheduler). Rank 0 waits for every rank, then fails unless:

- there is one capture per rank, ranks 0 to world size - 1, each closed cleanly, every record
  naming its own rank, and the ranks' losses differ (each trained on its own shard);
- each rank's recorded steps are exactly the `global_step - 1` its `Trainer` reached at each
  `on_step_end`, 0 to `--steps` - 1, ending at `global_step == --steps`;
- `train/loss` and `clip_grad_norm` are recorded on every step of every rank; the mean over ranks
  of each step's recorded loss, averaged over each logging interval, equals the loss `Trainer`
  logged (`Trainer` gathers and averages it over the ranks), and every rank's `clip_grad_norm` at
  each logged step equals the `grad_norm` `Trainer` logged (DDP has averaged the gradients);
- under fp16, every rank's `step_skipped` flags equal, step by step, the skips Accelerate reported,
  and `grad_scale` is on every record; how many there were is printed, whatever it is;
- on every rank the activation detector finds the outlier at the seeded step, first in the seeded
  block; with compile, no frame compiled twice and no graph break outside the breaks of
  `DistributedDataParallel.forward` itself (`DDP_WRAPPER_BREAKS`), which `Trainer` compiles around
  the model; those are counted and printed.

    torchrun --nproc-per-node 2 examples/trainer_ddp_check.py --out runs/trainer-ddp
    GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run --nproc-per-node 2 \\
        examples/trainer_ddp_check.py --toy --eager --out runs/trainer-ddp-cpu

`--toy` is a tiny random Llama on random tokens, on CPU with gloo. Exit 0 when every check passes, 1
when one fails; ranks other than 0 exit 0 once rank 0 has written `result.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import tensorquill

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_outlier_check import FAULTS, MODEL, install_outlier, load_batches, outlier_channels  # noqa: E402
from trainer_check import (  # noqa: E402
    agrees, recompiled_frames, recorded, reset_compile_state, trainer_agreement, _number,
)

RUN_ID = "trainer-ddp-check"
WAIT_SECONDS = 180.0


def _write_json(path: Path, document) -> None:
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_text(json.dumps(document), encoding="utf-8")
    os.replace(scratch, path)


def _wait_for(paths: list[Path], seconds: float) -> list[Path]:
    deadline = time.time() + seconds
    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing or time.time() > deadline:
            return missing
        time.sleep(0.1)


def mean_over_ranks(per_rank: list[list]) -> list:
    """Each step's loss averaged over the ranks, as `Trainer` gathers it; None where a rank has none."""
    out = []
    for values in zip(*per_rank):
        numbers = [_number(v) for v in values]
        out.append(None if None in numbers else sum(numbers) / len(numbers))
    return out


# Breaks in `torch.nn.parallel.DistributedDataParallel.forward` itself, which `Trainer` compiles when the
# model is wrapped (Accelerate compiles the DDP module): its `Logger.set_runtime_stats_and_log` and its
# `_inside_ddp_forward`, which Dynamo skips on purpose. Seen on 2 CPU ranks under aot_eager, with
# PyTorch 2.15; they are torch's, not the model's or TensorQuill's hooks'.
DDP_WRAPPER_BREAKS = ("DistributedDataParallel._inside_ddp_forward", "set_runtime_stats_and_log` of class `Logger`")


def split_breaks(counts: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Dynamo's graph-break counter split into the DDP wrapper's own breaks and every other."""
    wrapper = {k: v for k, v in counts.items() if any(mark in k for mark in DDP_WRAPPER_BREAKS)}
    return wrapper, {k: v for k, v in counts.items() if k not in wrapper}


def skip_agreement(records: list[dict], skipped: list[bool]) -> list[str]:
    """Where a rank's recorded `step_skipped` differs from what Accelerate reported for the same step
    (`skipped[i]` for step i). Empty when every step agrees and every record has `grad_scale`."""
    problems = []
    by_step = {r["step"]: r for r in records}
    for step, was in enumerate(skipped):
        record = by_step.get(step)
        if record is None:
            problems.append(f"step {step}: not recorded")
            continue
        if bool(record.get("step_skipped")) != bool(was):
            problems.append(f"step {step}: recorded step_skipped={bool(record.get('step_skipped'))}, "
                            f"Accelerate step_was_skipped={bool(was)}")
        if "grad_scale" not in record:
            problems.append(f"step {step}: no grad_scale")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4, help="per device")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed-step", type=int, default=250, help="step the outlier starts")
    parser.add_argument("--fault", choices=FAULTS, default="massive")
    parser.add_argument("--channels", type=int, nargs="+", default=[7, 300])
    parser.add_argument("--factor", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50, help="Trainer's logging_steps")
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], help="default: fp16 on CUDA, else fp32")
    parser.add_argument("--eager", action="store_true", help="do not compile")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend (aot_eager to skip Inductor)")
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, CPU allowed")
    args = parser.parse_args(argv)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 2:
        parser.error("run under torchrun with --nproc-per-node 2 or more (trainer_check.py is the one-process check)")
    cuda = torch.cuda.is_available() and not args.toy
    if not args.toy and not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")
    if not 50 <= args.seed_step < args.steps:
        parser.error("--seed-step must be at least 50 and before --steps")
    precision = args.precision or ("fp16" if cuda else "fp32")
    if precision != "fp32" and not cuda:
        parser.error("mixed precision needs CUDA")
    if args.out.exists() and any(p.name != "trainer" for p in args.out.iterdir()) and rank == 0:
        parser.error(f"{args.out} is not empty; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments

    from tensorquill.huggingface import TensorQuillCallback

    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    reset_compile_state()
    torch.manual_seed(0)  # the same weights on every rank
    rows_needed = args.steps * world
    if args.toy:
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)
        args.channels = [c % config.hidden_size for c in args.channels]
        args.batch_size, batches = 2, torch.randint(0, 128, (rows_needed, 2, 32))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
        batches = load_batches(args.model, args.batch_size, args.seq_len, rows_needed, "cpu")
    model.config.use_cache = False
    model.to(device)
    rows = batches.reshape(-1, batches.shape[-1])
    dataset = [{"input_ids": row, "labels": row} for row in rows]  # in order; Accelerate shards it by rank
    fixed = batches[0].to(device)

    blocks = [name for name, _ in tensorquill.capture._select_layers(model, "blocks")]
    seeded_block = blocks[len(blocks) // 2]
    channels = outlier_channels(model, seeded_block, args.fault, args.channels, model.config.hidden_size, fixed)
    flag, handle = install_outlier(
        model.get_submodule(seeded_block), model.config.hidden_size, channels, args.factor, device,
        next(model.parameters()).dtype,
    )

    class SeedOutlier(TrainerCallback):
        def on_step_begin(self, _args, state, control, **kwargs):
            if state.global_step == args.seed_step:
                flag.fill_(True)

    class TrainerOwn(TrainerCallback):
        """What the Trainer itself says each step: its global_step and whether Accelerate skipped it."""

        def __init__(self) -> None:
            self.global_steps: list[int] = []
            self.skipped: list[bool] = []

        def on_step_end(self, _args, state, control, optimizer=None, **kwargs):
            self.global_steps.append(state.global_step)
            self.skipped.append(bool(getattr(optimizer, "step_was_skipped", False)))

    own = TrainerOwn()
    jsonl = args.out / f"{RUN_ID}.rank{rank}.jsonl"
    callback = TensorQuillCallback(
        run_id=RUN_ID, capture="full", every=50, sink=tensorquill.JsonlSink(jsonl),
        probe=lambda m: m(input_ids=fixed, use_cache=False), probe_every=args.probe_every,
    )
    training_args = TrainingArguments(
        output_dir=str(args.out / "trainer"), max_steps=args.steps, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, max_grad_norm=args.max_grad_norm, logging_steps=args.log_every, report_to="none",
        save_strategy="no", disable_tqdm=True, fp16=precision == "fp16", bf16=precision == "bf16", use_cpu=not cuda,
        torch_compile=not args.eager, torch_compile_backend=None if args.eager else args.backend,
        dataloader_drop_last=True, logging_nan_inf_filter=False,
    )

    class InOrder(Trainer):
        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    trainer = InOrder(model=model, args=training_args, train_dataset=dataset,
                      callbacks=[SeedOutlier(), own, callback])
    try:
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            trainer.train()
    finally:
        callback.close()
        handle.remove()
    compiled, recompiles, breaks = 0, [], {}
    if not args.eager:
        compiled, recompiles = recompiled_frames(torch._dynamo.utils.get_compilation_metrics())
        breaks = dict(torch._dynamo.utils.counters["graph_break"])
    history = [e for e in trainer.state.log_history if "loss" in e and "grad_norm" in e]
    _write_json(args.out / f"done.rank{rank}.json", {
        "rank": rank, "world_size": trainer.args.world_size, "global_step": trainer.state.global_step,
        "global_steps": own.global_steps, "skipped": own.skipped, "history": history, "seeded_block": seeded_block,
        "compiled": compiled, "recompiles": recompiles, "graph_breaks": breaks,
        "distributed_type": str(trainer.accelerator.distributed_type),
    })
    if rank != 0:
        _wait_for([args.out / "result.json"], WAIT_SECONDS)
        return 0

    name = torch.cuda.get_device_name() if cuda else "CPU"
    mode = "eager" if args.eager else args.backend
    print(f"\n{torch.__version__} on {name} x{world} | {trainer.accelerator.distributed_type} | "
          f"{'toy Llama' if args.toy else args.model} | {mode} | {precision} | clip {args.max_grad_norm:g} | "
          f"seeding {seeded_block} from step {args.seed_step} "
          f"({args.fault}: {'all channels' if args.fault == 'block' else channels})", flush=True)
    failures: list[str] = []

    def check(ok: bool, text: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {text}")
        if not ok:
            failures.append(text)

    from tensorquill.sink import read_end_marker

    missing = _wait_for([args.out / f"done.rank{r}.json" for r in range(world)], WAIT_SECONDS)
    check(not missing, f"every rank finished ({[p.name for p in missing]} missing)" if missing else
          f"all {world} ranks finished")
    done = {r: json.loads((args.out / f"done.rank{r}.json").read_text(encoding="utf-8"))
            for r in range(world) if (args.out / f"done.rank{r}.json").exists()}
    captures = sorted(p.name for p in args.out.glob(f"{RUN_ID}.rank*.jsonl"))
    check(captures == sorted(f"{RUN_ID}.rank{r}.jsonl" for r in range(world)),
          f"one capture per rank: {captures}")
    check(all(d["world_size"] == world for d in done.values()), f"Trainer's world_size is {world} on every rank")

    records = {r: tensorquill.load_records(args.out / f"{RUN_ID}.rank{r}.jsonl")
               if (args.out / f"{RUN_ID}.rank{r}.jsonl").exists() else [] for r in range(world)}
    losses = {r: [(x.get("scalars") or {}).get("train/loss") for x in records[r]] for r in range(world)}
    ranks_summary = {}
    for r in range(world):
        print(f"\n--- rank {r} ---")
        d = done.get(r, {"global_steps": [], "global_step": None, "skipped": [], "history": []})
        path = args.out / f"{RUN_ID}.rank{r}.jsonl"
        end = read_end_marker(path) if path.exists() else {"state": "missing"}
        check(end.get("state") == "closed" and end.get("outcome") == "closed",
              f"rank {r}: capture closed cleanly ({end.get('state')}, {end.get('outcome')})")
        check(bool(records[r]) and all(x.get("rank") == r for x in records[r]), f"rank {r}: every record names rank {r}")
        steps = [x["step"] for x in records[r]]
        expected = [g - 1 for g in d["global_steps"]]
        check(steps == expected == list(range(args.steps)) and d["global_step"] == args.steps,
              f"rank {r}: {len(steps)} recorded steps equal the Trainer's global_step - 1 at each of its "
              f"{len(expected)} on_step_end calls, 0 to {args.steps - 1}; ended at global_step {d['global_step']}")
        norms = [x.get("clip_grad_norm") for x in records[r]]
        check(all(recorded(v) for v in losses[r]) and all(recorded(v) for v in norms),
              f"rank {r}: train/loss and clip_grad_norm recorded on every step")
        agreement = trainer_agreement(losses[r], norms, d["history"], args.log_every)
        compared = agreement["grad_norm"]
        check(agrees(compared), f"rank {r}: clip_grad_norm agrees with Trainer's logged grad_norm on {compared['pairs']} "
                                f"of {len(d['history'])} logged steps (largest difference {100 * compared['worst']:.3f}%)")
        for problem in compared["problems"][:5]:
            print(f"  {problem}")
        skips = None
        if precision == "fp16":
            problems = skip_agreement(records[r], d["skipped"])
            skips = sum(d["skipped"])
            check(not problems, f"rank {r}: step_skipped equals Accelerate's step_was_skipped on every step "
                                f"({skips} real scaler skips; grad_scale on every record)")
            for problem in problems[:5]:
                print(f"  {problem}")
        evidence = tensorquill.build_evidence(records[r])
        spike = next((i for i in evidence["incidents"] if i["kind"] == "activation_spike"), None)
        first = spike["modules"][0]["module"] if spike else None
        hit = spike is not None and spike["step"] == args.seed_step and first == d.get("seeded_block")
        check(hit, f"rank {r}: activation spike at step {args.seed_step}, first in {d.get('seeded_block')}"
                   + ("" if hit or spike is None else f" (found step {spike['step']}, first in {first})")
                   + (" (none found)" if spike is None else ""))
        probes = [x["probe"] for x in records[r] if "probe" in x]
        check(bool(probes) and all("error" not in p for p in probes), f"rank {r}: {len(probes)} probe passes, none failed")
        if not args.eager:
            wrapper, other = split_breaks(d["graph_breaks"])
            check(not other, f"rank {r}: {sum(other.values())} graph breaks outside DDP's own wrapper"
                             f" ({sum(wrapper.values())} in DistributedDataParallel.forward, which Trainer compiles"
                             " around the model: reported, not failed)")
            for reason, count in other.items():
                print(f"  {count}x {reason.splitlines()[0]}: {reason.splitlines()[1].strip() if len(reason.splitlines()) > 1 else ''}")
            check(d["compiled"] > 0 and not d["recompiles"],
                  f"rank {r}: {d['compiled']} frames compiled, none more than once"
                  + ("" if not d["recompiles"] else ": " + "; ".join(d["recompiles"])))
        ranks_summary[r] = {"records": len(records[r]), "skips": skips,
                            "spike": None if spike is None else {"step": spike["step"], "first": first}}

    print("\n--- Across ranks ---")
    apart = sum(1 for step in zip(*losses.values()) if len({_number(v) for v in step}) == world)
    check(apart > 0, f"the ranks' losses differ on {apart} of {min(map(len, losses.values()))} steps "
                     "(each rank trained on its own shard)")
    history = done.get(0, {}).get("history", [])
    mean = mean_over_ranks([losses[r] for r in range(world)])
    agreement = trainer_agreement(mean, [x.get("clip_grad_norm") for x in records[0]], history, args.log_every)
    compared = agreement["loss"]
    check(agrees(compared), f"interval mean of the ranks' mean train/loss agrees with Trainer's logged loss on "
                            f"{compared['pairs']} of {len(history)} logged steps (largest difference "
                            f"{100 * compared['worst']:.3f}%)")
    for problem in compared["problems"][:5]:
        print(f"  {problem}")
    if precision == "fp16":
        flags = {tuple(done[r]["skipped"]) for r in done}
        check(len(flags) == 1, "Accelerate skipped the same steps on every rank")

    document = {"check": "trainer_ddp", "world_size": world, "precision": precision, "toy": args.toy,
                "compile": None if args.eager else args.backend, "device": "cuda" if cuda else "cpu",
                "distributed_type": done.get(0, {}).get("distributed_type"), "seed_step": args.seed_step,
                "passed": not failures, "failures": failures, "captures": captures, "ranks": ranks_summary}
    _write_json(args.out / "result.json", document)
    print(f"\nResult: {args.out / 'result.json'}")
    print("All checks passed" if not failures else f"\n{len(failures)} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
