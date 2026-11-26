"""Acceptance check for the Hugging Face `Trainer` callback on a dense transformer.

Fine-tunes a small dense model on WikiText-2 through `transformers.Trainer` with
`torch_compile=True`, gradient clipping and (on CUDA) fp16 mixed precision, which puts a
`GradScaler` in the loop. From `--seed-step` it scales part of one middle block's output by
`--factor`, as `dense_outlier_check.py` does: the channels with the largest outputs by default,
or the whole block (`--fault`; see there for why not a few ordinary channels). It fails unless:

- TensorQuill recorded every optimizer step, numbered as `Trainer` numbers them;
- the loss and the gradient norm before clipping are on every step, and agree with the
  interval averages and norms `Trainer` itself logs;
- the activation detector finds the outlier at the seeded step and block;
- there are no graph breaks and no recompiles after the first step.

    python examples/trainer_check.py --out runs/trainer-check

`--toy` uses a tiny random Llama on random tokens, for checking the script on CPU; add
`--backend aot_eager` where Inductor is not set up.

`--smoke` is the same run without the plant, never scored (the multi-GPU test plan's T3). Run with
two GPUs visible and no torchrun, the `Trainer` sets `n_gpu` to the GPU count and wraps the model in
`nn.DataParallel` (Transformers 5.23.0, `training_args.py` and `Trainer._wrap_model`); the smoke
writes `smoke.json` under `--out` with `torch.cuda.device_count()`, the `Trainer`'s `n_gpu`, whether
the model it trained was wrapped in `nn.DataParallel`, and each check's PASS or FAIL (the spike check
"not checked": nothing is planted). It exits 0 when training finished, whatever the checks say, and 3
when training raised (the error is in `smoke.json`).

    python examples/trainer_check.py --out runs/smoke-trainer-two-visible --precision bf16 --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

import tensorquill

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_outlier_check import FAULTS, MODEL, install_outlier, load_batches, outlier_channels  # noqa: E402


def reset_compile_state() -> None:
    """Forget what Dynamo compiled earlier in this process; `reset()` leaves the metrics list alone."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    clear = getattr(torch._dynamo.utils, "clear_compilation_metrics", None)
    if clear is not None:
        clear()


def recompiled_frames(metrics) -> tuple[int, list[str]]:
    """The number of frames Dynamo compiled, and a line for each one it compiled more than once.

    Only forward compiles count. Under Inductor each backward graph is a metric of its own with
    no code object (`co_filename` is None, `is_forward` False): on the first A100 run those
    made one key look compiled many times and `Path(None)` raised.
    """
    frames: dict[tuple, int] = {}
    for m in metrics:
        if getattr(m, "is_forward", True) is False or m.co_filename is None:
            continue
        key = (m.co_filename, m.co_name, m.co_firstlineno)
        frames[key] = frames.get(key, 0) + 1
    lines = [f"{k[1]} ({Path(k[0]).name}:{k[2]}) compiled {n} times" for k, n in frames.items() if n > 1]
    return sum(frames.values()), lines


NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}  # as the sink writes them


def recorded(value) -> bool:
    """A value was recorded: a number, or the string the sink writes for a non-finite one."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)) or value in NONFINITE


def _number(value) -> float | None:
    if isinstance(value, str):
        return NONFINITE.get(value)
    return float(value) if recorded(value) else None


def _same(ours: float, theirs: float) -> tuple[bool, float]:
    """Whether two values agree, and their relative difference when both are finite. A
    non-finite value agrees only with the same one: NaN with NaN, +inf with +inf."""
    if math.isfinite(ours) and math.isfinite(theirs):
        off = abs(ours - theirs) / max(abs(theirs), 1e-8)
        return True, off
    same = (math.isnan(ours) and math.isnan(theirs)) or ours == theirs
    return same, 0.0


def trainer_agreement(losses: list, norms: list, history: list[dict], log_every: int) -> dict[str, dict]:
    """Each of Trainer's logged values beside what TensorQuill recorded for the same steps.

    For every entry of `history` (Trainer's `log_history` with a loss and a grad_norm): the mean
    of the recorded `train/loss` over the interval it averages, and the recorded `clip_grad_norm`
    of its last step. Returns, for "loss" and "grad_norm", how many pairs were compared, the
    largest relative difference between finite pairs, and a line for each pair that does not
    agree: a value not recorded, or a non-finite value on one side only (or of another kind).
    """
    result = {name: {"pairs": 0, "worst": 0.0, "problems": []} for name in ("loss", "grad_norm")}
    for entry in history:
        end = entry["step"]
        start = max(0, end - log_every)
        pairs = {"loss": None, "grad_norm": None}
        window = [_number(v) for v in losses[start:end]]
        if window and None not in window:
            pairs["loss"] = (sum(window) / len(window), f"steps {start} to {end - 1}")
        else:
            result["loss"]["problems"].append(f"steps {start} to {end - 1}: train/loss not recorded on every step")
        norm = _number(norms[end - 1]) if 0 < end <= len(norms) else None
        if norm is not None:
            pairs["grad_norm"] = (norm, f"step {end - 1}")
        else:
            result["grad_norm"]["problems"].append(f"step {end - 1}: no clip_grad_norm recorded")
        for name, pair in pairs.items():
            if pair is None:
                continue
            ours, where = pair
            theirs = float(entry[name])
            same, off = _same(ours, theirs)
            result[name]["pairs"] += 1
            result[name]["worst"] = max(result[name]["worst"], off)
            if not same:
                result[name]["problems"].append(f"{where}: recorded {ours}, Trainer logged {theirs}")
    return result


def agrees(compared: dict, tolerance: float = 0.01) -> bool:
    """At least one pair compared, every pair agreeing, finite ones within `tolerance`."""
    return compared["pairs"] > 0 and not compared["problems"] and compared["worst"] < tolerance


def data_parallel_wrap(trainer) -> tuple[bool, str]:
    """Whether the model the `Trainer` trained (`model_wrapped`) is an `nn.DataParallel`, looked for
    inside a `torch.compile` wrapper too, and the wrapper's type as a short text."""
    wrapped = trainer.model_wrapped
    inner = getattr(wrapped, "_orig_mod", None)
    if inner is not None:
        return isinstance(inner, torch.nn.DataParallel), f"{type(wrapped).__name__}({type(inner).__name__})"
    return isinstance(wrapped, torch.nn.DataParallel), type(wrapped).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed-step", type=int, default=250, help="step the outlier starts")
    parser.add_argument("--fault", choices=FAULTS, default="massive",
                        help="what to scale: the largest channels, the whole block, or the --channels named")
    parser.add_argument("--channels", type=int, nargs="+", default=[7, 300],
                        help="for --fault channels; for massive, only how many channels")
    parser.add_argument("--factor", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--log-every", type=int, default=50, help="Trainer's logging_steps")
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], help="default: fp16 on CUDA, else fp32")
    parser.add_argument("--eager", action="store_true", help="do not compile")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend (aot_eager to skip Inductor)")
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, CPU allowed")
    parser.add_argument("--smoke", action="store_true",
                        help="no plant, never scored: write smoke.json (GPU count, n_gpu, the DataParallel wrap, each check)")
    args = parser.parse_args()

    from transformers import (
        AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments,
    )

    from tensorquill.huggingface import TensorQuillCallback

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if not args.toy and not cuda:
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")
    if not 50 <= args.seed_step < args.steps:
        parser.error("--seed-step must be at least 50 and before --steps")
    precision = args.precision or ("fp16" if cuda else "fp32")
    if precision != "fp32" and not cuda:
        parser.error("mixed precision needs CUDA")

    args.out.mkdir(parents=True)
    reset_compile_state()
    torch.manual_seed(0)
    micro = args.steps * args.accumulate
    if args.toy:
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)
        args.channels = [c % config.hidden_size for c in args.channels]
        args.batch_size, batches = 2, torch.randint(0, 128, (micro, 2, 32))
    else:
        # float32 weights: fp16 and bf16 come from autocast, as Trainer sets them up.
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
        batches = load_batches(args.model, args.batch_size, args.seq_len, micro, "cpu")
    model.config.use_cache = False
    model.to(device)
    rows = batches.reshape(-1, batches.shape[-1])
    dataset = [{"input_ids": row, "labels": row} for row in rows]  # in order: no shuffling below
    fixed = batches[0].to(device)

    blocks = [name for name, _ in tensorquill.capture._select_layers(model, "blocks")]
    seeded_block = blocks[len(blocks) // 2]
    # Before the Trainer and its callback, so the hook runs before TensorQuill's hook.
    if args.smoke:
        channels, flag, handle = None, None, None
    else:
        channels = outlier_channels(model, seeded_block, args.fault, args.channels, model.config.hidden_size, fixed)
        flag, handle = install_outlier(
            model.get_submodule(seeded_block), model.config.hidden_size, channels, args.factor, device,
            next(model.parameters()).dtype,
        )

    class SeedOutlier(TrainerCallback):
        def on_step_begin(self, _args, state, control, **kwargs):
            if state.global_step == args.seed_step:
                flag.fill_(True)

    jsonl = args.out / "trainer.jsonl"
    callback = TensorQuillCallback(
        run_id="trainer-check", capture="full", every=50, sink=tensorquill.JsonlSink(jsonl),
        probe=lambda m: m(input_ids=fixed, use_cache=False), probe_every=args.probe_every,
    )
    training_args = TrainingArguments(
        output_dir=str(args.out / "trainer"), max_steps=args.steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate, learning_rate=args.lr, max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_every, report_to="none", save_strategy="no", disable_tqdm=True,
        fp16=precision == "fp16", bf16=precision == "bf16", use_cpu=not cuda,
        torch_compile=not args.eager, torch_compile_backend=None if args.eager else args.backend,
        dataloader_drop_last=True,
        # Trainer's default replaces a NaN or infinite step loss with its running average before
        # logging; off, the logged loss is the plain interval mean that `trainer_agreement` compares.
        logging_nan_inf_filter=False,
    )

    class InOrder(Trainer):
        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    trainer = InOrder(model=model, args=training_args, train_dataset=dataset,
                      callbacks=[callback] if args.smoke else [SeedOutlier(), callback])
    name = torch.cuda.get_device_name() if cuda else "CPU"
    mode = "eager" if args.eager else args.backend
    if args.smoke:
        print(f"{torch.__version__} on {name} | {'toy Llama' if args.toy else args.model} | {mode} | {precision} | "
              f"clip {args.max_grad_norm:g} | smoke: nothing planted | {torch.cuda.device_count()} CUDA devices "
              f"visible, Trainer n_gpu {training_args.n_gpu}", flush=True)
    else:
        print(f"{torch.__version__} on {name} | {'toy Llama' if args.toy else args.model} | {mode} | {precision} | "
              f"clip {args.max_grad_norm:g} | seeding {seeded_block} from step {args.seed_step} "
              f"({args.fault}: {'all channels' if args.fault == 'block' else channels})", flush=True)

    recompiles: list[str] = []
    train_error = None
    try:
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            trainer.train()
    except Exception as error:  # noqa: BLE001  the smoke records it; the scored run raises as before
        if not args.smoke:
            raise
        train_error = f"{type(error).__name__}: {error}"
        print(f"training raised: {train_error}", flush=True)
    finally:
        callback.close()
        if handle is not None:
            handle.remove()
    compiled = 0
    if not args.eager:
        compiled, recompiles = recompiled_frames(torch._dynamo.utils.get_compilation_metrics())

    # ---- checks ----------------------------------------------------------------------
    records = tensorquill.load_records(jsonl)
    history = [e for e in trainer.state.log_history if "loss" in e and "grad_norm" in e]
    failures: list[str] = []
    results: dict[str, str] = {}  # for the smoke: each check's result by a short name

    def check(ok: bool, text: str, key: str | None = None) -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {text}")
        if key is not None:
            results[key] = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(text)

    print("\n--- Steps, loss and clipping ---")
    steps = [r["step"] for r in records]
    check(steps == list(range(args.steps)) and trainer.state.global_step == args.steps,
          f"{len(records)} records, steps 0 to {args.steps - 1}, matching Trainer's global_step", "steps")
    losses = [r.get("scalars", {}).get("train/loss") for r in records]
    norms = [r.get("clip_grad_norm") for r in records]
    check(all(recorded(v) for v in losses), "train/loss recorded on every step", "train/loss")
    check(all(recorded(v) for v in norms), "clip_grad_norm recorded on every step", "clip_grad_norm")
    agreement = trainer_agreement(losses, norms, history, args.log_every)
    for name, label in (("loss", "interval mean of train/loss"), ("grad_norm", "clip_grad_norm")):
        compared = agreement[name]
        check(agrees(compared), f"{label} agrees with Trainer's logged {name} on {compared['pairs']} of "
                                f"{len(history)} logged steps (largest difference {100 * compared['worst']:.3f}%)",
              "loss agreement" if name == "loss" else "grad_norm agreement")
        for problem in compared["problems"][:5]:
            print(f"  {problem}")
    if precision == "fp16":
        skipped = sum(1 for r in records if r.get("step_skipped"))
        check(all("grad_scale" in r for r in records), f"grad_scale recorded on every step ({skipped} steps skipped)",
              "grad_scale")

    print("\n--- Detection ---")
    evidence = tensorquill.build_evidence(records)
    print(tensorquill.incident_summary(evidence))
    spike = next((i for i in evidence["incidents"] if i["kind"] == "activation_spike"), None)
    # The fault shows in every later block too; the incident must lead with the first, the seeded one.
    hit = spike is not None and spike["step"] == args.seed_step and spike["modules"][0]["module"] == seeded_block
    if args.smoke:
        print("NOT CHECKED: activation spike (no plant: --smoke)")
        results["activation spike"] = "not checked: no plant"
    else:
        check(hit, f"activation spike at step {args.seed_step}, first in {seeded_block}")
    if spike is not None and not hit and not args.smoke:
        print(f"  First spike: step {spike['step']}, modules {[m['module'] for m in spike['modules'][:5]]}")
    probes = [r["probe"] for r in records if "probe" in r]
    check(bool(probes) and all("error" not in p for p in probes), f"{len(probes)} probe passes, none failed", "probe")

    if not args.eager:
        print("\n--- Compile ---")
        breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        check(breaks == 0, f"{breaks} graph breaks", "graph breaks")
        check(compiled > 0 and not recompiles,
              f"{compiled} frames compiled, none more than once" + ("" if not recompiles else ": " + "; ".join(recompiles)),
              "recompiles")

    print(f"\nCapture: {jsonl}")
    if args.smoke:
        import transformers

        wrapped, kind = data_parallel_wrap(trainer)
        smoke = {
            "scored": False, "plant": None, "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "n_gpu": training_args.n_gpu, "data_parallel": wrapped, "model_wrapped": kind,
            "torchrun": "WORLD_SIZE" in os.environ, "precision": precision, "compile": mode,
            "torch": torch.__version__, "transformers": transformers.__version__, "train_error": train_error,
            "checks": results, "failures": failures,
        }
        (args.out / "smoke.json").write_text(json.dumps(smoke, indent=1), encoding="utf-8")
        print(f"Smoke, not scored: {args.out / 'smoke.json'} ({torch.cuda.device_count()} CUDA devices, n_gpu "
              f"{training_args.n_gpu}, DataParallel {wrapped}: {kind}; {len(failures)} check(s) failed)")
        sys.exit(3 if train_error is not None else 0)
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nAll checks passed")


if __name__ == "__main__":
    main()
