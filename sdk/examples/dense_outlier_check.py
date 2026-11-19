"""Check TensorQuill on a dense transformer with a seeded activation outlier.

Trains a small dense Hugging Face model on WikiText-2 twice in lockstep, with capture
off and with full capture plus the probe pass, and from `--seed-step` scales part of one
middle block's output by `--factor` in both (`--fault`): by default the few channels with
the largest outputs, the massive activations that precede many loss spikes growing
further; or the whole block output. Scaling a few ordinary channels (`--fault channels`)
is not seen by a block summary on a model that already has massive activations. It reports:

- full-capture overhead on a dense model, and what a probe step and a flush step add;
- whether the activation detector finds the outlier at the seeded step and block;
- the probe's histogram and attention statistics for that block before and after, and a
  PNG of both histograms on one shared axis (if matplotlib is installed);
- graph breaks under torch.compile, and a Rerun recording of the run.

    python examples/dense_outlier_check.py --out runs/dense-outlier

`--toy` uses a tiny random Llama on random tokens, for checking the script on CPU.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch

import tensorquill

MODEL = "HuggingFaceTB/SmolLM2-360M"
FLUSH_EVERY = 50


def load_batches(model_name: str, batch_size: int, seq_len: int, count: int, device) -> torch.Tensor:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"
    )
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    need = batch_size * seq_len * count
    tokens: list[int] = []
    for start in range(0, len(rows), 256):
        tokens.extend(tokenizer("\n".join(rows[start : start + 256]), add_special_tokens=False)["input_ids"])
        if len(tokens) >= need:
            break
    if len(tokens) < need:
        raise RuntimeError("not enough WikiText-2 tokens for the requested batches")
    return torch.tensor(tokens[:need], device=device).view(count, batch_size, seq_len)


def build_models(args, device):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    if args.toy:
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
        )
        models = [LlamaForCausalLM(config) for _ in range(2)]
        models[1].load_state_dict(models[0].state_dict())
    else:
        models = [
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
            for _ in range(2)
        ]
    models = [m.to(device).train() for m in models]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=args.lr, fused=device.type == "cuda") for m in models]
    return models, optimizers


FAULTS = ("massive", "block", "channels")


def outlier_channels(model, block_name: str, fault: str, channels: list[int], hidden: int, batch) -> list[int]:
    """The hidden channels a seeded outlier scales.

    - `massive`: as many channels as `channels` names, but the ones with the largest output on
      `batch`: the massive activations trained LLMs carry, growing further.
    - `block`: every channel, a change of the whole block output.
    - `channels`: the ones named. Block summaries cannot see this on a model that already has
      massive activations: SmolLM2's block 16 output has `abs_max` near 20,000 and `std` near
      32, and two ordinary channels times 100 moved `std` 1.1 times and `abs_max` not at all
      (the first A100 run, although the loss went from 2.3 to 6.9).
    """
    if fault == "channels":
        return list(channels)
    if fault == "block":
        return list(range(hidden))
    peaks: list[torch.Tensor] = []

    def record(_module, _args, output):
        states = output[0] if isinstance(output, tuple) else output
        peaks.append(states.detach().abs().amax(dim=tuple(range(states.dim() - 1))).float())

    handle = model.get_submodule(block_name).register_forward_hook(record)
    try:
        with torch.no_grad():
            model(input_ids=batch, use_cache=False)
    finally:
        handle.remove()
    return sorted(peaks[0].topk(len(channels)).indices.tolist())


def install_outlier(block: torch.nn.Module, hidden: int, channels: list[int], factor: float, device, dtype):
    """Scale `channels` of the block's output by `factor` once `enabled` is set. A flag on
    the device rather than a Python bool, so switching it on does not recompile."""
    enabled = torch.zeros((), dtype=torch.bool, device=device)
    scale = torch.ones(hidden, device=device, dtype=dtype)  # no cast inside the compiled hook
    scale[channels] = factor

    def hook(_module, _args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        seeded = torch.where(enabled, hidden_states * scale, hidden_states)
        return (seeded, *output[1:]) if isinstance(output, tuple) else seeded

    # Registered before tensorquill.init, so it runs before TensorQuill's hook and the
    # capture sees the seeded output. `prepend=True` after init would do the same, but on
    # PyTorch 2.10 with Hugging Face models it crashes Dynamo's guard building.
    return enabled, block.register_forward_hook(hook)


def train_step(model, optimizer, batch, session=None):
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
    loss.backward()
    if session is not None:
        # Before optimizer.step(), where the step ends. A copy on the device: no sync.
        session.log({"train/loss": loss})
    optimizer.step()
    return loss.detach()


class StepTimer:
    """Per-step times without a sync per step: CUDA events, read once per chunk."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.pending: list[tuple[int, object, object]] = []
        self.times: dict[int, float] = {}

    def start(self):
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, step: int, begin) -> None:
        if self.cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.pending.append((step, begin, end))
        else:
            self.times[step] = (time.perf_counter() - begin) * 1000

    def collect(self) -> None:
        if self.pending:
            torch.cuda.synchronize()
            for step, begin, end in self.pending:
                self.times[step] = begin.elapsed_time(end)
            self.pending = []


def _median(values):
    return statistics.median(values) if values else float("nan")


def histogram_figure(before: dict, after: dict, labels: tuple[str, str], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    for hist, label, color in ((before, labels[0], "#4a7bd0"), (after, labels[1], "#d0574a")):
        counts = hist["counts"]
        width = (hist["hi"] - hist["lo"]) / len(counts)
        edges = [hist["lo"] + width * i for i in range(len(counts))]
        ax.bar(edges, counts, width=width, align="edge", alpha=0.55, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("activation value (one shared axis; each histogram keeps its own 64 bins)")
    ax.set_ylabel("count (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def finite(values) -> list[float]:
    """The finite numbers among `values`: a non-finite one is the string the sink writes for it."""
    return [float(v) for v in values or ()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]


def split_probes(records, seed_step: int):
    """The probe passes in `records`: the failed ones as (step, error), kept for the summary, and
    the others before and after `seed_step` as (step, probe)."""
    failed, before, after = [], [], []
    for r in records:
        if "probe" not in r:
            continue
        step, probe = r["step"], r["probe"]
        if "error" in probe:
            failed.append((step, probe["error"]))
        else:
            (before if step < seed_step else after).append((step, probe))
    return failed, before, after


def attention_text(a) -> str:
    """One block's attention summary, over the finite values only, with `n/a` where none is."""
    logits, share = finite(a.get("max_logit")), finite(a.get("entropy_share"))
    largest = f"{max(logits):.2f}" if logits else "n/a"
    mean = f"{sum(share) / len(share):.2f}" if share else "n/a"
    return f"attention largest logit {largest}, entropy share {mean}, {a.get('invalid_rows', '?')} invalid rows"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=50, help="steps before timing starts")
    parser.add_argument("--steps", type=int, default=350, help="timed steps after warmup")
    parser.add_argument("--seed-step", type=int, default=250, help="step the outlier starts")
    parser.add_argument("--layer", type=int, help="block to seed (default: the middle one)")
    parser.add_argument("--fault", choices=FAULTS, default="massive",
                        help="what to scale: the largest channels, the whole block, or the --channels named")
    parser.add_argument("--channels", type=int, nargs="+", default=[7, 300],
                        help="for --fault channels; for massive, only how many channels")
    parser.add_argument("--factor", type=float, default=100.0)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--eager", action="store_true", help="do not compile")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend (aot_eager to skip Inductor)")
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, CPU allowed")
    args = parser.parse_args()
    total = args.warmup + args.steps
    if any(n <= 0 or n % FLUSH_EVERY for n in (args.warmup, args.steps)):
        parser.error("warmup and steps must be positive multiples of 50")
    if not args.warmup < args.seed_step < total:
        parser.error("--seed-step must fall inside the timed steps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.toy and device.type != "cuda":
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")

    args.out.mkdir(parents=True)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    models, optimizers = build_models(args, device)
    if args.toy:
        args.channels = [c % models[0].config.hidden_size for c in args.channels]
        batches = torch.randint(0, 128, (total, 2, 32), device=device)
    else:
        batches = load_batches(args.model, args.batch_size, args.seq_len, total, device)
    fixed = batches[0]
    jsonl = args.out / "dense.jsonl"
    labels = ("capture off", "full capture + probe")
    name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    mode = "eager" if args.eager else f"{args.backend}, CUDA graphs off"
    print(f"{torch.__version__} on {name} | {'toy Llama' if args.toy else args.model} | {mode}", flush=True)

    timers = [StepTimer(device), StepTimer(device)]
    with ExitStack() as stack:
        stack.enter_context(torch._dynamo.config.patch(skip_nnmodule_hook_guards=False))
        # The same blocks full capture watches by default.
        blocks = [name for name, _ in tensorquill.capture._select_layers(models[1], "blocks")]
        layer = args.layer if args.layer is not None else len(blocks) // 2
        seeded_block = blocks[layer]
        hidden = models[0].config.hidden_size
        # Both models start from the same weights, so one calibration serves both.
        channels = outlier_channels(models[0], seeded_block, args.fault, args.channels, hidden, fixed)
        flags = []
        for model in models:
            flag, handle = install_outlier(
                model.get_submodule(seeded_block), hidden, channels, args.factor, device,
                next(model.parameters()).dtype,
            )
            flags.append(flag)
            stack.callback(handle.remove)
        session = stack.enter_context(tensorquill.init(
            models[1], run_id="dense-outlier", capture="full", every=FLUSH_EVERY,
            sink=tensorquill.JsonlSink(jsonl),
            probe=lambda m: m(input_ids=fixed, use_cache=False), probe_every=args.probe_every,
        ))
        assert session.layer_names == blocks
        scaled = f"all {hidden} channels" if args.fault == "block" else f"channels {channels}"
        print(f"Seeding block {seeded_block} ({layer + 1} of {len(blocks)}), {scaled} ({args.fault}) "
              f"× {args.factor:g}, from step {args.seed_step}", flush=True)
        runners = models if args.eager else [
            torch.compile(
                m, backend=args.backend, fullgraph=True,
                options={"triton.cudagraphs": False} if args.backend == "inductor" else None,
            )
            for m in models
        ]

        losses = [[], []]
        for chunk in range(0, total, FLUSH_EVERY):
            order = (0, 1) if (chunk // FLUSH_EVERY) % 2 == 0 else (1, 0)
            for index in order:
                for step in range(chunk, chunk + FLUSH_EVERY):
                    if step == args.seed_step:
                        flags[index].fill_(True)
                    begin = timers[index].start()
                    loss = train_step(runners[index], optimizers[index], batches[step], session if index == 1 else None)
                    timers[index].stop(step, begin)
                    if step % 10 == 9:
                        losses[index].append((step, loss))
                timers[index].collect()
        for index in range(2):
            for step, loss in losses[index]:
                if not math.isfinite(float(loss)):
                    print(f"Note: non-finite loss in {labels[index]} at step {step}")
                    break

    # ---- timing --------------------------------------------------------------------
    timed = range(args.warmup, total)
    probe_steps = [s for s in timed if (s + 1) % args.probe_every == 0]
    flush_steps = [s for s in timed if (s + 1) % FLUSH_EVERY == 0 and s not in probe_steps]
    plain = [s for s in timed if (s + 1) % FLUSH_EVERY != 0]
    off = _median([timers[0].times[s] for s in plain])
    on = _median([timers[1].times[s] for s in plain])
    on_flush = _median([timers[1].times[s] for s in flush_steps])
    on_probe = _median([timers[1].times[s] for s in probe_steps])
    off_all = _median([timers[0].times[s] for s in timed])
    print("\n--- Timing (median ms per step, timed steps only) ---")
    print(f"Capture off: {off:.3f}  |  full capture: {on:.3f}  |  added {on - off:+.3f} ms ({100 * (on / off - 1):+.2f}%)")
    print(f"Flush step (every {FLUSH_EVERY}): {on_flush:.3f} ms, {on_flush - on:+.3f} ms over a plain step")
    print(f"Probe step (every {args.probe_every}, with a flush): {on_probe:.3f} ms, "
          f"{on_probe - on_flush:+.3f} ms over a flush step")
    probe_cost = on_probe - on_flush
    for every in sorted({args.probe_every, 1000}):
        print(f"  Probe every {every} steps: {probe_cost / every:.3f} ms per step on average "
              f"({100 * probe_cost / every / off_all:.2f}% of a step)")

    # ---- detection -----------------------------------------------------------------
    records = tensorquill.load_records(jsonl)
    evidence = tensorquill.build_evidence(records)
    spike = next((i for i in evidence["incidents"] if i["kind"] == "activation_spike"), None)
    print("\n--- Detection ---")
    print(tensorquill.incident_summary(evidence))
    # The fault shows in every later block too; the incident must lead with the first, the seeded one.
    hit = spike is not None and spike["step"] == args.seed_step and spike["modules"][0]["module"] == seeded_block
    print(f"Activation spike at step {args.seed_step}, first in {seeded_block}: {'PASS' if hit else 'FAIL'}")
    if spike is not None and not hit:
        print(f"  First spike: step {spike['step']}, modules {[m['module'] for m in spike['modules'][:5]]}")

    # ---- probe before and after ----------------------------------------------------
    failed, before, after = split_probes(records, args.seed_step)
    print(f"\n--- Probe ({len(before) + len(after) + len(failed)} passes, {len(failed)} failed) ---")
    for s, error in failed:
        print(f"Probe at step {s} failed: {error}")
    # The seeded block's own attention runs before its output is scaled; the next block's
    # attention is the first to read the outlier.
    reader = blocks[layer + 1] if layer + 1 < len(blocks) else seeded_block
    if before and after:
        (s0, p0), (s1, p1) = before[-1], after[0]
        print(f"Histogram of {seeded_block}; attention of {reader}, the first block to read it")
        for tag, s, p in (("before", s0, p0), ("after", s1, p1)):
            h = p["histograms"].get(seeded_block)
            a = p["attention"].get(reader)
            line = f"Step {s} ({tag}): "
            if h:
                line += f"histogram [{h['lo']:.2f}, {h['hi']:.2f}], largest |x| {max(abs(h['lo']), abs(h['hi'])):.2f}"
            if a:
                line += f"; {attention_text(a)}"
            print(line)
        h0, h1 = p0["histograms"].get(seeded_block), p1["histograms"].get(seeded_block)
        if h0 and h1:
            span0 = max(abs(h0["lo"]), abs(h0["hi"]))
            span1 = max(abs(h1["lo"]), abs(h1["hi"]))
            print(f"Largest |x| in {seeded_block} grew {span1 / span0:.1f}× between the probes")
            png = args.out / "histogram-before-after.png"
            if histogram_figure(h0, h1, (f"step {s0} (before)", f"step {s1} (after)"), png):
                print(f"Histograms on one shared axis: {png}")
            else:
                print("matplotlib is not installed; skipped the histogram figure")
    else:
        print("Need a probe before and after the seed step; adjust --probe-every or --seed-step.")

    # ---- compile and export --------------------------------------------------------
    breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    if not args.eager:
        print(f"\nGraph breaks: {breaks}")
    try:
        rrd = tensorquill.export_rerun(evidence, args.out / "dense.rrd", records=records)
        print(f"Rerun recording: {rrd}")
    except ImportError:
        print("rerun-sdk is not installed; skipped the Rerun recording")
    print(f"Capture: {jsonl}")


if __name__ == "__main__":
    main()
