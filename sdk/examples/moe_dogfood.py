"""Benchmark TensorLens during Granite MoE training on WikiText-2."""

import argparse
import math
import statistics
import time
from pathlib import Path

import torch

import tensorlens

MODEL = "ibm-granite/granite-3.0-1b-a400m-base"
FLUSH_EVERY = 50


def load_batches(args, batch_count):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    path = hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokens_per_batch = args.batch_size * args.seq_len
    count = tokens_per_batch * batch_count
    tokens = []
    for start in range(0, len(rows), 256):
        text = "\n".join(rows[start : start + 256])
        tokens.extend(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
        if len(tokens) >= count:
            break
    if len(tokens) < count:
        raise RuntimeError("Not enough tokens in the selected WikiText-2 rows")
    return torch.tensor(tokens[:count], device="cuda", dtype=torch.long).view(
        batch_count, args.batch_size, args.seq_len
    )


def train_step(model, optimizer, batch, cuda_graphs):
    optimizer.zero_grad(set_to_none=True)
    if cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
    loss.backward()
    optimizer.step()
    return loss.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200, help="timed steps per round")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    compile_mode = parser.add_mutually_exclusive_group()
    compile_mode.add_argument(
        "--eager", action="store_true", help="diagnose model or capture without compiling"
    )
    compile_mode.add_argument(
        "--cuda-graphs", action="store_true", help="enable CUDA graphs on a compatible GPU and PyTorch build"
    )
    args = parser.parse_args()
    if min(args.batch_size, args.rounds) < 1 or args.seq_len < 2 or args.seq_len > 4096:
        parser.error("batch size and rounds must be positive and sequence length must be 2–4096")
    if any(n <= 0 or n % FLUSH_EVERY for n in (args.steps, args.warmup)):
        parser.error("warmup and timed steps must be positive multiples of 50")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("a CUDA GPU with BF16 support is required")

    from transformers import AutoModelForCausalLM
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeTopKRouter

    args.out.mkdir(parents=True)
    capture_path = args.out / "granite.jsonl"
    mode = "eager" if args.eager else f"Inductor, CUDA graphs {'on' if args.cuda_graphs else 'off'}"
    print(f"{torch.__version__} on {torch.cuda.get_device_name()} | {MODEL} | {mode}", flush=True)
    batches = load_batches(args, args.warmup + args.steps)
    models = [
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, experts_implementation="grouped_mm"
        ).to("cuda").train()
        for _ in range(2)
    ]
    routers = {
        name: tensorlens.Router(module.num_experts, indices=lambda out: out[2])
        for name, module in models[1].named_modules()
        if isinstance(module, GraniteMoeTopKRouter)
    }
    if len(routers) != models[1].config.num_hidden_layers:
        raise RuntimeError(f"Unexpected Granite router count: {len(routers)}")
    expected_assignments = args.batch_size * args.seq_len * models[1].config.num_experts_per_tok
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True) for model in models]
    times = [[], []]
    counters = torch._dynamo.utils.counters
    counters.clear()
    # Both models share forward code. Guard their hooks so the capture run
    # cannot reuse the baseline graph, which has no TensorLens hooks.
    with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False), tensorlens.init(
        models[1], run_id="granite", routers=routers, every=FLUSH_EVERY,
        sink=tensorlens.JsonlSink(capture_path),
    ):
        # The grouped_mm fallback can copy expert offsets to CPU. On the RTX PRO
        # 6000 with PyTorch 2.14.1+cu130 this fails during CUDA graph recording.
        runners = models if args.eager else [
            torch.compile(
                model,
                backend="inductor",
                fullgraph=True,
                options={"triton.cudagraphs": args.cuda_graphs},
            )
            for model in models
        ]
        for index in range(2):
            for batch in batches[: args.warmup]:
                loss = train_step(runners[index], optimizers[index], batch, args.cuda_graphs)
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite warmup loss in run {index}")

        for round_index in range(args.rounds):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for index in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for batch in batches[args.warmup :]:
                    loss = train_step(runners[index], optimizers[index], batch, args.cuda_graphs)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000 / args.steps
                times[index].append(elapsed)
                if not math.isfinite(loss.item()):
                    raise RuntimeError(f"Non-finite loss in round {round_index + 1}, run {index}")
                print(f"Round {round_index + 1} capture {'on' if index else 'off'}: {elapsed:.3f} ms/step", flush=True)

    records = tensorlens.load_records(capture_path)
    expected_steps = args.warmup + args.steps * args.rounds
    if [record["step"] for record in records] != list(range(expected_steps)):
        raise RuntimeError("Capture did not record the expected optimizer steps")
    for record in records:
        for name in routers:
            counts = record["moe"][name]
            total = sum(counts)
            if total != expected_assignments:
                raise RuntimeError(
                    f"{name} at step {record['step']}: expected {expected_assignments} "
                    f"assignments, got {total} ({counts})"
                )

    off, on = map(statistics.median, times)
    overhead = 100 * (on / off - 1)
    print(f"Median: off {off:.3f} ms/step, on {on:.3f} ms/step, overhead {overhead:+.2f}%")
    print(f"Verified {expected_steps} recorded steps across {len(routers)} routers: {capture_path}")
    if not args.eager:
        print(f"1% overhead target: {'PASS' if overhead < 1 else 'FAIL'}")
    if args.cuda_graphs:
        skips = counters["inductor"]["cudagraph_skips"]
        if skips:
            print(f"CUDA graph skips: {skips}. See TORCH_LOGS=cudagraphs for the reason.")


if __name__ == "__main__":
    main()
