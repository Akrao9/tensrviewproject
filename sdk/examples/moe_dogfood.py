"""Benchmark TensorLens during Granite MoE training on WikiText-2."""

import argparse
from contextlib import ExitStack
import gc
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


def build_pair():
    from transformers import AutoModelForCausalLM
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeTopKRouter

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
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True) for model in models]
    return models, optimizers, routers


def verify_capture(
    capture_path: Path,
    routers: dict[str, tensorlens.Router],
    expected_steps: int,
    expected_assignments: int,
    *,
    expect_grads: bool,
    expect_layers: bool,
) -> None:
    records = tensorlens.load_records(capture_path)
    if [record["step"] for record in records] != list(range(expected_steps)):
        raise RuntimeError(f"{capture_path}: capture did not record the expected optimizer steps")
    for record in records:
        if expect_grads and (record["grad_norm"] is None or not record["grad_norms"]):
            raise RuntimeError(f"{capture_path} at step {record['step']}: expected gradient norms")
        if not expect_grads and (record["grad_norm"] is not None or record["grad_norms"]):
            raise RuntimeError(f"{capture_path} at step {record['step']}: unexpected gradient norms")
        if expect_layers and not record["layers"]:
            raise RuntimeError(f"{capture_path} at step {record['step']}: expected layer stats")
        if not expect_layers and record["layers"]:
            raise RuntimeError(f"{capture_path} at step {record['step']}: unexpected layer stats")
        for name in routers:
            counts = record["moe"][name]
            total = sum(counts)
            if total != expected_assignments:
                raise RuntimeError(
                    f"{name} at step {record['step']}: expected {expected_assignments} "
                    f"assignments, got {total} ({counts})"
                )


def run_comparison(
    args, batches, labels, init_kwargs_pair
) -> tuple[float, float, dict[str, tensorlens.Router], int, int]:
    torch._dynamo.reset()
    models, optimizers, routers = build_pair()
    expected_assignments = args.batch_size * args.seq_len * models[1].config.num_experts_per_tok
    times = [[], []]
    with ExitStack() as stack:
        stack.enter_context(torch._dynamo.config.patch(skip_nnmodule_hook_guards=False))
        for model, kwargs in zip(models, init_kwargs_pair):
            if kwargs is not None:
                stack.enter_context(tensorlens.init(model, routers=routers, every=FLUSH_EVERY, **kwargs))

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
                raise RuntimeError(f"Non-finite warmup loss in {labels[index]}")

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
                    raise RuntimeError(f"Non-finite loss in round {round_index + 1}, {labels[index]}")
                print(f"Round {round_index + 1} {labels[index]}: {elapsed:.3f} ms/step", flush=True)

    expected_steps = args.warmup + args.steps * args.rounds
    del runners, optimizers, models
    gc.collect()
    torch.cuda.empty_cache()
    t0, t1 = map(statistics.median, times)
    return t0, t1, routers, expected_steps, expected_assignments


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

    args.out.mkdir(parents=True)
    capture_path = args.out / "granite.jsonl"
    no_grads_path = args.out / "granite-no-grads.jsonl"
    router_only_path = args.out / "granite-router-only.jsonl"
    mode = "eager" if args.eager else f"Inductor, CUDA graphs {'on' if args.cuda_graphs else 'off'}"
    print(f"{torch.__version__} on {torch.cuda.get_device_name()} | {MODEL} | {mode}", flush=True)
    batches = load_batches(args, args.warmup + args.steps)
    counters = torch._dynamo.utils.counters

    print("\n--- Stage 1: full capture (grads off) vs. full capture (grads on) ---", flush=True)
    no_grads_ms, full_ms, routers, expected_steps, expected_assignments = run_comparison(
        args,
        batches,
        ("full (grads off)", "full (grads on)"),
        (
            {"run_id": "granite-no-grads", "grads": False, "sink": tensorlens.JsonlSink(no_grads_path)},
            {"run_id": "granite", "grads": True, "sink": tensorlens.JsonlSink(capture_path)},
        ),
    )
    verify_capture(
        no_grads_path, routers, expected_steps, expected_assignments, expect_grads=False, expect_layers=True
    )
    verify_capture(
        capture_path, routers, expected_steps, expected_assignments, expect_grads=True, expect_layers=True
    )
    grad_delta_ms = full_ms - no_grads_ms
    grad_delta_pct = 100 * (full_ms / no_grads_ms - 1)
    print(
        f"Median: grads off {no_grads_ms:.3f} ms/step, grads on {full_ms:.3f} ms/step, "
        f"grad norm cost {grad_delta_ms:+.3f} ms/step ({grad_delta_pct:+.2f}%)",
        flush=True,
    )

    print("\n--- Stage 2: baseline (capture off) vs. router-only ---", flush=True)
    off_ms, router_ms, routers, expected_steps, expected_assignments = run_comparison(
        args,
        batches,
        ("capture off", "router-only"),
        (
            None,
            {
                "run_id": "granite-router-only",
                "layers": (),
                "grads": False,
                "sink": tensorlens.JsonlSink(router_only_path),
            },
        ),
    )
    verify_capture(
        router_only_path, routers, expected_steps, expected_assignments, expect_grads=False, expect_layers=False
    )
    router_delta_ms = router_ms - off_ms
    router_overhead = 100 * (router_ms / off_ms - 1)
    print(
        f"Median: off {off_ms:.3f} ms/step, router-only {router_ms:.3f} ms/step, "
        f"overhead {router_delta_ms:+.3f} ms/step ({router_overhead:+.2f}%)",
        flush=True,
    )
    print(f"Verified {expected_steps} recorded steps across {len(routers)} routers: {capture_path}")
    if not args.eager:
        print(f"1% overhead target (router-only): {'PASS' if router_overhead < 1 else 'FAIL'}")
    if args.cuda_graphs:
        skips = counters["inductor"]["cudagraph_skips"]
        if skips:
            print(f"CUDA graph skips: {skips}. See TORCH_LOGS=cudagraphs for the reason.")


if __name__ == "__main__":
    main()
