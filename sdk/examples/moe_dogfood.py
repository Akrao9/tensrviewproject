"""Benchmark TensorQuill during Granite MoE training on WikiText-2."""

import argparse
from contextlib import ExitStack
import gc
import math
import statistics
import time
from pathlib import Path

import torch

import tensorquill

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


def train_step(model, optimizer, batch, cuda_graphs, session=None):
    optimizer.zero_grad(set_to_none=True)
    if cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
    loss.backward()
    if session is not None:
        # Before optimizer.step(), where the step ends. A copy on the device: no sync.
        session.log({"train/loss": loss})
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
        name: tensorquill.Router(module.num_experts, top_k=module.top_k, indices=lambda out: out[2])
        for name, module in models[1].named_modules()
        if isinstance(module, GraniteMoeTopKRouter)
    }
    if len(routers) != models[1].config.num_hidden_layers:
        raise RuntimeError(f"Unexpected Granite router count: {len(routers)}")
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True) for model in models]
    return models, optimizers, routers


def install_collapse(router):
    enabled = torch.zeros((), dtype=torch.bool, device=router.weight.device)
    top_k = router.top_k
    forced_ids = torch.arange(top_k, device=enabled.device)

    def inject(_module, _args, output):
        logits, weights, indices = output
        return (
            logits,
            torch.where(enabled, torch.full_like(weights, 1 / top_k), weights),
            torch.where(enabled, forced_ids, indices),
        )

    handle = router.register_forward_hook(inject, prepend=True)
    return enabled, handle


def verify_capture(
    capture_path: Path,
    routers: dict[str, tensorquill.Router],
    expected_steps: int,
    expected_assignments: int,
    collapse_step: int | None,
    collapse_layer: int | None,
    top_k: int,
    *,
    expect_grads: bool,
    expect_layers: bool,
) -> dict:
    records = tensorquill.load_records(capture_path)
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
    evidence = tensorquill.build_evidence(records)
    if collapse_step is None or collapse_layer is None:
        return evidence
    hit = tensorquill.first_incident(evidence)
    router_names = list(routers)
    collapse_router = router_names[collapse_layer - 1]
    num_experts = routers[collapse_router].n_experts
    expected_overloaded = list(range(1, top_k + 1))
    expected_dead = list(range(top_k + 1, num_experts + 1))
    expected_counts = [expected_assignments // top_k] * top_k + [0] * len(expected_dead)
    for record in records:
        if record["step"] >= collapse_step and record["moe"][collapse_router] != expected_counts:
            raise RuntimeError(f"{capture_path}: incorrect forced routing at step {record['step']}")
    if (
        hit is None
        or hit["step"] != collapse_step
        or hit["layer"] != collapse_layer
        or hit["router"] != collapse_router
        or hit["overloaded"] != expected_overloaded
        or hit["dead"] != expected_dead
    ):
        raise RuntimeError(
            f"{capture_path}: expected collapse at step {collapse_step}, layer {collapse_layer} "
            f"({collapse_router}) with overloaded={expected_overloaded} and dead={expected_dead}, "
            f"got {hit}"
        )
    return evidence


def run_comparison(
    args, batches, labels, init_kwargs_pair
) -> tuple[float, float, dict[str, tensorquill.Router], int, int, int | None, int]:
    torch._dynamo.reset()
    models, optimizers, routers = build_pair()
    top_k = models[1].config.num_experts_per_tok
    expected_assignments = args.batch_size * args.seq_len * top_k
    collapse_step = None
    times = [[], []]
    with ExitStack() as stack:
        collapse_flags = []
        if args.collapse_layer is not None:
            if not 1 <= args.collapse_layer <= len(routers):
                raise ValueError(f"--collapse-layer must be between 1 and {len(routers)}")
            collapse_router = list(routers)[args.collapse_layer - 1]
            collapse_step = args.warmup + (args.rounds - 1) * args.steps
            for model in models:
                flag, handle = install_collapse(model.get_submodule(collapse_router))
                collapse_flags.append(flag)
                stack.callback(handle.remove)
        stack.enter_context(torch._dynamo.config.patch(skip_nnmodule_hook_guards=False))
        probe_kwargs = {}
        if args.probe_every:
            fixed = batches[0]
            probe_kwargs = {
                "probe": lambda m: m(input_ids=fixed, use_cache=False),
                "probe_every": args.probe_every,
            }
        # The captured arm logs its loss, as a real job does, so that cost is in the comparison.
        sessions = [None, None]
        for slot, (model, kwargs) in enumerate(zip(models, init_kwargs_pair)):
            if kwargs is not None:
                sessions[slot] = stack.enter_context(
                    tensorquill.init(model, routers=routers, every=FLUSH_EVERY, **probe_kwargs, **kwargs)
                )

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
                loss = train_step(runners[index], optimizers[index], batch, args.cuda_graphs, sessions[index])
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite warmup loss in {labels[index]}")

        for round_index in range(args.rounds):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for index in order:
                if collapse_flags and round_index == args.rounds - 1:
                    collapse_flags[index].fill_(True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for batch in batches[args.warmup :]:
                    loss = train_step(runners[index], optimizers[index], batch, args.cuda_graphs, sessions[index])
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
    return t0, t1, routers, expected_steps, expected_assignments, collapse_step, top_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200, help="timed steps per round")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--layers",
        choices=("blocks", "all"),
        default="blocks",
        help="activation capture: one hook per decoder layer (default) or every leaf module",
    )
    parser.add_argument(
        "--probe-every", type=int, default=0,
        help="run the probe pass (histograms, attention) every N steps in the capture arms; 0 is off",
    )
    parser.add_argument("--collapse-layer", type=int, help="inject a collapse into this 1-based MoE layer in the final round")
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
    full_path = args.out / "granite-full.jsonl"
    mode = "eager" if args.eager else f"Inductor, CUDA graphs {'on' if args.cuda_graphs else 'off'}"
    print(f"{torch.__version__} on {torch.cuda.get_device_name()} | {MODEL} | {mode}", flush=True)
    batches = load_batches(args, args.warmup + args.steps)
    counters = torch._dynamo.utils.counters
    if args.collapse_layer is not None:
        print(f"Collapse validation on layer {args.collapse_layer}. Timings include the fault hook.", flush=True)

    print("\n--- Stage 1: full capture (grads off) vs. full capture (grads on) ---", flush=True)
    no_grads_ms, full_ms, routers, expected_steps, expected_assignments, collapse_step, top_k = run_comparison(
        args,
        batches,
        ("full (grads off)", "full (grads on)"),
        (
            # A weight check records that step's gradient norms, so this arm turns it off.
            {"run_id": "granite-no-grads", "capture": "full", "grads": False, "weights": False,
             "layers": args.layers, "sink": tensorquill.JsonlSink(no_grads_path)},
            {"run_id": "granite", "capture": "full", "layers": args.layers,
             "sink": tensorquill.JsonlSink(capture_path)},
        ),
    )
    verify_capture(
        no_grads_path,
        routers,
        expected_steps,
        expected_assignments,
        collapse_step,
        args.collapse_layer,
        top_k,
        expect_grads=False,
        expect_layers=True,
    )
    verify_capture(
        capture_path,
        routers,
        expected_steps,
        expected_assignments,
        collapse_step,
        args.collapse_layer,
        top_k,
        expect_grads=True,
        expect_layers=True,
    )
    grad_delta_ms = full_ms - no_grads_ms
    grad_delta_pct = 100 * (full_ms / no_grads_ms - 1)
    print(
        f"Median: grads off {no_grads_ms:.3f} ms/step, grads on {full_ms:.3f} ms/step, "
        f"grad norm cost {grad_delta_ms:+.3f} ms/step ({grad_delta_pct:+.2f}%)",
        flush=True,
    )

    print("\n--- Stage 2: baseline (capture off) vs. router-only ---", flush=True)
    off_ms, router_ms, routers, expected_steps, expected_assignments, collapse_step, top_k = run_comparison(
        args,
        batches,
        ("capture off", "router-only"),
        (
            None,
            {
                "run_id": "granite-router-only",
                "capture": "routers",
                "sink": tensorquill.JsonlSink(router_only_path),
            },
        ),
    )
    router_evidence = verify_capture(
        router_only_path,
        routers,
        expected_steps,
        expected_assignments,
        collapse_step,
        args.collapse_layer,
        top_k,
        expect_grads=False,
        expect_layers=False,
    )
    router_delta_ms = router_ms - off_ms
    router_overhead = 100 * (router_ms / off_ms - 1)
    print(
        f"Median: off {off_ms:.3f} ms/step, router-only {router_ms:.3f} ms/step, "
        f"overhead {router_delta_ms:+.3f} ms/step ({router_overhead:+.2f}%)",
        flush=True,
    )
    print(f"Verified {expected_steps} recorded steps across {len(routers)} routers: {router_only_path}")
    print(tensorquill.incident_summary(router_evidence))

    print(f"\n--- Stage 3: baseline (capture off) vs. full capture (defaults, layers={args.layers}) ---", flush=True)
    off_full_ms, full_default_ms, routers, expected_steps, expected_assignments, collapse_step, top_k = run_comparison(
        args,
        batches,
        ("capture off", "full capture"),
        (None, {"run_id": "granite-full", "capture": "full", "layers": args.layers,
                   "sink": tensorquill.JsonlSink(full_path)}),
    )
    verify_capture(
        full_path,
        routers,
        expected_steps,
        expected_assignments,
        collapse_step,
        args.collapse_layer,
        top_k,
        expect_grads=True,
        expect_layers=True,
    )
    full_delta_ms = full_default_ms - off_full_ms
    full_overhead = 100 * (full_default_ms / off_full_ms - 1)
    print(
        f"Median: off {off_full_ms:.3f} ms/step, full capture {full_default_ms:.3f} ms/step, "
        f"overhead {full_delta_ms:+.3f} ms/step ({full_overhead:+.2f}%)",
        flush=True,
    )
    if not args.eager and args.collapse_layer is None:
        print(f"1% overhead target (router-only): {'PASS' if router_overhead < 1 else 'FAIL'}")
        print(f"1% overhead target (full capture): {'PASS' if full_overhead < 1 else 'FAIL'}")
    if args.probe_every:
        probes = [r["probe"] for r in tensorquill.load_records(full_path) if "probe" in r]
        print(f"Probe passes in {full_path.name}: {len(probes)} (every {args.probe_every} steps)")
        if probes:
            last = probes[-1]["attention"]
            print(f"Last probe: {len(probes[-1]['histograms'])} histograms, attention in {len(last)} layers")
            for name, att in list(last.items())[:: max(1, len(last) // 4)]:
                share = sum(att["entropy_share"]) / len(att["entropy_share"])
                print(
                    f"  {name}: largest logit {max(att['max_logit']):.2f}, entropy {share:.2f} of maximum, "
                    f"{att['masked_rows']} masked and {att['invalid_rows']} invalid rows"
                )
    if args.cuda_graphs:
        skips = counters["inductor"]["cudagraph_skips"]
        if skips:
            print(f"CUDA graph skips: {skips}. See TORCH_LOGS=cudagraphs for the reason.")


if __name__ == "__main__":
    main()
