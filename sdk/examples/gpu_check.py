"""Check TensorQuill on a GPU: correctness under `torch.compile`, then training overhead.

    python examples/gpu_check.py                      # CUDA: Inductor, mode="reduce-overhead"
    python examples/gpu_check.py --dim 2048 --blocks 12 --batch 8192
    TORCH_LOGS="graph_breaks,recompiles,cudagraphs" python examples/gpu_check.py

Part 1 trains a small MoE model with capture on and fails unless there are zero graph
breaks, zero recompiles and zero CUDA-graph skips, the captured router counts equal a
direct calculation on every step, and the detector finds a seeded router collapse.

Part 2 trains two identical larger models, one with capture and one without, in
alternating timed blocks, and reports the median time per step of each. The percentage
depends on the model: a small model makes fixed per-step costs look large, so set
`--dim`, `--blocks` and `--batch` close to your real job before trusting the number.

Without CUDA the script falls back to the `aot_eager` backend, which checks the script
itself but says nothing about Inductor, CUDA graphs or GPU overhead.
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorquill  # noqa: E402

N_EXPERTS = 8


class Router(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, N_EXPERTS)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h):
        logits = self.proj(h)
        return logits, logits.argmax(dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.router = Router(dim)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(N_EXPERTS))

    def forward(self, h):
        _, choice = self.router(h)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=1)
        index = choice[:, None, None].expand(-1, 1, h.shape[-1])
        return h + torch.relu(stacked.gather(1, index).squeeze(1))


class MoE(nn.Module):
    def __init__(self, dim: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(blocks))
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def routers(model: MoE) -> dict[str, int]:
    return {f"blocks.{i}.router": N_EXPERTS for i in range(len(model.blocks))}


def compile_model(model: nn.Module, args: argparse.Namespace):
    if args.backend == "inductor":
        return torch.compile(model, mode=args.mode)
    return torch.compile(model, backend=args.backend)


def train_step(compiled, opt, x) -> None:
    loss = (compiled(x) - x.sum(dim=1, keepdim=True)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_correctness(args: argparse.Namespace, device: torch.device) -> list[str]:
    """Return a list of failures; empty means the check passed."""
    dim, batch, steps, break_at = 32, 256, 16, 8
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = MoE(dim, blocks=2).to(device)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, run_id="gpu-check", every=4, sink=sink, routers=routers(model))
    compiled = compile_model(model, args)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    proj = model.blocks[0].router.proj

    expected = []
    torch._dynamo.utils.counters.clear()
    with torch._dynamo.config.patch(error_on_recompile=True):
        for step in range(steps):
            if step == break_at:
                with torch.no_grad():
                    proj.bias[0] += 50.0  # the seeded failure
            x = torch.randn(batch, dim, device=device)
            with torch.no_grad():
                # Computed without calling the router module, so its hook does not fire.
                choice = F.linear(x, proj.weight, proj.bias).argmax(dim=-1)
                expected.append(torch.bincount(choice, minlength=N_EXPERTS).tolist())
            try:
                train_step(compiled, opt, x)
            except torch._dynamo.exc.RecompileError as error:
                session.close()
                return [f"recompile at step {step}: {error}"]
    session.close()

    counters = torch._dynamo.utils.counters
    failures = []
    if counters["graph_break"]:
        failures.append(f"graph breaks: {dict(counters['graph_break'])}")
    if counters["frames"]["ok"] < 1:
        failures.append("the model was never traced by the compiler")
    skips = counters["inductor"]["cudagraph_skips"]
    if skips:
        failures.append(f"{skips} CUDA-graph skip(s); rerun with TORCH_LOGS=cudagraphs for the reason")
    records = sink.records
    if [record["step"] for record in records] != list(range(steps)):
        failures.append(f"expected steps 0..{steps - 1}, got {[r['step'] for r in records]}")
        return failures
    wrong = [r["step"] for r, want in zip(records, expected) if r["moe"]["blocks.0.router"] != want]
    if wrong:
        failures.append(f"router counts differ from a direct calculation at steps {wrong}")
    hit = tensorquill.first_incident(tensorquill.build_evidence(records))
    if hit is None or hit["step"] != break_at:
        failures.append(f"collapse seeded at step {break_at}, detector reported {hit and hit['step']}")
    return failures


def measure_overhead(args: argparse.Namespace, device: torch.device) -> None:
    def build(capture: bool):
        torch.manual_seed(0)
        model = MoE(args.dim, args.blocks).to(device)
        session = None
        if capture:
            session = tensorquill.init(
                model,
                run_id="gpu-check-overhead",
                capture="full",
                every=args.every,
                sink=tensorquill.JsonlSink(Path(tmp) / "overhead.jsonl"),  # real file writes
                routers=routers(model),
            )
        return compile_model(model, args), torch.optim.SGD(model.parameters(), lr=1e-4), session

    # Both models share forward code but only one has capture hooks.
    with (
        torch._dynamo.config.patch(skip_nnmodule_hook_guards=False),
        tempfile.TemporaryDirectory() as tmp,
    ):
        torch._dynamo.reset()
        base, base_opt, _ = build(capture=False)
        on, on_opt, session = build(capture=True)
        x = torch.randn(args.batch, args.dim, device=device)
        for compiled, opt in ((base, base_opt), (on, on_opt)):
            for _ in range(args.warmup):
                train_step(compiled, opt, x)
        sync(device)

        times: dict[str, list[float]] = {"off": [], "on": []}
        for _ in range(args.rounds):
            for name, compiled, opt in (("off", base, base_opt), ("on", on, on_opt)):
                sync(device)
                start = time.perf_counter()
                for _ in range(args.block_steps):
                    train_step(compiled, opt, x)
                sync(device)
                times[name].append((time.perf_counter() - start) / args.block_steps)
        watched = len(session.layer_names)
        session.close()

        records = tensorquill.load_records(Path(tmp) / "overhead.jsonl")
        expected_steps = args.warmup + args.rounds * args.block_steps
        if [record["step"] for record in records] != list(range(expected_steps)):
            raise RuntimeError("Overhead run did not record the expected optimizer steps")
        router_names = [f"blocks.{i}.router" for i in range(args.blocks)]
        for record in records:
            for name in router_names:
                counts = record["moe"][name]
                if sum(counts) != args.batch:
                    raise RuntimeError(
                        f"Overhead run {name} at step {record['step']}: "
                        f"expected {args.batch} assignments, got {sum(counts)} ({counts})"
                    )

    off, on_time = statistics.median(times["off"]), statistics.median(times["on"])
    print(
        f"overhead model: dim={args.dim} blocks={args.blocks} batch={args.batch}, "
        f"{watched} watched layers, {args.blocks} routers, flush every {args.every} steps"
    )
    print(f"  capture off: {off * 1e3:.3f} ms/step (median of {args.rounds} blocks of {args.block_steps})")
    print(f"  capture on:  {on_time * 1e3:.3f} ms/step")
    print(f"  overhead:    {(on_time - off) * 1e6:+.1f} us/step, {(on_time / off - 1) * 100:+.2f}%")
    spread = (max(times["off"]) - min(times["off"])) / off * 100
    print(f"  noise:       capture-off blocks vary by {spread:.2f}% of the median")
    print(f"Verified routing assignment totals across {expected_steps} recorded steps")


def main() -> None:
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="inductor" if cuda else "aot_eager")
    parser.add_argument("--mode", default="reduce-overhead", help="torch.compile mode for Inductor")
    parser.add_argument("--dim", type=int, default=1024 if cuda else 64)
    parser.add_argument("--blocks", type=int, default=8 if cuda else 2)
    parser.add_argument("--batch", type=int, default=4096 if cuda else 256)
    parser.add_argument("--every", type=int, default=50, help="steps between transfers to the CPU")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block-steps", type=int, default=100)
    parser.add_argument("--skip-overhead", action="store_true")
    args = parser.parse_args()
    if args.block_steps % args.every:
        parser.error("--block-steps must be a multiple of --every so each block has the same flushes")

    device = torch.device("cuda" if cuda else "cpu")
    name = torch.cuda.get_device_name(0) if cuda else "CPU (no CUDA: Inductor and CUDA graphs not checked)"
    mode = f', mode="{args.mode}"' if args.backend == "inductor" else ""
    print(f"torch {torch.__version__} · {name} · backend={args.backend}{mode}")

    failures = check_correctness(args, device)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        sys.exit(1)
    print(
        "PASS: no graph breaks, recompiles or CUDA-graph skips; "
        "router counts exact on all 16 steps; collapse detected at step 8"
    )
    if not args.skip_overhead:
        measure_overhead(args, device)


if __name__ == "__main__":
    main()
