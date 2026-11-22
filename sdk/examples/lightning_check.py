"""Check `tensorquill.lightning.TensorQuillCallback` on a real `LightningModule` under DDP or FSDP.

The module wraps a small dense Hugging Face causal LM (SmolLM2-360M on WikiText-2, or a tiny
random Llama with `--toy`) and, from `--seed-step`, scales part of one middle block's output by
`--factor` on every rank, as `dense_outlier_check.py` plants it. Lightning launches the ranks
itself (`devices=N`, its subprocess launcher), and every rank records under one run ID through the
callback. The module keeps its own log of every `training_step` (Lightning's `global_step`,
`global_rank`, `world_size` and the loss it returned), written as `module.rank<R>.json`. Rank 0
waits for every rank's files, then fails unless:

- there is one capture per rank, ranks 0 to world_size - 1, each closed cleanly (its end marker),
  every record naming its own rank, and the run manifest's `world_size` equal to the module's;
- each rank's recorded steps are exactly the `global_step`s its module logged, 0 to `--steps` - 1,
  and the `Trainer` on every rank ended at `global_step == --steps`;
- each rank's recorded `train/loss` equals, step by step, the loss its module returned;
- on every rank the activation detector finds the outlier at the planted step, first in the
  planted block: the block TensorQuill watches in the module the callback was given that is, or
  holds, the planted layer, found by identity (under FSDP's per-layer wrap, the FSDP wrapper around
  it; the planted layer's own name is written beside it in `result.json`).

`--strategy ddp` (the default) or `--strategy fsdp` (Lightning's `FSDPStrategy`, FSDP1 with
`use_orig_params=True`; `--fsdp-wrap blocks` wraps each decoder layer, `none` the module as one
unit). Lightning 2.6.6 refuses `fsdp` without a CUDA accelerator, so on CPU that arm stops with
Lightning's own message and exit status 2, as it does when a package, the model or the data is
missing (one line says what). Writes the captures, the module logs and `result.json`
under `--out`; exit 0 when every check passes, 1 when one fails, 2 when it cannot run here.

    python examples/lightning_check.py --out runs/lightning-ddp --devices 2
    python examples/lightning_check.py --out runs/lightning-fsdp --devices 2 --strategy fsdp --fsdp-wrap blocks
    GLOO_SOCKET_IFNAME=lo0 python examples/lightning_check.py --toy --out runs/lightning-cpu --devices 2

`--toy` runs on CPU with gloo (`GLOO_SOCKET_IFNAME=lo0` on macOS, `lo` on Linux).
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
from dense_outlier_check import FAULTS, MODEL, load_batches, outlier_channels  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_CANNOT_RUN = 0, 1, 2
RUN_ID = "lightning-check"
WAIT_SECONDS = 180.0  # how long rank 0 waits for every rank's files, and the others for its verdict


def _is_launched_rank() -> bool:
    """A rank Lightning's launcher (or torchrun) started, not the process the user ran: the run's
    folder is made by the latter only."""
    return any(os.environ.get(name, "0") != "0" for name in ("RANK", "LOCAL_RANK", "NODE_RANK"))


def _write_json(path: Path, document: dict) -> None:
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_text(json.dumps(document), encoding="utf-8")
    os.replace(scratch, path)


def _wait_for(paths: list[Path], seconds: float) -> list[Path]:
    """The paths among `paths` still missing after waiting up to `seconds` for them."""
    deadline = time.time() + seconds
    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing or time.time() > deadline:
            return missing
        time.sleep(0.1)


def watched_block(root: torch.nn.Module, target: torch.nn.Module) -> str | None:
    """The name of the block TensorQuill watches in `root` (`"blocks"`) that is `target` or holds it."""
    return next((name for name, block in tensorquill.capture._select_layers(root, "blocks")
                 if block is target or any(m is target for m in block.modules())), None)


def build(args):
    """The Lightning classes, imported here so `--help` needs no Lightning."""
    import lightning.pytorch as L
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    from tensorquill.lightning import TensorQuillCallback

    class CheckModule(L.LightningModule):
        """A causal LM with a planted outlier, logging what it did on every training step."""

        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            if args.toy:
                config = LlamaConfig(
                    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
                )
                self.model = LlamaForCausalLM(config)
                args.channels = [c % config.hidden_size for c in args.channels]
            else:
                self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
            self.model.config.use_cache = False
            hidden = self.model.config.hidden_size
            blocks = [name for name, _ in tensorquill.capture._select_layers(self.model, "blocks")]
            self.seeded_inner = blocks[len(blocks) // 2]
            self.seeded_block_module = self.model.get_submodule(self.seeded_inner)
            channels = outlier_channels(self.model, self.seeded_inner, args.fault, args.channels, hidden, args.fixed)
            self.channels = channels
            # Buffers, so Lightning moves them with the module; not persistent, so no checkpoint holds them.
            scale = torch.ones(hidden)
            scale[channels] = args.factor
            self.register_buffer("seed_scale", scale, persistent=False)
            self.register_buffer("seed_flag", torch.zeros((), dtype=torch.bool), persistent=False)
            # Before the callback's init (in on_fit_start), so it runs before TensorQuill's hook.
            self.seeded_block_module.register_forward_hook(self._plant)
            self.own_log: list[dict] = []

        def _plant(self, _module, _args, output):
            states = output[0] if isinstance(output, tuple) else output
            seeded = torch.where(self.seed_flag, states * self.seed_scale.to(states.dtype), states)
            return (seeded, *output[1:]) if isinstance(output, tuple) else seeded

        def on_train_batch_start(self, batch, batch_idx):
            if self.global_step == args.seed_step:
                self.seed_flag.fill_(True)

        def training_step(self, batch, batch_idx):
            loss = self.model(input_ids=batch, labels=batch, use_cache=False).loss
            self.own_log.append({"global_step": self.global_step, "rank": self.global_rank,
                                 "world_size": self.trainer.world_size, "loss": float(loss.detach())})
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=args.lr)

        def train_dataloader(self):
            # In order; under DDP and FSDP Lightning adds a DistributedSampler that keeps it (shuffle off).
            return torch.utils.data.DataLoader(list(args.rows), batch_size=args.batch_size, shuffle=False, drop_last=True)

    class RankFileCallback(TensorQuillCallback):
        """The package's callback, writing each rank's capture into the run's folder, and keeping the
        name under which the planted block is known in the module `init` is given."""

        def on_fit_start(self, trainer, pl_module):
            self.init_kwargs["sink"] = tensorquill.JsonlSink(args.out / f"{RUN_ID}.rank{trainer.global_rank}.jsonl")
            # The block TensorQuill watches that is, or holds, the planted module: under FSDP's
            # per-layer wrap the decoder layer is FSDP's `_fsdp_wrapped_module` inside the watched wrapper.
            target = pl_module.seeded_block_module
            self.planted_name = next((n for n, m in pl_module.named_modules() if m is target), None)
            self.seeded_name = watched_block(pl_module, target)
            super().on_fit_start(trainer, pl_module)

    return L, CheckModule, RankFileCallback


def strategy_for(args):
    if args.strategy == "ddp":
        return "ddp"
    from lightning.pytorch.strategies import FSDPStrategy

    if args.fsdp_wrap == "blocks":
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        return FSDPStrategy(auto_wrap_policy=ModuleWrapPolicy({LlamaDecoderLayer}))
    return FSDPStrategy()


def judge(args, world: int, done: dict[int, dict], seeded: str | None) -> tuple[list[str], dict]:
    """Rank 0's checks over every rank's capture and module log."""
    from tensorquill import manifest as _manifest
    from tensorquill.sink import read_end_marker

    failures: list[str] = []
    ranks: dict[int, dict] = {}

    def check(ok: bool, text: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {text}")
        if not ok:
            failures.append(text)

    captures = sorted(p.name for p in args.out.glob(f"{RUN_ID}.rank*.jsonl"))
    expected = [f"{RUN_ID}.rank{r}.jsonl" for r in range(world)]
    check(captures == sorted(expected), f"one capture per rank: {captures} (world size {world})")
    manifest = _manifest.read_manifest(_manifest.local_manifest_path(args.out / expected[0], RUN_ID)) or {}
    check(manifest.get("world_size") == world, f"the run manifest's world_size {manifest.get('world_size')} is the "
                                               f"module's world_size {world}")
    for rank in range(world):
        print(f"\n--- rank {rank} ---")
        path = args.out / f"{RUN_ID}.rank{rank}.jsonl"
        own = json.loads((args.out / f"module.rank{rank}.json").read_text(encoding="utf-8")) if (
            args.out / f"module.rank{rank}.json").exists() else []
        records = tensorquill.load_records(path) if path.exists() else []
        end = read_end_marker(path) if path.exists() else {"state": "missing"}
        check(end.get("state") == "closed" and end.get("outcome") == "closed",
              f"rank {rank}: capture closed cleanly ({end.get('state')}, {end.get('outcome')})")
        check(bool(records) and all(r.get("rank") == rank for r in records)
              and all(e["rank"] == rank and e["world_size"] == world for e in own),
              f"rank {rank}: every record names rank {rank}; the module logged rank {rank} of world size {world}")
        steps = [r["step"] for r in records]
        own_steps = [e["global_step"] for e in own]
        check(steps == own_steps == list(range(args.steps)) and done[rank]["global_step"] == args.steps,
              f"rank {rank}: {len(steps)} recorded steps equal the {len(own_steps)} global_steps the module logged, "
              f"0 to {args.steps - 1}; Trainer ended at global_step {done[rank]['global_step']}")
        losses = [(r.get("scalars") or {}).get("train/loss") for r in records]
        off = [abs(a - e["loss"]) / max(abs(e["loss"]), 1e-8) for a, e in zip(losses, own)
               if isinstance(a, (int, float))]
        worst = max(off, default=float("inf"))
        check(len(off) == len(own) == args.steps and worst < args.loss_tolerance,
              f"rank {rank}: train/loss equals the module's loss on {len(off)} of {len(own)} steps "
              f"(largest relative difference {worst:.2e})")
        evidence = tensorquill.build_evidence(records)
        spike = next((i for i in evidence["incidents"] if i["kind"] == "activation_spike"), None)
        first = spike["modules"][0]["module"] if spike else None
        hit = spike is not None and spike["step"] == args.seed_step and first == seeded
        check(hit, f"rank {rank}: activation spike at step {args.seed_step}, first in {seeded}"
                   + ("" if hit or spike is None else f" (found step {spike['step']}, first in {first})")
                   + (" (none found)" if spike is None else ""))
        ranks[rank] = {"records": len(records), "steps_match": steps == own_steps, "loss_worst": worst,
                       "spike": None if spike is None else {"step": spike["step"], "first": first}}
    if world > 1:
        logs = [json.loads((args.out / f"module.rank{r}.json").read_text(encoding="utf-8"))
                if (args.out / f"module.rank{r}.json").exists() else [] for r in range(world)]
        apart = sum(1 for step in zip(*logs) if len({e["loss"] for e in step}) == world)
        check(apart > 0, f"the ranks' losses differ on {apart} of {min(map(len, logs))} steps: each rank trained on "
                         "its own shard, so each capture's agreement above is with its own rank")
    return failures, {"captures": captures, "manifest_world_size": manifest.get("world_size"), "ranks": ranks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--fsdp-wrap", choices=["none", "blocks"], default="blocks",
                        help="with --strategy fsdp: wrap each decoder layer, or the module as one unit")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed-step", type=int, default=200, help="step the outlier starts")
    parser.add_argument("--fault", choices=FAULTS, default="massive")
    parser.add_argument("--channels", type=int, nargs="+", default=[7, 300])
    parser.add_argument("--factor", type=float, default=100.0)
    parser.add_argument("--precision", default=None, help="Lightning's precision (default: bf16-mixed on CUDA, 32-true on CPU)")
    parser.add_argument("--loss-tolerance", type=float, default=1e-5, help="relative; raise it for mixed precision")
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, on CPU with gloo")
    args = parser.parse_args(argv)

    cuda = torch.cuda.is_available() and not args.toy
    if not args.toy and not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")
    if not 50 <= args.seed_step < args.steps:
        parser.error("--seed-step must be at least 50 and before --steps")
    precision = args.precision or ("bf16-mixed" if cuda else "32-true")
    if args.out.exists() and not _is_launched_rank() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=True)

    world = args.devices * args.num_nodes
    if args.toy:
        args.batch_size, seq = 2, 32
        torch.manual_seed(0)
        args.rows = torch.randint(0, 128, (args.steps * args.batch_size * world, seq))
    else:
        try:
            args.rows = load_batches(args.model, args.batch_size, args.seq_len, args.steps * world, "cpu").reshape(-1, args.seq_len)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"lightning_check.py: cannot load WikiText-2 for {args.model} ({type(error).__name__}: {error}); install "
                  "transformers, huggingface_hub and pyarrow and give the machine network access or a filled Hugging Face cache")
            return EXIT_CANNOT_RUN
    args.fixed = args.rows[: args.batch_size]

    try:
        L, CheckModule, RankFileCallback = build(args)
    except ImportError as error:
        print(f"lightning_check.py: cannot run: {error}; install lightning and transformers")
        return EXIT_CANNOT_RUN
    try:
        module = CheckModule()
    except OSError as error:
        print(f"lightning_check.py: cannot load {args.model} ({error}); give the machine network access or a filled "
              "Hugging Face cache")
        return EXIT_CANNOT_RUN
    callback = RankFileCallback(run_id=RUN_ID, capture="full", every=50)
    try:
        trainer = L.Trainer(
            default_root_dir=args.out / "lightning", max_steps=args.steps, callbacks=[callback], logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            accelerator="cuda" if cuda else "cpu", devices=args.devices, num_nodes=args.num_nodes,
            strategy=strategy_for(args), precision=precision,
        )
    except ValueError as error:
        if "requires a GPU accelerator" in str(error):
            print(f"lightning_check.py: cannot run here: Lightning {L.__version__} says: {error}")
            return EXIT_CANNOT_RUN
        raise
    trainer.fit(module)
    rank = trainer.global_rank
    world = trainer.world_size
    _write_json(args.out / f"module.rank{rank}.json", module.own_log)  # type: ignore[arg-type]
    _write_json(args.out / f"done.rank{rank}.json", {"rank": rank, "global_step": trainer.global_step,
                                                      "seeded": callback.seeded_name, "planted": callback.planted_name,
                                                      "world_size": world})
    if rank != 0:
        _wait_for([args.out / "result.json"], WAIT_SECONDS)
        return EXIT_PASS

    print(f"\n{torch.__version__} | Lightning {L.__version__} | {type(trainer.strategy).__name__} on "
          f"{'CUDA' if cuda else 'CPU'} x{world} | {precision} | {'toy Llama' if args.toy else args.model} | "
          f"seeding {callback.planted_name} (watched as {callback.seeded_name}) from step {args.seed_step} "
          f"({args.fault}: {'all channels' if args.fault == 'block' else module.channels})", flush=True)
    missing = _wait_for([args.out / f"done.rank{r}.json" for r in range(world)], WAIT_SECONDS)
    failures: list[str] = []
    if missing:
        failures.append(f"no done file from {[p.name for p in missing]}")
        print(f"FAIL: {failures[-1]}")
    done = {}
    for r in range(world):
        path = args.out / f"done.rank{r}.json"
        done[r] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"global_step": None}
    found, summary = judge(args, world, done, callback.seeded_name)
    failures += found
    if len({d.get("seeded") for d in done.values()}) != 1:
        failures.append(f"the planted block's name differs across ranks: {[d.get('seeded') for d in done.values()]}")
        print(f"FAIL: {failures[-1]}")
    document = {"check": "lightning", "strategy": type(trainer.strategy).__name__, "fsdp_wrap": args.fsdp_wrap
                if args.strategy == "fsdp" else None, "world_size": world, "precision": precision, "toy": args.toy,
                "device": "cuda" if cuda else "cpu", "seed_step": args.seed_step, "seeded_block": callback.seeded_name,
                "planted_layer": callback.planted_name,
                "passed": not failures, "failures": failures, **summary}
    _write_json(args.out / "result.json", document)
    print(f"\nResult: {args.out / 'result.json'}")
    print("All checks passed" if not failures else f"\n{len(failures)} check(s) failed")
    return EXIT_PASS if not failures else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
