"""Run lm-resiliency's fault-injection campaign with a TensorQuill session on every rank.

Lane AI-3, Nov 22, 2026. A torchrun entry point of ours: it imports the campaign's own module
(`tests.validation.fault_injection.pytorch` in an lm-resiliency checkout) and runs its `main()`,
unchanged, after replacing two names in that module's namespace. Nothing in the sibling repository
is modified.

- `enable_fault_injection`: wrapped so that `tensorquill.init` runs just before it, on the DDP
  model's module, after the campaign has registered SCOUT (`enable_resiliency`). TensorQuill's
  optimizer hooks are therefore registered after SCOUT's and before the injector's. The injector
  arms each fault when it fires (forward hooks, gradient hooks, in-place tensor edits; `local.py`),
  which is later than `init` in any case, so its forward hooks run after TensorQuill's: a corrupted
  block output is seen by TensorQuill on the next block's input, not on the block itself; a
  corrupted gradient is seen in the gradient norms; a corrupted weight in the weight check and
  in the cross-rank replica observation, taken at the optimizer step, before the campaign's
  `EvaluationStateReset` restores the clean state after that step.
- `_teardown`: wrapped so that the session is closed (its last step written) before the campaign's
  own teardown.

The session: `capture="full"`, `every=1`, `weights=1`, `replica_group=WORLD`, `replica_every=1`
(every step compared across ranks), `start_step=1` so that TensorQuill's step N is the campaign's
iteration N (their loop counts from 1; TensorQuill from `start_step`). Captures go to
`<artifact-dir>/tensorquill/<run_id>.rank<R>.jsonl`, beside the campaign's `injection.json`, and
rank 0 writes `<artifact-dir>/tensorquill/attach.json` saying how they were made, which the
comparison (`fault_injection_check.py`) reads.

Before anything starts, each rank checks what the campaign needs and stops with one line naming
what is missing: the checkout's files, `lm_resiliency` importable, PyTorch 2.10 to 2.13
(`torch>=2.10,<2.14` in its `pyproject.toml`), eight ranks and a CUDA device for each.
`--cpu-stub` skips those checks; it exists for the CPU test, which runs a stub module tree of its
own in place of the campaign, and proves only that the wrapper attaches and the captures are written.

    # GPU machine, from anywhere (8 GPUs):
    torchrun --standalone --nproc-per-node=8 sdk/examples/fault_injection_check_entry.py \\
        --lm-resiliency /path/to/lm-resiliency --artifact-dir runs/fault-injection
    python sdk/examples/fault_injection_check.py --artifact-dir runs/fault-injection
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

USAGE = 2
CAMPAIGN_MODULE = "tests.validation.fault_injection.pytorch"
PATCHED = ("enable_fault_injection", "_teardown", "main")
START_STEP = 1


def preflight(root: Path) -> str | None:
    """What the campaign needs that is missing, in one line, or None."""
    for needed in ("tests/validation/fault_injection/pytorch.py", "examples/production_loops/pytorch.py"):
        if not (root / needed).is_file():
            return f"{root} is not an lm-resiliency checkout (no {needed}); give --lm-resiliency"
    try:
        import lm_resiliency  # noqa: F401
    except ImportError:
        return f"lm_resiliency is not installed: pip install -e {root}"
    import torch

    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if not (2, 10) <= version < (2, 14):
        return (f"the campaign needs PyTorch 2.10 to 2.13 (lm-resiliency's torch>=2.10,<2.14); this is "
                f"{torch.__version__}")
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world != 8:
        return f"the campaign needs exactly eight ranks (WORLD_SIZE is {world or 'unset'}): torchrun --nproc-per-node=8"
    if not torch.cuda.is_available() or torch.cuda.device_count() < int(os.environ.get("LOCAL_WORLD_SIZE", "8")):
        return "the campaign needs a CUDA device for each local rank"
    return None


def _run_id() -> str:
    job = os.environ.get('TORCHELASTIC_RUN_ID')
    return f"fault-injection-{job if job and job != 'none' else 'local'}"


def environment() -> dict[str, Any]:
    """The exact versions this campaign ran on, read inside its own environment."""
    import platform

    import torch

    import tensorquill

    cuda = torch.cuda.is_available()
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": platform.python_version(),
            "tensorquill": getattr(tensorquill, "__version__", None), "tensorquill_file": tensorquill.__file__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())) if cuda else None,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu", "device_count": torch.cuda.device_count()}


def attach(module: ModuleType, artifact_dir: Path, *, every: int = 1) -> dict[str, Any]:
    """Replace `enable_fault_injection` and `_teardown` in the campaign's module. Returns the state
    the wrappers share (`session` once started)."""
    import torch.distributed as dist

    import tensorquill

    for name in PATCHED:
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"{module.__name__} has no {name}(): the campaign changed; this wrapper needs updating")
    state: dict[str, Any] = {"session": None}
    original_inject, original_teardown = module.enable_fault_injection, module._teardown

    @functools.wraps(original_inject)
    def enable_fault_injection(model: Any, optimizer: Any, *args: Any, **kwargs: Any) -> Any:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        run_id = _run_id()
        folder = artifact_dir / "tensorquill"
        folder.mkdir(parents=True, exist_ok=True)
        inner = getattr(model, "module", model)  # the DDP wrapper's module
        state["session"] = tensorquill.init(
            inner, run_id=run_id, sink=tensorquill.JsonlSink(folder / f"{run_id}.rank{rank}.jsonl"),
            every=every, capture="full", weights=1, start_step=START_STEP,
            replica_group=dist.group.WORLD if dist.is_initialized() else None, replica_every=1,
        )
        if rank == 0:
            (folder / "attach.json").write_text(json.dumps({
                "run_id": run_id, "world_size": world, "start_step": START_STEP,
                "step_to_iteration": "TensorQuill step N is the campaign's iteration N",
                "files": f"{run_id}.rank<R>.jsonl", "every": every, "capture": "full", "weights": 1,
                "replica_every": 1, "registered": "after enable_resiliency, before enable_fault_injection",
                "campaign_module": module.__name__, "campaign_file": getattr(module, "__file__", None),
                "environment": environment(),
            }, indent=2) + "\n", encoding="utf-8")
        return original_inject(model, optimizer, *args, **kwargs)

    @functools.wraps(original_teardown)
    def _teardown(*args: Any, **kwargs: Any) -> Any:
        session = state.get("session")
        if session is not None:
            try:
                session.close()
            except Exception as error:  # noqa: BLE001  the campaign's teardown still runs
                print(f"fault_injection_check_entry: closing the TensorQuill session raised {error!r}", file=sys.stderr)
        return original_teardown(*args, **kwargs)

    module.enable_fault_injection = enable_fault_injection
    module._teardown = _teardown
    return state


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--lm-resiliency", type=Path, default=os.environ.get("LM_RESILIENCY_ROOT"),
                        help="the lm-resiliency checkout (default: LM_RESILIENCY_ROOT)")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="the campaign's own argument, also read here")
    parser.add_argument("--tensorquill-every", type=int, default=1)
    parser.add_argument("--cpu-stub", action="store_true",
                        help="skip the campaign's checks: for the CPU test's stub module only")
    ours, theirs = parser.parse_known_args(argv)
    if ours.lm_resiliency is None:
        parser.error("give --lm-resiliency (the lm-resiliency checkout) or set LM_RESILIENCY_ROOT")
    root = ours.lm_resiliency.resolve()
    if not ours.cpu_stub:
        missing = preflight(root)
        if missing:
            if os.environ.get("RANK", "0") == "0":
                print(f"fault_injection_check_entry: {missing}", flush=True)
            return USAGE
    sys.path.insert(0, str(root))  # the campaign imports `examples.` and `tests.` from the checkout's root
    module = importlib.import_module(CAMPAIGN_MODULE)
    attach(module, ours.artifact_dir, every=ours.tensorquill_every)
    sys.argv = [str(getattr(module, "__file__", CAMPAIGN_MODULE)), "--artifact-dir", str(ours.artifact_dir), *theirs]
    module.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
