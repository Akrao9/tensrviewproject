"""Tensors the capture read wrongly or raised on, from the outside review checked Nov 18, 2026.

SDK-08: `_stats_hook` raised inside the training forward when a watched block's output was on
another device than the session's buffers, or was a DTensor. SDK-10: a complex trainable
parameter made the weight check (`_record_weights`) and the drift fingerprints
(`distributed.py`) raise, and the fingerprints' sum and projection dropped the imaginary part.
SDK-05: under FSDP2 the update size (`_read_updates`) failed at step 0 on the DTensor
optimizer state and was turned off. Nothing here may raise into training.

`meta` stands in for another device; `mps` is a real one where the machine has it. Distributed
tests use one gloo process on the CPU with an in-memory store, or two with a file store, so no
port is taken.
"""

from __future__ import annotations

import json
import math
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import distributed as D
from tensorquill.capture import STATS

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="needs Apple MPS")
COMPILED = pytest.mark.parametrize("compile_backend", [None, "eager"], ids=["eager", "compiled"])


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _hook_problems(model):
    """Each watched module's list of reasons its statistics were not read (`_stats_hook`)."""
    out = {}
    for name, module in model.named_modules():
        for hook in module._forward_hooks.values():
            if isinstance(getattr(hook, "problems", None), list):
                out[name] = hook.problems
    return out


def _train(tmp_path, model, *, steps=3, compile_backend=None, make_input=None, **options):
    """Train `steps` AdamW steps under a session; returns the records and each hook's problems."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "r1.jsonl"
    session = tensorquill.init(model, run_id="r1", sink=tensorquill.JsonlSink(path), every=steps, **options)
    problems = _hook_problems(model)
    stepped = torch.compile(model, backend=compile_backend) if compile_backend else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        optimizer.zero_grad()
        x = make_input(generator) if make_input else torch.randn(8, 4, generator=generator)
        stepped(x).backward()
        optimizer.step()
    session.close()
    return _records(path), problems


# ---- SDK-08: a block's output on another device, or a DTensor ---------------------------------


class Block(nn.Module):
    """A composite block, so `_block_layers` watches it. Its input moves to its weight's device
    (as `device_map="auto"`'s hooks do); its output is `make(output)`."""

    def __init__(self, make=None) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.make = make

    def forward(self, x):
        out = self.lin(x.to(self.lin.weight.device))
        return self.make(out) if self.make is not None else out


class Watched(nn.Module):
    """Two blocks; the second one's output is watched and not used, so it can be anything."""

    def __init__(self, make) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(make)])

    def forward(self, x):
        h = self.blocks[0](x)
        self.blocks[1](h)
        return h.pow(2).mean()


class Chain(nn.Module):
    """As the review's script: block 1 takes block 0's output, so the loss is on block 1's device."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x.pow(2).mean()


@COMPILED
def test_a_block_whose_output_holds_no_values_is_left_out_not_raised(tmp_path, compile_backend):
    records, problems = _train(tmp_path, Watched(lambda out: out.to("meta")), capture="full",
                               compile_backend=compile_backend)
    assert len(records) == 3
    assert all("blocks.0" in r["layers"] and "blocks.1" not in r["layers"] for r in records)
    assert problems == {"blocks.0": [], "blocks.1": ["its output is on meta, which holds no values"]}


@needs_mps
@pytest.mark.parametrize(("first", "second"), [("cpu", "mps"), ("mps", "cpu")], ids=["cpu-then-mps", "mps-then-cpu"])
def test_blocks_on_cpu_and_mps_train_and_only_a_read_back_is_left_out(tmp_path, first, second):
    """The review's case (`sdk08.py`) and its reverse. The buffers are on the first parameter's
    device. Block 1's statistics on MPS would have to be read back to the CPU, a wait for MPS at
    every forward, so that block is left out with its reason and training goes on (it raised).
    The reverse, a CPU output with the buffers on MPS, was read before with three blocking copies
    and is read now with one. A weight check every step reads parameters on both devices."""
    torch.manual_seed(0)
    model = Chain()
    model.blocks[0].to(first)
    model.blocks[1].to(second)
    records, problems = _train(tmp_path, model, capture="full", weights=1)
    on = {name: str(torch.empty(0, device=d).device) for name, d in (("first", first), ("second", second))}
    assert len(records) == 3
    assert problems["blocks.0"] == []
    if second == "mps":
        assert all(set(r["layers"]) == {"blocks.0"} for r in records)
        assert problems["blocks.1"] == [
            f"its output is on {on['second']}, not on {on['first']} where the layer statistics are kept, and moving "
            "its statistics there would make the host wait for a device at every forward"
        ]
    else:
        assert all(set(r["layers"]) == {"blocks.0", "blocks.1"} for r in records)
        assert problems["blocks.1"] == []
    # Every parameter's norm, on both devices, is read and true. Where the buffers are on the CPU
    # (float64 compared exactly), MPS's float32 norms cannot be compared exactly, so its two
    # parameters are never listed as unchanged; on an MPS session no parameter is.
    last = records[-1]
    for block in model.blocks:
        for name, param in (("weight", block.lin.weight), ("bias", block.lin.bias)):
            key = f"blocks.{list(model.blocks).index(block)}.lin.{name}"
            real = torch.linalg.vector_norm(param.detach().cpu().double()).item()
            assert last["weight_norms"][key] == pytest.approx(real, rel=1e-6), key
    assert not {"blocks.1.lin.weight", "blocks.1.lin.bias"} & set(last.get("weights_unchanged", []))


def test_between_two_gpus_and_from_the_cpu_the_four_numbers_are_moved_not_left_out():
    """A model split over GPUs (`device_map="auto"`): the statistics are worked out on the
    output's GPU and the four numbers copied to the buffers' GPU, a device-to-device copy that
    PyTorch orders with stream events, so the host does not wait. A CPU output is copied to the
    buffers' device in one copy. No GPU here: the decision is checked on fake tensors, and the
    copy's devices with them; the hook's whole path on two GPUs, and the copy's timing (read in
    PyTorch's source), were not run. A read back from a GPU to the CPU is left out."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tensorquill.capture import _output_unread

    gpu0, cpu = torch.device("cuda:0"), torch.device("cpu")
    with FakeTensorMode():
        on_gpu1, on_cpu = torch.randn(8, 4, device="cuda:1"), torch.randn(8, 4)
        assert _output_unread(on_gpu1, gpu0) is None and _output_unread(on_cpu, gpu0) is None
        std, mean = torch.std_mean(on_gpu1)
        moved = torch.stack((mean, std, on_gpu1.abs().amax(), (on_gpu1 == 0).float().mean())).to(gpu0)
        assert moved.device == gpu0 and moved.shape == (len(STATS),)
        assert _output_unread(torch.randn(2, device="cuda:0"), cpu) == (
            "its output is on cuda:0, not on cpu where the layer statistics are kept, and moving its statistics "
            "there would make the host wait for a device at every forward"
        )


def _dtensor_runs(tmp_path, placement):
    """The `layers` of a run whose second block hands out its output as a DTensor with
    `placement`, and of the same run with a plain output, eager and compiled; and the problems."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    mesh = init_device_mesh("cpu", (1,))
    out = {}
    for compile_backend in (None, "eager"):
        torch._dynamo.reset()
        for wrapped in (False, True):
            torch.manual_seed(0)
            make = (lambda o: DTensor.from_local(o, mesh, [placement])) if wrapped else None
            records, problems = _train(tmp_path / f"{compile_backend}-{wrapped}", Watched(make), capture="full",
                                       compile_backend=compile_backend)
            out[compile_backend, wrapped] = ([r["layers"] for r in records], problems)
    return out


@pytest.fixture
def one_gloo_rank():
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or dist.is_initialized():
        pytest.skip("needs torch.distributed and no process group yet")
    pytest.importorskip("torch.distributed.tensor")
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("placement", ["Replicate", "Shard"])
def test_a_dtensor_output_is_read_on_its_local_tensor(tmp_path, one_gloo_rank, placement):
    """Replicate: the local tensor is the output. Shard: it is this rank's part, read as
    `grad_norms` and `weight_norms` already are under FSDP2 (on one rank, all of it)."""
    from torch.distributed import tensor as dt

    runs = _dtensor_runs(tmp_path, dt.Replicate() if placement == "Replicate" else dt.Shard(0))
    for compile_backend in (None, "eager"):
        plain, _ = runs[compile_backend, False]
        layers, problems = runs[compile_backend, True]
        assert problems == {"blocks.0": [], "blocks.1": []}
        assert all("blocks.1" in step for step in layers)
        assert layers == plain


def test_a_dtensor_output_with_a_partial_placement_is_left_out(tmp_path, one_gloo_rank):
    """Each rank holds an addend of the output, not its values: no statistic of it is true."""
    from torch.distributed.tensor import Partial

    runs = _dtensor_runs(tmp_path, Partial())
    for compile_backend in (None, "eager"):
        layers, problems = runs[compile_backend, True]
        assert all("blocks.0" in step and "blocks.1" not in step for step in layers)
        assert problems["blocks.1"] == [
            "its output is a DTensor with a Partial placement: each rank holds an addend of the output, "
            "not its values"
        ]


def _float8(out):
    return out.to(torch.float8_e4m3fn)


@COMPILED
@pytest.mark.parametrize(
    ("make", "recorded"),
    [
        (_float8, True),
        (lambda out: out.double(), True),
        (lambda out: out.to_sparse(), False),
        (lambda out: out.reshape(-1)[:1].reshape(()), True),
        (lambda out: out[:0], False),
        (lambda out: out.round().to(torch.int64), False),
    ],
    ids=["float8", "float64", "sparse", "one-element", "empty", "integer"],
)
def test_other_outputs_are_read_or_left_out_never_raised(tmp_path, make, recorded, compile_backend):
    records, problems = _train(tmp_path, Watched(make), capture="full", compile_backend=compile_backend)
    assert len(records) == 3
    assert all(("blocks.1" in r["layers"]) == recorded for r in records)
    if make is _float8:
        reference = [r["layers"]["blocks.1"]["zero_frac"] for r in records]
        assert all(0.0 <= z <= 1.0 for z in reference)


def test_in_eager_mode_anything_else_reading_the_output_raises_on_is_caught(tmp_path):
    """The checks are what keep a compiled hook from raising; in eager mode whatever they do not
    foresee is caught as well. A nested tensor is one."""

    def nested(out):
        return torch.nested.nested_tensor([out[0], out[1, :3]])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")
        records, problems = _train(tmp_path, Watched(nested), capture="full")
    assert len(records) == 3 and all("blocks.1" not in r["layers"] for r in records)
    assert problems["blocks.1"] == ["its output is a nested tensor"]


def test_in_eager_mode_an_error_the_checks_do_not_foresee_is_caught_and_said(tmp_path):
    """A tensor subclass whose every operation raises stands for what no check foresees."""

    class Raises(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func in (torch.Tensor.detach, torch.Tensor.numel) or getattr(func, "__name__", "") in (
                "__get__", "is_nested", "layout", "device", "is_complex", "is_floating_point", "dtype",
            ):
                return super().__torch_function__(func, types, args, kwargs or {})
            raise RuntimeError("no operation on this tensor")

    records, problems = _train(tmp_path, Watched(lambda out: out.detach().as_subclass(Raises)), capture="full")
    assert len(records) == 3 and all("blocks.1" not in r["layers"] for r in records)
    assert problems["blocks.1"] == ["reading its output raised RuntimeError: no operation on this tensor"]


def test_a_training_forward_under_inference_mode_is_read(tmp_path):
    """A module in train mode run under `torch.inference_mode` counts as training
    (`_evaluating`); the hook's write must not raise there."""
    model = Watched(None)
    path = tmp_path / "r1.jsonl"
    session = tensorquill.init(model, run_id="r1", sink=tensorquill.JsonlSink(path), every=2, capture="full")
    problems = _hook_problems(model)
    with torch.inference_mode():
        model(torch.randn(8, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(8, 4)).backward()
        optimizer.step()
    session.close()
    assert problems == {"blocks.0": [], "blocks.1": []}
    assert all({"blocks.0", "blocks.1"} <= set(r["layers"]) for r in _records(path))


# ---- SDK-10: complex parameters ----------------------------------------------------------------


class ComplexModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, dtype=torch.complex64))
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin((x * self.w).abs()).pow(2).mean()


def _pairs_norm(t):
    return torch.linalg.vector_norm(torch.view_as_real(t.detach().resolve_conj()).double()).item()


def test_a_complex_parameter_goes_through_the_weight_check(tmp_path):
    """The weight check raised inside the training loop (at step 50 by default; every step
    here). A complex parameter's norm is its real pairs' norm, which is |z|'s. The record holds
    the norm in float32; the float64 one only decides `weights_unchanged`."""
    torch.manual_seed(0)
    model = ComplexModel()
    records, _ = _train(tmp_path, model, steps=4, weights=1, updates=True)
    assert len(records) == 4
    last = records[-1]
    assert last["weight_norms"]["w"] == pytest.approx(_pairs_norm(model.w), rel=1e-7)
    assert last["weight_dtypes"] == {"complex64": 1, "float32": 2}
    assert "w" not in last["weights_unchanged"]  # AdamW moved it
    assert set(last["update_rms"]) == {"w", "lin.weight", "lin.bias"}


def test_the_weight_check_at_its_default_step_does_not_raise_on_a_complex_parameter(tmp_path):
    """The review's script (`sdk10.py`): 55 steps with the default weight check at step 50."""
    torch.manual_seed(0)
    records, _ = _train(tmp_path, ComplexModel(), steps=55, capture="full")
    checked = [r for r in records if "weight_norms" in r]
    assert [r["step"] for r in checked] == [49]


def test_the_fingerprints_read_a_complex_tensor_as_real_pairs():
    torch.manual_seed(0)
    z = torch.randn(5, 3, dtype=torch.complex64)
    pairs = torch.view_as_real(z)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "discards the imaginary part"
        fp = D.parameter_fingerprints([z])
        assert torch.equal(fp, D.parameter_fingerprints([pairs]))
        assert D._norms_f64([z])[0].item() == pytest.approx(
            torch.linalg.vector_norm(z.to(torch.complex128)).item(), rel=1e-12)
        assert D._sum_f64(z).item() == pytest.approx(pairs.double().sum().item(), rel=1e-12)
        assert torch.equal(D._projection_f64(z, 0), D._projection_f64(pairs, 0))
        # Only the imaginary part changes: all three move.
        moved = z.clone()
        moved.imag[2, 1] += 0.5
        changed = D.parameter_fingerprints([moved])
        assert (changed != fp).all()
        # A conjugate view is read as the values it shows.
        assert torch.equal(D.parameter_fingerprints([z.conj()]), D.parameter_fingerprints([z.conj().resolve_conj()]))
    # complex32 (whose creation warns that its support is experimental) and an empty complex tensor.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ComplexHalf support is experimental")
        half = torch.zeros(4, dtype=torch.chalf)
        assert D.parameter_fingerprints([half, z[:0]]).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


class Odd(nn.Module):
    """Trainable parameters the weight check cannot read as the model's others: a conjugate
    view, complex32, and one on `meta` (no values). Only `lin` is used, so training runs."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)
        self.conj = nn.Parameter(torch.randn(3, dtype=torch.complex64).conj())
        self.small = nn.Parameter(torch.zeros(3, dtype=torch.chalf))
        self.ghost = nn.Parameter(torch.empty(3, device="meta"))

    def forward(self, x):
        return self.lin(x).pow(2).mean()


def test_the_weight_check_never_raises_and_never_makes_up_a_norm(tmp_path):
    torch.manual_seed(0)
    model = Odd()
    records, _ = _train(tmp_path, model, steps=3, weights=1)
    assert len(records) == 3
    last = records[-1]
    norms = last["weight_norms"]
    assert norms["conj"] == pytest.approx(_pairs_norm(model.conj), rel=1e-7)
    assert norms["small"] == 0.0
    assert norms["ghost"] == "nan"  # no values to read: not a number, never a made-up 0
    assert "ghost" not in last["weights_unchanged"]  # not known, so not listed
    assert {"conj", "small"} <= set(last["weights_unchanged"])  # no gradient: unchanged, truly
    assert last["weight_dtypes"] == {"float32": 3, "complex64": 1, "complex32": 1}


# ---- SDK-05: the update size under FSDP2 -------------------------------------------------------


def _cpu_fsdp2_supported() -> bool:
    """Whether `fully_shard` can run on the CPU: its all-gather records an event on a
    `torch.cpu.Stream`, which has `record_event` only from PyTorch 2.7. On 2.6 it raises inside
    PyTorch's own FSDP2 (`_fsdp_collectives.foreach_all_gather`), before TensorQuill reads anything.
    Asked of the stream itself, not of a version string."""
    stream = getattr(getattr(torch, "cpu", None), "Stream", None)
    return stream is not None and hasattr(stream, "record_event")


needs_cpu_fsdp2 = pytest.mark.skipif(
    not _cpu_fsdp2_supported(), reason="PyTorch's CPU Stream lacks record_event, which FSDP2 needs on the CPU"
)


def _fsdp2_run(tmp_path, *, shard, steps=3):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 1))
    if shard:
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    path = tmp_path / f"fsdp2-{shard}.jsonl"
    session = tensorquill.init(model, run_id="r1", sink=tensorquill.JsonlSink(path), every=steps, updates=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(8, 4, generator=generator)).pow(2).mean().backward()
        optimizer.step()
    session.close()
    return _records(path)


@needs_cpu_fsdp2
def test_fsdp2_updates_are_read_from_the_local_shard(tmp_path, one_gloo_rank):
    """The review's script (`sdk05.py`): step 0 said `update_not_read` with the DTensor error and
    nothing was read after. On one rank the local shard is the whole parameter, so the sizes are
    a plain model's."""
    plain = _fsdp2_run(tmp_path, shard=False)
    sharded = _fsdp2_run(tmp_path, shard=True)
    assert not any("update_not_read" in r for r in sharded)
    assert [set(r["update_rms"]) for r in sharded] == [{"0.weight", "0.bias", "1.weight", "1.bias"}] * 3
    for a, b in zip(plain, sharded):
        assert b["update_rms"] == pytest.approx(a["update_rms"], rel=1e-5)


class _ShardedLinear(nn.Module):
    """A linear layer whose weight and bias are DTensor parameters sharded by rows, as FSDP2
    keeps them, made without `fully_shard`. The forward reads their local tensors, so the
    gradients, and AdamW's `exp_avg` and `exp_avg_sq`, are DTensors as under FSDP2."""

    def __init__(self, linear, mesh):
        from torch.distributed.tensor import DTensor, Shard

        super().__init__()
        self.weight = nn.Parameter(DTensor.from_local(linear.weight.detach().clone(), mesh, [Shard(0)]))
        self.bias = nn.Parameter(DTensor.from_local(linear.bias.detach().clone(), mesh, [Shard(0)]))

    def forward(self, x):
        return nn.functional.linear(x, self.weight.to_local(), self.bias.to_local())


def test_dtensor_optimizer_state_is_read_without_fully_shard(tmp_path, one_gloo_rank):
    """SDK-05 where `fully_shard` cannot run on the CPU (PyTorch 2.6): the update size reads
    AdamW's state when it is a DTensor, through its local tensor. On one rank the local shard is
    the whole parameter, so the sizes are a plain model's."""
    from torch.distributed.device_mesh import init_device_mesh

    torch.manual_seed(0)
    plain_model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 1))
    mesh = init_device_mesh("cpu", (1,))
    sharded_model = nn.Sequential(*(_ShardedLinear(layer, mesh) for layer in plain_model))

    def run(model, name):
        path = tmp_path / f"{name}.jsonl"
        session = tensorquill.init(model, run_id=name, sink=tensorquill.JsonlSink(path), every=3, updates=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(1)
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.randn(8, 4, generator=generator)).pow(2).mean().backward()
            optimizer.step()
        session.close()
        return _records(path), optimizer

    plain, _ = run(plain_model, "plain")
    sharded, optimizer = run(sharded_model, "sharded")
    from torch.distributed.tensor import DTensor

    assert all(isinstance(state["exp_avg"], DTensor) for state in optimizer.state.values())
    assert not any("update_not_read" in r for r in sharded)
    assert [set(r["update_rms"]) for r in sharded] == [{"0.weight", "0.bias", "1.weight", "1.bias"}] * 3
    for a, b in zip(plain, sharded):
        assert b["update_rms"] == pytest.approx(a["update_rms"], rel=1e-5)


def _two_rank_worker(rank, store_path, out_dir):
    import torch.distributed as dist

    store = dist.FileStore(store_path, 2)
    dist.init_process_group("gloo", rank=rank, world_size=2, store=store)
    try:
        from pathlib import Path

        _fsdp2_two_ranks(Path(out_dir), rank)
    finally:
        dist.destroy_process_group()


def _fsdp2_two_ranks(out_dir, rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 1))
    fully_shard(model, mesh=init_device_mesh("cpu", (2,)))
    path = out_dir / f"rank{rank}.jsonl"
    session = tensorquill.init(model, run_id="r1", sink=tensorquill.JsonlSink(path), every=2, updates=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(8, 4, generator=generator)).pow(2).mean().backward()
        optimizer.step()
    session.close()
    sizes = {name: p.to_local().numel() for name, p in model.named_parameters()}
    (out_dir / f"sizes{rank}.json").write_text(json.dumps(sizes))


@needs_cpu_fsdp2
def test_on_two_ranks_each_rank_reads_the_rms_of_the_rows_it_holds(tmp_path):
    """FSDP2 shards each parameter by rows. A rank's `update_rms` is the RMS over its own rows;
    weighting each rank's square by its element count gives the whole parameter's (the plain
    run's). A rank whose shard of a parameter is empty (`1.bias`, one element, on rank 1)
    reads nothing for it."""
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("needs torch.distributed")
    pytest.importorskip("torch.distributed.tensor")
    import torch.multiprocessing as mp

    mp.spawn(_two_rank_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2, join=True)
    ranks = [_records(tmp_path / f"rank{r}.jsonl") for r in (0, 1)]
    sizes = [json.loads((tmp_path / f"sizes{r}.json").read_text()) for r in (0, 1)]
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        plain = _fsdp2_run(plain_dir, shard=False, steps=2)
    finally:
        dist.destroy_process_group()
    assert sizes[1]["1.bias"] == 0 and "1.bias" not in ranks[1][0]["update_rms"]
    assert not any("update_not_read" in r for records in ranks for r in records)
    for step in range(2):
        for name, whole in plain[step]["update_rms"].items():
            held = [(records[step]["update_rms"].get(name), size[name]) for records, size in zip(ranks, sizes)]
            mean_square = sum(rms * rms * n for rms, n in held if n) / sum(n for _, n in held)
            assert math.sqrt(mean_square) == pytest.approx(whole, rel=1e-5), name
