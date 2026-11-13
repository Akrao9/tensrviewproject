"""The CI gate from the plan, on the parts that need no GPU: zero graph breaks and zero
recompiles with the SDK on. `aot_eager` runs Dynamo and AOTAutograd without building C++."""

import pytest
import torch

import tensorlens


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_hooks_cause_no_graph_breaks_or_recompiles(toy_moe, backend):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=3, routers={"router": 4})
    compiled = torch.compile(toy_moe, backend=backend)
    opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)

    torch._dynamo.utils.counters.clear()
    with torch._dynamo.config.patch(error_on_recompile=True):
        for _ in range(7):
            opt.zero_grad()
            compiled(torch.randn(16, 8)).sum().backward()
            opt.step()
    session.close()

    assert dict(torch._dynamo.utils.counters["graph_break"]) == {}
    assert torch._dynamo.utils.counters["frames"]["ok"] >= 1  # the model really was traced
    assert [record["step"] for record in sink.records] == list(range(7))
    assert all(sum(record["moe"]["router"]) == 16 for record in sink.records)


def test_compiled_statistics_equal_eager_statistics(toy_moe):
    x = torch.randn(16, 8)
    results = []
    for compile_it in (False, True):
        torch.manual_seed(0)
        model = type(toy_moe)()
        sink = tensorlens.MemorySink()
        session = tensorlens.init(model, sink=sink, every=1, routers={"router": 4}, auto_step=False)
        runner = torch.compile(model, backend="aot_eager") if compile_it else model
        runner(x).sum().backward()
        session.step()
        session.close()
        results.append(sink.records[0])
    eager, compiled = results
    assert compiled["moe"] == eager["moe"]
    for name, stats in eager["layers"].items():
        assert compiled["layers"][name] == pytest.approx(stats, rel=1e-5, abs=1e-7)
    assert compiled["grad_norm"] == pytest.approx(eager["grad_norm"], rel=1e-5)


def test_hook_buffers_are_marked_static_for_cudagraph_trees(toy_moe):
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    captured = []

    def fw_compiler(gm, example_inputs):
        ctx = torch._guards.TracingContext.try_get()
        meta = ctx.fw_metadata
        captured.append((set(meta.mutated_inp_runtime_indices), set(meta.static_input_indices)))
        return make_boxed_func(gm.forward)

    session = tensorlens.init(toy_moe, sink=tensorlens.MemorySink(), every=2, routers={"router": 4})
    compiled = torch.compile(toy_moe, backend=aot_autograd(fw_compiler=fw_compiler))
    compiled(torch.randn(16, 8)).sum().backward()
    session.close()

    assert captured, "fw_compiler must have been called"
    for mutated, static in captured:
        assert mutated, "forward hooks must mutate the preallocated buffers"
        assert mutated <= static, "every mutated hook buffer must be marked as a static input"

