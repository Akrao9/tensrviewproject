import json
import pytest
import torch

import tensorlens


def test_build_evidence_detects_moe_collapse_and_formats_table(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(
        toy_moe, run_id="moe-test", sink=sink, every=1, routers={"router": 4}, auto_step=False
    )
    # Step 10: balanced routing (1 token per expert)
    with torch.no_grad():
        toy_moe.inp.weight.copy_(torch.eye(8))
        toy_moe.inp.bias.zero_()
        toy_moe.router.proj.weight.zero_()
        toy_moe.router.proj.weight[:4, :4] = torch.eye(4)
        toy_moe.router.proj.bias.zero_()
    x0 = torch.zeros(4, 8)
    x0[:4, :4] = torch.eye(4)
    toy_moe(x0).sum().backward()
    session.step(10)

    # Step 1: force all 4 tokens to expert 0
    with torch.no_grad():
        toy_moe.router.proj.bias.copy_(torch.tensor([100.0, 0.0, 0.0, 0.0]))
    toy_moe(x0).sum().backward()
    session.step(20)
    session.close()

    evidence = tensorlens.build_evidence(sink.records)
    assert evidence["layer_count"] == 1
    assert evidence["expert_count"] == 4
    assert evidence["router_names"] == ["router"]

    hit = tensorlens.first_incident(evidence)
    assert hit is not None
    assert hit["step"] == 20
    assert hit["layer"] == 1
    assert hit["overloaded"] == [1]
    assert hit["dead"] == [2, 3, 4]
    assert hit["diversity"] == pytest.approx(0.25)

    summary = tensorlens.incident_summary(evidence)
    assert "First sampled bad step 20  ·  Layer 01  ·  Diversity 0.25" in summary
    assert "Overloaded: Expert 1 (100%)" in summary
    assert "Dead experts: 02, 03, 04" in summary

    table = tensorlens.expert_load_table(evidence, 20, 1)
    lines = table.splitlines()
    assert lines[0] == "**Layer 01 expert load** · step 20 vs. 10"
    assert lines[4] == "| #001 | 100.00% | +75.00 | Overloaded |"


def test_non_finite_and_replica_drift_detectors_from_capture_session(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch import nn

    model_a = nn.Sequential(nn.Linear(4, 2, bias=False))
    model_b = nn.Sequential(nn.Linear(4, 2, bias=False))
    with torch.no_grad():
        model_b[0].weight.copy_(model_a[0].weight)

    group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)

    def fake_all_gather(out_list, tensor, group=None):
        out_list[0].copy_(tensor)
        out_list[1].copy_(tensorlens.parameter_fingerprints(model_b.parameters()))

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    jsonl = tmp_path / "run.jsonl"
    session = tensorlens.init(
        model_a,
        run_id="r1",
        sink=tensorlens.JsonlSink(jsonl),
        every=1,
        auto_step=False,
        replica_group=group,
    )
    # Step 0: identical replicas, finite activations and gradients
    model_a(torch.ones(2, 4)).sum().backward()
    session.step(0)

    # Step 5: replica B's weights drift and model A sees a NaN batch
    with torch.no_grad():
        model_b[0].weight[0, 0] += 0.05
    model_a(torch.full((2, 4), float("nan"))).sum().backward()
    session.step(5)
    session.close()

    records = tensorlens.load_records(jsonl)
    assert "replica_spread" in records[0]
    assert records[0]["replica_spread"]["0.weight"] == [0.0, 0.0]

    evidence = tensorlens.build_evidence(jsonl)
    kinds = {inc["kind"]: inc for inc in evidence["incidents"]}
    assert set(kinds) == {"non_finite", "replica_drift"}
    assert kinds["non_finite"]["step"] == 5
    assert kinds["non_finite"]["layers"] == ["0"]
    assert kinds["non_finite"]["gradients"] == ["0.weight"]
    assert kinds["replica_drift"]["step"] == 5
    assert "0.weight" in kinds["replica_drift"]["parameters"]


def test_imbalance_rule_allows_balanced_two_expert_router_and_preserves_bakeoff():
    # Bake-off (128 experts, 128k tokens) and toy MoE (8 experts, 256 tokens) keep 0.35
    assert tensorlens.effective_overload_threshold(128, 128_000) == pytest.approx(0.35)
    assert tensorlens.effective_overload_threshold(8, 256) == pytest.approx(0.35)

    records = [
        {"run_id": "two-exp", "step": 0, "moe": {"router": [50, 50]}},
        {"run_id": "two-exp", "step": 1, "moe": {"router": [2, 2]}},
    ]
    evidence = tensorlens.build_evidence(records)
    assert tensorlens.first_incident(evidence) is None
    assert evidence["incidents"] == []
    assert all(s["overloaded"] == [] and s["dead"] == [] for s in evidence["samples"])

    collapsed = tensorlens.build_evidence(
        [
            {"run_id": "two-exp", "step": 0, "moe": {"router": [50, 50]}},
            {"run_id": "two-exp", "step": 1, "moe": {"router": [100, 0]}},
        ]
    )
    hit = tensorlens.first_incident(collapsed)
    assert hit is not None
    assert hit["step"] == 1
    assert hit["overloaded"] == [1]
    assert hit["dead"] == [2]

    # Pretrained 32-expert top-8 router (8,192 assignments) with 1 unused expert and
    # healthy diversity (~0.79) must not fire a false-positive dead-expert collapse.
    granite_healthy = [560] * 8 + [220] * 12 + [97] * 11 + [0]
    healthy_ev = tensorlens.build_evidence(
        [{"run_id": "granite", "step": 1, "moe": {"layer-17": granite_healthy}}]
    )
    assert healthy_ev["samples"][0]["diversity"] == pytest.approx(0.79, abs=0.02)
    assert healthy_ev["samples"][0]["overloaded"] == []
    assert healthy_ev["samples"][0]["dead"] == []
    assert tensorlens.first_incident(healthy_ev) is None

    # Seeded collapse in a 32-expert top-8 Granite router onto experts 1..8 (12.50% each)
    granite_collapsed = [1024] * 8 + [0] * 24
    granite_ev = tensorlens.build_evidence(
        [
            {"run_id": "granite-router-only", "step": 449, "moe": {"layer-12": [256] * 32}},
            {"run_id": "granite-router-only", "step": 450, "moe": {"layer-12": granite_collapsed}},
        ]
    )
    granite_hit = tensorlens.first_incident(granite_ev)
    assert granite_hit is not None
    assert granite_hit["step"] == 450
    assert granite_hit["layer"] == 1
    assert granite_hit["overloaded"] == list(range(1, 9))
    assert granite_hit["dead"] == list(range(9, 33))
    assert granite_hit["diversity"] == pytest.approx(0.25)
    assert "Expert 1 (12.5%)" in tensorlens.incident_summary(granite_ev)
    granite_table = tensorlens.expert_load_table(granite_ev, 450, 1).splitlines()
    assert granite_table[4] == "| #001 | 12.50% | +9.38 | Overloaded |"
    assert granite_table[-1] == "| #032 | 0.00% | -3.12 | Dead |"


def test_missing_routing_data_and_mixed_expert_counts():
    records = [
        {
            "run_id": "mixed",
            "step": 0,
            "moe": {"r4": [10, 10, 10, 10], "r8": [5] * 8},
        },
        {
            "run_id": "mixed",
            "step": 1,
            # r4 skipped (all zeros), r8 omitted from record
            "moe": {"r4": [0, 0, 0, 0]},
        },
    ]
    evidence = tensorlens.build_evidence(records)
    assert evidence["layer_count"] == 2
    assert evidence["expert_count"] == 8
    assert tensorlens.first_incident(evidence) is None

    by_key = {(s["step"], s["layer"]): s for s in evidence["samples"]}
    # Step 0, layer 1 (4 experts padded to 8): 4 active blue + 4 neutral gray
    row0_l1 = list(tensorlens.routing_colors(by_key[0, 1], evidence["expert_count"]))
    assert len(row0_l1) == 8
    assert all(c != (236, 84, 99) for c in row0_l1)
    assert row0_l1[4:] == [tensorlens.MISSING_COLOR] * 4

    # Step 1, both routers skipped/missing: all neutral gray, never dead red
    for layer in (1, 2):
        colors = list(tensorlens.routing_colors(by_key[1, layer], evidence["expert_count"]))
        assert colors == [tensorlens.MISSING_COLOR] * 8
        table = tensorlens.expert_load_table(evidence, 1, layer)
        assert "Missing" in table
        assert "Dead" not in table

    # Step 2 after a skipped step 1: heading and delta must both reference step 0
    records_with_step2 = [
        {"run_id": "skip", "step": 0, "moe": {"r2": [50, 50]}},
        {"run_id": "skip", "step": 1, "moe": {"r2": [0, 0]}},
        {"run_id": "skip", "step": 2, "moe": {"r2": [100, 0]}},
    ]
    ev2 = tensorlens.build_evidence(records_with_step2)
    table2 = tensorlens.expert_load_table(ev2, 2, 1).splitlines()
    assert table2[0] == "**Layer 01 expert load** · step 2 vs. 0"
    assert table2[4] == "| #001 | 100.00% | +50.00 | Overloaded |"


def test_export_rerun_writes_explicit_width_and_height_for_single_layer(tmp_path, monkeypatch):
    import sys
    import types
    import numpy as np

    records = [
        {"run_id": "toy", "step": 0, "moe": {"router": [32] * 8}},
        {"run_id": "toy", "step": 1, "moe": {"router": [256, 0, 0, 0, 0, 0, 0, 0]}},
    ]
    evidence = tensorlens.build_evidence(records)
    assert (evidence["layer_count"], evidence["expert_count"]) == (1, 8)

    image_calls = []
    tensor_calls = []

    fake_rr = types.ModuleType("rerun")
    fake_rrb = types.ModuleType("rerun.blueprint")
    for name in ("Blueprint", "Vertical", "Horizontal", "Tabs", "Spatial2DView", "TensorView", "TextDocumentView", "TimeSeriesView"):
        setattr(fake_rrb, name, lambda *a, **kw: (name, a, kw))
    fake_rr.blueprint = fake_rrb
    fake_rr.MediaType = types.SimpleNamespace(MARKDOWN="markdown")
    fake_rr.init = lambda *a, **kw: None
    fake_rr.save = lambda *a, **kw: None
    fake_rr.send_blueprint = lambda *a, **kw: None
    fake_rr.set_time = lambda *a, **kw: None
    fake_rr.log = lambda *a, **kw: None
    fake_rr.disconnect = lambda *a, **kw: None
    fake_rr.TextDocument = lambda *a, **kw: ("TextDocument", a, kw)
    fake_rr.SeriesPoints = lambda *a, **kw: ("SeriesPoints", a, kw)
    fake_rr.Scalars = lambda *a, **kw: ("Scalars", a, kw)

    def record_image(*args, **kwargs):
        image_calls.append((args, kwargs))
        return ("Image", args, kwargs)

    def record_tensor(arr, **kwargs):
        tensor_calls.append((np.asarray(arr), kwargs))
        return ("Tensor", arr, kwargs)

    fake_rr.Image = record_image
    fake_rr.Tensor = record_tensor

    monkeypatch.setitem(sys.modules, "rerun", fake_rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", fake_rrb)

    rrd_path = tmp_path / "toy.rrd"
    tensorlens.export_rerun(evidence, rrd_path, records=records)

    assert len(image_calls) == 2
    for _, kwargs in image_calls:
        assert kwargs["width"] == 8
        assert kwargs["height"] == 1
        assert kwargs["color_model"] == "RGB"
        assert len(kwargs["bytes"]) == 1 * 8 * 3
    assert all(arr.shape == (1, 8) for arr, _ in tensor_calls)

    # Also verify stored .rrd dimensions end-to-end when rerun-sdk is installed
    monkeypatch.undo()
    try:
        import rerun  # noqa: F401
        from rerun_bindings import RrdReader
    except ImportError:
        return

    real_rrd = tmp_path / "real_toy.rrd"
    tensorlens.export_rerun(evidence, real_rrd, records=records)
    store = RrdReader(real_rrd).store()
    view = store.view(contents={"/routing/status": ["Image:format"]})
    for batch in view.select():
        col = batch.column("Image:format:ImageFormat")
        if len(col) and col[0].is_valid:
            fmt = col[0].as_py()[0]
            assert (fmt["width"], fmt["height"]) == (8, 1)


def test_zero_token_expert_not_flagged_as_dead_is_idle_not_red():
    # One of 32 experts gets no tokens in a healthy layer: the detector ignores it.
    counts = [264] * 31 + [0]
    evidence = tensorlens.build_evidence(
        [{"run_id": "idle", "step": 0, "moe": {"r": [256] * 32}},
         {"run_id": "idle", "step": 1, "moe": {"r": counts}}]
    )
    sample = evidence["samples"][1]
    assert sample["dead"] == []
    assert tensorlens.first_incident(evidence) is None
    colors = list(tensorlens.routing_colors(sample))
    assert colors[-1] == tensorlens.IDLE_COLOR
    assert tensorlens.DEAD_COLOR not in colors
    table = tensorlens.expert_load_table(evidence, 1, 1)
    assert "| #032 | 0.00% | -3.12 | Idle |" in table
    assert "Dead" not in table


def _starving_records(steps=60, starve_from=0):
    # 8 experts, 1024 assignments per step. Expert 8 gets 1 token every third step from
    # `starve_from` on, far below its even share of 128; the others share the rest.
    records = []
    for step in range(steps):
        last = (1 if step % 3 == 0 else 0) if step >= starve_from else 128
        rest = 1024 - last
        counts = [rest // 7 + (1 if i < rest % 7 else 0) for i in range(7)] + [last]
        records.append({"run_id": "starve", "step": step, "moe": {"r": counts}})
    return records


def test_starved_expert_is_flagged_after_a_full_window():
    evidence = tensorlens.build_evidence(_starving_records())
    hit = next(i for i in evidence["incidents"] if i["kind"] == "starved_experts")
    assert hit["step"] == 49  # first step with a full 50-step window
    assert [(e["layer"], e["expert"]) for e in hit["experts"]] == [(1, 8)]
    assert hit["experts"][0]["window_share"] == pytest.approx(17 / (50 * 1024))
    assert tensorlens.first_incident(evidence) is None  # no single step looks bad

    by_step = {s["step"]: s for s in evidence["samples"]}
    assert by_step[48]["starved"] == []
    assert by_step[59]["starved"] == [8]
    colors = list(tensorlens.routing_colors(by_step[59]))
    assert colors[-1] == tensorlens.STARVED_COLOR
    assert "| Starved |" in tensorlens.expert_load_table(evidence, 59, 1)
    summary = tensorlens.incident_summary(evidence)
    assert "Starved experts (share over 50 steps, from step 49): Layer 01 expert 8 (0.033%)" in summary
    assert "violet = starved over 50 steps" in summary


def test_recently_starved_expert_waits_for_the_window_and_dead_wins():
    # Healthy for 40 steps, then starved: the window average crosses 10% of an even share
    # only once most of the window is starved.
    evidence = tensorlens.build_evidence(_starving_records(steps=100, starve_from=40))
    hit = next(i for i in evidence["incidents"] if i["kind"] == "starved_experts")
    assert 80 < hit["step"] < 90

    # A collapsed router's dead experts stay red, not violet.
    collapsed = [
        {"run_id": "c", "step": s, "moe": {"r": [1024] + [0] * 7 if s >= 10 else [128] * 8}}
        for s in range(80)
    ]
    ev = tensorlens.build_evidence(collapsed)
    last = ev["samples"][-1]
    assert last["dead"] == list(range(2, 9)) and last["starved"] == []
    assert all(i["kind"] != "starved_experts" for i in ev["incidents"])


def test_small_routers_and_short_runs_are_not_judged_on_noise():
    # 4 experts, 8 tokens per step: an even share gets 100 assignments only after 50 steps.
    records = [
        {"run_id": "tiny", "step": s, "moe": {"r": [3, 3, 2, 0] if s % 2 else [2, 2, 2, 2]}}
        for s in range(40)
    ]
    ev = tensorlens.build_evidence(records)
    assert all(i["kind"] != "starved_experts" for i in ev["incidents"])
