import json
import pytest
import torch

import tensorquill


def test_build_evidence_detects_moe_collapse_and_formats_table(toy_moe):
    sink = tensorquill.MemorySink()
    session = tensorquill.init(
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

    evidence = tensorquill.build_evidence(sink.records)
    assert evidence["layer_count"] == 1
    assert evidence["expert_count"] == 4
    assert evidence["router_names"] == ["router"]

    hit = tensorquill.first_incident(evidence)
    assert hit is not None
    assert hit["step"] == 20
    assert hit["layer"] == 1
    assert hit["overloaded"] == [1]
    assert hit["dead"] == [2, 3, 4]
    assert hit["diversity"] == pytest.approx(0.25)

    summary = tensorquill.incident_summary(evidence)
    assert "First sampled bad step 20  ·  Layer 01  ·  Diversity 0.25" in summary
    assert "Overloaded: Expert 1 (100%)" in summary
    assert "Dead experts: 02, 03, 04" in summary

    table = tensorquill.expert_load_table(evidence, 20, 1)
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
        # The session gathers each parameter's fingerprints and whether it was read.
        from tensorquill.distributed import _fingerprint_table

        out_list[0].copy_(tensor)
        out_list[1].copy_(_fingerprint_table(model_b.parameters(), projection=True)[0])

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    jsonl = tmp_path / "run.jsonl"
    session = tensorquill.init(
        model_a,
        run_id="r1",
        sink=tensorquill.JsonlSink(jsonl),
        every=1,
        auto_step=False,
        replica_group=group,
        replica_every=1,
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

    records = tensorquill.load_records(jsonl)
    assert "replica_spread" in records[0]
    assert records[0]["replica_spread"]["0.weight"] == [0.0, 0.0, 0.0]

    evidence = tensorquill.build_evidence(jsonl)
    kinds = {inc["kind"]: inc for inc in evidence["incidents"]}
    assert set(kinds) == {"non_finite", "replica_drift"}
    assert kinds["non_finite"]["step"] == 5
    assert kinds["non_finite"]["layers"] == ["0"]
    assert kinds["non_finite"]["gradients"] == ["0.weight"]
    assert kinds["replica_drift"]["step"] == 5
    assert "0.weight" in kinds["replica_drift"]["parameters"]


def test_imbalance_rule_allows_balanced_two_expert_router_and_preserves_bakeoff():
    # Bake-off (128 experts, 128k tokens) and toy MoE (8 experts, 256 tokens) keep 0.35
    assert tensorquill.effective_overload_threshold(128, 128_000) == pytest.approx(0.35)
    assert tensorquill.effective_overload_threshold(8, 256) == pytest.approx(0.35)

    records = [
        {"run_id": "two-exp", "step": 0, "moe": {"router": [50, 50]}},
        {"run_id": "two-exp", "step": 1, "moe": {"router": [2, 2]}},
    ]
    evidence = tensorquill.build_evidence(records)
    assert tensorquill.first_incident(evidence) is None
    assert evidence["incidents"] == []
    assert all(s["overloaded"] == [] and s["dead"] == [] for s in evidence["samples"])

    collapsed = tensorquill.build_evidence(
        [
            {"run_id": "two-exp", "step": 0, "moe": {"router": [50, 50]}},
            {"run_id": "two-exp", "step": 1, "moe": {"router": [100, 0]}},
        ]
    )
    hit = tensorquill.first_incident(collapsed)
    assert hit is not None
    assert hit["step"] == 1
    assert hit["overloaded"] == [1]
    assert hit["dead"] == [2]

    # Pretrained 32-expert top-8 router (8,192 assignments) with 1 unused expert and
    # healthy diversity (~0.79) must not fire a false-positive dead-expert collapse.
    granite_healthy = [560] * 8 + [220] * 12 + [97] * 11 + [0]
    healthy_ev = tensorquill.build_evidence(
        [{"run_id": "granite", "step": 1, "moe": {"layer-17": granite_healthy}}]
    )
    assert healthy_ev["samples"][0]["diversity"] == pytest.approx(0.79, abs=0.02)
    assert healthy_ev["samples"][0]["overloaded"] == []
    assert healthy_ev["samples"][0]["dead"] == []
    assert tensorquill.first_incident(healthy_ev) is None

    # Seeded collapse in a 32-expert top-8 Granite router onto experts 1..8 (12.50% each)
    granite_collapsed = [1024] * 8 + [0] * 24
    granite_ev = tensorquill.build_evidence(
        [
            {"run_id": "granite-router-only", "step": 449, "moe": {"layer-12": [256] * 32}},
            {"run_id": "granite-router-only", "step": 450, "moe": {"layer-12": granite_collapsed},
             "moe_top_k": {"layer-12": 8}},
        ]
    )
    granite_hit = tensorquill.first_incident(granite_ev)
    assert granite_hit is not None
    assert granite_hit["step"] == 450
    assert granite_hit["layer"] == 1
    assert granite_hit["overloaded"] == list(range(1, 9))
    assert granite_hit["dead"] == list(range(9, 33))
    assert granite_hit["overload_rule"]["rule"] == "collapse"
    assert granite_hit["diversity"] == pytest.approx(0.25)
    assert "Expert 1 (12.5%)" in tensorquill.incident_summary(granite_ev)
    granite_table = tensorquill.expert_load_table(granite_ev, 450, 1).splitlines()
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
    evidence = tensorquill.build_evidence(records)
    assert evidence["layer_count"] == 2
    assert evidence["expert_count"] == 8
    assert tensorquill.first_incident(evidence) is None

    by_key = {(s["step"], s["layer"]): s for s in evidence["samples"]}
    # Step 0, layer 1 (4 experts padded to 8): 4 active blue + 4 neutral gray
    row0_l1 = list(tensorquill.routing_colors(by_key[0, 1], evidence["expert_count"]))
    assert len(row0_l1) == 8
    assert all(c != (236, 84, 99) for c in row0_l1)
    assert row0_l1[4:] == [tensorquill.MISSING_COLOR] * 4

    # Step 1, both routers skipped/missing: all neutral gray, never dead red
    for layer in (1, 2):
        colors = list(tensorquill.routing_colors(by_key[1, layer], evidence["expert_count"]))
        assert colors == [tensorquill.MISSING_COLOR] * 8
        table = tensorquill.expert_load_table(evidence, 1, layer)
        assert "Missing" in table
        assert "Dead" not in table

    # Step 2 after a skipped step 1: heading and delta must both reference step 0
    records_with_step2 = [
        {"run_id": "skip", "step": 0, "moe": {"r2": [50, 50]}},
        {"run_id": "skip", "step": 1, "moe": {"r2": [0, 0]}},
        {"run_id": "skip", "step": 2, "moe": {"r2": [100, 0]}},
    ]
    ev2 = tensorquill.build_evidence(records_with_step2)
    table2 = tensorquill.expert_load_table(ev2, 2, 1).splitlines()
    assert table2[0] == "**Layer 01 expert load** · step 2 vs. 0"
    assert table2[4] == "| #001 | 100.00% | +50.00 | Overloaded |"


def test_export_rerun_writes_explicit_width_and_height_for_single_layer(tmp_path, monkeypatch):
    import sys
    import types
    np = pytest.importorskip("numpy")  # the Rerun export needs it; the core SDK does not

    records = [
        {"run_id": "toy", "step": 0, "moe": {"router": [32] * 8}},
        {"run_id": "toy", "step": 1, "moe": {"router": [256, 0, 0, 0, 0, 0, 0, 0]}},
    ]
    evidence = tensorquill.build_evidence(records)
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
    tensorquill.export_rerun(evidence, rrd_path, records=records)

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
    tensorquill.export_rerun(evidence, real_rrd, records=records)
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
    evidence = tensorquill.build_evidence(
        [{"run_id": "idle", "step": 0, "moe": {"r": [256] * 32}},
         {"run_id": "idle", "step": 1, "moe": {"r": counts}}]
    )
    sample = evidence["samples"][1]
    assert sample["dead"] == []
    assert tensorquill.first_incident(evidence) is None
    colors = list(tensorquill.routing_colors(sample))
    assert colors[-1] == tensorquill.IDLE_COLOR
    assert tensorquill.DEAD_COLOR not in colors
    table = tensorquill.expert_load_table(evidence, 1, 1)
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
    evidence = tensorquill.build_evidence(_starving_records())
    hit = next(i for i in evidence["incidents"] if i["kind"] == "starved_experts")
    assert hit["step"] == 49  # first step with a full 50-step window
    assert [(e["layer"], e["expert"]) for e in hit["experts"]] == [(1, 8)]
    assert hit["experts"][0]["window_share"] == pytest.approx(17 / (50 * 1024))
    assert tensorquill.first_incident(evidence) is None  # no single step looks bad

    by_step = {s["step"]: s for s in evidence["samples"]}
    assert by_step[48]["starved"] == []
    assert by_step[59]["starved"] == [8]
    colors = list(tensorquill.routing_colors(by_step[59]))
    assert colors[-1] == tensorquill.STARVED_COLOR
    assert "| Starved |" in tensorquill.expert_load_table(evidence, 59, 1)
    summary = tensorquill.incident_summary(evidence)
    assert "Starved experts (share over 50 steps, from step 49): Layer 01 expert 8 (0.033%)" in summary
    assert "violet = starved over 50 steps" in summary


def test_recently_starved_expert_waits_for_the_window_and_dead_wins():
    # Healthy for 40 steps, then starved: the window average crosses 10% of an even share
    # only once most of the window is starved.
    evidence = tensorquill.build_evidence(_starving_records(steps=100, starve_from=40))
    hit = next(i for i in evidence["incidents"] if i["kind"] == "starved_experts")
    assert 80 < hit["step"] < 90

    # A collapsed router's dead experts stay red, not violet.
    collapsed = [
        {"run_id": "c", "step": s, "moe": {"r": [1024] + [0] * 7 if s >= 10 else [128] * 8}}
        for s in range(80)
    ]
    ev = tensorquill.build_evidence(collapsed)
    last = ev["samples"][-1]
    assert last["dead"] == list(range(2, 9)) and last["starved"] == []
    assert all(i["kind"] != "starved_experts" for i in ev["incidents"])


def test_small_routers_and_short_runs_are_not_judged_on_noise():
    # 4 experts, 8 tokens per step: an even share gets 100 assignments only after 50 steps.
    records = [
        {"run_id": "tiny", "step": s, "moe": {"r": [3, 3, 2, 0] if s % 2 else [2, 2, 2, 2]}}
        for s in range(40)
    ]
    ev = tensorquill.build_evidence(records)
    assert all(i["kind"] != "starved_experts" for i in ev["incidents"])


def _topk_records(steps=40, shift_at=30):
    # 32 experts, top-8, 1024 tokens (8192 assignments) per step. Expert 1 is picked by
    # every token throughout, as some Granite experts are, and expert 20 jumps to 90% of
    # tokens for single steps, as small batches make some experts do. At `shift_at`,
    # experts 9-15 take nearly every token while every other expert keeps a few.
    records = []
    for step in range(steps):
        counts = [1024] + [231] * 31
        if step % 7 == 3:
            counts[19] = 922
        if step >= shift_at:
            counts = [1024] + [8] * 31
            for e in range(9, 16):
                counts[e - 1] = 1000
        counts[1] += 8192 - sum(counts)
        records.append({"run_id": "topk", "step": step, "moe": {"r": counts}, "moe_top_k": {"r": 8}})
    return records


def test_top_k_jump_flags_concentration_without_dead_experts():
    evidence = tensorquill.build_evidence(_topk_records())
    assert evidence["top_k"] == {"r": 8}
    hit = tensorquill.first_incident(evidence)
    assert (hit["step"], hit["overloaded"], hit["dead"]) == (30, list(range(9, 16)), [])
    assert hit["overload_rule"]["rule"] == "diversity_drop"
    # The always-on expert 1 and the one-step jumps of expert 20 are never flagged.
    assert all(not {1, 20} & set(s["overloaded"]) for s in evidence["samples"])

    # Without k the old share threshold (unreachable at 12.5%) misses it.
    assert tensorquill.first_incident(tensorquill.build_evidence(_topk_records(), top_k=1)) is None


def test_top_k_collapse_keeps_its_baseline_until_recovery():
    rows = _topk_records(steps=145)
    healthy = _topk_records(steps=145, shift_at=145)
    rows[100:120] = healthy[100:120]
    # One missing sample interrupts reporting without adding a healthy sample to the baseline.
    rows[70] = {**rows[70], "moe": {}}
    evidence = tensorquill.build_evidence(rows, starve_window=5)
    flagged = {s["step"]: s["overloaded"] for s in evidence["samples"] if s["overloaded"]}
    expected = (set(range(30, 100)) - {70}) | set(range(120, 145))
    assert flagged == {step: list(range(9, 16)) for step in expected}
    assert tensorquill.first_incident(evidence)["step"] == 30


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("count, overloaded", [(3400, []), (3500, []), (3501, [1])])
def test_dead_experts_do_not_lower_the_share_threshold(top_k, count, overloaded):
    rest, remainder = divmod(10_000 - count, 5)
    counts = [count, *[rest + (i < remainder) for i in range(5)], 0, 0]
    sample = tensorquill.first_incident(tensorquill.build_evidence(
        [{"step": 0, "moe": {"r": counts}, "moe_top_k": {"r": top_k}}],
    ))
    assert sample["dead"] == [7, 8]
    assert sample["overloaded"] == overloaded
    assert sample["overload_rule"] == {"rule": "share", "threshold": 0.35}


def test_mature_top_k_does_not_attribute_a_dead_expert_event_to_a_stably_hot_expert():
    # Expert 1 is always picked by every token at top-4. Six experts go dead,
    # but no surviving expert's selection rate rises by 25 points.
    normal = [1000] + [200] * 15
    bad = [1000] + [334] * 3 + [333] * 6 + [0] * 6
    rows = [{"step": s, "moe": {"r": normal if s < 20 else bad}, "moe_top_k": {"r": 4}}
            for s in range(22)]
    hit = tensorquill.first_incident(tensorquill.build_evidence(rows))
    assert hit["step"] == 20 and hit["dead"] == list(range(11, 17))
    assert hit["overloaded"] == []
    assert hit["overload_rule"]["rule"] == "diversity_drop"


def _grad_records(norms):
    return [
        {"run_id": "g", "step": step, "grad_norm": (a * a + b * b) ** 0.5,
         "grad_norms": {"a": a, "b": b}}
        for step, (a, b) in enumerate(norms)
    ]


def test_grad_spike_names_the_parameter_where_it_started():
    norms = [(1.0, 0.1 + 0.01 * (s % 3)) for s in range(30)] + [(1.0, 4.0), (1.2, 30.0), (1.0, 0.1)]
    evidence = tensorquill.build_evidence(_grad_records(norms))
    hit = next(i for i in evidence["incidents"] if i["kind"] == "grad_spike")
    assert hit["step"] == 31 and hit["spike_count"] == 1
    top = hit["parameters"][0]
    assert top["name"] == "b" and top["rising_since"] == 30
    assert "Gradient spike at step 31" in tensorquill.incident_summary(evidence)
    assert "first in b" in tensorquill.incident_summary(evidence)

    quiet = [(1.0 + 0.05 * (s % 4), 0.5) for s in range(60)]
    assert tensorquill.detect_grad_spikes(_grad_records(quiet)) is None


def test_stalled_weights_detector_and_skipped_steps():
    records = []
    for step in range(8):
        records.append({
            "run_id": "w", "step": step, "grad_norm": 1.0,
            "grad_norms": {"moving": 1.0, "frozen": 0.5, "no_grad": 0.0, "blip": 0.5},
            "weight_norms": {"moving": 10.0 + step, "frozen": 3.0, "no_grad": 2.0, "blip": 1.0},
            # "blip" is unchanged on two steps only, as a few-element bf16 update can be
            "weights_unchanged": ["frozen", "no_grad"] + (["blip"] if step in (2, 3) else []),
        })
    records[5]["grad_norm"] = "inf"  # a skipped mixed-precision step is not counted
    hit = tensorquill.detect_stalled_weights(records)
    assert hit["step"] == 2 and hit["parameters"] == ["frozen"]
    assert hit["stalled"] == 1 and hit["with_gradients"] == 3 and hit["stalled_steps"] == 5

    # Older captures without the exact flag are not judged on float32 norms.
    for record in records:
        del record["weights_unchanged"]
    assert tensorquill.detect_stalled_weights(records) is None


def test_parameter_left_out_of_the_optimizer_is_reported(toy_moe):
    sink = tensorquill.MemorySink()
    session = tensorquill.init(toy_moe, sink=sink, every=2, weights=1, routers={"router": 4})
    opt = torch.optim.SGD(
        [p for n, p in toy_moe.named_parameters() if not n.startswith("out.")], lr=0.1
    )
    for _ in range(5):
        opt.zero_grad()
        toy_moe(torch.randn(16, 8)).sum().backward()
        opt.step()
    session.close()
    assert "weights_unchanged" not in sink.records[0]  # nothing to compare with yet
    # The router's projection gets no gradient through argmax, so it is unchanged too, but
    # only parameters with a gradient are reported.
    assert sorted(sink.records[1]["weights_unchanged"]) == [
        "out.bias", "out.weight", "router.proj.bias", "router.proj.weight"
    ]
    hit = tensorquill.detect_stalled_weights(sink.records)
    assert hit is not None and hit["step"] == 3
    assert sorted(hit["parameters"]) == ["out.bias", "out.weight"]
    # No optimizer that stepped holds them: the likely cause, said as such.
    assert hit["optimizer"]["state"] == "known"
    assert sorted(hit["optimizer"]["not_held"]) == ["out.bias", "out.weight"]
    assert hit["optimizer"]["held_count"] == 0
    summary = tensorquill.incident_summary(tensorquill.build_evidence(sink.records))
    assert "no optimizer that stepped holds any of them" in summary
    assert "do not prove the weights are equal" in summary


def _run_with_optimizer(toy_moe, make_optimizer, **init_kwargs):
    sink = tensorquill.MemorySink()
    session = tensorquill.init(toy_moe, sink=sink, every=2, weights=1, routers={"router": 4}, **init_kwargs)
    opt = make_optimizer(toy_moe)
    for _ in range(5):
        opt.zero_grad()
        toy_moe(torch.randn(16, 8)).sum().backward()
        opt.step()
        if not session.auto_step:
            session.step()
    session.close()
    return sink


def test_a_stalled_parameter_an_optimizer_holds_points_away_from_the_optimizer(toy_moe):
    # Every parameter is held but the learning rate is zero: the cause is not ownership.
    sink = _run_with_optimizer(toy_moe, lambda m: torch.optim.SGD(m.parameters(), lr=0.0))
    hit = tensorquill.detect_stalled_weights(sink.records)
    assert hit["optimizer"] == {"state": "known", "not_held": [], "not_held_count": 0, "held_count": hit["stalled"]}
    assert "an optimizer holds all of them" in tensorquill.incident_summary(tensorquill.build_evidence(sink.records))


def test_every_parameter_no_optimizer_holds_is_recorded_not_just_the_first_twenty():
    model = torch.nn.Sequential(*[torch.nn.Linear(2, 2) for _ in range(15)])  # 30 tensors
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2, weights=1)
    opt = torch.optim.SGD(model[0].parameters(), lr=0.1)  # the other 28 are left out
    for _ in range(6):
        model(torch.randn(3, 2)).sum().backward()
        opt.step()
        opt.zero_grad()
    session.close()
    ownership = sink.records[-1]["optimizer_ownership"]
    assert ownership["unheld_count"] == 28 and len(ownership["unheld"]) == 28
    hit = tensorquill.detect_stalled_weights(sink.records)
    assert hit["optimizer"]["not_held_count"] == hit["stalled"] == 28 and hit["optimizer"]["held_count"] == 0
    assert "no optimizer that stepped holds any of them" in tensorquill.incident_summary(
        tensorquill.build_evidence(sink.records))


def test_ownership_is_unknown_when_it_cannot_be_established(toy_moe):
    # Hooks off: nothing was seen stepping.
    sink = _run_with_optimizer(toy_moe, lambda m: torch.optim.SGD(m.parameters(), lr=0.0), auto_step=False)
    hit = tensorquill.detect_stalled_weights(sink.records)
    assert hit["optimizer"]["state"] == "unknown" and "auto_step=False" in hit["optimizer"]["reason"]
    summary = tensorquill.incident_summary(tensorquill.build_evidence(sink.records))
    assert "whether an optimizer holds them is unknown" in summary

    sink = tensorquill.MemorySink()
    session = tensorquill.init(toy_moe, sink=sink, every=2, weights=1, routers={"router": 4}, auto_step=False)
    assert session._optimizer_ownership()["state"] == "unknown"
    session.close()

    # Records from before ownership was recorded.
    old = [{"run_id": "w", "step": s, "grad_norm": 1.0, "grad_norms": {"a": 1.0},
            "weight_norms": {"a": 1.0}, "weights_unchanged": ["a"]} for s in range(4)]
    assert tensorquill.detect_stalled_weights(old)["optimizer"] == {
        "state": "unknown", "reason": "not recorded by this capture"}


def test_ownership_unknown_when_the_optimizer_holds_none_of_the_models_parameters(toy_moe):
    other = torch.nn.Linear(2, 2)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(toy_moe, sink=sink, every=2, weights=1, routers={"router": 4})
    opt = torch.optim.SGD(other.parameters(), lr=0.1)  # steps, but not this model's parameters
    for _ in range(3):
        other(torch.randn(3, 2)).sum().backward()
        opt.step()
    assert session._optimizer_ownership() == {
        "state": "unknown", "reason": "no optimizer that has stepped holds this model's parameters"}
    session.close()


def test_repeated_steps_after_resume_and_start_step(toy_moe):
    def run(sink, **kwargs):
        session = tensorquill.init(toy_moe, sink=sink, every=1, routers={"router": 4}, **kwargs)
        opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
        for _ in range(3):
            opt.zero_grad()
            toy_moe(torch.randn(16, 8)).sum().backward()
            opt.step()
        session.close()

    sink = tensorquill.MemorySink()
    run(sink)
    run(sink)  # resumed without start_step
    hit = tensorquill.detect_repeated_steps(sink.records)
    assert hit["steps"] == [0, 1, 2]
    assert "pass start_step=" in tensorquill.incident_summary(tensorquill.build_evidence(sink.records))

    sink = tensorquill.MemorySink()
    run(sink)
    run(sink, start_step=3)
    assert tensorquill.detect_repeated_steps(sink.records) is None


def _act_records(steps=120, spike=None, drift_from=None, dead_from=None):
    # Four blocks with healthy noise; block 1 carries massive outliers from the start,
    # as trained LLMs do, which must not be reported by itself.
    records = []
    for step in range(steps):
        wobble = 1.0 + 0.05 * ((step * 7) % 5 - 2) / 2
        layers = {}
        for b in range(4):
            std = (2.6 if b == 1 else 0.5) * wobble
            amax = (800.0 if b == 1 else 10.0) * std
            zero = 0.1 if b == 3 else 0.0
            if spike == (step, b):
                amax *= 20
            if drift_from is not None and b == 2 and step >= drift_from:
                std *= 1.03 ** (step - drift_from)
                amax = 10.0 * std
            if dead_from is not None and b == 3 and step >= dead_from:
                zero = 0.6
            layers[f"blocks.{b}"] = {"mean": 0.0, "std": std, "abs_max": amax, "zero_frac": zero}
        records.append({"run_id": "act", "step": step, "layers": layers})
    return records


def test_activation_detectors_on_planted_failures():
    assert tensorquill.detect_activation_anomalies(_act_records()) == []

    hits = tensorquill.detect_activation_anomalies(_act_records(spike=(70, 0)))
    assert [(h["kind"], h["step"], h["modules"][0]["module"]) for h in hits] == [
        ("activation_spike", 70, "blocks.0")
    ]

    drift = tensorquill.detect_activation_anomalies(_act_records(drift_from=60))
    hit = next(h for h in drift if h["kind"] == "activation_drift")
    assert hit["modules"][0]["module"] == "blocks.2" and hit["modules"][0]["ratio"] >= 3
    assert 60 < hit["step"] < 120

    dead = tensorquill.detect_activation_anomalies(_act_records(dead_from=80))
    hit = next(h for h in dead if h["kind"] == "dead_activations")
    assert (hit["modules"][0]["module"], hit["step"]) == ("blocks.3", 85)

    evidence = tensorquill.build_evidence(_act_records(spike=(70, 0), dead_from=80))
    summary = tensorquill.incident_summary(evidence)
    assert "Activation spike at step 70 in blocks.0: abs_max 19× its recent median; 1 step affected" in summary
    assert "Outputs going dead from step 85 in blocks.3: zero fraction 10% → 60%" in summary


def _propagating_records(first, ratios, steps=60, seed_step=40):
    """Eight blocks; from `seed_step` the blocks from `first` on show a spike with these ratios (cycled
    if fewer), as a change in one block's output does as it travels down the residual stream."""
    records = []
    for step in range(steps):
        layers = {}
        for b in range(8):
            amax = 10.0
            if step == seed_step and b >= first:
                amax *= ratios[(b - first) % len(ratios)]
            layers[f"blocks.{b}"] = {"mean": 0.0, "std": 1.0, "abs_max": amax, "zero_frac": 0.0}
        records.append({"run_id": "act", "step": step, "layers": layers})
    return records


def test_an_activation_spike_that_travels_down_the_stack_leads_with_its_first_block():
    # Seeded in block 2, growing with depth: the largest ratio is in block 7, the first in block 2.
    hits = tensorquill.detect_activation_anomalies(_propagating_records(2, [6, 8, 10, 14, 20, 30]))
    spike = next(h for h in hits if h["kind"] == "activation_spike")
    assert spike["step"] == 40 and spike["affected"] == 6
    names = [m["module"] for m in spike["modules"]]
    assert names[0] == "blocks.2"  # first in the forward pass, though its ratio is the smallest
    assert names[1] == "blocks.7"  # then by ratio, so the largest is second
    assert names[2:] == ["blocks.6", "blocks.5", "blocks.4", "blocks.3"]
    summary = tensorquill.incident_summary(tensorquill.build_evidence(
        _propagating_records(2, [6, 8, 10, 14, 20, 30])))
    assert "Activation spike at step 40 first in blocks.2: abs_max 6× its recent median, and 5 more modules " \
           "(the largest, abs_max 30×, is in blocks.7); 1 step affected" in summary


def test_the_first_block_in_the_model_leads_when_names_do_not_sort_in_model_order(tmp_path):
    # model.layers.0 to 11 sort as 0, 1, 10, 11, 2, ...: through a JSONL file and back, block 2 still leads.
    path = tmp_path / "run.rank0.jsonl"
    sink = tensorquill.JsonlSink(path)
    for step in range(60):
        layers = {
            f"model.layers.{b}": {"mean": 0.0, "std": 1.0, "zero_frac": 0.0,
                                  "abs_max": 10.0 * (6 + b if step == 40 and b >= 2 else 1)}
            for b in range(12)
        }
        sink.write([{"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "layers": layers}])
    sink.close()
    records = tensorquill.load_records(path)
    assert list(records[0]["layers"]) == [f"model.layers.{b}" for b in range(12)]
    spike = next(h for h in tensorquill.detect_activation_anomalies(records) if h["kind"] == "activation_spike")
    assert [m["module"] for m in spike["modules"]][:2] == ["model.layers.2", "model.layers.11"]


def test_the_first_block_leads_even_when_it_is_also_the_largest():
    spike = next(h for h in tensorquill.detect_activation_anomalies(_propagating_records(3, [30, 12, 10, 8, 6]))
                 if h["kind"] == "activation_spike")
    assert [m["module"] for m in spike["modules"]][:2] == ["blocks.3", "blocks.4"]
    summary = tensorquill.incident_summary(tensorquill.build_evidence(_propagating_records(3, [30, 12, 10, 8, 6])))
    assert "first in blocks.3: abs_max 30× its recent median, and 4 more modules; 1 step affected" in summary
    assert "the largest" not in summary


def test_a_cut_list_of_activation_spikes_keeps_the_first_block_and_the_largest():
    records = []
    for step in range(60):
        layers = {}
        for b in range(40):
            amax = 10.0 * (b + 6 if step == 40 else 1)  # block 0 has the smallest ratio (6), block 39 the largest
            layers[f"blocks.{b:02d}"] = {"mean": 0.0, "std": 1.0, "abs_max": amax, "zero_frac": 0.0}
        records.append({"run_id": "act", "step": step, "layers": layers})
    spike = next(h for h in tensorquill.detect_activation_anomalies(records) if h["kind"] == "activation_spike")
    names = [m["module"] for m in spike["modules"]]
    assert spike["affected"] == 40 and len(names) == 20
    assert names[:3] == ["blocks.00", "blocks.39", "blocks.38"]


def test_activation_detectors_skip_non_finite_values():
    records = _act_records(steps=30)
    records[20]["layers"]["blocks.0"]["abs_max"] = "nan"
    assert tensorquill.detect_activation_anomalies(records) == []


def test_replica_drift_check_runs_every_replica_every_steps(monkeypatch):
    import torch.distributed as dist
    from torch import nn

    model = nn.Linear(4, 2, bias=False)
    gathers = []
    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)

    def fake_all_gather(out_list, tensor, group=None):
        gathers.append(1)
        for out in out_list:
            out.copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(
        model, sink=sink, every=2, auto_step=False, replica_group=object(), replica_every=3
    )
    for _ in range(6):
        model(torch.ones(2, 4)).sum().backward()
        session.step()
    session.close()

    assert len(gathers) == 2
    assert [("replica_spread" in r) for r in sink.records] == [False, False, True] * 2
    with pytest.raises(ValueError, match="replica_every"):
        tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink(), replica_every=0)


def test_incidents_carry_logged_scalars_and_export_plots_them(tmp_path, monkeypatch):
    norms = [(1.0, 0.1 + 0.01 * (s % 3)) for s in range(30)] + [(1.0, 4.0), (1.2, 30.0), (1.0, 0.1)]
    records = _grad_records(norms)
    for record in records:
        record["scalars"] = {"train/loss": 2.0, "rl/kl": 0.01}
    records[31]["scalars"] = {"train/loss": 9.5, "rl/kl": "nan"}

    evidence = tensorquill.build_evidence(records)
    hit = next(i for i in evidence["incidents"] if i["kind"] == "grad_spike")
    assert hit["scalars"]["train/loss"] == {"value": 9.5, "median_before": 2.0}
    assert hit["scalars"]["rl/kl"]["value"] == "nan"
    summary = tensorquill.incident_summary(evidence)
    assert "Logged at step 31: rl/kl nan (median before 0.01), train/loss 9.5 (median before 2)" in summary
    assert tensorquill.scalar_context(records, 999) == {}

    rr = pytest.importorskip("rerun")
    logged = []
    real_log = rr.log
    monkeypatch.setattr(rr, "log", lambda path, *a, **kw: (logged.append(path), real_log(path, *a, **kw)))
    tensorquill.export_rerun(evidence, tmp_path / "run.rrd", records=records)
    assert "/logged/train/loss" in logged and "/logged/rl/kl" in logged
    assert logged.count("/logged/rl/kl") == len(records) - 1  # the NaN value is skipped


def _steps(attempt, steps, rank=0, **more):
    return [{"run_id": "r", "rank": rank, "attempt": attempt, "step": s, "grad_norm": float(attempt), **more}
            for s in steps]


def test_a_resumed_attempt_replaces_the_steps_it_rolled_back():
    # Attempt 1 reached step 149 and crashed; the job resumed from the checkpoint at step
    # 100 and has reached step 119. Steps 120 to 149 of attempt 1 never happened either.
    records = _steps(1, range(150)) + _steps(2, range(100, 120)) + _steps(1, range(10), rank=1)
    kept = tensorquill.load_records(records)
    assert [(r["attempt"], r["step"]) for r in kept if r["rank"] == 0] == (
        [(1, s) for s in range(100)] + [(2, s) for s in range(100, 120)]
    )
    assert [r["step"] for r in kept if r["rank"] == 1] == list(range(10))  # each rank on its own
    assert len(tensorquill.load_records(records, all_attempts=True)) == 180
    assert tensorquill.load_records(records, rank=1) == _steps(1, range(10), rank=1)
    # A resume from a checkpoint is not an incident.
    assert tensorquill.detect_repeated_steps(records) is None
    assert not [i for i in tensorquill.build_evidence(records, rank=0)["incidents"] if i["kind"] == "repeated_steps"]
    # A third attempt from step 50 rolls back both earlier ones.
    kept = tensorquill.load_records(records + _steps(3, range(50, 60)), rank=0)
    assert [(r["attempt"], r["step"]) for r in kept] == [(1, s) for s in range(50)] + [(3, s) for s in range(50, 60)]


def test_an_attempt_that_restarts_the_numbering_is_still_reported():
    records = _steps(1, range(30)) + _steps(2, range(5))  # resumed without start_step
    hit = tensorquill.detect_repeated_steps(records)
    assert hit["steps"] == [0, 1, 2, 3, 4]
    evidence = tensorquill.build_evidence(records)
    assert "pass start_step=" in tensorquill.incident_summary(evidence)
    # Records written before attempts existed all count as one attempt, as before.
    old = [{"run_id": "r", "step": s} for s in (0, 1, 1, 2)]
    assert tensorquill.load_records(old) == old
    assert tensorquill.detect_repeated_steps(old)["steps"] == [1]
    assert tensorquill.detect_repeated_steps([{"run_id": "r", "step": 4}, {"run_id": "r", "rank": 0, "step": 4}])["steps"] == [4]


def _write_parts(root, records, rows_per_part=40):
    """Lay records out as tensorquill-server does: rank=<rank>/attempt=<attempt>/part-*.parquet."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from tensorquill.flight import records_to_batch

    groups = {}
    for record in records:
        groups.setdefault((record.get("rank", 0), record.get("attempt", 0)), []).append(record)
    for (rank, attempt), group in groups.items():
        directory = root / f"rank={rank}" / f"attempt={attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(group), rows_per_part):
            chunk = group[start : start + rows_per_part]
            name = f"part-{chunk[0]['step']:08}-{chunk[-1]['step']:08}.parquet"
            pq.write_table(pa.Table.from_batches([records_to_batch(chunk)]), directory / name)


def _capture(toy_moe, path, steps, **kwargs):
    session = tensorquill.init(
        toy_moe, run_id="pq", sink=tensorquill.JsonlSink(path), every=10, capture="full",
        routers={"router": 4}, weights=5, probe=torch.randn(4, 8), probe_every=20, **kwargs,
    )
    opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
    for step in range(steps):
        opt.zero_grad()
        loss = toy_moe(torch.randn(16, 8)).pow(2).mean()
        loss.backward()
        session.log({"train/loss": loss, "sometimes_nan": float("nan") if step % 7 == 0 else 1.0})
        opt.step()
    session.close()


def test_evidence_from_parquet_parts_equals_evidence_from_the_jsonl_file(toy_moe, tmp_path):
    path = tmp_path / "pq.rank0.jsonl"
    _capture(toy_moe, path, 130)
    _capture(toy_moe, path, 40, start_step=90)  # resumed from the checkpoint at step 90
    written = tensorquill.load_records(path, all_attempts=True)
    assert len(written) == 170 and len({r["attempt"] for r in written}) == 2
    assert any("probe" in r for r in written) and any(r.get("moe_stats") for r in written)

    run_dir = tmp_path / "store" / "runs" / "pq"
    _write_parts(run_dir, written)
    for source in (run_dir, run_dir / "rank=0"):
        assert tensorquill.load_records(source, all_attempts=True) == written
        assert tensorquill.load_records(source) == tensorquill.load_records(path)
        # A server's store is expected to hold the server's progress status and this one has
        # none, which is what a local file reads when a status is named and cannot be read.
        # The file alone has no status to miss: its stall kinds are not configured.
        from_store = tensorquill.build_evidence(source)
        local = tensorquill.build_evidence(path, watchdog=run_dir / "watchdog.json")
        assert local.pop("run_end")["state"] == "closed"  # only a local file has an end marker (row S4)
        assert from_store == local
        assert from_store["kind_checks"]["job_stalled"]["value"] == "unavailable"
        alone = tensorquill.build_evidence(path)
        assert alone["kind_checks"]["job_stalled"]["value"] == "not_configured"
        assert {k: v for k, v in from_store.items() if k not in ("kind_checks", "run_state")} == {
            k: v for k, v in alone.items() if k not in ("kind_checks", "run_state", "run_end")
        }
    assert [r["step"] for r in tensorquill.load_records(run_dir)] == list(range(130))
    one_part = sorted(run_dir.rglob("part-*.parquet"))[0]
    assert tensorquill.load_records(one_part, all_attempts=True) == written[:40]


def test_watch_follows_a_growing_jsonl_file_and_new_parquet_parts(tmp_path):
    import json

    from tensorquill.evidence import watch

    records = _steps(1, range(60), moe={"r": [10, 10]})
    path = tmp_path / "w.rank0.jsonl"
    seen = []

    def grow(evidence, out):
        assert not out.with_name(out.name + ".tmp").exists()  # replaced by rename
        assert json.loads(out.read_text()) == evidence
        seen.append(len(evidence["samples"]))
        with path.open("a") as file:
            if len(seen) == 1:
                file.writelines(json.dumps(r) + "\n" for r in records[20:40])
                file.write(json.dumps(records[40])[:15])  # a line still being written
            elif len(seen) == 2:
                file.write(json.dumps(records[40])[15:] + "\n")

    path.write_text("".join(json.dumps(r) + "\n" for r in records[:20]))
    out = watch(path, interval=0.0, polls=6, on_update=grow)
    assert out == tmp_path / "w.rank0.evidence.json"
    assert seen == [20, 40, 41]  # one rebuild per change, none while nothing changed
    assert json.loads(out.read_text()) == tensorquill.build_evidence(records[:41], records_file=path)

    pytest.importorskip("pyarrow.parquet")
    run_dir = tmp_path / "store" / "runs" / "r"
    _write_parts(run_dir, records[:40])
    seen.clear()

    def more_parts(evidence, out):
        seen.append(len(evidence["samples"]))
        if len(seen) == 1:
            _write_parts(run_dir, records[40:] + _steps(2, range(30, 50), moe={"r": [10, 10]}))

    out = watch(run_dir, tmp_path / "live.evidence.json", interval=0.0, polls=4, on_update=more_parts)
    assert seen == [40, 50]  # attempt 2 resumed at step 30 and reached 49
    live = json.loads(out.read_text())
    # A store with parts and no incidents.json: the detectors ran here, and the evidence says why.
    detection = live.pop("detection")
    assert (detection["source"], detection["state"], detection["stored_last_step"]) == ("python", "missing", 49)
    # (The watcher names the store's status file, which is not there: a status was expected.)
    assert live == tensorquill.build_evidence(
        records + _steps(2, range(30, 50), moe={"r": [10, 10]}), watchdog=run_dir / "watchdog.json"
    )
    assert "no_status" in [r["code"] for r in live["run_state"]["reasons"]]
    assert watch(tmp_path / "nothing-here.jsonl", interval=0.0, polls=2) is None


def test_reading_one_rank_leaves_the_other_ranks_parts_unread(tmp_path, monkeypatch):
    from tensorquill import evidence

    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from tensorquill.flight import records_to_batch

    ranks = {rank: _steps(1, range(25 + rank), rank=rank, moe={"r": [10 + rank, 10]}) for rank in range(3)}
    run_dir = tmp_path / "runs" / "r"
    _write_parts(run_dir, [record for group in ranks.values() for record in group], rows_per_part=10)
    read = []
    real = evidence._read_part
    monkeypatch.setattr(evidence, "_read_part", lambda path: (read.append(path), real(path))[1])

    def steps_of(records):
        return [(r["rank"], r["step"]) for r in records]

    for rank, written in ranks.items():
        read.clear()
        assert steps_of(tensorquill.load_records(run_dir, rank=rank, all_attempts=True)) == steps_of(written)
        assert len(read) == -(-len(written) // 10) and {p.parent.parent.name for p in read} == {f"rank={rank}"}
        read.clear()
        tail = evidence._Tail(run_dir, rank)
        assert tail.poll() and steps_of(tail.records) == steps_of(written)
        assert {p.parent.parent.name for p in read} == {f"rank={rank}"}
        read.clear()
        assert not tail.poll() and not read  # nothing new, nothing read
    read.clear()
    assert tensorquill.load_records(run_dir, rank=7) == [] and not read  # no folder for that rank
    everyone = tensorquill.load_records(run_dir, all_attempts=True)
    assert sorted(steps_of(everyone)) == sorted(steps_of([r for group in ranks.values() for r in group]))
    assert evidence._parquet_parts(run_dir, 1) == [p for p in evidence._parquet_parts(run_dir) if "rank=1" in p.parts]

    # Without the server's folders nothing can be skipped by name, and the records are filtered.
    flat = tmp_path / "flat"
    flat.mkdir()
    mixed = [r for step in range(6) for rank in range(3) for r in _steps(1, [step], rank=rank)]
    pq.write_table(pa.Table.from_batches([records_to_batch(mixed)]), flat / "part-00000000-00000005.parquet")
    assert steps_of(tensorquill.load_records(flat, rank=1, all_attempts=True)) == [(1, s) for s in range(6)]
    tail = evidence._Tail(flat, 2)
    assert tail.poll() and steps_of(tail.records) == [(2, s) for s in range(6)]


def test_cli_reads_a_directory_of_parts(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from tensorquill.evidence import main

    pytest.importorskip("pyarrow.parquet")
    run_dir = tmp_path / "runs" / "r"
    _write_parts(run_dir, _steps(1, range(30), moe={"r": [10, 10]}) + _steps(1, range(5), rank=1, moe={"r": [1, 1]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(run_dir)])
    main()
    assert "reading rank 0" in capsys.readouterr().out
    assert len(json.loads((tmp_path / "r.evidence.json").read_text())["samples"]) == 30
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(run_dir), "--rank", "1", "-o", "one.json"])
    main()
    assert len(json.loads((tmp_path / "one.json").read_text())["samples"]) == 5


def _lr(attempt, steps, fn, rank=0):
    return [{"run_id": "r", "rank": rank, "attempt": attempt, "step": s, "lr": [fn(s)]} for s in steps]


def _warmup_then_decay(step):
    return 1e-3 * min(1.0, (step + 1) / 20) * (0.999 ** max(0, step - 20))


def test_a_resume_that_restores_the_scheduler_is_not_reported():
    # Attempt 1 reached step 59; the job resumed from step 40 with the scheduler restored.
    records = _lr(1, range(60), _warmup_then_decay) + _lr(2, range(40, 60), _warmup_then_decay)
    assert tensorquill.detect_resume_lr_changes(records) is None
    # The same when the earlier attempt stopped before the resume step (extrapolated basis).
    gap = _lr(1, range(30), _warmup_then_decay) + _lr(2, range(32, 50), _warmup_then_decay)
    assert tensorquill.detect_resume_lr_changes(gap) is None
    assert not [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] == "resume_lr_change"]


def test_warmup_starting_over_after_a_resume_is_reported_as_an_observation():
    restarted = lambda s: _warmup_then_decay(s - 40)  # the scheduler began again at step 40
    records = _lr(1, range(60), _warmup_then_decay) + _lr(2, range(40, 60), restarted)
    hit = tensorquill.detect_resume_lr_changes(records)
    assert hit["kind"] == "resume_lr_change" and hit["step"] == 40
    change = hit["changes"][0]
    assert change["basis"] == "same_step"
    assert change["resumed_lr"] == [pytest.approx(5e-5)]
    assert change["expected_lr"] == [pytest.approx(_warmup_then_decay(40))]
    assert change["equals_first_lr"] is True  # the run began at 5e-5 too
    summary = tensorquill.incident_summary(tensorquill.build_evidence(records))
    assert "after the resume at step 40" in summary and "warmup starts over" in summary and "on purpose" in summary


def test_a_resume_to_the_first_learning_rate_is_labelled():
    flat = lambda s: 3e-4
    records = _lr(1, range(30), lambda s: 3e-4 * 0.5 if s > 10 else 3e-4) + _lr(2, range(25, 35), flat)
    hit = tensorquill.detect_resume_lr_changes(records)
    assert hit["changes"][0]["equals_first_lr"] is True


def test_a_resume_after_the_earlier_attempt_stopped_uses_its_slope():
    ramp = lambda s: 1e-4 * (s + 1)
    ok = _lr(1, range(30), ramp) + _lr(2, range(34, 40), ramp)
    assert tensorquill.detect_resume_lr_changes(ok) is None
    jump = _lr(1, range(30), ramp) + _lr(2, range(34, 40), lambda s: 1e-4)
    hit = tensorquill.detect_resume_lr_changes(jump)
    assert hit["changes"][0]["basis"] == "extrapolated"


def test_a_resume_without_a_recorded_learning_rate_is_listed_as_not_checked():
    records = _steps(1, range(30)) + _steps(2, range(20, 30))
    skipped = []
    assert tensorquill.detect_resume_lr_changes(records, not_checked=skipped) is None
    assert skipped == [{"check": "resume_lr", "step": 20, "rank": 0, "reason": "no learning rate recorded in the resumed attempt"}]
    assert tensorquill.build_evidence(records)["not_checked"] == skipped
    # A restart of the numbering is left to repeated_steps.
    restart = _lr(1, range(30), lambda s: 1e-3) + _lr(2, range(5), lambda s: 5e-4)
    assert tensorquill.detect_resume_lr_changes(restart) is None


def test_weight_norm_trends_are_numbers_by_kind_and_never_an_incident():
    def rec(step, scale):
        return {"run_id": "w", "step": step, "weight_norms": {
            "embed_tokens.weight": 10.0 * scale, "layers.0.input_layernorm.weight": 5.0,
            "layers.0.q_proj.weight": 20.0, "layers.1.q_proj.weight": 40.0 * (2 - scale),
            "layers.0.q_proj.bias": 1.0}}
    records = [rec(49, 1.0), rec(99, 1.0), rec(149, 0.5)]  # the embedding halves, one matrix grows
    trends = tensorquill.weight_norm_trends(records)
    assert (trends["baseline_step"], trends["latest_step"]) == (49, 149)
    assert set(trends["kinds"]) == {"embedding", "norm", "matrix", "bias"}
    assert trends["kinds"]["embedding"]["median_ratio"] == pytest.approx(0.5)
    matrix = trends["kinds"]["matrix"]
    assert matrix["parameters"] == 2 and matrix["highest"] == {"name": "layers.1.q_proj.weight", "ratio": pytest.approx(1.5)}
    assert tensorquill.weight_norm_trends(records[:1]) is None
    evidence = tensorquill.build_evidence(records)
    assert evidence["weight_trends"] == trends
    assert evidence["incidents"] == []  # no cutoff is chosen: numbers only
    assert "weight_trends" not in tensorquill.build_evidence(records[:1])


def test_a_third_attempt_is_compared_with_the_history_that_survives_the_rollbacks():
    # Attempt 1 reached step 149, attempt 2 resumed at 100, attempt 3 resumed at 50 and
    # rolls both back. Step 50's learning rate comes from attempt 1, not from attempt 2.
    fn = _warmup_then_decay
    base = _lr(1, range(150), fn) + _lr(2, range(100, 120), fn)
    skipped = []
    assert tensorquill.detect_resume_lr_changes(base + _lr(3, range(50, 60), fn), not_checked=skipped) is None
    assert skipped == []
    hit = tensorquill.detect_resume_lr_changes(base + _lr(3, range(50, 60), lambda s: 7e-4))
    assert hit["step"] == 50 and hit["resumes"] == 1
    assert hit["changes"][0]["previous_attempt"] == 1 and hit["changes"][0]["basis"] == "same_step"


def test_a_scheduler_milestone_at_the_resume_step_is_not_reported_when_the_earlier_attempt_ended_before_it():
    # A hard kill lost the last steps, so the earlier attempt ends at step 39. The restored
    # scheduler drops the learning rate at step 40, as a StepLR milestone does. That looks
    # the same as a reset, so it is listed as not checked, with its numbers, not reported.
    step_lr = lambda s: 1e-3 if s < 40 else 1e-4
    records = _lr(1, range(40), step_lr) + _lr(2, range(40, 50), step_lr)
    skipped = []
    assert tensorquill.detect_resume_lr_changes(records, not_checked=skipped) is None
    assert len(skipped) == 1 and skipped[0]["step"] == 40
    assert skipped[0]["expected_lr"] == [pytest.approx(1e-3)] and skipped[0]["resumed_lr"] == [pytest.approx(1e-4)]
    # Warmup starting over from the same ending point is still reported.
    reset = _lr(1, range(40), lambda s: 1e-3 if s > 5 else 1e-4 * (s + 1)) + _lr(2, range(40, 50), lambda s: 1e-4)
    hit = tensorquill.detect_resume_lr_changes(reset)
    assert hit["changes"][0]["basis"] == "extrapolated" and hit["changes"][0]["equals_first_lr"] is True
    # Small learning rates are compared relatively, not against an absolute floor.
    tiny = lambda s: 1e-9 * (1 + s)
    assert tensorquill.detect_resume_lr_changes(_lr(1, range(30), tiny) + _lr(2, range(30, 40), tiny)) is None


def test_build_evidence_reads_one_rank_of_a_run_directory(tmp_path):
    # Two ranks' steps merged repeat every step number and give incidents that belong to
    # neither rank, so a directory or a list of records with several ranks needs a rank.
    pytest.importorskip("pyarrow.parquet")
    from tensorquill.evidence import watch

    def moe(rank, collapse):
        out = _steps(1, range(40), rank=rank)
        for record in out:
            record["moe"] = {"router": [25] * 8}
            if collapse and record["step"] >= 20:
                record["moe"] = {"router": [100, 0, 0, 0, 0, 0, 0, 0]}
        return out

    records = moe(0, collapse=False) + moe(1, collapse=True)
    _write_parts(tmp_path, records)
    healthy = tensorquill.build_evidence(tmp_path, rank=0)
    broken = tensorquill.build_evidence(tmp_path, rank=1)
    assert not healthy["incidents"]
    assert broken["incidents"] and broken["incidents"][0]["step"] >= 20
    assert tensorquill.build_evidence(records, rank=1)["incidents"] == broken["incidents"]
    with pytest.raises(ValueError, match=r"ranks \[0, 1\].*pass rank="):
        tensorquill.build_evidence(tmp_path)
    with pytest.raises(ValueError, match="pass rank="):
        tensorquill.build_evidence(records)
    assert tensorquill.build_evidence(moe(1, collapse=False))["incidents"] == []  # one rank needs no argument
    with pytest.raises(ValueError, match="pass rank="):  # --watch on a run of several ranks
        watch(tmp_path, tmp_path / "e.json", polls=1)
    assert watch(tmp_path, tmp_path / "e.json", rank=1, polls=1) == tmp_path / "e.json"
