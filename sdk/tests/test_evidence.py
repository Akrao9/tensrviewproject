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
    np = pytest.importorskip("numpy")  # the Rerun export needs it; the core SDK does not

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
    evidence = tensorlens.build_evidence(_topk_records())
    assert evidence["top_k"] == {"r": 8}
    hit = tensorlens.first_incident(evidence)
    assert (hit["step"], hit["overloaded"], hit["dead"]) == (30, list(range(9, 16)), [])
    assert hit["overload_rule"]["rule"] == "diversity_drop"
    # The always-on expert 1 and the one-step jumps of expert 20 are never flagged.
    assert all(not {1, 20} & set(s["overloaded"]) for s in evidence["samples"])

    # Without k the old share threshold (unreachable at 12.5%) misses it.
    assert tensorlens.first_incident(tensorlens.build_evidence(_topk_records(), top_k=1)) is None


def _grad_records(norms):
    return [
        {"run_id": "g", "step": step, "grad_norm": (a * a + b * b) ** 0.5,
         "grad_norms": {"a": a, "b": b}}
        for step, (a, b) in enumerate(norms)
    ]


def test_grad_spike_names_the_parameter_where_it_started():
    norms = [(1.0, 0.1 + 0.01 * (s % 3)) for s in range(30)] + [(1.0, 4.0), (1.2, 30.0), (1.0, 0.1)]
    evidence = tensorlens.build_evidence(_grad_records(norms))
    hit = next(i for i in evidence["incidents"] if i["kind"] == "grad_spike")
    assert hit["step"] == 31 and hit["spike_count"] == 1
    top = hit["parameters"][0]
    assert top["name"] == "b" and top["rising_since"] == 30
    assert "Gradient spike at step 31" in tensorlens.incident_summary(evidence)
    assert "first in b" in tensorlens.incident_summary(evidence)

    quiet = [(1.0 + 0.05 * (s % 4), 0.5) for s in range(60)]
    assert tensorlens.detect_grad_spikes(_grad_records(quiet)) is None


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
    hit = tensorlens.detect_stalled_weights(records)
    assert hit["step"] == 2 and hit["parameters"] == ["frozen"]
    assert hit["stalled"] == 1 and hit["with_gradients"] == 3 and hit["stalled_steps"] == 5

    # Older captures without the exact flag are not judged on float32 norms.
    for record in records:
        del record["weights_unchanged"]
    assert tensorlens.detect_stalled_weights(records) is None


def test_parameter_left_out_of_the_optimizer_is_reported(toy_moe):
    sink = tensorlens.MemorySink()
    session = tensorlens.init(toy_moe, sink=sink, every=2, weights=1, routers={"router": 4})
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
    hit = tensorlens.detect_stalled_weights(sink.records)
    assert hit is not None and hit["step"] == 3
    assert sorted(hit["parameters"]) == ["out.bias", "out.weight"]


def test_repeated_steps_after_resume_and_start_step(toy_moe):
    def run(sink, **kwargs):
        session = tensorlens.init(toy_moe, sink=sink, every=1, routers={"router": 4}, **kwargs)
        opt = torch.optim.SGD(toy_moe.parameters(), lr=0.01)
        for _ in range(3):
            opt.zero_grad()
            toy_moe(torch.randn(16, 8)).sum().backward()
            opt.step()
        session.close()

    sink = tensorlens.MemorySink()
    run(sink)
    run(sink)  # resumed without start_step
    hit = tensorlens.detect_repeated_steps(sink.records)
    assert hit["steps"] == [0, 1, 2]
    assert "pass start_step=" in tensorlens.incident_summary(tensorlens.build_evidence(sink.records))

    sink = tensorlens.MemorySink()
    run(sink)
    run(sink, start_step=3)
    assert tensorlens.detect_repeated_steps(sink.records) is None


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
    assert tensorlens.detect_activation_anomalies(_act_records()) == []

    hits = tensorlens.detect_activation_anomalies(_act_records(spike=(70, 0)))
    assert [(h["kind"], h["step"], h["modules"][0]["module"]) for h in hits] == [
        ("activation_spike", 70, "blocks.0")
    ]

    drift = tensorlens.detect_activation_anomalies(_act_records(drift_from=60))
    hit = next(h for h in drift if h["kind"] == "activation_drift")
    assert hit["modules"][0]["module"] == "blocks.2" and hit["modules"][0]["ratio"] >= 3
    assert 60 < hit["step"] < 120

    dead = tensorlens.detect_activation_anomalies(_act_records(dead_from=80))
    hit = next(h for h in dead if h["kind"] == "dead_activations")
    assert (hit["modules"][0]["module"], hit["step"]) == ("blocks.3", 85)

    evidence = tensorlens.build_evidence(_act_records(spike=(70, 0), dead_from=80))
    summary = tensorlens.incident_summary(evidence)
    assert "Activation spike at step 70 in blocks.0: abs_max 19× its recent median; 1 step affected" in summary
    assert "Outputs going dead from step 85 in blocks.3: zero fraction 10% → 60%" in summary


def test_activation_detectors_skip_non_finite_values():
    records = _act_records(steps=30)
    records[20]["layers"]["blocks.0"]["abs_max"] = "nan"
    assert tensorlens.detect_activation_anomalies(records) == []


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
    sink = tensorlens.MemorySink()
    session = tensorlens.init(
        model, sink=sink, every=2, auto_step=False, replica_group=object(), replica_every=3
    )
    for _ in range(6):
        model(torch.ones(2, 4)).sum().backward()
        session.step()
    session.close()

    assert len(gathers) == 2
    assert [("replica_spread" in r) for r in sink.records] == [False, False, True] * 2
    with pytest.raises(ValueError, match="replica_every"):
        tensorlens.init(nn.Linear(2, 2), sink=tensorlens.MemorySink(), replica_every=0)


def test_incidents_carry_logged_scalars_and_export_plots_them(tmp_path, monkeypatch):
    norms = [(1.0, 0.1 + 0.01 * (s % 3)) for s in range(30)] + [(1.0, 4.0), (1.2, 30.0), (1.0, 0.1)]
    records = _grad_records(norms)
    for record in records:
        record["scalars"] = {"train/loss": 2.0, "rl/kl": 0.01}
    records[31]["scalars"] = {"train/loss": 9.5, "rl/kl": "nan"}

    evidence = tensorlens.build_evidence(records)
    hit = next(i for i in evidence["incidents"] if i["kind"] == "grad_spike")
    assert hit["scalars"]["train/loss"] == {"value": 9.5, "median_before": 2.0}
    assert hit["scalars"]["rl/kl"]["value"] == "nan"
    summary = tensorlens.incident_summary(evidence)
    assert "Logged at step 31: rl/kl nan (median before 0.01), train/loss 9.5 (median before 2)" in summary
    assert tensorlens.scalar_context(records, 999) == {}

    rr = pytest.importorskip("rerun")
    logged = []
    real_log = rr.log
    monkeypatch.setattr(rr, "log", lambda path, *a, **kw: (logged.append(path), real_log(path, *a, **kw)))
    tensorlens.export_rerun(evidence, tmp_path / "run.rrd", records=records)
    assert "/logged/train/loss" in logged and "/logged/rl/kl" in logged
    assert logged.count("/logged/rl/kl") == len(records) - 1  # the NaN value is skipped
