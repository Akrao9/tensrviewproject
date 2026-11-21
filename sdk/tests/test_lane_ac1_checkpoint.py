"""Lane AC1: CHECKPOINT_TENSOR_CONTRACT.md built in the SDK and the Python detectors (TrainCheck B14).

Every `coverage` case of `fixtures/checkpoint/cases.json` is run through a session: a real
`torch.nn.Module` built from the case's rows, a real checkpoint of tensors with the case's
shapes, `session.checkpoint(state, event="load")` (or `checkpoint_files`, or a save), one step,
and the load's manifest read back from the written record. Every `records` case is run through
`detect_resume_state` and `build_evidence(detectors="python")`. Then the regressions the author
named, the Lightning adapter, and what does not change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import evidence as ev
from tensorquill.data import checkpoint_manifest, files_manifest, load_manifest
from tensorquill.data_checks import detect_resume_state

DOCUMENT = json.loads((Path(__file__).parent / "fixtures" / "checkpoint" / "cases.json").read_text())
CASES = DOCUMENT["cases"]
COVERAGE = [(name, body) for name, body in CASES.items() if body["group"] == "coverage"]
RECORDS = [(name, body) for name, body in CASES.items() if body["group"] == "records"]
LSTM_ROWS = CASES["two_lstm_tensors_left_out"]["given"]["model"]
DROPPED = ("lstm.weight_hh_l0", "lstm.bias_hh_l0")


# --- the models and checkpoints of a case ---------------------------------------------------


class ShapeRaises(nn.Parameter):
    """A parameter whose shape raises when read, once armed (after `init` has read it)."""

    armed = False

    @property
    def shape(self):  # type: ignore[override]
        if type(self).armed:
            raise RuntimeError("this parameter's shape cannot be read")
        return torch.Tensor.shape.__get__(self, type(self))


class Calls:
    def __init__(self):
        self.count = 0


class GatheringExport(nn.Module):
    """Stands in for a distributed wrapper whose `state_dict()` gathers from other ranks: it records
    every call of `state_dict()` and of its state-dict hooks."""

    def __init__(self, calls: Calls):
        super().__init__()
        self._calls = calls

    def state_dict(self, *args, **kwargs):  # noqa: D102
        self._calls.count += 1
        return super().state_dict(*args, **kwargs)


class OwnParameterWalk(nn.Module):
    """A module that replaces `named_parameters` with its own, which records being called."""

    def __init__(self, calls: Calls):
        super().__init__()
        self._calls = calls

    def named_parameters(self, *args, **kwargs):  # noqa: D102
        self._calls.count += 1
        return super().named_parameters(*args, **kwargs)


def build_model(rows, root: nn.Module | None = None, parameter=nn.Parameter) -> nn.Module:
    """A module tree with the rows' names: `[name, shape, kind]`, kind "p" a trainable parameter,
    "f" a frozen one, "b" a buffer, and a fourth element naming the parameter it is tied to."""
    root = root if root is not None else nn.Module()
    made: dict[str, nn.Parameter] = {}
    for row in rows:
        name, shape, kind = row[0], row[1], row[2]
        *path, leaf = name.split(".")
        module = root
        for part in path:
            if part not in module._modules:
                module.add_module(part, nn.Module())
            module = module._modules[part]
        if kind == "b":
            module.register_buffer(leaf, torch.zeros(shape))
        else:
            tensor = made[row[3]] if len(row) > 3 else parameter(torch.zeros(shape), requires_grad=kind == "p")
            module.register_parameter(leaf, tensor)
            made[name] = tensor
    return root


def tensors(entry):
    return {name: torch.zeros(shape) for name, shape in entry.items()}


def optimizer_state():
    return {"state": {0: {"step": torch.tensor(9.0)}}, "param_groups": [{"lr": 1e-3}]}


def checkpoint_of(given):
    if "bare" in given:
        return tensors(given["bare"])
    return {key: optimizer_state() if isinstance(value, str) else tensors(value)
            for key, value in given["checkpoint"].items()}


def session_for(model, path, **kwargs):
    return tensorquill.init(model, run_id="ac1", auto_step=False, heartbeat_seconds=None, stack_seconds=None,
                            sink=tensorquill.JsonlSink(path), **kwargs)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load_through_a_session(tmp_path, model, state=None, *, folder=None, model_state=None, before=None, after=None):
    """The manifest a session writes for a load: `checkpoint(state, event="load")` (or
    `checkpoint_files`), one step, the record read back. `before` runs just before the call and
    `after` just after it, so a stand-in counts the calls made by the load alone."""
    path = tmp_path / "records.jsonl"
    session = session_for(model, path)
    if before is not None:
        before()
    if folder is not None:
        session.checkpoint_files(folder, event="load")
    else:
        session.checkpoint(state, event="load", model_state=model_state)
    if after is not None:
        after()
    session.step()
    session.close()
    [record] = read(path)
    return record["checkpoint"]["load"]


def model_component(manifest):
    return "model" in manifest["components"]


# --- every coverage case, through a session ------------------------------------------------


@pytest.mark.parametrize("name,body", COVERAGE, ids=[n for n, _ in COVERAGE])
def test_a_coverage_case_is_what_the_session_records(tmp_path, name, body):
    given, expected = body["given"], body["expected"]
    calls = Calls()
    seen: list[int] = []
    before = None
    if name == "the_comparison_raises":
        ShapeRaises.armed = False
        model = build_model(LSTM_ROWS, parameter=ShapeRaises)
        before = lambda: setattr(ShapeRaises, "armed", True)  # noqa: E731
    elif name == "a_model_whose_state_export_gathers":
        model = build_model(given["model"], GatheringExport(calls))

        def hook(*args, **kwargs):
            calls.count += 1

        model.register_state_dict_pre_hook(hook)
        model.register_state_dict_post_hook(hook)
        model._modules["lstm"].register_state_dict_pre_hook(hook)
        before = lambda: setattr(calls, "count", 0)  # noqa: E731  init's own reads are not counted
    elif name == "a_model_that_replaces_its_parameter_walk":
        model = build_model(given["model"])
        inner = OwnParameterWalk(calls)
        for leaf, tensor in list(model._modules["lstm"]._parameters.items()):
            inner.register_parameter(leaf, tensor)
        model._modules["lstm"] = inner
        before = lambda: setattr(calls, "count", 0)  # noqa: E731
    else:
        model = build_model(given["model"])
    try:
        if "folder" in given:
            folder = tmp_path / "checkpoint-4"
            folder.mkdir()
            for file in given["folder"]:
                (folder / file).write_bytes(b"")
            manifest = load_through_a_session(tmp_path, model, folder=folder)
        elif given.get("event") == "save":
            state = checkpoint_of(given)
            path = tmp_path / "records.jsonl"
            session = session_for(model, path)
            session.step()
            session.checkpoint(state, event="save")
            session.step()
            session.close()
            [manifest] = [r["checkpoint"]["save"] for r in read(path) if "checkpoint" in r]
            assert {k: v for k, v in manifest.items() if k != "after_step"} == checkpoint_manifest(state)
        else:
            state = checkpoint_of(given)
            loader = state[given["loader"]] if "loader" in given else None
            manifest = load_through_a_session(tmp_path, model, state, model_state=loader, before=before,
                                              after=lambda: seen.append(calls.count))
    finally:
        ShapeRaises.armed = False
    assert model_component(manifest) == expected["model_component"]
    if "tensors" in expected:
        assert manifest["tensors"] == expected["tensors"]
    else:
        assert expected["tensors_key"] is False and "tensors" not in manifest
    for key in ("state_dict_called", "state_dict_hook_called", "override_called"):
        if key in expected:
            assert (seen[0] > 0) == expected[key], f"{key}: {seen[0]} calls at the load"


# --- every records case, through the Python detectors --------------------------------------


@pytest.mark.parametrize("name,body", RECORDS, ids=[n for n, _ in RECORDS])
def test_a_records_case_is_what_the_python_detectors_say(name, body):
    expected = body["expected"]
    found = detect_resume_state([dict(r) for r in body["records"]])[1]
    assert (found is not None) == expected["fires"]
    if found is not None:
        assert found["changes"][0]["findings"] == expected["findings"]
    evidence = ev.build_evidence(body["records"], detectors="python")
    assert evidence.get("checkpoint_tensors") == expected["checkpoint_tensors"]
    [hit] = [i for i in evidence["incidents"] if i["kind"] == "resume_incomplete"] or [None]
    assert (hit is not None) == expected["fires"]
    sentence = ev._describe(hit) if hit is not None else ""
    if "words" in expected:
        assert expected["words"] in sentence, sentence
    if "words_absent" in expected:
        assert expected["words_absent"] not in sentence, sentence


def test_the_incidents_sentence_names_five_and_says_how_many_more():
    body = CASES["a_resume_with_more_missing_than_named"]
    [hit] = [i for i in ev.build_evidence(body["records"])["incidents"] if i["kind"] == "resume_incomplete"]
    sentence = ev._describe(hit)
    named = ", ".join(sorted(f"blocks.{i:02d}.weight" for i in range(60))[:5])
    assert f"model 60 trainable tensors not in the checkpoint: {named} and 55 more" in sentence
    assert len(hit["changes"][0]["findings"][0]["tensors"]) == 50  # the finding keeps every name it was given


# --- the regressions the author named ------------------------------------------------------


def _lstm_state(order):
    entries = {"model": tensors({n: s for n, s, _ in LSTM_ROWS if n not in DROPPED}),
               "ema": tensors({n: s for n, s, _ in LSTM_ROWS}), "optimizer": optimizer_state()}
    return {key: entries[key] for key in order}


@pytest.mark.parametrize("order", [("model", "ema", "optimizer"), ("ema", "optimizer", "model")])
def test_an_incomplete_model_beside_a_complete_ema_names_both_missing_lstm_tensors(tmp_path, order):
    manifest = load_through_a_session(tmp_path, build_model(LSTM_ROWS), _lstm_state(order))
    assert manifest["tensors"]["entry"] == "model" and manifest["tensors"]["by"] == "name"
    assert manifest["tensors"]["missing"] == {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_two_equally_plausible_unnamed_entries_give_unavailable_in_either_order(tmp_path, order):
    full = {n: s for n, s, _ in LSTM_ROWS}
    entries = {"a": tensors(full), "b": tensors(full)}
    manifest = load_through_a_session(tmp_path, build_model(LSTM_ROWS), {key: entries[key] for key in order})
    assert manifest["tensors"] == {"state": "unavailable", "why": "ambiguous", "candidates": ["a", "b"]}


def test_no_state_export_and_no_state_dict_hook_runs_at_a_load(tmp_path):
    calls = Calls()
    model = build_model(LSTM_ROWS, GatheringExport(calls))
    hooks = Calls()

    def hook(*args, **kwargs):
        hooks.count += 1

    for module in model.modules():
        module.register_state_dict_pre_hook(hook)
        module.register_state_dict_post_hook(hook)
        module._register_load_state_dict_pre_hook(hook)

    def reset():
        calls.count = hooks.count = 0

    seen = []
    state = {"model": tensors({n: s for n, s, _ in LSTM_ROWS})}
    manifest = load_through_a_session(tmp_path, model, state, before=reset,
                                      after=lambda: seen.append((calls.count, hooks.count)))
    assert manifest["tensors"]["state"] == "compared"
    assert seen == [(0, 0)]
    reset()
    model.state_dict()  # the stand-ins do record a call
    assert calls.count == 1 and hooks.count > 0


@pytest.mark.parametrize("method", ["named_parameters", "named_buffers", "named_modules", "named_children"])
def test_a_module_that_replaces_its_walk_is_not_called_and_not_read(tmp_path, method):
    calls = Calls()

    def own(self, *args, **kwargs):
        calls.count += 1
        return getattr(nn.Module, method)(self, *args, **kwargs)

    replaced = type("Replaced", (nn.Module,), {method: own})
    model = build_model(LSTM_ROWS)
    inner = replaced()
    for leaf, tensor in list(model._modules["head"]._parameters.items()):
        inner.register_parameter(leaf, tensor)
    model._modules["head"] = inner
    state = {"model": tensors({n: s for n, s, _ in LSTM_ROWS})}
    seen = []
    manifest = load_through_a_session(tmp_path, model, state, before=lambda: setattr(calls, "count", 0),
                                      after=lambda: seen.append(calls.count))
    assert manifest["tensors"] == {"state": "unavailable", "why": "model_not_readable"}
    assert seen == [0]


def test_a_method_replaced_on_the_instance_is_not_called_either():
    calls = Calls()
    model = build_model(LSTM_ROWS)
    model._modules["lstm"].named_buffers = lambda *a, **k: (calls.__setattr__("count", calls.count + 1), iter(()))[1]
    manifest = load_manifest({"model": tensors({n: s for n, s, _ in LSTM_ROWS})}, model)
    assert manifest["tensors"] == {"state": "unavailable", "why": "model_not_readable"} and calls.count == 0


def test_no_tensor_value_is_read_from_the_model_or_the_checkpoint():
    """Meta tensors have shapes and no values: a comparison that read one would raise, and give `error`."""
    with torch.device("meta"):
        model = build_model(LSTM_ROWS)
        state = {"model": {n: torch.empty(s) for n, s, _ in LSTM_ROWS if n not in DROPPED}}
    manifest = load_manifest(state, model)
    assert manifest["tensors"]["missing"] == {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}


# --- the rest of sections 1 to 3 ------------------------------------------------------------


def test_a_buffer_that_is_not_saved_is_not_listed():
    model = build_model([["w", [2], "p"]])
    model.register_buffer("kept", torch.zeros(2))
    model.register_buffer("scratch", torch.zeros(2), persistent=False)
    manifest = load_manifest({"model": {"w": torch.zeros(2)}}, model)
    assert manifest["tensors"]["model"] == 2 and manifest["tensors"]["missing_buffers"] == {"count": 1, "names": ["kept"]}


def test_a_parameter_frozen_at_the_call_is_frozen_and_one_frozen_after_is_trainable():
    model = build_model(LSTM_ROWS)
    model._modules["lstm"].weight_hh_l0.requires_grad_(False)
    state = {"model": tensors({n: s for n, s, _ in LSTM_ROWS if n not in DROPPED})}
    tensors_at = load_manifest(state, model)["tensors"]
    assert tensors_at["missing"]["names"] == ["lstm.bias_hh_l0"]
    assert tensors_at["missing_frozen"]["names"] == ["lstm.weight_hh_l0"] and tensors_at["trainable"] == 6


def test_a_real_lstm_saved_without_two_tensors_and_loaded_with_strict_false():
    """B14 on a real `nn.LSTM`, as the workload has it."""
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm, self.head = nn.LSTM(8, 32, batch_first=True), nn.Linear(32, 10)

    model = Classifier()
    saved = {k: v.clone() for k, v in model.state_dict().items() if k not in DROPPED}
    state = {"model": saved, "optimizer": torch.optim.AdamW(model.parameters()).state_dict(), "step": 100}
    fresh = Classifier()
    assert sorted(fresh.load_state_dict(state["model"], strict=False).missing_keys) == sorted(DROPPED)
    manifest = load_manifest(state, fresh)
    assert manifest["tensors"] == {"state": "compared", "entry": "model", "by": "name", "model": 6, "trainable": 6,
                                   "checkpoint": 4, "matched": 4,
                                   "missing": {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}}


def test_a_model_state_that_is_not_an_entry_is_compared_under_no_key():
    model = build_model(LSTM_ROWS)
    copy = tensors({n: s for n, s, _ in LSTM_ROWS})
    manifest = load_manifest({"model": tensors({n: s for n, s, _ in LSTM_ROWS})}, model, model_state=copy)
    assert manifest["tensors"]["by"] == "loader" and manifest["tensors"]["entry"] is None


def test_a_model_state_one_level_down_is_named_by_its_path():
    model = build_model(LSTM_ROWS)
    inner = tensors({n: s for n, s, _ in LSTM_ROWS})
    manifest = load_manifest({"training": {"weights": inner, "epoch": 3}}, model, model_state=inner)
    assert (manifest["tensors"]["entry"], manifest["tensors"]["by"]) == ("training.weights", "loader")


def test_components_model_is_the_entry_compared():
    model = build_model(LSTM_ROWS)
    full = {n: s for n, s, _ in LSTM_ROWS}
    state = {"ema": tensors(full), "model": tensors({n: s for n, s in full.items() if n not in DROPPED})}
    assert checkpoint_manifest(state)["components"]["model"]["keys"] == ["ema", "model"]
    loaded = load_manifest(state, model)["components"]["model"]
    assert loaded["keys"] == ["model", "ema"] and loaded["tensors"] == 5
    assert loaded["keys_hash"] == checkpoint_manifest({"model": state["model"]})["components"]["model"]["keys_hash"]


def test_a_bare_state_dict_is_the_model_and_no_longer_other():
    model = build_model(LSTM_ROWS)
    state = tensors({n: s for n, s, _ in LSTM_ROWS})
    assert "model" not in checkpoint_manifest(state)["components"] and checkpoint_manifest(state)["other"]
    manifest = load_manifest(state, model)
    assert manifest["components"]["model"]["keys"] == [] and "other" not in manifest


def test_an_entry_not_identified_leaves_components_and_goes_to_other():
    model = build_model(LSTM_ROWS)
    state = {"features": {"layers.0.k": torch.zeros(2, 2)}, "step": 3}
    assert checkpoint_manifest(state)["components"]["model"]["keys"] == ["features"]
    manifest = load_manifest(state, model)
    assert "model" not in manifest["components"] and manifest["other"] == ["features"]
    assert manifest["counters"] == {"step": 3}


def test_a_folder_is_marked_at_a_load_and_not_at_a_save(tmp_path):
    folder = tmp_path / "checkpoint-2"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"")
    path = tmp_path / "records.jsonl"
    session = session_for(build_model(LSTM_ROWS), path)
    session.checkpoint_files(folder, event="load")
    session.step()
    session.checkpoint_files(folder, event="save")
    session.step()
    session.close()
    marked = [r["checkpoint"] for r in read(path) if "checkpoint" in r]
    [load] = [m["load"] for m in marked if "load" in m]
    [save] = [m["save"] for m in marked if "save" in m]
    assert load["tensors"] == {"state": "unavailable", "why": "folder"}
    assert {k: v for k, v in save.items() if k != "after_step"} == files_manifest(["model.safetensors"])


def test_the_comparison_on_several_thousand_tensors_takes_little_time():
    """The time is reported by the lane; this only keeps it from growing by orders of magnitude."""
    import time

    rows = [[f"layers.{i}.{part}", [4, 4] if part == "weight" else [4], "p"]
            for i in range(2500) for part in ("weight", "bias")]
    model = build_model(rows)
    state = {"model": {f"net.{n}": torch.zeros(s) for n, s, _ in rows}}  # the slow path: a prefix is searched
    started = time.perf_counter()
    manifest = load_manifest(state, model)
    assert time.perf_counter() - started < 5.0
    assert manifest["tensors"]["matched"] == 5000 and manifest["tensors"]["prefix"]["removed"] == "net."
    # An entry that shares a few names and fails: no prefix helps, and none is counted name by name.
    state = {"model": {**{f"other.{i}.{n}": torch.zeros(1) for i, (n, _, _) in enumerate(rows)},
                       **{n: torch.zeros(s) for n, s, _ in rows[:10]}}}
    started = time.perf_counter()
    manifest = load_manifest(state, model)
    assert time.perf_counter() - started < 5.0
    assert manifest["tensors"] == {"state": "unavailable", "why": "not_identified",
                                   "closest": {"entry": "model", "checkpoint": 5010, "matched": 10}}


# --- Lightning -----------------------------------------------------------------------------


def test_lightning_compares_its_state_dict_at_a_resume(tmp_path):
    L = pytest.importorskip("lightning.pytorch")
    from tensorquill.lightning import TensorQuillCallback

    class Module(L.LightningModule):
        def __init__(self, drop=False):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
            self.drop = drop
            self.strict_loading = False

        def training_step(self, batch, batch_idx):
            return self.net(batch).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)

        def train_dataloader(self):
            torch.manual_seed(0)
            return torch.utils.data.DataLoader(torch.randn(64, 4), batch_size=8)

        def on_save_checkpoint(self, checkpoint):
            if self.drop:  # the checkpointing logic leaves the first layer's bias out
                del checkpoint["state_dict"]["net.0.bias"]

    def fit(steps, sink, drop, ckpt_path=None):
        saver = L.callbacks.ModelCheckpoint(dirpath=tmp_path / f"ckpt-{drop}", every_n_train_steps=2, save_top_k=-1)
        trainer = L.Trainer(default_root_dir=tmp_path, max_steps=steps, logger=False, enable_progress_bar=False,
                            enable_model_summary=False, accelerator="cpu", devices=1,
                            callbacks=[TensorQuillCallback(sink=sink, run_id=f"lightning-{drop}"), saver])
        trainer.fit(Module(drop), ckpt_path=ckpt_path)

    for drop in (False, True):
        first, second = tensorquill.MemorySink(), tensorquill.MemorySink()
        fit(4, first, drop)
        assert all("tensors" not in r["checkpoint"]["save"] for r in first.records if "checkpoint" in r)
        fit(6, second, drop, ckpt_path=str(tmp_path / f"ckpt-{drop}" / "epoch=0-step=4.ckpt"))
        load = second.records[0]["checkpoint"]["load"]
        assert (load["tensors"]["entry"], load["tensors"]["by"]) == ("state_dict", "loader")
        evidence = tensorquill.build_evidence(first.records + second.records)
        [hit] = [i for i in evidence["incidents"] if i["kind"] == "resume_incomplete"]
        findings = hit["changes"][0]["findings"]
        assert evidence["checkpoint_tensors"]["entries"][0]["state"] == "compared"
        if drop:
            assert load["tensors"]["missing"] == {"count": 1, "names": ["net.0.bias"]}
            assert findings[-1] == {"component": "model", "finding": "missing_tensors", "count": 1, "tensors": ["net.0.bias"]}
        else:
            assert "missing" not in load["tensors"] and load["tensors"]["matched"] == 4
            assert findings == [{"component": "rng", "finding": "missing"}]  # Lightning saves no RNG state


# --- what does not change -------------------------------------------------------------------


def test_the_all_ranks_view_carries_each_judged_ranks_block():
    missing = CASES["a_resume_with_two_tensors_verified_missing"]
    plain = CASES["a_resume_recorded_before_this_check"]
    per_rank = {0: ev.build_evidence(missing["records"]),
                1: ev.build_evidence([{**r, "rank": 1} for r in plain["records"]])}
    combined = ev.combine_rank_evidence(per_rank)
    assert combined["checkpoint_tensors_by_rank"] == [
        {"rank": 0, "checkpoint_tensors": missing["expected"]["checkpoint_tensors"]}]
    assert "checkpoint_tensors_by_rank" not in ev.combine_rank_evidence({1: per_rank[1]})


BLOCK = CASES["a_resume_with_two_tensors_verified_missing"]["expected"]["checkpoint_tensors"]


def _store(tmp_path, block):
    import time

    run = tmp_path / "store" / "runs" / "r1"
    folder = run / "rank=0" / "attempt=1"
    folder.mkdir(parents=True)
    (folder / "part-00000000-00000009.parquet").write_bytes(b"")
    report = {"schema_version": 1, "source": "tensorquill-server", "run_id": "r1", "rank": 0, "attempt": 1,
              "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
              "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, "kind_checks": ev.kind_checks(ev.input_counts([]), []),
              **({"checkpoint_tensors": block} if block is not None else {})}
    (run / "rank=0" / "incidents.json").write_text(json.dumps(report))
    status = {"schema_version": 1, "run_id": "r1", "summary": "active", "server_running": True, "refresh_seconds": 30.0,
              "written_at_ms": int(time.time() * 1000), "world_size": 1, "missing_ranks": [], "incidents": [],
              "ranks": [{"rank": 0, "attempt": 1, "last_step": 9, "state": "active"}]}
    (run / "watchdog.json").write_text(json.dumps(status))
    return run


def test_one_ranks_evidence_from_a_server_report_carries_its_block_unchanged(tmp_path):
    run = _store(tmp_path, BLOCK)
    status = json.loads((run / "watchdog.json").read_text())
    assert ev._report_evidence(run / "rank=0", ev.read_server_report(run / "rank=0"), status)["checkpoint_tensors"] == BLOCK
    assert ev.all_ranks_evidence(run)["checkpoint_tensors_by_rank"] == [{"rank": 0, "checkpoint_tensors": BLOCK}]
    clean = _store(tmp_path / "b", None)
    status = json.loads((clean / "watchdog.json").read_text())
    assert "checkpoint_tensors" not in ev._report_evidence(clean / "rank=0", ev.read_server_report(clean / "rank=0"), status)
    assert "checkpoint_tensors_by_rank" not in ev.all_ranks_evidence(clean)


def test_the_rust_path_carries_the_servers_block(monkeypatch):
    """`build_evidence(detectors="rust")` takes `checkpoint_tensors` from the server's report, as it
    takes `skipped_updates`: a stand-in report with the block, and one without."""
    import tensorquill.server_detect as server_detect

    body = CASES["a_resume_with_two_tensors_verified_missing"]
    python = ev.build_evidence(body["records"], detectors="python")
    for block in (BLOCK, None):
        def report(records, **kwargs):
            samples = []
            return {"incidents": [dict(i) for i in python["incidents"] if i["kind"] in ev.SERVER_KINDS],
                    "checked": list(ev.SERVER_KINDS), "samples": samples,
                    "thresholds": {"overload_threshold": ev.OVERLOAD_THRESHOLD, "drift_tolerance": ev.DRIFT_TOLERANCE,
                                   "starve_window": ev.STARVE_WINDOW, "starve_ratio": ev.STARVE_RATIO, "top_k": None,
                                   "update_window": ev.UPDATE_WINDOW, "spike_sustain": ev.SPIKE_SUSTAIN},
                    "kind_checks": {k: python["kind_checks"][k] for k in ev.SERVER_KINDS},
                    **({"checkpoint_tensors": block} if block is not None else {})}
        monkeypatch.setattr(server_detect, "server_report", report)
        evidence = ev.build_evidence(body["records"], detectors="rust")
        assert evidence.get("checkpoint_tensors") == block
