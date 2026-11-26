"""Wave AO, lane AO-8: the helpers (`datawait.py`, `mirror.py`, `collectives.py`, `data.py`).

Each section holds the test that showed one finding on the candidate and the tests that pin what
the fix must leave as it was. Optional packages (NumPy, wandb, mlflow, PyTorch's Flight Recorder
analyzer) are asked for test by test, so the file runs on an interpreter with PyTorch alone.
"""

from __future__ import annotations

import time

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import tensorquill
from tensorquill import capture, datawait
from tensorquill.sink import MemorySink


def _model():
    torch.manual_seed(0)
    return nn.Linear(1, 1)


def _open(model, **kwargs):
    sink = MemorySink()
    options = dict(sink=sink, run_id="ao8", layers=(), grads=False, weights=False, heartbeat_seconds=None,
                   stack_seconds=None)
    options.update(kwargs)
    return tensorquill.init(model, **options), sink


# ---- BUG-17: a DataLoader fetched inside another DataLoader's fetch ---------------------------------


@pytest.fixture(autouse=True)
def _no_session_left_watching():
    yield
    for session in list(datawait._watching):
        datawait.unwatch(session)


class _Clock:
    """`time` for `datawait` and `capture`, with `perf_counter` moved by hand; the rest is `time`'s."""

    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    stand_in = _Clock()
    monkeypatch.setattr(datawait, "time", stand_in)
    monkeypatch.setattr(capture, "time", stand_in)
    return stand_in


def _charging(clock, seconds):
    """A collate function that spends `seconds` on the stand-in clock and hands its batch over."""

    def collate(batch):
        clock.now += seconds
        return batch

    return collate


def _run_steps(session, clock, fetch, steps, *, train=4.0):
    """`steps` steps that each fetch once (`fetch()`) and then train for `train` seconds."""
    session.step()  # the clock starts at the first step's end
    for _ in range(steps):
        fetch()
        clock.now += train
        session.step()


def _waits(sink):
    return [(r["data_wait"], r["step_time"]) for r in sink.records if r.get("step_time") is not None]


def test_a_loader_fetched_inside_another_loaders_fetch_is_counted_once(clock):
    inner = iter(DataLoader(range(64), collate_fn=_charging(clock, 1.0)))
    outer = iter(DataLoader(range(64), collate_fn=lambda _: next(inner)))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        _run_steps(session, clock, lambda: next(outer), 5)
    assert _waits(sink) == [(1.0, 5.0)] * 5


def test_nested_loaders_with_a_fifth_of_the_step_waiting_raise_no_starvation(clock):
    """The two runs spend the same simulated time: one second in the loader, four training."""
    verdicts, shares = {}, {}
    for nested in (False, True):
        inner = iter(DataLoader(range(64), collate_fn=_charging(clock, 1.0)))
        loader = iter(DataLoader(range(64), collate_fn=lambda _: next(inner))) if nested else inner
        session, sink = _open(_model(), auto_step=False, data_wait=True)
        with session:
            _run_steps(session, clock, lambda: next(loader), 24)
        shares[nested] = sorted({wait / took for wait, took in _waits(sink)})
        incidents = tensorquill.build_evidence(sink.records, detectors="python")["incidents"]
        verdicts[nested] = [i["kind"] for i in incidents if i["kind"] == "input_starvation"]
    assert verdicts == {False: [], True: []}
    assert shares == {False: [0.2], True: [0.2]}


def test_a_time_data_iterator_fetched_inside_a_loaders_fetch_is_counted_once(clock):
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        inner = iter(session.time_data(DataLoader(range(64), collate_fn=_charging(clock, 1.0))))
        outer = iter(DataLoader(range(64), collate_fn=lambda _: next(inner)))
        _run_steps(session, clock, lambda: next(outer), 3)
    assert _waits(sink) == [(1.0, 5.0)] * 3


class _Sleeping(Dataset):
    def __init__(self, seconds):
        self.seconds = seconds

    def __len__(self):
        return 64

    def __getitem__(self, i):
        time.sleep(self.seconds)
        return torch.ones(1)


def test_on_the_wall_clock_nested_loaders_never_wait_longer_than_the_step():
    inner = iter(DataLoader(_Sleeping(0.01), batch_size=None))
    outer = iter(DataLoader(range(64), batch_size=None, collate_fn=lambda _: next(inner)))
    model = _model()
    session, sink = _open(model, auto_step=False, data_wait=True)
    with session:
        session.step()
        for _ in range(6):
            model(next(outer).reshape(1, 1)).sum().backward()
            session.step()
    waits = _waits(sink)
    assert len(waits) == 6 and all(0 < wait <= took for wait, took in waits)


# Pinned beside the fix: what counted before counts as before.


def test_the_flag_is_down_after_an_epoch_ends_and_the_next_epoch_counts(clock):
    loader = DataLoader(range(2), collate_fn=_charging(clock, 1.0))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        session.step()
        for _epoch in range(2):
            for _batch in loader:  # each epoch's last `next()` raises StopIteration inside the timer
                clock.now += 4.0
                session.step()
            assert session._fetching is False
    assert _waits(sink) == [(1.0, 5.0)] * 4


def test_the_flag_is_down_after_a_fetch_that_raises_and_the_next_fetch_counts(clock):
    calls = []

    def collate(batch):
        calls.append(batch)
        clock.now += 1.0
        if len(calls) == 1:
            raise RuntimeError("a bad batch")
        return batch

    loader = iter(DataLoader(range(8), collate_fn=collate))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        session.step()
        with pytest.raises(RuntimeError, match="a bad batch"):
            next(loader)
        assert session._fetching is False
        next(loader)
        clock.now += 4.0
        session.step()
    assert _waits(sink) == [(2.0, 6.0)]


def test_two_loaders_one_after_the_other_in_a_step_each_count(clock):
    first = iter(DataLoader(range(8), collate_fn=_charging(clock, 1.0)))
    second = iter(DataLoader(range(8), collate_fn=_charging(clock, 2.0)))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        _run_steps(session, clock, lambda: (next(first), next(second)), 2)
    assert _waits(sink) == [(3.0, 7.0)] * 2


def test_time_data_around_a_loader_counts_once(clock):
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        loader = iter(session.time_data(DataLoader(range(8), collate_fn=_charging(clock, 1.0))))
        _run_steps(session, clock, lambda: next(loader), 2)
    assert _waits(sink) == [(1.0, 5.0)] * 2


def test_two_sessions_on_one_thread_each_count_a_nested_fetch_once(clock):
    inner = iter(DataLoader(range(8), collate_fn=_charging(clock, 1.0)))
    outer = iter(DataLoader(range(8), collate_fn=lambda _: next(inner)))
    first, first_sink = _open(_model(), auto_step=False, data_wait=True, run_id="ao8-a")
    second, second_sink = _open(_model(), auto_step=False, data_wait=True, run_id="ao8-b")
    with first, second:
        first.step()
        second.step()
        for _ in range(2):
            next(outer)
            clock.now += 4.0
            first.step()
            second.step()
    assert _waits(first_sink) == _waits(second_sink) == [(1.0, 5.0)] * 2


def test_a_fetch_under_the_eval_phase_is_not_counted_and_leaves_the_flag_down(clock):
    inner = iter(DataLoader(range(8), collate_fn=_charging(clock, 1.0)))
    outer = iter(DataLoader(range(8), collate_fn=lambda _: next(inner)))
    session, sink = _open(_model(), auto_step=False, data_wait=True)
    with session:
        session.step()
        with session.phase("eval"):
            next(outer)
            assert session._fetching is False
        next(outer)
        clock.now += 4.0
        session.step()
    assert _waits(sink) == [(1.0, 5.0)]


# ---- BUG-27: a bool tensor in a list of Python ints ----------------------------------------------


def test_a_bool_tensor_after_an_int_is_a_bool_not_the_integer_one():
    from tensorquill.data import batch_hash

    ones = batch_hash([1, 1])
    for flag in (torch.tensor(True), torch.tensor([True])):
        assert batch_hash([1, flag]) != ones
        assert batch_hash([1, flag]) == batch_hash([1, True])  # as a bool tensor first already was
    assert batch_hash([1, torch.tensor(False)]) == batch_hash([1, False]) != batch_hash([1, 0])


def test_ranks_drawing_one_and_a_bool_tensor_are_not_a_duplicate_batch():
    from tensorquill.data import batch_hash
    from tensorquill.data_checks import detect_duplicate_batches

    def records(second):
        return [{"rank": rank, "step": 9, "data": {"batches": 1, "batch_hash": batch_hash([1, value])}}
                for rank, value in enumerate((1, second))]

    assert detect_duplicate_batches(records(1)) is not None  # the control: the same batch on two ranks
    assert detect_duplicate_batches(records(True)) is None
    assert detect_duplicate_batches(records(torch.tensor(True))) is None


# Pinned beside the fix: every other id keeps its hash.


def test_integer_ids_of_other_types_after_an_int_keep_the_hash_of_the_ints():
    import enum

    from tensorquill.data import batch_hash

    class Shard(enum.IntEnum):
        TWO = 2

    expected = batch_hash([1, 2])
    assert expected == "v2:cb1bef76b5e68e77"
    for two in (torch.tensor(2), torch.tensor([2]), torch.tensor(2, dtype=torch.uint8),
                torch.tensor(2, dtype=torch.int32), Shard.TWO):
        assert batch_hash([1, two]) == expected, two
    assert batch_hash(torch.tensor([1, 2])) == expected


def test_numpy_integer_ids_after_an_int_keep_the_hash_of_the_ints():
    np = pytest.importorskip("numpy")
    from tensorquill.data import batch_hash

    assert batch_hash([1, np.int64(2)]) == batch_hash([1, np.int32(2)]) == batch_hash([1, 2])


def test_the_hashes_written_before_stay_the_same():
    """Values of the candidate's own encoding, for ids the fix does not touch."""
    from tensorquill.data import batch_hash

    assert batch_hash([1, 1]) == "v2:bb153dc27acf9bbf"
    assert batch_hash([1, True]) == "v3:a8d6c9c56c1210a5"
    assert batch_hash([True, 1]) == batch_hash([torch.tensor(True), 1]) == "v3:67e453c50d1b70b3"
    assert batch_hash(list(range(1024))) == batch_hash(torch.arange(1024))
    assert batch_hash([1, 2**63]) == "v2:5cf5707f7c6801b7"  # beyond int64: typed items
    assert batch_hash(["a", 1]) == "v2:cc40eb8dfc65b276"
    assert batch_hash([1, 2.5]) == "v2:bab08894da12d9bd"


# ---- BUG-19: PyTorch Flight Recorder groups and the analyzer's "group info" line -----------------
#
# The dumps below have the keys and value types PyTorch's writer gives them (`FlightRecorderDetail.hpp`,
# `getPgConfig` and the entry dictionary, format "2.10" from PyTorch 2.10 to 2.15): `pg_config` maps each
# group's uid to its name, its description and its ranks as a string, and each entry names the group as
# `(uid, description)`. A group made without a description has "undefined" (`distributed_c10d.py`,
# `_new_process_group_helper`). The analyzer is PyTorch's own, as installed.


def _analyzer():
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")  # PyTorch 2.10 or later
    from tensorquill.collectives import analyze_flight_recorder

    return analyze_flight_recorder


def _fr_entry(record_id, uid, desc, seq, op, state="completed", sizes=((4, 4),)):
    shapes = [list(s) for s in sizes]
    return {
        "record_id": record_id, "pg_id": int(uid), "process_group": (uid, desc), "thread_name": "pt_main_thread",
        "thread_id": "1", "collective_seq_id": seq, "p2p_seq_id": 0, "op_id": seq, "profiling_name": f"nccl:{op}",
        "time_created_ns": 1_000 * (record_id + 1), "input_sizes": shapes, "input_dtypes": ["Float"] * len(shapes),
        "output_sizes": shapes, "output_dtypes": ["Float"] * len(shapes), "state": state,
        "time_discovered_started_ns": None, "time_discovered_completed_ns": None, "retired": state == "completed",
        "timeout_ms": 600_000, "is_p2p": False, "frames": [],
    }


def _fr_dumps(directory, groups, entries_by_rank):
    """`groups`: {uid: (description, ranks)}; each rank's dump lists the groups it is in."""
    import pickle

    directory.mkdir()
    for rank, entries in entries_by_rank.items():
        config = {uid: {"name": uid, "desc": desc, "ranks": str(list(ranks))}
                  for uid, (desc, ranks) in groups.items() if rank in ranks}
        document = {"version": "2.10", "comm_lib_version": "2.27.3", "pg_config": config,
                    "pg_status": {uid: {"last_enqueued_collective": 0, "last_started_collective": 0,
                                        "last_completed_collective": 0} for uid in config},
                    "entries": entries}
        (directory / f"fr_trace_rank_{rank}").write_bytes(pickle.dumps(document))
    return directory


def _desync(uid, desc, ranks=(0, 1), *, at=2, other="all_gather"):
    """Every rank of `ranks` completes all_reduce #1; at #`at` the last rank issues `other`."""
    entries = {}
    for rank in ranks:
        op = other if rank == ranks[-1] else "all_reduce"
        entries[rank] = [_fr_entry(0, uid, desc, 1, "all_reduce"), _fr_entry(1, uid, desc, at, op, "scheduled")]
    return entries


@pytest.mark.parametrize("desc, shown", [("undefined", "0"), ("default_pg", "0:default_pg")])
def test_a_desync_in_a_group_keeps_the_group_s_size_and_the_culprit(tmp_path, desc, shown):
    analyze = _analyzer()
    directory = _fr_dumps(tmp_path / "fr", {"0": (desc, (0, 1))}, _desync("0", desc))
    problem = analyze(directory)["problem"]
    assert problem["type"] == "desync" and (problem["collective"], problem["collective_seq_id"]) == ("all_reduce", 2)
    assert problem["group"] == {"id": "0", "desc": shown, "size": 2}
    assert problem["mismatch"] is not None and problem["mismatch"].startswith("Culprit rank 1")
    assert "all_gather" in problem["mismatch"]


def test_a_hang_in_a_subgroup_has_the_subgroup_s_size_not_the_world_s(tmp_path):
    analyze = _analyzer()
    world, sub = (0, 1, 2), (0, 1)
    entries = {rank: [_fr_entry(0, "0", "default_pg", 1, "all_reduce")] for rank in world}
    for rank in sub:
        entries[rank].append(_fr_entry(1, "1", "undefined", 1, "all_reduce"))
    entries[0].append(_fr_entry(2, "1", "undefined", 2, "all_reduce", "started"))  # rank 1 never entered #2
    directory = _fr_dumps(tmp_path / "fr", {"0": ("default_pg", world), "1": ("undefined", sub)}, entries)
    problem = analyze(directory)["problem"]
    assert problem["type"] == "hang" and problem["missing_ranks"] == [1] and problem["expected_ranks"] == [0, 1]
    assert problem["group"] == {"id": "1", "desc": "1", "size": 2}


def test_two_groups_on_the_same_ranks_whose_uids_share_a_prefix_are_told_apart(tmp_path):
    """Groups "1" and "10" hold the same ranks and both desync at #2, over different collectives:
    the culprit sentence is the one the analyzer wrote for the group of the problem reported."""
    analyze = _analyzer()
    one, ten = _desync("1", "undefined", other="all_gather"), _desync("10", "undefined", other="reduce_scatter")
    for rank, mine in ten.items():
        for index, entry in enumerate(mine):
            entry["record_id"] = 10 + index
    entries = {rank: one[rank] + ten[rank] for rank in (0, 1)}
    directory = _fr_dumps(tmp_path / "fr", {"1": ("undefined", (0, 1)), "10": ("undefined", (0, 1))}, entries)
    problem = analyze(directory)["problem"]
    other = {"1": "all_gather", "10": "reduce_scatter"}[problem["group"]["id"]]
    assert problem["group"]["size"] == 2 and problem["group"]["desc"] == problem["group"]["id"]
    assert problem["mismatch"].startswith("Culprit rank 1") and other in problem["mismatch"]


def test_the_incident_names_the_group_as_pytorch_prints_it(tmp_path):
    analyze = _analyzer()
    from tensorquill.collectives import flight_recorder_incident

    directory = _fr_dumps(tmp_path / "fr", {"0": ("undefined", (0, 1))}, _desync("0", "undefined"))
    incident = flight_recorder_incident(analyze(directory), step=12)
    text = tensorquill.incident_summary({"incidents": [incident], "samples": [], "layer_count": 0,
                                          "expert_count": 0, "router_names": []})
    assert "at all_reduce #2 in group 0: Culprit rank 1" in text


def test_an_analyzer_that_keys_its_groups_otherwise_still_gives_the_size_when_one_group_fits(tmp_path, monkeypatch):
    analyze = _analyzer()
    from torch.distributed.flight_recorder.components import builder

    build = builder.build_db

    def keyed_otherwise(*args, **kwargs):
        db = build(*args, **kwargs)
        return db._replace(groups=[g._replace(id=g.id + "-v") for g in db.groups],
                           memberships=[m._replace(group_id=m.group_id + "-v") for m in db.memberships])

    monkeypatch.setattr(builder, "build_db", keyed_otherwise)
    directory = _fr_dumps(tmp_path / "fr", {"0": ("undefined", (0, 1))}, _desync("0", "undefined"))
    assert analyze(directory)["problem"]["group"] == {"id": "0", "desc": "0", "size": 2}


# ---- BUG-18: a tracker's mirrored integer and an explicit `session.log` of the same integer -------
#
# Stand-ins for W&B and MLflow carry only the names the mirror wraps (`wandb.log(data)`,
# `mlflow.log_metric(key, value)`), each passing the caller's own objects, as the real ones do
# (`wandb/sdk/wandb_run.py` `Run.log`, `mlflow/tracking/fluent.py` `log_metric`). The real packages
# follow, where installed, offline.


def _stand_in_trackers(monkeypatch):
    import sys
    import types

    wandb = types.ModuleType("wandb")
    wandb.log = lambda data, step=None, commit=None: None
    mlflow = types.ModuleType("mlflow")
    mlflow.log_metric = lambda key, value, step=None, **kwargs: None
    mlflow.log_metrics = lambda metrics, step=None, **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    return wandb, mlflow


def _step_once(model, session):
    model(torch.ones(1, 1)).sum().backward()
    session.step()


_SHARED = int("500")  # one object, beyond the integers CPython keeps one of each


@pytest.mark.parametrize("value", [1, 0, -3, 256, _SHARED, 1.0], ids=["1", "0", "-3", "256", "500-shared", "1.0"])
def test_an_explicit_integer_is_kept_when_the_tracker_logs_the_key_again(monkeypatch, value):
    wandb, _ = _stand_in_trackers(monkeypatch)
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    with session:
        wandb.log({"k": value})
        session.log({"k": value})
        wandb.log({"k": 999})
        _step_once(model, session)
    assert [r["scalars"] for r in sink.records] == [{"k": value}]


@pytest.mark.parametrize("value", [1, _SHARED, 1.0], ids=["1", "500-shared", "1.0"])
def test_under_eval_an_explicit_integer_on_the_newest_step_is_kept(monkeypatch, value):
    wandb, _ = _stand_in_trackers(monkeypatch)
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    with session:
        _step_once(model, session)
        with session.phase("eval"):
            wandb.log({"eval/k": value})
            session.log({"eval/k": value})
            wandb.log({"eval/k": 999})
    assert [r.get("scalars") for r in sink.records] == [{"eval/k": value}]


def test_an_explicit_integer_after_the_step_ended_beats_the_mirror_s_carried_value(monkeypatch):
    wandb, _ = _stand_in_trackers(monkeypatch)
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    with session:
        wandb.log({"k": 1})  # the mirror's, on the step in progress
        _step_once(model, session)
        with session.phase("eval"):
            session.log({"k": 1})  # explicit, on the step that just ended
            wandb.log({"k": 999})
    assert [r["scalars"] for r in sink.records] == [{"k": 1}]


def test_mlflow_log_metric_does_not_replace_an_explicit_integer(monkeypatch):
    _, mlflow = _stand_in_trackers(monkeypatch)
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    with session:
        mlflow.log_metric("epoch", 3)
        session.log({"epoch": 3})
        mlflow.log_metric("epoch", 4)
        mlflow.log_metrics({"epoch": 5})
        _step_once(model, session)
    assert [r["scalars"] for r in sink.records] == [{"epoch": 3}]


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_real_wandb_does_not_replace_an_explicit_integer(tmp_path, monkeypatch, mode):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    run = wandb.init(project="ao8", mode=mode, dir=str(tmp_path))
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    try:
        run.log({"k": 1, "f": 1.0}, commit=False)
        session.log({"k": 1, "f": 1.0})
        wandb.log({"k": 2, "f": 2.0})
        _step_once(model, session)
    finally:
        session.close()
        run.finish()
    assert [r["scalars"] for r in sink.records] == [{"k": 1, "f": 1.0}]


def test_real_mlflow_does_not_replace_an_explicit_integer(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    try:
        with mlflow.start_run():
            mlflow.log_metric("epoch", 3)
            session.log({"epoch": 3})
            mlflow.log_metric("epoch", 4)
            _step_once(model, session)
    finally:
        session.close()
    assert [r["scalars"] for r in sink.records] == [{"epoch": 3}]


# Pinned beside the fix: what the mirror did and still does.


def test_a_key_only_the_tracker_logged_keeps_its_last_value(monkeypatch):
    wandb, _ = _stand_in_trackers(monkeypatch)
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True)
    with session:
        wandb.log({"k": 1, "n": 7})
        wandb.log({"k": 2})
        _step_once(model, session)
        wandb.log({"k": 1})  # the mirror's, on the next step in progress
        _step_once(model, session)
        wandb.log({"k": 3})
        _step_once(model, session)
        with session.phase("eval"):
            wandb.log({"k": 4})  # the mirror's own value from the step: replaced, as before
    assert [r["scalars"] for r in sink.records] == [{"k": 2, "n": 7}, {"k": 1}, {"k": 4}]


def test_a_mirrored_integer_is_written_as_a_plain_integer(monkeypatch, tmp_path):
    import json

    from tensorquill.sink import JsonlSink

    wandb, _ = _stand_in_trackers(monkeypatch)
    model = _model()
    path = tmp_path / "run.jsonl"
    session, _ = _open(model, auto_step=False, mirror_scalars=True, sink=JsonlSink(path))
    with session:
        wandb.log({"k": 1, "big": 2**40, "f": 0.5})
        _step_once(model, session)
    [line] = [json.loads(text) for text in path.read_text().splitlines() if '"scalars"' in text]
    assert line["scalars"] == {"k": 1, "big": 2**40, "f": 0.5}
    assert '"k":1,' in path.read_text() and '"big":1099511627776,' in path.read_text()


def _mirrored_then_explicit_records(monkeypatch, **options):
    wandb, _ = _stand_in_trackers(monkeypatch)
    model = _model()
    session, sink = _open(model, auto_step=False, mirror_scalars=True, **options)
    with pytest.warns(RuntimeWarning, match="left out the scalar 'k'"), session:
        for step in range(5):
            wandb.log({"k": step % 2, "only_mirrored": 7})
            if step == 3:
                session.log({"k": step % 2})
                wandb.log({"k": 9})
            _step_once(model, session)
            if step == 2:
                # With every=3 the third step is kept back when the ring fills, its row copied to the
                # CPU: the mirror's integers there are still its own, an explicit one is kept.
                with session.phase("eval"):
                    wandb.log({"only_mirrored": 8})
                    session.log({"k": 1})
                    wandb.log({"k": 9})
        session.flush()
        wandb.log({"k": 5})  # no step in memory: left out
    return sink.records


def test_the_mirror_s_integers_through_held_steps_a_flush_and_the_eval_phase(monkeypatch):
    records = _mirrored_then_explicit_records(monkeypatch, every=3)
    assert [r["scalars"] for r in records] == [
        {"k": 0, "only_mirrored": 7}, {"k": 1, "only_mirrored": 7}, {"k": 1, "only_mirrored": 8},
        {"k": 1, "only_mirrored": 7}, {"k": 0, "only_mirrored": 7},
    ]


def test_the_mirror_s_integers_make_the_same_arrow_row_as_plain_integers(monkeypatch):
    pytest.importorskip("pyarrow")
    from tensorquill.flight import records_to_batch

    records = _mirrored_then_explicit_records(monkeypatch)
    plain = [{**r, "scalars": {k: (int(v) if isinstance(v, int) else v) for k, v in r["scalars"].items()}}
             for r in records]
    assert any(type(v) is not int and isinstance(v, int) for r in records for v in r["scalars"].values())
    assert records_to_batch(records).equals(records_to_batch(plain))
