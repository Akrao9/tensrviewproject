"""Lane E of wave AF: the user documents and the scripts a reader runs.

1. The agent instructions name each key an MCP tool answers with that the audit after wave AD found
   missing (OPEN_BUGS.md, section 11, DOC-01), and each named key is in the answer of the tool the
   page names, on evidence built from the contracts' own cases (read here, never changed).
2. The stale scripts (DOC-02): `experiments/nanomoe/drift_measure.py` runs its published command and
   says which rule its table is; `experiments/rule-selection/inputs.py` and `measure_thresholds.py`
   hold no path of one machine. (`examples/benchmark_detectors.py` is held in
   `test_benchmark_detectors.py`.)
3. Two stated limits: router-only capture never records a parameter unfrozen after `init` while
   full capture does (SDK-01), and the restart keys are scoped only under one torchrun agent (L25).
4. The replica collectives: the guide and the README say the explicit calls block like any
   collective, and the scheduled check issues none.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import tracker
from tensorquill.capture import Router
from tensorquill.evidence import load_records
from tensorquill.sink import JsonlSink, MemorySink

SDK = Path(__file__).resolve().parents[1]
REPO = SDK.parent
FIXTURES = Path(__file__).parent / "fixtures"
RULES = json.loads((FIXTURES / "rules" / "cases.json").read_text())["cases"]
CHECKPOINT = json.loads((FIXTURES / "checkpoint" / "cases.json").read_text())["cases"]
AGENT = (SDK / "AGENT_INSTRUCTIONS.md").read_text()
GUIDE = (SDK / "RUN_ON_YOUR_JOB.md").read_text()
README = (SDK / "README.md").read_text()


def flat(text: str) -> str:
    return " ".join(text.split())


def section(text: str, title: str) -> str:
    """One `## ` section of a page, to its next `## ` heading."""
    start = text.index(f"## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def child_env() -> dict[str, str]:
    # Without NumPy (CI's streaming job), importing PyTorch warns on standard error, ahead of the one line a
    # command prints there; the filter names only that warning.
    return {**os.environ, "PYTHONPATH": str(SDK / "src"), "PYTHONWARNINGS": "ignore:Failed to initialize NumPy"}


# --- 1. The agent instructions and what the tools answer -------------------------------------------


def call(where: Path, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """A tool's answer as an agent gets it, through `call_tool`."""
    reply = mcp_server.call_tool(mcp_server.Source(where), 1, {"name": name, "arguments": arguments or {}})
    assert reply["result"]["isError"] is False, reply
    return json.loads(reply["result"]["content"][0]["text"])


def plain_rows(run: str, steps: range) -> list[dict[str, Any]]:
    return [{"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, "step": s, "ts": 1000.0 + s,
             "grad_norm": 1.0, "grad_norms": {"w": 1.0 + s / 100}} for s in steps]


def expected(name: str) -> Any:
    return json.loads(json.dumps(RULES[name]["expected"]))


def served(tmp_path: Path, name: str, **keys: Any) -> Path:
    """One rank's evidence as `build_evidence` writes it, with the contract's keys put in."""
    found = ev.build_evidence(plain_rows(name, range(20)))
    found.update(keys)
    return ev.write_evidence(found, tmp_path / f"{name}.rank0.evidence.json")


OTHER_KEYS = section(AGENT, "Other keys the tools give")


@pytest.mark.parametrize("key", [
    "checkpoint_tensors", "checkpoint_tensors_by_rank", "missing_tensors", "shape_differs", "scores", "score_note",
    "display", "since_start", "reportable_step", "when", "starting_state", "lr", "lr_before", "gradient_trend",
    "attention_trends", "update_trends", "weight_jumps", "layers_not_read", "modules_not_judged", "coverage",
    "freshness", "judged_to_step", "records_to_step", "capture_stopped.step",
])
def test_the_agent_page_names_each_key_the_audit_found_missing(key):
    spans = re.findall(r"`([^`]+)`", OTHER_KEYS) + re.findall(r"\*\*([a-z_]+)\*\*", OTHER_KEYS)
    assert any(span == key or span.endswith(f".{key}") for span in spans), key


def test_the_new_section_keeps_the_rules_the_page_already_keeps():
    """`test_mcp_server.py` reads every backticked word that starts as a verb (`update_...`) as a tool
    name, and allows a kind's name only on a line that names its class: so `update_trends` is in bold,
    and the section names the kinds in words, as the rest of the page does."""
    assert "`update_" not in AGENT
    named = {kind for line in AGENT.splitlines() for kind in ev.INCIDENT_CLASSES if f"`{kind}`" in line}
    assert not named & {"resume_incomplete", "moe_imbalance", "update_spike"}


def test_checkpoint_tensors_and_the_two_findings_are_where_the_page_says(tmp_path):
    rows = [dict(r) for r in CHECKPOINT["a_resume_with_two_tensors_verified_missing"]["records"]]
    capture = tmp_path / "r1.rank0.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in rows))
    listed = call(capture, "list_incidents")
    block = listed["checkpoint_tensors"]
    assert {"resumes", "entries"} <= set(block)
    entry = block["entries"][0]
    assert {"step", "rank", "attempt", "state", "entry", "by", "model", "trainable", "checkpoint", "matched"} <= set(entry)
    assert entry["state"] == "compared" and set(entry["missing"]) == {"count", "names"}
    [resume] = [i for i in listed["incidents"] if i["kind"] == "resume_incomplete"]
    one = call(capture, "get_incident", {"id": resume["id"]})
    assert one["checkpoint_tensors"] == block
    [finding] = one["incident"]["changes"][0]["findings"]
    assert finding == {"component": "model", "finding": "missing_tensors", "count": 2,
                       "tensors": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}
    assert "`changes[].findings` with `component` `model`, a `count` and the `tensors` named" in flat(OTHER_KEYS)


def test_scores_since_start_and_the_learning_rate_are_where_the_page_says(tmp_path):
    grad = expected("score_on_grad_spike_by_the_total")["incident"]
    spike = expected("update_spike_carries_the_learning_rate")["incident"]
    collapsed = expected("start_ten_samples_is_reported")["incident"]
    path = served(tmp_path, "r1", incidents=[grad, spike, collapsed])
    scores = call(path, "get_incident", {"id": "grad_spike@60"})["scores"]
    assert {"at", "ratio", "score", "display"} <= set(scores[0]) and scores[0]["display"].startswith("score ")
    on_update = call(path, "get_incident", {"id": f"update_spike@{spike['step']}"})["incident"]
    assert len(on_update["lr"]) == len(on_update["lr_before"]) == 2  # one a parameter group
    [brief] = [i for i in call(path, "list_incidents")["incidents"] if i["kind"] == "moe_imbalance"]
    assert brief["when"] == f"collapsed since step {brief['step']}, reportable at step {brief['reportable_step']}"
    page = flat(OTHER_KEYS)
    assert '"collapsed since step S, reportable at step R"' in page
    assert "A null score is no score, never 0" in page


def store_folder(root: Path, run: str, reports: dict[int, dict[str, Any]], judged: bool = True) -> Path:
    """A run folder in a store with a report for each rank and its status (as lane AD5's tests write
    one). Not `judged`: each rank has a stored part newer than its report, so the report is not current."""
    import time

    folder = root / "runs" / run
    for rank, extra in reports.items():
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-00000000-00000009.parquet").write_bytes(b"")
        if not judged:
            (parts / "part-00000010-00000019.parquet").write_bytes(b"")
        (parts.parent / "incidents.json").write_text(json.dumps({
            "schema_version": 1, "source": "tensorquill-server", "run_id": run, "rank": rank, "attempt": 1,
            "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
            "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, **extra}))
    if True:
        (folder / "watchdog.json").write_text(json.dumps({
            "schema_version": 1, "run_id": run, "summary": "active", "server_running": True,
            "written_at_ms": int(time.time() * 1000), "world_size": len(reports), "missing_ranks": [],
            "incidents": [], "pauses": {}, "recommended_silent_seconds": None,
            "ranks": [{"rank": r, "attempt": 1, "last_step": 9, "phase": "train", "state": "active",
                       "severity": None, "stream": "open"} for r in reports]}))
    return folder


def test_the_measurements_reach_one_ranks_answers_and_a_run_folders_by_rank(tmp_path):
    blocks = {key: expected(name)[key] for key, name in (
        ("gradient_trend", "gradient_trend_a_steady_rise"), ("weight_jumps", "weight_jumps_one_tensor_jumps"))}
    listed = call(served(tmp_path, "r1", **blocks), "list_incidents")
    assert all(listed[key] == block for key, block in blocks.items())
    # A run folder (wave AF, EV-04): each judged rank's block, with its rank, under `<block>_by_rank`.
    source = {"r": {"form": "scores", "point": "requested"}}
    folder = store_folder(tmp_path / "store", "run", {0: {**blocks, "moe_source": source}, 1: {}})
    for answer in (call(folder, "get_all_ranks_status"), call(folder, "list_incidents")):
        assert answer["gradient_trend_by_rank"] == [{"rank": 0, "gradient_trend": blocks["gradient_trend"]}]
        assert answer["weight_jumps_by_rank"] == [{"rank": 0, "weight_jumps": blocks["weight_jumps"]}]
        assert answer["moe_source_by_rank"] == [{"rank": 0, "moe_source": source}]
        assert not {"gradient_trend", "weight_jumps", "moe_source"} & set(answer)  # never added up
    one = call(folder, "list_incidents", {"rank": 1})
    assert not {"gradient_trend_by_rank", "moe_source_by_rank"} & set(one)  # that rank's entries only
    # A rank that is not judged is in none of them.
    unjudged = store_folder(tmp_path / "unjudged", "run", {0: {**blocks}}, judged=False)
    assert "gradient_trend_by_rank" not in call(unjudged, "get_all_ranks_status")
    page = flat(OTHER_KEYS)
    assert "`gradient_trend_by_rank`, `attention_trends_by_rank`, **update_trends_by_rank**, `weight_jumps_by_rank` and `moe_source_by_rank`" in page
    assert "A rank that is not judged is in none of them." in page
    assert "carry neither the measurements" not in AGENT  # the sentence of 0a17420, true before wave AF


def test_modules_not_read_reach_a_tool_as_the_page_says(tmp_path):
    some = expected("layers_not_read_some_modules")
    found = ev.build_evidence(plain_rows("r1", range(20)))
    found["not_checked"] = some["not_checked"]
    for kind, entry in some["kind_checks"].items():
        found["kind_checks"][kind] = {**found["kind_checks"][kind], **entry}
    listed = call(ev.write_evidence(found, tmp_path / "r1.rank0.evidence.json"), "list_incidents")
    [layers] = [e for e in listed["not_checked"] if e.get("check") == "layers"]
    assert {"modules", "reason", "coverage"} <= set(layers) and layers["coverage"] == "partial"
    assert listed["kind_checks"]["activation_drift"] == {"modules_not_judged": ["blk1"]}
    assert "layers_not_read" not in listed  # the capture's key; a tool shows it as above
    assert "a `not_checked` entry with `check` `layers`" in flat(OTHER_KEYS)


def test_freshness_and_capture_stopped_are_described_as_the_tools_describe_them():
    tools = {tool["name"]: flat(tool["description"]) for tool in mcp_server.TOOLS}
    assert "`freshness`, on a local run's row only" in tools["list_runs"]
    assert "`step` 0 is step 0; `step` null means it committed none" in tools["get_all_ranks_status"]
    page = flat(OTHER_KEYS)
    assert "`step` 0 is step 0. `step` null means it committed none." in page
    for why in ("records_newer", "nothing_judged", "thresholds_changed"):
        assert f"`{why}`" in page


def test_the_setup_paragraph_lists_what_the_tools_pass_on():
    setup = flat(section(AGENT, "Setup"))
    for key in ("gradient_trend", "moe_source", "modules_not_judged", "starting_state"):
        assert f"`{key}`" in setup, key
    for key in mcp_server.KIND_CHECK_KEYS:
        assert f"`{key.removesuffix('_by_rank')}`" in setup or key.endswith("_by_rank"), key


# --- 2. The stale scripts ----------------------------------------------------------------------------


def full_capture(path: Path, steps: int) -> Path:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    session = tensorquill.init(model, run_id="drift", capture="full", sink=JsonlSink(path), heartbeat_seconds=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = model(torch.randn(16, 8)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    session.close()
    return path


def test_drift_measure_runs_its_published_command_and_names_the_rule_of_its_table(tmp_path):
    capture = full_capture(tmp_path / "drift.rank0.jsonl", ev.ACT_DRIFT_NEEDS + 20)
    script = REPO / "experiments" / "nanomoe" / "drift_measure.py"
    assert "drift_measure.py balance-on=<capture> balance-off=<capture>" in (script.parent / "README.md").read_text()
    done = subprocess.run([sys.executable, str(script), f"balance-on={capture}", f"balance-off={capture}"],
                          cwd=REPO, env=child_env(), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert out.count("detector today:") == 2 and out.count("the rule before wave AD") == 2
    assert "balance-on: " in out and "balance-off: " in out and "factor 3.0: first step" in out
    doc = script.read_text().split('"""')[1]
    assert "before wave AD" in doc and "Same arithmetic as evidence.detect_activation_anomalies" not in doc


def test_the_rule_selection_scripts_hold_no_path_of_one_machine():
    folder = REPO / "experiments" / "rule-selection"
    for name in ("inputs.py", "measure_thresholds.py", "THRESHOLDS.md"):
        assert "/Users/" not in (folder / name).read_text(), name


def run_inputs(env: dict[str, str]) -> list[str]:
    folder = REPO / "experiments" / "rule-selection"
    done = subprocess.run([sys.executable, "-c", "import inputs; print(inputs.MAIN); print(inputs.WORKTREE)"],
                          cwd=folder, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def test_the_main_checkout_is_named_by_the_variable_or_found_by_git(tmp_path):
    env = {k: v for k, v in child_env().items() if k != "TENSORQUILL_MAIN_CHECKOUT"}
    main, worktree = run_inputs(env)
    assert Path(worktree) == REPO
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    assert Path(main) == (REPO / common).resolve().parent  # the main working tree, also from a worktree
    named, _ = run_inputs({**env, "TENSORQUILL_MAIN_CHECKOUT": str(tmp_path)})
    assert Path(named) == tmp_path.resolve()


# --- 3. Two stated limits ---------------------------------------------------------------------------


class Unfrozen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(8, 4)
        self.head = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x) + self.router(x).sum(-1, keepdim=True)


def unfrozen_run(capture: str | None, by_hand: bool, **options: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """`head` frozen at `init` and unfrozen at step 2, as a fine-tuning schedule does."""
    torch.manual_seed(0)
    model = Unfrozen()
    model.head.requires_grad_(False)
    sink = MemorySink()
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id="u", sink=sink, heartbeat_seconds=None,
                                   routers={"router": Router(4, top_k=1)},
                                   **({"capture": capture} if capture is not None else {}), **options)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for step in range(4):
            if step == 2:
                model.head.requires_grad_(True)
            loss = model(torch.randn(4, 8)).sum()
            optimizer.zero_grad()
            loss.backward()
            if by_hand:
                session.record_grads()
            optimizer.step()
        session.close()
    return sink.records, [str(w.message) for w in said]


@pytest.mark.parametrize("by_hand", [False, True])
def test_router_only_capture_never_records_a_parameter_unfrozen_after_init(by_hand):
    records, said = unfrozen_run("routers", by_hand)
    assert len(records) == 4
    assert not any(name.startswith("head.") for r in records for name in r.get("grad_norms") or {})
    assert not any("frozen when the session started" in text for text in said)
    if by_hand:  # the call records the parameters trainable at `init`, and only those
        assert all(set(r["grad_norms"]) == {"router.weight", "router.bias"} for r in records)


@pytest.mark.parametrize(("capture", "options"), [("full", {}), (None, {"grads": True})])
def test_full_capture_or_grads_records_it_from_the_step_it_is_unfrozen_and_says_so(capture, options):
    records, said = unfrozen_run(capture, False, **options)
    named = [{n for n in r["grad_norms"] if n.startswith("head.")} for r in records]
    assert named == [set(), set(), {"head.weight", "head.bias"}, {"head.weight", "head.bias"}]
    assert sum("frozen when the session started" in text for text in said) == 1


def test_the_guide_and_the_readme_state_the_unfreeze_limit():
    for text in (flat(GUIDE), flat(README)):
        assert "a parameter frozen at `init` and unfrozen later is never recorded, with no warning, also when" in text
        assert "`routers=...` with `grads=True`" in text
    assert "If your schedule unfreezes layers, use `capture=\"full\"` or add `grads=True`." in flat(
        section(GUIDE, "2. Choose a capture mode"))


SIGTERM_JOB = """
import os, signal, sys, torch, tensorquill
from torch import nn
if sys.argv[1] == "handler":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
model = nn.Linear(4, 4)
session = tensorquill.init(model, run_id=sys.argv[1], capture="full", every=10, heartbeat_seconds=None,
                           sink=tensorquill.JsonlSink(sys.argv[1] + ".rank0.jsonl"))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
with session:
    for _ in range(25):
        loss = model(torch.randn(2, 4)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    os.kill(os.getpid(), signal.SIGTERM)
    for _ in range(1000):
        pass
"""


@pytest.mark.parametrize(("mode", "lines", "end"), [("default", 20, None), ("handler", 25, "raised")])
def test_sigterm_as_the_guide_and_the_readme_describe_it(tmp_path, mode, lines, end):
    """Lane AF-B's sentence, entered because this tree already does it: no handler is installed, so a
    default SIGTERM loses the steps not yet written and leaves no marker; a handler that raises keeps them."""
    (tmp_path / "job.py").write_text(SIGTERM_JOB)
    done = subprocess.run([sys.executable, "job.py", mode], cwd=tmp_path, env=child_env(), capture_output=True,
                          text=True, timeout=120)
    assert done.returncode in (-15, 143), done.stderr
    steps = [json.loads(line)["step"] for line in (tmp_path / f"{mode}.rank0.jsonl").read_text().splitlines()]
    assert steps == list(range(len(steps)))  # whole lines, a gapless run from step 0
    # With the default handling the steps since the last flush (every=10) are lost, and any still in
    # the writer's queue: 20 lines when the writer had caught up, as it had in the run by hand.
    assert len(steps) == lines if end is not None else len(steps) <= lines
    marker = tmp_path / f"{mode}.rank0.end.json"
    assert (json.loads(marker.read_text())["outcome"] if marker.exists() else None) == end
    answer = call(tmp_path / f"{mode}.rank0.jsonl", "list_incidents")
    assert answer["run_end"]["state"] == ("not_closed" if end is None else "closed")
    assert "TensorQuill installs no signal handler" in flat(GUIDE) and "TensorQuill installs no signal handler" in flat(README)


@pytest.mark.parametrize(("env", "scoped"), [
    ({"TORCHELASTIC_RESTART_COUNT": "1", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "2"}, 1),
    ({"TORCHELASTIC_RESTART_COUNT": "1", "GROUP_WORLD_SIZE": "2", "WORLD_SIZE": "2"}, None),
    ({"TORCHELASTIC_RESTART_COUNT": "0", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "2"}, None),
])
def test_restart_keys_are_scoped_under_one_agent_only(monkeypatch, env, scoped):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert tracker._generation(2) == scoped


def test_the_guide_and_the_readme_say_several_agents_are_not_supported_in_v1():
    guide = flat(GUIDE)
    assert "**Not supported in v1: a restart under a torchrun launch with several agents.**" in guide
    assert guide.index("Not supported in v1: a restart") < guide.index("### Plain PyTorch")
    assert "Restart keys under a torchrun launch with several agents (nodes) are not supported in v1" in flat(README)


def test_the_numbers_the_pages_give_of_torch_are_the_installed_torchs():
    """Read from PyTorch's own code, not from TensorQuill's: torchrun's agent stops workers with SIGTERM
    and a SIGKILL 30 s later (a close can wait 60 s), and the process group's default timeouts."""
    import inspect

    import torch.distributed as dist
    from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent

    shutdown = inspect.signature(LocalElasticAgent._shutdown).parameters
    assert shutdown["death_sig"].default.name == "SIGTERM" and shutdown["timeout"].default == 30
    assert "Default value is 10 minutes for NCCL and 30 minutes for other backends" in flat(
        dist.init_process_group.__doc__)
    from tensorquill import capture
    assert capture.CLOSE_SECONDS == 60.0
    for text in (flat(GUIDE), flat(README)):
        assert "30 seconds after its SIGTERM" in text and "10 minutes on NCCL" in text


# --- 4. The replica collectives ---------------------------------------------------------------------


def test_the_guide_says_the_explicit_replica_calls_block_and_the_scheduled_check_does_not():
    gpus = flat(section(GUIDE, "4. Add TensorQuill to the training loop").split("### Several GPUs")[1])
    assert "It is not a collective and no rank waits for another." in gpus
    assert "**The two checks you call yourself are collectives.**" in gpus
    assert "each call waits for the others as any collective does" in gpus
    assert "a gather that fails once issued raises into your code" in gpus


def test_the_readme_says_it_once_where_a_reader_looks_for_what_can_wait_or_raise():
    things = flat(section(README, "Things to know"))
    assert things.index("**What can wait or raise in training.**") < things.index("A `Trainer.train()` that raises")
    assert "each is a collective. Every rank of the group must call it on the same step" in things


# --- 5. Wave AF/AG: the sentences the lanes left, entered once the merged code does what they say ------

from test_flight_e2e import BINARY  # noqa: E402

from tensorquill import compare as compare_module  # noqa: E402
from tensorquill import divergence  # noqa: E402
from tensorquill import flight  # noqa: E402
from tensorquill import identity  # noqa: E402
from tensorquill import runs as runs_module  # noqa: E402

AG_CASES = json.loads((FIXTURES / "wave_af" / "cases_ag.json").read_text())["cases"]
needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")
ENGINES = ["python", pytest.param("rust", marks=needs_server)]
NEW = flat(section(README, "Records that cannot be placed, every rank's keys, and routers with no tokens"))


def identity_file(path: Path, **bad: Any) -> Path:
    """Five valid records, the fourth (line 4) with `bad` put in."""
    rows = [{"schema_version": 1, "run_id": "m", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0}
            for s in range(5)]
    rows[3].update(bad)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def server_env() -> dict[str, str]:
    return {**child_env(), **({"TENSORQUILL_SERVER_BIN": str(BINARY)} if BINARY else {})}


@pytest.mark.parametrize("bad", [{"attempt": 1.0}, {"rank": True}, {"step": -3}, {"schema_version": 99},
                                 {"attempt": None}, {"step": "1"}])
@pytest.mark.parametrize("engine", ENGINES)
def test_a_record_that_cannot_be_placed_refuses_the_input_in_every_offline_reader(tmp_path, monkeypatch, bad, engine):
    if BINARY:
        monkeypatch.setenv("TENSORQUILL_SERVER_BIN", str(BINARY))
    capture = identity_file(tmp_path / "m.rank0.jsonl", **bad)
    [(field, value)] = bad.items()
    with pytest.raises(identity.MalformedRecordError) as raised:
        ev.build_evidence(ev.load_records(capture), detectors=engine)
    error = raised.value
    assert isinstance(error, ValueError) and (error.field, error.value) == (field, value)
    assert str(error).startswith(f"{field} is ") and f"{capture}, line 4" in str(error) and error.reason
    done = subprocess.run([sys.executable, "-m", "tensorquill", str(capture), "--detectors", engine], cwd=tmp_path,
                          env=server_env(), capture_output=True, text=True, timeout=120)
    assert done.returncode == 2 and done.stdout == ""
    assert done.stderr.splitlines() == [f"tensorquill: error: {error}"]  # one line, no traceback
    assert not list(tmp_path.glob("*.evidence.json"))  # no evidence written


def test_the_readme_s_example_line_is_the_one_the_command_prints(tmp_path):
    capture = identity_file(tmp_path / "run.rank0.jsonl", attempt=1.0)
    done = subprocess.run([sys.executable, "-m", "tensorquill", capture.name, "--detectors", "python"],
                          cwd=tmp_path, env=child_env(), capture_output=True, text=True, timeout=120)
    said = done.stderr.strip()
    assert said == "tensorquill: error: attempt is 1.0 at run.rank0.jsonl, line 4: tensorquill reads an integer of 0 or more"
    assert f"`{said}`" in NEW
    assert said.removeprefix("tensorquill: error: ").replace("run.rank0.jsonl", "<file>") in flat(GUIDE)
    assert f"`{said.removeprefix('tensorquill: error: ')}`" in flat(AGENT)


def test_an_mcp_tool_answers_the_error_and_the_next_call_and_the_listing_does_not_raise(tmp_path):
    capture = identity_file(tmp_path / "m.rank0.jsonl", step=-3)
    reply = mcp_server.call_tool(mcp_server.Source(capture), 1, {"name": "list_incidents", "arguments": {}})
    assert reply["result"]["isError"] is True
    assert reply["result"]["content"][0]["text"].startswith("step is -3 at ")
    identity_file(capture)  # put right: the next call is answered
    assert call(capture, "list_incidents")["incidents"] == []
    identity_file(capture, step=-3)
    [row] = runs_module.list_runs(tmp_path)
    assert row["run_state"] is None and row["evidence"]["state"] == "missing"


@needs_server
@pytest.mark.parametrize(("column", "value", "said"), [
    ("step", -3, "row 3: step is -3, not an integer of 0 or more; the input is refused"),
    ("rank", -3, "row 3: rank is -3, not an integer of 0 or more; the input is refused"),
    # Version 2 is protocol 5's row (wave AL, section 3), so 3 is the version refused since then.
    ("schema_version", 3, "row 3: schema_version is 3; this server reads versions 0, 1 and 2; the input is refused"),
    (None, None, None),
])
def test_offline_detect_refuses_a_negative_key_and_carries_attempt_order(tmp_path, column, value, said):
    import pyarrow as pa

    rows = [{"schema_version": 1, "run_id": "m", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0} for s in range(5)]
    table = pa.Table.from_batches([flight.records_to_batch(rows)])
    if column:
        index = table.schema.get_field_index(column)
        values = table.column(column).to_pylist()
        values[2] = value
        table = table.set_column(index, table.schema.field(column), pa.array(values, table.schema.field(column).type))
    stream = tmp_path / "rows.arrow"
    with pa.OSFile(str(stream), "wb") as out, pa.ipc.new_stream(out, table.schema) as writer:
        writer.write_table(table)
    done = subprocess.run([str(BINARY), "detect", str(stream)], capture_output=True, text=True, timeout=120)
    if column is None:
        assert done.returncode == 0 and json.loads(done.stdout)["attempt_order"] == "first_seen"
        assert "carry `attempt_order` (`first_seen`), as a live report does" in NEW
    else:
        assert done.returncode == 1 and done.stderr.strip() == f"tensorquill-server: {said}"
        assert "exits with status 1 and one line naming the row, the field and the value" in NEW


def test_the_capture_writes_a_negative_step_as_given_and_the_file_is_then_refused(tmp_path):
    path = tmp_path / "n.rank0.jsonl"
    # Wave AO (BUG-05, lane AO-1, accepted Nov 26): `init` refuses a negative `start_step` before
    # anything is opened; `session.step(step=...)` below 0 is still written as given.
    with pytest.raises(ValueError, match="start_step must be an integer of 0 or more, not -2"):
        tensorquill.init(nn.Linear(2, 2), run_id="n", sink=JsonlSink(path), heartbeat_seconds=None,
                         auto_step=False, start_step=-2)
    assert path.read_text() == ""  # the sink, built before `init`, made the file: nothing is written to it
    session = tensorquill.init(nn.Linear(2, 2), run_id="n", sink=JsonlSink(path), heartbeat_seconds=None,
                               auto_step=False, start_step=0)
    session.step()
    session.step(step=-5)
    session.close()
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [0, -5]
    with pytest.raises(identity.MalformedRecordError):
        ev.load_records(path)
    for text in (NEW, flat(GUIDE)):
        # Wave AO (BUG-05): `init` refuses a negative `start_step`, so the page names `session.step(step=...)`.
        assert "writes a `session.step(step=...)` below 0 as given" in text


def test_the_sender_skips_and_counts_what_it_does_not_send_and_sends_an_unread_version():
    good = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0}
    records = [good, {**good, "step": 1, "attempt": -3}, {**good, "step": 2, "rank": -1}, {**good, "step": 3, "attempt": None},
               {**good, "step": 4, "schema_version": 99}, {**good, "step": 5}]
    refused: list = []
    _, steps, skipped = flight.rows_to_send(records, flight.flight_schema(), refused=refused)
    assert steps == [0, 4, 5] and skipped == 3  # the unread schema_version 99 is sent, for the server to count
    assert [(field, value) for _, field, value in refused] == [("attempt", -3), ("rank", -1), ("attempt", None)]
    assert flight.SENDER_FIELDS == ("rank", "attempt", "step")
    assert "counts it in `sender.skipped_lines`, says the first such line once on stderr" in NEW
    assert "counts it under `unsupported_schema_version` (`dropped_row_keys` in the status)" in NEW


def test_each_ranks_not_checked_and_kind_entries_reach_the_all_ranks_view_read_from_local_files():
    def rank(entries: list, modules: list, starting: list) -> dict:
        found = ev.build_evidence(plain_rows("r", range(20)))
        found["not_checked"] = entries
        found["kind_checks"]["activation_drift"] = {**found["kind_checks"]["activation_drift"], "modules_not_judged": modules}
        found["kind_checks"]["moe_imbalance"] = {**found["kind_checks"]["moe_imbalance"], "starting_state": starting}
        return found
    entry = expected("layers_not_read_some_modules")["not_checked"][0]
    view = ev.combine_rank_evidence({0: rank([entry], ["blk1"], [{"router": "r", "samples": 5}]),
                                     1: rank([], [], [{"router": "r", "samples": 3}])})
    assert view["not_checked_by_rank"] == [{"rank": 0, "not_checked": [entry]}]
    drift = view["kind_checks"]["activation_drift"]
    assert drift["modules_not_judged"] == ["blk1"] and drift["modules_not_judged_by_rank"] == {"0": ["blk1"]}
    moe = view["kind_checks"]["moe_imbalance"]
    assert moe["starting_state"] == [{"router": "r", "samples": 3}] and set(moe["starting_state_by_rank"]) == {"0", "1"}
    assert "under `not_checked_by_rank` (`{\"rank\", \"not_checked\"}`) in a view read from local files" in NEW
    assert "`modules_not_judged_by_rank` and `starting_state_by_rank`" in NEW and "`starting_state_by_rank`**" in AGENT


def test_two_runs_observed_differently_say_so_under_provenance_differs():
    def run(name: str, source: dict) -> list[dict]:
        rows = plain_rows(name, range(40))
        rows[0]["moe_source"] = source
        return rows
    found = divergence.first_divergence(run("a", {"r": {"form": "scores", "point": "requested"}}),
                                        run("b", {"r": {"form": "map", "point": "dispatched"}}))
    assert set(found["provenance_differs"]) == {"moe_source"}
    assert set(found["provenance_differs"]["moe_source"]) == {"a", "b"}
    assert "moe_source" not in (found.get("left_out") or {}).get("not_compared", [])
    assert "The runs differ in how the routers' counts were read" in divergence.divergence_summary(found)
    assert "`provenance_differs`" in NEW and "`provenance_differs`**" in AGENT


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(AG_CASES))
def test_a_router_recorded_with_no_tokens_is_named_as_the_pages_say(monkeypatch, engine, name):
    if BINARY:
        monkeypatch.setenv("TENSORQUILL_SERVER_BIN", str(BINARY))
    given = AG_CASES[name]["given"]
    found = ev.build_evidence(given["records"] if isinstance(given, dict) else given, detectors=engine)
    entries = [e for e in found.get("not_checked", []) if e.get("check") == "moe_routing"]
    values = {kind: found["kind_checks"][kind]["value"] for kind in ("moe_imbalance", "starved_experts")}
    if name == "every_router_recorded_with_no_tokens":
        assert values == {"moe_imbalance": "unavailable", "starved_experts": "unavailable"}
        assert found["run_state"]["state"] == "degraded"
    else:
        assert values == {"moe_imbalance": "checked", "starved_experts": "checked"}
    if name == "routers_that_were_read":
        assert entries == []
    else:
        [entry] = entries
        assert entry["kinds"] == ["moe_imbalance", "starved_experts"] and entry["routers"] == sorted(entry["routers"])
    assert "`{\"check\": \"moe_routing\", \"kinds\": [\"moe_imbalance\", \"starved_experts\"], \"routers\": [...], \"reason\": ...}`" in NEW
    assert "a `not_checked` entry `moe_routing` names them" in flat(GUIDE)


def test_a_rewrite_in_place_with_a_later_time_is_read_again_and_one_that_grows_is_not(tmp_path):
    path = tmp_path / "w.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in plain_rows("w", range(200))))
    tail = ev._Tail(path, None)
    assert tail.poll() and len(tail.records) == 200
    text = path.read_text()
    middle = len(text) // 2
    at = text.index('"grad_norm": 1.0', middle)
    rewritten = text[:at] + '"grad_norm": 9.0' + text[at + len('"grad_norm": 1.0'):]
    before = os.stat(path)
    path.write_text(rewritten)  # same length, same first and last 4 KiB
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert tail.poll() and tail.restarts == 1
    assert sum(r["grad_norm"] == 9.0 for r in tail.records) == 1
    # A rewrite that also grows the file, with the same ends where it was read up to: still read as an append.
    grown = rewritten.replace('"grad_norm": 9.0', '"grad_norm": 7.0', 1) + json.dumps(plain_rows("w", range(200, 201))[0]) + "\n"
    path.write_text(grown)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    tail.poll()
    assert tail.restarts == 1 and not any(r["grad_norm"] == 7.0 for r in tail.records)
    readme = flat(README)
    assert "or whose modification time (`st_mtime_ns`) changed while it did not grow since the last read" in readme
    assert "both of those 4 KiB stretches and the modification time, or that also grows the file" in readme


def test_the_probe_s_overflow_positions_are_the_ones_the_readme_names():
    from tensorquill import probe

    def call_of(finite_logits: bool, gap_sq: float) -> dict:
        one = torch.tensor([1.0], dtype=torch.float64)
        return {"vocab": 4, "norm": one, "mean": one, "var": one, "log_z": one, "gap_sq": torch.tensor([gap_sq], dtype=torch.float64),
                "logits_finite": torch.tensor([finite_logits]), "input_dtype": "torch.float32", "weight_dtype": "torch.float32"}
    found = probe._head_numbers([call_of(True, float("inf"))])
    assert found["overflow_positions"] == 1 and found["nonfinite_positions"] == 0
    assert found["not_read"].startswith("every position's FP32 logits are finite but the BF16 pass overflows")
    assert "Positions whose FP32 logits are finite but whose BF16 pass is not" in flat(README)
    assert "whose finite logits overflow the sums" not in README


def checkpoint_tensors(model: nn.Module, state: dict) -> dict:
    sink = MemorySink()
    session = tensorquill.init(model, run_id="k", sink=sink, capture="full", heartbeat_seconds=None, start_step=10)
    session.checkpoint({"model": state}, event="load", model_state=state)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.randn(2, 4)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    session.close()
    return next(r["checkpoint"]["load"]["tensors"] for r in sink.records if "load" in (r.get("checkpoint") or {}))


class SharedChild(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(4, 4)
        self.a = shared
        self.b = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class WithBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("b", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) + self.b


def test_lane_a_s_three_checkpoint_cases_read_as_the_readme_says():
    shared = SharedChild()
    found = checkpoint_tensors(shared, dict(shared.state_dict()))
    assert found["state"] == "compared" and found["model"] == 4  # a.* and b.*: held twice, named twice
    plain = nn.Linear(4, 4)
    state = dict(plain.state_dict())
    state["module.weight"] = state["weight"]
    assert checkpoint_tensors(plain, state) | {"closest": None} == {
        "state": "unavailable", "why": "ambiguous", "collides": "checkpoint", "closest": None}
    buffered = WithBuffer()
    del buffered.__dict__["_non_persistent_buffers_set"]
    assert checkpoint_tensors(buffered, {"lin.weight": buffered.lin.weight.detach(), "lin.bias": buffered.lin.bias.detach(),
                                         "b": torch.zeros(1)}) == {"state": "unavailable", "why": "model_not_readable"}
    readme = flat(README)
    assert "so a module held twice by one parent is named twice" in readme
    assert "`unavailable`, `ambiguous`, with `collides` naming the side (`model` or `checkpoint`)" in readme


def manual_loop(order: str) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    sink = MemorySink()
    session = tensorquill.init(model, run_id="a", sink=sink, capture="full", auto_step=False, heartbeat_seconds=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        model(torch.randn(3, 4)).sum().backward()
        if order == "record_grads":
            session.record_grads()
        optimizer.step()
        if order == "step_first":
            session.step()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            session.step()
    session.close()
    return sink.records


def test_with_auto_step_off_the_gradients_are_read_at_session_step():
    """AUD-2 of the correctness audit, run here first."""
    assert all(not r.get("grad_norms") and r.get("grad_norm") is None for r in manual_loop("zero_grad_first"))
    for order in ("step_first", "record_grads"):
        assert all(len(r["grad_norms"]) == 4 and r["grad_norm"] > 0 for r in manual_loop(order)), order
    assert "**Call `session.step()` before `optimizer.zero_grad()`.**" in flat(GUIDE)
    assert "call it before `optimizer.zero_grad()`, or call `session.record_grads()` before clearing" in flat(README)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_checkpoint_loaded_inside_one_attempt_is_recorded_and_judged_by_no_detector(monkeypatch, engine):
    """AUD-3 of the correctness audit, run here first."""
    if BINARY:
        monkeypatch.setenv("TENSORQUILL_SERVER_BIN", str(BINARY))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    sink = MemorySink()
    session = tensorquill.init(model, run_id="c", sink=sink, capture="full", heartbeat_seconds=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(30):
        if step == 20:
            state = {k: v for k, v in model.state_dict().items() if k != "2.weight"}
            session.checkpoint({"model": state}, event="load", model_state=state)
            model.load_state_dict(state, strict=False)
        loss = model(torch.randn(3, 4)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    session.close()
    [load] = [r for r in sink.records if "load" in (r.get("checkpoint") or {})]
    assert load["checkpoint"]["load"]["tensors"]["missing"] == {"count": 1, "names": ["2.weight"]}
    assert len({r["attempt"] for r in sink.records}) == 1
    found = ev.build_evidence(sink.records, detectors=engine)
    assert found["kind_checks"]["resume_incomplete"]["value"] == "checked"
    assert "resume_incomplete" not in [i["kind"] for i in found["incidents"]] and "checkpoint_tensors" not in found
    assert "**Only a load at the start of a later attempt is judged.**" in flat(GUIDE) and "**Only a load at the start of a later attempt is judged.**" in flat(README)


class CheckpointedMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(8, 4)
        self.out = nn.Linear(8, 8)

    def block(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x) + self.router(x).softmax(-1).sum(-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self.block, x, use_reentrant=True)


def routed(mode: str) -> list[list[int]]:
    torch.manual_seed(0)
    model = CheckpointedMoE()
    model.train(mode == "train")
    sink = MemorySink()
    session = tensorquill.init(model, run_id="e", sink=sink, routers={"router": Router(4, top_k=1)}, heartbeat_seconds=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.manual_seed(1)
    for _ in range(4):
        x = torch.randn(6, 8, requires_grad=True)
        loss = model(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    session.close()
    return [r["moe"]["router"] for r in sink.records]


def test_training_in_eval_mode_under_reentrant_checkpointing_is_counted_as_in_train_mode():
    """Lane B's wave AG fix (contract section 8): the counts come from the recomputation inside backward."""
    train, evaluated = routed("train"), routed("eval")
    assert evaluated == train and all(sum(counts) == 6 for counts in evaluated)
    readme = flat(README)
    assert "**Training in eval mode under reentrant activation checkpointing**" in readme
    assert "training with modules kept in eval mode, or without gradients as reentrant activation checkpointing runs its forward, is recorded as before" not in readme


def test_the_trend_memo_is_stated_without_a_claim_of_speed():
    assert "a series with no new point gives the trend found before. The report is the same as before. Its cost was not measured." in NEW
    assert "faster" not in NEW and "speed" not in NEW


# --- 6. The incremental Rust watcher, protocol 4, and the compact file's replica lists -------------

from tensorquill import compact as compact_module  # noqa: E402
from tensorquill import server_detect as sd  # noqa: E402

WATCH_RUST = flat(README[README.index("**Watching through Rust.**"):README.index("**All ranks.** `--rank all --detectors rust`")])
WATCH_GUIDE = flat(GUIDE[GUIDE.index("**Watching while the job trains.**"):GUIDE.index("**All ranks at once.**")])
COMPACT = flat(section(README, "Compact evidence and the trigger"))
needs_protocol_5 = pytest.mark.skipif(BINARY is None or sd.binary_protocol(BINARY) != sd.PROTOCOL,
                                      reason="no tensorquill-server of this SDK's protocol (TENSORQUILL_SERVER_BIN)")


def watched_records(steps, *, attempt: int = 0, router: str = "r0") -> list[dict[str, Any]]:
    return [{"schema_version": 1, "run_id": "w", "rank": 0, "attempt": attempt, "step": s, "ts": 1000.0 + s,
             "grad_norm": 1.0 + 0.001 * (s % 7), "scalars": {"train/loss": 1.0 / (s + 1)},
             "moe": {router: [5 + s % 3, 4, 6, 5]}, "moe_top_k": {router: 1}} for s in steps]


def protocol_4(tmp: Path) -> Path:
    """A stand-in binary of protocol 4, the one before protocol 5's row layout: it answers `protocol`
    and fails anything else."""
    path = tmp / "old" / "tensorquill-server"
    path.parent.mkdir(parents=True)
    path.write_text('#!/bin/sh\nif [ "$1" = protocol ]; then echo 4; exit 0; fi\necho "ran $*" >&2; exit 3\n')
    path.chmod(0o755)
    return path


def run_command(tmp: Path, binary: Path, *args: str) -> subprocess.CompletedProcess:
    path = tmp / "w.rank0.jsonl"
    if not path.exists():
        path.write_text("".join(json.dumps(r) + "\n" for r in watched_records(range(30))))
    env = {**child_env(), sd.BINARY_ENV: str(binary)}
    return subprocess.run([sys.executable, "-m", "tensorquill", str(path), *args, "--interval", "0.05",
                           "-o", str(tmp / "out.json")], cwd=tmp, env=env, capture_output=True, text=True, timeout=120)


def quoted(text: str, start: str) -> str:
    """The backticked message in `text` that starts with `start`."""
    return re.search(r"`(" + re.escape(start) + r"[^`]*)`", text).group(1)


def test_the_protocol_the_pages_give_is_the_sdks():
    assert sd.PROTOCOL == 5
    assert ("both speak protocol 5 since Nov 24, 2026 (4 before), when protocol 5's row layout was added; "
            "protocol 4 came on Nov 21, 2026 with `detect --follow`") in flat(README)
    assert "The SDK and `tensorquill-server` must be built from the same version (protocol 5)." in WATCH_GUIDE
    assert "must be built again after you upgrade the SDK" in flat(README) and "must be built again after you upgrade the SDK" in WATCH_GUIDE


@needs_protocol_5
def test_this_trees_binary_prints_the_protocol_the_pages_give():
    done = subprocess.run([str(BINARY), "protocol"], capture_output=True, text=True, timeout=30)
    assert (done.returncode, done.stdout) == (0, f"{sd.PROTOCOL}\n")


def test_a_watch_with_a_binary_of_another_protocol_prints_the_message_the_readme_quotes(tmp_path):
    old = protocol_4(tmp_path)
    said = quoted(flat(README), "--watch needs the Rust detectors: <binary> speaks protocol 4").replace("<binary>", str(old))
    for args in (("--watch",), ("--watch", "--detectors", "rust")):
        done = run_command(tmp_path, old, *args)
        assert done.returncode == 2 and not (tmp_path / "out.json").exists()
        assert done.stderr.strip().splitlines()[-1] == f"__main__.py: error: {said}"
        assert "ran " not in done.stderr  # stopped before any look


def test_one_read_with_a_binary_of_another_protocol_refuses_rust_and_falls_back_with_auto(tmp_path):
    old = protocol_4(tmp_path)
    mismatch = quoted(flat(README), "--watch needs the Rust detectors: ").split(": ", 1)[1].split(". To watch")[0]
    mismatch = mismatch.replace("<binary>", str(old))
    done = run_command(tmp_path, old, "--detectors", "rust")
    assert done.returncode == 2 and not (tmp_path / "out.json").exists()
    assert done.stderr.strip().splitlines()[-1].endswith(mismatch)
    warning = quoted(flat(README), "tensorquill: warning: using the Python detectors")
    head = warning.replace("<binary>", str(old))
    tail = "`--detectors python` asks for the Python detectors without this warning"
    assert "ends by saying that `--detectors python` asks for the Python detectors without this warning" in flat(README)
    for args in ((), ("--detectors", "auto")):
        (tmp_path / "out.json").unlink(missing_ok=True)
        done = run_command(tmp_path, old, *args)
        assert done.returncode == 0 and (tmp_path / "out.json").exists()
        warned = [line for line in done.stderr.splitlines() if line.startswith("tensorquill: warning: ")]
        assert len(warned) == 1 and warned[0].startswith(head) and warned[0].endswith(tail), warned


@needs_protocol_5
def test_the_follower_judges_an_append_alone_and_every_change_the_pages_name_in_full(monkeypatch):
    monkeypatch.setenv(sd.BINARY_ENV, str(BINARY))
    pytest.importorskip("pyarrow")
    first = watched_records(range(20))
    changes = {
        "late rows": [*first, *watched_records([5])],
        "a resume": [*first, *watched_records([10], attempt=1)],
        "a truncated file": first[:15],
        "a router not seen before": [*first, *watched_records([20], router="r9")],
    }
    follow = sd.FollowedDetect()
    try:
        follow.report(first)
        assert follow.last["mode"] == "full" and follow.last["why"] == "first look"
        follow.report([*first, *watched_records(range(20, 25))])
        assert (follow.last["mode"], follow.last["records_sent"], follow.last["rows_read"]) == ("append", 5, 5)
        follow.report([*first, *watched_records(range(20, 25))], thresholds={"update_window": 20})
        assert follow.last["mode"] == "full" and follow.last["why"] == "the thresholds changed"
        for name, records in changes.items():
            follow.close()
            follow.report(first)
            follow.report(records)
            assert follow.last["mode"] == "full" and follow.last["records_sent"] == len(records), (name, follow.last)
    finally:
        follow.close()
    assert follow._process is None
    for words in ("keeps one `tensorquill-server detect --samples --follow` process for the whole watch",
                  "late rows of an earlier step, a new attempt (a resume), a file rewritten, truncated or replaced",
                  "is judged again at once, in full, with no new record needed",
                  "refuses one that names a router it has not seen"):
        assert words in WATCH_RUST, words
    assert "Each rebuild still converts every record read so far" not in README
    assert "those run over every record read so far at each rebuild made here" not in README
    assert "keeps one `tensorquill-server detect --follow` process for the whole watch" in WATCH_GUIDE


@needs_protocol_5
def test_the_watch_command_s_follow_child_ends_when_the_watcher_is_killed(tmp_path):
    import signal
    import time

    pytest.importorskip("pyarrow")
    path = tmp_path / "w.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in watched_records(range(30))))
    env = {**child_env(), sd.BINARY_ENV: str(BINARY)}
    watcher = subprocess.Popen([sys.executable, "-m", "tensorquill", str(path), "--watch", "--interval", "0.05",
                                "-o", str(tmp_path / "out.json")], cwd=tmp_path, env=env,
                               stdout=subprocess.DEVNULL, stderr=(tmp_path / "watcher.stderr").open("wb"))

    def table() -> list[str]:
        # -ww: whole lines. On Linux CI (981fe6c) no child was found here; the cause is not known (see
        # test_lane_af_w_watch), so a failure below says what the table held.
        return subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,command="], capture_output=True,
                              text=True).stdout.splitlines()

    def followers() -> list[int]:
        return [int(line.split()[0]) for line in table()
                if line.split()[1:2] == [str(watcher.pid)] and "detect --samples --follow" in line]

    def seen() -> str:
        return (f"no evidence or no follow child: watcher pid {watcher.pid}, ended with {watcher.poll()}, evidence "
                f"written {(tmp_path / 'out.json').exists()}; lines naming the server: "
                f"{[line for line in table() if 'tensorquill-server' in line]}; the watcher's standard error: "
                f"{(tmp_path / 'watcher.stderr').read_text(errors='replace')[-600:]!r}")

    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "out.json").exists() or not followers():
            assert time.monotonic() < deadline and watcher.poll() is None, seen()
            time.sleep(0.05)
        child = followers()[0]
        watcher.send_signal(signal.SIGKILL)
        watcher.wait(timeout=30)
        deadline = time.monotonic() + 30
        while subprocess.run(["ps", "-p", str(child)], capture_output=True).returncode == 0:
            assert time.monotonic() < deadline, "the follow child outlived its killed watcher"
            time.sleep(0.05)
    finally:
        if watcher.poll() is None:
            watcher.kill()
            watcher.wait()
    assert "when the watcher is killed with SIGTERM or SIGKILL, because it then reads the end of its input" in WATCH_RUST


def test_the_memory_the_pages_give_is_the_detector_child_s_not_the_watcher_s():
    assert ("the detector child had about 237 MB resident after the first look and between looks, 239 MB after one "
            "more part, and a peak of 386 MB during the first look. These are the detector child's figures, not the "
            "watcher's") in WATCH_RUST
    assert "about 240 MB resident between looks and 386 MB at its peak, the detector process's own memory" in WATCH_GUIDE


def one_rank_replica_run(steps: int) -> dict[str, Any]:
    sys.path.insert(0, str(SDK / "examples"))
    import benchmark_detectors as bd

    template = bd.capture_template(blocks=2, experts=4, top_k=1)
    return tensorquill.build_evidence([bd.synth_record(template, 1, 0, step) for step in range(steps)])


def test_the_compact_file_s_replica_markers_have_the_keys_the_pages_give():
    full = one_rank_replica_run(2_000)
    before = json.dumps(full, sort_keys=True, default=repr)
    small = compact_module.compact_evidence(full)
    assert json.dumps(full, sort_keys=True, default=repr) == before  # the full evidence is unchanged
    steps = small["compact"]["replica_steps"]
    assert list(steps) == ["total", "kept", "omitted", "not_routine", "around", "kept_steps", "observed_steps"]
    assert steps["omitted"] == steps["total"] == len(full["replica"]["steps"]) and small["replica"]["steps"] == []
    gaps = small["compact"]["replica_not_compared"]
    assert list(gaps) == ["listed", "kept", "removed", "not_compared_omitted"]
    full_cov = full["kind_checks"]["replica_drift"]["coverage"]
    small_cov = small["kind_checks"]["replica_drift"]["coverage"]
    assert small_cov["not_compared"] == [full_cov["not_compared"][-1]]
    assert small_cov["not_compared_omitted"] == full_cov["not_compared_omitted"] + gaps["removed"]
    assert ev.replica_line(small) == ev.replica_line(full)
    assert "leaving out" in compact_module.describe(small)
    with pytest.raises(compact_module.ReplicaStepsOmitted):
        compact_module.require_all_replica_steps(small, "this reader")
    assert issubclass(compact_module.ReplicaStepsOmitted, compact_module.SamplesOmitted)
    for page in (COMPACT, flat(AGENT)):
        assert "`total`, `kept`, `omitted`, `not_routine`, `around`, `kept_steps`" in page and "`observed_steps`" in page
        assert "`listed`, `kept`, `removed`" in page
    assert "The marker `compact.replica_steps` gives" in COMPACT and "`compact.replica_not_compared` gives" in flat(AGENT)
