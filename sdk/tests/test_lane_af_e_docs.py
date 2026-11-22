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
    return {**os.environ, "PYTHONPATH": str(SDK / "src")}


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


def test_the_measurements_reach_one_ranks_answers_and_not_a_run_folders(tmp_path):
    blocks = {key: expected(name)[key] for key, name in (
        ("gradient_trend", "gradient_trend_a_steady_rise"), ("weight_jumps", "weight_jumps_one_tensor_jumps"))}
    listed = call(served(tmp_path, "r1", **blocks), "list_incidents")
    assert all(listed[key] == block for key, block in blocks.items())
    # A run folder whose ranks' reports hold them: no answer carries them, as the page now says.
    folder = tmp_path / "store" / "runs" / "run"
    for rank in (0, 1):
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-00000000-00000009.parquet").write_bytes(b"")
        (parts.parent / "incidents.json").write_text(json.dumps({
            "schema_version": 1, "source": "tensorquill-server", "run_id": "run", "rank": rank, "attempt": 1,
            "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
            "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, **blocks,
            "moe_source": {"r": {"form": "scores", "point": "requested"}}}))
    for answer in (call(folder, "get_all_ranks_status"), call(folder, "list_incidents"),
                   call(folder, "list_incidents", {"rank": 0})):
        assert not {"gradient_trend", "weight_jumps", "moe_source"} & set(answer)
    assert "A run folder's answers carry neither the measurements nor `moe_source` in this version" in flat(OTHER_KEYS)


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
