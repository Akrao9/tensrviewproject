"""Lane AK-3 of wave AK: two documentation changes, each checked against the code.

1. Dual-MCP correlation (the plan's Phase 1 item 2; the architecture page, section 6). The agent
   instructions tell a coding agent that also has the team's W&B or MLflow MCP server how to go from a
   TensorQuill incident to the tracker's run: by the run ID, and only where the run ID came from that
   tracker. The page names the keys a capture really has (read here from a capture `tensorquill.init`
   writes on the CPU), and the source words it lists are exactly the words the code can write (read
   from `tracker.py` itself, so a later change to either side fails here).
2. Rerun's usage analytics (the plan's Phase 7 item 4 and its risk table). The README says, beside the
   no-telemetry statement and in the section that installs the `rerun` extra, that Rerun's own viewer
   and command line send anonymous usage analytics unless `rerun analytics disable` is run, for the
   Rerun version the extra pins.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import compact as compact_mod
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import tracker

SDK = Path(__file__).resolve().parents[1]
AGENT = (SDK / "AGENT_INSTRUCTIONS.md").read_text(encoding="utf-8")
README = (SDK / "README.md").read_text(encoding="utf-8")
PYPROJECT = (SDK / "pyproject.toml").read_text(encoding="utf-8")

DUAL_MCP = "With a W&B or MLflow MCP server beside TensorQuill"


def flat(text: str) -> str:
    return " ".join(text.split())


def section(text: str, title: str) -> str:
    """One `## ` section of a page, to its next `## ` heading."""
    start = text.index(f"## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


# --- 1. The source words the code can write -----------------------------------------------------


def written_source_words() -> set[str]:
    """Every source word `tracker.py` can return beside a run ID, read from its text: the second item
    of each two-item `return` in `_resolve_here` (the whole order: explicit, a tracker, the launcher,
    generated) and in `tracker_run_id` (the trackers). `RunInfo.source` and the manifest's
    `run_id_source` are this word (`capture.py` passes `run.source` to `build_manifest`)."""
    tree = ast.parse(Path(tracker.__file__).read_text(encoding="utf-8"))
    words: set[str] = set()
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        if function.name not in ("_resolve_here", "tracker_run_id"):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2:
                second = node.value.elts[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    words.add(second.value)
    return words


@pytest.fixture
def no_tracker(monkeypatch):
    """No tracker run, no launcher ID: the order falls through to a generated ID."""
    for name in ("MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID"):
        monkeypatch.delenv(name, raising=False)
    for module in ("mlflow", "wandb"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    return monkeypatch


def fake_wandb(run_id: str) -> types.ModuleType:
    module = types.ModuleType("wandb")
    module.run = types.SimpleNamespace(id=run_id, project="tracker-project", name="tracker-name", group=None)
    return module


def fake_mlflow(run_id: str) -> types.ModuleType:
    module = types.ModuleType("mlflow")
    active = types.SimpleNamespace(info=types.SimpleNamespace(run_id=run_id, run_name="tracker-name"))
    module.active_run = lambda: active
    return module


def resolved_word(case: str, monkeypatch) -> str:
    if case == "explicit":
        return tracker._resolve_here(0, "given-by-the-job")[1]
    if case == "env":
        monkeypatch.setenv("MLFLOW_RUN_ID", "0123456789abcdef0123456789abcdef")
    elif case == "mlflow":
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow("fedcba9876543210fedcba9876543210"))
    elif case == "wandb":
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb("abc123xy"))
    elif case == "launcher":
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job-7")
    return tracker._resolve_here(0)[1]


CASES = ("explicit", "env", "mlflow", "wandb", "launcher", "generated")


def test_each_branch_of_the_order_writes_one_of_the_words_read_from_the_code(no_tracker):
    """Code only (passes before the documentation): the words read from `tracker.py` are the words
    its branches really return, one for each branch."""
    seen = {}
    for case in CASES:  # each branch run alone, with nothing else in the environment
        with pytest.MonkeyPatch.context() as alone:
            for name in ("MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID"):
                alone.delenv(name, raising=False)
            for module in ("mlflow", "wandb"):
                alone.delitem(sys.modules, module, raising=False)
            seen[case] = resolved_word(case, alone)
    assert seen == {case: case for case in CASES}
    assert set(seen.values()) == written_source_words()
    assert len(set(seen.values())) == len(CASES)


def listed_source_words() -> list[str]:
    """The words the dual-MCP section lists, one bullet each: `- \\`word\\`: ...`."""
    return re.findall(r"^- `([a-z_]+)`", section(AGENT, DUAL_MCP), flags=re.M)


def test_the_instructions_list_exactly_the_source_words_the_code_can_write():
    listed = listed_source_words()
    assert len(listed) == len(set(listed)), listed
    assert set(listed) == written_source_words()


# --- 1. The keys a capture really has ------------------------------------------------------------


@pytest.fixture(scope="module")
def capture(tmp_path_factory) -> dict[str, Any]:
    """A tiny CPU run as `tensorquill.init` writes it under a W&B run, with one planted loss spike, and
    what the evidence files and the MCP tools say of it."""
    out = tmp_path_factory.mktemp("ak3")
    patch = pytest.MonkeyPatch()
    try:
        for name in ("MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "TENSORQUILL_NAME"):
            patch.delenv(name, raising=False)
        patch.delitem(sys.modules, "mlflow", raising=False)
        patch.setitem(sys.modules, "wandb", fake_wandb("abc123xy"))
        patch.chdir(out)
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        session = tensorquill.init(model, every=1, project="demo")
        for step in range(80):
            loss = ((model(torch.randn(16, 8)) - torch.randn(16, 1)) ** 2).mean()
            if step == 60:
                loss = loss * 500
            optimizer.zero_grad()
            loss.backward()
            session.log({"loss": loss.detach()})
            optimizer.step()
        session.close()
    finally:
        patch.undo()
    folder = out / "tensorquill-out"
    records = folder / "abc123xy.rank0.jsonl"
    evidence = ev.build_evidence(records)
    path = ev.write_evidence(evidence, folder / "abc123xy.rank0.evidence.json")
    compact = compact_mod.compact_evidence(evidence)

    def call(name: str, **arguments: Any) -> dict[str, Any]:
        reply = mcp_server.call_tool(mcp_server.Source(path), 1, {"name": name, "arguments": arguments})
        assert reply["result"]["isError"] is False, reply
        return json.loads(reply["result"]["content"][0]["text"])

    listed = call("list_incidents")
    first = listed["incidents"][0]["id"]
    return {
        "manifest": json.loads((folder / "abc123xy.run.json").read_text(encoding="utf-8")),
        "evidence": evidence, "compact": compact,
        "list_incidents": listed, "get_incident": call("get_incident", id=first),
        "list_runs": call("list_runs"),
    }


def test_the_capture_has_the_run_id_where_the_instructions_say(capture):
    """Code only (passes before the documentation): where the run ID and its source really are."""
    assert capture["manifest"]["run_id"] == "abc123xy" and capture["manifest"]["run_id_source"] == "wandb"
    for document in (capture["evidence"], capture["compact"]):
        assert document["run_id"] == "abc123xy" and document["run"]["run_id"] == "abc123xy"
    assert capture["list_incidents"]["source"]["run_id"] == "abc123xy"
    assert capture["get_incident"]["source"]["run_id"] == "abc123xy"
    assert capture["list_runs"]["runs"][0]["run_id"] == "abc123xy"
    assert "loss" in capture["get_incident"]["incident"]["scalars"]
    # The gap the page states: the source word is in the run manifest only, in no evidence file and in
    # no tool's answer. If a later change adds it there, this fails and the page must say so.
    for document in (capture["evidence"], capture["compact"], capture["list_incidents"], capture["get_incident"],
                     capture["list_runs"]):
        assert "run_id_source" not in json.dumps(document, default=str)
    # The `source` of a tool's answer is where the evidence was read, and a run's `source` in
    # `list_runs` is where the run is kept: neither is where the run ID came from.
    assert capture["list_incidents"]["source"]["kind"] == "evidence"
    assert capture["list_runs"]["runs"][0]["source"] == "local"
    assert '"TENSORQUILL_RUN_ID": payload["run_id"]' in (SDK / "src" / "tensorquill" / "trigger.py").read_text()


def test_the_instructions_name_the_real_keys_and_files():
    text = section(AGENT, DUAL_MCP)
    for key in ("`run_id_source`", "`<run_id>.run.json`", "`runs/<run_id>/run.json`", "`run.run_id`",
                "`source.run_id`", "`TENSORQUILL_RUN_ID`", "`scalars`", "`list_incidents`", "`get_incident`"):
        assert key in text, key
    # The architecture page's `run.run_id_source` is not what the code writes: the page must not say it.
    assert "run.run_id_source" not in AGENT
    # The scalar mirror is lane AK-1's, documented there.
    assert "mirror_scalars" not in AGENT


def test_the_instructions_give_the_order_and_say_when_the_ids_do_not_match():
    text = flat(section(AGENT, DUAL_MCP))
    order = [text.index(phrase) for phrase in (
        "Find the incident in TensorQuill first", "Read the scalars recorded on the incident's step",
        "Only then, and only if you need more")]
    assert order == sorted(order)
    for phrase in ("exact run ID", "shares no credentials", "makes no call to a tracker",
                   "will not match", "say so", "do not guess", "Only read from the tracker",
                   "not the tracker's step"):
        assert phrase in text, phrase
    # The page's rule on outside services is narrowed for this one case, not contradicted.
    must_not = flat(section(AGENT, "What you must not do"))
    assert "Do not send the run's data, names or numbers to any outside service" in must_not
    assert "the run ID alone" in must_not


# --- 2. Rerun's usage analytics ------------------------------------------------------------------


def pinned_rerun() -> str:
    found = re.search(r'"rerun-sdk==([0-9.]+)"', PYPROJECT)
    assert found, "the rerun extra no longer pins rerun-sdk"
    return found.group(1)


@pytest.mark.parametrize("title", ["No telemetry", "Develop"])
def test_the_readme_gives_the_rerun_opt_out_beside_the_statement_and_the_extra(title):
    text = flat(section(README, title))
    assert "`rerun analytics disable`" in text
    if title == "Develop":  # the section that installs the extra
        assert "sdk[dev,rerun]" in text


def test_the_telemetry_statement_says_what_was_verified_for_the_pinned_rerun():
    text = flat(section(README, "No telemetry"))
    # Read in Rerun's source at the version the extra pins: a new pin needs a new reading.
    assert f"Rerun {pinned_rerun()}" in text
    for phrase in ("TensorQuill itself sends nothing", "`FORCE_RERUN_ANALYTICS`", "`rerun analytics config`",
                   "writing an `.rrd` file", "web viewer"):
        assert phrase in text, phrase
    assert "have their own settings" not in text  # the sentence this one replaces
