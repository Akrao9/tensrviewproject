"""Wave AO, lane AO-9: BUG-30's workflow and test-collection parts.

1. The workflow's Install steps run `packaging/requirements.py` so that its failure fails the
   step. Each Install block of `.github/workflows/ci.yml` that names `requirements.py` is run
   here as the runner runs it, by bash with the runner's flags, with stand-ins for `python` and
   `pip`: when `requirements.py` fails, the block must fail; when it succeeds, pip must be handed
   what it printed. The workflow is read as text (no YAML package).
2. `test_lane_ai1_compiler_gate.py` can be collected in a checkout whose path is short: its lookup
   of a NeuralDBG clone beside the project indexed the path's parents past the root.

They need bash; each skips where there is none.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SDK = TESTS.parent
WORKFLOW = SDK.parent / ".github" / "workflows" / "ci.yml"
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash here")

# How the runner calls bash for a `run:` step: with no `shell:` (this workflow names none) and
# with `shell: bash` (GitHub's documentation, "Default shell").
RUNNER_FLAGS = {"default": ["-e"], "shell-bash": ["--noprofile", "--norc", "-eo", "pipefail"]}

FAKE_PYTHON = """#!/bin/sh
case "$1" in
  packaging/requirements.py)
    if [ -n "$FAIL_REQUIREMENTS" ]; then echo "requirements.py: no extra named 'x'" >&2; exit 3; fi
    echo "dep-from-requirements==1"; exit 0;;
  *) exit 0;;
esac
"""
FAKE_PIP = """#!/bin/sh
echo "pip $*" >> "$PIP_LOG"
while [ $# -gt 0 ]; do
  if [ "$1" = "-r" ]; then echo "--- $2"; cat "$2"; echo "---"; fi >> "$PIP_LOG"
  shift
done
"""


def install_blocks(text: str) -> list[tuple[int, str]]:
    """Each `- name: Install` step's `run: |` block that names requirements.py, with the line it
    starts on, as the runner hands it to the shell: indentation removed, `${{ ... }}` filled in."""
    lines = text.splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if line.strip() != "- name: Install":
            continue
        run = next(i for i in range(index + 1, len(lines)) if lines[i].strip().startswith("run:"))
        assert lines[run].strip() == "run: |", f"line {run + 1}: an Install step's run is not a block"
        indent = len(lines[run]) - len(lines[run].lstrip())
        body = []
        for following in lines[run + 1:]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        width = min(len(b) - len(b.lstrip()) for b in body if b.strip())
        script = "\n".join(b[width:] for b in body)
        if "requirements.py" in script:
            blocks.append((run + 2, re.sub(r"\$\{\{[^}]*\}\}", "x", script)))
    return blocks


def run_block(script: str, flags: list[str], tmp_path: Path, fail: bool) -> tuple[int, str, Path]:
    bin_dir, work, temp = tmp_path / "bin", tmp_path / "sdk", tmp_path / "runner-temp"
    for folder in (bin_dir, work, temp):
        folder.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("pip", FAKE_PIP)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "pip.log"
    log.touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "RUNNER_TEMP": str(temp), "PIP_LOG": str(log),
           "HOME": str(tmp_path), **({"FAIL_REQUIREMENTS": "1"} if fail else {})}
    (tmp_path / "step.sh").write_text(script + "\n")
    done = subprocess.run([BASH, *flags, str(tmp_path / "step.sh")], cwd=work, env=env, capture_output=True, text=True, timeout=60)
    return done.returncode, log.read_text(), work


def test_the_workflow_has_the_four_install_blocks_that_run_requirements_py():
    starts = [start for start, _ in install_blocks(WORKFLOW.read_text(encoding="utf-8"))]
    assert len(starts) == 4, starts  # the cpu matrix, extras, server and gpu jobs


@needs_bash
@pytest.mark.parametrize("flags", list(RUNNER_FLAGS), ids=list(RUNNER_FLAGS))
@pytest.mark.parametrize("block", range(4), ids=["cpu", "extras", "server", "gpu"])
def test_a_failing_requirements_py_fails_the_install_step(tmp_path, flags, block):
    start, script = install_blocks(WORKFLOW.read_text(encoding="utf-8"))[block]
    code, log, _ = run_block(script, RUNNER_FLAGS[flags], tmp_path, fail=True)
    assert code != 0, f"the Install block at ci.yml line {start} went on after requirements.py failed:\n{log}"
    assert "pip install -r" not in log, log


@needs_bash
@pytest.mark.parametrize("block", range(4), ids=["cpu", "extras", "server", "gpu"])
def test_pip_is_handed_what_requirements_py_printed(tmp_path, block):
    start, script = install_blocks(WORKFLOW.read_text(encoding="utf-8"))[block]
    code, log, work = run_block(script, RUNNER_FLAGS["default"], tmp_path, fail=False)
    assert code == 0, f"ci.yml line {start}:\n{log}"
    handed = re.findall(r"^--- [^\n]*\n(.*?)^---$", log, flags=re.M | re.S)
    assert handed == ["dep-from-requirements==1\n"], log
    assert not any(work.iterdir()), "the requirements were written into the checkout"


@needs_bash
@pytest.mark.parametrize("flags", list(RUNNER_FLAGS), ids=list(RUNNER_FLAGS))
def test_the_construction_itself(tmp_path, flags):
    """The two constructions side by side, outside the workflow: pip reading a process
    substitution goes on after the command in it failed; a file written first does not."""
    substituted = 'pip install -r <(python packaging/requirements.py dev)'
    written = ('python packaging/requirements.py dev > "$RUNNER_TEMP/requirements.txt"\n'
               'pip install -r "$RUNNER_TEMP/requirements.txt"')
    for name in ("substituted", "written", "written-ok"):
        (tmp_path / name).mkdir()
    code, log, _ = run_block(substituted, RUNNER_FLAGS[flags], tmp_path / "substituted", fail=True)
    assert code == 0 and "pip install -r /dev/fd/" in log  # the failure is lost, and pip read nothing
    assert re.search(r"^--- /dev/fd/\d+\n---$", log, flags=re.M), log
    code, log, _ = run_block(written, RUNNER_FLAGS[flags], tmp_path / "written", fail=True)
    assert code != 0 and log == ""
    code, log, _ = run_block(written, RUNNER_FLAGS[flags], tmp_path / "written-ok", fail=False)
    assert code == 0 and "dep-from-requirements==1" in log


# --- 2. Collecting the compiler gate's tests in a short path ----------------------------------


def test_the_compiler_gate_tests_import_in_a_checkout_whose_path_is_short():
    """The test file and the example it imports, copied to a folder right under /tmp (the
    temporary folders pytest makes are too deep for it), imported in a fresh interpreter."""
    root = Path("/tmp")
    if not root.is_dir() or not os.access(root, os.W_OK):
        pytest.skip("no writable /tmp here")
    folder = Path(tempfile.mkdtemp(dir=root))
    try:
        (folder / "sdk" / "tests").mkdir(parents=True)
        (folder / "sdk" / "examples").mkdir()
        copied = folder / "sdk" / "tests" / "test_lane_ai1_compiler_gate.py"
        shutil.copy(TESTS / copied.name, copied)
        shutil.copy(SDK / "examples" / "granite_compiler_gate.py", folder / "sdk" / "examples")
        depth = len(copied.resolve().parents)
        if depth >= 7:
            pytest.skip(f"{folder} is {depth} folders deep: too deep to show the short path")
        import tensorquill
        src = str(Path(tensorquill.__file__).resolve().parents[1])
        code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('gate_tests', sys.argv[1]);"
                " module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module);"
                " print('NEURALDBG', module.NEURALDBG)")
        done = subprocess.run([sys.executable, "-c", code, str(copied)], capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PYTHONPATH": src}, cwd=folder)
        assert done.returncode == 0, done.stderr[-3000:]
        assert "NEURALDBG None" in done.stdout
    finally:
        shutil.rmtree(folder, ignore_errors=True)
