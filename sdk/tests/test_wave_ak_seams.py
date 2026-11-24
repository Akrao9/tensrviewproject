"""Wave AK, the merge's seams: three statements the lanes left for the merge, each outside the
lane that made it untrue or incomplete (bug list, section 20).

- Lane AK-2 made a checkpoint folder's load read the `.safetensors` headers; the Accelerate
  adapter's docstring and the agent instructions still described a folder as file names only.
- Lane AK-3 put Rerun's analytics opt-out in the README; `RUN_ON_YOUR_JOB.md` introduces the
  `rerun` extra and suggests the web viewer, which the opt-out does not reach, and said neither.

Documents and a docstring only: no behaviour is tested here. The docstring is read from the
adapter's source, not by importing it: the module imports `accelerate`, which the plain CI jobs do
not install, and a page test should run there too.
"""

from __future__ import annotations

import ast
from pathlib import Path

SDK = Path(__file__).resolve().parents[1]
GUIDE = (SDK / "RUN_ON_YOUR_JOB.md").read_text()
INSTRUCTIONS = (SDK / "AGENT_INSTRUCTIONS.md").read_text()


def flat(text: str) -> str:
    return " ".join(text.split())


def line_with(text: str, start: str) -> str:
    """The one line of a page that starts with `start`."""
    found = [line for line in text.splitlines() if line.startswith(start)]
    assert len(found) == 1, (start, len(found))
    return found[0]


def docstring_of(source: Path, function: str) -> str:
    """A top-level function's docstring, read from the file without importing it."""
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{function} is not a top-level function of {source.name}")


def test_the_accelerate_adapter_says_a_folder_load_reads_the_safetensors_headers():
    said = flat(docstring_of(SDK / "src" / "tensorquill" / "accelerator.py", "watch_accelerator"))
    assert "recorded from its file names (`session.checkpoint_files`)" in said
    # At a load the model's weights are compared too, from headers alone (lane AK-2).
    assert "at a load the model's tensors are also compared, from the folder's `.safetensors` headers alone" in said
    assert "no tensor is read" in said


def test_the_agent_instructions_say_what_entry_is_for_a_checkpoint_folder():
    bullet = line_with(INSTRUCTIONS, "- **`checkpoint_tensors`**")
    # A folder is no longer always unavailable: its `.safetensors` file or index is the entry.
    assert "for a checkpoint folder, `entry` is the `.safetensors` file or index that was read" in bullet
    assert "a checkpoint folder whose weights are not in readable `.safetensors` files" in bullet
    assert "(a checkpoint folder, an ambiguous source" not in bullet


def test_the_guide_gives_reruns_opt_out_where_it_introduces_the_extra():
    extra = line_with(GUIDE, "- `rerun`:")
    assert "`rerun analytics disable`" in extra
    assert "TensorQuill sends none" in extra
    assert "README.md#no-telemetry" in extra


def test_the_guide_says_the_opt_out_does_not_reach_the_web_viewer():
    viewer = line_with(GUIDE, "The second command opens the recording in the Rerun viewer.")
    assert "`rerun --web-viewer <file>.rrd`" in viewer
    assert "`rerun analytics disable` does not reach the web viewer" in viewer
    assert "open the file in the native viewer" in viewer
