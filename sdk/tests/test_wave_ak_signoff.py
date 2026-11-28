"""Wave AK, the author's sign-off of Nov 24, 2026 ("Yes, sign all"): the checkpoint contract's folder
section is signed, and the four signed statements it names are amended in place (bug list, section 20).

Pages only: no behaviour is tested here; `test_lane_ak2_safetensors.py` tests what the section says.
"""

from __future__ import annotations

from pathlib import Path

import conftest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "sdk" / "README.md").read_text()


def section(number: int) -> str:
    """One numbered section of the contract, heading included. Read at the call, so that with no page in the
    checkout the test that asked is skipped and the README's test still runs."""
    CONTRACT = conftest.contract_page("CHECKPOINT_TENSOR_CONTRACT.md")
    start = CONTRACT.index(f"\n## {number}. ")
    end = CONTRACT.find("\n## ", start + 1)
    return CONTRACT[start : end if end != -1 else len(CONTRACT)]


def line_with(text: str, start: str) -> str:
    found = [line for line in text.splitlines() if line.startswith(start)]
    assert len(found) == 1, (start, len(found))
    return found[0]


def test_the_folder_section_is_signed():
    eight = section(8)
    assert eight.startswith("\n## 8. A checkpoint folder's tensors (lane AK-2; signed by the author on Nov 24, 2026)")
    assert "**Signed by the author on Nov 24, 2026.**" in eight
    for unsigned in ("Not signed", "awaiting the author", "for the author to confirm", "if signed", "the author decides whether"):
        assert unsigned not in eight, unsigned


def test_the_moe_reading_is_signed_as_a_stated_limit_with_no_matching_rule():
    eight = section(8)
    assert "**A model whose saved names are not its own: signed as a stated limit.**" in eight
    assert "No matching rule is added" in eight
    assert "is compared as it stands" in eight


def test_the_bounds_are_signed_as_chosen():
    assert "**The bounds, signed as chosen.** 1,024 files and 32 MiB of headers at one load" in section(8)


def test_the_four_signed_statements_are_amended_in_place():
    why = line_with(section(3), "| `folder` |")
    assert "file names only, no file opened" not in why
    assert "no `.safetensors` weights in the folder, or their headers refused (`refused` says why)" in why
    assert "section 8" in why
    apart = line_with(section(4), "| A folder")
    assert "with no `.safetensors` weights, or whose headers are refused" in apart
    assert "read as any compared load (section 8)" in apart
    unseen = line_with(section(5), "- **A checkpoint folder")
    assert "not proposed here" not in unseen
    assert "pickled weights" in unseen and "section 8" in unseen
    assert "`checkpoint_files()` compares a folder load from its `.safetensors` headers (section 8)" in section(6)
    assert "marks a folder load" not in section(6)


def test_the_readme_no_longer_says_the_section_awaits_sign_off():
    bullet = line_with(README, "  - **A checkpoint folder** (lane AK-2")
    assert "awaits the author's sign-off" not in bullet
    assert "the contract's section 8, signed Nov 24, 2026" in bullet
