"""Lane AK-4 (wave AK): the loader delay, documented.

The delay of one `next()` (bug list, section 14) was measured to be the interpreter's one owed full
garbage collection, brought forward by the capture's per-flush allocations and run on the
`tensorquill-writer` thread (the lane's findings, Nov 24, 2026). The coordinator chose, for the
owner, to leave the collector alone and say so: TensorQuill changes no collector setting, and a
user who needs the pause gone calls `gc.collect()` after `tensorquill.init(...)`. This pins that the
README's "GPU memory, NVML and data wait" section, where a reader looking at one slow `data_wait`
or `step_time` reads, says it, with the scope that was measured.
"""

from __future__ import annotations

from pathlib import Path

SDK = Path(__file__).resolve().parents[1]
README = (SDK / "README.md").read_text()


def flat(text: str) -> str:
    return " ".join(text.split())


def section(text: str, title: str) -> str:
    """One `## ` section of a page, to its next `## ` heading."""
    start = text.index(f"## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def test_the_data_wait_section_says_what_the_one_slow_step_is_and_how_to_remove_it():
    text = flat(section(README, "GPU memory, NVML and data wait"))
    start = text.index("**One slow step from the garbage collector.**")
    passage = text[start:text.index("- **", start + 10)]
    # What runs it, and where.
    assert "full garbage collection" in passage and "`tensorquill-writer`" in passage
    assert "interpreter lock" in passage
    # The scope that was measured, with its limits.
    assert "once per run" in passage and "20,000 steps" in passage
    assert "45 to 110 ms" in passage and "Python 3.12" in passage
    assert "A100" in passage and "owes" in passage
    # The remedy is the user's, and TensorQuill sets nothing itself.
    assert "`gc.collect()` after `tensorquill.init(...)`" in passage
    assert "`gc.freeze()`" in passage and "never collected" in passage
    assert "TensorQuill does not change the collector's settings" in passage
