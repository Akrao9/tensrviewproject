"""Wave AL, step 1's seam (`WAVE_AL_CONTRACT.md` sections 3, 5 and 9): what no lane owned.

1. Protocol 5: `server_detect.PROTOCOL` and `PROTOCOL` in `server/src/main.rs` are both 5 (D13), each
   with a line saying what 5 changed, and this tree's binary prints it. The pages give the same number.
2. One row-version list in the server: `ingest::SUPPORTED_ROW_VERSIONS` is what ingest stores and
   what `detect` reads (0, 1 and 2).

Every test here runs without pyarrow and without the server binary, but the one that runs the binary,
which skips without it. None needs git history.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tensorquill import server_detect as sd

SDK = Path(__file__).resolve().parents[1]
REPO = SDK.parent
SERVER_SRC = REPO / "server" / "src"
README = (SDK / "README.md").read_text(encoding="utf-8")
GUIDE = (SDK / "RUN_ON_YOUR_JOB.md").read_text(encoding="utf-8")


def flat(text: str) -> str:
    return " ".join(text.split())


def server_source(name: str) -> str:
    path = SERVER_SRC / name
    if not path.exists():
        pytest.skip(f"no server/src/{name} in this checkout")
    return path.read_text(encoding="utf-8")


def _binary() -> str | None:
    path = os.environ.get(sd.BINARY_ENV)
    return path if path and Path(path).is_file() else None


BINARY = _binary()
needs_binary = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


# --- 1. Protocol 5 -----------------------------------------------------------------------------


def test_the_sdk_and_the_server_source_speak_protocol_5():
    assert sd.PROTOCOL == 5
    [number] = re.findall(r"^const PROTOCOL: u32 = (\d+);$", server_source("main.rs"), re.M)
    assert int(number) == 5


def test_each_side_says_what_protocol_5_changed():
    """A line for 5 beside the lines for 3 and 4, naming the contract and what `detect` now reads."""
    main = flat(server_source("main.rs"))
    sdk = flat((SDK / "src" / "tensorquill" / "server_detect.py").read_text(encoding="utf-8"))
    for text, start in ((main, "/// 5 (Nov 24, 2026, wave AL"), (sdk, "# 5 (wave AL, Nov 24, 2026)")):
        assert start in text, start
        line = text[text.index(start):][:600]
        for words in ("`WAVE_AL_CONTRACT.md`", "`layers` and `moe_stats`", "`probe_maps`", "`probe_histograms`",
                      "row version 2", "`record_version`"):
            assert words in line, (start, words)


@needs_binary
def test_this_trees_binary_prints_protocol_5():
    done = subprocess.run([BINARY, "protocol"], capture_output=True, text=True, timeout=30)
    assert (done.returncode, done.stdout, done.stderr) == (0, "5\n", "")
    assert sd.binary_protocol(BINARY) == sd.PROTOCOL


def test_the_pages_give_protocol_5_and_quote_a_binary_of_protocol_4():
    readme, guide = flat(README), flat(GUIDE)
    assert "both speak protocol 5 since Nov 24, 2026 (4 before)" in readme
    assert "The SDK and `tensorquill-server` must be built from the same version (protocol 5)." in guide
    for message in ("`--watch needs the Rust detectors: <binary> speaks protocol 4 and this tensorquill needs protocol 5:",
                    "`tensorquill: warning: using the Python detectors, not the Rust ones: <binary> speaks protocol 4 "
                    "and this tensorquill needs protocol 5: install"):
        assert message in readme, message
    for page in (readme, guide):
        assert "needs protocol 4" not in page
        assert "(protocol 4)" not in page


# --- 2. One row-version list -------------------------------------------------------------------


def test_the_server_keeps_one_row_version_list_read_by_ingest_and_detect():
    ingest, detect = server_source("ingest.rs"), server_source("detect.rs")
    assert "pub const SUPPORTED_ROW_VERSIONS: [i32; 3] = [0, 1, 2];" in ingest
    assert "STORED_ROW_VERSIONS" not in ingest and "STORED_ROW_VERSIONS" not in detect
    assert not re.search(r"\bconst ROW_VERSIONS\b", detect)
    assert "!SUPPORTED_ROW_VERSIONS.contains(" in ingest  # what ingest stores (RS10)
    assert "!crate::ingest::SUPPORTED_ROW_VERSIONS.contains(" in detect  # what `check_keys` reads
