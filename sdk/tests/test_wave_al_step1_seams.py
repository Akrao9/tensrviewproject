"""Wave AL, step 1's seam (`WAVE_AL_CONTRACT.md` sections 3, 5 and 9): what no lane owned.

1. Protocol 5: `server_detect.PROTOCOL` and `PROTOCOL` in `server/src/main.rs` are both 5 (D13), each
   with a line saying what 5 changed, and this tree's binary prints it. The pages give the same number.
2. One row-version list in the server: `ingest::SUPPORTED_ROW_VERSIONS` is what ingest stores and
   what `detect` reads (0, 1 and 2).

Every test here runs without pyarrow and without the server binary, but the one that runs the binary,
which skips without it. None needs git history.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tensorquill import flight, identity
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
        assert "same version (protocol 4)" not in page  # history ("Since Nov 21, 2026 (protocol 4) ...") stays


# --- 2. One row-version list -------------------------------------------------------------------


def test_the_server_keeps_one_row_version_list_read_by_ingest_and_detect():
    ingest, detect = server_source("ingest.rs"), server_source("detect.rs")
    assert "pub const SUPPORTED_ROW_VERSIONS: [i32; 3] = [0, 1, 2];" in ingest
    assert "STORED_ROW_VERSIONS" not in ingest and "STORED_ROW_VERSIONS" not in detect
    assert not re.search(r"\bconst ROW_VERSIONS\b", detect)
    assert "!SUPPORTED_ROW_VERSIONS.contains(" in ingest  # what ingest stores (RS10)
    assert "!crate::ingest::SUPPORTED_ROW_VERSIONS.contains(" in detect  # what `check_keys` reads


# --- 3. One wording for the server's refusal, read by both sides (the owner's seam note) ------------

REFUSAL = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "refusal.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("server", ["protocol_4_server", "this_server"])
def test_the_sender_recognises_both_wordings_of_the_refusal_and_reads_the_versions(server):
    """A protocol 4 server's singular "reads version 1" and this server's plural "reads versions 1 and
    2" (pinned on the server's side by lane_al2_protocol5_test.rs) are both an older server to a sender
    of the refused version, and the versions it reads are read out of each."""
    case = REFUSAL[server]
    assert flight._older_server_reads(case["text"], case["sent"]) == case["reads"]
    wrapped = f"Flight returned invalid argument error, with message: {case['text']}. gRPC client debug context: x"
    assert flight._older_server_reads(wrapped, case["sent"]) == case["reads"]
    assert flight._older_server_reads(case["text"], case["sent"] + 1) is None  # another sender's version


def test_the_sender_of_this_package_says_an_older_server_reads_its_versions():
    from types import SimpleNamespace

    assert REFUSAL["protocol_4_server"]["sent"] == flight.SCHEMA_VERSION
    stand_in = SimpleNamespace(endpoint="grpc://host:50051", path=Path("run.rank0.jsonl"), _life={})
    singular = str(flight.FlightSender._older(stand_in, ConnectionError(REFUSAL["protocol_4_server"]["text"])))
    assert "It reads batch schema_version 1 and this tensorquill sends version 2" in singular
    plural = REFUSAL["this_server"]["text"].replace("schema_version 3 ", "schema_version 2 ").replace("1 and 2", "1 and 3")
    said = str(flight.FlightSender._older(stand_in, ConnectionError(plural)))
    assert "It reads batch schema_versions 1 and 3 and this tensorquill sends version 2" in said


# --- 4. Known differences on a name Arrow cannot encode, pinned on each side (not fixed) --------------
# The owner, Nov 24, 2026, on lane AL-3's decisions 3 and 4. The Rust sides are
# `a_protocol_4_row_with_a_name_arrow_cannot_encode_reads_no_layer_in_rust` and
# `a_name_arrow_cannot_encode_is_not_read_by_non_finite` (`server/src/detect/lane_al3_tests.rs`).

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text(encoding="utf-8"))
SURROGATE_CASE = next(case for case in CASES["cases"] if case["name"] == "layers_name_arrow_cannot_encode")
ACTIVATION_KINDS = ("activation_spike", "activation_drift", "dead_activations")


def _protocol_4_row(record: dict) -> dict:
    """`record` as a protocol 4 sender's row: the layers as JSON text in `layers_json`, row version 1."""
    row = {key: record[key] for key in ("run_id", "rank", "attempt", "step", "schema_version", "ts", "grad_norm")}
    row.update(layers_json=json.dumps(record["layers"]), moe_stats_json=None, extra_json=None)
    return row


def _activation_checks(records) -> dict:
    from tensorquill import evidence

    checks = evidence.build_evidence(records)["kind_checks"]
    return {kind: checks[kind]["value"] for kind in ACTIVATION_KINDS}


def test_python_reads_every_layer_of_a_protocol_4_row_with_a_name_arrow_cannot_encode():
    """Rust reads no layer of such a row (its activation kinds `not_configured`, the case's `today`
    `rust_raw`); Python reads the row back to the record, every layer and the name included."""
    records = SURROGATE_CASE["records"]
    assert any("bad\ud800" in record["layers"] for record in records)
    rows = [_protocol_4_row(record) for record in records]
    assert "bad\\ud800" in rows[1]["layers_json"]  # the escape serde_json refuses
    back = [flight.row_to_record(row) for row in rows]
    assert back == records
    assert _activation_checks(back) == dict.fromkeys(ACTIVATION_KINDS, "checked")
    today = SURROGATE_CASE["today"]
    assert {kind: today["python"]["kind_checks"][kind] for kind in ACTIVATION_KINDS} == _activation_checks(back)
    assert {kind: today["rust_raw"]["kind_checks"][kind] for kind in ACTIVATION_KINDS} == dict.fromkeys(
        ACTIVATION_KINDS, "not_configured")


def test_python_names_a_non_finite_field_of_a_name_arrow_cannot_encode_where_rust_does_not():
    """In protocol 5's layout the entry travels in `extra_json`, whose whole text Rust cannot parse (the
    lone surrogate's escape), so Rust's `non_finite` names no layer there; Python reads the row back to
    the record and names the layer."""
    records = [{"schema_version": 1, "run_id": "s", "rank": 0, "attempt": 1, "step": step, "ts": float(step),
                "grad_norm": 1.0, "layers": {"model.layers.0": {"mean": 0.0, "std": 1.0, "abs_max": 3.0,
                                                                "zero_frac": 0.0}}} for step in range(3)]
    records[1]["layers"]["bad\ud800"] = {"p99": "nan"}
    rows = [flight.record_to_row(record) for record in records]
    assert json.loads(rows[1]["extra_json"]) == {"layers": {"bad\ud800": {"p99": "nan"}}}
    assert "bad\\ud800" in rows[1]["extra_json"]
    back = [flight.row_to_record(row) for row in rows]
    assert back == records
    from tensorquill import evidence

    found = [incident for incident in evidence.build_evidence(back)["incidents"] if incident["kind"] == "non_finite"]
    assert found and found[0]["step"] == 1 and "bad\ud800" in json.dumps(found[0], ensure_ascii=False), found
