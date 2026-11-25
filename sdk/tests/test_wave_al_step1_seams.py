"""Wave AL, step 1's seam (`WAVE_AL_CONTRACT.md` sections 3, 5 and 9): what no lane owned, and the
owner's answers to the lanes' decisions (Nov 24, 2026).

1. Protocol 5: `server_detect.PROTOCOL` and `PROTOCOL` in `server/src/main.rs` are both 5 (D13), each
   with a line saying what 5 changed, and this tree's binary prints it. The pages give the same number.
2. One row-version list in the server: `ingest::SUPPORTED_ROW_VERSIONS` is what ingest stores and
   what `detect` reads (0, 1 and 2).
3. One wording of the server's refusal of a batch metadata version, read by both sides from
   `fixtures/wave_al/refusal.json`: the sender recognises a protocol 4 server's and this server's.
4. The known differences on a name Arrow cannot encode, the Python side (the Rust side is in
   `server/src/detect/lane_al3_tests.rs`).
5. The limit of lane AL-1's row-version rule: a record of version 2 is stored and its part refused.
6. The README's passages on the versions, an older server, a downgrade and the probe's columns.

Every test here runs without pyarrow and without the server binary, but the ones that start the binary,
which skip without it. None needs git history."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
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


# --- 5. The limit of lane AL-1's row-version rule, pinned (the owner, on AL-1's decision 1) -------------
# A record keeps its own version as its row's when no reader takes it (99 -> row 99, which a server
# drops and counts, RS10), but the version readers take, 0 and 1, and also 2 get row version 2. So a
# record of version 2, which step 1's package does not read (step 2 makes it the scalar-only record,
# in the same push), is stored, and `load_records` then refuses the part holding it.

try:
    import pyarrow as pa
    import pyarrow.flight as pa_flight
    import pyarrow.parquet as pq
except ImportError:
    pa = pa_flight = pq = None

needs_server = pytest.mark.skipif(BINARY is None or pa is None,
                                  reason="needs the server binary and pyarrow (TENSORQUILL_SERVER_BIN)")


def test_a_record_of_version_2_gets_row_version_2_with_its_own_in_record_version():
    record = {"schema_version": 2, "run_id": "v2", "rank": 0, "attempt": 1, "step": 2, "grad_norm": 1.0}
    row = flight._row(record)  # the sender's row, which `rows_to_send` builds without the identity check
    assert (row["schema_version"], row["record_version"]) == (flight.ROW_VERSION, 2) == (2, 2)
    row = flight._row({**record, "schema_version": 99})
    assert (row["schema_version"], row["record_version"]) == (99, 99)  # a version no reader takes


class _Server:
    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen([BINARY, "--listen", "127.0.0.1:0", "--store", str(store),
                                         "--watchdog-seconds", "0.5", "--flush-seconds", "0.2"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def close(self) -> None:
        import signal

        self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        self.process.stdout.close()
        self.process.stderr.close()


def _records(run_id: str, versions: dict[int, int]) -> list[dict]:
    return [{"schema_version": versions.get(step, 1), "run_id": run_id, "rank": 0, "attempt": 1, "step": step,
             "ts": float(step), "grad_norm": 1.0} for step in range(5)]


def _stored(store: Path, run_id: str) -> tuple[Path, list, list, dict]:
    """The run's one part, its row and record versions, and the rank's entry in `watchdog.json`."""
    [part] = sorted((store / "runs" / run_id).rglob("part-*.parquet"))
    table = pq.read_table(part, partitioning=None)
    status = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
    [entry] = status["ranks"] if isinstance(status["ranks"], list) else status["ranks"].values()
    return part, table.column("schema_version").to_pylist(), table.column("record_version").to_pylist(), entry


def _refusal(store: Path, run_id: str) -> str:
    from tensorquill import evidence

    with pytest.raises(identity.MalformedRecordError) as raised:
        evidence.load_records(store / "runs" / run_id)
    return str(raised.value)


@needs_server
def test_this_server_stores_a_record_of_version_2_and_load_records_refuses_its_part(tmp_path):
    store, run_id = tmp_path / "store", "v2"
    path = tmp_path / f"{run_id}.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _records(run_id, {2: 2})))
    server = _Server(store)
    try:
        assert flight.send_file(path, server.endpoint, run_id, 0, timeout=60) is True  # every byte acknowledged
        time.sleep(1.5)  # a watchdog pass writes the rank's status
    finally:
        server.close()
    part, rows, records, entry = _stored(store, run_id)
    assert (rows, records) == ([2] * 5, [1, 1, 2, 1, 1])
    assert "dropped_rows" not in entry and "dropped_row_keys" not in entry, entry  # not dropped, not counted
    assert _refusal(store, run_id) == f"schema_version is 2 at {part}, row 2: this tensorquill reads schema versions 0 and 1"


@needs_server
def test_load_records_refuses_a_part_holding_a_record_version_it_does_not_take_under_row_version_2(tmp_path):
    """The lane's reason for its rule, run: had a version 99 record been given row version 2, as the
    contract first had it, this server would store it and `load_records` would refuse the whole part."""
    store, run_id = tmp_path / "store", "v99"
    rows = [flight._row(record) for record in _records(run_id, {3: 99})]
    rows[3]["schema_version"] = 2
    assert rows[3]["record_version"] == 99
    batch = pa.RecordBatch.from_pylist(rows, schema=flight.flight_schema())
    server = _Server(store)
    try:
        client = pa_flight.FlightClient(server.endpoint)
        writer, reader = client.do_put(pa_flight.FlightDescriptor.for_path("runs", run_id, "rank=0"), batch.schema)
        metadata = {"schema_version": flight.SCHEMA_VERSION, "run_id": run_id, "rank": 0, "min_step": 0,
                    "max_step": 4, "end_byte_offset": 5}
        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
        assert json.loads(reader.read().to_pybytes())["end_byte_offset"] == 5
        writer.done_writing()
        writer.close()
        client.close()
        time.sleep(1.5)
    finally:
        server.close()
    part, row_versions, records, entry = _stored(store, run_id)
    assert (row_versions, records) == ([2] * 5, [1, 1, 1, 99, 1])
    assert "dropped_row_keys" not in entry, entry
    assert _refusal(store, run_id) == f"schema_version is 99 at {part}, row 3: this tensorquill reads schema versions 0, 1 and 2"


@needs_server
def test_under_the_lane_s_rule_a_record_of_version_99_is_dropped_and_counted_and_the_rest_is_read(tmp_path):
    store, run_id = tmp_path / "store", "v99rule"
    path = tmp_path / f"{run_id}.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _records(run_id, {3: 99})))
    server = _Server(store)
    try:
        assert flight.send_file(path, server.endpoint, run_id, 0, timeout=60) is True
        time.sleep(1.5)
    finally:
        server.close()
    _, row_versions, records, entry = _stored(store, run_id)
    assert (row_versions, records) == ([2] * 4, [1] * 4)
    assert entry["dropped_row_keys"] == {"unsupported_schema_version": 1}, entry
    from tensorquill import evidence

    assert [r["step"] for r in evidence.load_records(store / "runs" / run_id)] == [0, 1, 2, 4]


# --- 6. The README's passages on protocol 5 (section 9: the versions, an older server, the probe) ---


STREAMING = README[README.index("## Streaming to an ingest server\n"):README.index("\n## Detector benchmark on 64 ranks")]


def versions_passage() -> str:
    """From "**Protocol 5's versions.**" to the end of the streaming section."""
    return STREAMING[STREAMING.index("**Protocol 5's versions.**"):]


def version_table() -> dict[str, list[str]]:
    """The passage's table, by its first cell: the four numbers of section 3."""
    rows = [line for line in versions_passage().splitlines() if line.startswith("| ")]
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    cells = [row for row in cells if set("".join(row)) != {"-"}]  # not the separator row
    assert cells[0] == ["Number", "Where", "Protocol 5 (before)", "What it gates"], cells[0]
    return {row[0]: row[1:] for row in cells[1:]}


def test_the_version_table_gives_section_3_s_four_numbers_as_the_code_has_them():
    table = version_table()
    assert list(table) == ["The protocol", "Batch metadata `schema_version`", "Row `schema_version`",
                           "Record `schema_version`"]
    protocol, metadata, row, record = table.values()
    assert "`server_detect.PROTOCOL`" in protocol[0] and protocol[1].startswith(f"{sd.PROTOCOL} (4)")
    assert "`flight.SCHEMA_VERSION`" in metadata[0] and metadata[1].startswith(f"{flight.SCHEMA_VERSION} (1)")
    assert "`flight.ROW_VERSION`" in row[0] and row[1].startswith(f"{flight.ROW_VERSION} (0 or 1: the record's own), with the record's own version in `record_version`")
    assert "`identity.SCHEMA_VERSIONS`" in record[0]
    assert record[1].startswith(" or ".join(map(str, identity.SCHEMA_VERSIONS)) + ", unchanged")
    # What the server takes and stores, as `ingest.rs` has it.
    ingest = server_source("ingest.rs")
    assert "pub const SCHEMA_VERSIONS: [i64; 2] = [1, 2];" in ingest
    assert "the server takes 1 and 2" in metadata[1]
    assert "this server reads versions 1 and 2" in ingest and "this server reads versions 1 and 2" in metadata[2]
    assert "the server stores 0, 1 and 2" in row[1]
    assert "`unsupported_schema_version`" in row[2] and "`tensorquill-server detect` refuses the input" in row[2]
    # Not step 2 of the wave: no scalar-only record is described as built.
    assert "scalar-only" not in versions_passage() and "scalars-" not in versions_passage()


def test_the_older_server_message_the_readme_quotes_is_the_senders():
    from types import SimpleNamespace

    passage = flat(versions_passage())
    quoted = re.search(r"`(the server at <endpoint> speaks an older protocol[^`]*)`", passage).group(1)
    stand_in = SimpleNamespace(endpoint="grpc://host:50051", path=Path("run.rank0.jsonl"), _life={})
    refused = ConnectionError("INVALID_ARGUMENT: schema_version 2 is not supported; this server reads version 1")
    said = str(flight.FlightSender._older(stand_in, refused))
    assert said.startswith(quoted.replace("<endpoint>", "grpc://host:50051")), (said, quoted)
    for words in ("`schema_version 2 is not supported; this server reads version 1`", "stores nothing",
                  "says once, on stderr and in `session._sink.sender.last_error`", "keeps every line",
                  "A sender older than protocol 5 streams to a protocol 5 server as before"):
        assert words in passage, words


def test_the_readme_says_a_downgrade_is_not_supported_and_how_it_shows():
    passage = flat(versions_passage())
    start = passage.index("**A downgrade is not supported.**")
    downgrade = passage[start:start + 900]
    for words in ("A protocol 4 server", "protocol 5 parts", "without a word", "no layers", "no MoE statistics",
                  "no attention maps",
                  # Corrected Nov 24 from the run across a real protocol 4 build (experiments/protocol-v5/
                  # cross_version.py, row 4): the old load_records and a read refuse such a store; its runs does not.
                  "`load_records` and a read of the run", "refuse a store that holds protocol 5 parts, naming the part and the row",
                  "its `runs` lists the run from the server's own reports"):
        assert words in downgrade, words


def test_the_columns_bullet_gives_protocol_5_s_columns_and_the_probe_split():
    [columns] = [line for line in STREAMING.splitlines() if line.startswith("- **Columns.**")]
    for words in ("`layers` and `moe_stats` are typed columns", "in the model's order", "`probe_maps`",
                  "`probe_histograms`", "`probe_json`", "The `.jsonl` file holds the whole probe",
                  "`layers_json` and `moe_stats_json`", "still read", "`extra_json`"):
        assert words in columns, words
    assert "`layers`, `moe_stats` and `probe` are JSON strings" not in columns


def test_the_identity_rules_name_the_row_versions_detect_reads():
    identity_section = flat(README[README.index("**A record's identity fields.**"):
                                   README.index("**Several refused batches in a row.**")])
    assert "| `schema_version` | read as 0, the legacy version | the integer 0 or 1 |" in identity_section
    assert "a row the server stores carries row version 2, with the record's own in `record_version`" in identity_section
    assert "and a `schema_version` other than 0, 1 or 2." in identity_section
    assert "other than 0 or 1" not in identity_section
