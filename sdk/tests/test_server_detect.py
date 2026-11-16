"""`build_evidence(detectors="rust")` against the Python detectors, on the same records.

The Rust path runs `tensorquill-server detect` on the records in hand and takes the incidents
of the server's kinds from its report; everything else in the evidence is still made in
Python. Both paths must give the same evidence: incidents, `not_checked` and the run state
compared as the server parity tests compare them (numbers to 1e-12, a NaN as `null`), the rest
exactly. The two places where they differ on purpose are pinned further down.

The five threshold options go to the Rust rules as the server's one `--thresholds` JSON
object. `CASES` stays at the defaults; the options have their own table (`OPTIONS`).

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import tensorquill  # noqa: E402
from tensorquill.evidence import SERVER_KINDS  # noqa: E402
from tensorquill.server_detect import BINARY_ENV, server_binary, server_report  # noqa: E402

import test_attention  # noqa: E402
import test_data_server  # noqa: E402
import test_evidence_system  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402
from test_server_detectors import differences, edges, everything, moe_edges, one_parameter, record, wobble  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
FIXTURES = Path(__file__).resolve().parents[2] / "experiments" / "moe-bakeoff" / "artifacts"
JUDGED = ("incidents", "not_checked", "run_state")


@pytest.fixture(autouse=True)
def binary(monkeypatch):
    monkeypatch.setenv(BINARY_ENV, str(BINARY))


def both(records, **options) -> list[str]:
    """Where the Rust path's evidence differs from the Python path's, both given `options`."""
    python = tensorquill.build_evidence(records, **options)
    rust = tensorquill.build_evidence(records, detectors="rust", **options)
    assert "detectors" not in python and rust.pop("detectors") == "rust"
    found = [d for key in JUDGED for d in differences(python.get(key, []), rust.get(key, []), key)]

    def rest(evidence):
        return json.dumps({k: v for k, v in evidence.items() if k not in JUDGED}, sort_keys=True)

    return found + (["the keys made in Python differ"] if rest(python) != rest(rust) else [])


def lr_resumes() -> list[list[dict]]:
    """A run with resumes and restarts, as the stretches it was written in."""
    def warmup(start):
        return lambda step: 1e-3 * min(1.0, (step - start + 1) / 100)

    def attempt(number, steps, lr):
        return [
            record(step, number, grad_norm=wobble(step), scalars={"train/loss": 2.0 - step / 1000 + number / 100},
                   **({} if lr is None else {"lr": [lr(step), 0.1 * lr(step)]}))
            for step in steps
        ]

    return [
        attempt(1, range(150), warmup(0)),
        attempt(2, range(100, 180), warmup(100)),  # warmup starts over at a step the run had
        attempt(3, range(200, 220), lambda step: 5e-4),  # a resume past the stored steps
        attempt(4, range(230, 240), lambda step: 1e-5),
        attempt(5, range(235, 250), lambda step: 1e-5),
        attempt(6, range(220, 245), None),  # no learning rate
        attempt(7, range(120, 130), warmup(0)),
        attempt(8, range(0, 120), warmup(0)),  # a restart from step 0
        attempt(9, range(0, 10), lambda step: 3e-4),
    ]


def growing(stretches) -> list[list[dict]]:
    sent: list[dict] = []
    return [list(sent := sent + stretch) for stretch in stretches]


def cases() -> dict[str, list[dict]]:
    out = {"everything": everything(), "one parameter": one_parameter(), "moe edges": moe_edges()}
    out.update({f"edge {name}": records for name, records in edges().items()})
    stretches = test_data_server.attempts()
    out.update({
        f"data, up to {name}": records
        for (name, _), records in zip(stretches, growing(records for _, records in stretches))
    })
    out.update({f"data edge {name}": records for name, records in test_data_server.edges().items()})
    out["attention"] = test_attention.parity_run()
    out.update({f"attention edge {run}": test_attention.edge_run(run) for run in ("a", "b")})
    out.update({f"resumes, {n + 1} attempts written": records for n, records in enumerate(growing(lr_resumes()))})
    system = test_evidence_system.system_run()
    # Ends inside and at a reading, and at or just past each incident, as the server's stretches do.
    out.update({f"system, {upto} steps": system[:upto] for upto in (10, 11, 112, 115, 190, 230, 260, 300)})
    out.update({f"system edge {name}": records for name, records in test_evidence_system.edges().items()})
    # Checks that could not be made: too few probes, and resumes with nothing to compare.
    out["attention, the first probes"] = test_attention.parity_run()[:30]
    row, tag = test_data_server.row, test_data_server.tag
    out["a resume after a rollback"] = (
        [row(s, 1, tag(s)) for s in range(30)] + [row(s, 2, f"b-{s}") for s in range(10, 15)]
        + [row(s, 3, tag(s)) for s in range(20, 25)]
    )
    unloadable = [row(s, 1, tag(s)) for s in range(10)]
    unloadable[3]["checkpoint"] = {"save": {"components": test_data_server.FULL, "after_step": 30}}
    out["a resume from a checkpoint it cannot have loaded"] = unloadable + [row(s, 2, tag(s)) for s in range(5, 12)]
    out["no run id and no attempt"] = [{"step": s, "ts": float(s), "grad_norm": wobble(s)} for s in range(30)]
    return out


CASES = cases()


def test_the_cases_reach_every_kind_the_server_has():
    seen = {i["kind"] for records in CASES.values() for i in tensorquill.build_evidence(records)["incidents"]}
    assert set(SERVER_KINDS) <= seen, set(SERVER_KINDS) - seen
    # And the checks that could not be made, which the Rust path lists in Python's place.
    listed = {n["check"] for records in CASES.values() for n in tensorquill.build_evidence(records).get("not_checked", [])}
    assert {"resume_lr", "resume_data", "resume_state", "attention", "cuda_memory", "nvml"} <= listed, listed


@pytest.mark.parametrize("name", CASES)
def test_the_rust_path_gives_the_python_paths_evidence(name):
    assert both(CASES[name]) == []


@pytest.mark.parametrize("fixture", sorted(p.name for p in FIXTURES.iterdir() if (p / f"{p.name}.jsonl").exists()))
def test_the_gate_1_fixtures_give_the_same_evidence(fixture):
    records = tensorquill.load_records(FIXTURES / fixture / f"{fixture}.jsonl", all_attempts=True)
    assert both(records) == []
    assert tensorquill.build_evidence(records, detectors="rust")["incidents"], "every fixture has a planted failure"


def test_the_report_is_the_document_the_server_writes():
    report = server_report(everything())
    assert report["source"] == "tensorquill-server" and report["rebuilds"] == 0
    assert (report["run_id"], report["rank"], report["steps"], report["last_step"]) == ("detect", 0, 300, 299)
    assert tuple(report["checked"]) == tuple(SERVER_KINDS)
    assert all("class" in incident for incident in report["incidents"])


# --- where the two paths differ, on purpose -------------------------------------------------


def test_a_step_written_twice_in_one_attempt_is_reported_with_the_scalars_of_its_second_writing():
    def row(step, salt):
        return record(step, 1, grad_norm=wobble(step), scalars={"train/loss": 2.0 - step / 100 + salt})

    records = [row(s, 0) for s in range(40)] + [row(s, 1) for s in range(30, 50)]
    python = tensorquill.build_evidence(records)["incidents"]
    rust = tensorquill.build_evidence(records, detectors="rust")["incidents"]
    assert [i["kind"] for i in python] == [i["kind"] for i in rust] == ["repeated_steps"]
    # The incident is the same: the server's own store never holds such rows, so only this path finds it.
    assert {k: v for k, v in python[0].items() if k != "scalars"} == {k: v for k, v in rust[0].items() if k != "scalars"}
    assert (python[0]["step"], python[0]["repeated"]) == (30, 10)
    # Python gives what was logged when step 30 was first written, Rust when it was written again.
    assert python[0]["scalars"]["train/loss"]["value"] == pytest.approx(1.7)
    assert rust[0]["scalars"]["train/loss"]["value"] == pytest.approx(2.7)


def test_attempts_out_of_order_in_the_input_are_read_oldest_first_as_a_store_holds_them():
    first = [record(s, 1, grad_norm=wobble(s), lr=[1e-3 * min(1.0, (s + 1) / 100)], scalars={"train/loss": 2.0 - s / 1000}) for s in range(120)]
    resumed = [record(s, 2, grad_norm=wobble(s), lr=[1e-3 * min(1.0, (s - 99) / 100)], scalars={"train/loss": 2.02 - s / 1000}) for s in range(100, 150)]
    rust = tensorquill.build_evidence(resumed + first, detectors="rust")["incidents"]
    assert [i["kind"] for i in rust] == ["resume_lr_change"]
    assert differences(tensorquill.build_evidence(first + resumed)["incidents"], rust) == []


# --- what the Rust path refuses -------------------------------------------------------------


# --- the five options, through `--thresholds` -------------------------------------------------

OPTIONS: list[dict] = (
    [{"overload_threshold": v} for v in (0.0, 0.2, 0.5, 0.9)]
    + [{"drift_tolerance": v} for v in (0.0, 1e-3, 0.5)]
    + [{"starve_window": v} for v in (1, 10, 20)]
    + [{"starve_ratio": v} for v in (0.0, 0.5, 0.9)]
    + [{"top_k": v} for v in (1, 2, 3, 8)]
    + [{"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2}]
)


def option_sets() -> dict[str, list[dict]]:
    """The record sets these options can change: the ones with MoE counts or a replica spread."""
    sets = dict(CASES)
    for folder in sorted(FIXTURES.iterdir()):
        capture = folder / f"{folder.name}.jsonl"
        if capture.exists():
            sets[f"fixture {folder.name}"] = tensorquill.load_records(capture, all_attempts=True)
    return {name: records for name, records in sets.items() if any(r.get("moe") or r.get("replica_spread") for r in records)}


OPTION_SETS = option_sets()
_usual: dict[str, list] = {}


def moves_python(name: str, **options) -> bool:
    """Whether the options change the incidents Python reports on this record set."""
    if name not in _usual:
        _usual[name] = tensorquill.build_evidence(OPTION_SETS[name])["incidents"]
    return tensorquill.build_evidence(OPTION_SETS[name], **options)["incidents"] != _usual[name]


def test_the_option_sets_hold_the_data_the_options_judge():
    assert len(OPTION_SETS) >= 8, sorted(OPTION_SETS)
    recorded = {k for records in OPTION_SETS.values() for r in records for k in (r.get("moe_top_k") or {}).values()}
    # Top-1 routers for the share rule, top-k of three or more for the baseline rule, and a
    # recorded value on every row for an override to stand against.
    assert 1 in recorded and any(k >= 3 for k in recorded), recorded
    assert any(r.get("replica_spread") for records in OPTION_SETS.values() for r in records)


@pytest.mark.parametrize("option", OPTIONS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_an_option_reaches_the_rust_rules_and_gives_pythons_result(option):
    moved = []
    for name, records in OPTION_SETS.items():
        assert both(records, **option) == [], name
        if moves_python(name, **option):
            moved.append(name)
    # Without this the test would pass with the option ignored on both sides.
    assert moved, "this value changes no incident on any record set"


# One router of a record set, and a value for it other than the one recorded on every row,
# chosen because the override moves an incident there.
ONE_ROUTER = {
    "everything": ("a", 2),
    "moe edges": ("early", 1),
    "fixture unseen-3": ("layers.11.router", 2),
    "fixture unseen-4": ("layers.08.router", 3),
    "fixture unseen-6": ("layers.03.router", 8),
}


@pytest.mark.parametrize("name", ONE_ROUTER)
def test_a_top_k_for_one_router_stands_against_the_value_recorded_on_every_row(name):
    records = OPTION_SETS[name]
    target, override = ONE_ROUTER[name]
    recorded = {
        router: {r["moe_top_k"][router] for r in records if router in (r.get("moe_top_k") or {})}
        for router in {router for r in records for router in (r.get("moe") or {})}
    }
    assert recorded[target] and override not in recorded[target], "the override must differ from what the rows carry"
    option = {"top_k": {target: override, "not-a-router": 5}}
    assert both(records, **option) == []
    assert moves_python(name, **option), "this override changes no incident, so the test would pass with it ignored"
    evidence = tensorquill.build_evidence(records, detectors="rust", **option)
    assert evidence["top_k"][target] == override and "not-a-router" not in evidence["top_k"]
    # Every other router keeps what its rows recorded.
    others = [router for router in recorded if router != target]
    assert others and all(evidence["top_k"][router] == max(recorded[router], default=1) for router in others)


def test_the_report_lists_the_thresholds_it_applied():
    usual = server_report(everything())["thresholds"]
    assert (usual["overload_threshold"], usual["drift_tolerance"], usual["starve_window"], usual["starve_ratio"]) == (0.35, 1e-6, 50, 0.1)
    assert usual["top_k"] is None
    assert server_report(everything(), thresholds={})["thresholds"] == usual  # every key left out
    asked = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": {"r1": 2}}
    applied = server_report(everything(), thresholds=asked)["thresholds"]
    assert {name: applied[name] for name in asked} == asked
    assert {name: value for name, value in applied.items() if name not in asked} == {
        name: value for name, value in usual.items() if name not in asked
    }
    # The drift incident says which tolerance it was judged by.
    drift = [i for i in server_report(everything(), thresholds={"drift_tolerance": 0.0})["incidents"] if i["kind"] == "replica_drift"]
    assert drift and drift[0]["tolerance"] == 0.0


@pytest.mark.parametrize("option, reason", [
    ({"starve_window": 0}, "starve_window must be at least 1"),
    ({"top_k": 0}, "top_k must be at least 1"),
    ({"top_k": {"r1": 0}}, "top_k must be at least 1"),
    ({"overload_threshold": -0.1}, "overload_threshold must be a finite number of at least 0"),
    ({"drift_tolerance": -1.0}, "drift_tolerance must be"),
    ({"starve_ratio": float("nan")}, "cannot be sent as JSON"),
    ({"overload_threshold": float("inf")}, "cannot be sent as JSON"),
])
def test_values_the_rust_rules_refuse_raise_value_error_with_the_reason(option, reason):
    with pytest.raises(ValueError, match=reason):
        tensorquill.build_evidence(moe_edges(), detectors="rust", **option)


def test_an_unknown_threshold_key_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown field `overload_treshold`"):
        server_report(moe_edges(), thresholds={"overload_treshold": 0.5})
    with pytest.raises(ValueError, match="unknown field `spike_factor`"):  # listed in a report, but not an option
        server_report(moe_edges(), thresholds={"spike_factor": 4.0})


def test_thresholds_are_compared_to_one_part_in_1e12_in_case_json_text_moves_a_last_place():
    # A precaution: no run has shown a value come back changed.
    from tensorquill.evidence import _same_threshold

    assert both(moe_edges(), overload_threshold=0.1 + 0.2, drift_tolerance=1e-6 * 3, starve_ratio=1 / 3) == []
    assert _same_threshold(0.35, 0.35) and _same_threshold(0.35, 0.35000000000000003) and _same_threshold(1, 1.0)
    assert not _same_threshold(0.35, 0.5) and not _same_threshold(0.35, None) and not _same_threshold(None, 0.35)
    assert _same_threshold(None, None) and _same_threshold({"a": 2}, {"a": 2}) and _same_threshold(50, 50)
    assert not _same_threshold(2, None) and not _same_threshold({"a": 2}, {"a": 3}) and not _same_threshold(50, 20)


def test_a_server_started_with_thresholds_applies_them_and_a_reader_asking_for_the_same_finds_it_current(tmp_path):
    from tensorquill.evidence import read_server_report
    from test_server_detectors import FAST, Stream

    asked = {"overload_threshold": 0.5, "drift_tolerance": 1e-3, "starve_window": 20, "starve_ratio": 0.5, "top_k": 2}
    records = everything()

    def reference(upto):
        return [i for i in tensorquill.build_evidence(records[:upto], **asked)["incidents"] if i["kind"] in SERVER_KINDS]

    assert reference(300) != [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in SERVER_KINDS]
    stream = Stream(tmp_path, *FAST, "--thresholds", json.dumps(asked))
    sent = 0
    for upto in (60, 130, 200):  # each stretch is appended to the detectors the server holds
        document = stream.send(records[sent:upto])
        sent = upto
        assert differences(reference(upto), document["incidents"]) == [], f"after {upto} steps"
    assert {name: document["thresholds"][name] for name in asked} == asked and document["rebuilds"] == 1
    stream.restart()  # the new server rebuilds from its store, with the thresholds it was started with
    document = stream.send(records[200:])
    assert differences(reference(300), document["incidents"]) == []
    assert {name: document["thresholds"][name] for name in asked} == asked

    folder = stream.incidents_file.parent
    assert read_server_report(folder, **asked)["state"] == "current"
    look = read_server_report(folder)
    assert look["state"] == "other_thresholds"
    assert "top_k 2 where None was asked for" in look["reason"] and "overload_threshold 0.5 where 0.35" in look["reason"]
    assert read_server_report(folder, **{**asked, "top_k": {"r1": 2}})["state"] == "other_thresholds"
    assert stream.server.terminate() == 0

    # The watcher takes the server's incidents when it asks for the same values, and runs
    # the Python detectors when it asks for others.
    from test_server_reports import look

    evidence, _ = look(stream, tmp_path / "same", **asked)
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("server", "current")
    assert differences(reference(300), [i for i in evidence["incidents"] if i["kind"] in SERVER_KINDS]) == []
    evidence, _ = look(stream, tmp_path / "usual")
    assert (evidence["detection"]["source"], evidence["detection"]["state"]) == ("python", "other_thresholds")
    # (A logged NaN is in these incidents, so they are compared by kind and step.)
    usual = tensorquill.build_evidence(records)["incidents"]
    assert [(i["kind"], i["step"]) for i in evidence["incidents"]] == [(i["kind"], i["step"]) for i in usual]
    assert [(i["kind"], i["step"]) for i in usual] != [(i["kind"], i["step"]) for i in reference(300)]


def test_the_server_refuses_bad_thresholds_before_it_listens_or_reads(tmp_path):
    store = tmp_path / "store"
    for arguments in (
        ["--store", str(store), "--thresholds", '{"bogus": 1}'],
        ["detect", "--thresholds", '{"bogus": 1}'],
        ["--thresholds", '{"bogus": 1}', "detect"],  # the option is taken before the subcommand too
        ["detect", "--thresholds", "[0.5]"],
        ["detect", "--thresholds", "0.5"],
        ["detect", "--thresholds", "null"],
    ):
        done = subprocess.run([str(BINARY), *arguments], input=b"", capture_output=True, timeout=30)
        assert done.returncode == 2, arguments
        assert b"--thresholds" in done.stderr and (b"unknown field `bogus`" in done.stderr or b"expected a JSON object" in done.stderr)
        assert done.stdout == b""
    assert not store.exists()
    # Before the subcommand, a good value is applied, not dropped.
    stream = __import__("tensorquill.server_detect", fromlist=["records_stream"]).records_stream(moe_edges())
    done = subprocess.run([str(BINARY), "--thresholds", '{"starve_window": 7}', "detect"], input=stream, capture_output=True, timeout=30)
    assert done.returncode == 0 and json.loads(done.stdout)["thresholds"]["starve_window"] == 7


def test_an_unknown_engine_and_a_missing_binary_say_so(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="'python' or 'rust'"):
        tensorquill.build_evidence(moe_edges(), detectors="server")
    monkeypatch.delenv(BINARY_ENV)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match=BINARY_ENV):
        tensorquill.build_evidence(moe_edges(), detectors="rust")
    with pytest.raises(RuntimeError, match="is not a file"):
        server_binary(tmp_path / "missing")
    # No records: nothing to judge, and the binary is not needed.
    assert tensorquill.build_evidence([], detectors="rust")["incidents"] == []


def test_the_binary_refuses_two_ranks_and_its_message_is_passed_on():
    rows = [record(s, 1) for s in range(5)] + [{**record(s, 1), "rank": 1} for s in range(5)]
    with pytest.raises(RuntimeError, match="send one rank of one run"):
        server_report(rows)
    with pytest.raises(ValueError, match="pass rank="):
        tensorquill.build_evidence(rows, detectors="rust")
    assert both([r for r in rows if r["rank"] == 1]) == []


def test_a_server_built_from_other_rules_is_refused(monkeypatch):
    import tensorquill.server_detect as module

    real = module.server_report
    change = {}

    def older(records, **kwargs):
        report = real(records, **kwargs)
        report["thresholds"].update(change.get("thresholds", {}))
        report["checked"] = [kind for kind in report["checked"] if kind not in change.get("unchecked", ())]
        return report

    monkeypatch.setattr(module, "server_report", older)
    tensorquill.build_evidence(moe_edges(), detectors="rust")  # unchanged, it is taken
    change["thresholds"] = {"overload_threshold": 0.5}
    with pytest.raises(RuntimeError, match="overload_threshold 0.5, here 0.35"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")
    change.clear()
    change["unchecked"] = ("input_starvation", "attention_collapse")
    with pytest.raises(RuntimeError, match="does not check attention_collapse, input_starvation"):
        tensorquill.build_evidence(moe_edges(), detectors="rust")


# --- the command line -----------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "tensorquill", *args], capture_output=True, text=True, timeout=120)


def test_the_command_line_takes_the_incidents_from_rust(tmp_path):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in everything()), encoding="utf-8")
    plain = run_cli(str(capture), "-o", str(tmp_path / "python.json"))
    rust = run_cli(str(capture), "-o", str(tmp_path / "rust.json"), "--detectors", "rust")
    assert plain.returncode == 0 and rust.returncode == 0, rust.stderr
    written = json.loads((tmp_path / "rust.json").read_text(encoding="utf-8"))
    assert written.pop("detectors") == "rust"
    assert differences(json.loads((tmp_path / "python.json").read_text(encoding="utf-8")), written, "evidence") == []
    # The summary a person reads is the same, line for line, but for the file names and a
    # logged NaN: it is `null` in the server's report, which is strict JSON, and reads
    # "not finite" where Python's own value reads "nan".
    assert "not finite" in rust.stdout and "not finite" not in plain.stdout
    assert plain.stdout.replace("python.json", "X").replace(" nan ", " not finite ") == rust.stdout.replace("rust.json", "X")


def test_the_command_lines_thresholds_go_to_rust(tmp_path):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in everything()), encoding="utf-8")
    options = ("--top-k", "2", "--overload-threshold", "0.5")
    plain = run_cli(str(capture), "-o", str(tmp_path / "python.json"), *options)
    rust = run_cli(str(capture), "-o", str(tmp_path / "rust.json"), "--detectors", "rust", *options)
    assert plain.returncode == 0 and rust.returncode == 0, rust.stderr
    python_evidence = json.loads((tmp_path / "python.json").read_text(encoding="utf-8"))
    written = json.loads((tmp_path / "rust.json").read_text(encoding="utf-8"))
    assert written.pop("detectors") == "rust" and written["overload_threshold"] == 0.5 and set(written["top_k"].values()) == {2}
    assert differences(python_evidence, written, "evidence") == []
    assert python_evidence["incidents"] != tensorquill.build_evidence(everything())["incidents"]


@pytest.mark.parametrize("extra", [("--watch",), ("--rank", "all")])
def test_the_command_line_refuses_what_the_rust_path_cannot_do(tmp_path, extra):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in moe_edges()), encoding="utf-8")
    done = run_cli(str(capture), "--detectors", "rust", *extra)
    assert done.returncode == 2, done.stdout
    assert "rust" in done.stderr.lower()
