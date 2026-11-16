"""`build_evidence(detectors="rust")` against the Python detectors, on the same records.

The Rust path runs `tensorquill-server detect` on the records in hand and takes the incidents
of the server's kinds from its report; everything else in the evidence is still made in
Python. Both paths must give the same evidence: incidents, `not_checked` and the run state
compared as the server parity tests compare them (numbers to 1e-12, a NaN as `null`), the rest
exactly. The two places where they differ on purpose are pinned at the end.

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


def both(records) -> list[str]:
    """Where the Rust path's evidence differs from the Python path's."""
    python = tensorquill.build_evidence(records)
    rust = tensorquill.build_evidence(records, detectors="rust")
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


@pytest.mark.parametrize("option", [
    {"overload_threshold": 0.5}, {"drift_tolerance": 1e-3}, {"starve_window": 10}, {"starve_ratio": 0.5}, {"top_k": 2},
])
def test_other_thresholds_are_refused_because_the_rust_rules_take_none(option):
    with pytest.raises(ValueError, match=next(iter(option))):
        tensorquill.build_evidence(moe_edges(), detectors="rust", **option)
    tensorquill.build_evidence(moe_edges(), **option)  # the Python path still takes them


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


@pytest.mark.parametrize("extra", [("--watch",), ("--rank", "all"), ("--top-k", "2")])
def test_the_command_line_refuses_what_the_rust_path_cannot_do(tmp_path, extra):
    capture = tmp_path / "run.jsonl"
    capture.write_text("".join(json.dumps(r) + "\n" for r in moe_edges()), encoding="utf-8")
    done = run_cli(str(capture), "--detectors", "rust", *extra)
    assert done.returncode == 2, done.stdout
    assert "rust" in done.stderr.lower() or "top_k" in done.stderr
