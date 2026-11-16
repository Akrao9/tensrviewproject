"""`--rank all` with `--detectors rust`: accepted, and the same evidence.

The all-ranks view takes one source, a run folder in a server's store, and runs no detector:
every rank's incidents are read from the report its server wrote, which the Rust detectors
made. So the option has nothing to switch there. What is held here:

- every test of `test_watch_all_ranks.py` passes again with the option given, through
  `watch(..., rank="all")` and through the command line, with no server binary to be found
  by the watcher and no part read;
- the evidence is equal, key for key, with and without the option;
- an engine that does not exist is refused.

Needs `pyarrow` (the `flight` extra) and the server binary to make the stores
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill.evidence import watch  # noqa: E402
from tensorquill.server_detect import BINARY_ENV  # noqa: E402

import test_watch_all_ranks as base  # noqa: E402
from test_all_ranks_look import mixed_run  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402
from test_watch_all_ranks import parts_are_never_read, servers_are_stopped  # noqa: E402,F401  (autouse here too)

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

GIVEN: list[str] = []  # how the option reached the view in the test that just ran
PLAIN: dict = {}  # the command line as `test_watch_all_ranks.py` has it, with nothing added


@pytest.fixture(autouse=True)
def the_rust_option_is_given(monkeypatch, tmp_path):
    """Every `watch(..., rank="all")` and every `--rank all` command of the tests taken from
    `test_watch_all_ranks.py` gets the option, and the watcher is given no binary to find."""
    monkeypatch.setenv(BINARY_ENV, str(tmp_path / "no-such-binary"))
    real_watch, real_cli = base.watch, base.tensorquill
    GIVEN.clear()
    PLAIN["cli"] = real_cli

    def with_option(source, *args, **options):
        if options.get("rank") == "all":
            options.setdefault("detectors", "rust")
            GIVEN.append("watch")
        return real_watch(source, *args, **options)

    def cli_with_option(*args: str, cwd: Path):
        if "all" in args:
            args = (*args, "--detectors", "rust")
            GIVEN.append("cli")
        return real_cli(*args, cwd=cwd)

    monkeypatch.setattr(base, "watch", with_option)
    monkeypatch.setattr(base, "tensorquill", cli_with_option)


# Every test of the all-ranks view, collected here a second time to run with the option.
TAKEN = sorted(name for name in dir(base) if name.startswith("test_"))
globals().update({name: getattr(base, name) for name in TAKEN})


def test_the_whole_all_ranks_file_was_taken_and_the_option_reaches_it(tmp_path):
    assert len(TAKEN) >= 32 and "test_ranks_that_are_all_current_with_no_incidents_are_said_to_be_clean" in TAKEN
    run = base.healthy_run(tmp_path, 2)
    base.look(run, tmp_path)
    assert GIVEN == ["watch"]
    assert base.tensorquill(str(run.folder), "--rank", "all", cwd=tmp_path).returncode == 0
    assert GIVEN == ["watch", "cli"]
    assert run.server.terminate() == 0


def test_the_evidence_is_the_same_with_and_without_the_option(tmp_path):
    run = mixed_run(tmp_path)

    def one_look(name: str, **options) -> dict:
        path = watch(run.folder, tmp_path / name, interval=0.0, polls=1, rank="all", **options)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence.pop("written_at_ms")
        return evidence

    python, rust = one_look("python.json", detectors="python"), one_look("rust.json", detectors="rust")
    assert python == rust == one_look("default.json")
    assert rust["judged_ranks"] == [0, 1, 2, 3] and len(rust["incident_groups"]) == 2 and rust["run_state"]["state"] == "unhealthy"
    assert "detectors" not in rust  # nothing ran here, so nothing says an engine was chosen

    # The command line, with thresholds, a compact file and one look.
    for name, extra in (("cli-python.json", ()), ("cli-rust.json", ("--detectors", "rust"))):
        done = PLAIN["cli"](str(run.folder), "--rank", "all", "-o", str(tmp_path / name), *extra,
                            "--overload-threshold", "0.35", "--compact", cwd=tmp_path)
        assert done.returncode == 0, done.stderr
    written = [json.loads((tmp_path / name).read_text()) for name in ("cli-python.json", "cli-rust.json")]
    for evidence in written:
        evidence.pop("written_at_ms")
    assert written[0] == written[1] and "compact" in written[1] and written[1]["incident_groups"] == rust["incident_groups"]


def test_an_engine_that_does_not_exist_is_refused(tmp_path):
    run = mixed_run(tmp_path)
    with pytest.raises(ValueError, match="'python' or 'rust'"):
        watch(run.folder, tmp_path / "x.json", rank="all", polls=1, detectors="server")
    assert not (tmp_path / "x.json").exists()
