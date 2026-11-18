"""Which detectors run, and which binary: the SDK side of the packaging decision.

The author's rules (v3 plan, Decisions: "Packaging and the release plan"), each held here:

- "The binary reports one protocol number, and the SDK checks it."
- "`--watch` requires the Rust binary and stops with the install hint; `--detectors python`
  overrides that. One-shot reads use `auto` and fall back to Python with one warning."
- "Keep the Python detectors": a library call still uses them unless it asks otherwise.

`tests/test_packaging.py` holds the dependency and the `tensorquill-mcp` script.

Most of this file needs neither `pyarrow` nor the server binary: a stand-in program answers
`protocol`, and the lookup beside the interpreter is pointed at a folder of the test's own.
The tests that run the real Rust detectors are skipped without the binary and `pyarrow`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import tensorquill
import tensorquill.evidence as ev
from tensorquill import server_detect as sd

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stand-in binaries are shell scripts")


def real_binary() -> Path | None:
    named = os.environ.get(sd.BINARY_ENV)
    if named:
        return Path(named)
    target = ROOT / "server" / "target"
    built = [p for p in (target / "debug" / sd.BINARY_NAME, target / "release" / sd.BINARY_NAME) if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


REAL = real_binary()
try:
    import pyarrow  # noqa: F401
    HAVE_ARROW = True
except ImportError:
    HAVE_ARROW = False
needs_rust = pytest.mark.skipif(REAL is None or not HAVE_ARROW, reason="needs the server binary and pyarrow")


def collapse(steps: int = 40) -> list[dict]:
    """A router that sends nearly everything to one expert from step 20: `moe_imbalance`."""
    return [
        {"schema_version": 1, "run_id": "auto", "rank": 0, "attempt": 1, "step": step, "ts": 1000.0 + step,
         "moe": {"router": [250] * 8 if step < 20 else [1930, 10, 10, 10, 10, 10, 10, 10]},
         "moe_top_k": {"router": 1}}
        for step in range(steps)
    ]


def stand_in(path: Path, protocol_lines: str, rest: str = "exit 3") -> Path:
    """A program that answers `protocol` with `protocol_lines` (shell) and anything else with `rest`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\nif [ "$1" = protocol ]; then\n{protocol_lines}\nfi\n{rest}\n', encoding="utf-8")
    path.chmod(0o755)
    return path


def speaks(path: Path, number: int) -> Path:
    return stand_in(path, f"echo {number}; exit 0")


def before_the_number(path: Path) -> Path:
    """What clap prints for a subcommand it does not have: a binary from before `protocol`."""
    return stand_in(path, "echo \"error: unrecognized subcommand 'protocol'\" >&2; exit 2")


@pytest.fixture
def nowhere(monkeypatch, tmp_path):
    """No binary by any of the three ways: not named, not beside the interpreter, not on PATH."""
    monkeypatch.delenv(sd.BINARY_ENV, raising=False)
    monkeypatch.setattr(sd, "_beside_interpreter", lambda: None)
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return empty


def run_main(monkeypatch, capsys, *argv: str) -> tuple[int, str, str]:
    """`python -m tensorquill <argv>` in this process: exit status, standard output, standard error."""
    monkeypatch.setattr(sys, "argv", ["tensorquill", *map(str, argv)])
    status = 0
    try:
        ev.main()
    except SystemExit as stop:
        status = stop.code if isinstance(stop.code, int) else 1
    out = capsys.readouterr()
    return status, out.out, out.err


def capture(tmp_path: Path) -> Path:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in collapse()), encoding="utf-8")
    return path


# --- "The binary reports one protocol number, and the SDK checks it." --------------------------


def test_the_server_and_the_sdk_are_built_for_the_same_protocol_number():
    main_rs = ROOT / "server" / "src" / "main.rs"
    if not main_rs.exists():
        pytest.skip("server/ not in this checkout")
    [number] = re.findall(r"^const PROTOCOL: u32 = (\d+);$", main_rs.read_text(encoding="utf-8"), re.M)
    assert int(number) == sd.PROTOCOL, "PROTOCOL in server/src/main.rs and in server_detect.py must be the same"


@pytest.mark.skipif(REAL is None, reason="tensorquill-server is not built")
def test_the_real_binary_reports_the_number_and_is_taken():
    done = subprocess.run([str(REAL), "protocol"], capture_output=True, text=True, timeout=30)
    assert (done.returncode, done.stdout, done.stderr) == (0, f"{sd.PROTOCOL}\n", "")
    assert sd.binary_protocol(REAL) == sd.PROTOCOL
    assert sd.server_binary(REAL) == str(REAL)


def test_a_binary_is_taken_only_when_it_reports_this_number(tmp_path):
    assert sd.server_binary(speaks(tmp_path / "same", sd.PROTOCOL)) == str(tmp_path / "same")
    # A newer binary is refused as an older one is: its reports are not the ones this module reads.
    for other in (sd.PROTOCOL + 1, sd.PROTOCOL - 1):
        with pytest.raises(RuntimeError) as refused:
            sd.server_binary(speaks(tmp_path / f"speaks-{other}", other))
        message = str(refused.value)
        assert f"speaks protocol {other}" in message and f"needs protocol {sd.PROTOCOL}" in message
        assert "pip install --upgrade tensorquill tensorquill-server" in message


@pytest.mark.parametrize("answer", [
    "echo \"error: unrecognized subcommand 'protocol'\" >&2; exit 2",  # from before the number
    "echo one; exit 0",                                               # not a number
    "echo 1.5; exit 0",
    "echo -1; exit 0",
    "exit 0",                                                         # nothing printed
    "echo 1; exit 1",                                                 # a number, and a failure
], ids=["no-subcommand", "a-word", "a-fraction", "negative", "nothing", "failed"])
def test_a_binary_that_reports_no_number_is_refused_with_what_to_install(tmp_path, answer):
    old = stand_in(tmp_path / "old", answer)
    with pytest.raises(RuntimeError) as refused:
        sd.server_binary(old)
    message = str(refused.value)
    assert "does not report a protocol number" in message and f"needs protocol {sd.PROTOCOL}" in message
    assert sd.INSTALL_HINT in message


def test_the_number_is_checked_before_the_binary_is_given_any_records(tmp_path):
    """`server_report` and `server_run_report` ask for the number first: a binary that fails
    the check is never started on `detect`."""
    ran = tmp_path / "ran"
    other = stand_in(tmp_path / "other", f"echo {sd.PROTOCOL + 1}; exit 0", f"touch {ran}; echo '{{\"incidents\": []}}'")
    for call in (sd.server_report, sd.server_run_report):
        with pytest.raises(RuntimeError, match="speaks protocol"):
            call(collapse(), binary=other)
    assert not ran.exists()


def test_a_binary_is_asked_once_and_again_only_when_its_file_changes(tmp_path):
    asked = tmp_path / "asked"
    binary = stand_in(tmp_path / "counted", f"echo x >> {asked}; echo {sd.PROTOCOL}; exit 0")
    for _ in range(3):
        assert sd.server_binary(binary) == str(binary)
    assert asked.read_text() == "x\n"
    # Replaced by another build: the answer kept for the old file is not used for the new one.
    stand_in(binary, f"echo y >> {asked}; echo {sd.PROTOCOL + 7}; exit 0  # another build")
    with pytest.raises(RuntimeError, match=f"speaks protocol {sd.PROTOCOL + 7}"):
        sd.server_binary(binary)
    assert asked.read_text() == "x\ny\n"


def test_a_binary_that_never_answers_is_refused_and_not_waited_for(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "_PROTOCOL_TIMEOUT", 0.3)
    with pytest.raises(RuntimeError, match="no answer in 0.3 s"):
        sd.server_binary(stand_in(tmp_path / "hangs", "exec sleep 30"))


# --- finding the binary: the name given, the environment, beside the interpreter, PATH --------


def test_the_binary_is_looked_for_in_order(tmp_path, monkeypatch):
    on_path = speaks(tmp_path / "path" / sd.BINARY_NAME, sd.PROTOCOL)
    beside = speaks(tmp_path / "venv" / "bin" / sd.BINARY_NAME, sd.PROTOCOL)
    named = speaks(tmp_path / "named" / "server", sd.PROTOCOL)
    given = speaks(tmp_path / "given" / "server", sd.PROTOCOL)
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setenv(sd.BINARY_ENV, str(named))
    monkeypatch.setattr(sd.sysconfig, "get_path", lambda name, *scheme: str(beside.parent))
    assert sd.find_binary(given) == str(given)
    assert sd.find_binary() == sd.server_binary() == str(named)
    monkeypatch.delenv(sd.BINARY_ENV)
    assert sd.find_binary() == sd.server_binary() == str(beside)   # installed with this Python: before PATH
    beside.unlink()
    monkeypatch.setattr(sd.sys, "executable", str(tmp_path / "elsewhere" / "python"))
    assert sd.find_binary() == sd.server_binary() == str(on_path)
    on_path.unlink()
    assert sd.find_binary() is None
    with pytest.raises(RuntimeError) as none:
        sd.server_binary()
    assert sd.INSTALL_HINT in str(none.value) and sd.BINARY_ENV in str(none.value)


def test_beside_the_interpreter_is_its_scripts_folder_its_own_folder_and_the_user_scripts(tmp_path, monkeypatch):
    """Where a `tensorquill-server` wheel puts the binary, on PATH or not: the scripts folder
    of the running interpreter, the folder the interpreter itself is in (a virtual
    environment's `python` is a link there, and is not followed), and `pip install --user`'s."""
    scripts, own, user = tmp_path / "scripts", tmp_path / "venv" / "bin", tmp_path / "user" / "bin"
    for folder in (scripts, own, user):
        folder.mkdir(parents=True)
    base = tmp_path / "base" / "python"
    base.parent.mkdir()
    base.write_text("", encoding="utf-8")
    (own / "python").symlink_to(base)
    monkeypatch.setattr(sd.sys, "executable", str(own / "python"))
    monkeypatch.setattr(sd.sysconfig, "get_path", lambda name, *scheme: str(user if scheme else scripts))
    monkeypatch.delenv(sd.BINARY_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "base"))
    assert sd._beside_interpreter() is None and sd.find_binary() is None
    for folder in (user, own, scripts):  # each found, and the later ones in this loop win over the earlier
        (folder / sd.BINARY_NAME).write_text("", encoding="utf-8")
        assert sd._beside_interpreter() == sd.find_binary() == str(folder / sd.BINARY_NAME)


def test_a_wrong_name_in_the_environment_is_said_and_not_passed_over(tmp_path, monkeypatch):
    speaks(tmp_path / "path" / sd.BINARY_NAME, sd.PROTOCOL)
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.setenv(sd.BINARY_ENV, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="is not a file"):
        sd.server_binary()


# --- `auto`: Rust when it is all there, and otherwise Python, said once ------------------------


def test_auto_is_rust_only_when_the_binary_the_number_and_pyarrow_are_all_there(tmp_path, monkeypatch, nowhere):
    assert sd.resolve_detectors("python") == ("python", None)
    assert sd.resolve_detectors("rust") == ("rust", None)  # asked for: not checked here, refused where it runs
    with pytest.raises(ValueError, match="'auto', 'python' or 'rust'"):
        sd.resolve_detectors("server")

    chosen, why = sd.resolve_detectors("auto")
    assert chosen == "python" and "using the Python detectors" in why and sd.INSTALL_HINT in why

    chosen, why = sd.resolve_detectors("auto", speaks(tmp_path / "newer", sd.PROTOCOL + 1))
    assert chosen == "python" and f"speaks protocol {sd.PROTOCOL + 1}" in why and "pip install --upgrade" in why

    chosen, why = sd.resolve_detectors("auto", before_the_number(tmp_path / "older"))
    assert chosen == "python" and "does not report a protocol number" in why and sd.INSTALL_HINT in why

    good = speaks(tmp_path / "good", sd.PROTOCOL)
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # `import pyarrow` now fails, as when it is not installed
    chosen, why = sd.resolve_detectors("auto", good)
    assert chosen == "python" and "pip install 'tensorquill[flight]'" in why
    if HAVE_ARROW:
        monkeypatch.undo()
        assert sd.resolve_detectors("auto", good) == ("rust", None)


def test_a_library_call_keeps_the_python_detectors_and_starts_no_process(tmp_path, monkeypatch):
    """`build_evidence` and `watch` default to "python" whatever is installed: the same evidence
    on every machine, and nothing started, unless the caller asks. (The MCP server relies on it.)"""
    monkeypatch.setenv(sd.BINARY_ENV, str(speaks(tmp_path / "good", sd.PROTOCOL)))

    def no_process(*args, **kwargs):
        raise AssertionError(f"a process was started: {args}")

    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evidence = tensorquill.build_evidence(collapse())
        path = capture(tmp_path)
        written = ev.watch(path, interval=0.0, polls=1)
    assert "detectors" not in evidence and [i["kind"] for i in evidence["incidents"]] == ["moe_imbalance"]
    assert "detectors" not in json.loads(written.read_text())


def test_auto_in_a_library_call_falls_back_with_exactly_one_warning(tmp_path, nowhere):
    reference = tensorquill.build_evidence(collapse(), detectors="python")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evidence = tensorquill.build_evidence(collapse(), detectors="auto")
    assert evidence == reference
    [warning] = caught
    assert warning.category is RuntimeWarning and sd.INSTALL_HINT in str(warning.message)
    assert warning.filename == __file__  # it points at the caller

    # A watcher decides once, before its first look, and not again at each rebuild.
    path = capture(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with path.open("a", encoding="utf-8") as fh:
            seen = []

            def more(evidence, _):
                seen.append(len(evidence["samples"]))
                fh.write(json.dumps({**collapse()[-1], "step": 40 + len(seen)}) + "\n")
                fh.flush()

            ev.watch(path, interval=0.0, polls=3, detectors="auto", on_update=more)
    assert seen == [40, 41, 42]  # three rebuilds
    assert [str(w.message) for w in caught].count(str(warning.message)) == 1 and len(caught) == 1


@needs_rust
def test_auto_in_a_library_call_is_rust_when_it_is_all_there(monkeypatch):
    monkeypatch.setenv(sd.BINARY_ENV, str(REAL))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auto = tensorquill.build_evidence(collapse(), detectors="auto")
        rust = tensorquill.build_evidence(collapse(), detectors="rust")
    assert auto["detectors"] == "rust" and auto == rust


# --- the command line: one read uses `auto` ------------------------------------------------------


def test_one_read_without_the_binary_uses_python_and_warns_once(tmp_path, monkeypatch, capsys, nowhere):
    path = capture(tmp_path)
    status, out, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "auto.json")
    assert status == 0 and "wrote evidence" in out
    [line] = err.splitlines()  # exactly one warning
    assert line.startswith("tensorquill: warning: using the Python detectors")
    assert sd.INSTALL_HINT in line and "--detectors python" in line
    # `--detectors auto` typed is the same; `--detectors python` is the same evidence without the warning.
    status, _, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "typed.json", "--detectors", "auto")
    assert status == 0 and err.splitlines() == [line]
    status, _, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "python.json", "--detectors", "python")
    assert (status, err) == (0, "")
    auto, typed, python = (json.loads((tmp_path / f"{name}.json").read_text()) for name in ("auto", "typed", "python"))
    assert auto == typed == python and "detectors" not in python
    assert [i["kind"] for i in python["incidents"]] == ["moe_imbalance"]
    # Asked for by name, the Rust detectors are not replaced: the read stops, and writes nothing.
    status, _, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "rust.json", "--detectors", "rust")
    assert status == 2 and sd.INSTALL_HINT in err and not (tmp_path / "rust.json").exists()


def test_one_read_falls_back_on_another_protocol_number_and_on_missing_pyarrow(tmp_path, monkeypatch, capsys, nowhere):
    path = capture(tmp_path)
    monkeypatch.setenv(sd.BINARY_ENV, str(speaks(tmp_path / "newer", sd.PROTOCOL + 1)))
    status, _, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "a.json")
    [line] = err.splitlines()
    assert status == 0 and f"speaks protocol {sd.PROTOCOL + 1}" in line and "pip install --upgrade" in line
    assert "detectors" not in json.loads((tmp_path / "a.json").read_text())

    monkeypatch.setenv(sd.BINARY_ENV, str(speaks(tmp_path / "good", sd.PROTOCOL)))
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    status, _, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "b.json")
    [line] = err.splitlines()
    assert status == 0 and "pip install 'tensorquill[flight]'" in line
    assert "detectors" not in json.loads((tmp_path / "b.json").read_text())


@needs_rust
def test_one_read_with_the_binary_uses_the_rust_detectors_and_says_nothing(tmp_path, monkeypatch, capsys, nowhere):
    path = capture(tmp_path)
    # Found beside the interpreter, with nothing named and nothing on PATH: what an installed wheel gives.
    monkeypatch.setattr(sd, "_beside_interpreter", lambda: str(REAL))
    status, out, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "auto.json")
    assert (status, err) == (0, "") and "wrote evidence" in out
    evidence = json.loads((tmp_path / "auto.json").read_text())
    assert evidence["detectors"] == "rust" and [i["kind"] for i in evidence["incidents"]] == ["moe_imbalance"]
    # Chosen by `auto`, a refusal of the Rust rules is an error, not a quiet change of detectors.
    status, _, err = run_main(monkeypatch, capsys, path, "-o", tmp_path / "bad.json", "--starve-window", "0")
    assert status == 2 and "refuse these thresholds" in err and not (tmp_path / "bad.json").exists()


# --- the command line: `--watch` requires the Rust binary ----------------------------------------


@pytest.mark.parametrize("typed", [(), ("--detectors", "auto"), ("--detectors", "rust")], ids=["default", "auto", "rust"])
def test_watch_without_the_binary_stops_before_the_first_look_with_the_install_hint(
        tmp_path, monkeypatch, capsys, nowhere, typed):
    path = capture(tmp_path)
    monkeypatch.setattr(ev, "watch", lambda *a, **k: pytest.fail("the watcher was started"))
    status, out, err = run_main(monkeypatch, capsys, path, "--watch", *typed)
    assert status == 2 and "watching" not in out and "Traceback" not in err
    assert sd.INSTALL_HINT in err
    if typed != ("--detectors", "rust"):
        assert "--watch needs the Rust detectors" in err and "add --detectors python" in err
    assert not path.with_suffix(".evidence.json").exists()


def test_watch_stops_on_another_protocol_number_and_on_missing_pyarrow(tmp_path, monkeypatch, capsys, nowhere):
    path = capture(tmp_path)
    monkeypatch.setattr(ev, "watch", lambda *a, **k: pytest.fail("the watcher was started"))
    monkeypatch.setenv(sd.BINARY_ENV, str(before_the_number(tmp_path / "older")))
    status, _, err = run_main(monkeypatch, capsys, path, "--watch")
    assert status == 2 and "does not report a protocol number" in err and sd.INSTALL_HINT in err
    monkeypatch.setenv(sd.BINARY_ENV, str(speaks(tmp_path / "good", sd.PROTOCOL)))
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    status, _, err = run_main(monkeypatch, capsys, path, "--watch")
    assert status == 2 and "pip install 'tensorquill[flight]'" in err and "add --detectors python" in err


def test_watch_with_detectors_python_runs_without_the_binary(tmp_path, nowhere):
    """The real command, as a process: it watches, writes the evidence, and is stopped from outside."""
    path = capture(tmp_path)
    env = {**os.environ, "PATH": str(nowhere), "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "sdk" / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    env.pop(sd.BINARY_ENV, None)
    child = subprocess.Popen(
        [sys.executable, "-m", "tensorquill", str(path), "--watch", "--interval", "0.1", "--detectors", "python"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path, env=env,
    )
    try:
        lines = []
        for line in child.stdout:
            lines.append(line)
            if line.startswith("updated "):
                break
    finally:
        child.terminate()
        child.wait(timeout=30)
        err = child.stderr.read()
    assert lines[0].startswith("watching ") and "Traceback" not in err and "warning" not in err
    evidence = json.loads(path.with_suffix(".evidence.json").read_text())
    assert "detectors" not in evidence and [i["kind"] for i in evidence["incidents"]] == ["moe_imbalance"]


def test_watch_asks_the_watcher_for_rust_unless_python_is_typed(tmp_path, monkeypatch, capsys, nowhere):
    path = capture(tmp_path)
    monkeypatch.setenv(sd.BINARY_ENV, str(speaks(tmp_path / "good", sd.PROTOCOL)))
    monkeypatch.setitem(sys.modules, "pyarrow", type(sys)("pyarrow"))  # importable, which is all the check asks
    asked = []
    monkeypatch.setattr(ev, "watch", lambda *a, **options: asked.append(options["detectors"]))
    for typed in ((), ("--detectors", "auto"), ("--detectors", "rust"), ("--detectors", "python")):
        status, out, err = run_main(monkeypatch, capsys, path, "--watch", *typed)
        assert (status, err) == (0, "") and out.startswith("watching ")
    assert asked == ["rust", "rust", "rust", "python"]


def test_the_all_ranks_view_runs_no_detector_and_asks_for_no_binary(tmp_path, monkeypatch, capsys, nowhere):
    """`--rank all` reads each rank's report from its server and judges nothing here, so it is
    not stopped for a binary it would never start, with `--watch` or without."""
    monkeypatch.setattr(sd, "rust_ready", lambda *a, **k: pytest.fail("the binary was looked for"))
    asked = []
    monkeypatch.setattr(ev, "watch", lambda *a, **options: asked.append(options) or tmp_path / "x.json")
    for typed in (("--watch",), ()):
        status, _, err = run_main(monkeypatch, capsys, tmp_path, "--rank", "all", *typed)
        assert (status, err) == (0, "")
    assert [o["rank"] for o in asked] == ["all", "all"]
