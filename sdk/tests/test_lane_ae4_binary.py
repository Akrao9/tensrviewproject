"""Lane AE4: the tests and the Rust detectors find the same server binary.

`conftest.py` names the binary built under `server/target` in `TENSORQUILL_SERVER_BIN` when nothing
else has, so `_server_binary()` in test_flight_e2e.py (which decides whether a test skips) and
`server_detect.find_binary` (which `detectors="rust"` uses and which does not look under
`server/target`) agree. These tests need no built server: they use empty folders and fake files.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import conftest
from tensorquill import server_detect

VAR = conftest.SERVER_BIN_ENV


def build(target: Path, *flavours: str) -> dict[str, Path]:
    made = {}
    for age, flavour in enumerate(flavours):
        binary = target / flavour / "tensorquill-server"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n")
        stamp = time.time() - 1000 + age * 100  # later in the argument list is newer
        os.utime(binary, (stamp, stamp))
        made[flavour] = binary
    return made


def test_nothing_built_sets_nothing_and_chooses_none(tmp_path):
    env: dict[str, str] = {}
    assert conftest.built_server_binary(tmp_path / "target") is None
    assert conftest.name_built_server(env, tmp_path / "target") is None
    assert VAR not in env
    # a target folder that exists but holds no binary is the same
    (tmp_path / "target" / "debug").mkdir(parents=True)
    assert conftest.name_built_server(env, tmp_path / "target") is None
    assert env == {}


def test_unset_variable_gets_the_newer_of_debug_and_release(tmp_path):
    target = tmp_path / "target"
    made = build(target, "release", "debug")  # debug is newer
    env: dict[str, str] = {}
    assert conftest.name_built_server(env, target) == made["debug"]
    assert env[VAR] == str(made["debug"])

    other = tmp_path / "other"
    made = build(other, "debug", "release")  # release is newer
    env = {}
    conftest.name_built_server(env, other)
    assert env[VAR] == str(made["release"])

    only = build(tmp_path / "only", "release")
    env = {}
    conftest.name_built_server(env, tmp_path / "only")
    assert env[VAR] == str(only["release"])


def test_empty_variable_counts_as_unset_as_both_lookups_do(tmp_path):
    made = build(tmp_path / "target", "debug")
    env = {VAR: ""}
    conftest.name_built_server(env, tmp_path / "target")
    assert env[VAR] == str(made["debug"])


@pytest.mark.parametrize("named", ["/somewhere/else/tensorquill-server", "/no/such/file"])
def test_a_variable_already_set_is_left_exactly_as_it_is(tmp_path, named):
    build(tmp_path / "target", "debug", "release")
    env = {VAR: named}
    assert conftest.name_built_server(env, tmp_path / "target") is None
    assert env == {VAR: named}


def test_the_tests_lookup_and_the_conftest_choice_agree(tmp_path, monkeypatch):
    """`_server_binary()` of test_flight_e2e.py, which skips the tests, picks what conftest names."""
    pytest.importorskip("pyarrow")
    import test_flight_e2e

    monkeypatch.setenv(VAR, "placeholder")  # so the teardown puts back whatever the run had
    monkeypatch.delenv(VAR)
    conftest.name_built_server(os.environ)
    assert test_flight_e2e._server_binary() == conftest.built_server_binary()
    if conftest.built_server_binary() is not None:
        assert Path(os.environ[VAR]) == conftest.built_server_binary()
    else:
        assert VAR not in os.environ


def test_the_sdk_lookup_finds_what_conftest_named(tmp_path):
    made = build(tmp_path / "target", "debug")
    env = {}
    conftest.name_built_server(env, tmp_path / "target")
    # `find_binary` reads os.environ; give it this one for the call
    old = os.environ.get(VAR)
    os.environ[VAR] = env[VAR]
    try:
        assert server_detect.find_binary() == str(made["debug"])
    finally:
        if old is None:
            os.environ.pop(VAR, None)
        else:
            os.environ[VAR] = old


def test_a_fresh_pytest_run_names_the_binary_before_a_test_module_is_imported(tmp_path):
    """Run pytest in a child on a copy of conftest with a fake built `server/target`: the module-level
    call must have set the variable by the time a test module reads it."""
    sdk_tests = Path(conftest.__file__).resolve().parent
    root = tmp_path / "checkout"
    (root / "sdk" / "tests").mkdir(parents=True)
    made = build(root / "server" / "target", "debug")
    (root / "sdk" / "tests" / "conftest.py").write_text(
        (sdk_tests / "conftest.py").read_text()
    )
    (root / "sdk" / "tests" / "test_reads.py").write_text(textwrap.dedent(f'''
        import os
        SEEN_AT_IMPORT = os.environ.get("TENSORQUILL_SERVER_BIN")

        def test_seen():
            assert SEEN_AT_IMPORT == {str(made["debug"])!r}
    '''))
    env = {k: v for k, v in os.environ.items() if k != VAR}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           str(root / "sdk" / "tests" / "test_reads.py")],
                          cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    env[VAR] = "/named/by/the/user"
    (root / "sdk" / "tests" / "test_reads.py").write_text(textwrap.dedent('''
        import os

        def test_untouched():
            assert os.environ["TENSORQUILL_SERVER_BIN"] == "/named/by/the/user"
    '''))
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           str(root / "sdk" / "tests" / "test_reads.py")],
                          cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
