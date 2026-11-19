"""Lane U8 (Nov 19, 2026): release checks that could not fail.

- X05: `sdk/packaging/check_archives.py` passed a wheel with its METADATA removed.
- RS18: `server/packaging/check_wheel.py` read only the platform field of a wheel's file name.
- X04: `sdk/examples/benchmark_detectors.py` deleted a `--store` of another shape, with whatever
  else was in it; `server/examples/simulated_ranks.sh` deleted an earlier result directory of the
  same label and exited 0 after its load client or summary failed.

Every archive here is built by the test, and every delete path runs only inside the test's own
temporary directories. Nothing needs the server binary, torch or pyarrow.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "sdk"
CHECK_ARCHIVES = SDK / "packaging" / "check_archives.py"
CHECK_WHEEL = ROOT / "server" / "packaging" / "check_wheel.py"
DRIVER = ROOT / "server" / "examples" / "simulated_ranks.sh"
LICENSES = {name: (SDK / name).read_bytes() for name in ("LICENSE-APACHE", "LICENSE-MIT")}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- X05: the SDK's wheel and sdist ---------------------------------------------------------

SDK_METADATA = (
    "Metadata-Version: 2.4\nName: tensorquill\nVersion: 0.0.1\nLicense-Expression: Apache-2.0 OR MIT\n"
    "License-File: LICENSE-APACHE\nLicense-File: LICENSE-MIT\nRequires-Dist: torch>=2.6\n"
    "Requires-Dist: tensorquill-server>=0.0.1,<0.1\n\nThe description.\n"
)
SDK_INFO = "tensorquill-0.0.1.dist-info"


def sdk_archives(folder: Path, metadata: list[str] | None = None) -> Path:
    """A wheel and an sdist shaped as `python -m build` writes them; `metadata` lists the paths
    the wheel holds the metadata record at (default: the one right place)."""
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "tensorquill-0.0.1-py3-none-any.whl", "w") as wheel:
        wheel.writestr("tensorquill/__init__.py", "")
        for path in [f"{SDK_INFO}/METADATA"] if metadata is None else metadata:
            wheel.writestr(path, SDK_METADATA)
        wheel.writestr(f"{SDK_INFO}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr(f"{SDK_INFO}/entry_points.txt", "[console_scripts]\ntensorquill = tensorquill.evidence:main\n"
                                                       "tensorquill-mcp = tensorquill.mcp_server:main\n")
        for name, data in LICENSES.items():
            wheel.writestr(f"{SDK_INFO}/licenses/{name}", data)
        wheel.writestr(f"{SDK_INFO}/RECORD", "")
    with tarfile.open(folder / "tensorquill-0.0.1.tar.gz", "w:gz") as sdist:
        files = {**LICENSES, "README.md": b"readme", "pyproject.toml": b"[project]", "PKG-INFO": SDK_METADATA.encode(),
                 "src/tensorquill/__init__.py": b""}
        for name, data in files.items():
            info = tarfile.TarInfo(f"tensorquill-0.0.1/{name}")
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))
    return folder


def check_archives(folder: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(CHECK_ARCHIVES), str(folder)], capture_output=True, text=True,
                          timeout=60)


def test_sdk_archives_as_built_pass(tmp_path):
    result = check_archives(sdk_archives(tmp_path / "dist"))
    assert result.returncode == 0 and result.stdout.endswith("archive contents OK\n"), result.stdout


@pytest.mark.filterwarnings("ignore:Duplicate name")  # the duplicate record is the point
@pytest.mark.parametrize("metadata,found", [
    ([], 0),  # the audit's case: METADATA removed
    ([f"{SDK_INFO}/METADATA", f"{SDK_INFO}/METADATA"], 2),
    ([f"{SDK_INFO}/sub/METADATA"], 0),  # a record pip never reads
])
def test_a_wheel_without_exactly_one_metadata_record_fails(tmp_path, metadata, found):
    result = check_archives(sdk_archives(tmp_path / "dist", metadata))
    assert result.returncode == 1, result.stdout
    assert f"FAIL: wheel: expected exactly one {SDK_INFO}/METADATA, found {found}" in result.stdout
    assert "archive contents OK" not in result.stdout


def test_the_package_job_blocks_the_workflow():
    """The plan kept `continue-on-error` until the job had run once; it passed at 3124fd2 and e693e2f."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = workflow.split("\n  package:\n", 1)[1].split("\n  gpu:\n", 1)[0]
    assert "check_archives.py" in job and "continue-on-error" not in job


# --- RS18: the server's wheel file name -----------------------------------------------------

check_wheel_module = _load("lane_u8_check_wheel", CHECK_WHEEL)
VERSION = check_wheel_module.crate_version()
MANYLINUX = "manylinux_2_17_x86_64.manylinux2014_x86_64"  # maturin's compressed tag set


def server_wheel(folder: Path, file_name: str | None = None, *, platform: str = "any", name: str = "tensorquill-server",
                 version: str = VERSION, dist_info: str | None = None, tags: list[str] | None = None,
                 records: tuple[str, ...] = ("METADATA", "WHEEL")) -> Path:
    """A wheel of the server as maturin (or `fallback_wheel.py`, for `any`) writes it, with one field changeable."""
    folder.mkdir(parents=True, exist_ok=True)
    dist_info = dist_info or f"tensorquill_server-{version}.dist-info"
    tags = tags or [f"py3-none-{p}" for p in platform.split(".")]
    path = folder / (file_name or f"tensorquill_server-{version}-py3-none-{platform}.whl")
    with zipfile.ZipFile(path, "w") as wheel:
        for record in records:
            if record == "METADATA":
                wheel.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
                                                        "License-Expression: Apache-2.0 OR MIT\n"
                                                        "License-File: LICENSE-APACHE\nLicense-File: LICENSE-MIT\n\n")
            else:
                wheel.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                                                     + "".join(f"Tag: {t}\n" for t in tags))
        for license_name, data in LICENSES.items():
            wheel.writestr(f"{dist_info}/licenses/{license_name}", data)
        if platform != "any":
            program = zipfile.ZipInfo(f"{dist_info[: -len('.dist-info')]}.data/scripts/tensorquill-server")
            program.external_attr = 0o755 << 16
            wheel.writestr(program, b"\x7fELF not a real program")
    return path


@pytest.fixture
def no_objdump(monkeypatch):
    """The program in these wheels is a stand-in, so the GLIBC reading (Linux CI has objdump) is off."""
    monkeypatch.setattr(check_wheel_module.shutil, "which", lambda name: None)


@pytest.mark.parametrize("platform,checked_as", [
    ("any", "any"),
    (MANYLINUX, "manylinux_2_17_x86_64"),
    ("macosx_11_0_arm64", "macosx_arm64"),
])
def test_the_release_wheels_names_pass(tmp_path, capsys, no_objdump, platform, checked_as):
    path = server_wheel(tmp_path, platform=platform)
    check_wheel_module.check_wheel(path, checked_as)
    assert capsys.readouterr().out.endswith(f"{path.name}: ok ({checked_as})\n")


@pytest.mark.parametrize("file_name,message", [
    ("wrongproject-{v}-py3-none-any.whl", "file name distribution 'wrongproject' is not METADATA's Name"),
    ("tensorquill_server-9.9-py3-none-any.whl", "file name version '9.9' is not METADATA's Version"),
    ("tensorquill_server-{v}-1-py3-none-any.whl", "file name has the build tag 1"),
    ("tensorquill_server-{v}-cp311-none-any.whl", "file name interpreter tags ['cp311'] disagree with WHEEL's ['py3']"),
    ("tensorquill_server-{v}-py3-cp311-any.whl", "file name ABI tags ['cp311'] disagree with WHEEL's ['none']"),
    ("tensorquill_server-{v}-py3-none-manylinux_2_17_x86_64.whl",
     "file name platform tags ['manylinux_2_17_x86_64'] disagree with WHEEL's ['any']"),
    ("tensorquill_server-{v}-py3-none.whl", "not a wheel file name"),
    # The audit's wheel: every field but the platform is wrong, and it printed `ok (any)`.
    ("wrongproject-9.9-cp311-cp311-any.whl", "file name distribution 'wrongproject'"),
])
def test_each_field_of_the_file_name_is_compared(tmp_path, file_name, message):
    path = server_wheel(tmp_path, file_name.format(v=VERSION))
    with pytest.raises(SystemExit) as refused:
        check_wheel_module.check_wheel(path, "any")
    assert message in str(refused.value), refused.value


def test_the_audits_wheel_fails_from_the_command_line(tmp_path):
    path = server_wheel(tmp_path, "wrongproject-9.9-cp311-cp311-any.whl")
    result = subprocess.run([sys.executable, "-I", str(CHECK_WHEEL), "wheel", str(path), "any"], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 1 and "ok (any)" not in result.stdout, result
    assert "file name distribution 'wrongproject'" in result.stderr


@pytest.mark.filterwarnings("ignore:Duplicate name")
@pytest.mark.parametrize("change,message", [
    ({"dist_info": "other-0.0.1.dist-info"}, "dist-info directory other-0.0.1.dist-info does not match"),
    ({"dist_info": "tensorquill_server-0.0.2.dist-info"}, "dist-info directory tensorquill_server-0.0.2.dist-info"),
    ({"records": ("WHEEL",)}, "expected exactly one tensorquill_server-{v}.dist-info/METADATA, found 0"),
    ({"records": ("METADATA", "WHEEL", "WHEEL")}, "expected exactly one tensorquill_server-{v}.dist-info/WHEEL, found 2"),
    # WHEEL claims two platforms, the file name one: the name is a smaller set than the wheel says.
    ({"platform": "manylinux_2_17_x86_64", "tags": ["py3-none-manylinux_2_17_x86_64", "py3-none-manylinux2014_x86_64"]},
     "file name platform tags ['manylinux_2_17_x86_64'] disagree"),
])
def test_the_name_must_match_the_wheels_own_records(tmp_path, no_objdump, change, message):
    platform = change.get("platform", "any")
    path = server_wheel(tmp_path, **change)
    with pytest.raises(SystemExit) as refused:
        check_wheel_module.check_wheel(path, platform)
    assert message.format(v=VERSION) in str(refused.value), refused.value


# --- X04: the detector benchmark's --store --------------------------------------------------

sys.path.insert(0, str(SDK / "examples"))
import benchmark_detectors as bd  # noqa: E402  (its module level is the standard library only)

SHAPE = {"blocks": 2, "experts": 4, "top_k": 1}


@pytest.fixture
def no_capture(monkeypatch):
    """Building a store trains a model; these tests only need to know whether one was built."""
    built = []
    monkeypatch.setattr(bd, "capture_template", lambda **shape: {"shape": shape, "attempt": 0})

    def write_store(store, template, ranks, steps, workers):
        store.mkdir(parents=True, exist_ok=True)
        (store / "template.json").write_text(json.dumps(template))
        manifest = {"ranks": ranks, "steps": steps, "shape": template["shape"]}
        (store / "manifest.json").write_text(json.dumps(manifest))
        built.append(store)
        return manifest
    monkeypatch.setattr(bd, "write_store", write_store)
    return built


def stored(store: Path, shape: dict, ranks: int = 4, steps: int = 600) -> Path:
    store.mkdir(parents=True)
    (store / "template.json").write_text(json.dumps({"shape": shape, "attempt": 0}))
    (store / "manifest.json").write_text(json.dumps({"shape": shape, "ranks": ranks, "steps": steps}))
    (store / "notes-the-user-keeps-here.txt").write_text("not the benchmark's")
    return store


@pytest.mark.parametrize("needs", [
    {"shape": {**SHAPE, "experts": 8}},  # another shape
    {"ranks": 8},  # more ranks than stored
    {"steps": 1200},  # more steps than stored
])
def test_a_store_that_does_not_fit_is_refused_and_kept(tmp_path, no_capture, needs):
    store = stored(tmp_path / "store", SHAPE)
    before = sorted(p.name for p in store.iterdir())
    with pytest.raises(SystemExit) as refused:
        bd.prepare_store(store, needs.get("shape", SHAPE), needs.get("ranks", 4), needs.get("steps", 600), 1)
    assert "Nothing was deleted" in str(refused.value)
    assert sorted(p.name for p in store.iterdir()) == before and not no_capture
    assert (store / "notes-the-user-keeps-here.txt").read_text() == "not the benchmark's"


def test_a_directory_that_is_not_a_store_is_refused_and_kept(tmp_path, no_capture):
    store = tmp_path / "somewhere"
    store.mkdir()
    (store / "results.json").write_text("{}")
    (store / "manifest.json").write_text("not json")  # with a template, an unreadable manifest is refused too
    (store / "template.json").write_text("{}")
    with pytest.raises(SystemExit) as refused:
        bd.prepare_store(store, SHAPE, 4, 600, 1)
    assert "unreadable manifest.json" in str(refused.value)
    (store / "template.json").unlink()
    with pytest.raises(SystemExit) as refused:
        bd.prepare_store(store, SHAPE, 4, 600, 1)
    assert "is not empty and holds no benchmark store" in str(refused.value)
    assert sorted(p.name for p in store.iterdir()) == ["manifest.json", "results.json"] and not no_capture


def test_a_new_or_empty_store_is_built_and_a_fitting_one_reused(tmp_path, no_capture):
    for store in (tmp_path / "new", tmp_path / "empty"):
        if store.name == "empty":
            store.mkdir()
        _, manifest = bd.prepare_store(store, SHAPE, 4, 600, 1)
        assert manifest["ranks"] == 4 and no_capture[-1] == store
    big = stored(tmp_path / "big", SHAPE, ranks=8, steps=1200)
    template, manifest = bd.prepare_store(big, SHAPE, 4, 600, 1)
    assert manifest["ranks"] == 8 and no_capture[-1] != big


# --- X04: the simulated-ranks driver --------------------------------------------------------

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
# The driver measures with macOS's `/usr/bin/time -l` and `sysctl vm.swapusage`; it is a macOS script.
needs_macos = pytest.mark.skipif(sys.platform != "darwin" or shutil.which("nc") is None,
                                 reason="the driver's measurements are macOS's (/usr/bin/time -l, sysctl)")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def fake_build(tmp_path: Path, client: str) -> Path:
    """Stand-ins for the release build: a server that listens until SIGTERM and writes its PID,
    and a load client running `client` (shell)."""
    bins = tmp_path / "bin"
    (bins / "examples").mkdir(parents=True)
    server = bins / "tensorquill-server"
    server.write_text(f"""#!/bin/bash
exec {sys.executable} -I -c '
import os, signal, socket, sys
host, port = sys.argv[sys.argv.index("--listen") + 1].rsplit(":", 1)
listener = socket.socket(); listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind((host, int(port))); listener.listen()
open(os.environ["FAKE_PID_FILE"], "w").write(str(os.getpid()))
print("listening", flush=True)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
while True:
    listener.accept()[0].close()
' "$@"
""")
    (bins / "examples" / "live_streams").write_text(f"#!/bin/bash\n{client}\n")
    for program in (server, bins / "examples" / "live_streams"):
        program.chmod(0o755)
    return bins


def drive(tmp_path: Path, bins: Path, out: Path, label: str = "r2") -> subprocess.CompletedProcess:
    source = tmp_path / "source"
    (source / "runs" / "bench" / "rank=0" / "attempt=0").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OUT=str(out), BIN_DIR=str(bins), PORT=str(free_port()),
               FAKE_PID_FILE=str(tmp_path / "server.pid"))
    return subprocess.run(["bash", str(DRIVER), label, str(source), "2"], env=env, capture_output=True, text=True,
                          timeout=120)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@needs_bash
def test_an_earlier_result_directory_is_refused_not_deleted(tmp_path):
    out = tmp_path / "results"
    earlier = out / "r2"
    earlier.mkdir(parents=True)
    (earlier / "client-first.txt").write_text("the earlier run's result")
    result = drive(tmp_path, fake_build(tmp_path, "exit 0"), out)
    assert result.returncode == 2, result
    assert "is not empty" in result.stderr and "nothing was deleted" in result.stderr
    assert (earlier / "client-first.txt").read_text() == "the earlier run's result"
    assert not (tmp_path / "server.pid").exists()  # refused before anything started


@needs_bash
@needs_macos
def test_a_failed_load_client_fails_the_run_stops_the_server_and_keeps_the_logs(tmp_path):
    out = tmp_path / "results"
    result = drive(tmp_path, fake_build(tmp_path, "echo 'client starting'; exit 3"), out)
    assert result.returncode == 3, result
    assert "the load client (first) failed (status 3); logs kept in" in result.stderr
    assert "results in" not in result.stdout
    pid = int((tmp_path / "server.pid").read_text())
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(pid), "the server the driver started is still running"
    run = out / "r2"
    assert "client starting" in (run / "client-first.txt").read_text()
    assert (run / "server-first.log").exists() and (run / "store").is_dir()


@needs_bash
@needs_macos
def test_a_failed_summary_fails_the_run(tmp_path):
    out = tmp_path / "results"
    result = drive(tmp_path, fake_build(tmp_path, "echo '{not json'"), out)  # the summary reads the client's JSON line
    assert result.returncode == 1, result
    assert "the summary failed (status 1)" in result.stderr and "results in" not in result.stdout
    assert (out / "r2" / "store").is_dir()  # kept with the logs


@needs_bash
@needs_macos
def test_a_completed_run_exits_0_and_stops_the_server(tmp_path):
    out = tmp_path / "results"
    result = drive(tmp_path, fake_build(tmp_path, """echo '{"seconds": 0.1}'"""), out)
    assert result.returncode == 0, result
    assert f"results in {out / 'r2'}" in result.stdout
    assert not alive(int((tmp_path / "server.pid").read_text()))
    assert not (out / "r2" / "store").exists()  # made by this run, removed without KEEP=1
    # The same label again is refused, and the first run's results stay.
    again = drive(tmp_path, fake_build(tmp_path / "again", "exit 0"), out)
    assert again.returncode == 2 and (out / "r2" / "client-first.txt").exists()


@needs_bash
@needs_macos
def test_a_busy_port_is_refused_before_anything_is_made(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        source = tmp_path / "source"
        source.mkdir()
        env = dict(os.environ, OUT=str(tmp_path / "results"), BIN_DIR=str(fake_build(tmp_path, "exit 0")),
                   PORT=str(busy.getsockname()[1]), FAKE_PID_FILE=str(tmp_path / "server.pid"))
        result = subprocess.run(["bash", str(DRIVER), "r2", str(source), "2"], env=env, capture_output=True, text=True,
                                timeout=60)
    assert result.returncode == 1 and "already listens on port" in result.stderr, result
    assert not (tmp_path / "results" / "r2").exists()  # the same label can be run again at once
