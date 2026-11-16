"""Cheap packaging checks: no network, no build, no import of `tensorquill`.

They pin what a release must carry: the dual `Apache-2.0 OR MIT` license in both packages with
both license files shipped, the sdist kept to source, and no author or URL metadata that the
project owner has not chosen yet. The wheel and sdist contents themselves are checked by the
build job (see the packaging notes), because that needs a real `python -m build`.
"""

import hashlib
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which `requires-python` still allows
    tomllib = pytest.importorskip("tomli")

SDK = Path(__file__).resolve().parents[1]
ROOT = SDK.parent
SERVER = ROOT / "server"

LICENSE_EXPRESSION = "Apache-2.0 OR MIT"
LICENSE_FILES = ["LICENSE-APACHE", "LICENSE-MIT"]
# SHA-256 of the unmodified Apache-2.0 text served at https://www.apache.org/licenses/LICENSE-2.0.txt
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
MIT_COPYRIGHT = "Copyright (c) 2026 TensorQuill contributors"

needs_server = pytest.mark.skipif(not (SERVER / "Cargo.toml").exists(), reason="server/ not in this checkout")


def _toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _project() -> dict:
    return _toml(SDK / "pyproject.toml")["project"]


def _read(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("name", LICENSE_FILES)
@pytest.mark.parametrize("where", ["root", "sdk", "server"])
def test_license_file_exists_and_is_not_empty(where, name):
    directory = {"root": ROOT, "sdk": SDK, "server": SERVER}[where]
    if where == "server" and not SERVER.exists():
        pytest.skip("server/ not in this checkout")
    path = directory / name
    assert path.is_file(), f"{path} is missing, so the package built from {directory} would not ship it"
    assert path.stat().st_size > 500, f"{path} is empty or truncated"


def test_apache_text_is_the_unmodified_original():
    text = _read(ROOT / "LICENSE-APACHE")
    assert hashlib.sha256(text).hexdigest() == APACHE_SHA256, "LICENSE-APACHE differs from the apache.org text"


def test_mit_text_uses_the_project_copyright_line():
    text = _read(ROOT / "LICENSE-MIT").decode()
    assert text.startswith("MIT License")
    assert MIT_COPYRIGHT in text
    assert "Permission is hereby granted, free of charge" in text
    assert len(re.findall(r"^Copyright \(c\)", text, flags=re.M)) == 1


@pytest.mark.parametrize("name", LICENSE_FILES)
def test_package_license_copies_match_the_root_ones(name):
    root = (ROOT / name).read_bytes()
    assert (SDK / name).read_bytes() == root, f"sdk/{name} drifted from the root copy"
    if SERVER.exists():
        assert (SERVER / name).read_bytes() == root, f"server/{name} drifted from the root copy"


def test_sdk_license_is_the_dual_spdx_expression():
    project = _project()
    assert project["license"] == LICENSE_EXPRESSION
    assert project["license-files"] == LICENSE_FILES


def test_sdk_has_no_license_classifier():
    # Setuptools rejects a `License ::` classifier next to a license expression (PEP 639).
    assert not [c for c in _project().get("classifiers", []) if c.startswith("License ::")]


def test_sdk_build_backend_understands_pep_639():
    requires = _toml(SDK / "pyproject.toml")["build-system"]["requires"]
    versions = [int(m.group(1)) for r in requires if (m := re.fullmatch(r"setuptools\s*>=\s*(\d+)(?:\.\S*)?", r.strip()))]
    assert versions and max(versions) >= 77, "license expressions and license-files need setuptools>=77"


def test_sdk_readme_is_declared_and_present():
    assert _project()["readme"] == "README.md"
    assert (SDK / "README.md").is_file()


def test_sdist_is_kept_to_source():
    manifest = (SDK / "MANIFEST.in").read_text().splitlines()
    for line in ("prune tests", "prune examples"):
        assert line in manifest, f"MANIFEST.in must contain '{line}': the sdist would ship tests without their conftest"


@needs_server
def test_server_license_matches_the_sdk():
    package = _toml(SERVER / "Cargo.toml")["package"]
    assert package["license"] == LICENSE_EXPRESSION
    assert "license-file" not in package, "`license` and `license-file` together confuse the crates.io license check"


@needs_server
def test_server_crate_ships_both_license_files():
    package = _toml(SERVER / "Cargo.toml")["package"]
    include = package.get("include")
    if include is not None:  # without `include`, cargo packages every tracked file
        for name in LICENSE_FILES:
            assert name in include, f"{name} is missing from [package] include, so `cargo package` would drop it"


@needs_server
def test_server_crate_metadata_is_present():
    package = _toml(SERVER / "Cargo.toml")["package"]
    assert package["description"].strip()
    assert package["keywords"] and len(package["keywords"]) <= 5
    assert package["categories"] and len(package["categories"]) <= 5
    assert "readme" not in package or (SERVER / package["readme"]).is_file()


def test_no_contact_metadata_the_owner_has_not_chosen():
    """Authors and URLs are the project owner's call. Delete or relax this once they are chosen."""
    project = _project()
    for key in ("authors", "maintainers", "urls", "optional-urls"):
        assert key not in project, f"[project] {key} is set; the owner picks it at first publish"
    texts = {"sdk/pyproject.toml": (SDK / "pyproject.toml").read_text()}
    if (SERVER / "Cargo.toml").exists():
        package = _toml(SERVER / "Cargo.toml")["package"]
        for key in ("authors", "repository", "homepage", "documentation"):
            assert key not in package, f"server [package] {key} is set; the owner picks it at first publish"
        texts["server/Cargo.toml"] = (SERVER / "Cargo.toml").read_text()
    for name, text in texts.items():
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text), f"{name} contains an email address"
