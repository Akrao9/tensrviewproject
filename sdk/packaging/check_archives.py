"""Check what a built tensorquill wheel and sdist actually contain.

    python packaging-check-archives.py <dist-dir>

Fails (exit 1) if tests, examples, experiments or bytecode ship, if either license file is missing
or differs between the two archives, if the metadata lacks the dual license expression, or if it
carries author, maintainer or URL fields the project owner has not chosen yet.
"""
import re
import sys
import tarfile
import zipfile
from pathlib import Path

LICENSES = ["LICENSE-APACHE", "LICENSE-MIT"]
FORBIDDEN_TOP = {"tests", "test", "examples", "experiments", "server", "docs"}
problems: list[str] = []


def check(ok: bool, message: str) -> None:
    if not ok:
        problems.append(message)


def check_metadata(text: str, where: str) -> None:
    header = text.split("\n\n", 1)[0]
    check(re.search(r"^License-Expression: Apache-2\.0 OR MIT$", header, re.M) is not None, f"{where}: License-Expression is not 'Apache-2.0 OR MIT'")
    check(sorted(re.findall(r"^License-File: (.+)$", header, re.M)) == LICENSES, f"{where}: License-File lines are not exactly {LICENSES}")
    check(not re.search(r"^(Author|Author-email|Maintainer|Maintainer-email|Project-URL|Home-page|Download-URL):", header, re.M), f"{where}: has author, maintainer or URL fields")
    check(not re.search(r"^Classifier: License ::", header, re.M), f"{where}: has a License classifier")


dist = Path(sys.argv[1])
wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
check(len(wheels) == 1 and len(sdists) == 1, f"expected exactly one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}")
wheel_licenses: dict[str, bytes] = {}
sdist_licenses: dict[str, bytes] = {}

if wheels:
    with zipfile.ZipFile(wheels[0]) as z:
        names = z.namelist()
        print(f"wheel {wheels[0].name}: {len(names)} files")
        distinfo = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        check(len(distinfo) == 1, f"wheel: expected one dist-info directory, found {distinfo}")
        for n in names:
            top = n.split("/", 1)[0]
            check(top == "tensorquill" or top in distinfo, f"wheel: unexpected top-level entry {n}")
            check("__pycache__" not in n and not n.endswith((".pyc", ".pyo")), f"wheel: bytecode shipped: {n}")
        for lic in LICENSES:
            member = f"{next(iter(distinfo))}/licenses/{lic}"
            check(member in names, f"wheel: {member} missing")
            if member in names:
                wheel_licenses[lic] = z.read(member)
        meta = [n for n in names if n.endswith(".dist-info/METADATA")]
        if meta:
            check_metadata(z.read(meta[0]).decode(), "wheel METADATA")

if sdists:
    with tarfile.open(sdists[0]) as t:
        names = t.getnames()
        print(f"sdist {sdists[0].name}: {len(names)} entries")
        root = names[0].split("/", 1)[0]
        for n in names:
            parts = n.split("/")
            check(parts[0] == root, f"sdist: entry outside {root}/: {n}")
            check(len(parts) < 2 or parts[1] not in FORBIDDEN_TOP, f"sdist: ships {n}")
            check("__pycache__" not in n and not n.endswith((".pyc", ".pyo")), f"sdist: bytecode shipped: {n}")
        for needed in [*LICENSES, "README.md", "pyproject.toml", "PKG-INFO", "src/tensorquill/__init__.py"]:
            check(f"{root}/{needed}" in names, f"sdist: {needed} missing")
        for lic in LICENSES:
            if f"{root}/{lic}" in names:
                sdist_licenses[lic] = t.extractfile(f"{root}/{lic}").read()
        if f"{root}/PKG-INFO" in names:
            check_metadata(t.extractfile(f"{root}/PKG-INFO").read().decode(), "sdist PKG-INFO")

for lic in LICENSES:
    check(wheel_licenses.get(lic) == sdist_licenses.get(lic), f"{lic} differs between the wheel and the sdist")
    check(len(wheel_licenses.get(lic, b"")) > 500, f"{lic} is missing or truncated")
check(b"Apache License" in wheel_licenses.get("LICENSE-APACHE", b"") and b"Version 2.0, January 2004" in wheel_licenses.get("LICENSE-APACHE", b""), "LICENSE-APACHE is not the Apache-2.0 text")
check(b"Copyright (c) 2026 TensorQuill contributors" in wheel_licenses.get("LICENSE-MIT", b""), "LICENSE-MIT lacks the project copyright line")

if problems:
    print("\n".join(f"FAIL: {p}" for p in problems))
    sys.exit(1)
print("archive contents OK")
