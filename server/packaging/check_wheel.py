"""Check one built `tensorquill-server` wheel, or which one pip installed. Standard library only.

    python check_wheel.py wheel <file.whl> <platform>
    python check_wheel.py installed <platform>

<platform> is one of: manylinux_2_17_x86_64, manylinux_2_17_aarch64, macosx_arm64, any.

`wheel` checks the file: its tags, that it holds the program (or, for `any`, that it holds none),
both license files, the license expression, and that the version is the crate's. For a Linux
wheel it also reads the program's own GLIBC symbol versions with `objdump`, when that is on PATH.

`installed` checks the environment it is run in: the tag in the installed WHEEL file says which
wheel pip chose, and the program is, or for `any` is not, in the scripts directory. It is how a
job with both a platform wheel and the fallback on offer shows that pip took the right one.
"""

import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

PROGRAM = "tensorquill-server"
LICENSES = {"LICENSE-APACHE", "LICENSE-MIT"}
# The tags each platform's wheel may carry. `manylinux2014` is the older name of `manylinux_2_17`
# (PEP 600), and maturin writes both. Anything else, such as `manylinux_2_28` or `linux`, fails.
ALLOWED = {
    "manylinux_2_17_x86_64": {"py3-none-manylinux_2_17_x86_64", "py3-none-manylinux2014_x86_64"},
    "manylinux_2_17_aarch64": {"py3-none-manylinux_2_17_aarch64", "py3-none-manylinux2014_aarch64"},
    "any": {"py3-none-any"},
}
MAC = re.compile(r"py3-none-macosx_1[1-9]_\d+_arm64")
GLIBC_FLOOR = (2, 17)


def fail(message: str) -> None:
    raise SystemExit(f"check_wheel: {message}")


def check_tags(tags: set, platform: str, where: str) -> None:
    if platform == "macosx_arm64":
        ok = bool(tags) and all(MAC.fullmatch(t) for t in tags)
    else:
        required = f"py3-none-{platform}"
        ok = required in tags and tags <= ALLOWED[platform]
    if not ok:
        fail(f"{where}: tags {sorted(tags)} are not those of a {platform} wheel")


def wheel_tags(text: str) -> set:
    return {line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("Tag:")}


def crate_version() -> str:
    cargo = (Path(__file__).resolve().parents[1] / "Cargo.toml").read_text()
    package = cargo.split("[package]", 1)[1].split("\n[", 1)[0]
    return re.search(r'^version\s*=\s*"([^"]+)"', package, flags=re.M).group(1)


def newest_glibc(program: Path):
    out = subprocess.run(["objdump", "-T", str(program)], capture_output=True, text=True, check=True).stdout
    found = {tuple(int(n) for n in m.split(".")) for m in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", out)}
    if not found:
        fail(f"objdump found no GLIBC symbol version in {program.name}")
    return max(found)


def check_wheel(path: Path, platform: str) -> None:
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        metadata = wheel.read(f"{dist_info}/METADATA").decode()
        tags = wheel_tags(wheel.read(f"{dist_info}/WHEEL").decode())
        check_tags(tags, platform, path.name)
        # The file name is what pip and PyPI read; it must say the same as WHEEL.
        name_tags = path.name[: -len(".whl")].split("-")[-1].split(".")
        if {f"py3-none-{p}" for p in name_tags} != tags:
            fail(f"{path.name}: file name platforms {name_tags} disagree with WHEEL tags {sorted(tags)}")

        head = metadata.split("\n\n", 1)[0]
        fields = [tuple(part.strip() for part in line.split(":", 1)) for line in head.splitlines() if ":" in line]
        get = lambda key: [v for k, v in fields if k == key]
        if get("Name") != [PROGRAM]:
            fail(f"{path.name}: Name is {get('Name')}")
        if get("Version") != [crate_version()]:
            fail(f"{path.name}: Version {get('Version')} is not Cargo.toml's {crate_version()}")
        if get("License-Expression") != ["Apache-2.0 OR MIT"]:
            fail(f"{path.name}: License-Expression is {get('License-Expression')}")
        if set(get("License-File")) != LICENSES:
            fail(f"{path.name}: License-File lines are {get('License-File')}")
        for key in ("Author", "Author-email", "Maintainer", "Maintainer-email", "Home-page", "Project-URL"):
            if get(key):
                fail(f"{path.name}: {key} is set; the owner chooses it at first publish")
        for license_name in LICENSES:
            if f"{dist_info}/licenses/{license_name}" not in names:
                fail(f"{path.name}: {license_name} is not in the wheel")

        outside = [n for n in names if not n.startswith(dist_info + "/")]
        script = f"{dist_info[: -len('.dist-info')]}.data/scripts/{PROGRAM}"
        if platform == "any":
            if outside:
                fail(f"{path.name}: the fallback wheel must hold nothing outside dist-info, found {outside}")
            print(f"{path.name}: ok ({platform})")
            return
        if outside != [script]:
            fail(f"{path.name}: expected only {script} outside dist-info, found {outside}")
        if wheel.getinfo(script).external_attr >> 16 & 0o111 == 0:
            fail(f"{path.name}: {script} is not marked executable")
        if platform.startswith("manylinux") and shutil.which("objdump"):
            with tempfile.TemporaryDirectory() as scratch:
                newest = newest_glibc(Path(wheel.extract(script, path=scratch)))
            if newest > GLIBC_FLOOR:
                fail(f"{path.name}: the program needs GLIBC_{'.'.join(map(str, newest))}, newer than 2.17")
            print(f"{path.name}: newest GLIBC symbol version is {'.'.join(map(str, newest))}")
    print(f"{path.name}: ok ({platform})")


def check_installed(platform: str) -> None:
    from importlib import metadata

    dist = metadata.distribution(PROGRAM)
    tags = wheel_tags(dist.read_text("WHEEL"))
    check_tags(tags, platform, "installed wheel")
    program = Path(sysconfig.get_path("scripts")) / (PROGRAM + (".exe" if sys.platform == "win32" else ""))
    if platform == "any":
        if program.exists() or shutil.which(PROGRAM):
            fail(f"the fallback wheel is installed but a {PROGRAM} program was found")
        print(f"installed: the fallback wheel {sorted(tags)}, version {dist.version}, and no {PROGRAM} program")
        return
    if not program.is_file():
        fail(f"{program} is missing")
    print(f"installed: {sorted(tags)}, version {dist.version}, program at {program}")


if __name__ == "__main__":
    args = sys.argv[1:]
    known = set(ALLOWED) | {"macosx_arm64"}
    if len(args) == 3 and args[0] == "wheel" and args[2] in known:
        check_wheel(Path(args[1]), args[2])
    elif len(args) == 2 and args[0] == "installed" and args[1] in known:
        check_installed(args[1])
    else:
        raise SystemExit(__doc__)
