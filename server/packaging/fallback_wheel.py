"""Make the fallback wheel of `tensorquill-server` from one built platform wheel.

    python fallback_wheel.py <built platform wheel> <output directory>

The SDK depends on `tensorquill-server` with no platform markers, so pip must find some wheel on
every platform. Where there is no build (Windows, Intel macOS, musl Linux), it finds this one:
tagged `py3-none-any`, holding the metadata and the two license files and nothing else. No program
and no stand-in script are in it, so "is `tensorquill-server` in the scripts directory" stays a true
test of whether the binary is installed.

It is derived from a built wheel, not built on its own, so its name, version and METADATA are
byte for byte those of the platform wheels: a resolver that reads one wheel's metadata for the
whole release is told the truth. Standard library only.
"""

import base64
import hashlib
import sys
import zipfile
from pathlib import Path

TAG = "py3-none-any"


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def make(source: Path, out_dir: Path) -> Path:
    name, version = source.name.split("-")[:2]
    dist_info = f"{name}-{version}.dist-info"
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(source) as wheel:
        for entry in wheel.namelist():
            # Kept: METADATA and licenses/. Dropped: the program under .data/, the bill of
            # materials of that program (sboms/), and WHEEL and RECORD, which are rewritten.
            if entry == f"{dist_info}/METADATA" or entry.startswith(f"{dist_info}/licenses/"):
                files[entry] = wheel.read(entry)
    if f"{dist_info}/METADATA" not in files:
        raise SystemExit(f"{source.name}: no {dist_info}/METADATA")
    if len(files) != 3:
        raise SystemExit(f"{source.name}: expected METADATA and two license files, found {sorted(files)}")
    files[f"{dist_info}/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: tensorquill fallback_wheel.py\nRoot-Is-Purelib: true\nTag: {TAG}\n"
    ).encode()
    record = [_record_line(entry, data) for entry, data in sorted(files.items())]
    record.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = ("\n".join(record) + "\n").encode()

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}-{version}-{TAG}.whl"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as wheel:
        for entry, data in sorted(files.items()):
            # A fixed date, as maturin writes, so the same input gives the same file.
            wheel.writestr(zipfile.ZipInfo(entry, date_time=(1980, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
    return target


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(make(Path(sys.argv[1]), Path(sys.argv[2])))
