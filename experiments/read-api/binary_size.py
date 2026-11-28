"""The release binary's size, as WAVE_AN_CONTRACT.md section 13.4 states it: at most 2.5 MB of growth over
the base's with `rusqlite` bundled and the rest of wave AN, on macOS arm64.

    .venv/bin/python experiments/read-api/binary_size.py base=<the base's binary> wave=<the wave's binary>

Prints each binary's bytes and, for every binary after the first, its growth over the first, in bytes and
MB (10^6 bytes). It builds nothing: the lane builds the wave's binary after the pin (section 16.1), and a
release build of the base is the one to compare (the shared binary of `a6c1e2c` is one). With no argument
it reports the binary `TENSORQUILL_SERVER_BIN` names.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    pairs = [arg.split("=", 1) for arg in sys.argv[1:]]
    if not pairs:
        path = os.environ.get("TENSORQUILL_SERVER_BIN")
        if not path:
            raise SystemExit("give label=path pairs, or set TENSORQUILL_SERVER_BIN")
        pairs = [["server", path]]
    sizes = [(label, Path(path).stat().st_size) for label, path in pairs]
    first = sizes[0][1]
    out = {}
    for index, (label, size) in enumerate(sizes):
        out[label] = {"bytes": size, "mb": round(size / 1e6, 3)}
        if index:
            out[label].update({"growth_bytes": size - first, "growth_mb": round((size - first) / 1e6, 3)})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
