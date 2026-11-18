"""Print what tensorquill requires, one requirement per line, without `tensorquill-server`.

    python packaging/requirements.py [extra ...]            # from pyproject.toml, with those extras
    python packaging/requirements.py --wheel <tensorquill-*.whl>   # from a built wheel's metadata

A stopgap, to be deleted with its callers' two-step installs: `tensorquill` depends on
`tensorquill-server`, which is not on PyPI before the first release, so `pip install tensorquill`
(or `-e .`) cannot resolve it yet. CI and the wheel smoke test install these lines and then the
package itself with `--no-deps`. Both modes fail (exit 1) when the `tensorquill-server` requirement
is missing or carries a platform marker, so the two-step install cannot hide that it was dropped.
"""
import re
import sys
import zipfile
from pathlib import Path

SERVER = "tensorquill-server"


def name_of(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;(]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def split_server(requirements: list[str], where: str) -> list[str]:
    server = [r for r in requirements if name_of(r) == SERVER]
    if len(server) != 1 or ";" in server[0]:
        sys.exit(f"{where}: expected one `{SERVER}` requirement with no marker, found {server}")
    return [r for r in requirements if name_of(r) != SERVER]


def from_pyproject(extras: list[str]) -> list[str]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with path.open("rb") as fh:
        project = tomllib.load(fh)["project"]
    lines = split_server(list(project["dependencies"]), str(path))
    for extra in extras:
        if extra not in project.get("optional-dependencies", {}):
            sys.exit(f"{path}: no extra named {extra!r}")
        lines += project["optional-dependencies"][extra]
    return lines


def from_wheel(wheel: str) -> list[str]:
    with zipfile.ZipFile(wheel) as z:
        [metadata] = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
        header = z.read(metadata).decode().split("\n\n", 1)[0]
    required = re.findall(r"^Requires-Dist: (.+)$", header, re.M)
    return split_server([r for r in required if "extra ==" not in r], wheel)


if __name__ == "__main__":
    args = sys.argv[1:]
    print("\n".join(from_wheel(args[1]) if args[:1] == ["--wheel"] else from_pyproject(args)))
