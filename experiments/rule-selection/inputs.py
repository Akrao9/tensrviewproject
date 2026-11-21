"""The only door to the inputs a detector rule of wave AD may be chosen from.

`load(id)` gives the record sets of one entry of `allowlist.json`, each as (name, records in
step order, the entry). Anything not on the list is refused, and so is any path that holds one
of the list's refused words: Gate 1's six held-out cases are never read here, in any form.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
MAIN = Path("/Users/ak/Documents/tensrview/tensrviewproject")
GENERATED = HERE / "generated"  # ignored: what the two generator entries wrote
LIST = json.loads((HERE / "allowlist.json").read_text())
ENTRIES = {entry["id"]: entry for entry in LIST["inputs"]}

sys.path.insert(0, str(WORKTREE / "sdk" / "src"))
from tensorquill import evidence as ev  # noqa: E402


def _allowed(path: Path) -> Path:
    text = str(path)
    for word in LIST["refused_words"]:
        if word in text:
            raise PermissionError(f"{text} holds {word!r}: not a development input")
    return path


def _file(path: Path) -> dict[int, list[dict]]:
    """One capture file's records by rank, attempts resolved, in step order."""
    rows = ev.load_records(_allowed(path))
    by_rank: dict[int, list[dict]] = {}
    for row in rows:
        by_rank.setdefault(int(row.get("rank", 0)), []).append(row)
    return {rank: sorted(found, key=lambda r: int(r["step"])) for rank, found in by_rank.items()}


def _module(path: Path):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"allowed_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def load(entry_id: str) -> list[tuple[str, list[dict], dict]]:
    entry = ENTRIES[entry_id]  # a KeyError is the refusal: it is not on the list
    kind = entry["kind"]
    if kind == "file":
        sets = _file(Path(entry["path"]).expanduser())
        return [(entry_id if len(sets) == 1 else f"{entry_id}.rank{rank}", rows, entry) for rank, rows in sets.items()]
    if kind in ("glob", "glob-many"):
        root = MAIN if entry.get("root") == "main" else WORKTREE
        found = []
        for name in sorted(glob.glob(str(root / entry["path"]))):
            path = Path(name)
            for rank, rows in _file(path).items():
                label = path.name.split(".rank")[0] if kind == "glob-many" else entry_id
                found.append((f"{label}.rank{rank}" if kind == "glob-many" else label, rows, entry))
        return found
    if kind == "python":
        if entry["call"] == "BY_NAME['seeded'].records":
            # generate.py is the bake-off's generator. Importing it builds nothing; only the visible
            # practice fixture's records are asked for, by name.
            module = _module(MAIN / entry["module"])
            rows = module.BY_NAME["seeded"].records()
        else:
            rows = getattr(_module(WORKTREE / entry["module"]), entry["call"])()
        return [(entry_id, sorted(ev.resolve_attempts(rows), key=lambda r: int(r["step"])), entry)]
    if kind == "generated":
        found = []
        for name in sorted(glob.glob(str(GENERATED / entry_id / "*.jsonl"))):
            for rank, rows in _file(Path(name)).items():
                found.append((entry_id, rows, entry))
        return found
    if kind == "validation":  # run once against the proposed rules; never part of `everything()`
        return [(entry_id, rows, entry) for name in sorted(glob.glob(str(WORKTREE / entry["path"])))
                for rows in _file(Path(name)).values()]
    raise ValueError(f"unknown kind {kind!r}")


def everything() -> list[tuple[str, list[dict], dict]]:
    return [one for entry_id, entry in ENTRIES.items() if entry["kind"] not in ("in-script", "validation")
            for one in load(entry_id)]


def role(name: str, entry: dict) -> str:
    """healthy or seeded. The TrainCheck runs say it in their names."""
    if entry["role"] != "by-name":
        return entry["role"]
    return "healthy" if "-healthy" in name else "seeded"
