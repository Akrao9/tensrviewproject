"""Wave AI: dumps PyTorch's own analyzer cannot build a picture from are "not checked", never a traceback.

Found by lane AI-3's hang harness on CPU: PyTorch 2.15's Gloo dumps name the process group in the
entries (`("0", "default_pg")`) and not in `pg_config`, and `build_db` raises `KeyError: ('0', 2)`.
`analyze_flight_recorder` guarded only the reading of the files, so the error escaped and
`python -m tensorquill ... --flight-recorder DIR` ended in a traceback.
"""

from __future__ import annotations

import json
import pickle

import pytest

pytest.importorskip("torch.distributed.flight_recorder")

import tensorquill  # noqa: E402
from tensorquill.collectives import analyze_flight_recorder  # noqa: E402
from tensorquill.evidence import main  # noqa: E402

import test_collectives as tc  # noqa: E402


def gloo_like_dumps(directory):
    """Two ranks' dumps whose `pg_config` does not name the group their entries name."""
    directory.mkdir()
    entries = [tc._entry(0, "all_reduce", 1)]
    for rank in (0, 1):
        tc._dump(directory, rank, entries)
        path = next(p for p in directory.iterdir() if p.name.endswith(str(rank)))
        dump = pickle.loads(path.read_bytes())
        dump["pg_config"] = {"": {"name": "", "desc": "", "ranks": "[0, 1]"}}
        path.write_bytes(pickle.dumps(dump))
    return directory


def test_the_analyzer_s_own_error_is_a_value_error_that_names_it(tmp_path):
    dumps = gloo_like_dumps(tmp_path / "dumps")
    with pytest.raises(ValueError, match="PyTorch's Flight Recorder analyzer could not") as caught:
        analyze_flight_recorder(dumps)
    assert "KeyError" in str(caught.value)


def test_evidence_lists_the_dumps_as_not_checked(tmp_path):
    dumps = gloo_like_dumps(tmp_path / "dumps")
    records = [{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": step, "grad_norm": 1.0} for step in range(4)]
    evidence = tensorquill.build_evidence(records, flight_recorder=dumps)
    [entry] = [e for e in evidence["not_checked"] if e.get("check") == "flight_recorder"]
    assert "could not" in entry["reason"] and "KeyError" in entry["reason"]
    assert not [i for i in evidence["incidents"] if i["kind"] in ("collective_hang", "collective_desync")]


def test_the_command_ends_normally_with_no_traceback(tmp_path, capsys, monkeypatch):
    dumps = gloo_like_dumps(tmp_path / "dumps")
    source = tmp_path / "r.jsonl"
    source.write_text("".join(json.dumps({"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": step,
                                          "grad_norm": 1.0}) + "\n" for step in range(4)))
    out = tmp_path / "r.evidence.json"
    monkeypatch.setattr("sys.argv", ["tensorquill", str(source), "--flight-recorder", str(dumps), "--detectors", "python",
                                     "--output", str(out)])
    main()
    printed = capsys.readouterr()
    assert "Traceback" not in printed.out + printed.err
    written = json.loads(out.read_text())
    assert any(e.get("check") == "flight_recorder" for e in written["not_checked"])
