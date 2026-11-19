"""Lane V2 (wave V): probe.py, data.py, huggingface.py and system.py.

Each test reproduces a row of OPEN_BUGS.md and failed on the base commit (66b52d7).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from tensorquill import data as D

FIXTURE = Path(__file__).parent / "fixtures" / "c16-batch-hash-versions.json"


# --- C16: batch hashes -------------------------------------------------------------------------


@pytest.mark.parametrize("left, right", [
    (["a\x00b", "c"], ["a", "b\x00c"]),  # the audit's pair: the old writer joined text with NUL
    ([b"\xff"], [b"\xfe"]),  # bytes that are not UTF-8: the old writer decoded them with replacement
    (["1"], [1]),  # text and a number: one item's type is part of its bytes
    ([1, "a"], ["1", "a"]),
    ([True], [1]),
    ([1.0], [1]),
    (["ab", "c"], ["a", "bc"]),
    ([b"a"], ["a"]),
    ([2**63], [str(2**63)]),  # beyond int64: typed items, not an unknown batch
])
def test_c16_different_batches_hash_apart(left, right):
    assert D.batch_hash(left) != D.batch_hash(right)


def test_c16_a_batch_of_int64_and_its_typed_twin_hash_apart():
    # [1] goes the int64 way; a NumPy-free stand-in for an index the int64 path refuses.
    class Index(int):
        pass

    assert D.batch_hash([1, 2]) == D.batch_hash(torch.tensor([1, 2]))
    assert D.batch_hash([1, 2]) == D.batch_hash([Index(1), Index(2)])
    assert D.batch_hash([1, 2]) != D.batch_hash([1.0, 2.0])


def test_c16_an_index_beyond_int64_still_hashes():
    assert D.batch_hash([2**64, 1]) is not None


def test_c16_text_with_a_lone_surrogate_hashes():
    assert D.batch_hash(["\ud800"]) not in (None, D.batch_hash(["\ud801"]))


def test_c16_hashes_carry_their_version():
    value = D.batch_hash([0, 1, 2])
    assert value.startswith(D.HASH_PREFIX) and D.HASH_PREFIX == "v2:"
    digest = value[len(D.HASH_PREFIX):]
    assert len(digest) == 16 and int(digest, 16) >= 0


def _legacy(*batches: list) -> str:
    """The writer before C16 (data.py at 66b52d7), for integer and text batches."""
    h = hashlib.blake2b(digest_size=8)
    for values in batches:
        if all(isinstance(v, int) for v in values):
            raw = b"".join(v.to_bytes(8, "little", signed=True) for v in values)
        else:
            raw = "\x00".join(map(str, values)).encode()
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def test_c16_the_shared_fixture_holds_both_writers_hashes():
    doc = json.loads(FIXTURE.read_text())
    assert doc["batches"]
    for name, entry in doc["batches"].items():
        assert entry["legacy"] == _legacy(entry["ids"]), name
        assert entry["v2"] == D.batch_hash(entry["ids"]), name
        assert ":" not in entry["legacy"] and entry["v2"].split(":", 1)[0] == "v2"
    hashes = {name: e for name, e in doc["batches"].items()}
    for case in doc["cases"]:
        for record in case["records"]:
            value = record["data"]["batch_hash"]
            assert any(value in (e["legacy"], e["v2"]) for e in hashes.values()), case["name"]
        kinds = {r["data"]["batch_hash"].partition(":")[0] if ":" in r["data"]["batch_hash"] else ""
                 for r in case["records"]}
        assert (len(kinds) == 1) == case["compared"], case["name"]
