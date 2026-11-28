"""Where a test finds a contract page: `conftest.contract_page`.

The contract pages (`*_CONTRACT.md`) are kept outside this repository since Nov 27, 2026. A test that
holds a page to the code reads it through `conftest.contract_page`, which these tests hold to its three
answers: the folder the variable names; a page at the repository's root when the variable is unset; and a
skip of the calling test when there is none, with a reason CI's server job does not take for a missing
binary or a missing pyarrow. A folder that is named and lacks the page is an error, not a skip, so a run
that was told where the pages are cannot pass without reading them.
"""

from __future__ import annotations

import re

import conftest
import pytest

# The pattern of CI's server job (.github/workflows/ci.yml), which fails the job on a skip it matches.
CI_REFUSES = re.compile(r"tensorquill-server|TENSORQUILL_SERVER_BIN|server binary|pyarrow")


def test_the_named_folder_is_read_before_the_root(tmp_path, monkeypatch):
    (tmp_path / "named").mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "named" / "X_CONTRACT.md").write_text("from the named folder é", encoding="utf-8")
    (tmp_path / "root" / "X_CONTRACT.md").write_text("from the root", encoding="utf-8")
    monkeypatch.setattr(conftest, "_ROOT", tmp_path / "root")
    monkeypatch.setenv(conftest.CONTRACT_PAGES_ENV, str(tmp_path / "named"))
    assert conftest.contract_page("X_CONTRACT.md") == "from the named folder é"
    monkeypatch.delenv(conftest.CONTRACT_PAGES_ENV)
    assert conftest.contract_page("X_CONTRACT.md") == "from the root"
    monkeypatch.setenv(conftest.CONTRACT_PAGES_ENV, "")
    assert conftest.contract_page("X_CONTRACT.md") == "from the root"


def test_a_page_missing_from_the_named_folder_is_an_error_not_a_skip(tmp_path, monkeypatch):
    (tmp_path / "X_CONTRACT.md").write_text("at the root", encoding="utf-8")
    (tmp_path / "named").mkdir()
    monkeypatch.setattr(conftest, "_ROOT", tmp_path)
    monkeypatch.setenv(conftest.CONTRACT_PAGES_ENV, str(tmp_path / "named"))
    with pytest.raises(FileNotFoundError):
        conftest.contract_page("X_CONTRACT.md")


def test_with_no_page_the_calling_test_is_skipped_for_a_reason_ci_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(conftest, "_ROOT", tmp_path)
    monkeypatch.delenv(conftest.CONTRACT_PAGES_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        conftest.contract_page("X_CONTRACT.md")
    reason = str(skipped.value)
    assert "X_CONTRACT.md" in reason and conftest.CONTRACT_PAGES_ENV in reason
    assert not CI_REFUSES.search(reason), reason
