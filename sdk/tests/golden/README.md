# Golden files

What `tensorquill.build_evidence(records)` gave with the Python detectors, for each record set the parity tests use: the `CASES` of `../test_server_detect.py` and the seven Gate 1 fixtures (`gate-1-*.json`). `../test_golden.py` holds the Python path and the Rust path (`detectors="rust"`) to these files; its docstring describes them in full.

- Each file holds `incidents`, `not_checked`, `kind_checks` (what was checked, added Nov 16, 2026), `run_state` and the per-step MoE `samples`, beside the case's name and the count and SHA-256 of the records it was frozen from.
- The files are strict JSON with sorted keys, one incident, check or sample to a line. A NaN is written `{"$float": "nan"}`, an infinity `{"$float": "inf"}` or `{"$float": "-inf"}`.
- Do not edit a file by hand. To rewrite the files on purpose, run `pytest tests/test_golden.py --write-golden` from `sdk/` and read the diff.
