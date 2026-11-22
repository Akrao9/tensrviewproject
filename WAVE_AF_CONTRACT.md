# Wave AF contract: identity fields, the views, the status write, offline against live

**Nothing on this page is built.** It is cut from `main` at `76a3b5d` (code as verified at `f5ee53e`) for the integrity lanes of the v1 plan, which the author approved on Nov 21, 2026. Lanes C, D and F build from it; lanes A, B and E do not need it and are listed here only for who owns which file.

Where this page states something the author's words do not, it says **this page's reading**, and that reading is the author's to change. The readings are listed together at the end.

Three rules hold for every section:

- **Nothing here blocks, pauses, terminates or raises into training.** Every refusal on this page is a reader's: it happens where a file or a stream is read, never inside a training step and never in the capture.
- **A supported capture reads as before.** An evidence file, a report and a status made from records whose identity fields are valid are byte for byte what they are on the base, apart from the keys a section adds and names.
- **Gate 1.** What its arms are shown is compared with the base at the trial merge and never read to choose a rule. A difference is brought to the author before the merge.

## Who owns which file

One owner for each source file and each test file. A lane writes new tests in its own new files. An existing test file is changed only by its owner below, and only where this page changes the behaviour it asserts. A file on nobody's list is not changed: the lane reports what it needs.

| Lane | Source | New test files | Existing test files it may change |
|---|---|---|---|
| A. Checkpoint comparison | `sdk/src/tensorquill/data.py` | `sdk/tests/test_lane_af_a_*.py` | `test_checkpoint_tensor_contract.py`, `test_lane_ac1_checkpoint.py`, `test_data_*.py`, `test_lane_y6_manifest_data.py` |
| B. Capture | `capture.py`, `probe.py` | `test_lane_af_b_*.py` | `test_capture*.py`, `test_probe.py`, `test_head_probe.py`, `test_close_and_clipping.py`, `test_lane_*_capture.py`, `test_lane_*_probe_data.py`, `test_lane_w1_capture_sender.py` |
| C. Evidence and readers | `evidence.py`, `divergence.py`, `mcp_server.py`, `runs.py`, `compare.py` | `test_lane_af_c_*.py` | `test_evidence*.py`, `test_divergence.py`, `test_mcp_server.py`, `test_runs_compare.py`, `test_watch_*.py`, `test_all_ranks_*.py`, `test_lane_*_readers.py`, `test_lane_*_evidence.py`, `test_lane_ac4_divergence.py`, `test_lane_v4_runs_compare.py`, `test_lane_x5_runs_mcp.py`, `test_lane_z2_reader.py`, `test_kind_checks.py`, `test_run_state.py`, `test_compact.py` |
| D. Rust server | everything under `server/` | `server/src/detect/lane_af_d_tests.rs`, `server/tests/lane_af_d_*.rs`, `sdk/tests/test_lane_af_d_*.py` | every Rust test; `test_lane_*_server.py`, `test_lane_ae1_ingest.py`, `test_server_reports.py`, `test_server_detectors.py`, `test_run_detect.py`, `test_rs10_acceptance.py`, `test_status_before_first_batch.py`, `test_watchdog.py` |
| E. Docs and experiments | `sdk/README.md`, `sdk/RUN_ON_YOUR_JOB.md`, `sdk/AGENT_INSTRUCTIONS.md`, `sdk/examples/`, `experiments/` (never Gate 1's fixtures, `generate.py`, the held-out cases or `PREREGISTRATION.md`) | `test_lane_af_e_*.py` | `test_lane_v8_examples.py`, `test_lane_y5_examples.py`, `test_benchmark_detectors.py`, `test_seeded_failures.py`, `test_trainer_check.py`, `test_profile_overhead.py`, the tests inside `experiments/` |
| F. The wire row | `flight.py`, `server_detect.py`, and the new `sdk/src/tensorquill/identity.py` | `test_lane_af_f_*.py` | `test_flight.py`, `test_flight_e2e.py`, `test_server_detect.py`, `test_lane_*_writer.py`, `test_lane_*_sender.py`, `test_rs10_sender.py`, `test_lane_u4_flight_sink.py`, `test_lane_v7_server_detect.py` |
| This page | `WAVE_AF_CONTRACT.md`, `sdk/tests/fixtures/wave_af/`, `sdk/tests/test_wave_af_contract.py` | | |

Lanes A to D and F do not edit the documents. Each leaves the sentences a document needs in its notes, and lane E or the integration enters them. The bug list and the plan are changed at the integration only.

## 1. The identity fields of a record (D1; L32, SDK-06, the offline half of RS-04)

**The fault.** A record's `schema_version`, `rank`, `attempt` and `step` say which rank, attempt and step an observation belongs to. On the base the two engines disagree on a value that is not an integer, and neither names the field:

| A one-record file with | Python detectors | Rust detectors | `tensorquill-server detect`, given the row |
|---|---|---|---|
| `attempt`, `schema_version` or `step` as `null`, `true`, `1.0`, `1.5` or `"1"`; `rank` as `true`, a float or text | Read; exit 0 (`step: null` and `rank: null` apart, below) | `TypeError: not an integer`, exit 1 with a traceback | `null`: Arrow refuses the stream. The others cannot be put in an Int32 column |
| `rank: null`, `step: null` | A bare `TypeError` from `int()`, traceback | The same | Arrow refuses the stream |
| `step` absent | `KeyError: 'step'`, traceback | The same | Cannot be expressed |
| `attempt`, `rank`, `step` or `schema_version` of -3; `schema_version` of 99 | Read; exit 0 | Read; exit 0 | **Judged**: a report with 1 step, and for the rank, `rank: -3` |

All of it was run on `76a3b5d`; each case's `today` in `cases.json` is the record.

**What is valid.** The same in both engines and at every boundary:

| Field | Absent | Valid when present |
|---|---|---|
| `schema_version` | Read as 0, the legacy version, as on the base | The integer 0 or 1 (the versions the server stores) |
| `rank` | Read as on the base (0, or the reader's default rank) | An integer of 0 or more |
| `attempt` | Read as 0, as on the base | An integer of 0 or more |
| `step` | **Not valid** | An integer of 0 or more |
| `run_id` | As on the base | As on the base: this page does not change how a missing, null or non-text run ID is read (pins `run_id_absent`, `run_id_null`, `run_id_number`) |

- A Boolean is not an integer. A float is not an integer, **also when it is whole** (`1.0`): this page's reading, the stricter of what the two engines do today.
- Text is not an integer (`"1"`).

**What a reader does with a record that is not valid.** It refuses the input with one named error and judges nothing. The record is not dropped and the rest judged: an observation that cannot be placed in a rank, an attempt and a step is a missing observation, and a missing observation never reads as a clean check. This page's reading.

- **The error.** `tensorquill.identity.MalformedRecordError`, a `ValueError`, with `field`, `value`, `where` and `reason` as attributes. Its text names the field, says what was found and what is read, and says where: the file and the 1-based line for a `.jsonl` file, the Parquet part and the row for a store, the position in the list for records passed in memory. For an unsupported `schema_version` it says which versions this package reads.
- **One check, in one place.** `identity.check(record, where)` is the only code that decides validity in Python (lane F). `flight.record_to_row` calls it before anything else, so a refusal leaves the caller's `moved` list as it was passed (SDK-06). The readers of lane C call it where records are loaded, before either engine is chosen, so the two engines cannot disagree.

| Boundary | A record that is not valid | Owner |
|---|---|---|
| A local `.jsonl` file or records in memory: `build_evidence`, `watch`, `watch_all_ranks`, `first_divergence`, `compare`, with either engine | `MalformedRecordError`; no evidence file is written or replaced | C, calling F's check |
| The row for the Rust detectors and for the sender: `flight.record_to_row`, `server_detect.records_stream`, `run_stream` | `MalformedRecordError`, raised before `moved` is touched | F |
| The command line: `python -m tensorquill <input>`, with `--watch`, `--rank all`, `compare`, with either engine | Exit status 2, one line `tensorquill: error: ...` with the error's text, no traceback | C |
| The MCP tools that read records (`list_incidents`, `get_incident`, `get_series`, `compare_runs`) | A `ToolError` with the error's text; the server answers the next call | C |
| `runs.discover_runs`, the MCP `list_runs` | As on the base: they do not raise (pin). What they show for such a file is recorded by lane C and brought to the author; no key is added by this page | C |
| Offline `tensorquill-server detect`, with and without `--run`, given an Arrow stream | Exit status 1 and one line naming the column, the value and the row. A null: Arrow's own refusal, as on the base, when its text names the column; otherwise the same line. A negative `rank`, `attempt` or `step`, and a `schema_version` that is not 0 or 1: refused, where the base judges them | D |
| Live ingest | **Unchanged** (RS10, decided by the author on Nov 20, 2026): a row with a null key is refused and counted under its key; a row of another `schema_version` is refused and counted under `unsupported_schema_version`; a batch with a negative `attempt` is refused whole with `invalid_argument`. Pinned by the tests the base has. What live ingest does with a negative `rank` or `step` is read and run by lane D and reported; it is not changed here | D |
| The sender reading its own `.jsonl` file | As on the base. Lane F runs it on a file with one such line between valid ones and records what is sent, what is counted and what is said. A sender that stops sending the valid lines after it, or raises into training, is a finding to report, not to fix in passing | F |

**The difference that stays.** Offline, a file is refused as a whole. Live, the server refuses the row or the batch and counts it, since it cannot refuse a run that is still arriving. Lane E says so in the documents.

**Cases** (`cases.json`, section 1). One per field and value: absent, `null`, `true`, `1.0`, `1.5`, `"1"`, -3, and `schema_version` 99, for `schema_version`, `rank`, `attempt` and `step`; three pins for `run_id`. Each has `today` and `expected` for the Python detectors, the Rust detectors, the command line with each, and, where an Int32 column can hold the value, the offline `detect`. A case whose `today` already equals its `expected` at every boundary is a **pin**; the others are **new**.

**Order between lanes.** Lane F builds `identity.py` first. Lane C does its other items first and wires the check after lane F is in the trial tree, continuing from that tree.

## 2. The same observation in every view (EV-01 to EV-04)

No engine is involved: these are the Python readers, one owner. The page fixes the rule and the regressions; lane C chooses the shape of a new key, following the `<block>_by_rank` blocks `combine_rank_evidence` already has (`skipped_updates_by_rank`, `head_trends_by_rank`), and enters it in this section when it is built.

**The rule.** A key that says what was *not* observed, or how an observation was made, reaches every view that shows the check it qualifies: the one-rank evidence, the all-ranks evidence from local files and from a store, the command line's summary, and the MCP tools. In the all-ranks view it keeps which ranks it came from. A view that cannot show it says the check is not stated; it never shows the check as clean.

| Row | The base | The target, and its regression |
|---|---|---|
| EV-01 | `first_divergence` lists `layers_not_read` and `moe_source` under `left_out.not_compared`, as if they were series | They are not series and are in no list of series. Where the two runs differ in either, the answer says so under a key of its own. Case `divergence_provenance_keys_are_not_series` |
| EV-02 | `not_checked` of a rank is lost on a path of the all-ranks view | Every judged rank's `not_checked` entries are in the all-ranks evidence with their ranks, from local files and from a store, and in the MCP answer for the run |
| EV-03 | The all-ranks kind checks drop `modules_not_judged` and `starting_state` | Kept, with their ranks. The audit only read this one; the lane first makes a run that produces each key on one rank, so the regression fails on the base |
| EV-04 | The all-ranks view drops `moe_source` and the four measurements (`gradient_trend` reproduced) | Kept, with their ranks, for each of the four and for `moe_source`; each reproduced on the base first |
| The stale post-pass | `grads_not_read` is applied by a second pass that can disagree with the first | One place decides it; the evidence of a run where it applies is unchanged |

Each regression runs the same records through every view the row names and asserts the field in each.

## 3. A file rewritten in place under `--watch` (EV-07, D3 at its default)

**The base.** The watcher takes a file as unchanged when its length and both 4 KiB ends are what they were. A rewrite that keeps all three is missed, a stated limit.

**The target.** The file's modification time (`st_mtime_ns`) joins the signature. A rewrite of the same length and the same ends with a later modification time is read again, by both engines. What stays a stated limit, in the class's note and the README: a rewrite that keeps the length, both ends and the modification time.

- An append is still read as an append: a changed modification time with a longer file and an unchanged head does not cause a full read where the base reads the tail. The regression asserts it by counting what is read, not by timing.
- Regressions: the rewrite is seen; an append is not made slower; a `touch` with no change of content gives the same evidence bytes.

## 4. The status after a refusal and after late rows (the second half of RS-01)

**The base.** `write_statuses` writes `watchdog.json` at once when one of a set of values changed. `refused_batch_keys`, `rows_not_newer` and `late_steps_confirmed` are not in that set, so a status that only gained a refusal or a late row waits for the next periodic write.

**The target.** A change in any of the three is a change that writes the status at once, by the path the other counts take. Nothing else about when a status is written changes.

- Regression, through a real server: one refused batch, and separately one late row, each reach `watchdog.json` without any other event and before the periodic write would have happened; the test fails on the base.
- After this is in the trial tree, the author decides whether RS10 is closed (D4).

## 5. Offline `detect` against live ingest (RS-04)

| | The base | The target |
|---|---|---|
| `attempt_order` | In a live report; absent from the offline one | In both, with the same value for the same rows |
| A negative `rank`, `attempt` or `step`; an unsupported `schema_version` | Offline judges them; live refuses | Offline refuses, section 1; live as it is |
| Steps out of order within an attempt | Offline puts them in order and judges, as the Python detectors do; live counts a row that is not newer and does not judge it again | **Both stay** (pins). The difference is stated: offline is given the whole file, live judges as rows arrive |

**The shared cases** (lane D writes them beside its tests, since both sides are Rust): the same valid rows, of one attempt and of two, in step order, sent to a live server and given to `detect`. The two reports are equal after removing the keys that only a live server can have, and lane D lists those keys here, each with why. A key that differs and is not on that list fails the test.

## 6. The cross-rank state that is never pruned (RS-05)

**The base.** Two maps of the cross-rank state have nothing that removes from them, so they grow with the run.

**The target.** They are pruned by the rule the state's other maps are already pruned by; no new setting. If lane D finds no such rule to follow, it builds nothing here and brings the choice to the author with what each map holds and how fast it grows. A verdict that the base gives is unchanged: the regression runs a long stream, asserts the maps' sizes stay bounded, and asserts the status and the reports equal the base's on the same rows.

Router counts of mixed sign (`[-2, 10]`) read as checked with no incident in both engines. That is RS17's accepted limit and stays (pin).

## 7. A sender whose batches are refused (D5): a test before any limit

No limit is adopted on this page. Lane D adds one test and reports what it shows, and the author decides D5 from it.

- **The test** (`sdk/tests/test_lane_af_d_refused_sender.py`): the production sender, `flight.py` as it is, against a real server. Several batches the server explicitly refuses, then a valid batch, then a reconnect.
- **What it asserts:** the valid batch is eventually stored. The sender advances only across bytes the server confirmed stored or explicitly refused for that exact range. A range that was neither stays pending and is sent again.
- If the sender fails it, the fix is in `flight.py`, which is lane F's file: lane D reports and does not edit it.

## Wave AG: four findings of the trial merge, fixed on the author's word of Nov 21, 2026 ("Yes, fix all four")

Sections 8 to 11 were added at the trial merge (`trial/af-wave`, `3c1df61`). Each lane merges `trial/af-wave` into its branch first. Ownership is as in the table above. Where a section corrects an earlier one, it says so.

## 8. A router recorded with no tokens (AUD-1 of the correctness audit)

**The fault**, run on the trial tree: a model trained in `eval()` mode with reentrant activation checkpointing has every router's counts recorded as 0 on every step (and, under full capture, no block statistics). Both engines read the MoE kinds as `not_configured` and the run as `healthy`. The same model in train mode reports `moe_imbalance`.

**The capture (lane B).** A training step's forward is counted also when the modules are in eval mode and the forward runs without gradients as reentrant checkpointing runs it. An evaluation forward is still not counted. Lane B finds what tells the two apart (the step has a backward and an optimizer step; the recomputation inside backward is the same forward) and says in its notes what it chose and what it cannot tell apart. Where a step's forward cannot be counted, the record does not carry zeros as if they were read: the router is left out of that record's `moe`, by the mechanism the capture already has for a router it did not read, and no new record key is added. Regressions: the audit's reproducer (`wave-af-proof-correctness-audit/f1_evalmode_reentrant.py` in the notes folder) gives the counts of its train-mode control; the same under full capture gives the block statistics; an evaluation forward under `no_grad` in eval mode between steps is still not counted; capture on and off give the same final parameters.

**The readers, both engines (lanes C and D), for captures already written and for whatever the capture still cannot read.** A router that the records name and whose counts are 0 on every record was not read:

| The records | `moe_imbalance`, `starved_experts` | `not_checked` |
|---|---|---|
| Every router named has only zero counts (`every_router_recorded_with_no_tokens`, new) | `unavailable`, where the base says `not_configured` | One entry `{"check": "moe_routing", "kinds": ["moe_imbalance", "starved_experts"], "routers": [...], "reason": ...}`, the routers sorted by name |
| Some routers have only zero counts, others were read (`one_router_with_no_tokens_beside_one_that_was_read`, new) | Judged on the routers that were read, as on the base | The same entry, naming only the routers not read |
| Every router was read on at least one record (`routers_that_were_read`, pin) | As on the base | No such entry |

- The run state is what the existing rule gives a run with an `unavailable` kind; it is not `healthy` in the first row. If the existing rule would still give `healthy`, the lane reports and builds nothing more.
- A router with zero counts on some records and counts on others is as on the base: this section is about a router never read.
- The entry reaches the all-ranks views with its ranks, by section 2's rule, and the MCP tools.
- Cases: `sdk/tests/fixtures/wave_af/cases_ag.json`, three cases whose `today` is both engines' answer on `3c1df61` (they agree). Lane C turns them into tests of the Python detectors and lane D of the Rust detectors, offline and in the live report.

## 9. Several refused batches in a row (D5; RS10)

**The fault** (lane D's `test_several_refused_batches_then_a_valid_batch_then_a_reconnect`, a strict expected failure on the trial tree): `Ingest::confirm_refusal` names a refused range from the stream's accepted end, else the rank's last acknowledged offset, and reads the batch's claimed start only when nothing was ever acknowledged for the rank. After one confirmed refusal, the next refused batch is named from the last stored acknowledgement, the sender cannot confirm a range that does not start at its offset, and it sends that batch for the rest of the run.

**The target (lane D).** The server keeps, per rank, the end of the last refusal it confirmed. A refused batch whose claimed start equals that end is named from it. The claim is then a position the server itself named, never the sender's word alone. A refusal whose confirmation was lost moves nothing, as now. Across a restart the kept end follows the rule RS10's restart fix already uses for the acknowledged offset; if it cannot be kept across a restart, the lane says what a sender then sees and tests it.

- The expected-failure mark is removed and the test passes: three refused batches in a row, then a valid batch, then a reconnect; the valid batch is stored; the sender advances only across bytes confirmed stored or refused for that exact range.
- The one-refusal test and RS10's acceptance tests pass unchanged.

## 10. A negative `attempt`, `rank` or `step` at the sender and at live ingest

This section corrects section 1's rows "Live ingest" and "The sender reading its own `.jsonl` file", and its paragraph "The difference that stays", for these three values.

**The faults**, run on the base and on the trial tree. A line with a negative `attempt` or `rank` is refused by the server with its whole batch and no range, and the sender sends that batch for the rest of the run, with nothing after it stored. A negative `step` is acknowledged, stored and judged by the live server, and the readers then refuse that store by name.

**The sender (lane F).** `rows_to_send` does not send a line whose `rank`, `attempt` or `step` is negative: it is skipped and counted in `skipped_lines`, as a line whose key cannot be stored already is, and the lines after it are sent. The first such line is said once on stderr, naming the line and the field; the base's silent skip of a non-integer key is said the same way. A `schema_version` the server does not read is still sent, for the server to count (RS10, unchanged). Nothing raises into training. Regression with a real server: the file of lane F's probe with a negative `attempt`, and one with a negative `rank`, each store every valid line; `close()` returns True.

**Live ingest (lane D).** A batch holding a negative `step` is refused whole with `invalid_argument`, as a batch holding a negative `attempt` is, and nothing of it is stored. The server's own stores and reports are then always readable by section 1's readers. Regression: the probe that stored `part--0000003--0000002.parquet` on the trial tree stores nothing and acknowledges nothing for that batch, and the next valid batch is stored.

**The difference that stays, corrected.** Offline, a file with such a record is refused as a whole. Live, a row with a null key or an unread `schema_version` is refused and counted, and a batch with a negative `attempt` or `step` is refused whole; the SDK's own sender never sends the latter.

## 11. Entered with the above, no code

Lane E enters, once the code is in the trial tree: the sentences the lanes left for the documents, section 10's corrected paragraph, and two stated behaviours from the audit, each checked against the code first: with `auto_step=False` the gradient norms are read at `session.step()`, so `zero_grad()` goes after it (AUD-2); a checkpoint load inside one attempt is recorded and compared but judged by no detector (AUD-3).

## Not on this page

The live report's trend memoization follows lane D's changes above, and the incremental Rust watcher follows the integration of the reader and detector changes. Each gets its own acceptance, as the plan states it, when it starts.

## This page's readings, for the author

1. A whole float (`1.0`) is not a valid `rank`, `attempt`, `step` or `schema_version`.
2. A reader refuses the whole input on the first record that is not valid; it does not drop the record and judge the rest.
3. A negative `step` and a negative `rank` are not valid.
4. `run_id` is left as it is.
5. The command line's exit status for such an input is 2.
6. `list_runs` and `discover_runs` gain no key here.
7. Offline and live keep their difference on steps out of order.
