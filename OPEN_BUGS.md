# TensorQuill: open bugs, consolidated

One list of the open bugs known from the two code reviews and the fix lanes, as of `main` at `d2a9106` (pushed as `73eac65` with its docs commit), after all four merge groups of waves V and W, the six lanes of wave X (Nov 19, 2026) and the six lanes of wave Y (Nov 20, 2026). It joins three sources that were tracked apart:

- `CODEBASE_AUDIT.md`, the second audit: 70 finding groups with IDs C, RS and X. Its evidence-reader rows have no IDs, so they are numbered E1 to E8 here.
- `codebase_review_findings.md`, the first outside review: 87 findings with IDs SDK, DET, CLI and RUST.
- The open items the fix lanes raised themselves, numbered L1 to L24 here.
- A cross-check of the pending-work list against the code, done on `main` at `a31dd37` (Nov 19, 2026) and kept in `PENDING_CODING_TASKS.md`, which is untracked. Its ten reproduced findings are numbered XC1 to XC10 here (section 10).

Both audit files are untracked. The decision log for all of this is `TensorQuill v3 — Product and Build Plan.md`.

**How far to trust a row.** The "Checked" column says who last confirmed it:

- *Run*: reproduced with a script, by a fix lane or by lane T6. *Run on main* means the script was run again on `main` on Nov 19, 2026, after wave U.
- *Read*: seen in the code, not run.
- *Audit*: the audit's claim, not checked again by this project's sessions.
- *Cross-check*: reproduced by the cross-check's author with temporary files and CPU jobs, as that list records; not run again when it was copied here.

Only the rows marked *Run on main* were run again when this file was written. Rows last checked before wave U may have moved.

**Priority.** Each row keeps its source's own label: P1 to P3 from the second audit, Medium or Low from the first review. The one exception is SDK-22, Medium in the review and listed as P1 here because it can raise into training.

## Counts

| Area | P1 | P2 or Medium | P3 or Low | Total |
|---|---:|---:|---:|---:|
| Can raise into training, or stop it | 1 | 0 | 0 | 1 |
| Training SDK | 1 | 7 | 0 | 8 |
| Evidence readers, detectors, command line, MCP | 0 | 0 | 0 | 0 |
| Rust server | 0 | 1 | 1 | 2 |
| Experiments, examples, CI, docs | 0 | 2 | 0 | 2 |
| **All** | **2** | **10** | **1** | **13** |

There were 72 when this file was written. Waves V and W closed 51 of them in four merge groups, and CI on Nov 19, 2026 closed X11; wave X closed four more (DET-14, DET-17, CLI-13, CLI-14); wave Y closed three more (C06, SDK-23, DET-13); 13 remain, 10 of which are fixed in part and say what is still open. A recount by script on Nov 19, 2026 found 15 rows marked as fixed in part among the 21, not the 16 this file gave before. Two P1 rows remain, each fixed in part: SDK-22 (a process group that fails after `init`; decided Nov 20, 2026, not built) and C03 (every stop notice lost, accepted for v1). C06 (the length check) was the third until wave Y closed it. Section 6 holds what is left of the closed findings and the two reviews' disagreements, section 7 the items the lanes raised, section 8 the audit's notes on cost and narrow cases, and section 9 the wrong behaviour the plan records from measurement. None of those is counted here.

## 1. Can raise into training, or stop it

| ID | Priority | Issue | Checked | State |
|---|---|---|---|---|
| SDK-22 | P1 (Medium in the review) | **Open part after M3:** a process group that fails after `init` still fails inside the training step, because the gather runs there; the agreement at `init` does not cover it. **Decided Nov 20, 2026 ("Yes, approve"), not built:** the gather leaves the training step for v1; each rank records its own fingerprints and an all-ranks reader or the server compares them, with a missing rank read as insufficient coverage. It is a wave of its own after the next six lanes and changes what proof result 1 measures. Also decided that day: the two explicit calls (`replica_spread()`, `record_replica_spread(group)`) stay, documented as collectives every participating rank must call and that can block; old captures holding `replica_spread` keep their verdicts; XC2 is built in this wave; each rank also records an exact signature of each parameter, which is evidence and raises nothing until the eight-GPU run (v3 plan, "Decision, SDK-22" and "Decision for the SDK-22 wave"). Fixed (lane W4): every rank consistently skips a drift check known at `init` to be incompatible, so no rank issues the gather while another skips it. The public `replica_spread` on `meta` or sparse: see lane W4's note for its outcome | Run (W4), gloo on CPU and injected failures; not NCCL | Fixed in part |

## 2. Training SDK

### P1

| ID | Issue | Checked | State |
|---|---|---|---|
| C03 | **Open part after M4, accepted for v1 by the author:** if every stop notice is lost and the server sees a clean close, the run reads `finished`. Not a bug, by the author's decision of Nov 19, 2026 ("yes keep raising"): a stopped rank that then reads `unresponsive` or `lost` raises `job_stalled` or `rank_silent`, in both engines, as contract 3 has it; such a rank still sends heartbeats, so their stopping is new evidence. Still open from that decision: the incident does not say that capture had stopped earlier, at which step and why, so a reader cannot see that TensorQuill cannot tell whether training itself stalled (both engines, not started). Fixed (lanes W1, W2, W3, contract 3): the heartbeat says capture stopped on every beat, the server has a state and a summary word of its own for it that survive a restart, and a run with such a rank never reads healthy | Run (W1, W2, W3; the lifecycle test) | Fixed in part |

### P2, P3, Medium and Low

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| C11 | | P2 | A closed session's forward hooks still run on every forward. `close()` leaves them on purpose: its docstring says removing a hook from a compiled module triggers a recompile | Read |
| C12 | | P2 | **Open part after wave X:** a parameter unfrozen after `init` is still not recorded; it is named once in a warning. Recording it does not break the record schema, but both engines' `update_spike` then fires a false incident at the unfreeze step (L13); lane X1 kept the working patch out. L13 is fixed (lane Y2, wave Y), so the patch is no longer blocked; it is not applied yet. Fixed (lane X1): an optimizer that holds only frozen parameters ends steps, and so does a model whose every parameter is frozen at `init`, which catches TrainCheck B1 as `stalled_weights`. Fixed (lane V1): an optimizer given the model later by `add_param_group` ends steps | Run (V1, X1) |
| C14 | | P2 | **Open part after M1, a stated limit:** two different weights that share a float64 norm are still listed in `weights_unchanged` (swapped rows, for example). Lane V1 fixed what the record claims: its documentation now says an equal norm is necessary, not sufficient, and a parameter that was not compared is never listed (contract 2) | Run (V1) |
| C16 | | P2 | **Open part after M1:** the first resume after upgrading the SDK is not compared, once; the old writer's collisions stay in old stores. Fixed (lane X4): a batch of bools and other integers no longer hashes as its integers; a step that drew one is written `v3:`, so neither reader compares it with a `v2:` hash, and the protocol number is unchanged. Fixed (lanes V2, V3, V7): the new writer's `v2:` hashes do not collide, and both readers compare only hashes of one kind | Run (V2, V3, V7, X4) |
| C20 | SDK-12 | P2 | **Open part after M1:** an unmarked evaluation's time is still in `step_time`; there is no host signal for it. Fixed (lane V1): a marked pause is out of `step_time`, and evaluation fetches, marked or not, are out of `data_wait` | Run (V1) |
| C22 | SDK-11, SDK-17 | P2 | **Open part after wave X:** still counted as overflow, not read: on MPS, squares of norms above about 1.8e19 (not run, no MPS here); a logit too large for BF16; logits near float32's largest value with both signs in one row. Evidence built from a server's report now carries `head_trends` with `overflow_positions` (lane Y2, wave Y), so the overflow count shows there too; the all-ranks view still has none (L20). Fixed (lane X4): a mean whose float32 sum overflows is read. Fixed (lane X3): `head_trends.overflow_positions` gives the probes, the positions, the most in one probe and the first and latest step. The rest is fixed (lane V2) | Run (V2, X3, X4) |
| SDK-16 | | Medium | **Open part after M1:** the wrapper-module fix (lane V2) was not run with a real LightningModule, DDP or FSDP | Run (V2) |

## 3. Evidence readers, detectors, command line and MCP

None open after wave Y: DET-13 is closed (lane Y2), and the rest were closed in waves V, W and X.

## 4. Rust server

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| RS10 | RUST-11 | P2 | **Open part after wave Y:** a key column that is missing or of another type still refuses the whole batch and ends the stream; per-row `schema_version` values are not checked. Both need the author first: dropping a whole batch is a data-loss choice, and `flight.py` writes schema version 0 for a record that has none, so which values are valid has to be decided. Fixed (lane Y4): the dropped-row count is in `watchdog.json`, per rank inside each run (`dropped_rows`, `dropped_row_keys`), and each rank has its own limit on the stderr line. Fixed (lane X6): live ingest drops a row with a null run, rank, step, attempt or schema version, counts it by key, and stores and acknowledges the rest of its batch. Fixed (lane V7): the offline modes refuse null or mistyped run, rank, step and attempt keys | Run (V7, X6, Y4) |
| RS17 | RUST-15 | P3 | A negative expert count is judged as a routed sample. The audit ran it on a recorded vector `[-1, 3]`; T6 says no writer makes one, so it does not hold as a live bug | Audit |

RUST-17 (Low) was shown by lane V6 and fixed: rebuilds from the stream path had no shared bound; they now share one of 16.

## 5. Experiments, examples, CI and docs

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| X02 | | P2 | Gate 1's older speed gate reads zero tool calls as an infinite speedup. Needs the author's amendment to the preregistration | Audit |
| X09 | | P2 | **Open part after M2:** the detectors still judge a step Megatron skipped as one that updated the weights. Treating it as skipped needs a public `skipped=` on `session.step()`, the author's decision, parked. Fixed (lane V8): the guide's example records `megatron/update_successful` as a scalar | Run (V8) |

## 6. Parts of closed findings, and where the two reviews disagree

Not counted in the 72.

| ID | Issue | What each source says |
|---|---|---|
| SDK-06 | After a second optimizer steps, the drift check is not repeated. The check runs after the lead optimizer steps and before the second optimizer steps. With the default 50-step interval, drift the second optimizer introduces right after a check is detected at the next check only if it persists; if the run ends or the replicas agree again before then, it goes undetected | **Left open for v1, the author's decision (Nov 19, 2026):** the gather stays where it is. The current schedule gives every rank a predictable point to enter the collective; moving it would change that point and lose the last scheduled check when training ends. Lane W4's judgement: today's timing is collective-safe. The limit is stated in `sdk/README.md` |
| SDK-14 | The heartbeat discards its client on failure without an explicit close | T6 ran it: no client was alive after its beat or after `close()`, so it does not hold. The second audit still calls it a cleanup gap, with no leak shown |
| CLI-10 | A row built from an evidence file older than the records can show a healthy state for steps not yet judged | T6: documented, the row says it is as of those files. The second audit: still visible, and wants the newer interval marked unknown |

## 7. Raised by the fix lanes

These are not in either audit.

| # | Lane | Item |
|---|---|---|
| L1 | U1 | The tensor-subclass rule under `torch.compile` was run on `eager` and `aot_eager`, not on inductor |
| L3 | U4 | The server's ack carries no store ID |
| L10 | V7 | A lone rank with batch hashes is still listed in `ranks_compared`, though it has no rank to be compared with. Existing behaviour, left as it is; a question for the author |
| L12 | W2 | Limits of contract 1, stated in the contract: a stream that ends before its first batch is not attributed; two attempts first met by claim in one outage go by arrival; across local files, attempts no file orders go by `ts`, which is each host's clock; a lower-numbered replacement whose rows beat its first heartbeat, from a sender with no `sender_attempt`, is taken as backlog |
| L18 | X2 | Waits between two ranks, measured on PyTorch 2.15 with gloo on macOS: a rank that reaches `init_process_group` after its peer has given up waits with no limit; a stopped rank's `_gather_and_drop` swallows a gather timeout, so it can sit 60 s for each unmatched gather and say nothing. Both bear on SDK-22 |
| L19 | Y1 | After C06: on the `indices=` path a `Router(8, 12)` whose ids do number the tokens times 12 is still counted (run again on Nov 20, 2026: 120 routed for 10 tokens, no refusal), and by lane Y1's report, not run again, a routing map under `top_k=12` is still counted, with `moe_top_k` 12. The `indices=` path does refuse a `top_k` above the experts when the ids do not number tokens times `top_k` (lane X1's test). Either case could get C06's refusal in `capture.py` |
| L20 | Y2 | The all-ranks view carries no `head_trends`, and a report from an older server has none. The split of a pre-clipping gradient total by parameter assumes `clip_grad_norm_` clips all the recorded parameters together. The flight sender drops a record with a NaN inside a nested value, so such a probe never reaches the server (existing behaviour in `flight.py`) |
| L21 | Y3 | An attempt that appears only in the middle of a rank file over 1 MB is not seen by the run listing. An evidence file beside a file of several ranks does not say which rank it judged (`evidence.py`). Old runs with no manifest and rank files over 1 MB now list their rank count as "1+" |
| L22 | Y4 | No Python reader shows the dropped-row count the server now writes (`dropped_rows`, `dropped_row_keys` in `watchdog.json`). A batch sent again after a lost acknowledgement is counted again, so the count is an upper bound |
| L23 | Y6 | `evidence.py` has no `not_checked` entry for `grads_not_read`. `sink.py`'s `write_end_marker` uses a plain warning whose message is lost under warnings-as-errors. A failed manifest rename leaves a `.tmp` file. The model-state rule in `data.py` is a heuristic: it misses a custom model with only unusual top-level parameter names under a key that names no model, and accepts a cache with dotted or `*_weight` keys |
| L24 | Wave Y merge | Calls the lanes made that await the author: lane Y3 no longer lists an earlier attempt's incident as the rank's own when that rank is the one rank read (L14); lane Y2's `read_server_report` raises `ValueError` on a bad threshold, in readers only; lane Y5's healthy control needs at least 200 steps; lane Y1's SDK-23 watches a second block list outside the main stack, counts a list that repeats one module as one block, and watches `[encoder, decoder]` as two whole blocks. `sdk/README.md` lines 30, 39 and 40 are stale. L15 was not run end to end (a server's report into the MCP tool) |

## 8. The second audit's notes on cost and narrow cases

The audit lists these under simplifications and test quality, not as findings. Not counted in the 72, and none was checked here.

| Where | Note |
|---|---|
| `divergence.py` | Called directly with attempts not yet resolved, it can report a stale divergence step. The documented path resolves attempts first |
| `collectives.py` | It replaces the shared Flight Recorder logger's whole handler list for the length of an analysis, which two analyses at once could disturb |
| Hugging Face adapter | An interrupted `fit` can leave a session open until the next `train()` |
| `capture.py`, router counting | The count builds a boolean matrix of experts × assignments: 16,384 tokens × top-8 × 256 experts is about 33.5 million entries per forward |
| `capture.py`, the ring | Four floats per trainable parameter per ring step are allocated even in routers-only mode |
| `watchdog.rs`, status | Missing-rank detection searches rank lists with `.contains()`, quadratic in world size, while holding the progress lock |
| `evidence.py`, Rerun export | The sample lookup is rebuilt for every step, quadratic in the number of steps |
| `evidence.py`, gradient history | The detector keeps every total and every per-parameter pair where it reads only a trailing window |
| `test_packaging.py` | One test counts shell substrings, so a comment can change its verdict |
| The golden files | Python and Rust agreeing on them guards regressions; it cannot show that both are right. Both reproduced the stalled-weight error (E1) |

## 9. Wrong behaviour the plan records from measurement

Not from either review. Not counted in the 72.

| Item | State |
|---|---|
| `activation_drift` fired on 6 of 7 healthy small trainings on CPU (11 of their 12 runs), the TrainCheck re-creations. Since C24's fix (M1) a model that is one leaf module is observed too, and it fired there as well: step 75 on a healthy bare `nn.Linear` in a trigger test | Open. Anomalies do not fire the trigger by default until thresholds are checked on a real job |
| `resume_incomplete` does not compare a checkpoint's tensors with the model's (TrainCheck B14). Reproduced on CPU: two LSTM tensors (`lstm.weight_hh_l0`, `lstm.bias_hh_l0`) were missing from the checkpoint, were fresh after the resume, and the final loss was 1.76 against 1.35 healthy; the check, which looks only at which components a checkpoint holds, said nothing | Open. The fix is a checkpoint-to-model comparison at resume that raises an incident naming each missing tensor. Notes: `experiments/traincheck-bugs/README.md` |
| Thresholds (spike 5×, diversity drop 0.6, starved ratio, stall count) rest on Granite and synthetic fixtures | Open until a real job runs without false alarms |
| A probe flush of about 150 ms delays a live acknowledgement; 620 ms was seen once | Logged as an open gap |
| `--watch` with the Python detectors recomputes the whole run: 8.1 s per update at 10,000 steps | Open gap |
| Full capture costs +4.15% on the RTX PRO 6000 | Kept opt-in |
| Headless screenshots have only been run on macOS | Blocks Gate 3's step 20 |

## 10. Found in the cross-check of the pending-work list

Not from either review or a fix lane, and not counted in the 72. From `PENDING_CODING_TASKS.md`, "Findings recovered in the cross-check", made on `main` at `a31dd37` before wave X was merged; none was run again after the merge. After wave Y (Nov 20, 2026), XC1 and XC3 to XC9 are fixed and are out of the table (see "Closed, for the record"); XC2 remains; XC10 was closed by CI on `73eac65` the same day. Where that list says "main and X5", the finding was also reproduced on lane X5's branch, so wave X did not fix it.

| # | Where | Finding | Checked |
|---|---|---|---|
| XC2 | `sdk/examples/replica_check.py` | The driver refuses any `WORLD_SIZE` but 2 at argument validation (line 20), before distributed initialization, so it cannot run the eight-rank proof or D1's attribution. It tests divergence on three planted steps but asserts only the parameter and the spread. Extend it or add a driver, and keep the two-rank inconclusive and three-rank majority controls. Held for the SDK-22 wave by the author's decision of Nov 20, 2026 | Cross-check; the `WORLD_SIZE` check read again on Nov 20, 2026 |

That list also names work measured or specified earlier and missing from any active list (compiler and placement rechecks, a forced collective hang with real dumps, unexplained HTTP/2 stream ends in 3 of 22 scaling runs, detector blind spots B6 and B19, the GPU schedule, package metadata). Those are measurements and proofs, not bugs, and are not copied here.

## What to fix first

- **First:** find why `test_a_stopped_rank_keeps_entering_the_drift_checks_gather` timed out once in CI on Python 3.12 with PyTorch 2.6 (Nov 19, 2026) and passed on a re-run. The log does not say where the two ranks stopped; a rank left waiting in a gather is what this project must not do, so a pass on re-run does not close it. Lane X2 (wave X) did not find the cause: 550 two-rank runs with no hang (300 under CPU load) and 3,600 randomized runs with no mismatch in the stopped rank's gather schedule, all on PyTorch 2.15 on macOS, which says little about PyTorch 2.6 on Linux. The test now prints each rank's progress and every thread's stack if it times out again. **Seen again in CI on `73eac65` (Nov 20, 2026), on Python 3.11 with PyTorch 2.15, and this time not as a timeout:** lane X2's progress lines show both ranks completing all four gathers, closing their sessions and destroying the process group; then rank 0, the rank whose writer was made to fail, aborted as the process exited (`terminate called without an active exception`, SIGABRT, exit code -6), and no stack was dumped. It passed on a re-run. So at least this occurrence is a crash at interpreter shutdown, not a rank left waiting in a gather; whether the earlier timeout was the same fault is not known. A training process that exits with an abort code after finishing would be reported as failed by a launcher. **Looked into on Nov 20, 2026; cause not found.** The lead first written here, an SDK daemon thread still in native code at shutdown, is withdrawn: `close` returned at once on rank 0, where a join that times out waits 1 s; the worker uses a JSONL sink, so there is no sender or heartbeat thread; and locally no SDK thread was alive after `close`, with the failing sink or without (20 runs each). Found while looking, and worth fixing on its own: a session stopped by a writer error is kept alive by a reference cycle through that error's traceback (its `_drain` frame holds the session), so the session and the process group it holds are freed only by a garbage collection, where a session that closed normally is freed as soon as its caller drops it. That this plays a part in the abort is not shown: in this test lane X2's wrappers keep the session alive on both ranks until shutdown anyway. Not reproduced: macOS does not abort this way and no Linux was at hand. **Done the same day, not pushed:** the leak is fixed (the error is kept without its traceback, as `flight.py` keeps a sender's, and a closed session lets go of its process group; `sdk/tests/test_stopped_session_is_freed.py`), and the test's worker now dumps every thread's stack on an abort and writes which threads are alive as it returns and at exit (locally only the main thread, on both ranks). The abort itself stays unexplained.
- **Also unexplained, from CI on `e158ed8` (Nov 20, 2026):** `test_watch_all_ranks.py::test_a_rank_between_a_part_and_its_report_is_no_news_but_is_not_left_stuck_once_written` failed once in the ingest and streaming job on Linux (`judged_ranks` was `[1, 2]`, rank 0 left out, after a new part for rank 2) and passed on a re-run of the job with no change. It passed on `99c3b8f`, the push before wave X. Not reproduced on this Mac: 25 plain runs, 20 under CPU load, and 36 runs of the same scenario with the server's status file forced to exist, which it never does here in the test's 2 seconds and would on a slow runner (the server first writes it after 10). Lane X3's L6 change reads that file to decide which ranks are judged, which is a suspicion and not a finding: the log does not say why rank 0 was left out. One of the 36 forced runs failed another way, with a second write when the status file first appeared. Next step: make the test's assertion print each rank's state and reason, so a recurrence says why. **A second occurrence, in CI on `73eac65` (Nov 20, 2026), in another test:** `test_all_ranks_rust.py::test_the_same_kind_at_the_same_step_naming_different_things_is_two_groups` gave the first incident group ranks `[1, 2]` where `[0, 1, 2]` was expected, after one look at a store four ranks had just sent to; it passed on a re-run. Rank 0 left out of an all-ranks reading in the Linux server job on two pushes in a row is a pattern, not a one-off. **Cause found and fixed on Nov 20, 2026 (not pushed; Linux CI has not run the fix).** The server's status gives a rank whose stream is open and whose first batch is not stored yet `attempt` 0 and `last_step` null (`starting`). A status left at that snapshot, which the default 10 s interval leaves for up to 10 s, makes `_not_current_by_status` (`evidence.py`) read "attempt 1 was judged, and the server's status shows attempt 0" and call the rank `earlier_attempt`, so no value judges it. Reproduced with the real server at its default interval and the test's own arguments, by starting the first send about 10 s after the server: 3 of 12 trials left `watchdog.json` at that snapshot and gave judged ranks `[1, 2, 3]` and the first group ranks `[1, 2]`, the CI result exactly; with the status written often, 12 of 12 were right. That the runner stalled that long before the first send is inferred, not seen. Measured only for a sender that names no attempt before its first batch, which an uploaded file (`send_file`) is; a live session names its attempt through its heartbeat, so its exposure is likely narrower and was not measured. The fix: an entry with attempt 0, `last_step` null and no `attempt_without_steps` is not compared; every other case is as before, a claim of another attempt that has stored nothing included (`sdk/tests/test_status_before_first_batch.py`, whose end-to-end case gives judged ranks `[1, 2, 3]` without the fix). Not looked at: a status marked `attempt_without_steps` that stands beside a report of that same attempt, which would be the same kind of stale reading.
- **Before the GPU runs:** replica drift names the rank (D1 / Q4, the author's decision of Nov 19, 2026; pending, not started). L9, which stood here, is fixed (lane X1). XC1 is fixed (lane Y5): the healthy control no longer passes with nothing recorded. XC2, the replica driver that refuses eight ranks, is held for the SDK-22 wave.
- **Before a release:** the two P1 rows, each fixed in part: SDK-22 and C03 (C06 closed in wave Y). SDK-22's open part is decided (Nov 20, 2026: the gather leaves the training step) and not built. C03's open part is now only the incident's wording (Nov 19, 2026).
- **From the cross-check, not counted in the 13:** XC2 (section 10), held for the SDK-22 wave. XC10 is closed by CI on `73eac65`.
- **Found during waves V, W, X and Y, not fixed, and not counted in the 13:**
  - L18 (section 7), from wave X, and L19 to L24, from wave Y. L13 to L17 are fixed; C12's saved patch is no longer blocked.
  - `activation_drift` fires on healthy small trainings, now including single-leaf models (section 9). It needs thresholds checked on a real job.
  - L10: a lone rank with batch hashes is listed as "compared" (section 7). A question for the author.
- **A demonstrated gap with a clear fix:** checkpoint-to-model verification (section 9, TrainCheck B14). An alert that names the missing tensors gives the user an immediate action.
- **Needs the author, not code:** X02 (an amendment), X09 (`skipped=` on `session.step()`), L3 and L10. For SDK-22 the direction and the two explicit calls are decided; proof result 1's row in the Focused Plan is to be amended, with a deadline, before the SDK-22 wave. From wave Y: RS10's two open parts and the lanes' calls in L24.

## Closed, for the record

| Source | Closed |
|---|---|
| First review, high (27) | 22 held and are fixed. 4 did not hold: DET-02, DET-03, DET-06, RUST-02. RUST-05 was narrow and is fixed |
| First review, medium and low (60) | Fixed, and run again on main on Nov 19, 2026 with T6's scripts: SDK-18, SDK-19, CLI-11. Do not hold (T6): SDK-13, SDK-21, SDK-24, DET-08. T6 also says SDK-14, CLI-10, RUST-12 and RUST-15 do not hold; the second audit disagrees on each (section 6, and RS17 in section 4) |
| The sweep after the first review | All six ways to stop or hang training are fixed |
| Second audit (70 groups) | 30 fixed: C01, C02, C04, C05, C07 to C10, C17 to C19; C03 and C06 in part; RS01 to RS04, RS18; X01, X03, X04, X05; and eight evidence rows (five P1 and three small ones) |
| Wave V, merge group M1 (`main` at `351ad85`, Nov 19, 2026; lanes V1, V2, V3, V6, V7) | Fixed, each with a test that failed on `66b52d7`: C13, C15, C21, C23, DET-12, DET-18, E6, L4, L7, RS06, RS07, RS08, RS09, RS11, RS12, RS13, RS14, RS15, RS16, RS19, RUST-13, RUST-16, RUST-18, SDK-15, SDK-25, SDK-26. C24: V1: the capture half is fixed (a model that is one leaf module is observed under its class name); the sink half did not reproduce on `66b52d7`. RUST-17: V6: shown for concurrency (40 rebuilds at once after a restart) and fixed with a bound of 16. Fixed in part, still listed above with what is open: C12, C14, C16, C20, C22, SDK-16, E3, DET-14, DET-17, RS10, L6, CLI-24. Checked and left as they are: C11, SDK-23, RS17, L3 (options written, nothing changed). Verified on the trial merge, macOS and CPU only; not on Linux, PyTorch 2.6 or a GPU |
| Wave V, merge group M2 (`main` at `4b0be90`, Nov 19, 2026; lanes V4, V5, V8) | Fixed, each with a test that failed on `66b52d7` (X06, X07 and X10 by reading, since the code had no seam to test before): CLI-08, CLI-09, CLI-15, CLI-17, CLI-19, CLI-20, CLI-21, CLI-22, CLI-24, DET-07, E2, E4, E7, E8, L8, X06, X07, X08, X10. DET-07's "paused" half does not hold, as T6 said. Fixed in part, still listed above: CLI-13, X09, X11. Checked and left as they are: CLI-10, DET-13, CLI-14. `seeded_failures.py` now writes every case flat into `--out`. Verified on the trial merge, macOS and CPU only |
| Wave W, merge group M3 (`main` at `af846f7`, Nov 19, 2026; lane W4) | SDK-22's first criterion and L5 (ranks on different device types), with W4's one call in `capture.py`. SDK-22 stays open above for a failure after `init`. Verified on the trial merge with gloo on CPU; not NCCL, MPI or a GPU |
| Wave W, merge group M4 (`main` at `c0b63ab`, Nov 19, 2026; lanes W1, W2, W3) | Four of the five P1 rows: C03 and C06 (each with a part still listed above), E1 and RS05. Also E5, DET-09, DET-10, DET-11, DET-16, E3's count half, L2, RUST-05's residual and RUST-12's coverage case. The protocol number is 3. Golden files: 73 lines in 72 files, all from E3 (the `needs` text and what counts as a checkpoint input), RUST-12 and DET-16; none from E1, the step order, DET-09 or DET-11. Verified on the trial merge, with the attempt fixture, the lifecycle test, the cross-file cases and both mixed-version streaming runs; macOS and CPU only |
| Found by CI on Linux after the waves were pushed (`23940bb`), fixed in `367c529` | Two SDK faults seen only on older PyTorch, neither in either review: a compiled router that returns a routing map broke the graph on PyTorch 2.6 (`next()` with a default inside the hook); the probe's histogram of one repeated huge value raised on PyTorch 2.6 and 2.9 (`torch.histc` with equal bounds). One test fault on Python 3.13 and 3.14 (docstring indentation). Not yet run on CI after the fix |
| CI on `99c3b8f` (Nov 19, 2026) | X11: the two jobs it added, Python 3.10 with PyTorch 2.6 and Python 3.12 with PyTorch 2.9, ran and passed. The three faults CI had found on the push before are fixed in `367c529` and passed on every job. One job, Python 3.12 with PyTorch 2.6, failed once on `test_a_stopped_rank_keeps_entering_the_drift_checks_gather` (two gloo ranks did not finish in 120 s) and passed on a re-run with no change: intermittent, not explained, listed under "What to fix first" |
| Wave X (`main` at `dc7cad0`, Nov 19, 2026; lanes X1 to X6, one merge each) | Fixed, by each lane's report that every new test failed before its fix. Checked again at the merge only this far: the new test files of lanes X1, X3, X4 and X5 were run against `a31dd37` and 24 of 33, 31 of 56, 12 of 23 and 14 of 14 of their tests failed there, counted per file and not per item; lane X6's branch was checked with `cargo test` alone; lane X2's with its three two-rank test files. The items: DET-14, DET-17 (three modules; `trigger.py` parsed no `rank=` folder), CLI-13, CLI-14, L6, L9, and four open parts from lane V6's hand-over (`--silent-factor` validation, pause order across a restart, a failed rebuild tried again without a later part, a new attempt's elapsed time). Fixed in part, still listed above: C06, C12, C16, C22, DET-13, RS10. Not fixed: SDK-23, and the cause of the two-rank timeout. C03's Python-side change was made by lane X3 and reverted on the author's decision. Lane X4 changed one of lane V2's tests, which pinned the behaviour C22's fix removes |
| CI on `e158ed8` (Nov 20, 2026), the push of wave X | Server wheels: every job passed, the five installs included. CI: 14 of 15 jobs passed and the GPU compiler gate was skipped. Python 3.12 with PyTorch 2.6, the pairing that timed out on `99c3b8f`, passed with lane X2's new check of exactly four gathers. The ingest and streaming job failed once on one test (1 failed, 1,152 passed, 4 skipped; see "What to fix first") and passed on a re-run with no change. Nothing was closed by this run. The server job's file list still names no test file of lanes V7, W2, W3 or X (XC10) |
| Wave Y (`main` at `d2a9106`, Nov 20, 2026; lanes Y1 to Y6, one merge each, then two commits at the merge) | Fixed, by each lane's report that its new tests failed on `687e316` and pass with the fix; not run against the base again at the merge. The items: C06 (a `top_k` above `n_experts` on the default integer path is refused; equal stays valid), SDK-23, DET-13, L13 (both engines), L14, L15 (both halves, not run end to end), L16 (`evidence.rank_number`, made at the merge), L17, XC1, XC3 to XC9. Fixed in part, still listed above: RS10. Changed, not proven: XC10. At the merge, lane X1's guard test for the old C06 behaviour was removed. Run on the merged tree, macOS and CPU with PyTorch 2.15 only: the full SDK suite with the server built from that tree, 2,721 passed, none failed, none skipped; `cargo test` 209 passed and 1 ignored; `cargo clippy` clean with all features and with none; `cargo fmt --check` clean; the library tests without default features 148 passed; the experiment tests 112 passed; CI's four example commands and the healthy control passed |
| CI on `73eac65` (Nov 20, 2026), the push of wave Y | XC10: the server job ran the whole suite on Linux with the binary, 2,618 passed, 87 skipped and 1 failed in 23 minutes (31 for the job, against a 50-minute limit), and the suites of lanes V7, W2 and W3 executed there for the first time; none of the 87 skips was for want of the binary or pyarrow (optional packages: `transformers`, `lightning`, `rerun`, `accelerate`). Server wheels: every job passed. CI: 13 of 15 jobs passed and the GPU compiler gate was skipped; the two failures are the two unexplained intermittent tests ("What to fix first"), and both jobs passed on a re-run (2,619 passed in the server job, 1,759 in the Python 3.11 job) |
