# TensorQuill: open bugs, consolidated

One list of the open bugs known from the two code reviews and the fix lanes, as of `main` at `4b0be90`, after both merge groups of wave V (Nov 19, 2026). It joins three sources that were tracked apart:

- `CODEBASE_AUDIT.md`, the second audit: 70 finding groups with IDs C, RS and X. Its evidence-reader rows have no IDs, so they are numbered E1 to E8 here.
- `codebase_review_findings.md`, the first outside review: 87 findings with IDs SDK, DET, CLI and RUST.
- The open items the fix lanes raised themselves, numbered L1 to L8 here.

Both audit files are untracked. The decision log for all of this is `TensorQuill v3 — Product and Build Plan.md`.

**How far to trust a row.** The "Checked" column says who last confirmed it:

- *Run*: reproduced with a script, by a fix lane or by lane T6. *Run on main* means the script was run again on `main` on Nov 19, 2026, after wave U.
- *Read*: seen in the code, not run.
- *Audit*: the audit's claim, not checked again by this project's sessions.

Only the rows marked *Run on main* were run again when this file was written. Rows last checked before wave U may have moved.

**Priority.** Each row keeps its source's own label: P1 to P3 from the second audit, Medium or Low from the first review. The one exception is SDK-22, Medium in the review and listed as P1 here because it can raise into training.

## Counts

| Area | P1 | P2 or Medium | P3 or Low | Total |
|---|---:|---:|---:|---:|
| Can raise into training, or stop it | 1 | 0 | 0 | 1 |
| Training SDK | 2 | 7 | 1 | 10 |
| Evidence readers, detectors, command line, MCP | 1 | 10 | 1 | 12 |
| Rust server | 1 | 1 | 1 | 3 |
| Experiments, examples, CI, docs | 0 | 3 | 0 | 3 |
| **All** | **5** | **21** | **3** | **29** |

There were 72 when this file was written. Wave V closed 43 of them in its two merge groups (and, outside the 72, the lanes' items L4, L7 and L8 and the never-shown RUST-17); 29 remain, 14 of which are fixed in part and say what is still open. All five P1 rows are still open: they are wave W's. Section 6 holds five more that are parts of closed findings or that the two reviews disagree on, section 7 the items the lanes raised, section 8 the audit's notes on cost and narrow cases, and section 9 the wrong behaviour the plan records from measurement. None of those is counted here.

## 1. Can raise into training, or stop it

| ID | Priority | Issue | Checked | State |
|---|---|---|---|---|
| SDK-22 | P1 (Medium in the review) | **Open part after M3:** a process group that fails after `init` still fails inside the training step, because the gather runs there; the agreement at `init` does not cover it. The long-term fix (no collective in the training path) changes what proof result 1 measures and is the author's decision. Fixed (lane W4): every rank consistently skips a drift check known at `init` to be incompatible, so no rank issues the gather while another skips it. The public `replica_spread` on `meta` or sparse: see lane W4's note for its outcome | Run (W4), gloo on CPU and injected failures; not NCCL | Fixed in part |

## 2. Training SDK

### P1

| ID | Issue | Checked | State |
|---|---|---|---|
| C03 | After capture stops on a streaming run, the server reads a suspected hang about 2 s later, or `finished` if the session is closed while training goes on | Run (U1) | The local sink is fixed; this part is open |
| C06 | A router's integer output with indices out of range is recorded as zeros with no refusal (`[-1, 8]` for 4 experts gives `[0, 0, 0, 0]`) | Run | The bool routing map is fixed; this part is open |

### P2, P3, Medium and Low

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| C11 | | P2 | A closed session's forward hooks still run on every forward. `close()` leaves them on purpose: its docstring says removing a hook from a compiled module triggers a recompile | Read |
| C12 | | P2 | **Open part after M1:** a parameter unfrozen after `init` is still not recorded; it is now named once in a warning. The other half is fixed (lane V1): an optimizer given the model later by `add_param_group` ends steps | Run (V1) |
| C14 | | P2 | **Open part after M1, a stated limit:** two different weights that share a float64 norm are still listed in `weights_unchanged` (swapped rows, for example). Lane V1 fixed what the record claims: its documentation now says an equal norm is necessary, not sufficient, and a parameter that was not compared is never listed (contract 2) | Run (V1) |
| C16 | | P2 | **Open part after M1:** the first resume after upgrading the SDK is not compared, once; the old writer's collisions stay in old stores; a list starting with an int that holds a bool hashes as its integers. Fixed (lanes V2, V3, V7): the new writer's `v2:` hashes do not collide, and both readers compare only hashes of one kind | Run (V2, V3, V7) |
| C20 | SDK-12 | P2 | **Open part after M1:** an unmarked evaluation's time is still in `step_time`; there is no host signal for it. Fixed (lane V1): a marked pause is out of `step_time`, and evaluation fetches, marked or not, are out of `data_wait` | Run (V1) |
| C22 | SDK-11, SDK-17 | P2 | **Open part after M1:** on MPS squares of norms above about 1.8e19 still overflow (shown as `overflow_positions`, not as NaN; not run, no MPS here); a mean that overflows float32 is counted as overflow, not read. The rest is fixed (lane V2) | Run (V2) |
| SDK-16 | | Medium | **Open part after M1:** the wrapper-module fix (lane V2) was not run with a real LightningModule, DDP or FSDP | Run (V2) |
| SDK-23 | | Low | On a stack with two layer classes only one class's blocks are watched; an encoder-decoder gets only the larger stack | Run (T6) |

## 3. Evidence readers, detectors, command line and MCP

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| E1 | | P1 | A parameter that changes without a gradient is still called a stalled-weight invariant. Held out of wave U because it changes a detector rule | Audit |
| E3 | | P2 | **Open part after M1:** a scaler-only resume that finds nothing still reads `not_configured`; counting the loss scale needs both engines changed together (lane W3). Fixed (lane V3): a kind that reports an incident is counted as checked | Run (V3) |
| E5 | | P2 | The physical order of steps in the file changes which fault is named first | Audit |
| DET-09 | | Medium | Learning rates seen only in rolled-back steps can make `scheduler` an expected component, so a resume can be called incomplete (Python and Rust) | Read |
| DET-10 | | Medium | Records out of order within an attempt give the Rust detectors different incidents than Python. The capture writes steps in order | Run (T6) |
| DET-11 | | Medium | In a top-k router's warm-up with `n_experts <= 2k`, `overloaded` is always empty (Python and Rust). Dead experts are still flagged | Read |
| DET-13 | | Medium | The compact file's table refuses a baseline it cannot show, and only the earliest starvation onset anchors the window | Read |
| DET-14 | | Medium | **Open part after M1:** at a run's first step the expert-load table compares the step with itself. Fixed (lane V3): the table is shown at the first step and layer names are escaped | Run (V3) |
| DET-16 | | Medium | A bool is read as the number 1.0 (Python and Rust) | Run (T6) |
| DET-17 | | Low | **Open part after M1:** `attempt=<n>` folders are still parsed with the looser rule (lane W3's code), and four other modules parse `rank=` on their own. Fixed (lane V3): the evidence readers share one `rank=` parser | Run (V3) |
| CLI-13 | | Medium | **Open part after M2:** when a run and rank has two files, their order is path order and the `end` cell reads only the first file; both wait for contract 1's rule across files (wave W). Fixed (lane V4): the second file is no longer dropped | Run (V4) |
| CLI-14 | | Medium | MCP serves no head or attention probe series | Read |

## 4. Rust server

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| RS05 | | P1 | A late new step from an older attempt takes ownership of live progress. Needs the author's choice of four ordering options (lane U6's entry in the plan) | Run (U6) |
| RS10 | RUST-11 | P2 | **Open part after M1:** live ingest's per-row check (`keep_new_rows`) is unchanged. Fixed (lane V7): the offline modes refuse null or mistyped run, rank, step and attempt keys | Run (V7) |
| RS17 | RUST-15 | P3 | A negative expert count is judged as a routed sample. The audit ran it on a recorded vector `[-1, 3]`; T6 says no writer makes one, so it does not hold as a live bug | Audit |

RUST-17 (Low) was shown by lane V6 and fixed: rebuilds from the stream path had no shared bound; they now share one of 16.

## 5. Experiments, examples, CI and docs

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| X02 | | P2 | Gate 1's older speed gate reads zero tool calls as an infinite speedup. Needs the author's amendment to the preregistration | Audit |
| X09 | | P2 | **Open part after M2:** the detectors still judge a step Megatron skipped as one that updated the weights. Treating it as skipped needs a public `skipped=` on `session.step()`, the author's decision, parked. Fixed (lane V8): the guide's example records `megatron/update_successful` as a scalar | Run (V8) |
| X11 | | P2 | **Open part after M2:** the two new CI jobs (Python 3.10 with PyTorch 2.6, Python 3.12 with PyTorch 2.9) have never run; this machine has neither. They run first after a push | Read |

## 6. Parts of closed findings, and where the two reviews disagree

Not counted in the 72.

| ID | Issue | What each source says |
|---|---|---|
| SDK-06 | After a second optimizer steps, the drift check is not repeated. The check runs after the lead optimizer steps and before the second optimizer steps. With the default 50-step interval, drift the second optimizer introduces right after a check is detected at the next check only if it persists; if the run ends or the replicas agree again before then, it goes undetected | **Left open for v1, the author's decision (Nov 19, 2026):** the gather stays where it is. The current schedule gives every rank a predictable point to enter the collective; moving it would change that point and lose the last scheduled check when training ends. Lane W4's judgement: today's timing is collective-safe. The limit is stated in `sdk/README.md` |
| RUST-05 | On offline input with steps out of order, the resume comparison takes the first physical row, not the lowest step, so a learning-rate change or a numbering restart can be missed | The finding is fixed (lane T4). The second audit lists this as a remaining offline case. Audit, read |
| RUST-12 | Four skipped rows with `update_window=2` give `checked, records=4, enough_history=true` although no comparison was eligible | Open, moved to lane W3: Python counts skipped steps the same way and a parity test holds the two together. T6: the finding does not hold as a parity bug. The second audit ran this coverage case and lists it. Audit, run |
| SDK-14 | The heartbeat discards its client on failure without an explicit close | T6 ran it: no client was alive after its beat or after `close()`, so it does not hold. The second audit still calls it a cleanup gap, with no leak shown |
| CLI-10 | A row built from an evidence file older than the records can show a healthy state for steps not yet judged | T6: documented, the row says it is as of those files. The second audit: still visible, and wants the newer interval marked unknown |

## 7. Raised by the fix lanes

These are not in either audit.

| # | Lane | Item |
|---|---|---|
| L1 | U1 | The tensor-subclass rule under `torch.compile` was run on `eager` and `aot_eager`, not on inductor |
| L2 | U1 | The writer's bounded wait for NVML can stop capture above about 200 steps a second |
| L3 | U4 | The server's ack carries no store ID |
| L6 | U6 | **Open part after M1:** in the all-ranks evidence a rank with `attempt_without_steps` is still counted among judged ranks; only `run_state` calls it unknown. Fixed (lane V3): the readers look at the flag |

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
| Thresholds (spike 5×, diversity drop 0.6, starved ratio, stall count) rest on Granite and synthetic fixtures | Open until a real job runs without false alarms |
| A probe flush of about 150 ms delays a live acknowledgement; 620 ms was seen once | Logged as an open gap |
| `--watch` with the Python detectors recomputes the whole run: 8.1 s per update at 10,000 steps | Open gap |
| Full capture costs +4.15% on the RTX PRO 6000 | Kept opt-in |
| Headless screenshots have only been run on macOS | Blocks Gate 3's step 20 |

## What to fix first

- **Before a release:** the five P1 rows: SDK-22, C03, C06, E1 and RS05. RS05 and E1 each need a decision first.
- **For the GPU runs:** no row is shown to change them. Two were considered:
  - C20 moves evaluation time into `data_wait` and `step_time` on the step after a pause. It matters to the group C cell only if that cell runs evaluation between training steps; no evaluation loop was found in the cell's description.
  - C14 needs two different weights whose float64 norms are exactly equal. The stalled-weights seeded case freezes a weight, which is a true stall, so it does not create that.
- **Needs the author, not code:** X02 (an amendment) and RS05 (an ordering policy).

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
