# TensorQuill: open bugs, consolidated

One list of every known open bug, as of `main` after wave U (Nov 19, 2026). It joins three sources that were tracked apart:

- `CODEBASE_AUDIT.md`, the second audit: 70 finding groups with IDs C, RS and X. Its evidence-reader rows have no IDs, so they are numbered E1 to E8 here.
- `codebase_review_findings.md`, the first outside review: 87 findings with IDs SDK, DET, CLI and RUST.
- The open items the fix lanes raised themselves, numbered L1 to L8 here.

Both audit files are untracked. The decision log for all of this is `TensorQuill v3 — Product and Build Plan.md`.

**How far to trust a row.** The "Checked" column says who last confirmed it:

- *Run*: reproduced with a script, by a fix lane or by lane T6.
- *Read*: seen in the code, not run.
- *Audit*: the audit's claim, not checked again by this project's sessions.

No row was run again when this file was written. Rows last checked before wave U may have moved.

**Priority.** Each row keeps its source's own label: P1 to P3 from the second audit, Medium or Low from the first review.

## Counts

| Area | P1 | P2 or Medium | P3 or Low | Total |
|---|---:|---:|---:|---:|
| Can raise into training, or stop it | 1 | 0 | 0 | 1 |
| Training SDK | 2 | 12 | 4 | 18 |
| Evidence readers, detectors, command line, MCP | 1 | 19 | 6 | 26 |
| Rust server | 1 | 9 | 7 | 17 |
| Experiments, examples, CI, docs | 0 | 8 | 2 | 10 |
| **All** | **5** | **48** | **19** | **72** |

The 72 are the second audit's 40 unfixed rows, its 2 rows fixed in part (C03, C06), and from the first review 25 open findings that the audit does not list and 5 fixed in part. Eight more items raised by the lanes are listed at the end, and one finding that was never shown (RUST-17) is noted under the server. Neither is counted above.

## 1. Can raise into training, or stop it

| ID | Priority | Issue | Checked | State |
|---|---|---|---|---|
| SDK-22 | P1 (Medium in the review) | A drift-check gather that fails after it is issued raises out of `optimizer.step()`. The public `replica_spread` on a `meta` or sparse parameter raises too | Run (T6, U5) | Fixed in part by lane U5: the check is skipped on every rank where the gather cannot run. A catch was tried and left the other ranks waiting for the group's timeout, so none shipped |

## 2. Training SDK

### P1

| ID | Issue | Checked | State |
|---|---|---|---|
| C03 | After capture stops on a streaming run, the server reads a suspected hang about 2 s later, or `finished` if the session is closed while training goes on | Run (U1) | The local sink is fixed; this part is open |
| C06 | A router's integer output with indices out of range is recorded as zeros with no refusal (`[-1, 8]` for 4 experts gives `[0, 0, 0, 0]`) | Run | The bool routing map is fixed; this part is open |

### P2, P3, Medium and Low

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| C11 | | P2 | A closed session's forward hooks still run on every forward. Kept on purpose: the C03 decision never removes them | Audit |
| C12 | | P2 | Adding a parameter group to an optimizer after `init` silently disables step capture for it | Audit |
| C13 | | P2 | The on-device index cache grows with every distinct set of parameters that has gradients | Audit |
| C14 | | P2 | Two different weights with equal L2 norms are named in `weights_unchanged` | Audit |
| C15 | | P2 | A replica-spread table with too few rows is broadcast to the other parameters | Audit |
| C16 | | P2 | Different batches can encode to the same batch hash | Audit |
| C20 | SDK-12 | P2 | Evaluation time is counted as training data wait and step duration | Run (T6) |
| C21 | SDK-20 | P2 | A checkpoint of a model with `get_extra_state` loses its model component in the manifest | Run (T6) |
| C22 | SDK-11, SDK-17 | P2 | Large finite statistics come out wrong or as nonfinite in the probe (at magnitudes near 1e20 and above) | Run (T6) |
| C23 | | P2 | Two sessions built at the same time can send heartbeats with each other's attempt ID | Audit |
| C24 | | P3 | Two reporting paths treat missing data as coverage (`state="closed"` when a marker exists and its file does not) | Audit |
| SDK-15 | related to C14 | Medium | A trainable parameter with no elements (an empty shard) is in `weights_unchanged` at every check. The decode half is fixed | Run (T6) |
| SDK-16 | | Medium | A wrapper module with no `get_output_embeddings` or `config` hides the inner model's head and attention settings from the probe. The composite-config half is fixed (lane U3) | Run (T6) |
| SDK-23 | | Low | On a stack with two layer classes only one class's blocks are watched; an encoder-decoder gets only the larger stack | Run (T6) |
| SDK-25 | | Low | Blocks that return a dict get no probe histogram | Run (T6) |
| SDK-26 | | Low | Three small ones left: `nvml: {}` in every record after every metric fails once; a checkpoint folder given as a `Path` records no `load` manifest; the `auto_step` docstring disagrees with the scaler option. The offset's scratch file is fixed (lane U4) | Run (T6) |

## 3. Evidence readers, detectors, command line and MCP

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| E1 | | P1 | A parameter that changes without a gradient is still called a stalled-weight invariant. Held out of wave U because it changes a detector rule | Audit |
| E2 | CLI-07 | P2 | Rollback is applied to the detectors' input and not to `compare` or MCP progress: after a resume, `last_step` is the highest step of every attempt | Run (T6) |
| E3 | | P2 | A scaler-reset incident is reported while `kind_checks` says that check was never configured | Audit |
| E4 | | P2 | A benign observation marks a run `unhealthy` in `list_runs` | Audit |
| E5 | | P2 | The physical order of steps in the file changes which fault is named first | Audit |
| E6 | DET-15 | P2 | The Python detectors accept invalid settings, such as a negative tolerance or `top_k=0`; Rust refuses each | Run (T6) |
| E7 | CLI-12 | P2 | A run with only the compact evidence file reads as `missing` in the catalog | Run (T6) |
| E8 | DET-19 | P3 | The TrainCheck import merges violations whose names share their first 200 characters, and can change the first step | Run (T6) |
| DET-07 | | Medium | A `job_stalled` episode stays open when the rank later reads `finished` | Run (T6) |
| DET-09 | | Medium | Learning rates seen only in rolled-back steps can make `scheduler` an expected component, so a resume can be called incomplete (Python and Rust) | Read |
| DET-10 | | Medium | Records out of order within an attempt give the Rust detectors different incidents than Python. The capture writes steps in order | Run (T6) |
| DET-11 | | Medium | In a top-k router's warm-up with `n_experts <= 2k`, `overloaded` is always empty (Python and Rust). Dead experts are still flagged | Read |
| DET-12 | | Medium | A watcher whose default rank has no record writes nothing and says nothing | Read |
| DET-13 | | Medium | The compact file's table refuses a baseline it cannot show, and only the earliest starvation onset anchors the window | Read |
| DET-14 | | Medium | The Rerun export shows no expert-load table at the first step | Read |
| DET-16 | | Medium | A bool is read as the number 1.0 (Python and Rust) | Run (T6) |
| DET-17 | | Low | Two parsers accept different spellings of `rank=<n>` | Read |
| DET-18 | | Low | The cache stamp matches Flight Recorder files by substring | Read |
| CLI-08 | | Medium | One unreadable rank folder in a store ends the divergence for every rank | Read |
| CLI-09 | | Medium | A run with no judged rank lists its incidents as none | Read |
| CLI-13 | | Medium | Of two files of one run and rank only the first is kept, with no word | Run (T6) |
| CLI-14 | | Medium | MCP serves no head or attention probe series | Read |
| CLI-15 | | Medium | `get_incident`'s follow-up calls carry no rank. Not read again after lane U7 changed how MCP picks a rank | Read |
| CLI-20 | | Low | A constant series ranks first among series leaving at the same step | Read |
| CLI-21 | | Low | A run kept as files and in a store is listed twice | Run (T6) |
| CLI-22 | | Low | A compact file's previous kept sample is called the previous sample; a store rank's cache can miss a late part of an older attempt | Read |

## 4. Rust server

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| RS05 | | P1 | A late new step from an older attempt takes ownership of live progress. Needs the author's choice of four ordering options (lane U6's entry in the plan) | Run (U6) |
| RS06 | RUST-07 | P2 | A crashed evaluation inflates `recommended_silent_seconds` for the next attempt | Run (T6) |
| RS07 | RUST-08 | P2 | Equal phase and work instants are saved at different precisions, so after a restart a pause can read `silent` | Run (T6) |
| RS08 | RUST-09 | P2 | Before the first stored batch the severity reads `critical` ahead of its own interval | Run (T6) |
| RS09 | RUST-10 | P2 | The deduplication watermark and the detector replay disagree about which files exist | Read |
| RS10 | RUST-11 | P2 | The offline modes accept null run, rank or step keys as run "", rank 0, step 0 | Read |
| RS11 | | P2 | A `checkpoint_json` that does not parse is counted as a checkpoint for `resume_incomplete` | Audit |
| RS12 | RUST-06 | P2 | A repeated offline step attaches the later row's loss and the wrong median to the incident | Run (T6) |
| RS13 | | P2 | A long-lived server keeps finished and abandoned runs' buffers and state | Audit |
| RS14 | | P2 | With the watchdog interval at zero, shutdown still writes a status file | Audit |
| RS15 | | P3 | Invalid duration flags panic with exit code 101 | Audit |
| RS16 | RUST-14 | P3 | The largest int64 step is read as no cutoff | Read |
| RS17 | RUST-15 | P3 | A negative expert count is judged as a routed sample. T6: no writer makes one | Audit |
| RS19 | | P3 | The `detect_store --bench` example panics on an empty run | Audit |
| RUST-13 | | Low | The merged pauses' `last_seconds` is the highest rank's, not the newest | Run (T6) |
| RUST-16 | | Low | A replay error is retried at every flush with no backoff | Read |
| RUST-18 | | Low | A stream error has no counter or structured line on the server | Read |

RUST-17 (Low) is plausible and was never shown: flushes from the stream path are not bounded, which would matter with hundreds of ranks after a restart. It is not counted above.

## 5. Experiments, examples, CI and docs

| ID | Also known as | Priority | Issue | Checked |
|---|---|---|---|---|
| X02 | | P2 | Gate 1's older speed gate reads zero tool calls as an infinite speedup. Needs the author's amendment to the preregistration | Audit |
| X06 | | P2 | `trainer_check.py` counts string norms as recorded and then skips every comparison | Audit |
| X07 | CLI-18, CLI-23 | P2 | `dense_outlier_check.py` drops failed probes before the loop that prints their errors; `moe_dogfood.py` raises on a failed probe's record | Read |
| X08 | CLI-16 | P2 | `toy_moe.py` writes an evidence name the reader does not pair with a rank file; `smoke.sh` tests that name | Read |
| X09 | | P2 | The Megatron example does not record `update_successful`. The guide says so; the example is unchanged | Audit |
| X10 | | P3 | `capture_rerun.py` ignores an explicit step 0 | Audit |
| X11 | | P2 | CI does not run Python 3.10 or PyTorch 2.9, both of which the package declares | Audit |
| CLI-17 | | Medium | CI runs clippy with `--no-default-features` only, so the S3 and GCS code is built and tested but not linted. The `package` job now blocks (lane U8) | Read |
| CLI-19 | | Medium | `tensorquill runs` finds nothing in the folders `seeded_failures.py` writes | Read |
| CLI-24 | | Low | Comments and tool text name a kind count or a file that has changed | Read |

## 6. Raised by the fix lanes

These are not in either audit.

| # | Lane | Item |
|---|---|---|
| L1 | U1 | The tensor-subclass rule under `torch.compile` was run on `eager` and `aot_eager`, not on inductor |
| L2 | U1 | The writer's bounded wait for NVML can stop capture above about 200 steps a second |
| L3 | U4 | The server's ack carries no store ID |
| L4 | U4 | The server acks a dead stream's held offset |
| L5 | U5 | The drift check does not handle ranks on different device types |
| L6 | U6 | Python's readers do not look at `attempt_without_steps` |
| L7 | U6, U7 | For growing memory at one repeated step, Rust writes `steps_to_total: 0.0` where Python writes null |
| L8 | U7 | The MCP path that reads another rank from a store has no test |

## What to fix first

- **Before a release:** the five P1 rows: SDK-22, C03, C06, E1 and RS05. RS05 and E1 each need a decision first.
- **Before the GPU runs that read them:**
  - C20 puts evaluation time into the data-wait readings (the group C cell).
  - C14 can name a changed weight as unchanged (the stalled-weights seeded case).
- **Needs the author, not code:** X02 (an amendment), RS05 (an ordering policy), and the choice on C11 (leave the hooks, or accept a recompile to remove them).

## Closed, for the record

| Source | Closed |
|---|---|
| First review, high (27) | 22 held and are fixed. 4 did not hold: DET-02, DET-03, DET-06, RUST-02. RUST-05 was narrow and is fixed |
| First review, medium and low (60) | Fixed: SDK-18, SDK-19, CLI-11. Do not hold (T6): SDK-13, SDK-14, SDK-21, SDK-24, DET-08, CLI-10, RUST-12, RUST-15 |
| The sweep after the first review | All six ways to stop or hang training are fixed |
| Second audit (70 groups) | 30 fixed: C01, C02, C04, C05, C07 to C10, C17 to C19; C03 and C06 in part; RS01 to RS04, RS18; X01, X03, X04, X05; and eight evidence rows (five P1 and three small ones) |
