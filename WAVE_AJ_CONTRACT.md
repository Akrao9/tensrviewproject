# Wave AJ contract: a map entry the column cannot hold

Base: `trial/ai-wave` at `212205d`. Read by lanes AJ-1 (`flight.py`, `server_detect.py`), AJ-2 (`evidence.py`) and AJ-3 (`detect.rs`). The shared cases are in `sdk/tests/fixtures/wave_aj/cases.json`. Each case states what every path must report, and every path must give for `records` what it gives for `clean_records`.

## 1. The rule (A3, and A5's Rust half)

**One refused entry leaves the column alone.** For `grad_norms`, `weight_norms`, `update_rms`, `moe_top_k`, `moe` (column `moe_counts`) and `replica_spread`:
- An entry the column cannot hold is left out of the typed column.
- The rest of the map stays in the column.
- Before this, one such entry sent the whole map to `extra_json`, so the Rust detectors missed the rest.

**What a column holds** is what `flight.py`'s converter for it takes, plus Arrow's check. That check runs on every path: the sender's (`rows_to_send`) and the raw `detect` stream (`server_detect._stream`, through `flight.rows_batch`). Before, a value only Arrow refuses made the raw path raise.

| Column | An entry is held when it is |
|---|---|
| `grad_norms`, `weight_norms`, `update_rms` | a number that is not a bool, `null` (stored as null, as before), or one of the strings `"nan"`, `"inf"`, `"-inf"` |
| `moe_top_k` | an integer that is not a bool and fits int32 |
| `moe` | a list of integers (not bools) that fit int64. `[]` and negative integers are held |
| `replica_spread` | a list of numbers (not bools) or nulls. `[]` is held |

Every other entry is refused. This includes `true`, any other string (`"NaN"` too), a `null` under the list maps or `moe_top_k`, an integer too large for its type, and a name Arrow cannot encode.

**Where refused entries go.** They travel in `extra_json` under the record's own key, holding only those entries. For example, `{"moe": {"bad": "x"}}` sits beside the typed `moe_counts` column `{"good": [100, 0, 0, 0]}`.

**`grad_norms` keeps a refused entry in the column as null** (the author, Nov 22, through AJ-3). The entry is a gradient listed with no value, which is how Python reads the raw `true` or string.
- Its raw value still travels in `extra_json`: `{"good": NaN, "bad": null}` in the column, `{"grad_norms": {"bad": true}}` beside it. This holds even when every entry is refused.
- So in both engines an update comparison where either side has a refused gradient takes the whole-total fallback (Rust's `shared_totals` ends on a non-finite norm; Python's `_split_totals` on a value `_finite` refuses).
- A stalled weight's streak resets on it in both engines. `non_finite` still names a NaN sibling.
- A name Arrow cannot encode cannot sit in the column even as null. It travels only in `extra_json`. For that form, Rust reads `extra_json`'s `grad_norms` (AJ-3's `refused_gradients`, `28444b1`), so the whole-total fallback and the streak's reset hold there too. One side effect: a whole `grad_norms` map in `extra_json` from an older sender now resets the streak for every name in it, as Python does on the raw values.
- The other five maps leave a refused entry out of the column, so it is absent.

**When the map is not split.** If the value is not an object, or (outside `grad_norms`) no entry is held, the whole value goes to `extra_json` and the column is null, as before. A map that loses every entry is never sent as an empty typed map.

**Reading a row back.** `row_to_record` merges `extra_json[key]` into the typed map. So `row_to_record(record_to_row(r)) == r` holds for every shared case.

**A map with no refused entry** is converted by the same code as before, so its row is byte-identical. The seven recorded GPU captures are checked for this.

**`moved`** (the sender's count, `rows_to_send`) is unchanged in meaning:
- It names the record's key once per row when an entry was refused by Arrow or by an overflow.
- A type refusal (a bool, a string) is not counted, as before.

**The Arrow schema and the protocol number do not change.**

## 2. What both engines read

**Refused entries are absent to every detector**, as if the entry were not in the record. The exception is `grad_norms`, where a refused entry reads as a listed gradient with no value, as `null` does (section 1).
- Rust reads only the typed columns. Its cross-rank replica reader (`replica.rs`) falls back to `extra_json["replica_spread"]` only when the column is null.
- **Python must read the same** (AJ-2, `evidence.py`): an entry the table above would refuse is skipped, and nothing is raised.

**A map with no held entry reads as an absent key**, in both engines. In Rust the column is null.
- This is the A9 case: `weight_norms` whose only entry is refused is not a weight check, so `weight_jumps.checks` is 2, not 3.

**A top-level `grad_norm` whose value its float64 column refuses reads as an absent key** (wave AR, AQ5-D3, Nov 27, 2026), in both engines and to every rule that reads it: a bool, a text other than the sink's "nan", "inf" and "-inf", a list, an object, an integer too large for a float.
- It does not break a stall's streak, and it is no value of a spike or a trend.
- A non-finite value the column carries is a value, and `non_finite`'s.
- Only the readers of `grad_norm` were audited for this. The other top-level float fields (`clip_grad_norm`, `grad_scale`, `data_wait`, `step_time`, `lr`) rest on the generated sweep, which found no disagreement on them, not on an audit of their readers.

**MoE routers.** A router is a name with a held `moe` entry on some record. Python's `_router_sizes` must count only those names.
- Then the SDK's sample guard in `build_evidence(detectors="rust")` agrees with the server, because both see the routers the column holds.
- The guard itself is not changed. It still says "build the server from the same version", which is then true only of a real mismatch.
- Until AJ-2 lands, the `moe` case raises on the SDK's Rust path. AJ-1 covers it on the raw report and the live server.

**Stored is not the same as judged** (A2). `moe` `[5, -5]` is held by the column, so it never goes to `extra_json`. The detectors judge a sample only when its counts sum above 0 (Rust, `detect.rs` `tokens <= 0`). So `[5, -5]` (sum 0) is a missing sample.
- A router with no judged sample in the run is not read.
- Both engines report `not_checked` `moe_routing` naming it, and the run state is `degraded`.
- Python must match this (AJ-2). Today it reads the router as read because one count is above 0.
- Not decided here: a list with a negative count but a positive sum (`[5, -1]`) is judged by Rust as it stands. That is left to the author.

## 3. Outside this contract

**`scalars`: closed, the rule changed (the author, Nov 26, 2026, lane AR-1, AQ5-D7).** The author: "A refused entry of `scalars` is absent and its siblings are kept, in both engines." Lane W3's whole-map rule, kept through wave AJ and listed here as open, is gone.
- **What the column holds** is what it held before: an entry is held when it is a number that is not a bool and fits a float, `null` (stored as null), or one of the strings `"nan"`, `"inf"`, `"-inf"`, under a name Arrow can encode.
- **A refused entry leaves the column alone** (the sender's `_scalars`, on `rows_to_send` and on the raw `detect` stream). Refused means a text other than those three, an integer too large for a float, a list, an object, or a name Arrow cannot encode. The held entries stay in the column.
- **Where it goes.** It travels in `extra_json` under `scalars`, holding only the entries the column does not hold, as the maps of section 1 do. A bool goes there too, and an integer past 2**53 travels there exactly beside its rounded value in the column, both as before (wave AO).
- **When no entry is held** and one is refused, the whole map goes to `extra_json` and the column is null, as before. A map with nothing refused but bools makes the same row as before, so does a map held whole.
- **`moved`** names `scalars` once per row when an entry was refused by an overflow or by Arrow, as before. A type refusal (a text, a list) is not counted.
- **Reading a row back.** `row_to_record` merges `extra_json["scalars"]` into the typed map, so the record comes back whole. The refused entries come after the column's, as a bool does.
- **What both engines read.** A refused entry is absent from an incident's scalar context and from the median before it, and its siblings are kept: Python's `scalar_context` skips it, and Rust never sees it in the column.
- **A row stored before the change** has a null `scalars` column and the whole map in `extra_json`. Rust reads such a row's held entries from `extra_json` when the column is null, as Python reads the record back, so both show the siblings there too. A number too large for a float in that text is read as unreadable (`detect/extra_json.rs`), so the rest of the row is kept.
- **A server built before the change** reads the new sender's rows from the column: it sees the siblings and never the refused entry, which is what the new engines report.
- **A `null` entry** is held by the column as null, as before, and is absent from an incident's scalar context in both engines: Rust's context skipped it already, and Python's now does (lane AR-1's choice, seed 3569, which AQ-5 filed under D7; before, Python showed it as a value of `None`). A null is no value, as in the median before an incident in both engines.

**`records_to_batch` and the raw `detect` stream** now build rows as before and redo with Arrow's check only the rows Arrow refuses. A valid record's row is unchanged, and a key column Arrow refuses still raises.

## 4. The watcher across a resume (A1)

**`FollowedDetect` takes the last step read from the newest attempt**, as `detect --follow` does with `last_step`: the step of the last row in stream order. Before, it took the highest step over every attempt.
- An append to the newest attempt after a resume below an earlier attempt's top is judged as an append, not a full restart.
- A late row of a superseded attempt still restarts in full.
- The report is unchanged.

## 5. The four measurement blocks under `detectors="rust"` (A9, no code change)

**`build_evidence(detectors="rust")` takes its four measurement blocks from Python**, computed from the records. The blocks are `gradient_trend`, `attention_trends`, `update_trends` and `weight_jumps`. The comment at `evidence.py`'s `measurement_blocks` call says so ("made here from the records under either detectors").

**The raw Rust report must equal Python's for all four**, absence included: a block is absent in one exactly when it is absent in the other.

**The audit's disagreement came from A3.** A `weight_norms` map whose only entry is a non-number gave 2 checks in raw Rust and 3 in Python. After section 1 the map is not a weight check in either engine, so both must give **2**. This is shared case `weight_norms_only_entry_refused`.

## 6. Skipped updates in `update_trends` (A7, decided Nov 22; lanes AJ-2 and AJ-3)

- An optimizer update that was skipped in full neither enters the `update_trends` baseline nor is reported as the lowest ratio.
- In a partial skip across several optimizers, an update that did happen still counts, when the record identifies it.
- A parameter whose optimizer the record cannot identify is left unjudged for that step.
- **As written for both engines (AJ-2):**
  - A record whose `step_skipped` is true is left out. It enters no parameter's baseline and gives no ratio.
  - That holds whoever reported the skip.
  - It also holds on a partial skip (`step_skipped_part`), because no record names which optimizer applied its update to a parameter. Every parameter of such a step is unjudged there, as `update_spike` reads it.
- Shared cases: AJ-2's `FULL_SKIP`, `SKIPS_FIRST`, `PARTIAL_SKIP` and `CALLER_SKIP` (`876f627`), plus only-skipped-updates, in `cases.json` as `a7_*`.

## 7. `weight_jumps` across a resume (A10, decided Nov 22; lanes AJ-2 and AJ-3)

- `weight_jumps` resets when the attempt changes.
- The first weight check after a resume is the new baseline.
- Only a jump inside the new attempt is listed.
- This applies to `weight_jumps` only.
- **As written for both engines (AJ-2):**
  - Two consecutive weight checks are compared only within one attempt.
  - The first check after a reload starts the comparison again, and `checks` still counts every weight check.
  - A weight check is a record whose `weight_norms` holds at least one value the float64 column carries (a null counts).
- Shared cases: AJ-2's `RELOAD` and `RELOAD_ONLY` (`876f627`), in `cases.json` as `a10_reload_then_jump` and `a10_reload_only`.

## 8. Open between AJ-2 and AJ-3

- **Closed: a held `replica_spread` entry `[]`** covers nothing, in both engines.
  - Rust, AJ-3 `1ab6a2a`: `replica.rs` matches AJ-2's rule. Its `extra_json` fallback reads entries as Python's `_spread_values` does.
  - On a joined tree, the `a7_*`, `a10_*` and `replica_spread_empty*` paths pass with `--runxfail` (AJ-3: 25 of 25).
- **Open, for the author: an integer too large for a float, carried into `extra_json`.** An example is `10**400`, written as a JSON integer.
  - Rust's `serde_json` refuses the whole `extra_json` text. That row's replica observation, `step_skipped_by`, `moe_source` and every other `extra_json` key are then lost, not only the entry.
  - This predates wave AJ. Lane AD4 already sent such values in `extra_json`, whole maps included. Wave AJ narrows what travels there, but does not change how it is written.
  - One option is for the sender to write such a number as text, which Rust's readers skip as unreadable. That would break `row_to_record`'s round trip for that value, so it is not done here.
  - Another option is for the Rust readers to parse `extra_json` with big numbers allowed.
  - AJ-3's test is `#[ignore]`d with the reason, in `replica/lane_aj3_replica_tests.rs`.
- **AJ-3's `update_spike` cases** are in `cases.json`, with explicit results and no clean control (deleting the refused entry also deletes the evidence that the map was incomplete).
- **The stalled-weight streak** resets on a refused gradient on every path. This was run, not only read: shared case `stalled_weight_streak_resets_on_a_refused_gradient` puts `stalled_weights` at step 4, as with a null, where an unlisted gradient puts it at step 3. Python, raw Rust, the SDK in Rust mode and the live server all agree. AJ-3's test 68aeb7b ("holds") was written for the earlier representation.
