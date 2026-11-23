# Wave AJ contract: a map entry the column cannot hold

Base: `trial/ai-wave` at `212205d`. Read by lanes AJ-1 (`flight.py`, `server_detect.py`), AJ-2 (`evidence.py`) and AJ-3 (`detect.rs`). The shared cases are in `sdk/tests/fixtures/wave_aj/cases.json`. Each case states what every path must report, and every path must give for `records` what it gives for `clean_records`.

## 1. The rule (A3, and A5's Rust half)

**One refused entry leaves the column alone.** For `grad_norms`, `weight_norms`, `update_rms`, `moe_top_k`, `moe` (column `moe_counts`) and `replica_spread`:
- An entry the column cannot hold is left out of the typed column.
- The rest of the map stays in the column.
- Before this, one such entry sent the whole map to `extra_json`, so the Rust detectors missed the rest.

**What a column holds** is what `flight.py`'s converter for it takes, plus Arrow's check on the sender's strict path:

| Column | An entry is held when it is |
|---|---|
| `grad_norms`, `weight_norms`, `update_rms` | a number that is not a bool, `null` (stored as null, as before), or one of the strings `"nan"`, `"inf"`, `"-inf"` |
| `moe_top_k` | an integer that is not a bool and fits int32 |
| `moe` | a list of integers (not bools) that fit int64. `[]` and negative integers are held |
| `replica_spread` | a list of numbers (not bools) or nulls. `[]` is held |

Every other entry is refused. This includes `true`, any other string (`"NaN"` too), a `null` under the list maps or `moe_top_k`, an integer too large for its type, and a name Arrow cannot encode.

**Where refused entries go.** They travel in `extra_json` under the record's own key, holding only those entries. For example, `{"grad_norms": {"bad": true}}` sits beside the typed `grad_norms` column `{"good": NaN}`.

**When the map is not split.** If the value is not an object, or no entry is held, the whole value goes to `extra_json` and the column is null, as before. A map that loses every entry is never sent as an empty typed map.

**Reading a row back.** `row_to_record` merges `extra_json[key]` into the typed map. So `row_to_record(record_to_row(r)) == r` holds for every shared case.

**A map with no refused entry** is converted by the same code as before, so its row is byte-identical. The seven recorded GPU captures are checked for this.

**`moved`** (the sender's count, `rows_to_send`) is unchanged in meaning:
- It names the record's key once per row when an entry was refused by Arrow or by an overflow.
- A type refusal (a bool, a string) is not counted, as before.

**The Arrow schema and the protocol number do not change.**

## 2. What both engines read

**Refused entries are absent to every detector**, as if the entry were not in the record.
- Rust reads only the typed columns. Its cross-rank replica reader (`replica.rs`) falls back to `extra_json["replica_spread"]` only when the column is null.
- **Python must read the same** (AJ-2, `evidence.py`): an entry the table above would refuse is skipped, and nothing is raised.

**A map with no held entry reads as an absent key**, in both engines. In Rust the column is null.
- This is the A9 case: `weight_norms` whose only entry is refused is not a weight check, so `weight_jumps.checks` is 2, not 3.

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

**`scalars`** keeps lane W3's rule.
- A bool entry is dropped from the row and does not travel in `extra_json`.
- Any other refused entry (a string, a number too large) still sends the whole `scalars` map to `extra_json`.
- This is unchanged in wave AJ, and listed as open.

**`server_detect._stream`** (the raw `detect` path) builds rows without Arrow's check, as before. A value only Arrow refuses (for example a `moe_top_k` of 2**40) is caught on the sender's path, not there.

## 4. The watcher across a resume (A1)

**`FollowedDetect` takes the last step read from the newest attempt**, as `detect --follow` does with `last_step`: the step of the last row in stream order. Before, it took the highest step over every attempt.
- An append to the newest attempt after a resume below an earlier attempt's top is judged as an append, not a full restart.
- A late row of a superseded attempt still restarts in full.
- The report is unchanged.
