# Wave AL contract: protocol 5

Base: `trial/ai-wave` at `f71d8e0` (lane AL-0's branch; `trial/ai-wave` has since moved to `5749a89`, tests only). This is Phase 2 item 1 of the v1 plan, in two gated steps under one protocol bump: the typed columns and the probe split first (sections 1 to 3), the scalar-only record second (section 4). The pinned-memory flush, `grads=N`, K2, K3, the catalog and compaction are not in it.

- **Shared cases:** `sdk/tests/fixtures/wave_al/cases.json`, written by `experiments/protocol-v5/build_cases.py`. Each case says what every path must report, and what the base reports today. `sdk/tests/test_wave_al_contract.py` reads them.
- **Measurements:** `experiments/protocol-v5/` (`prototype.py`, `captures.py`, `live.py`, `measure.py`, `rust-reader/`), results in `results/`. Every one ran on this Mac's CPU against the protocol 4 release binary built from this tree, with no product code changed. A "v5" figure is the prototype's layout, sent to and stored by the protocol 4 server.
- **Read by:** the coordinating session that cuts the lanes, and the lanes of section 9.

## Where each decision stands

**Decided** means the code or a measurement settles it; the evidence column says which. **Owner** means a product choice. The owner decided all nine on Nov 24 (the next section), and the same day brought one fault left open by wave AK into this wave (row L1, section 4.8).

| # | Decision | Stands | Evidence |
|---|---|---|---|
| D1 | An Arrow map keeps its entries' order from pyarrow through Flight, the server's Parquet part and `Storage::read_part`, and after a restart's rebuild. No other order carrier is needed for `layers`. | Decided | `results/live.json`, `order` (sent `…2, 10, 11, 3`, read back the same by pyarrow and by the server's own `read_part`, on the part and after a restart) |
| D2 | `layers` has four fields and `moe_stats` five: `routed_tokens`, `entropy`, `diversity`, `dead_experts`, `dead_frac`. The architecture page's `moe_stats` struct is wrong. | Decided | `capture.py` `STATS` and `_router_stats`; every record of the seven captures (`results/captures.json`) |
| D3 | `token_entropy` (K3) is not reserved. Readers look struct fields up by name, and K3 adds the field when it lands. | Decided | `results/live.json`, `struct_field_added`: one pyarrow reader and the Rust reader read a part without the field as absent and one with it as its value |
| D4 | Wave AJ's rule extended to struct entries, field by field, with a null struct keeping an unreadable entry's place (section 1.4). | Decided | the order rule (D1) and the round trip on every case and capture |
| D5 | `row_to_record(record_to_row(r)) == r` with every map's order kept, on all 3,800 records of the seven captures. On the 8,463 records of the repository's shared case files, the round trip is the same as today's. | Decided | `results/captures.json`: 0 not equal, 0 out of order. `results/fixtures.json`: 8,446 the same as today's round trip, 17 refused by both (wave AF's invalid identities), 0 different |
| D6 | The batch metadata `schema_version` goes from 1 to 2 for protocol 5's layout. | Decided | `results/live.json`: a protocol 4 server given the typed layout under version 1 stores it and judges no layer (`old_server_typed`); under row version 2 alone it drops the rows and acknowledges them, so they are lost to it (`old_server_row_v2`); under metadata version 2 it refuses the stream and acknowledges nothing (`old_server_meta_v2`) |
| D7 | A row of protocol 5's layout carries row `schema_version` 2. | Decided | `results/live.json`, `old_python_on_v5_part`: today's `load_records` reads a typed part with row version 1 as a run with no layers (`activation_spike` `not_configured`, no incident); with row version 2 it refuses the part naming it |
| D8 | A scalar-only row is never in a `part-*` file and never moves the step watermark. | Decided | `results/live.json`, `watermark`: a row of step 1000 in a part makes the server drop steps 4 to 9 after a restart, from the part's footer |
| D9 | The server reads protocol 4 parts for as long as they are stored. It does not convert them. | Decided | the store is append-only (section 20 of the ledger): a converted copy would be a second `part-*` file over the same steps, which `rank_parts`, `max_step` and `_parquet_parts` all read |
| D10 | Attention maps are stored as `List<List<Float32>>`, ragged, not as a fixed-size list. | Decided | `probe.py`: a map is `min(rows, 64)` by `min(keys, 64)`; all 524,288 map values in the three probe captures are exact float32 and finite (`results/captures.json`) |
| D11 | No detector, MCP tool, `compare`, `compact` or Gate 1 shown file reads a map or a histogram. | Decided | section 2.2's reader list |
| D12 | None of the 33 kinds judges a logged scalar. Scalars are incident context, `compare`'s series, `get_series`, the Rerun export and Gate 1's arm A CSV. | Decided | section 4.6 |
| D13 | `PROTOCOL` goes from 4 to 5 in `server_detect.py` and `main.rs`. | Decided | what `detect` reads changes (section 3) |
| D14 | The before-numbers and the acceptance figures of section 6. | Decided | `results/ack.json`, `results/live64.json`, `results/decode.json` |
| Q1 | What makes an older SDK refuse protocol 5's data: only what changed is version 2 (Q1 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q2 | The heavy probe arrays: columns in the same row (Q2 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q3 | The scalar-only record travels in the same `.jsonl` and stream, with `seq` (Q3 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q4 | Scalar-only records are series only, never incident context (Q4 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q5 | With no step in memory and no `step=`, the value is left out with one warning per session, which shows the explicit `session.log({...}, step=N)` for a baseline evaluation and for an evaluation after `flush()`. The Hugging Face callback passes `step=` itself, so `eval_on_start`'s baseline is recorded. The mirror's case is a documented limit. (e) is not taken now and stays compatible later (section Q5). | **The coordinator's choice on the owner's word, Nov 24** | owner: "keep the one-time warning and show how to record the value explicitly. Cover both a baseline evaluation before training and an evaluation after flush()", then "which is best decide" |
| Q6 | At one series point, the scalar-only record with the highest `seq` wins (Q6 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q7 | The mirror ignores a tracker's own `step=` (Q7 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q8 | `compare` reads scalar-only records as series points (Q8 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| Q9 | A downgrade is documented as unsupported (Q9 a). | **Owner, decided Nov 24** | "Yes, all but Q5", relayed by the coordinator |
| L1 | A value logged late inside a compiled function reaches its step's own record when a frozen parameter becomes trainable before the next step ends (step 2, lane AL-5). The stop of capture stays a stated limit (section 4.8). | **Owner, decided Nov 24** | "this will be fixed in AL", relayed by the coordinator; the place and the limit are this contract's recommendation |

## The owner's decisions

**The owner answered on Nov 24: "Yes, all but Q5".** Q1 to Q4 and Q6 to Q9 are decided as the recommended option (a) of each. The alternatives are kept as one line each, for the record. Q5 was decided later the same day (section Q5).

- **Q1 (a), decided: only what changed is version 2.**
  - Rows of protocol 5's layout carry row version 2, and a `record_version` column (Int32, null for a record with no `schema_version` key) holds the record's own version.
  - A step record in the `.jsonl` stays version 1; a scalar-only record is version 2.
  - Not taken: (b) every record and row version 2.
- **Q2 (a), decided: the heavy probe arrays go to `probe_maps` and `probe_histograms` in the same row.**
  - A replay projects them away. Section 2.5 (sidecars) is kept only as the record of what was weighed.
  - Not taken: (b) sidecar objects.
- **Q3 (a), decided: the scalar-only record travels in the same `.jsonl`, stream and byte cursor, with a `seq` column.**
  - Not taken: (b) a file and stream of its own.
- **Q4 (a), decided: scalar-only records are series only.** An incident's `scalars` context is read from its step's training record, as today, and the detectors never read scalar rows.
  - Not taken: (b) incident context that the live server re-attaches.
- **Q6 (a), decided: wherever values meet in one series point, the scalar-only record with the highest `seq` wins.**
  - Not taken: (b) the training record's value wins.
- **Q7 (a), decided: the mirror ignores a tracker's own `step=`.**
  - Not taken: (b) passing it on.
- **Q8 (a), decided: `first_divergence` reads scalar-only records as series points.**
  - Not taken: (b) `compare` leaves them out.
- **Q9 (a), decided: the README says a downgrade is not supported, and how it shows.** A protocol 4 server on a store with protocol 5 parts reads them silently, with no layers, no MoE statistics, no maps, and no `scalars-*`.
  - Not taken: (b) a `part5-*` prefix.

### Q5, decided Nov 24: a value with no step in memory and no `step=`

**The owner's words:** "For Q5, when there is no training step in memory and the caller supplies no step=, keep the one-time warning and show how to record the value explicitly. Cover both a baseline evaluation before training and an evaluation after flush()".
- Asked to weigh this against (e), the owner left the choice to the coordinator ("which is best decide").
- **The coordinator's choice, on the owner's word:** the owner's wording, plus one addition, the adapter's explicit step. (e) is not taken now.
- **Why:** only a value logged under "eval", or pending at `close()` after a flush, reaches the switch below. A plain `log()` before the first step, which is a one-line mirror user's baseline, already lands on step 0's record. So the values at stake come from code that already calls the session's API, or from TensorQuill's own adapters, and the addition covers the adapters.

**What reaches `Session._no_step_for`**, which stays the one switch:
- a value logged under `phase("eval")`, by the user or by the mirror, before any step was committed, or after `flush()` with no step since;
- a value pending at `close()` with no step in memory.

A plain `log()` outside "eval" before the first step is not one of these. It is pending and lands on step 0's training record, as today.

**What happens to it:** it is left out, with one warning per session, as in wave AK. No record is written. The warning now shows the explicit call that records it: `session.log({...}, step=N)`, which writes a scalar-only record (section 4.2). The Hugging Face callback does that itself (below).

**Which step the warning suggests.** Both numbers are host integers the session already keeps, the ones `_record_manifest` uses for `after_step`, so no device is read.

| Case | When `_no_step_for` sees it | `N` in the text | From |
|---|---|---|---|
| A baseline evaluation before training | no step committed yet (`_steps_committed == 0`) | the step training starts at: `start_step`, 0 unless resumed | `self._next_step` |
| An evaluation after `flush()`; a value pending at `close()` after a flush | a step committed, none in memory since | the last step training finished | `self._next_step - 1` |

A value pending at `close()` with no step at all (a session that never trained) is the first case.

**Why the text names a number in both cases.**
- After `flush()`, the last finished step is what the value followed, and it is the step wave AK puts such a value on while that step is in memory.
- For a baseline, `start_step` is where a trainer plots one. The number is given as the step about to start, so the user can choose another knowingly.

**The text, step 2 (lane AL-5).** `<names>` is as today: the first five names quoted, then "and K more". `<first>` is the first name.
- **Before the first step:** `TensorQuill left out the scalar <names>: it was logged before the first training step (a baseline evaluation), when no step is in memory. To record it, log it with an explicit step, which writes a scalar-only record: session.log({<first>: value}, step=<start_step>), the step training starts at. Said once per session.`
- **After `flush()`:** `TensorQuill left out the scalar <names>: it was logged after flush() with no training step since (an evaluation after flush()), when no step is in memory. To record it, log it with an explicit step, which writes a scalar-only record: session.log({<first>: value}, step=<last step>), the last step training finished. Said once per session.`
- **Pending at `close()` after a flush:** the same, with "it was still pending at close() after flush() with no training step since" in place of the first clause.
- **A value the mirror brought** gets this sentence before "Said once per session.": `It came from your tracker's log call through mirror_scalars, which cannot pass a step (the tracker's own step is ignored): call session.log(..., step=...) yourself for it.` This is a documented limit (Q7).
  - The mirror tells the session that the value is its own. Today its call with nowhere to go is a plain `session.log(dict(values))` (`mirror._log_into`), so `mirror.py` passes a mark that `_no_step_for` reads. That is lane AL-5's.

**Said once per session, as today.** The text is the case of the first value left out. Later ones are left out silently.

**Step 1 leaves the text as wave AK wrote it.** That text names protocol 5's record and no argument, and `step=` still raises `TypeError` until step 2. The push follows step 2, so no build tells a user to pass an argument that raises.

**The addition: the Hugging Face callback passes `step=` itself** (lane AL-5, step 2). It is the only TensorQuill adapter that records evaluation metrics after wave AK. Lightning's callback logs the loss only.
- **When:** in `on_evaluate`, when the session has no step in memory. That is `eval_on_start`'s baseline, or an evaluation after a `flush()` with no step since. It calls `session.log(values, step=N)`, which writes a scalar-only record, so nothing is left out and no warning is said.
  - With a step in memory, it logs as today: on the newest step, under "eval".
- **Which `N`.** `Trainer` counts finished optimizer steps in `state.global_step`. The callback starts the session at `start_step = state.global_step` (`on_train_begin`), and the session numbers each step it ends from there. So after TensorQuill's step `n` has ended, `global_step` is `n + 1`.
  - **No step ended in this session** (a baseline, also on a resume): `N = state.global_step`, which is `start_step` and the session's `_next_step`. It is 0 on a fresh run, the resumed step otherwise.
  - **After a flush:** `N = state.global_step - 1`, the last step that ended, which is the session's `_next_step - 1`.
  - The callback passes the session's own number. Lane AL-5's test shows the two agree on a real `Trainer` (with gradient accumulation too, since both count optimizer steps).
- **What shares a step.**
  - A baseline at `N = 0` shares its x with training step 0's record (`train/loss` and the rest). The names differ, so the series differ.
  - When W&B or MLflow is active, the mirror also brings the `Trainer`'s logged baseline. It arrives outside "eval", before `on_evaluate`, so it is pending and lands on step 0's own record, as today. The callback's record holds the same values under the same names.
  - By Q6 (decided) the scalar-only record wins at that series point. It is the same value.

**What remains, stated as limits:**
- **The mirror's values.** A value the mirror brings under "eval" with no step in memory is left out with the warning, which says so (Q7: the mirror passes no step). A documented limit.
- **One point at step 0.** A baseline recorded at `step=0`, and an evaluation after step 0 recorded on step 0's own record under the same name, are one series point. By Q6 the scalar-only record wins, so the series shows the baseline and not the evaluation after step 0. The evaluation is still on step 0's record and in incident context.
  - It happens only when an evaluation follows the very first step (`eval_steps=1`, or the user's own loop).
  - It needs one README line, in the `scalars` passage (section 9).

**Not taken now: (e), compatible later.**
- **What (e) was:** a record at `start_step` marked `"before_step": true` for a value before the first step, the last finished step otherwise, and `"pending": true` for a value from `close()`.
- **What it would add:** it would record everything `_no_step_for` leaves out, the mirror's values included, and it would remove the step-0 limit above (a "before" point sorts ahead of step 0's own).
- **What it would cost:** one optional key, a reader ordering rule, and a placement the user did not state.
- **Compatible later:** both keys are optional additions to a version-2 record, so they need no version change, and readers that ignore them read the record as today.
- **What would make it worth taking:** a real job whose evaluations reach the session under "eval" with no step in memory and are lost. For example, a mirror user whose own loop sets `phase("eval")` around evaluations logged only to W&B. Or a user who meets the step-0 point.

**Weighed and not taken:** (d) a record with a null step before the first step. `step` is a key in `identity.check`, the sender, the Arrow schema and the server's null-key check, so all four and every series reader would change.

## 1. `layers` and `moe_stats` as typed columns (lanes AL-1, AL-3)

### 1.1 The real entries

| Map | Fields the capture writes, in order | Type | Read by the detectors |
|---|---|---|---|
| `layers` | `mean`, `std`, `abs_max`, `zero_frac` (`capture.py` `STATS`) | float | Rust: `std`, `abs_max`, `zero_frac`, and every key for `non_finite` (`LayerVisitor`). Python: the same three (`_Activations`), every value for `non_finite` (`_NonFinite`) |
| `moe_stats` | `routed_tokens`, `entropy`, `diversity`, `dead_experts`, `dead_frac` (`_router_stats`) | int, float, float, int, float | `diversity` only, in both engines (`RouterStats`; `evidence.py` near line 7345) |

- **Other readers.** MCP `get_series` (`layers` stats and `moe_stats` diversity), `compare`'s `first_divergence` (every `layers` stat, with the layers' order as capture order: `divergence.py` `layer_order`), the Rerun export (every numeric field).
- **The captures agree.** 15,600 entries of `layers` and of `moe_stats` in each Granite capture with grads, 12,800 `layers` entries in each dense and `Trainer` capture. Every field is present with the type above. `granite-router-only.jsonl` writes `layers: {}` on all 650 records.
- **The architecture page's `moe_stats` struct** (`routed_tokens, diversity, dead_frac, token_entropy`) lacks `entropy` and `dead_experts`. It carries K3's `token_entropy`, which is not built.

### 1.2 The columns

- **`layers: Map<Utf8, Struct<mean: Float64, std: Float64, abs_max: Float64, zero_frac: Float64>>`.** The map's values are nullable, and so is every field.
- **`moe_stats: Map<Utf8, Struct<routed_tokens: Int64, entropy: Float64, diversity: Float64, dead_experts: Int64, dead_frac: Float64>>`**, nullable the same way.
- **`layers_json` and `moe_stats_json` are not written** in protocol 5's layout. They stay readable (section 3).
- **Fields are looked up by name** in every reader, Python and Rust. A field a part does not have reads as absent. When K3 lands it adds `token_entropy: Float64` to `moe_stats`, with no row-version change. `PROTOCOL` goes up with K3's new kind anyway. Its parts and older ones are read alike (D3).
  - Reserving the field now would cost a null column in every part.
  - It would also need a rule for an explicit null. If K3 is cut, the field would be dead in every part for good.

### 1.3 Order

- **The converter writes the entries in the record's order** (the model's, for `layers`) and nothing sorts them. The map is not marked `keys_sorted`.
- **Every reader reads entries in stored order.** Shown on a part and after a restart: D1.
- Phase 4's compaction must keep both row order and entry order.

### 1.4 An entry or field the column cannot hold (wave AJ, section 1, extended)

**A field holds** what `flight._float` takes, for a Float64 field: a number that is not a bool, or one of `"nan"`, `"inf"`, `"-inf"`, held as the non-finite float. An Int64 field holds an integer that is not a bool and fits int64.

| What the record holds | The column | `extra_json` |
|---|---|---|
| Every field held | the struct | nothing |
| A field refused: a bool, another string (`"NaN"`), an explicit `null`, a number too large, a float in an Int64 field, or a field the struct does not name | the struct, that field null | `{"layers": {"<name>": {"<field>": <value>}}}`: the refused fields only |
| A field missing | the struct, that field null | nothing; it reads back absent |
| An entry that is not an object (`5`) | a null struct, in its place | `{"layers": {"<name>": 5}}` |
| A name Arrow cannot encode (a lone surrogate) | not in the map | `{"layers": {"<name>": …}}`; read after the column's entries, so it loses its place. No capture writes one |
| A value that is not an object (`[1, 2]`) | null | the whole value under `layers` |
| `{}` | an empty map, not null | nothing |

- **An integer in a Float64 field is held as a float.** The round trip then holds by `==` (`0 == 0.0`) and not as JSON text, as for wave AJ's maps.
- **`moved` names `layers` or `moe_stats` once per row** when a field was refused by an overflow or by Arrow, as in wave AJ. A type refusal is not counted.
- **`row_to_record` puts the record back.** It merges an object of refused fields into its entry, replaces a null struct with its value, and appends a name the map does not hold. `row_to_record(record_to_row(r)) == r` for every case, with the map's order kept.

### 1.5 What both engines read

- **A refused field is absent to every detector.** An entry that is not an object is absent too. These are the activation kinds, `dead_activations` and the router's `diversity`. A layer missing `std`, `abs_max` or `zero_frac` is skipped, as today.
- **`non_finite` is the exception: it reads every field of a layer, the refused ones included.** An unknown field `"p99": "nan"` names the layer in both engines.
  - Today Python reads every value, and Rust's visitor ORs every key.
  - Rust then parses `extra_json` on a row whose text names `"layers"`, as `refused_gradients` does for `grad_norms`.
**Three cases are targets; today they differ from what the clean records give** (`cases.json`, `today`):
- **`layers_entry_not_object`:**
  - Python raises `AttributeError: 'int' object has no attribute 'get'`.
  - Rust's `Members<LayerStats>` fails on the entry and drops every layer of every row: the activation kinds are `not_configured`.
- **`layers_value_not_object`:** Python raises `AttributeError: 'list' object has no attribute 'items'`. Rust already reads no layers.
- **`layers_name_arrow_cannot_encode`:** Rust loses every layer. Arrow refuses the key, so the whole map goes to `extra_json` today. Under section 1.4 only that entry does.

**Who fixes them:**
- Python (lane AL-4′, in step 2 with the other Python readers) skips an entry that is not an object, and a `layers` value that is not an object, as absent.
- Rust (lane AL-3, with AL-1's layout) reads the rest of the map.

**Every other case gives today's results in both engines,** and the tests pin them (`-pin` in their IDs). That includes `moe_stats_diversity_refused`:
- Python reads `float(true)` as a diversity of 1.0, where Rust takes it from the counts. Both still report the same `moe_imbalance`.
- The difference shows only in the per-step `samples` (`build_evidence` against `detect --samples`). Lane AL-4′ makes Python fall back to the counts there as Rust does, and adds that comparison as its test.

### 1.6 The round trip on the recorded captures

`captures.py` sent every record of the seven captures (`granite-full`, `granite`, `granite-no-grads`, `granite-router-only`, `dense`, both `trainer`) through the prototype and back through a real Arrow batch: 3,800 records, none unequal, none with a map out of order. The rows that carry `extra_json` (13, 13, 0, 0, 8, 8, 8) hold `optimizer_ownership` and `weight_dtypes`, as today; none holds a refused field.

### 1.7 The cases

`layers_model_order`, `layers_empty_map`, `layers_non_finite_strings`, `layers_field_bool`, `layers_field_other_string`, `layers_field_missing`, `layers_field_extra`, `layers_field_null`, `layers_entry_not_object`, `layers_name_arrow_cannot_encode`, `layers_int_in_float_field`, `layers_value_not_object`, `moe_stats_real_fields`, `moe_stats_float_in_int_field`, `moe_stats_diversity_refused`, `moe_stats_int_past_int64`, `moe_stats_token_entropy_by_name`.

## 2. The heavy probe arrays (lanes AL-1, AL-2, AL-3)

### 2.1 What a probe holds

`probe.py` writes one of three things:
- `{histograms, attention, attention_note?, head?}`.
- `{not_run}`: the capture's note that a sharded model is not probed.
- `{error}`: the probe failed.

Its parts:
- **`histograms`:** `{block: {lo, hi, counts[64], nonfinite}}`.
- **`attention`:** `{label: {max_logit[h], entropy[h], entropy_share[h], masked_rows, invalid_rows, rows, query_len, key_len, map}}`, or `{label: {error}}`.
  - Labels are block names, or `name#n` for a second call in one block.
  - A map is `min(rows, 64)` by `min(keys, 64)` floats.
- **`head`:** `{module, found_by, …numbers}`, or `{…, not_read}`.

In the dense capture a probe record is 1.7 MB of JSON. 98% of it is the 32 maps (6.6 MB of 6.8 MB over its four probes); the histograms are 0.4%.

### 2.2 Who reads what

| Part | Read by |
|---|---|
| Maps | the Rerun export (`evidence.py` `_log_probe`, from records: a `.jsonl` or a store read by `load_records`). Nothing else: not `detect.rs` (its `Probe` visitor passes the map with `IgnoredAny`; `HeadTrends` parses the whole text as a `Value` when it holds `"head"`), not Python's detectors, not MCP `get_series` (`_probe_point`: "never … the attention map"), not `compare`, `compact`, `runs`, nor any Gate 1 shown file (the bake-off's fixtures run no probe) |
| Histograms | the Rerun export; `sdk/examples/dense_outlier_check.py` (bounds); `gpu_probe_coverage.py` and `moe_dogfood.py` (how many). No detector, no MCP tool |
| Per-head `max_logit`, `entropy_share` | both engines' attention kinds and `attention_trends` (`detect.rs` `Attention`, `Measurements::probe`; `evidence.py`), MCP `get_series` (`_attention_values`), the Rerun export |
| Per-head `entropy`, `masked_rows`, `invalid_rows`, `rows`, `query_len`, `key_len` | MCP `get_series` (`masked_rows`, `invalid_rows`, `rows`); the Rerun export |
| `head` | both engines' `head_trends` (`HeadTrends`; `evidence.py` near line 1324), MCP `get_series` and `_not_read` |
| `not_run`, `error` (the probe's or an entry's) | both engines' probe `not_checked` and attention coverage (`Attention::row`; `evidence.py` near line 2113), MCP `_probe_point` |

### 2.3 The two options, measured

**What the probe costs.**
- On one rank (`results/ack.json`; 7 rounds, load 2.3 to 2.6):
  - Today, a 50-step batch holding one of the dense capture's probes is acknowledged in 14.1 ms (median; 11.2 to 39.8, one outlier above 23). The same batch without the probe takes 5.3 ms.
  - Its part is 743 KB, against 151 KB without the probe, and the server's Parquet write alone takes 5.2 ms (5.1 to 6.4), against 1.7 ms.
- Under 64 streams at the capture's 65 ms a step (`results/live64.json`):
  - Today: p99 170 ms and worst 185 ms (medians of 7 rounds, load 2.2 to 8.2), against 82 and 92 ms with no probe. One round reached 283 ms, its only acknowledgement over 250 ms.
  - The prototype: p99 111 ms and worst 115 ms, none over 250 ms.
  - An earlier run of 5 rounds (load 2.8 to 9.2) had one round reach 379 ms, with 42 acknowledgements over 250 ms. That round itself began at load 2.8, and nothing like it recurred in these 7 rounds.
  - This is the plan's "acknowledgement spike", reproduced at under a third of the size the plan states (section 8).

**Prototype columns (Q2 a), measured.**
- The same batch, with maps and histograms in typed columns, is acknowledged in 9.1 ms (7.8 to 12.8).
- Its part is 539 KB (−27%), and its write takes 4.8 ms.
- The Arrow batch the sender builds is 1.42 MB, against 2.66 MB.
- `probe_json` falls to about 60 KB a probe. The detectors still parse that text, which no longer holds a map.

**Sidecar (Q2 b), estimated, not measured.** It needs product code to measure.
- The part would be close to the no-probe part, about 151 KB.
- The maps would still cross Flight, unless sent apart.
- Each probe would cost one more create-only `PUT`: two round trips on S3 with `If-None-Match`, one on local disk.

### 2.4 The layout (Q2 a)

- **`probe_json`** holds the probe without the maps it moved and without `histograms` when it moved them. Everything else stays where it was: per-head lists, row counts, `attention_note`, `head`, `not_run`, `error`, and an entry's `error`.
- **`probe_maps: Map<Utf8, List<List<Float32>>>`**, keyed by the module's label.
  - It holds a module's `map` when every value is exactly a float32, or one of the three non-finite strings.
  - Otherwise that map stays in `probe_json` (case `probe_map_not_float32`).
  - The capture's maps are float32 tensors, so all of them are held (D10).
- **`probe_histograms: Map<Utf8, Struct<lo: Float64, hi: Float64, counts: List<Int64>, nonfinite: Int64>>`**, all or none: one entry it cannot hold keeps every histogram of that probe in `probe_json`, so the map's order is never split (case `probe_histogram_irregular`).
- **`row_to_record`** puts `histograms` first, as the capture writes it, and each map back on its module's entry.
- **No detector reads `probe_maps` or `probe_histograms`.**
  - A replay must not decode them: `Storage::read_part`'s callers in `detect::replay` read with a projection that leaves them out (`ProjectionMask::roots`, arrow-rs `parquet/src/arrow/mod.rs`).
  - On the live path the batch is already decoded from Flight, so nothing changes there.
- **The `.jsonl` file holds the whole probe, as today.** The Rerun export of a local run, every golden file and Gate 1 are unchanged.

### 2.5 If the owner chooses sidecars (Q2 b): the rules a sidecar must keep

- **The object:** `runs/<run>/probes/rank=<r>/attempt=<a>/probe-<step:08>.bin.zst`, written by the server.
  - It is put with `PutMode::Create` before the batch that carries the probe is acknowledged, and never deleted.
  - The row's `probe_json` holds the light part and a reference `{"sidecar": "<key>", "bytes": N, "sha256": "<hex>"}`.
- **A replay that sends the line again** meets `AlreadyExists`.
  - Same digest: the object is stored, and the acknowledgement goes on.
  - Another digest: the first object is kept, the new one counted (`probe_conflicts` in the rank's status), and the row keeps the stored object's reference.
- **The sender's byte cursor and the `.jsonl` are unchanged.** The line holds the whole probe, and an acknowledgement of its offset means the row and the object are both stored.
- **A reader that finds the row and not the object** says so: `maps: "not stored"`, in the Rerun export's log and in `get_series` if it ever reads maps. It never raises and never guesses.
  - A put fails before the acknowledgement, so the sender sends the line again. Only a store that lost the object afterwards has "never".

### 2.6 The cases

`probe_split_maps_and_histograms`, `probe_map_not_float32`, `probe_entry_error`, `probe_not_run`, `probe_error_whole`, `probe_histogram_irregular`, `probe_attention_note`.

## 3. The version numbers, kept apart (lanes AL-1, AL-2)

There are four, not three.

| Number | Where | Today | Protocol 5 | What it gates |
|---|---|---|---|---|
| `PROTOCOL` | `server_detect.py`, `main.rs` | 4 | **5** | Whether the SDK runs a binary's `detect` and `detect --follow`: exact equality, else "speaks protocol N and this tensorquill needs protocol M", and `auto` falls back to Python. Not checked by the live sender |
| Batch metadata `schema_version` | `flight.py` `SCHEMA_VERSION`, `ingest.rs` `SCHEMA_VERSION`, the Flight `app_metadata` | 1 | **2** for protocol 5's layout. A protocol 5 server takes 1 and 2 | The live stream: a server refuses another with `INVALID_ARGUMENT`, "schema_version N is not supported; this server reads version 1", and acknowledges nothing (D6) |
| Row `schema_version` | the Arrow column; today equal to the record's | 0, 1 | **2** for a row of protocol 5's layout, with the record's own in `record_version` (Q1 a). A protocol 5 server stores 0, 1 and 2 | RS10: a server drops and counts a row of another version, and acknowledges it. Today's readers refuse a part holding one (D7) |
| Record `schema_version` | the `.jsonl`; `identity.SCHEMA_VERSIONS` | 0, 1 | step records stay 1; **2** is a scalar-only record (Q1 a) | Every offline reader: a record of another version refuses the whole input, naming it |

- **The Arrow schema carries no number.** A layout is told by its columns: `layers` against `layers_json`. Every Rust reader looks columns up by name already (`Columns::new`), so it reads a batch of either layout, per batch.
- **A part has one schema.** `accept_attempt` writes the buffered part when a batch's schema differs. So a rank whose protocol 4 backlog and protocol 5 rows meet has parts of both layouts, never a part of two.
- **Unchanged:** the evidence's own `schema_version` (1), the offset file's `OFFSET_VERSION` (2), `store.json` and `attempts.json`.

## 4. The scalar-only record (step 2 of the wave; lanes AL-5, AL-2, AL-4)

### 4.1 What wave AK left

- **`Session.log(..., step=...)` raises `TypeError`** naming protocol 5, for any value, `None` included. A dictionary key `"step"` is an ordinary scalar.
- **`Session._no_step_for` is the one place that decides a value with no step in memory.** That is a baseline evaluation before the first step, a value after `flush()`, or one pending at `close()` after such a flush. It leaves the value out with one warning per session.
- **From wave AK, now this wave's:** a scalar logged late inside a compiled function, then a flush because a frozen parameter became trainable. Wave AK left it open. The owner reversed that on Nov 24: "this will be fixed in AL". It is step 2 work for lane AL-5 (section 4.8).

### 4.2 The record

```json
{"schema_version": 2, "record": "scalars", "run_id": "...", "rank": 0, "attempt": 1732460000000,
 "step": 500, "seq": 0, "ts": 1732460123.4, "scalars": {"eval_loss": 1.8}}
```

- **Its keys are these nine and no others.**
  - `step` is the x-axis: an integer of 0 or more, as `identity.check` requires of every record.
  - `seq` is an integer of 0 or more, counted per session (so per attempt) from 0, in the order of the `log` calls.
  - `ts` is when `log` was called, in Unix seconds.
  - `scalars` holds numbers only, as `Session._scalar_values` gives them. A non-finite value is written as the sink's string.
- **The capture writes one for an explicit `step=`.** It never writes one for a value logged without it that a step in memory takes, even one naming the step in progress.
- **It writes none for a value with no step in memory and no `step=`** (Q5). That value is left out with one warning, which shows the explicit call. The Hugging Face callback passes `step=` itself in that situation, so its evaluation metrics are recorded.
- **`step=` refuses, before anything is recorded:** a bool, a float, a negative integer and anything not an integer (case `scalar_log_refusals`). The refusals are `TypeError`, or `ValueError` for a negative.
- **A tensor value is copied on its device** and read with the next flush, with that flush's records, so no sync is added.
- **The writer writes it with the next block of step records, or at `close()`.** A stop of capture loses a pending one, as it loses pending steps.
- **Not done:** a record for a value the mirror brings with a tracker's own `step=` (Q7 a).

### 4.3 What a protocol 4 reader does with it

Today's readers refuse the whole input, naming the line: "schema_version is 2 at …, line N: this tensorquill reads schema versions 0 and 1" (case `scalar_old_reader`). A protocol 5 reader given an old file reads it as today: it holds no such record.

### 4.4 How it travels (Q3 a)

- **The same `.jsonl`, the same byte cursor, the same stream.** The sender converts it like any line, to a row of protocol 5's layout:
  - `seq` set (a step row's `seq` is null) and `record_version` 2;
  - `scalars` in the `scalars` column, under wave W3's rule;
  - `record` restored by `row_to_record` from a non-null `seq`.
- **The sender's checks are unchanged:** `rank`, `attempt` and `step` are checked as for any line (`SENDER_FIELDS`).

### 4.5 The server (lane AL-2)

- **Rows whose `seq` is not null are taken out of the batch before `keep_new_rows`.** They never meet the step watermark and are never progress: not `last_step`, not a batch's `max_step` for the watchdog, not `repeated_steps`.
- **They are deduplicated by `(run_id, rank, attempt, seq)`:** a row at or below the attempt's `seq` watermark is dropped and counted as `scalars_not_newer` in the rank's status.
- **They are stored as `attempt=<a>/scalars-<first_seq:08>-<last_seq:08>.parquet`**, with `PutMode::Create`, never as `part-*` (D8). Today's `is_part`, `rank_parts`, `max_step` and `_parquet_parts` never list them.
- **The watermark across a restart** is the highest last `seq` in the names of the attempt's `scalars-*` objects. Their footers are read only for a name that does not parse, as `max_step` does.
  - A replay from byte 0, after a replaced store or rewritten bytes, sends every line again. The server drops each scalar row at or below the watermark, so nothing is stored twice (cases `scalar_replay_from_byte_0`, `scalar_seq_watermark_across_restart`).
- **An offset is acknowledged only when every row before it is stored**, of both kinds. A flush writes the step part, then the scalars object, then acknowledges.
- **A create that meets an existing object** is read for its `seq` range. If the range is the same, the rows are stored. If not, the rows are kept buffered and the error is logged.
- **The detectors never read a scalar row** (Q4 a), live or in a replay.
- **Phase 4's compaction and catalog** must list `scalars-*` as their own family.

### 4.6 Reading it back (lane AL-4)

**Every reader sets scalar-only records apart where records are loaded**, after `identity.check` (which now takes version 2), and before any engine.
- **Attempt resolution applies to both, with the cutoffs of the training records.** A later attempt that resumed below a scalar's step leaves that scalar out of the default read. `all_attempts` keeps it (case `scalar_rolled_back_by_a_later_attempt`).
- **A scalar's series is the training records' `scalars` values and the scalar-only records' points, by step** (Q6 a: at one step, the highest `seq` of a scalar-only record wins).
  - A step no training record has is a point of its own.
  - No reader ever makes a record, a step or a sample of it.

| Reader | What it does with them |
|---|---|
| `evidence.build_evidence`, both engines | Judges the training records alone. Incident context is unchanged (Q4 a). `kind_checks`' `records`, `repeated_steps` and the run state are what the training records alone give (case `scalar_not_a_training_step`). The Rust paths are never sent a scalar-only record: `server_detect._stream`, `FollowedDetect` and `stream_order` take training records |
| `compare` (`first_divergence`) | Reads them as series points (Q8 a) |
| `runs` | Takes the run ID from the first line, whatever it is, and the last step from the last training record (case `scalar_first_and_last_line`) |
| `compact` | Unchanged: it reads the evidence |
| MCP `get_series` (kind `scalar`), `list_runs`, `compare_runs` | The merged series; as `runs`; as `compare` |
| The Rerun export | Logs each point under `/logged/<name>` at its step |
| Gate 1's arm A `scalars.csv` | Unchanged for the fixtures, which log no `step=` |

**None of the 33 kinds judges a scalar** (D12).
- Scalars are read only as incident context: `scalar_context` in Python, `Scalars::context` in Rust.
- They are also read by `first_divergence`, `get_series`, the Rerun export and Gate 1's CSV.

**Edge cases.**
- **A step in the future, or one never written,** is a point at that step (case `scalar_step_never_written`). Today a row of a future step in a part loses every later training step (D8; case `scalar_future_step_keeps_training_rows`).
- **Two records for one step and name:** the higher `seq` wins (case `scalar_same_step_twice`).

### 4.7 The cases

`scalar_record_keys`, `scalar_not_a_training_step`, `scalar_future_step_keeps_training_rows`, `scalar_replay_from_byte_0`, `scalar_seq_watermark_across_restart`, `scalar_step_never_written`, `scalar_first_and_last_line`, `scalar_same_step_twice`, `scalar_rolled_back_by_a_later_attempt`, `scalar_old_reader`, `scalar_log_refusals`, `scalar_no_step_baseline_warning`, `scalar_no_step_baseline_on_a_resume_warning`, `scalar_no_step_after_flush_warning`, `scalar_no_step_pending_at_close_after_flush_warning`, `scalar_no_step_mirrored_warning`, `scalar_no_step_said_once`, `scalar_hf_eval_on_start_baseline`, `scalar_hf_eval_on_start_on_a_resume`, `scalar_baseline_and_step_0_share_a_point`, `scalar_eager_late_before_unfreeze`, `scalar_compiled_late_before_unfreeze`, `scalar_compiled_late_at_a_stop` (section 4.8).

### 4.8 A value logged late inside a compiled function (lane AL-5; the owner, Nov 24)

**The fault.** From lane AK-5's report, item 11, read again in `capture.py` at `f71d8e0`.
- **Where a traced value waits.** Traced, `_log_late` reads nothing of the ring: its length, its step numbers and the held step would each be a guard that changes at every step. It puts the value in `_late_traced`. Eager code moves it onto the newest step (`_take_traced_late`) at:
  - the next step's end (the start of `_commit`);
  - `flush()` and `close()`;
  - an eager late `log`;
  - the mirror's next value (`_scalars_for_log`).
- **The one path that writes the newest step without draining it is `_take_unfrozen`.** It runs on the first step on which a parameter frozen at `init` is trainable, from the optimizer's pre-step hook, or from a hand `record_grads()`. Before that step's gradients are read, it calls `_flush(hold_newest=False)`.
  - That writes step k, the step the evaluation followed, whether it is the ring's last row or the held step (at `every=1`).
  - Then `_commit` of step k+1 drains the buffer, finds no step in memory, and leaves the value out through `_no_step_for`.
  - The warning gives the wrong cause: "before the first step, or after flush() with no step since".
- **The same value logged eagerly is kept.** Eager `_log_late` puts it on the ring's last row at once, before that flush. So today an evaluation under `torch.compile` loses a value that the same evaluation run eagerly keeps.

**What the fix must make true.**
1. A value logged late inside a compiled function, after step k ended and before that flush, is on step k's own record. This is exactly where the same value logged eagerly is, at every `every`: on the ring's last row and on the held step alike. Case `scalar_compiled_late_before_unfreeze`. Its eager twin, `scalar_eager_late_before_unfreeze`, is the reference, and the base already gives it.
2. No "left out" warning is said for it. The unfreeze warning is said once, as today.
3. The evaluation still compiles once, as in wave AK's test (`frame_count == 1`). The fix is on the eager side only, and nothing the traced function reads changes.
4. **The rule, not just the call site: every path that hands the newest step to the writer drains `_late_traced` first.** Today that is `_commit`, `flush()` and `close()`, and the fix adds `_take_unfrozen`.
   - The form that also covers a later caller is a drain at the start of `_flush`, after its stop check.
   - The drains in `flush()` and `close()` may stay; a second drain finds the buffer empty.
   - In `close()` the drain stays before `_attach_pending`, as today.
5. Only then can the stop of capture be stated as below: a value in the buffer always belongs to a step not yet handed to the writer (or to no step, section Q5's cases).

**Where the value goes: step k's own record (recommended), not a scalar-only record.**
- **Step k's own record, recommended.**
  - It is where the value belongs, and where the eager run puts it, so compiling an evaluation changes nothing that is recorded.
  - Step k is still in memory when the buffer is drained, so wave AK's rule that a record already written is never changed still holds.
  - It needs nothing from protocol 5: one drain before a flush. So it is the smallest change, and AL-5 can land it first in its branch, ahead of the scalar-only record.
- **A scalar-only record at step k, not recommended.**
  - By Q6 it would win at the series point, so the series would be right.
  - But step k's own record would lack the value. An incident's `scalars` context, read from the training record (Q4 a), would then differ from the eager run's.
  - A protocol 4 reader would refuse the file over a value that the eager run writes as version 1.

**The stop of capture: a stated limit, like the steps not yet handed to the writer.**
- **What happens.** Capture stops (`_stop`) on a sink error, a writer that falls behind, a close timeout or a flush that raised. Then:
  - the ring and the held step are cleared;
  - every public call returns at once;
  - `close()` drains nothing.
  
  A value in `_late_traced` belongs to the newest step (rule 5), which is in the ring or held. So it is lost together with that step: no record is left for it to reach. The stop's own warning already says it: "the steps already handed to the writer are written, no later step is".
- **It cannot be closed** without writing after a stop, which wave C03's decision rules out: a stopped writer takes nothing, and a scalar-only record would have to pass through the same writer.
- **So it stays a stated limit,** and no "left out" warning is added: the stop's warning is the one said. Case `scalar_compiled_late_at_a_stop`, which the base already gives.
- **Lane AK-5's report words this differently.** Its item 11 puts the stop with the cases that leave the value out "with the one warning". At `f71d8e0` that warning is the stop's own, not `_no_step_for`'s:
  - `_closed_call` makes every public call on a stopped session return at once;
  - `_commit` returns before its drain;
  - `close()` drains nothing.
  
  So `_no_step_for` is never reached. This is read, not run; a scratch run confirms it once the machine is free.

**Tests.** `test_wave_al_contract.py`:
- `test_a_value_logged_late_inside_a_compiled_function_reaches_its_step`, at `every` 1 and 4, is the target. It is `xfail(strict=True, raises=AssertionError)`, so a compile that fails, or a scenario that does not reach the flush, fails the run instead of passing as expected.
  - The evaluation runs after steps 0 and 1. Requiring one compiled frame for both calls holds item 3 too: a fix that made it recompile fails the run.
  - Like wave AK's compiled tests, it runs unguarded on CI's whole matrix.
- `test_the_same_value_logged_eagerly_reaches_its_step` pins the eager twin.
- The stop case is AL-5's own test to write, failing first if it does not hold.

## 5. The matrix

"Measured" cells ran here against the protocol 4 binary (`results/live.json`). The protocol 5 cells are what the lanes must make true, with the case that tests each.

| Writer or store | Reader | Result |
|---|---|---|
| Protocol 4 sender (metadata 1, rows 0/1, JSON columns) | Protocol 5 server | **Works.** Stored as received, in its own parts. `detect.rs` reads `layers_json` where there is no `layers`. No conversion |
| Protocol 5 sender (metadata 2) | Protocol 4 server | **Refused.** `INVALID_ARGUMENT` "schema_version 2 is not supported; this server reads version 1"; nothing acknowledged; the lines stay in the sender's file (measured). The sender says once, on stderr and in `last_error`: "the server at … speaks an older protocol: upgrade tensorquill-server" (lane AL-1) |
| Protocol 5 server, restarting | A store of protocol 4 parts | **Works.** Replay reads each part by its columns. The step watermark is from names and footers, as today. No `scalars-*`, so no `seq` watermark |
| Protocol 4 server, restarting | A store with protocol 5 parts | **Silent:** no layers, no MoE statistics, no maps; `scalars-*` not listed (Q9) |
| Protocol 5 SDK | A protocol 4 binary, or the reverse | **Refused by `PROTOCOL`**: "speaks protocol 4 and this tensorquill needs protocol 5". `detectors="auto"` falls back to Python with that reason; `"rust"` raises. The watcher (`detect --follow`) does the same |
| Protocol 5 offline `detect` (through the SDK) | An old `.jsonl` | **Works:** converted to protocol 5's layout as it is read |
| Protocol 5 offline `detect` (through the SDK) | A new `.jsonl` | **Works:** scalar-only records are set apart before the stream |
| Protocol 5 Python readers | Old `.jsonl` and protocol 4 parts | **Work**, as today |
| Protocol 4 Python readers | A new `.jsonl` without scalar-only records | **Works** under Q1 (a); refused under Q1 (b) |
| Protocol 4 Python readers | A new `.jsonl` with one | **Refused, naming the line** (`scalar_old_reader`) |
| Protocol 4 Python readers | Protocol 5 parts | **Refused, naming the part and row** (measured) |
| Protocol 4 `runs`, `list_runs` | Such a file | Listed as a run with no evidence, as for any refused file |

The server reads protocol 4 parts forever (D9). A conversion could come only with Phase 4's compaction, under its manifest, which replay reads first.

## 6. Before-numbers and the acceptance figures (lanes AL-2, AL-3)

**What the "prototype" arm leaves out.** The protocol 4 server reads none of the prototype's typed columns: it judges no layer and no router statistic on them (`old_server_typed`). So in `ack` and `live64` the prototype arm's detectors skip the layer work that the `v4` and no-probe arms do.
- That flatters its acknowledgements only after the part is written: the detectors run after the acknowledgement. They show through the next batch's wait, and the `after_probe` batches in `ack` are no slower in either arm.
- The acceptance figures are ratios against a no-probe arm in the same layout, measured by the lane with protocol 5's own server, where the detectors read both arms alike.

**Where they were measured.** This Mac: 8 cores, Apple M1 Pro, Python 3.12.3, pyarrow 26.0.1, the release binary of this tree. Each run records the one-minute load average before each round. The figures below were re-measured on Nov 24 after the gate ended, at load 2.2 to 8.2 (each row gives its own); the first measurements, at load 4 to 11, are in the branch's history. The one-minute average also counts the previous round of the measurement itself, most of all in `live64`. Medians are given with the lowest and highest values. The plan's figures are corrected in section 8.

| What | Today (protocol 4) | Prototype (protocol 5 layout, protocol 4 server) | Acceptance figure for the wave, and how it is measured |
|---|---|---|---|
| One rank: a probe batch, sent to acknowledged (`measure.py ack`, 7 rounds, 4 probe batches each, load 2.3 to 2.6) | 14.1 ms (11.2 to 39.8) | 9.1 ms (7.8 to 12.8) | At most 1.7 times a plain batch's acknowledgement on the same run, against 2.2 today (plain: 6.4 ms in today's arm, 6.2 in the prototype's, 5.3 with no probe at all). `measure.py ack`, 5 or more rounds |
| Its Parquet part | 743 KB; write alone 5.2 ms | 539 KB; 4.8 ms | At most 75% of today's bytes on the dense capture's probe parts. `measure.py ack` |
| 64 ranks at 65 ms a step, every other part a probe (`measure.py live64`: the server's own `live_streams` example, 8 source ranks replayed as 64, `--flush-steps 50`, 7 rounds, load 2.2 to 8.2). Sent to acknowledged, median of the rounds' values (lowest to highest round) | median 71.1 ms (67.2 to 75.1), p99 170 (155 to 226), worst 185 (165 to 283); one round of seven had 1 of 448 acknowledgements over 250 ms (an earlier, loaded run: one round of five had 42, worst 379). Without any probe: median 48.7, p99 81.8 (79 to 99), worst 91.5 | median 51.0 (49.2 to 54.6), p99 111 (106 to 120), worst 115 (108 to 124); none over 250 ms | p99 at most 1.5 times the p99 of a no-probe arm in protocol 5's layout, on the same server and run (here: today 2.1, prototype 1.35, against the protocol 4 no-probe arm), and no acknowledgement over 250 ms in any of 5 or more rounds. `measure.py live64` |
| Decoding what the detectors read of the layer and router statistics: 1,000 Granite-shaped rows (24 blocks, 32 experts), with `detect.rs`' own visitors against the typed maps (`measure.py decode`, `rust-reader bench`, 7 rounds, load 2.3 to 2.4) | `layers_json` 4.89 ms (4.78 to 5.61), `moe_stats_json` 4.24 ms (4.16 to 4.63) | `layers` 0.18 ms (0.17 to 0.19), `moe_stats` 0.069 ms (0.066 to 0.085) | Both under 0.5 ms per 1,000 rows. `rust-reader bench` on the lane's parts |
| The detectors' whole time per flush of 64 ranks of 50 steps, one thread (`detect_store --bench`, the server's own example, 64 ranks × 1,000 steps, 7 runs, load 2.3 to 5.0) | 70.9 ms (70.6 to 71.4); of which the JSON decoding measured apart, 34.0 ms (33.8 to 34.4), 48% | Not measurable without product code. Expected about 38 ms: today's time without the decoding, plus the typed decode at 1 ms per 3,200 rows | At most 50 ms on the same store in protocol 5's layout, with `detect_store --bench` on the same machine. Today's 70.9 is to be re-measured beside it, and the figure is then at most 0.7 times today's |
| Its parts | 4.86 MB for 1,000 rows | 4.37 MB (−10%); `read_part` 20.1 ms (18.2 to 22.8) against 19.5 ms (17.2 to 28.2), within the spread | No figure: the replay is not on the live path |
| `load_records` on that store (Python) | 336 ms (264 to 494) | 298 ms (243 to 333) | No slower than today, beyond the spread |

## 7. The wave's invariant

- **The golden files and Gate 1's shown files stay byte-identical.** The golden files are under `sdk/tests/golden/`. Gate 1's are built by `experiments/gate1/harness.py build` and compared as the gate does, with `.rrd` files by content.
- **Both engines agree on every shared case.**
- **Nothing in sections 1 to 4, as recommended, changes either set:**
  - Step records and the `.jsonl` are unchanged.
  - The evidence keeps its shape.
  - The Gate 1 fixtures run no probe and log no `step=`.
  - The Python changes of section 1.5 touch only inputs no capture writes.
- **Under the other options:**
  - Q1 (b) changes every captured record's version byte. No golden file is expected to change, since the evidence does not carry the records' version, but the lane must show it.
  - Q4 (b) changes an incident's `scalars` only for a run with scalar-only records. No golden file has one.
  - Q2 (b) changes nothing local.
- **A change to a golden file is the owner's amendment, not a lane's.**

## 8. Corrections to the architecture page and the plan

1. **Section 4.2, `extra_json`.** It does not hold `checkpoint`, `data` or `system`: those have their own columns (`checkpoint_json`, `data_json`, `system_json`). `clip_groups` is a `not_checked` check, not a record key.
   - What it holds today, as `capture.py` writes it: `replica`, `replica_spread_scope`, `moe_refused`, `moe_source`, `moe_unassigned`, `moe_left_out`, `layers_not_read`, `grads_not_read`, `update_not_read`, `step_skipped_by`, `step_skipped_part`, `scaler_applied`, `clip_calls`, `optimizer_ownership` and `weight_dtypes`.
   - It also holds wave AJ's refused entries, and any value its column cannot hold.
2. **Section 4.2, key columns.** `data_wait`, `step_time`, `data_json`, `checkpoint_json` and `system_json` are missing from its list.
3. **Section 4.2, `moe_stats`.** The struct is wrong: section 1.1. `token_entropy` is not reserved (D3).
4. **Section 4.2, `probe_summary: Utf8`.** The column keeps its name, `probe_json`. Its heavy parts move to `probe_maps` and `probe_histograms` (Q2 a), not to "`probes/` sidecar blobs or dedicated binary columns".
5. **Sections 2.4, 4.1 and 4.2: "64 × 64 maps".** A map is `min(rows, 64)` by `min(keys, 64)`.
6. **Section 3.2: "The SDK verifies exact match before invoking Rust detection or sidecar IPC".** True of `detect`. The live sender checks no protocol: the batch metadata's `schema_version` is the live stream's gate (section 3).
7. **Section 4.1: the `probes/` path.** It stands only if the owner chooses Q2 (b).
8. **The plan, Phase 2: "the `620` to `744 ms` ack spike".**
   - The README records one acknowledgement of 620 ms in one of four filled runs under 64 live streams, unattributed. "A probe flush under load is likeliest" is its words, not a measurement.
   - No source for 744 ms was found in the repository or the ledger.
   - On one rank a probe batch costs 14.1 ms against 5.3 ms (section 6).
9. **The plan, Phase 2: "`155 ms` → `350 ms` Parquet flush penalty".** That is writing all 64 ranks' parts on a probe flush (README, "Re-measured with 25 kinds"), not one part. One probe part's write is 5.2 ms here.
10. **The plan, Phase 2: "`layers_json` / `moe_stats_json` cost `33 ms` of `54 ms`".**
    - The README's 54 ms is the first six kinds; twelve took 56 ms.
    - The 33 ms is "measured apart" by `detect_store`'s own decode timing.
    - Measured here on 64 ranks × 1,000 steps with the same example: 34.0 ms of 70.9 ms (48%), with today's per-rank kinds. "Two thirds" was the share with twelve kinds.
11. **Section 4.3: old parts "expire via bucket lifecycle policies".** Already withdrawn by the owner on Nov 24 (ledger section 20). Repeated here because a sidecar (Q2 b) would be a second family that rule could not tell apart either.

## 9. The lanes

**Two gated steps, one protocol bump, one push.**
- Step 1 is sections 1 to 3; step 2 is section 4. The Python readers change only in step 2, as the plan has it.
- Lanes are split by file, and two lanes never own one file.
- **`PROTOCOL` stays 4 in every lane's own tree.** The coordinator raises it to 5 in both `server_detect.py` and `main.rs` in step 1's seam commit. So each lane's tree keeps its `rust_raw` pins passing with a binary built from it.
- **Targets that need two lanes are judged on the joined tree,** with `--runxfail`, as wave AJ did. A `rust_raw` target needs AL-1's layout and AL-3's reader.
- A lane's tests are its cases' paths. Its targets in `test_wave_al_contract.py` lose their `xfail` mark when they pass on its branch, or at the merge for a joined target. The live-server, `runs` and MCP paths are tests each lane writes for its cases, failing first.
- **No lane owns `sdk/README.md`.** Each lane writes its passage in its report, and the coordinator puts them in at the seam, with the README's version table and the test that reads it. The passages owed:
  - **The `scalars` bullet (AL-5, step 2):**
    - `step=` writes a scalar-only record (section 4.2).
    - A value with no step in memory and no `step=` is left out with one warning per session, which shows `session.log({...}, step=N)`: `start_step` for a baseline evaluation, the last finished step after `flush()`.
    - The mirror's values cannot carry a step, so the user logs such a value with `step=` themself.
    - This replaces wave AK's "comes with protocol 5's scalar-only record".
    - The Hugging Face callback records `eval_on_start`'s baseline and an evaluation after `flush()` with an explicit step.
    - The one line on step 0: a baseline logged with `step=0` and an evaluation after the first step, logged under the same name, are one point at step 0, and the series shows the baseline.
    - Under `torch.compile`, a value logged under "eval" lands on the step it followed, as the same value logged eagerly does, also when a frozen parameter becomes trainable on the next step (section 4.8).
    - After capture stops, values logged since the last step written are lost with that step, compiled or not.
  - **The versions (AL-1, AL-2):** section 3's table; the older-server message; "a downgrade is not supported", with how it shows (Q9).
  - **The probe (AL-1):** the maps and histograms are in their own columns; the `.jsonl` is unchanged.

**Step 1: typed columns, the probe split, the versions.**

| Lane | Owns | Does | Cases it must make pass |
|---|---|---|---|
| AL-1 | `flight.py`, `server_detect.py` | Protocol 5's row layout: `flight_schema`, `record_to_row`, `row_to_record`, `rows_batch` for sections 1, 2 and 3 (`load_records` reads a store of either layout through it). Metadata version 2. The older-server message | every section 1 and 2 case's `row` path (`test_flight_gives_the_row`); `moved` for `moe_stats_int_past_int64` |
| AL-2 | `ingest.rs`, `storage.rs`, `main.rs` (but `PROTOCOL`), `watchdog.rs`, `server/examples/` | Metadata versions 1 and 2; row versions 0, 1 and 2. A read of a part that projects away `probe_maps` and `probe_histograms`, for replay. `live_streams` sends metadata version 2 for a source in protocol 5's layout (it writes 1 today, `live_streams.rs` line 264), so section 6's runs can be made | the matrix's server rows; `layers_model_order` on a live server across a restart |
| AL-3 | `detect.rs`, `detect/` | Reads `layers` and `moe_stats` from the typed columns where a batch has them, and the JSON columns where it does not. `non_finite` also reads the refused fields in `extra_json`. Never reads `probe_maps` or `probe_histograms`. `check_keys` takes row version 2 (it exits on anything but 0 or 1 today) | every section 1 and 2 case's `rust_raw` and `rust_sdk` paths, live server included (joined with AL-1) |

- **AL-1 first.** AL-2 and AL-3 can start beside it: AL-2's projection and AL-3's dual reader need only the column names of sections 1.2 and 2.4.
- **What step 1 leaves standing:** Python still raises on `layers_entry_not_object` and `layers_value_not_object`, as it does today, while Rust reads them. Their `python` targets are step 2's (AL-4′).
- **The step's gate:**
  - the whole SDK suite and `cargo test` with this tree's binary;
  - the golden and Gate 1 comparisons;
  - `measure.py ack`, `live64` and `decode` against section 6's figures;
  - `captures.py` on the seven captures.

**Step 2: the scalar-only record** (after step 1 is merged and gated).

| Lane | Owns | Does | Cases |
|---|---|---|---|
| AL-5 | `capture.py` (`Session.log`, `_no_step_for`, the writer's block), `mirror.py` (the mark on a value it brings), `huggingface.py` (`on_evaluate`'s explicit step), `sink.py` if needed | Writes the record of section 4.2. Takes away `step=`'s `TypeError`. The warning text of section Q5 | `scalar_record_keys` (capture), `scalar_log_refusals`, the six `scalar_no_step_*` cases, the two `scalar_hf_*` cases; wave AK's tests of the old warning text change with it. Section 4.8's drain before the flush of a newly trainable parameter: `scalar_compiled_late_before_unfreeze` (`test_a_value_logged_late_inside_a_compiled_function_reaches_its_step`), keeping `scalar_eager_late_before_unfreeze` and `scalar_compiled_late_at_a_stop` |
| AL-1′ | `flight.py`, `server_detect.py` | The record's row (`seq`, `record_version`). Scalar-only records set apart before `_stream`, `FollowedDetect` and `stream_order` | `scalar_record_keys` (row) |
| AL-2′ | `ingest.rs`, `storage.rs`, `watchdog.rs` (`scalars_not_newer`) | Section 4.5 | `scalar_future_step_keeps_training_rows`, `scalar_replay_from_byte_0`, `scalar_seq_watermark_across_restart` |
| AL-4′ | `evidence.py`, `identity.py`, `compare.py`, `divergence.py`, `runs.py`, `mcp_server.py`, `compact.py` | Section 4.6, and section 1.5's Python changes: an entry or a `layers` value that is not an object read as absent; a `diversity` that is not a finite number taken from the counts | every section 4 case's `python`, `runs` and `mcp_get_series` paths; `test_a_scalar_record_is_read_and_never_a_training_step`; the `python` targets of `layers_entry_not_object` and `layers_value_not_object`; the samples comparison of `moe_stats_diversity_refused` |

`detect.rs` has nothing to do in step 2 under Q4 (a). Under Q4 (b) it gains a lane, and the cases gain context expectations.

## 10. What the sibling clones do

- **arrow-rs** (`/Users/ak/Documents/tensrview/arrow-rs`, main of Nov 14; the server pins 60.1.0).
  - `DataType::Map(field, keys_sorted)` is `List<entries: Struct<key, value>>`. Its keys "should be hashable and unique", which is not enforced.
  - A map read from Parquet takes its `keys_sorted` flag from the stored Arrow schema (`parquet/src/arrow/schema/complex.rs`), or `false` without one (`array_reader/map_array.rs`). Nothing in the read path sorts entries; D1 measured that order holds.
  - `ArrowWriter::write` refuses a batch of another schema, so a part has one.
  - `ProjectionMask::roots` and `leaves` (`parquet/src/arrow/mod.rs`) read only the columns asked for.
- **wandb** (`core/internal/stream/handler.go` near line 994): a `log` with a `step` below the current one is ignored, with "Tried to log to step N that is less than the current step M … this data will be ignored", and pointed to `define_metric` for out-of-order axes. TensorQuill's scalar-only record takes out-of-order steps, since `seq` and not `step` orders it, and never touches the training step.
- **MLflow** (`mlflow/store/tracking/dbmodels/models.py`): a metric row's key is `(key, timestamp, step, run_uuid, value, is_nan)`, so the same value at the same time is stored once and every other value of a step is kept as history. `latest_metrics` keeps one row per key. TensorQuill deduplicates by its own `(run, rank, attempt, seq)`, which a replay repeats exactly, and reads the highest `seq` at a step.
- **InfluxDB 3** (`influxdb3_write/src/write_buffer/queryable_buffer.rs`, `sort_dedupe_persist`): a write's identity is its series and time, deduplicated at persist, last write winning. Its tables are flat columns, with no map or struct, and a field written with another type is refused (`influxdb3_catalog` `InvalidColumnType`). Read for the dedup rule; not copied.
- **datafusion-ducklake**: evolves `list`, `struct` and `map` columns recursively, with per-file name mappings (README). That is the by-name field lookup D3 asks for. It depends on DataFusion, a rejected non-goal, so ideas only, as the ledger already decided.
- **datafusion:** not read for this contract. The detectors read Arrow arrays directly, and DataFusion is a rejected non-goal.
