# Wave AL contract: protocol 5

Base: `trial/ai-wave` at `f71d8e0` (lane AL-0's branch; `trial/ai-wave` has since moved to `5749a89`, tests only). This is Phase 2 item 1 of the v1 plan, in two gated steps under one protocol bump: the typed columns and the probe split first (sections 1 to 3), the scalar-only record second (section 4). The pinned-memory flush, `grads=N`, K2, K3, the catalog and compaction are not in it.

- **Shared cases:** `sdk/tests/fixtures/wave_al/cases.json`, written by `experiments/protocol-v5/build_cases.py`. Each case says what every path must report, and what the base reports today. `sdk/tests/test_wave_al_contract.py` reads them.
- **Measurements:** `experiments/protocol-v5/` (`prototype.py`, `captures.py`, `live.py`, `measure.py`, `rust-reader/`), results in `results/`. Every one ran on this Mac's CPU against the protocol 4 release binary built from this tree, with no product code changed. A "v5" figure is the prototype's layout, sent to and stored by the protocol 4 server.
- **Read by:** the coordinating session that cuts the lanes, and the lanes of section 9.

## Where each decision stands

**Decided** means the code or a measurement settles it; the evidence column says which. **Owner** means it is a product choice: it is asked in the next section, with options and costs, and the contract is written for the recommended option until the owner answers.

| # | Decision | Stands | Evidence |
|---|---|---|---|
| D1 | An Arrow map keeps its entries' order from pyarrow through Flight, the server's Parquet part and `Storage::read_part`, and after a restart's rebuild. No other order carrier is needed for `layers`. | Decided | `results/live.json`, `order` (sent `…2, 10, 11, 3`, read back the same by pyarrow and by the server's own `read_part`, on the part and after a restart) |
| D2 | `layers` has four fields and `moe_stats` five: `routed_tokens`, `entropy`, `diversity`, `dead_experts`, `dead_frac`. The architecture page's `moe_stats` struct is wrong. | Decided | `capture.py` `STATS` and `_router_stats`; every record of the seven captures (`results/captures.json`) |
| D3 | `token_entropy` (K3) is not reserved. Readers look struct fields up by name, and K3 adds the field when it lands. | Decided | `results/live.json`, `struct_field_added`: one pyarrow reader and the Rust reader read a part without the field as absent and one with it as its value |
| D4 | Wave AJ's rule extended to struct entries, field by field, with a null struct keeping an unreadable entry's place (section 1.4). | Decided | the order rule (D1) and the round trip on every case and capture |
| D5 | `row_to_record(record_to_row(r)) == r` with every map's order kept, on all 3,800 records of the seven captures. | Decided | `results/captures.json`: 0 not equal, 0 out of order |
| D6 | The batch metadata `schema_version` goes from 1 to 2 for protocol 5's layout. | Decided | `results/live.json`: a protocol 4 server given the typed layout under version 1 stores it and judges no layer (`old_server_typed`); under row version 2 alone it drops the rows and acknowledges them, so they are lost to it (`old_server_row_v2`); under metadata version 2 it refuses the stream and acknowledges nothing (`old_server_meta_v2`) |
| D7 | A row of protocol 5's layout carries row `schema_version` 2. | Decided | `results/live.json`, `old_python_on_v5_part`: today's `load_records` reads a typed part with row version 1 as a run with no layers (`activation_spike` `not_configured`, no incident); with row version 2 it refuses the part naming it |
| D8 | A scalar-only row is never in a `part-*` file and never moves the step watermark. | Decided | `results/live.json`, `watermark`: a row of step 1000 in a part makes the server drop steps 4 to 9 after a restart, from the part's footer |
| D9 | The server reads protocol 4 parts for as long as they are stored. It does not convert them. | Decided | the store is append-only (section 20 of the ledger): a converted copy would be a second `part-*` file over the same steps, which `rank_parts`, `max_step` and `_parquet_parts` all read |
| D10 | Attention maps are stored as `List<List<Float32>>`, ragged, not as a fixed-size list. | Decided | `probe.py`: a map is `min(rows, 64)` by `min(keys, 64)`; all 524,288 map values in the three probe captures are exact float32 and finite (`results/captures.json`) |
| D11 | No detector, MCP tool, `compare`, `compact` or Gate 1 shown file reads a map or a histogram. | Decided | section 2.2's reader list |
| D12 | None of the 33 kinds judges a logged scalar. Scalars are incident context, `compare`'s series, `get_series`, the Rerun export and Gate 1's arm A CSV. | Decided | section 4.6 |
| D13 | `PROTOCOL` goes from 4 to 5 in `server_detect.py` and `main.rs`. | Decided | what `detect` reads changes (section 3) |
| D14 | The before-numbers and the acceptance figures of section 6. | Decided | `results/ack.json`, `results/live64.json`, `results/decode.json` |
| Q1 | What makes an older SDK refuse protocol 5's data. | **Owner** | section "The owner's questions" |
| Q2 | The heavy probe arrays: columns in the row, or sidecar objects. | **Owner** | same |
| Q3 | The scalar-only record: in the same `.jsonl` and stream, or a file and stream of its own. | **Owner** | same |
| Q4 | Whether a scalar-only record enters an incident's `scalars` context. | **Owner** | same |
| Q5 | A value logged with no step in memory and no `step=`. | **Owner** | same |
| Q6 | The same name on a training record and a scalar-only record of one step. | **Owner** | same |
| Q7 | Whether the mirror passes a tracker's own `step=` on. | **Owner** | same |
| Q8 | Whether `compare` reads scalar-only records. | **Owner** | same |
| Q9 | A protocol 4 server started on a store that holds protocol 5 parts. | **Owner** | same |

## The owner's questions

Each question lists its options with what each costs. The recommended option is first, and the rest of the page is written for it.

**Q1. What makes an older SDK refuse protocol 5's data.** Today's readers refuse a record or row whose `schema_version` is not 0 or 1, naming it (`identity.check`). They read a column they do not know as nothing, and that is silent (D7).
- **(a) Recommended: only what changed is version 2.**
  - Rows of protocol 5's layout carry row version 2, and a new `record_version` column (Int32, null for a record with no `schema_version` key) holds the record's own version, so the round trip is exact.
  - A step record in the `.jsonl` stays version 1, byte for byte. A scalar-only record is version 2.
  - Costs: an older SDK reads a new `.jsonl` file that has no scalar-only record, refuses one that has, and refuses every new store part. The row's `schema_version` no longer equals the record's, so `row_to_record` reads `record_version`. That is one more column, which compresses to almost nothing.
- **(b) Everything a protocol 5 SDK writes is version 2:** every record and every row.
  - Costs: an older SDK can read no new file, even one identical to a protocol 4 file apart from its version. Every step record the capture writes changes by one byte. Tests that assert `schema_version == 1` change. No golden file is expected to change, since the evidence carries its own `schema_version` and not the records'.
  - What it saves: no `record_version` column, and one meaning for the row's version.
- **(c) Nothing is refused:** an older SDK reads protocol 5's parts with no layers and no maps.
  - Not recommended: it is the silent case D7 measured.

**Q2. The heavy probe arrays** (section 2.3 has the measurements).
- **(a) Recommended: columns in the same row.**
  - Layout: `probe_maps: Map<Utf8, List<List<Float32>>>` and `probe_histograms: Map<Utf8, Struct<lo, hi, counts, nonfinite>>`; the rest stays in `probe_json`.
  - What it keeps: one stored object kind, the acknowledgement's meaning, replay and the `.jsonl` unchanged.
  - Costs: the maps still cross Flight and are still compressed in the part, about 1.4 MB of Arrow a probe row at the dense capture's shape, against 2.7 MB today. A replay must project the two columns away, or it decodes them for nothing.
- **(b) Sidecar objects** `runs/<run>/probes/rank=<r>/attempt=<a>/probe-<step:08>.bin.zst`, written create-only by the server.
  - What it saves: the part loses the maps altogether.
  - New object kind: it needs the Phase 4 rules: create-only puts, no delete, what a reader says when the object is missing, and what a replay does when the put meets an existing object (section 2.5).
  - Costs:
    - The maps still cross Flight, unless the sender uploads them on a call of their own. That is a second write path, with its own acknowledgement, joined to the byte cursor.
    - One more `PUT` a probe on S3 or GCS.
    - Every reader of the maps (the Rerun export of a store) needs a second fetch.
    - Phase 4's compaction and catalog gain a second object family to list.

**Q3. Where the scalar-only record travels** (section 4.4).
- **(a) Recommended: in the same `.jsonl`, in the same stream, with a `seq` column.**
  - One file, one byte cursor, one acknowledgement. The server routes the rows by `seq`.
  - Costs: under Q1 (a), an older SDK refuses a file holding one such record, naming the line and version. The record is in the training file, so every reader must set it aside before the detectors.
- **(b) A file and stream of their own** (`<run_id>.rank<N>.scalars.jsonl`, descriptor `["runs", run, "rank=N", "scalars"]`).
  - What it saves: an older SDK never sees the records and reads the training file as before.
  - Costs:
    - A second cursor and a second sender loop per rank.
    - Two streams per rank on the server; a protocol 4 server's descriptor parser refuses the four-part path.
    - `runs` and `tensorquill.init`'s file naming must learn the suffix. Today's `runs` globs `*.jsonl` (`runs.py` near line 601) and would read the file as a capture. This was read in the code, not run.

**Q4. Scalar-only records in incident context.**
- **(a) Recommended: not in context.**
  - An incident's `scalars` (value and `median_before`) is read from the training record of its step, as today. Scalar-only records are series (`get_series`, the Rerun export, and `compare` under Q8).
  - Neither engine's report then depends on when a scalar row arrives.
  - The detectors never read scalar rows, so `detect.rs` does nothing for section 4.
  - Cost: a user who logs `eval/loss` with `step=` does not see it on an incident at that step.
- **(b) In context.**
  - Python merges by step before `scalar_context`.
  - The live server re-attaches context to an incident already reported when a scalar row for its step arrives later. The report converges on the offline one, but is not equal to it until then.
  - Costs: `detect.rs` keeps every reported incident's step and re-reads scalar rows. A new kind of disagreement between the live report and `detect` on the same file, bounded by arrival. `median_before` needs a rule for which records count as before.

**Q5. A value logged with no step in memory and no `step=`.** Examples: a baseline evaluation before the first step, or a value after `flush()`. Today `Session._no_step_for` leaves it out with one warning.
- **(a) Recommended: unchanged.**
  - Only an explicit `step=` writes a scalar-only record, as the decision of Nov 24 is worded. The warning names `step=` as the way.
  - Cost: a baseline evaluation needs `step=0` written by the user.
- **(b) `_no_step_for` writes a scalar-only record at the step about to run** (`_next_step`): 0, or `start_step`, before the first step; the next step after a `flush()`.
  - Cost: an x-axis value the user did not state. A value after `flush()` sits on a step it was not logged in, which is the move wave AK took away.
- **(c) At the last committed step.**
  - Cost: wrong for a baseline evaluation, which precedes every step.

**Q6. The same name on a step's training record and on a scalar-only record of that step.**
- **(a) Recommended: the scalar-only record with the highest `seq` wins** wherever the two are merged into one series point.
  - This matches wave AK's rule that, under "eval", an explicit value replaces a same-named training value.
  - Cost: the training record's value is not on the series; it is still on the record, and in incident context.
- **(b) The training record's value wins.**
  - Cost: a value logged with `step=` to correct one is ignored where it was meant to count.

**Q7. The mirror and a tracker's own `step=`.** `wandb.log(..., step=N)` and `mlflow.log_metric(..., step=N)` carry the tracker's step. The mirror records on TensorQuill's step and ignores it.
- **(a) Recommended: unchanged.**
  - The tracker's step is the tracker's x-axis: often the global step, sometimes epochs or samples. Passing it as `step=` would put two axes on one series.
  - Cost: a value logged to W&B after the last step is left out (Q5), as today.
- **(b) Pass it when it is an integer of 0 or more.**
  - Cost: a run that logs W&B by samples gets scalar-only records at steps that do not exist.

**Q8. `compare` and scalar-only records.**
- **(a) Recommended: `first_divergence` reads them as points of the scalar's series**, merged as in Q6.
  - Cost: a pair of runs that logged `eval/loss` with `step=` can now diverge on it. Nothing changes for runs without such records.
- **(b) `compare` leaves them out in protocol 5.**

**Q9. A protocol 4 server started on a store that holds protocol 5 parts** (a downgrade).
- It reads them with no layers, no MoE statistics and no maps. It judges the activation kinds `not_configured` with no incident (the same reading as `old_server_typed` in `results/live.json`), and it never lists `scalars-*`.
- Nothing protocol 5 writes can make that binary refuse, short of giving protocol 5's parts another file prefix. That would hide them from it: its reports would then stop at the last protocol 4 part, which is also silent.
- **(a) Recommended:** the README says a downgrade is not supported and how it shows.
- **(b) Another prefix for protocol 5 parts**, `part5-*`.
  - Cost: every reader and Phase 4's manifest learn a second name, for a case with no user yet.

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
- Python (lane AL-4) skips an entry that is not an object, and a `layers` value that is not an object, as absent.
- Rust (lane AL-3, with AL-1's layout) reads the rest of the map.

**Every other case gives today's results in both engines,** and the tests pin them (`-pin` in their IDs). That includes `moe_stats_diversity_refused`:
- Python reads `float(true)` as a diversity of 1.0, where Rust takes it from the counts. Both still report the same `moe_imbalance`.
- The difference shows only in the per-step `samples` (`build_evidence` against `detect --samples`). Lane AL-4 makes Python fall back to the counts there as Rust does, and adds that comparison as its test.

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
- On one rank (`results/ack.json`; 7 rounds, load 9.3 to 11):
  - Today, a 50-step batch holding one of the dense capture's probes is acknowledged in 14.6 ms (median; 12.3 to 20.5). The same batch without the probe takes 5.6 ms.
  - Its part is 743 KB, against 151 KB without the probe, and the server's Parquet write alone takes 5.5 ms (5.2 to 8.1), against 1.8 ms.
- Under 64 streams at the capture's 65 ms a step (`results/live64.json`):
  - Today: p99 197 ms and worst 206 ms (medians of 5 rounds), against 85 and 89 ms with no probe. One round reached 379 ms, with 42 acknowledgements over 250 ms.
  - The prototype: p99 117 ms and worst 122 ms, none over 250 ms.
  - This is the plan's "acknowledgement spike", reproduced at a third of the size the plan states (section 8).

**Prototype columns (Q2 a), measured.**
- The same batch, with maps and histograms in typed columns, is acknowledged in 9.1 ms (8.0 to 12.9).
- Its part is 539 KB (−27%), and its write takes 4.9 ms.
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
- **Open from wave AK:** a scalar logged late inside a compiled function before the newly-trainable-parameter flush. It is not this wave's.

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
- **The capture writes one only for an explicit `step=`** (Q5 a). It never writes one for a value logged without it, even one naming the step in progress.
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

`scalar_record_keys`, `scalar_not_a_training_step`, `scalar_future_step_keeps_training_rows`, `scalar_replay_from_byte_0`, `scalar_seq_watermark_across_restart`, `scalar_step_never_written`, `scalar_first_and_last_line`, `scalar_same_step_twice`, `scalar_rolled_back_by_a_later_attempt`, `scalar_old_reader`, `scalar_log_refusals`.

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

**Where they were measured.** This Mac: 8 cores, Apple M1 Pro, Python 3.12.3, pyarrow 26.0.1, the release binary of this tree. Each run records the one-minute load average before it: 4 to 11 throughout, from other sessions on the machine. Medians are given with the lowest and highest values. The plan's figures are corrected in section 8.

| What | Today (protocol 4) | Prototype (protocol 5 layout, protocol 4 server) | Acceptance figure for the wave, and how it is measured |
|---|---|---|---|
| One rank: a probe batch, sent to acknowledged (`measure.py ack`, 7 rounds, 4 probe batches each) | 14.6 ms (12.3 to 20.5) | 9.1 ms (8.0 to 12.9) | At most 1.7 times a plain batch's acknowledgement on the same run, against 2.4 today (plain: 6.2 and 5.6 ms). `measure.py ack`, 5 or more rounds |
| Its Parquet part | 743 KB; write alone 5.5 ms | 539 KB; 4.9 ms | At most 75% of today's bytes on the dense capture's probe parts. `measure.py ack` |
| 64 ranks at 65 ms a step, every other part a probe (`measure.py live64`: the server's own `live_streams` example, 8 source ranks replayed as 64, `--flush-steps 50`, 5 rounds, load 2.8 to 9.2). Sent to acknowledged, median of the rounds' values (lowest to highest round) | median 76.6 ms (66.9 to 115.2), p99 197 (179 to 368), worst 206 (187 to 379); one round of five had 42 of 448 acknowledgements over 250 ms. Without any probe: median 50.3, p99 85.3 (85 to 97), worst 88.9 | median 52.6 (49.4 to 69.4), p99 117 (116 to 173), worst 122 (119 to 194); none over 250 ms | p99 at most 1.5 times the no-probe arm's p99 in the same run (today 2.3, prototype 1.4), and no acknowledgement over 250 ms in any of 5 rounds. `measure.py live64` |
| Decoding what the detectors read of the layer and router statistics: 1,000 Granite-shaped rows (24 blocks, 32 experts), with `detect.rs`' own visitors against the typed maps (`measure.py decode`, `rust-reader bench`, 5 rounds, load 4.6 to 5.3) | `layers_json` 4.88 ms (4.86 to 4.92), `moe_stats_json` 4.27 ms (4.26 to 4.29) | `layers` 0.18 ms (0.18 to 0.19), `moe_stats` 0.069 ms (0.068 to 0.071) | Both under 0.5 ms per 1,000 rows. `rust-reader bench` on the lane's parts |
| The detectors' whole time per flush of 64 ranks of 50 steps, one thread (`detect_store --bench`, the server's own example, 64 ranks × 1,000 steps, 5 runs) | 72.7 ms (72.6 to 72.8); of which the JSON decoding measured apart, 34.7 ms (34.65 to 34.86), 48% | Not measurable without product code. Expected about 39 ms: today's time without the decoding, plus the typed decode at 1 ms per 3,200 rows | At most 50 ms on the same store in protocol 5's layout, with `detect_store --bench` on the same machine. Today's 72.7 is to be re-measured beside it, and the figure is then at most 0.7 times today's |
| Its parts | 4.86 MB for 1,000 rows | 4.37 MB (−10%); `read_part` 19.8 ms against 18.3 ms (nested columns decode a little slower) | No figure: the replay is not on the live path |
| `load_records` on that store (Python) | 342 ms (276 to 400) | 304 ms (239 to 317) | No slower than today, beyond the spread |

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
   - On one rank a probe batch costs 14.6 ms against 5.6 ms (section 6).
9. **The plan, Phase 2: "`155 ms` → `350 ms` Parquet flush penalty".** That is writing all 64 ranks' parts on a probe flush (README, "Re-measured with 25 kinds"), not one part. One probe part's write is 5.5 ms here.
10. **The plan, Phase 2: "`layers_json` / `moe_stats_json` cost `33 ms` of `54 ms`".**
    - The README's 54 ms is the first six kinds; twelve took 56 ms.
    - The 33 ms is "measured apart" by `detect_store`'s own decode timing.
    - Measured here on 64 ranks × 1,000 steps with the same example: 34.7 ms of 72.7 ms (48%), with today's per-rank kinds. "Two thirds" was the share with twelve kinds.
11. **Section 4.3: old parts "expire via bucket lifecycle policies".** Already withdrawn by the owner on Nov 24 (ledger section 20). Repeated here because a sidecar (Q2 b) would be a second family that rule could not tell apart either.

## 9. The lanes

**Two gated steps, one protocol bump, one push.**
- Step 1 is sections 1 to 3; step 2 is section 4.
- `PROTOCOL` becomes 5 in step 1, and the push follows step 2.
- Lanes are split by file, and two lanes never own one file.
- A lane's tests are its cases' paths. Its targets in `test_wave_al_contract.py` lose their `xfail` mark when they pass on its branch. The live-server, `runs` and MCP paths are tests each lane writes for its cases, failing first.

**Step 1: typed columns, the probe split, the versions.**

| Lane | Owns | Does | Cases it must make pass |
|---|---|---|---|
| AL-1 | `flight.py`, `server_detect.py` | Protocol 5's row layout: `flight_schema`, `record_to_row`, `row_to_record`, `rows_batch` for sections 1, 2 and 3. Metadata version 2. The older-server message. `PROTOCOL = 5` | every section 1 and 2 case's `row` path (`test_flight_gives_the_row`) |
| AL-2 | `ingest.rs`, `storage.rs`, `main.rs` | Metadata versions 1 and 2; row versions 0, 1 and 2. `PROTOCOL` 5. A read of a part that projects away `probe_maps` and `probe_histograms`, for replay | the matrix's server rows; `layers_model_order` on a live server across a restart |
| AL-3 | `detect.rs`, `detect/` | Reads `layers` and `moe_stats` from the typed columns where a batch has them, and the JSON columns where it does not. `non_finite` also reads the refused fields in `extra_json`. Never reads `probe_maps` or `probe_histograms` | every section 1 and 2 case's `rust_raw` and `rust_sdk` paths, live server included |
| AL-4 | `evidence.py`, `identity.py` | Section 1.5's Python changes: an entry or a `layers` value that is not an object read as absent; a `diversity` that is not a finite number taken from the counts. Reads a store of either layout through `row_to_record` | the `python` targets of `layers_entry_not_object` and `layers_value_not_object`; the samples comparison of `moe_stats_diversity_refused` |

- **AL-1 first.** AL-2 and AL-3 can start beside it: AL-2's projection and AL-3's dual reader need only the column names of sections 1.2 and 2.4.
- **AL-4 is the one Python reader change in step 1.** It is small, and it does not depend on the layout.
- **The step's gate:**
  - the whole SDK suite and `cargo test` with this tree's binary;
  - the golden and Gate 1 comparisons;
  - `measure.py ack`, `live64` and `decode` against section 6's figures;
  - `captures.py` on the seven captures.

**Step 2: the scalar-only record** (after step 1 is merged and gated).

| Lane | Owns | Does | Cases |
|---|---|---|---|
| AL-5 | `capture.py` (`Session.log`, `_no_step_for` as Q5 decides, the writer's block), `sink.py` if needed | Writes the record of section 4.2 | `scalar_record_keys` (capture), `scalar_log_refusals` |
| AL-1′ | `flight.py`, `server_detect.py` | The record's row (`seq`, `record_version`). Scalar-only records set apart before `_stream`, `FollowedDetect` and `stream_order` | `scalar_record_keys` (row) |
| AL-2′ | `ingest.rs`, `storage.rs` | Section 4.5 | `scalar_future_step_keeps_training_rows`, `scalar_replay_from_byte_0`, `scalar_seq_watermark_across_restart` |
| AL-4′ | `evidence.py`, `identity.py`, `compare.py`, `divergence.py`, `runs.py`, `mcp_server.py`, `compact.py` | Section 4.6 | every section 4 case's `python`, `runs` and `mcp_get_series` paths; `test_a_scalar_record_is_read_and_never_a_training_step` |

`detect.rs` has nothing to do in step 2 under Q4 (a). Under Q4 (b) it gains a lane, and the cases gain context expectations.

## 10. What the sibling clones do

- **arrow-rs** (`/Users/ak/Documents/tensrview/arrow-rs`, main of Nov 14; the server pins 60.1.0).
  - `DataType::Map(field, keys_sorted)` is `List<entries: Struct<key, value>>`. Its keys "should be hashable and unique", which is not enforced.
  - The Parquet reader builds every map with `keys_sorted = false` (`parquet/src/arrow/array_reader/map_array.rs`), and nothing in the path sorts. D1 measured that order holds.
  - `ArrowWriter::write` refuses a batch of another schema, so a part has one.
  - `ProjectionMask::roots` and `leaves` (`parquet/src/arrow/mod.rs`) read only the columns asked for.
- **wandb** (`core/internal/stream/handler.go` near line 994): a `log` with a `step` below the current one is ignored, with "Tried to log to step N that is less than the current step M … this data will be ignored", and pointed to `define_metric` for out-of-order axes. TensorQuill's scalar-only record takes out-of-order steps, since `seq` and not `step` orders it, and never touches the training step.
- **MLflow** (`mlflow/store/tracking/dbmodels/models.py`): a metric row's key is `(key, timestamp, step, run_uuid, value, is_nan)`, so the same value at the same time is stored once and every other value of a step is kept as history. `latest_metrics` keeps one row per key. TensorQuill deduplicates by its own `(run, rank, attempt, seq)`, which a replay repeats exactly, and reads the highest `seq` at a step.
- **InfluxDB 3** (`influxdb3_write/src/write_buffer/queryable_buffer.rs`, `sort_dedupe_persist`): a write's identity is its series and time, deduplicated at persist, last write winning. Its tables are flat columns, with no map or struct, and a field written with another type is refused (`influxdb3_catalog` `InvalidColumnType`). Read for the dedup rule; not copied.
- **datafusion-ducklake**: evolves `list`, `struct` and `map` columns recursively, with per-file name mappings (README). That is the by-name field lookup D3 asks for. It depends on DataFusion, a rejected non-goal, so ideas only, as the ledger already decided.
- **datafusion:** not needed. The detectors read Arrow arrays directly.
