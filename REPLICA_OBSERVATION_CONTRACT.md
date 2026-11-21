# Replica observations: the record and how it is compared

Step 0 of the SDK-22 wave (Nov 20, 2026), revised the same day after the author's review. The decision behind it is in `TensorQuill v3 — Product and Build Plan.md`, "Decision, SDK-22" and "Decision for the SDK-22 wave": the automatic drift check no longer gathers inside the training step. Each rank records what it sees; a reader that has every rank's record compares them.

This page is what the writer, the Python reader, the server and the driver build against. The cases in `sdk/tests/fixtures/replica/` are its examples, and `sdk/tests/test_replica_contract.py` holds a small reference reading of the rules that every case is checked with. Where this page and a case disagree, that is a fault in one of them, to be fixed before a lane builds on either.

The author released the four lanes that build against this page on Nov 20, 2026 ("Release lanes 1–4"): Z1 the writer, Z2 the Python reader, Z3 the server, Z4 the driver. The four were merged into `main` on Nov 20, 2026 (at `8c45fb8`), with lanes Z5 and Z6: the writer, both readers and the driver now use the keys below. The author signed off this page's six calls beyond the review the same day: a nonce at each `init`, a group of one rank not compared, the rule for overlapping groups, two entries when an unscoped spread and the observations both drift, incidents that name launch and group, and the second read of the launch.

## 1. What a rank writes

On each scheduled step (`(step + 1) % replica_every == 0`, as today) a rank that was given a `replica_group` adds one key, `replica`, to that step's record:

```json
"replica": {
  "v": 1,
  "launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47",
  "group": [0, 1, 2],
  "device": "cuda",
  "projection": "name-v1",
  "signature": "fold64-v1",
  "params": {
    "layers.0.weight": {"shape": [8, 4], "dtype": "float32",
                        "fp": [1.25, -0.5, 0.125], "sig": "-4611686018427387904"}
  },
  "unread": {"layers.1.weight": "on meta"}
}
```

- **`launch`**: which start of the job this record belongs to (section 2). 32 hex digits, or null when the rank could not learn it.
- **`group`**: the global ranks of the replica group this rank expects to be compared with, itself included, sorted. For the world group that is every rank. It is what "every required rank" means to a reader.
- **`device`**: the device type the fingerprints were computed on (`cpu`, `cuda`, `mps`).
- **`projection`**: `"name-v1"` when the third fingerprint is recorded, null when it is not (defined below).
- **`signature`**: the name of the signature method, or null when no signature is recorded.
- **`params`**: one entry for each compared parameter, keyed by the parameter's name. `fp` is `[norm, sum]` or `[norm, sum, projection]` in float64. `sig` is the signature as a decimal string: a 64-bit number written as a JSON number loses digits in some readers. A fingerprint that is not finite is written as the string `"nan"`, `"inf"` or `"-inf"`, as layer statistics are, because the stream's encoder refuses a bare NaN.
- **`unread`**: parameters this rank could not read, with why. A parameter is in `params` or in `unread`, never both.

The key travels in `extra_json` on the stream and in Parquet, where every key without a column of its own already goes, so the stream's schema and protocol number do not change.

**`name-v1`.** A parameter's projection is the sum of its elements times fixed weights. `name-v1` takes the weights from the parameter's name, not from its place in a list, so two ranks that list the same parameters in another order compute the same number.

- The seed is the first four bytes of the SHA-256 of the name, read as a big-endian unsigned 32-bit number. The name is the string that is the parameter's key in `params`, encoded as UTF-8 and not normalised. Python's `hash()` is never used: it differs from one process to the next.
- The weight of element `i` (the tensor flattened in row-major order, a complex tensor as its real pairs) is, with every step taken modulo 2^32: `h = i·0x9E3779B1 + (seed + 1)·0x7F4A7C15`; `h = (h ^ (h >> 16))·0x85EBCA6B`; `h = (h ^ (h >> 13))·0xC2B2AE35`; `h = h ^ (h >> 16)`; weight `= h·2/2^32 − 1`. These are today's weights (`distributed._projection_weights`) with the seed changed, and that function was run with seeds up to `0xFFFFFFFF` without overflow.
- Test vector: `"layers.0.weight"` has seed `1749384365` (`0x68457CAD`). Its first four weights are `-0.055502989795058966`, `-0.19597370875999331`, `-0.6474399706348777`, `0.46693889796733856`, and its projection of the elements `1, 2, 3, 4` is `-0.5220147273503244`.

**`fold64-v1`.** A parameter's signature is a 64-bit number computed from its elements' bits, its dtype and its shape, and from nothing else: not its name, device, strides, storage offset or the order a device reduces in. Every step is taken modulo 2^64, and `>>` is a logical shift (no sign bit copied in).

- `mix(z)` is SplitMix64's finaliser: `z = (z ^ (z >> 30))·0xBF58476D1CE4E5B9`; `z = (z ^ (z >> 27))·0x94D049BB133111EB`; `z = z ^ (z >> 31)`.
- The elements are read as `name-v1` reads them: the tensor flattened in row-major order, a complex tensor as its real pairs. Element `i`'s word `b_i` is its bit pattern as an unsigned integer of the element's width (the IEEE 754 bits of a float, two's complement for an integer, 0 or 1 for a bool), so a 2-byte element gives a word below 2^16.
- `S = Σ mix(b_i ^ (i·0x9E3779B97F4A7C15))`. A sum, so any order of reduction gives the same `S`; each word mixes in its index, so a value moved to another place changes it. An empty tensor has `S = 0`.
- Then the dtype and the shape: `h = S`, and for each `v` in turn of `D`, the number of dimensions, and each size in order, `h = mix(h ^ v)`. `D` is the first eight bytes of the SHA-256 of the dtype's name as `dtype` writes it in `params` (`"float32"`, `"bfloat16"`), read as a big-endian unsigned 64-bit number. The shape is the one `shape` writes.
- The signature is `h` read as a signed two's complement 64-bit number, and `sig` is its decimal string.
- Test vector: `[[1, 2], [3, 4]]` as `float32` (words `0x3F800000`, `0x40000000`, `0x40400000`, `0x40800000`) has `S = 7798465200413876260` and signature `1799304254312201492`. The same values as `bfloat16` (words `0x3F80`, `0x4000`, `0x4040`, `0x4080`) give `5716884024385227157`; the `float32` words with shape `[4]` give `3672505199788284773`; an empty `float32` tensor of shape `[0, 3]` gives `-8628698763751150559`. `D` of `"float32"` is `7679619720248017433`.

The function is deterministic, so two different signatures under it prove that the bits, the dtype or the shape differ, and nothing else can make them differ. The converse is not claimed: different inputs can collide. The definition holds a writer to bits: `0.0` and `-0.0`, or two NaNs with different payloads, have different signatures and equal fingerprints. A parameter whose signature cannot be computed is not read: it is in `unread`, never in `params` without a `sig`.

A change to any of this is a new name (`name-v2`), never a new meaning for the old one. The same holds for a signature method.

**Signatures.** For one method, two different signatures of a parameter prove that its bytes, shape or dtype differ. Two equal signatures do not prove the bytes are equal: 64 bits can collide. Readers compare signatures for equality only, and only when both ranks name the same method.

**What an explicit gather adds.** `record_replica_spread()` still writes `replica_spread`, the spread it gathered, as a table keyed by parameter name. When it gathered through a process group whose global ranks it could read, and the rank knows its launch, it also writes what the spread covers:

```json
"replica_spread_scope": {"launch": "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47", "group": [0, 1, 2]}
```

The key is absent when either is unknown: a table the job computed itself and passed in, a group whose ranks could not be read, a rank with no launch, and every old record. The scope is never taken from a `replica` on the same record: the explicit call can name another group, and can fall on a step that has no observation.

Old records keep `replica_spread` alone. A reader gives such a record the verdict it always did (section 6).

## 2. Which start of the job a record belongs to

Each process makes its own `attempt` from its own clock, so healthy ranks of one start have different attempts. Attempts are therefore not matched between ranks. `launch` is.

`launch` has one source: **a nonce that rank 0 makes at each `init`**: 128 bits from the operating system's entropy (`secrets.token_hex(16)`), never from a random number generator the training uses. Rank 0 shares it through the process group's store beside the run ID, under a key of its own that it sets before the run ID's. A rank that has read the run ID then looks for that key once and does not wait for it, so the exchange has no new wait, the run ID's entry keeps its two items, and a rank on an older SDK still reads the run ID. Rank 0 never waits. A rank whose store failed, whose wait for the run ID ran out, or that found no launch beside the run ID records `launch` as null. A job of one rank makes its own nonce.

**A fresh nonce can still be read stale, so a rank confirms it.** torchrun keeps one store across a restart of its workers, and the exchange's key was the same in every start. (Since wave AA, Nov 20, 2026, the keys are scoped by torchrun's restart count where one agent launched every rank, so a restarted rank no longer finds the earlier start's entry there; with several agents the count is not trusted and the key is still the same in every start.) Run on Nov 20, 2026 (PyTorch 2.15, torchrun with two workers on the loopback address, one restart): the restarted rank 1 found the earlier start's entry in the store before rank 0 had set the new one. Two rules follow.

- **The writer reads the launch a second time before it uses it.** What a rank read at `init` is provisional and is never written. A rank other than rank 0 reads the key once more after its first training step, by which time rank 0 of the same start has long passed its own `init` and set the key, and records what that second read gives; if it fails, null. As built in wave AA (Nov 20, 2026), the read runs on a thread of its own, started at `init` and released at the end of the first step, and nothing waits for it: the writer thread never does. So an observation written before the read has returned carries a null launch, and its step reads `not_established` (section 4), never a launch that the second read would have replaced. At the default interval the first scheduled step is long after that.
- **The reader allows a rank one attempt in a launch.** Rank 0 makes a new nonce at each `init`, so one start never gives a rank two attempts under one launch. When a file holds them, the first attempt (by contract 1's order) keeps the launch, and the rows of any later attempt are set aside as `not_established`: they are never paired with the earlier start's rows.

It has to be a value that is new at every start of the job, and two candidates are not:

- **A launcher's restart count.** torchrun's `TORCHELASTIC_RESTART_COUNT` is 0 on every fresh launch, so a job stopped and started again by hand under the same run ID would carry the same count twice.
- **Rank 0's attempt**, which the first draft of this page used. It is wall-clock milliseconds with a counter that lives in one process. A job started again where the clock is behind (another host, a clock set back) can get the same number, and the old start's rows would pair with the new one's.

One source also means two ranks of one start can never disagree about which source they used.

A reader's input is one run's records. Two records can be compared only when they name the same run, the same `launch` (not null) and the same `group`. A reader never pairs records across launches: after a restart, an earlier launch's row at step 50 is not compared with the new launch's row at step 50.

What the two stale-launch rules above do not cover is in section 9.

## 3. What can be compared

For one run, launch, group and step, a reader has zero or one observation from each rank of the group. Between the ranks that have one, each parameter is compared by name, when all of these hold on every rank that reports it:

| Check | When it fails |
|---|---|
| The parameter is in `params` on every reporting rank | Not compared; why: the `unread` reason, or "not recorded on rank N" |
| Same `shape` and `dtype` | Not compared; why names both |
| Same `device` | Fingerprints not compared; why: device types differ. Signatures are still compared |
| Same `projection` | The projection column is not compared; norm and sum are |
| Same `signature` method | Signatures not compared |

A parameter that is not compared is listed with its reason. It is never counted as agreeing. A group of one rank has no rank to be compared with, so none of its parameters is compared.

## 4. The four facts a reader reports

They are separate, and a reader gives each one for each compared step. The steps are listed by step, then the rows that name no launch, then by launch, then by group, and a step's own entry before the rows set aside from it, so both engines give the same list.

**Coverage.**
- `complete`: every rank in `group` has an observation for this launch and step.
- `insufficient`: some do not. The reader names the missing ranks. This is not a clean result and not an incident.
- `not_established`: the records cannot say whether they belong to one start, or who is compared with whom: a `launch` is null; or two ranks of one launch and step name groups that overlap and are not the same, or a rank names a group it is not in, which leaves the whole launch and step not established, a group apart from the faulty ones too; or the row is a rank's later attempt under a launch (section 2). The reader says which. When some ranks of a step name a launch and one names none, both are said: the launch's step is `insufficient` without that rank, and that rank's row is `not_established`.

Coverage says who sent a record. It does not say anything was compared. Only a step with `complete` coverage can agree or drift, and a lone rank's file is `insufficient` for a group of more than one rank.

**Fingerprints.** For each compared parameter and each fingerprint column, the spread is the largest value minus the smallest, divided by the mean of the ranks' norms (never less than `1e-12`), as `spread_from_fingerprints` computes today.
- `drift` when any spread is above the tolerance (`1e-6` by default), or when any rank's fingerprint is not finite.
- `agree` when no spread is, and at least one parameter's fingerprints were compared.
- `not_compared` when none was, with why: the device types differ, no parameter could be compared (every one unread, missing, or of another shape or dtype), or the group has one rank.

**Signatures.**
- `differ` when any compared parameter's signatures are not all equal, with the parameters.
- `agree` when none differs and at least one parameter's signatures were compared. It means no difference was found, not that the bytes were proved equal.
- `not_compared` when none was, with why: the methods differ or a rank records none, no parameter could be compared, or the group has one rank.

**Attribution (D1).** Which rank a difference is attributed to is a fifth fact, with its own section, 4a. It never changes the four above.

## 4a. Which rank (D1)

Released by the author on Nov 20, 2026; written here on the same day, and built in both engines and the driver in wave AB (`main` at `ac0159a`): the same planted run read from local files and from a live server gave the same parameter, the same outlier ranks and one incident on 2, 3 and 8 CPU ranks. Its cases are in `sdk/tests/fixtures/replica/attribution_cases.json`, checked by `sdk/tests/test_replica_attribution_contract.py`.

**What it is.** Where the ranks of a step differ on a parameter, attribution says which ranks stand apart from the rest, when a strict majority does not. The two readings word that differently, and the difference is meant. For fingerprints: a strict majority lies within the tolerance of the median in every compared column. For signatures, which are exact: a strict majority holds one value. The two can give different results on the same ranks (`three_of_four_within_tolerance_of_the_median`). It describes a difference. It raises nothing and silences nothing: section 5 alone decides what raises, and the count of incidents is the same with or without it.

**Where it is read.** Only on a step whose coverage is `complete`, from the rows that were compared: one observation from each rank of the group, the first attempt's where a rank has two under one launch (section 2). A step that is `insufficient` or `not_established` has no attribution, even when every rank that reported agrees or two of them differ: the missing rank could change either reading. A gathered spread holds one number for a parameter and no rank's value, so it attributes nothing.

**By fingerprints,** for each parameter in the step's `drift_parameters`, with `n` the number of ranks in the group:

1. `n = 2`: `inconclusive`, why `two_ranks`. Two ranks that differ establish that the replicas diverge and cannot say which is wrong.
2. A rank with a fingerprint that is not finite stands apart.
3. For each compared column (norm and sum, and the projection where it is compared), take the median of the finite ranks' values: the middle one, or with an even number the mean of the two middle ones. A rank stands apart when its value is further from that median than the tolerance, measured as section 4 measures a spread: the distance divided by the mean of the finite ranks' norms, never less than `1e-12`.
4. No rank stands apart: `inconclusive`, why `no_outlier`. The spread from the lowest rank to the highest is over the tolerance and no rank is far from the middle.
5. The ranks that do not stand apart, each within the tolerance of the median in every compared column, are a strict majority, more than `n / 2`: `attributed`, with `outliers`, the ranks that stand apart, in order. They need not hold one value: three ranks at 0.9 of the tolerance either side of the median are a majority of four, though no two of them on opposite sides are within the tolerance of each other.
6. Otherwise: `inconclusive`, why `no_majority`.

**By signatures,** for each parameter in the step's `signature_parameters`: `n = 2` is `inconclusive`, why `two_ranks`; one signature held by more than `n / 2` of the ranks is `attributed`, with the other ranks as `outliers`; anything else is `inconclusive`, why `no_majority`. This is the rule of SCOUT's exact classifier (`lm_resiliency/detection/c3.py`, `classify_evidence`). It is evidence, as the signatures are: a signature-only difference is shown with the rank whose bits stand apart, and raises nothing.

The two readings are separate entries and may differ on one parameter (`not_finite_on_two_of_three`). Each parameter is attributed on its own, so one step can name different ranks for different parameters, and a reader never carries one parameter's outlier to another.

**The entry.** The result gains one key, `attribution`, a list present only when it has an entry, so a result with no difference is spelled exactly as before. Each entry has `step`, `launch`, `group`, `parameter`, `by` (`fingerprints` or `signatures`), `verdict` (`attributed` or `inconclusive`), and `outliers` when attributed or `why` when not. The list is ordered by step, launch, group, parameter, then `by`. The `steps` and `replica_drift` lists do not change.

**A rank with no peer** is never said to agree: one rank observed, zero peers compared (section 3). The same words hold for batch hashes.

**Where a named rank is shown.** In the cross-rank result, and in the incident a reader builds from it. The incident names ranks by the fingerprints' reading only, since fingerprints are what raised it: for each parameter, the ranks named or that none could be. The signatures' reading is shown beside it as evidence and never as the incident's own, so in `not_finite_on_two_of_three` the incident names no rank. No rank's own report changes, since no rank's records can say it.

**What a named rank means.** By fingerprints, that it lies further than the tolerance from the median while a strict majority lies within it; by signatures, that its signature is not the one a strict majority holds. It does not say that rank is at fault: a fault every other rank shares would name the healthy one.

**For the engines that build this.** An engine that writes `attribution` no longer gives the 14 earlier cases named under `earlier_cases` their old object, since each gains the key. A test that compares a whole result with one of the two earlier files compares everything but `attribution` with that file, and `attribution` with this one (an absent key and an empty list are the same there). That applies to the Python reader's test, the server's, and `sdk/tests/test_wave_z_seams.py`. The server lists an attribution entry with its step's entry and drops it with it; `omitted` then counts the step as before.

**How this differs from two earlier wordings.** The Focused Plan's item says "further than the tolerance from the group's median", and this page said "a strict majority of the ranks hold one value". Under a tolerance, holding one value is not a relation that sorts ranks into groups (a can be near b, and b near c, with a far from c), so the rule above takes the median as the point to measure from and asks that a strict majority lie within the tolerance of it. "A strict majority holds one value" is kept for signatures alone. For values far from the tolerance the two wordings and this rule give the same answer; `three_of_four_within_tolerance_of_the_median` is where they part.

**The cases.**

| Case | What it pins |
|---|---|
| `two_ranks_differ` | Two ranks: divergence, no rank named |
| `three_ranks_two_to_one` | Two to one names the one |
| `three_ranks_rank_0_is_the_outlier` | Rank 0 is not a reference to compare against |
| `three_ranks_all_differ` | Three values: no majority |
| `four_ranks_two_to_two` | A half is not a strict majority |
| `four_ranks_three_to_one` | Three of four names the fourth |
| `four_ranks_two_one_one` | The largest group is not a majority |
| `five_ranks_three_one_one` | Two outliers beside a majority of three |
| `a_peer_is_missing_and_the_reporters_agree` | Insufficient coverage: nothing attributed, nothing agrees |
| `a_peer_is_missing_and_the_reporters_differ` | The same, when the reporters differ |
| `two_parameters_with_different_outliers` | Each parameter names its own rank; one incident |
| `signature_only_two_to_one` | A signature-only difference is shown with its rank and raises nothing |
| `signature_only_between_two_ranks` | The same between two ranks: no rank named |
| `signed_zero_on_one_of_three` | `0.0` against `-0.0`: evidence only |
| `below_tolerance_on_one_of_three` | A difference far inside the tolerance: evidence only |
| `not_finite_on_one_of_three` | A value that is not finite stands apart |
| `not_finite_on_two_of_three` | The two readings can differ |
| `spread_over_three_ranks_with_no_outlier` | Drift with no rank apart: detecting a drift and locating it are two questions |
| `three_of_four_within_tolerance_of_the_median` | A majority within the tolerance of the median need not hold one value |
| `only_the_projection_separates_the_outlier` | The projection column counts |
| `clean_three_ranks` | No difference, no entry |
| `group_of_one` | One rank observed, zero peers compared |
| `one_step_under_two_launches_with_different_outliers` | Each launch on its own rows |
| `a_set_aside_later_attempt_does_not_vote` | Only compared rows vote |
| `device_differs_and_one_signature_stands_apart` | Signatures are read where fingerprints are not |
| `a_gathered_spread_drifts_and_the_observations_agree` | A gathered spread names no rank |
| `one_clean_step_and_one_attributed` | Entries only for the step that differs |

Every case keeps its values far from the tolerance, but for two that are about it: `spread_over_three_ranks_with_no_outlier` (0.8 and 1.6 of it) and `three_of_four_within_tolerance_of_the_median` (0.9 and 4 of it), each still far further from the line than any rounding. The file also says, under `earlier_cases`, what attribution adds to each of the 39 cases of sections 2 to 6; their two files are unchanged.

## 5. What raises `replica_drift`

One rule for v1: **`replica_drift` is raised when coverage is `complete` and fingerprints `drift`.** The incident names the parameters, the step, the launch and the group compared.

- Fingerprints within tolerance and signatures different: the evidence says the signatures differ and names the parameters. No incident. Whether this raises on its own is decided after the eight-GPU run.
- Coverage `insufficient` or `not_established`, or coverage `complete` with fingerprints `not_compared`: no incident, and the kind does not read as checked for that step.

**A gathered spread beside observations.** A `replica_spread` is always judged on its own record, as in section 6, whatever else the step holds. What it decides beyond that depends on what it says it covers:

- **Same run, launch, group and step** (its `replica_spread_scope` names the launch and the group of the compared step): for each parameter the gathered table holds, the gathered spread decides, whether it shows drift or not, and the comparison of observations for that parameter is shown as evidence only. A parameter the table does not hold is decided by the observations. One step raises at most one `replica_drift` for a parameter.
- **Anything else** (another launch, another group, or no scope at all): it decides nothing but itself. The observations of that step are judged as if it were not there, so a clean spread of one launch never silences drift in another. When such a spread and the observations both show drift on one step, both are listed, each with its source: they cannot be shown to be the same comparison.

Incidents are listed by step, then by source, then by launch and group.

**The words in a result.** The fixtures spell each fact this way, and both engines use the same words.

| Where | Key | Values |
|---|---|---|
| Every entry | `coverage` | `complete`, `insufficient` (with `missing`, the ranks), `not_established` (with `why`) |
| A `not_established` entry | `why` | `launch`, `group`, `attempt` (with `ranks`, whose rows were set aside) |
| A `complete` entry | `fingerprints` | `agree`, `drift` (with `drift_parameters`), `not_compared` (with `fingerprints_why`) |
| | `fingerprints_why` | `device`, `no_parameters`, `one_rank` |
| | `signatures` | `agree`, `differ` (with `signature_parameters`), `not_compared` (with `signatures_why`) |
| | `signatures_why` | `method`, `no_parameters`, `one_rank` |
| | `not_compared` | Each parameter left out, with `unread`, `not_recorded` or `shape_dtype` |
| | `projection_compared` | `false` when the ranks' projection settings differ; absent otherwise |
| | `gathered` | The parameters a gathered spread of this launch, group and step decides |
| An incident | `source` | `replica` (with `launch` and `group`) or `replica_spread` (with them when it has a scope) |
| The result, when it has an entry | `attribution` | A list of entries (section 4a) |
| An attribution entry | `by` | `fingerprints`, `signatures` |
| | `verdict` | `attributed` (with `outliers`, the ranks) or `inconclusive` (with `why`) |
| | `why` | `two_ranks`, `no_majority`, `no_outlier` |

## 6. Old captures

A record with `replica_spread` and no `replica` is judged as before: drift when a spread is above the tolerance, on that rank's record alone. Its verdict does not change, and it is not mixed with observations.

## 7. When a verdict is due

- **The clock starts** when the last required observation is accepted: for the server, when it has stored and acknowledged the batch that holds it; for a reader of files, when the file holds the line.
- **The clock stops** when the verdict is visible to a reader: for the server, when the status or report that carries it is written; for a reader of files, at its next look.
- **The bound**: one server status interval (`--watchdog-seconds`), or one look of a watching reader.

A rank that is late leaves the step `insufficient` until its observation arrives, then the step is judged. A rank that never reports leaves it `insufficient` for good, and the reader keeps saying which rank is missing. Neither holds up a training step on any rank: nothing in the SDK waits for another rank's observation.

The proof for this uses a rank that is alive and training while its telemetry is delayed or suppressed. Killing a rank would stall the training's own collectives and prove nothing about TensorQuill.

## 8. What each lane takes from this

- **Writer** (`capture.py`, `distributed.py`, and `tracker.py`, where the store exchange lives): sections 1 and 2; no collective in a step by default; the scope key on an explicit gather. It defines `fold64-v1` on this page to the precision `name-v1` has, with a test vector, before it merges. It times the signature and the device synchronisation it needs, and reports the record's size: with a parameter's name, shape and dtype written at every scheduled step, an observation of a few hundred parameters is tens of kilobytes (one recorded capture names 290).
- **Python reader** (`evidence.py`) and **server** (`detect.rs`, `ingest.rs`, `detect/data.rs`): sections 2 to 7; both pass every case in the fixtures with the same result.
- **Driver** (`replica_check.py`): section 7's two proofs, on two and three ranks; its end-to-end results are taken on the trial merge, where the real writer and readers are together.

**Where the server's result is carried.** The server writes the result of sections 2 to 6 for a run as one object under the key `replica` at the top level of that run's `watchdog.json`, beside `ranks_compared`: `{"v": 1, "tolerance": <number>, "steps": [...], "replica_drift": [...]}`, with `legacy` when there is one, and with `attribution` when it has an entry once D1 is built (section 4a), each list spelled as the fixtures spell it. The Python all-ranks reader, which opens no part and runs no detector, takes the result from there; a reader of local rank files computes the same object from the records. The driver's clock (section 7) stops at the first `watchdog.json` whose `replica` object holds the step's verdict. How many steps the object lists, and what the server holds for steps that stay `insufficient`, is the server lane's to bound and to say.

**What a bound drops.** The server lists at most 256 steps and 1 MiB of their entries, holds at most 32 MiB of observation JSON for a run, and keeps the first 64 incidents of steps it no longer lists. Only when one of these dropped something, the object carries one more key, `omitted`: `steps` (how many step numbers are no longer listed), `late_rows` (rows that arrived for a step no longer listed: counted, never listed again), `replica_drift` (incidents no longer listed), and `frozen`, a list of `{"launch", "step", "why", "rows_not_compared"}` for each launch and step whose entry no longer changes, with `frozen_more` counting those past 64. `why` is `held_bytes` (the unfinished observations were let go), `restart` (taken up from the last status after a restart, then met by a row that was not compared) or `re_pair` (an earlier attempt's backlog arrived after the others' observations were let go). A frozen entry keeps reading as it did and never becomes `complete`. A reader comparing the server's object with its own leaves `omitted` out. A server started with `--no-detect` writes no `replica` key at all, which a reader must not call an older server.

**Stored and shown.** Where two sources drift on one step, the result keeps two entries, each with its own scope, as section 5 has it. A reader may show them as one incident, and then shows both sources.

**Who edits what.** Only the writer lane edits this page, and only to define `fold64-v1`. No lane edits the fixtures, `make_cases.py` or `sdk/tests/test_replica_contract.py`: a lane that finds this page and a case disagreeing stops and reports.

## 9. Not settled here

- Whether a signature mismatch alone raises. The author's call of Nov 20, 2026: it stays evidence. The eight-GPU run measures how often signatures differ on healthy steps and plants a byte difference below the fingerprint tolerance, signed zero among them; whether any mismatch is an incident, and under which conditions, is decided after that. Megatron's hash is of a CPU float32 copy and `fold64-v1` is of the original bits, so the two can disagree about the same tensors.
- D1 is written (section 4a) and not built. The author reviewed it at `278e216` and signed off its rules on Nov 20, 2026, the two that went beyond the request among them: a rank is also named from signatures, as evidence, and only the fingerprints' reading gives an incident its ranks; and where the largest spread is over the tolerance and no rank is far from the median, the reading is `inconclusive` with why `no_outlier`, one incident and no rank named. The author's one change is made: "a strict majority holds one value" is said of signatures only, and the four-rank case that separates the two wordings is added. Still to be tested: that the two engines agree on the merged implementation. Its acceptance test on real processes is the driver's: three CPU ranks with one drifted rank name that rank, and two ranks report drift and name none.
- SDK-06: whether to record again after a second optimizer steps. Its cost is hashing, device synchronisation and record size, to be measured, not assumed free.
- **L25, fixed in wave AA for one agent and open for several.** The exchange keys are scoped by `TORCHELASTIC_RESTART_COUNT` when one torchrun agent launched every rank (`GROUP_WORLD_SIZE` is 1 and `WORLD_SIZE` matches the group), tested on the store with the restarted rank other than 0 arriving before rank 0. With several agents the count is not trusted (a restart for a membership change does not raise it, and a node that joins starts at 0), so the stale read stays possible there and the two rules above are what stand against it. A whole restarted job has not completed on the machine this was built on.
- **What the two stale-launch rules leave.** A rank whose second read is still stale (rank 0 of its start had not reached `init` by the rank's first observation, which needs a job whose ranks do not step together), under a launch in which the earlier start's same rank never reported: its rows would be the first attempt under the old launch and could pair with the earlier start's other ranks. A store left on disk by a crashed job (a `FileStore` file) is the same exposure from the first read on. Closing it for certain needs rank 0 to wait for the others or a collective at `init`, and this wave took both out of the training path. The same store entry also carries the run ID, which is read the same way today (bug list, L25).
- Whether a restarted torchrun job completes on the setup the probe ran on: in the probe's second start rank 0 printed nothing before the run was stopped. Not looked into.
