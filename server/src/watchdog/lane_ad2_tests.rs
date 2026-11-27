//! Lane AD2: the missing-rank lookup and the time the status holds the progress lock (register,
//! section 8: "Missing-rank detection searches rank lists with `.contains()`, quadratic in world
//! size, while holding the progress lock").
//!
//! The status a run's ranks give must be the same, byte for byte, as before the lookup was made
//! linear: [`the_status_and_run_state_are_the_bytes_they_were_before_the_lookup_was_made_linear`]
//! holds the fingerprints of the documents `eb5a71b` (the lane's base) wrote for the same input.

use super::*;

/// Seconds after `t0` at which every fleet below is judged: past the default silence floor (300 s),
/// so the ranks not seen are called missing, and before the default critical time of a fresh rank.
const JUDGED_AT: u64 = 400;

fn key(rank: i32) -> Key {
    ("run".to_string(), rank)
}

fn at(t0: Instant, seconds: u64) -> Instant {
    t0 + Duration::from_secs(seconds)
}

/// A run of `world` ranks of which `seen` connected, with the variety a status shows: pauses,
/// a stack, streams that finished and that dropped, a rank silent, rows dropped and batches
/// refused, a capture stopped, a new attempt without steps. Every event is at a fixed time after
/// `t0`, so two fleets built alike are alike. `declared`: the batches name the world size.
fn fleet(world: i32, seen: &[i32], declared: bool, t0: Instant) -> Watchdog {
    let size = declared.then_some(world);
    let dog = Watchdog::new(WatchdogConfig::default());
    for &rank in seen {
        let key = key(rank);
        dog.stream_opened(&key, at(t0, 0));
        dog.heartbeat(&key, "train", 5.0, Some(0), None, at(t0, 1));
        if rank % 5 == 0 {
            dog.heartbeat(&key, "eval", 5.0, Some(10), None, at(t0, 20));
            dog.heartbeat(&key, "train", 5.0, Some(12), None, at(t0, 50 + (rank % 3) as u64));
        }
        dog.batch(&key, 0, Some(49), size, at(t0, 60));
        if rank % 17 != 3 {
            // Rank 3 of every 17 stops after its first batch: silent at `JUDGED_AT`.
            dog.batch(&key, 0, Some(99), size, at(t0, 120));
            dog.batch(&key, 0, Some(149), size, at(t0, 380));
        }
        if rank % 13 == 0 {
            let stack = json!({"threads": [{"name": "MainThread", "frames": [{"file": "train.py", "line": rank, "function": "step"}]}]});
            dog.heartbeat(&key, "train", 5.0, Some(20), Some(stack), at(t0, 390));
        } else {
            dog.heartbeat(&key, "train", 5.0, Some(20), None, at(t0, 390));
        }
        if rank % 7 == 0 {
            dog.stream_ended(&key, true, at(t0, 385));
        } else if rank % 11 == 0 {
            dog.stream_ended(&key, false, at(t0, 385));
        }
        if rank % 29 == 1 {
            dog.capture_stopped(&key, "out_of_memory", at(t0, 100));
        }
        if rank % 31 == 2 {
            assert!(dog.attempt_seen(&key, 5, at(t0, 300)));
        }
        if rank % 19 == 0 {
            dog.dropped(&key, &[("step", 2)], at(t0, 61));
            dog.refused(&key, "rank", at(t0, 62));
        }
    }
    dog
}

/// Ranks below `world` that connected: all but one in eight (`most_seen`), or one in eight.
fn ranks(world: i32, most_seen: bool) -> Vec<i32> {
    (0..world).filter(|rank| (rank % 8 == 5) != most_seen).collect()
}

/// The run's status as `run_status` writes it, with a fixed write time, and its members in order.
fn status_of(dog: &Watchdog, now: Instant) -> Value {
    let ranks = dog.ranks.lock().unwrap();
    let mut members: Vec<(i32, &Progress)> = ranks.iter().map(|((_, rank), p)| (*rank, p)).collect();
    members.sort_by_key(|(rank, _)| *rank);
    run_status("run", &members, now, 0, &dog.config, true)
}

/// The reports a server's detectors could hold for the ranks seen: one in three judged and
/// current, one in nine of another attempt, one in ten behind, the rest none.
fn reports_for(seen: &[i32]) -> BTreeMap<i32, RankReport> {
    seen.iter()
        .filter(|rank| *rank % 3 != 2)
        .map(|&rank| {
            let report = RankReport {
                kinds: if rank % 23 == 0 {
                    vec!["loss_spike".to_string()]
                } else {
                    Vec::new()
                },
                not_checked: if rank % 4 == 0 {
                    vec!["update_size".to_string()]
                } else {
                    Vec::new()
                },
                attempt: if rank % 9 == 0 { 7 } else { 0 },
                last_step: Some(149),
                behind: rank % 10 == 0,
            };
            (rank, report)
        })
        .collect()
}

/// FNV-1a over the bytes, with their length: enough to tell two documents apart in a test.
fn fingerprint(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{}:{hash:016x}", text.len())
}

/// One input of the fingerprints: the world size, whether most ranks were seen, the ranks seen,
/// whether the batches declare the world size, and the second after `t0` the status is taken at.
type Case = (i32, bool, Vec<i32>, bool, u64);

/// The cases the fingerprints cover: every world size with most ranks seen and most missing,
/// judged past the silence floor with the size declared; then a rank above the declared world size
/// (which raises the expected count past it), no declared size (the gaps below the highest rank),
/// and a status taken inside the floor's grace (ranks not seen, none yet missing).
fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for world in [1, 8, 64, 512, 4096] {
        for most_seen in [true, false] {
            cases.push((world, most_seen, ranks(world, most_seen), true, JUDGED_AT));
        }
    }
    let mut beyond = ranks(64, true);
    beyond.push(70);
    cases.push((64, true, beyond, true, JUDGED_AT));
    for world in [64, 4096] {
        cases.push((world, false, ranks(world, false), false, JUDGED_AT));
        cases.push((world, false, ranks(world, false), true, 200));
        cases.push((world, true, ranks(world, true), true, 200));
    }
    cases
}

fn documents((world, _, seen, declared, judged_at): &Case) -> (String, String) {
    let t0 = Instant::now();
    let dog = fleet(*world, seen, *declared, t0);
    let now = at(t0, *judged_at);
    let status = status_of(&dog, now);
    let state = run_state(&status, &reports_for(seen), true);
    (status.to_string(), state.to_string())
}

/// Prints the fingerprints of every case (`cargo test fingerprints_print -- --ignored --nocapture`),
/// as they were taken at the base commit for the test below.
#[test]
#[ignore]
fn fingerprints_print() {
    for case in cases() {
        let (status, state) = documents(&case);
        let (world, most_seen, seen, declared, judged_at) = &case;
        println!(
            "({world}, {most_seen}, {}, {declared}, {judged_at}, \"{}\", \"{}\"),",
            seen.len(),
            fingerprint(&status),
            fingerprint(&state)
        );
    }
}

/// The median of `runs` timings of `f`, in milliseconds.
fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let started = Instant::now();
            f();
            started.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// The timings in the lane's report (`cargo test --release lock_hold_timings -- --ignored
/// --nocapture`): how long a status pass holds the progress lock, and how long the whole pass and
/// the run state take, at world sizes 8, 512 and 4,096, with most ranks seen and most missing.
#[test]
#[ignore]
fn lock_hold_timings() {
    for world in [8, 512, 4096] {
        for most_seen in [true, false] {
            let seen = ranks(world, most_seen);
            let t0 = Instant::now();
            let dog = fleet(world, &seen, true, t0);
            let now = at(t0, JUDGED_AT);
            let runs = if world > 1000 { 41 } else { 201 };
            let pass = median_ms(runs, || {
                std::hint::black_box(dog.statuses(now, true));
            });
            let held = median_ms(runs, || {
                std::hint::black_box(held_for_status(&dog, now));
            });
            let status = status_of(&dog, now);
            let reports = reports_for(&seen);
            let state = median_ms(runs, || {
                std::hint::black_box(run_state(&status, &reports, true));
            });
            println!(
                "world {world:>5}, {:>11}: {:>5} seen; lock held {held:.3} ms; statuses() {pass:.3} ms; run_state {state:.3} ms",
                if most_seen { "most seen" } else { "most missing" },
                seen.len()
            );
        }
    }
}

/// What a status pass does while it holds the progress lock: the copy. At the base commit it was
/// the whole pass, `dog.statuses(now, true)`, and the timings in the report were taken so.
fn held_for_status(dog: &Watchdog, _now: Instant) -> usize {
    dog.snapshot().len()
}

/// The fingerprints `fingerprints_print` gave at `eb5a71b`, the lane's base, where the lookup was a
/// search of a list and the lock was held for the whole pass: (world size, most seen, ranks seen,
/// size declared, second judged at, status, run state).
///
/// Eight of them (six statuses and two run states, in the six cases that hold ranks 275, 649, ... or
/// 2893) were taken again at wave AH, Nov 22, 2026: those ranks stop after their first batch and
/// their stream then drops, so they now read `reconnecting` with the warning they had and
/// `stopped_as`, and are listed as stopped (P1-1). Checked rank by rank when they were retaken: no
/// other rank's entry carries `stopped_as`. The other 26 are the base's.
///
/// Seven more (two statuses and five run states, all five cases of world 4,096) were taken again
/// at lane AR-4, Nov 27, 2026: a status lists at most 1,000 missing ranks and counts the others in
/// `missing_ranks_omitted`, and the run state's `ranks_unknown` lists at most 1,000 and counts the
/// others in `ranks_omitted`. Checked when they were retaken: each of the seven cuts a list of more
/// than 1,000 ranks (2,584 or 2,582 missing ranks counted, 1,448 to 2,830 unknown ones), and none of
/// the other 27, every case of world 512 or less among them, cuts any.
const AT_BASE: &[(i32, bool, usize, bool, u64, &str, &str)] = &[
    (1, true, 1, true, 400, "1223:e003d7fce7c72736", "371:a1d2652f4b8bc9c9"),
    (1, false, 0, true, 400, "331:15179483c82d6405", "108:3f785a961528264b"),
    (8, true, 7, true, 400, "5085:82608f7e9bb17c6b", "779:2fc696b7d1f3c5d1"),
    (8, false, 1, true, 400, "1347:b7233eb92b17e04b", "529:6f211b74fe7ecc7c"),
    (
        64,
        true,
        56,
        true,
        400,
        "34349:30b96943cd940a9a",
        "891:d05fb849631b5937",
    ),
    (64, false, 8, true, 400, "5831:bead73de2da542c8", "607:a130b11d64279cfe"),
    (
        512,
        true,
        448,
        true,
        400,
        "269766:956dfbd746d7ec46",
        "1983:b82bc65798191ef2",
    ),
    (
        512,
        false,
        64,
        true,
        400,
        "42416:ad27fe6f54849a6f",
        "2491:ce7f8b9c5118b2ed",
    ),
    (
        4096,
        true,
        3584,
        true,
        400,
        "2164038:1bbe9d6b7460e2e0",
        "5699:bc7a326935e704e1",
    ),
    (
        4096,
        false,
        512,
        true,
        400,
        "317288:355b8e288a420fab",
        "4737:0c9803a3efbd4099",
    ),
    (
        64,
        true,
        57,
        true,
        400,
        "35045:b91077196cdc8d8c",
        "912:5d2f6ae67ae83253",
    ),
    (
        64,
        false,
        8,
        false,
        400,
        "5823:013e4c395045e596",
        "601:6606fb7934b885f4",
    ),
    (64, false, 8, true, 200, "4627:e1eed769876e8391", "357:3b54ae8eab89d65e"),
    (
        64,
        true,
        56,
        true,
        200,
        "30153:65f4a90c5e0c07bf",
        "755:fc577581c2452f5f",
    ),
    (
        4096,
        false,
        512,
        false,
        400,
        "317288:9f2846b3511a0607",
        "4737:784261886b830bcb",
    ),
    (
        4096,
        false,
        512,
        true,
        200,
        "303791:02a67843428ed4d9",
        "4737:0c9803a3efbd4099",
    ),
    (
        4096,
        true,
        3584,
        true,
        200,
        "2123513:2742447490cf087d",
        "5699:bc7a326935e704e1",
    ),
];

/// Not a test that fails at the base commit: it holds the change to what it was, the same bytes
/// for the same input, over world sizes 1 to 4,096 with most ranks seen and most missing, a rank
/// above the declared size, no declared size, and a status inside the silence floor's grace.
#[test]
fn the_status_and_run_state_are_the_bytes_they_were_before_the_lookup_was_made_linear() {
    let cases = cases();
    assert_eq!(cases.len(), AT_BASE.len());
    for (case, base) in cases.iter().zip(AT_BASE) {
        let (world, most_seen, seen, declared, judged_at) = case;
        assert_eq!(
            (*world, *most_seen, seen.len(), *declared, *judged_at),
            (base.0, base.1, base.2, base.3, base.4)
        );
        let (status, state) = documents(case);
        assert_eq!(
            (fingerprint(&status).as_str(), fingerprint(&state).as_str()),
            (base.5, base.6),
            "world {world}, most seen {most_seen}, declared {declared}, at {judged_at} s"
        );
    }
}

/// The pass that copies the ranks and builds after the lock is let go writes what `run_status`
/// writes over the ranks held, run by run, with the pass's own write time.
#[test]
fn a_status_pass_built_from_the_copy_is_the_status_of_the_ranks_held() {
    let t0 = Instant::now();
    let dog = fleet(512, &ranks(512, true), true, t0);
    for rank in [0, 3, 9] {
        dog.batch(&("other".to_string(), rank), 0, Some(9), Some(16), at(t0, 10));
    }
    let now = at(t0, JUDGED_AT);
    let statuses = dog.statuses(now, true);
    assert_eq!(statuses.keys().collect::<Vec<_>>(), ["other", "run"]);
    let ranks = dog.ranks.lock().unwrap();
    for (run_id, status) in &statuses {
        let mut members: Vec<(i32, &Progress)> = ranks
            .iter()
            .filter(|((run, _), _)| run == run_id)
            .map(|((_, rank), p)| (*rank, p))
            .collect();
        members.sort_by_key(|(rank, _)| *rank);
        let written = status["written_at_ms"].as_i64().unwrap();
        let expected = run_status(run_id, &members, now, written, &dog.config, true);
        assert_eq!(status.to_string(), expected.to_string(), "{run_id}");
    }
}
