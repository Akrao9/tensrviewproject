//! Lane AD10 (Nov 21, 2026): the setting without the trend series, which no build of the server
//! uses (the author chose to keep `gradient_trend` and `attention_trends` on every path), tested so
//! that it is not left untested. It needs `RankDetectors::without_trend_series`, which the base does
//! not have, so this file is apart from `lane_ad10_tests.rs`, which compiles on the base.

use std::sync::Arc;

use super::lane_ad10_tests::{block_differs, measurement_cases};
use super::lane_ad8_tests::stored;
use super::{report, RankDetectors, Thresholds};

/// The setting no build uses, tested so that it is not left untested: detectors without the trend
/// series write neither trend (absent, never a shorter one) and the two bounded blocks as before.
#[test]
fn without_the_trend_series_neither_trend_is_written_and_the_bounded_blocks_are() {
    let mut out = Vec::new();
    for (name, body) in measurement_cases() {
        let records = body["records"].as_array().unwrap();
        let mut detectors = RankDetectors::new(&Arc::new(Thresholds::default())).without_trend_series();
        for record in records {
            detectors.append(0, &stored(std::slice::from_ref(record)));
        }
        let got = report("ad10", 0, &detectors, 0);
        for block in ["gradient_trend", "attention_trends"] {
            if let Some(found) = got.get(block) {
                out.push(format!("{name}: {block} written without its series: {found}"));
            }
        }
        for block in ["update_trends", "weight_jumps"] {
            if let Some(expected) = body["expected"].get(block) {
                block_differs(&name, &got, block, expected, &mut out);
            }
        }
    }
    assert!(out.is_empty(), "{}", out.join("\n"));
}
