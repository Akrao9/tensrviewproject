"""Writes the cases of D1, section 4a of `REPLICA_OBSERVATION_CONTRACT.md`. Run from the repo root:

    python sdk/tests/fixtures/replica/make_attribution_cases.py

Every expectation here is written by hand from the page, never computed: the test
(`sdk/tests/test_replica_attribution_contract.py`) reads the rule a second time and must agree.
Each case says three things apart: the coverage of every step, the attribution entries, and how
many incidents the step raises (section 5's rule, which D1 does not change).

This file does not touch the two earlier case files. `earlier_cases` says what attribution adds to
each of their 39 cases that has an entry; a case not named there gets none.
"""
import json

RUN = "r1"
A = {r: 1795000000101 + 113 * r for r in range(5)}
L0 = "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47"
L1 = "b09d7e6f5a4c4b3a8f2e1d0c9b8a7f65"
G2, G3, G4, G5 = (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4)


def P(norm, s, proj, sig, shape=(8, 4), dtype="float32"):
    return {"shape": list(shape), "dtype": dtype, "fp": [norm, s, proj], "sig": str(sig)}


W = P(3.5, -1.25, 0.75, -4611686018427387904)           # the value healthy ranks hold
WX = P(3.6, -1.2, 0.7, 999)                             # far from it in every column
WY = P(3.4, -1.3, 0.8, 555)                             # far from both
W_SIG = P(3.5, -1.25, 0.75, 123456789)                  # the same fingerprints, other bits
W_ZERO = P(3.5, -1.25, 0.75, -4611686018427387903)      # a 0.0 that is -0.0 on this rank
W_ULP = P(3.5000000001, -1.25, 0.75, 222)               # one element a step away: far inside the tolerance
W_PROJ = P(3.5, -1.25, 0.7, 1337)                       # two elements swapped: only the projection moves
W_NAN = P("nan", "nan", "nan", 8)
B = P(0.5, 0.25, -0.125, 1152921504606846977, shape=(8,))
BX = P(0.6, 0.25, -0.125, 4243, shape=(8,))


def rec(rank, group, w=W, b=B, step=99, launch=L0, attempt=None, device="cuda", spread=None, scope=None):
    out = {"schema_version": 1, "run_id": RUN, "rank": rank, "attempt": A[rank] if attempt is None else attempt, "step": step}
    if spread is not None:
        out["replica_spread"] = spread
    if scope is not None:
        out["replica_spread_scope"] = {"launch": scope[0], "group": list(scope[1])}
    out["replica"] = {"v": 1, "launch": launch, "group": list(group), "device": device, "projection": "name-v1",
                      "signature": "fold64-v1", "params": {"w": w, "b": b}}
    return out


def cov(group, coverage="complete", step=99, launch=L0, **more):
    return {"step": step, "launch": launch, "group": list(group), "coverage": coverage, **more}


def named(parameter, by, outliers, group, step=99, launch=L0):
    return {"step": step, "launch": launch, "group": list(group), "parameter": parameter, "by": by,
            "verdict": "attributed", "outliers": outliers}


def unnamed(parameter, by, why, group, step=99, launch=L0):
    return {"step": step, "launch": launch, "group": list(group), "parameter": parameter, "by": by,
            "verdict": "inconclusive", "why": why}


def both(parameter, outliers, group, **where):
    return [named(parameter, "fingerprints", outliers, group, **where), named(parameter, "signatures", outliers, group, **where)]


def neither(parameter, why, group, **where):
    return [unnamed(parameter, "fingerprints", why, group, **where), unnamed(parameter, "signatures", why, group, **where)]


def case(says, records, coverage, attribution, incidents):
    return {"says": says, "records": records, "expected": {"coverage": coverage, "attribution": attribution, "incidents": incidents}}


cases = {
 "two_ranks_differ": case(
  "Two ranks, and they differ. That establishes that the replicas diverge. It cannot say which one is wrong, so no rank is named.",
  [rec(0, G2), rec(1, G2, w=WX)], [cov(G2)], neither("w", "two_ranks", G2), 1),
 "three_ranks_two_to_one": case(
  "Three ranks, two hold one value and rank 2 another. A strict majority holds one value, so the other rank is named, for that parameter.",
  [rec(0, G3), rec(1, G3), rec(2, G3, w=WX)], [cov(G3)], both("w", [2], G3), 1),
 "three_ranks_rank_0_is_the_outlier": case(
  "The rank that drifted is rank 0. A reader that compares every rank with rank 0, as Megatron's check does, would name ranks 1 and 2.",
  [rec(0, G3, w=WX), rec(1, G3), rec(2, G3)], [cov(G3)], both("w", [0], G3), 1),
 "three_ranks_all_differ": case(
  "Three ranks, three values. No value is held by a majority, so no rank is named.",
  [rec(0, G3), rec(1, G3, w=WX), rec(2, G3, w=WY)], [cov(G3)], neither("w", "no_majority", G3), 1),
 "four_ranks_two_to_two": case(
  "Four ranks split two and two. Neither half is a strict majority, so no rank is named.",
  [rec(0, G4), rec(1, G4), rec(2, G4, w=WX), rec(3, G4, w=WX)], [cov(G4)], neither("w", "no_majority", G4), 1),
 "four_ranks_three_to_one": case(
  "Four ranks, three hold one value. Rank 3 is named.",
  [rec(0, G4), rec(1, G4), rec(2, G4), rec(3, G4, w=WX)], [cov(G4)], both("w", [3], G4), 1),
 "four_ranks_two_one_one": case(
  "Four ranks: two hold one value and the other two each hold their own. The two are the largest group and not a strict majority of four, so no rank is named.",
  [rec(0, G4), rec(1, G4), rec(2, G4, w=WX), rec(3, G4, w=WY)], [cov(G4)], neither("w", "no_majority", G4), 1),
 "five_ranks_three_one_one": case(
  "Five ranks, three hold one value and two each hold their own. Three of five is a strict majority, so both of the others are named.",
  [rec(0, G5), rec(1, G5), rec(2, G5), rec(3, G5, w=WX), rec(4, G5, w=WY)], [cov(G5)], both("w", [3, 4], G5), 1),
 "a_peer_is_missing_and_the_reporters_agree": case(
  "Rank 2's observation has not arrived and the two that reported agree. Coverage is insufficient: nothing is attributed and nothing reads as agreeing.",
  [rec(0, G3), rec(1, G3)], [cov(G3, "insufficient", missing=[2])], [], 0),
 "a_peer_is_missing_and_the_reporters_differ": case(
  "Rank 2's observation has not arrived and the two that reported differ. Still insufficient: no drift is raised and no rank is named until every rank of the group is there.",
  [rec(0, G3), rec(1, G3, w=WX)], [cov(G3, "insufficient", missing=[2])], [], 0),
 "two_parameters_with_different_outliers": case(
  "On one step rank 0 holds another bias and rank 2 another weight. Each parameter names its own rank; the step has one incident, which lists both parameters.",
  [rec(0, G3, b=BX), rec(1, G3), rec(2, G3, w=WX)], [cov(G3)], both("b", [0], G3) + both("w", [2], G3), 1),
 "signature_only_two_to_one": case(
  "The fingerprints agree on all three ranks and rank 1's signature differs. The difference is shown, with the rank whose bits stand apart, and raises nothing.",
  [rec(0, G3), rec(1, G3, w=W_SIG), rec(2, G3)], [cov(G3)], [named("w", "signatures", [1], G3)], 0),
 "signature_only_between_two_ranks": case(
  "Two ranks whose fingerprints agree and whose signatures differ. Shown, no rank named, nothing raised.",
  [rec(0, G2), rec(1, G2, w=W_SIG)], [cov(G2)], [unnamed("w", "signatures", "two_ranks", G2)], 0),
 "signed_zero_on_one_of_three": case(
  "One element is 0.0 on two ranks and -0.0 on rank 2. Every fingerprint is equal and the bits differ: evidence, with the rank, and no incident.",
  [rec(0, G3), rec(1, G3), rec(2, G3, w=W_ZERO)], [cov(G3)], [named("w", "signatures", [2], G3)], 0),
 "below_tolerance_on_one_of_three": case(
  "One element on rank 0 is the smallest step away from the others'. The fingerprints differ far inside the tolerance, so they agree; the signatures show the difference and the rank. No incident.",
  [rec(0, G3, w=W_ULP), rec(1, G3), rec(2, G3)], [cov(G3)], [named("w", "signatures", [0], G3)], 0),
 "not_finite_on_one_of_three": case(
  "Rank 1's fingerprints are not finite. A value that is not finite stands apart from any value, so rank 1 is named.",
  [rec(0, G3), rec(1, G3, w=W_NAN), rec(2, G3)], [cov(G3)], both("w", [1], G3), 1),
 "not_finite_on_two_of_three": case(
  "Two of three ranks are not finite, with the same bits. By fingerprints no finite value is held by a majority, so no rank is named. By signatures the two hold one value and rank 0 stands apart. The two readings are separate and may differ; neither changes that the step raises.",
  [rec(0, G3), rec(1, G3, w=W_NAN), rec(2, G3, w=W_NAN)], [cov(G3)],
  [unnamed("w", "fingerprints", "no_majority", G3), named("w", "signatures", [0], G3)], 1),
 "spread_over_three_ranks_with_no_outlier": case(
  "Three ranks whose norms step up evenly: the spread from the lowest to the highest is over the tolerance, and each rank is within it of the middle one. The step drifts and no rank stands apart.",
  [rec(0, G3), rec(1, G3, w=P(3.5000028, -1.25, 0.75, 71)), rec(2, G3, w=P(3.5000056, -1.25, 0.75, 72))], [cov(G3)],
  [unnamed("w", "fingerprints", "no_outlier", G3), unnamed("w", "signatures", "no_majority", G3)], 1),
 "only_the_projection_separates_the_outlier": case(
  "Two elements are swapped on rank 0: the norm and the sum are equal on every rank and the projection is not. Rank 0 is named.",
  [rec(0, G3, w=W_PROJ), rec(1, G3), rec(2, G3)], [cov(G3)], both("w", [0], G3), 1),
 "clean_three_ranks": case(
  "Three ranks that agree. There is no attribution entry: an entry exists only where something differs.",
  [rec(0, G3), rec(1, G3), rec(2, G3)], [cov(G3)], [], 0),
 "group_of_one": case(
  "A group of one rank. One rank observed, zero peers compared: it never reads as agreeing, and nothing is attributed.",
  [rec(0, (0,))], [cov((0,))], [], 0),
 "one_step_under_two_launches_with_different_outliers": case(
  "The job was started twice under one run ID, and step 99 drifted in both starts: rank 0 in the first and rank 2 in the second. Each launch is attributed on its own rows.",
  [rec(0, G3, w=WX), rec(1, G3), rec(2, G3)] + [rec(r, G3, w=WX if r == 2 else W, launch=L1, attempt=A[r] + 900000) for r in G3],
  [cov(G3), cov(G3, launch=L1)], both("w", [0], G3) + both("w", [2], G3, launch=L1), 2),
 "a_set_aside_later_attempt_does_not_vote": case(
  "Rank 2 drifted. Rank 1 was started again and read the earlier start's launch, so its later attempt's row is set aside (section 2); that row holds rank 2's value. If it voted, two ranks would hold that value and rank 0 would be named. It does not: rank 2 is named.",
  [rec(0, G3), rec(1, G3), rec(2, G3, w=WX), rec(1, G3, w=WX, attempt=A[1] + 900000)],
  [cov(G3), cov(G3, "not_established", why="attempt", ranks=[1])], both("w", [2], G3), 1),
 "device_differs_and_one_signature_stands_apart": case(
  "Rank 2 is on another device type, so fingerprints are not compared (section 3) and nothing can drift. Signatures are still compared, and rank 1's stands apart: shown with the rank, nothing raised.",
  [rec(0, G3), rec(1, G3, w=W_SIG), rec(2, G3, device="cpu")], [cov(G3)], [named("w", "signatures", [1], G3)], 0),
 "a_gathered_spread_drifts_and_the_observations_agree": case(
  "An explicit gather of this launch and group shows drift, and the ranks' observations agree. The gathered spread raises, as section 5 has it. It holds one number for each parameter and no rank's value, so it names no rank.",
  [rec(0, G2, spread={"w": [0.03, 0.01, 0.02], "b": [0.0, 0.0, 0.0]}, scope=(L0, G2)), rec(1, G2)], [cov(G2)], [], 1),
 "one_clean_step_and_one_attributed": case(
  "Two scheduled steps: the first agrees, and on the second rank 1 holds another weight. Only the second has an entry.",
  [rec(r, G3, step=99) for r in G3] + [rec(r, G3, step=199, w=WX if r == 1 else W) for r in G3],
  [cov(G3, step=99), cov(G3, step=199)], both("w", [1], G3, step=199), 1),
}

# What attribution adds to the earlier cases (`no_incident_cases.json`, `incident_cases.json`).
# All but one compare two ranks, which names no rank.
two = lambda *parameters, launch=L0, by=("fingerprints", "signatures"): [
    unnamed(p, b, "two_ranks", G2, launch=launch) for p in parameters for b in by]
earlier = {
 "signature_only_disagreement": two("w", by=("signatures",)),
 "explicit_spread_clean_and_observations_drift": two("w"),
 "drift_on_one_of_three": both("w", [1], G3),
 "drift_between_two_ranks": two("b"),
 "drift_in_the_projection_only": two("w"),
 "not_finite_on_one_rank": two("w"),
 "shape_differs": two("b"),
 "projection_setting_differs": two("b"),
 "explicit_spread_and_observations_on_one_step": two("w"),
 "explicit_spread_in_another_launch": two("w", launch=L1),
 "explicit_spread_of_another_group": two("w"),
 "explicit_spread_without_a_scope": two("w"),
 "explicit_spread_covers_one_parameter": two("b", "w"),
 "explicit_spread_without_a_scope_and_both_drift": two("w"),
}

head = {"contract": "REPLICA_OBSERVATION_CONTRACT.md, section 4a", "tolerance": 1e-6}
json.dump({**head, "cases": cases, "earlier_cases": earlier},
          open("sdk/tests/fixtures/replica/attribution_cases.json", "w"), indent=1, allow_nan=False)
print(len(cases), "cases;", len(earlier), "earlier cases with an entry")
