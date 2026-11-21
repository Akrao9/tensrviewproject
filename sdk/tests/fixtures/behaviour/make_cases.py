"""Writes the cases of `STEP_AND_EVIDENCE_CONTRACT.md`. Run from the repo root:

    python sdk/tests/fixtures/behaviour/make_cases.py

Written by hand from the page. `status` is `pin` for behaviour today's Python detectors already
have (the test runs those) or `new` for what is not built: there `expected` is the target, in the
page's words, for the lane that builds it. A `new` case over records also says, under `today`, what
today's Python detector answers on them, which is not the target: the test runs that, so the case
cannot pass before its rule is built. The lane that builds the rule removes `today` and makes the
case a pin. A value that is not finite is written as the string "nan", as JSON has no such number;
the test turns it back.
"""
import json

def steady(n=40, **last):
    """`n` quiet steps, then one more with `last` on it, then five quiet ones."""
    out = [{"step": s, "grad_norm": 1.0 + 0.01 * (s % 3)} for s in range(n)]
    return out + [{"step": n, **last}] + [{"step": n + 1 + k, "grad_norm": 1.0} for k in range(5)]

def pin(section, says, detector, records, fires, **more):
    return {"section": section, "status": "pin", "says": says, "detector": detector, "records": records,
            "expected": {"fires": fires, **more}}

def new(section, says, given, expected, today=None):
    return {"section": section, "status": "new", "says": says, "given": given, "expected": expected,
            **({"today": today} if today else {})}

def today(detector, fires, **more):
    """What today's Python detector answers on a `new` case's records: not the target."""
    return {"detector": detector, "fires": fires, **more}

BY = {"the_caller": "caller", "the_scaler": "scaler", "both": "both"}
WHO = {"caller": "only the caller reported", "scaler": "only the scaler reported", "both": "the caller and the scaler both reported"}

def skips(n, by, before=5):
    """`before` applied steps, then `n` skipped ones reported by `by`."""
    return [{"step": s, "step_skipped": False} for s in range(before)] + \
           [{"step": before + k, "step_skipped": True, "step_skipped_by": by} for k in range(n)]

def unchanged(steps, by=None):
    return [{"step": s, "step_skipped": True, **({"step_skipped_by": by} if by else {}),
             "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}} for s in range(steps)]

def around_a_caller_skip(middle, after=5, before=0):
    """`before` applied steps, five skips the scaler reported, one only the caller reported (with
    `middle` on it), then `after` more the scaler reported."""
    return skips(5, "scaler", before=before) + \
           [{"step": before + 5, "step_skipped": True, "step_skipped_by": "caller", **middle}] + \
           [{"step": before + 6 + k, "step_skipped": True, "step_skipped_by": "scaler"} for k in range(after)]

NAN_LAYER = {"blk": {"mean": "nan", "std": 1.0, "abs_max": 1.0}}

cases = {
 "a_spike_on_a_skipped_step": pin(1, "The total gradient norm is fifty times its usual size on a step whose update was skipped. Not judged.",
    "detect_grad_spikes", steady(grad_norm=50.0, step_skipped=True), False),
 "a_spike_on_an_applied_step": pin(1, "The same norm on a step that says its update was applied. Judged.",
    "detect_grad_spikes", steady(grad_norm=50.0, step_skipped=False), True, kind="grad_spike", step=40),
 "a_spike_with_nothing_said": pin(1, "The same norm on a step that says nothing of skipping, as every step did before the argument. Judged.",
    "detect_grad_spikes", steady(grad_norm=50.0), True, kind="grad_spike", step=40),
 "a_gradient_not_finite_on_a_skipped_step": pin(1, "A gradient norm that is not finite on a skipped step is what the skip is for. Nothing is raised for it.",
    "detect_non_finite", [{"step": 3, "grad_norm": "nan", "step_skipped": True}], False),
 "an_activation_not_finite_on_a_skipped_step": pin(1, "A layer statistic that is not finite is raised on a skipped step too: activations are still read.",
    "detect_non_finite", [{"step": 3, "grad_norm": "nan", "step_skipped": True,
                           "layers": {"blk": {"mean": "nan", "std": 1.0, "abs_max": 1.0}}}], True, kind="non_finite", step=3),
 "ten_skips_in_a_row_with_no_loss_scale": pin(1, "A capture written before `step_skipped_by`: ten skipped updates in a row, no loss scale, and nothing that says who reported them. They are read as a scaler's, and the streak is raised as before.",
    "detect_loss_scale_trouble", [{"step": s, "step_skipped": s >= 5} for s in range(15)], True,
    kind="loss_scale_trouble", step=14, reason="skips", consecutive_skips=10),
 "an_update_due_and_applied": new(1, "The caller says an update was due and was applied.",
    {"init": {"auto_step": False}, "calls": ["session.step(skipped=False)"]},
    {"records_carry": [{"step_skipped": False}], "records_lack": ["step_skipped_by"]}),
 "an_update_due_and_skipped": new(1, "The caller says an update was due and was skipped, and there is no scaler.",
    {"init": {"auto_step": False}, "calls": ["session.step(skipped=True)"]},
    {"records_carry": [{"step_skipped": True, "step_skipped_by": "caller"}]}),
 "a_skip_only_the_caller_reported": new(1, "The caller says the update was skipped; the SDK observed the scaler given to `init` apply it. The record says both.",
    {"init": {"auto_step": False, "scaler": "a GradScaler that applied this step"}, "calls": ["session.step(skipped=True)"]},
    {"records_carry": [{"step_skipped": True, "step_skipped_by": "caller", "scaler_applied": True}]}),
 "a_skip_the_caller_reported_with_no_scaler_outcome": new(1, "The caller says the update was skipped; a scaler was given to `init`, and it did not act on this update, so its outcome is unknown. The record does not say the scaler applied it.",
    {"init": {"auto_step": False, "scaler": "a GradScaler whose `step` was not called for this update"}, "calls": ["session.step(skipped=True)"]},
    {"records_carry": [{"step_skipped": True, "step_skipped_by": "caller"}], "records_lack": ["scaler_applied"]}),
 "a_skip_only_the_scaler_reported": new(1, "The caller says the update was applied; the scaler skipped it. Either one saying so makes it skipped.",
    {"init": {"auto_step": False, "scaler": "a GradScaler that skipped this step"}, "calls": ["session.step(skipped=False)"]},
    {"records_carry": [{"step_skipped": True, "step_skipped_by": "scaler"}]}),
 "a_skip_both_reported": new(1, "The caller and the scaler both say the same update was skipped.",
    {"init": {"auto_step": False, "scaler": "a GradScaler that skipped this step"}, "calls": ["session.step(skipped=True)"]},
    {"records_carry": [{"step_skipped": True, "step_skipped_by": "both"}]}),
 "an_accumulation_microstep": new(1, "Four forward and backward passes are accumulated into one update. The caller marks the one boundary, and the three microsteps before it end no record.",
    {"init": {"auto_step": False}, "calls": ["(backward)", "(backward)", "(backward)", "(backward)", "session.step(skipped=False)"]},
    {"records_carry": [{"step_skipped": False}], "records": 1}),
 "nothing_said_with_a_scaler": new(1, "The argument is left out and `init` was given a loss scaler. The record carries the scaler's verdict, as today, and says it is the scaler's.",
    {"init": {"auto_step": False, "scaler": "a GradScaler that skipped this step"}, "calls": ["session.step()"]},
    {"records_carry": [{"step_skipped": True, "step_skipped_by": "scaler"}]}),
 "nothing_said_without_a_scaler": new(1, "The argument is left out and there is no scaler. The record has no such key, as today.",
    {"init": {"auto_step": False}, "calls": ["session.step()"]}, {"records_lack": ["step_skipped"]}),
 "weights_unchanged_across_skipped_steps": new(1, "A parameter's norm is unchanged across three steps, and all three were skipped updates. That is not a stall: the weights were not meant to move.",
    {"records": unchanged(3)}, {"incident_kinds_absent": ["stalled_weights"]}),
 **{f"weights_unchanged_across_skips_{who}_reported": new(1, f"The same three steps, skipped updates that {WHO[by]}. Not a stall, whoever reported them.",
    {"records": unchanged(3, by)}, {"incident_kinds_absent": ["stalled_weights"]}) for who, by in BY.items()},
 "a_stall_across_a_skipped_step": new(1, "A parameter is unchanged at two applied steps, at a skipped one, and at one more applied step. The skipped step adds nothing to the streak and does not end it, so the stall is raised at the last step.",
    {"records": [{"step": s, **({"step_skipped": True, "step_skipped_by": "caller"} if s == 2 else {"step_skipped": False}),
                  "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}} for s in range(4)]},
    {"incident_kinds_present": ["stalled_weights"], "incident_carries": {"step": 3}}),
 "ten_skips_only_the_caller_reported": new(1, "Ten skipped updates in a row that only the caller reported. They are not the loss scale's trouble, and nothing is raised.",
    {"records": skips(10, "caller")}, {"incident_kinds_absent": ["loss_scale_trouble"]}),
 "ten_skips_the_scaler_reported": pin(1, "Ten skipped updates in a row that the scaler reported, on a capture that says so. Raised.",
    "detect_loss_scale_trouble", skips(10, "scaler"), True, kind="loss_scale_trouble", step=14, reason="skips", consecutive_skips=10),
 "ten_skips_both_reported": pin(1, "Ten skipped updates in a row that the caller and the scaler both reported. The scaler confirmed each, so the streak is raised.",
    "detect_loss_scale_trouble", skips(10, "both"), True, kind="loss_scale_trouble", step=14, reason="skips", consecutive_skips=10),
 "caller_skips_inside_a_run_of_scaler_skips": new(1, "Two applied steps, five skips the scaler reported, three only the caller reported, five more the scaler reported. The caller's are passed over: they neither add to the streak nor end it, and are in neither count. Ten in a row are reached at the last step.",
    {"records": skips(5, "scaler", before=2) + [{"step": 7 + k, "step_skipped": True, "step_skipped_by": "caller"} for k in range(3)]
                + [{"step": 10 + k, "step_skipped": True, "step_skipped_by": "scaler"} for k in range(5)]},
    {"incident_kinds_present": ["loss_scale_trouble"],
     "incident_carries": {"step": 14, "reason": "skips", "consecutive_skips": 10, "skipped_steps": 10, "recorded_steps": 12}}),
 "a_low_loss_scale_on_a_skip_only_the_caller_reported": pin(1, "A loss scale below the floor is recorded on a step whose skip only the caller reported. The scale is still read: it is the scale's own reading, not the skip's.",
    "detect_loss_scale_trouble", [{"step": 3, "step_skipped": True, "step_skipped_by": "caller", "grad_scale": 0.5}], True,
    kind="loss_scale_trouble", step=3, reason="scale"),
 **{f"an_activation_not_finite_on_a_skip_{who}_reported": pin(1, f"A layer statistic is not finite on a skip that {WHO[by]}. Raised for the layer, with no gradient named.",
    "detect_non_finite", [{"step": 3, "grad_norm": "nan", "step_skipped": True, "step_skipped_by": by, "layers": NAN_LAYER}], True,
    kind="non_finite", step=3, layers=["blk"], gradients=[]) for who, by in BY.items()},
 "a_gradient_not_finite_on_a_skip_the_caller_reported": new(1, "A parameter's gradient norm is not finite on a skip only the caller reported. No incident is raised from the gradient alone, and the evidence keeps it.",
    {"records": [{"step": 3, "grad_norm": "nan", "grad_norms": {"blk.weight": "nan"}, "step_skipped": True, "step_skipped_by": "caller"}]},
    {"incident_kinds_absent": ["non_finite"], "kind_values": {"non_finite": "checked"},
     "evidence_carries": {"skipped_updates": {"caller": 1, "scaler": 0, "both": 0,
                                              "non_finite_gradients": {"steps": 1, "first_step": 3, "gradients": ["blk.weight"]}}}}),
 "a_gradient_not_finite_on_a_skip_the_scaler_reported": new(1, "The same gradient on a skip only the scaler reported is what the scaler skipped for. No incident, and nothing kept but the count.",
    {"records": [{"step": 3, "grad_norm": "nan", "grad_norms": {"blk.weight": "nan"}, "step_skipped": True, "step_skipped_by": "scaler"}]},
    {"incident_kinds_absent": ["non_finite"], "evidence_carries": {"skipped_updates": {"caller": 0, "scaler": 1, "both": 0}}}),
 "a_gradient_not_finite_on_a_skip_both_reported": new(1, "The same gradient on a skip both reported. No incident, and the evidence keeps it, since the caller reported the skip.",
    {"records": [{"step": 3, "grad_norm": "nan", "grad_norms": {"blk.weight": "nan"}, "step_skipped": True, "step_skipped_by": "both"}]},
    {"incident_kinds_absent": ["non_finite"],
     "evidence_carries": {"skipped_updates": {"caller": 0, "scaler": 0, "both": 1,
                                              "non_finite_gradients": {"steps": 1, "first_step": 3, "gradients": ["blk.weight"]}}}}),
 "an_activation_and_a_gradient_not_finite_on_a_skip_the_caller_reported": new(1, "A layer statistic and a gradient norm are both not finite on a skip only the caller reported. The incident is raised for the layer and names no gradient; the gradient is kept in the evidence.",
    {"records": [{"step": 3, "grad_norm": "nan", "grad_norms": {"blk.weight": "nan"}, "step_skipped": True, "step_skipped_by": "caller", "layers": NAN_LAYER}]},
    {"incident_kinds_present": ["non_finite"], "incident_carries": {"step": 3, "layers": ["blk"], "gradients": []},
     "evidence_carries": {"skipped_updates": {"caller": 1, "scaler": 0, "both": 0,
                                              "non_finite_gradients": {"steps": 1, "first_step": 3, "gradients": ["blk.weight"]}}}}),
 "a_capture_written_before_the_key": new(1, "A skipped step with nothing that says who reported it, as every capture had before. The evidence has no `skipped_updates`, and is what it was.",
    {"records": [{"step": 3, "grad_norm": "nan", "step_skipped": True}]},
    {"incident_kinds_absent": ["non_finite"], "evidence_lacks": ["skipped_updates"]}),
 "a_caller_skip_of_unknown_scaler_outcome_inside_scaler_skips": new(1, "Five skips the scaler reported, one only the caller reported with the scaler's outcome unknown, five more the scaler reported. The streak is left as it is across the caller's skip and reaches ten at the last step: 10 recorded, 10 skipped, a streak of 10.",
    {"records": around_a_caller_skip({})},
    {"incident_kinds_present": ["loss_scale_trouble"],
     "incident_carries": {"step": 10, "reason": "skips", "consecutive_skips": 10, "skipped_steps": 10, "recorded_steps": 10},
     "counters_at_end": {"recorded_steps": 10, "skipped_steps": 10, "streak": 10}}),
 "a_caller_skip_the_scaler_applied_inside_scaler_skips": new(1, "The same steps, but the SDK observed the scaler apply the update the caller reported as skipped. The streak ends at five there, and five more do not raise the rule: 11 recorded, 10 skipped, a streak of 5.",
    {"records": around_a_caller_skip({"scaler_applied": True})},
    {"incident_kinds_absent": ["loss_scale_trouble"],
     "counters_at_end": {"recorded_steps": 11, "skipped_steps": 10, "streak": 5}}),
 "ten_scaler_skips_after_a_caller_skip_the_scaler_applied": new(1, "Two applied steps, five scaler skips, a caller's skip the scaler was seen to apply, then ten scaler skips. Raised at the last step with ten in a row; the caller's skip adds no skip and counts as a recorded observation of the scaler.",
    {"records": around_a_caller_skip({"scaler_applied": True}, after=10, before=2)},
    {"incident_kinds_present": ["loss_scale_trouble"],
     "incident_carries": {"step": 17, "reason": "skips", "consecutive_skips": 10, "skipped_steps": 15, "recorded_steps": 18}}),
 "a_nan_layer_beside_an_unread_gradient": new(2, "One parameter's gradient could not be read, and an unrelated layer's statistic is not finite. The finding stands, and the kind reads checked.",
    {"records": [{"step": 0, "grads_not_read": {"emb.weight": "a sparse gradient"}, "grad_norm": 1.0},
                 {"step": 1, "grad_norm": 1.0, "layers": {"blk": {"mean": "nan", "std": 1.0, "abs_max": 1.0}}}]},
    {"incident_kinds_present": ["non_finite"], "kind_values": {"non_finite": "checked"},
     "not_checked": [{"check": "gradients", "parameters": ["emb.weight"]}], "run_state": "unhealthy"}),
 "a_spike_beside_an_unread_gradient": new(2, "One parameter's gradient could not be read, so the total norm leaves it out. The kind is unavailable. A spike in the total as read is still raised.",
    {"records": [{"step": 0, "grads_not_read": {"emb.weight": "a sparse gradient"}, "grad_norm": 1.0}] + steady(grad_norm=50.0)[1:]},
    {"incident_kinds_present": ["grad_spike"], "kind_values": {"grad_spike": "unavailable", "non_finite": "checked"},
     "not_checked": [{"check": "gradients", "parameters": ["emb.weight"]}]}),
 "a_stall_beside_an_unread_gradient": new(2, "One parameter's gradient could not be read and another parameter stops changing. The stall is raised for the one that was read; the unread one is named and not judged.",
    {"records": [{"step": s, **({"grads_not_read": {"emb.weight": "a sparse gradient"}} if s == 0 else {}),
                  "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}} for s in range(4)]},
    {"incident_kinds_present": ["stalled_weights"], "kind_values": {"stalled_weights": "checked"},
     "not_checked": [{"check": "gradients", "parameters": ["emb.weight"]}]}),
 "one_rank_with_batch_hashes": new(2, "One rank sent batch hashes and no other rank did. It was compared with nothing.",
    {"ranks_with_batch_hashes": [0]},
    {"status": {"ranks_compared": []}, "kind_values": {"duplicate_batches": "not_configured"},
     "words": "one rank observed, zero peers compared"}),
 "silent_after_capture_stopped": new(3, "Rank 1's capture stopped after step 120 with a writer error, training went on, and later its heartbeats ceased. The silence is still raised, and the incident says when and why capture had stopped.",
    {"status_entry": {"rank": 1, "last_step": 120, "capture_stopped": {"reason": "sink_error", "seconds": 398.0}}, "then": "heartbeats cease"},
    {"incident_kinds_present": ["rank_silent"], "incident_carries": {"capture_stopped": {"reason": "sink_error", "seconds": 398.0}, "capture_last_step": 120},
     "words": "cannot tell from this rank whether training itself stalled"}),
 "silent_with_capture_running": new(3, "Rank 1's heartbeats cease while its capture was running. The incident is as today and has no such key.",
    {"status_entry": {"rank": 1, "last_step": 120}, "then": "heartbeats cease"},
    {"incident_kinds_present": ["rank_silent"], "incident_lacks": ["capture_stopped", "capture_last_step"]}),
 "silent_after_a_restart_that_stored_nothing": new(3, "Rank 1's earlier attempt reached step 100. The job was started again, the new attempt's capture stopped with a writer error before its first record, and later its heartbeats ceased. The incident gives the reason and a null last step: step 100 is not this attempt's.",
    {"earlier_attempt": {"attempt": 0, "last_step": 100},
     "status_entry": {"rank": 1, "attempt": 1, "last_step": None, "attempt_without_steps": True,
                      "capture_stopped": {"reason": "sink_error", "seconds": 42.0}}, "then": "heartbeats cease"},
    {"incident_kind_one_of": ["rank_silent", "job_stalled"],
     "incident_carries": {"capture_stopped": {"reason": "sink_error", "seconds": 42.0}, "capture_last_step": None},
     "words": "before this attempt stored a step", "words_absent": "100"}),
 "a_checkpoint_with_no_identified_model_state": new(4, "A resume from a checkpoint in which no entry is identified as the model's state. The reader does not know it is absent, only that the rule found none.",
    {"checkpoint_components": {"optimizer": "present", "model": "none identified"}},
    {"incident_kinds_present": ["resume_incomplete"], "findings": [{"component": "model", "finding": "not_identified"}],
     "words": "model state could not be identified", "words_absent": "model not in the checkpoint"}),
 "a_checkpoint_with_no_optimizer_state": new(4, "A resume from a checkpoint with no optimizer state. The other components keep the finding they have today.",
    {"checkpoint_components": {"optimizer": "absent", "model": "present"}},
    {"incident_kinds_present": ["resume_incomplete"], "findings": [{"component": "optimizer", "finding": "missing"}]}),
}
json.dump({"contract": "STEP_AND_EVIDENCE_CONTRACT.md", "cases": cases},
          open("sdk/tests/fixtures/behaviour/cases.json", "w"), indent=1, allow_nan=False)
print(len(cases), "cases:", sum(c["status"] == "pin" for c in cases.values()), "pins")
