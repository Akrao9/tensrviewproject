"""Writes the cases of `STEP_AND_EVIDENCE_CONTRACT.md`. Run from the repo root:

    python sdk/tests/fixtures/behaviour/make_cases.py

Written by hand from the page. `status` is `pin` for behaviour today's Python detectors already
have (the test runs those) or `new` for what is not built: there `expected` is the target, in the
page's words, for the lane that builds it. A value that is not finite is written as the string
"nan", as JSON has no such number; the test turns it back.
"""
import json

def steady(n=40, **last):
    """`n` quiet steps, then one more with `last` on it, then five quiet ones."""
    out = [{"step": s, "grad_norm": 1.0 + 0.01 * (s % 3)} for s in range(n)]
    return out + [{"step": n, **last}] + [{"step": n + 1 + k, "grad_norm": 1.0} for k in range(5)]

def pin(section, says, detector, records, fires, **more):
    return {"section": section, "status": "pin", "says": says, "detector": detector, "records": records,
            "expected": {"fires": fires, **more}}

def new(section, says, given, expected):
    return {"section": section, "status": "new", "says": says, "given": given, "expected": expected}

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
 "ten_skips_in_a_row_with_no_loss_scale": pin(1, "A caller reports ten skipped updates in a row and records no loss scale. The streak is raised.",
    "detect_loss_scale_trouble", [{"step": s, "step_skipped": s >= 5} for s in range(15)], True,
    kind="loss_scale_trouble", step=14, reason="skips", consecutive_skips=10),
 "an_update_due_and_applied": new(1, "The caller says an update was due and was applied.",
    {"init": {"auto_step": False}, "calls": ["session.step(skipped=False)"]}, {"records_carry": [{"step_skipped": False}]}),
 "an_update_due_and_skipped": new(1, "The caller says an update was due and was skipped.",
    {"init": {"auto_step": False}, "calls": ["session.step(skipped=True)"]}, {"records_carry": [{"step_skipped": True}]}),
 "an_accumulation_microstep": new(1, "Four forward and backward passes are accumulated into one update. The caller marks the one boundary, and the three microsteps before it end no record.",
    {"init": {"auto_step": False}, "calls": ["(backward)", "(backward)", "(backward)", "(backward)", "session.step(skipped=False)"]},
    {"records_carry": [{"step_skipped": False}], "records": 1}),
 "nothing_said_with_a_scaler": new(1, "The argument is left out and `init` was given a loss scaler. The record carries the scaler's verdict, as today.",
    {"init": {"auto_step": False, "scaler": "a GradScaler that skipped this step"}, "calls": ["session.step()"]}, {"records_carry": [{"step_skipped": True}]}),
 "nothing_said_without_a_scaler": new(1, "The argument is left out and there is no scaler. The record has no such key, as today.",
    {"init": {"auto_step": False}, "calls": ["session.step()"]}, {"records_lack": ["step_skipped"]}),
 "weights_unchanged_across_skipped_steps": new(1, "A parameter's norm is unchanged across three steps, and all three were skipped updates. That is not a stall: the weights were not meant to move.",
    {"records": [{"step": s, "step_skipped": True, "weights_unchanged": ["blk.weight"], "grad_norms": {"blk.weight": 0.5}} for s in range(3)]},
    {"incident_kinds_absent": ["stalled_weights"]}),
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
