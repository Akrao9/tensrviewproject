"""TensorQuill capture SDK.

    import tensorquill
    session = tensorquill.init(model)   # after model.to(device), before torch.compile
    model = torch.compile(model)
    ...train as usual...
    session.close()
"""

from .capture import Router, Session, init
from .data import batch_hash, checkpoint_manifest
from .data_checks import (
    detect_data_stalled,
    detect_duplicate_batches,
    detect_epoch_order_repeated,
    detect_resume_state,
)
from .distributed import parameter_fingerprints, replica_spread, spread_from_fingerprints
from .evidence import (
    DEAD_COLOR,
    IDLE_COLOR,
    STARVED_COLOR,
    MISSING_COLOR,
    build_evidence,
    detect_non_finite,
    detect_activation_anomalies,
    detect_grad_spikes,
    detect_loss_scale_trouble,
    scalar_context,
    detect_repeated_steps,
    detect_resume_lr_changes,
    detect_replica_drift,
    detect_stalled_weights,
    weight_norm_trends,
    detect_starved_experts,
    effective_overload_threshold,
    expert_load_table,
    export_rerun,
    first_incident,
    incident_summary,
    load_records,
    routing_colors,
)
from .compare import compare_runs
from .evidence_system import detect_memory_imbalance, detect_system_incidents, latest_memory
from .runs import list_runs
from .sink import JsonlSink, MemorySink, Sink
from .tracker import RunInfo, resolve_run

__all__ = [
    "DEAD_COLOR",
    "IDLE_COLOR",
    "STARVED_COLOR",
    "JsonlSink",
    "MISSING_COLOR",
    "MemorySink",
    "Router",
    "RunInfo",
    "Session",
    "Sink",
    "batch_hash",
    "build_evidence",
    "checkpoint_manifest",
    "compare_runs",
    "detect_data_stalled",
    "detect_duplicate_batches",
    "detect_epoch_order_repeated",
    "detect_resume_state",
    "detect_non_finite",
    "detect_activation_anomalies",
    "detect_grad_spikes",
    "detect_loss_scale_trouble",
    "detect_memory_imbalance",
    "detect_system_incidents",
    "scalar_context",
    "detect_repeated_steps",
    "detect_resume_lr_changes",
    "detect_replica_drift",
    "detect_stalled_weights",
    "weight_norm_trends",
    "detect_starved_experts",
    "effective_overload_threshold",
    "expert_load_table",
    "export_rerun",
    "first_incident",
    "incident_summary",
    "init",
    "latest_memory",
    "list_runs",
    "load_records",
    "parameter_fingerprints",
    "replica_spread",
    "resolve_run",
    "routing_colors",
    "spread_from_fingerprints",
]
__version__ = "0.0.1"
