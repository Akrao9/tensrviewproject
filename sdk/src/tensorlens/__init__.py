"""TensorLens capture SDK.

    import tensorlens
    session = tensorlens.init(model)   # after model.to(device), before torch.compile
    model = torch.compile(model)
    ...train as usual...
    session.close()
"""

from .capture import Router, Session, init
from .distributed import parameter_fingerprints, replica_spread, spread_from_fingerprints
from .evidence import (
    DEAD_COLOR,
    IDLE_COLOR,
    STARVED_COLOR,
    MISSING_COLOR,
    build_evidence,
    detect_non_finite,
    detect_grad_spikes,
    detect_repeated_steps,
    detect_replica_drift,
    detect_stalled_weights,
    detect_starved_experts,
    effective_overload_threshold,
    expert_load_table,
    export_rerun,
    first_incident,
    incident_summary,
    load_records,
    routing_colors,
)
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
    "build_evidence",
    "detect_non_finite",
    "detect_grad_spikes",
    "detect_repeated_steps",
    "detect_replica_drift",
    "detect_stalled_weights",
    "detect_starved_experts",
    "effective_overload_threshold",
    "expert_load_table",
    "export_rerun",
    "first_incident",
    "incident_summary",
    "init",
    "load_records",
    "parameter_fingerprints",
    "replica_spread",
    "resolve_run",
    "routing_colors",
    "spread_from_fingerprints",
]
__version__ = "0.0.1"
