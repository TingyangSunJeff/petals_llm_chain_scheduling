"""Shared data classes for petals_llm_chain_scheduling.

All core data structures used across algorithms, orchestrator, and experiment modules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InfeasiblePlacementError(Exception):
    """Raised when servers cannot cover all L blocks."""


class UnstableSystemError(Exception):
    """Raised when arrival rate >= total service rate (system cannot stabilize)."""


class NoFeasibleParameterError(Exception):
    """Raised when no feasible cache reservation parameter c exists."""


# ---------------------------------------------------------------------------
# Server and model parameters
# ---------------------------------------------------------------------------

@dataclass
class ServerParams:
    """Parameters for a single GPU server."""
    server_id: str
    memory_gb: float        # M_j in GB
    tau_c: float            # mean communication time (seconds)
    tau_p: float            # mean per-block computation time (seconds)
    mig_profile: str        # "3g.40gb" or "2g.20gb"
    peer_id: Optional[str] = None  # hivemind PeerID base58

    def __post_init__(self):
        if self.memory_gb <= 0:
            raise ValueError(f"memory_gb must be positive, got {self.memory_gb}")
        if self.tau_c <= 0:
            raise ValueError(f"tau_c must be positive, got {self.tau_c}")
        if self.tau_p <= 0:
            raise ValueError(f"tau_p must be positive, got {self.tau_p}")


@dataclass
class ModelParams:
    """Parameters for the served model."""
    num_blocks: int         # L (32 for LLaMA-2-7B)
    block_size_gb: float    # s_m in GB
    cache_size_gb: float    # s_c in GB per block per job
    model_name: str         # e.g. "meta-llama/Llama-2-7b-hf"

    def __post_init__(self):
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.block_size_gb <= 0:
            raise ValueError(f"block_size_gb must be positive, got {self.block_size_gb}")
        if self.cache_size_gb <= 0:
            raise ValueError(f"cache_size_gb must be positive, got {self.cache_size_gb}")


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    servers: List[ServerParams]
    model: ModelParams
    arrival_rate: float     # lambda (req/s)
    rho_bar: float = 0.7   # target load factor
    c_optimization: str = "theorem4_lower"  # or "surrogate"

    def __post_init__(self):
        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {self.arrival_rate}")
        if not (0 < self.rho_bar < 1):
            raise ValueError(f"rho_bar must be in (0, 1), got {self.rho_bar}")


# ---------------------------------------------------------------------------
# Block placement (GBP-CR output)
# ---------------------------------------------------------------------------

@dataclass
class ServerBlockAssignment:
    """Block assignment for a single server from GBP-CR."""
    server_id: str
    start_block: int        # a_j (0-indexed)
    num_blocks: int         # m_j(c)
    chain_index: int        # which chain this server belongs to


@dataclass
class ChainInfo:
    """A single disjoint chain from GBP-CR."""
    chain_id: int
    server_ids: List[str]
    service_time: float     # T_k
    service_rate: float     # mu_k = 1/T_k


@dataclass
class BlockPlacement:
    """Complete block placement result from GBP-CR."""
    server_assignments: Dict[str, ServerBlockAssignment]
    chains: List[ChainInfo]
    c: int                  # cache reservation parameter used
    total_service_rate: float


# ---------------------------------------------------------------------------
# Cache allocation (GCA output)
# ---------------------------------------------------------------------------

@dataclass
class RoutingEdge:
    """An edge in the routing topology for GCA."""
    from_server: str        # server i (or "dummy_start")
    to_server: str          # server j (or "dummy_end")
    blocks_at_to: int       # m_ij = a_j + m_j - a_i - m_i
    cost: float             # tau_c_j + tau_p_j * m_ij


@dataclass
class AllocatedChain:
    """A chain with allocated cache capacity from GCA."""
    chain_id: int
    server_sequence: List[str]
    edges: List[RoutingEdge]
    capacity: int           # c_k (max concurrent jobs)
    service_time: float     # T_k
    service_rate: float     # mu_k


@dataclass
class CacheAllocation:
    """Complete cache allocation result from GCA."""
    chains: List[AllocatedChain]
    total_service_rate: float  # sum(c_k * mu_k)
    total_capacity: int        # sum(c_k)


# ---------------------------------------------------------------------------
# Inference request / result
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """An incoming inference request."""
    request_id: int
    prompt: str
    max_new_tokens: int
    input_tokens: int
    submit_time: float


@dataclass
class PendingRequest:
    """A request waiting in the central queue."""
    request: InferenceRequest
    enqueue_time: float


@dataclass
class InferenceResult:
    """Result of a completed inference request."""
    request_id: int
    generated_text: str
    metrics: "RequestMetrics"


# ---------------------------------------------------------------------------
# Theorem 4 bounds
# ---------------------------------------------------------------------------

@dataclass
class ResponseTimeBounds:
    """Theorem 4 bounds on mean response time."""
    lower_bound: float
    upper_bound: float
    mean_occupancy_lower: float
    mean_occupancy_upper: float
    c: int
    num_chains: int


# ---------------------------------------------------------------------------
# Metrics (forward reference resolved at runtime)
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    """Per-request timing metrics."""
    request_id: int
    submit_time: float
    start_time: float       # when inference begins
    end_time: float
    response_time: float    # end - submit
    waiting_time: float     # start - submit
    service_time: float     # end - start
    chain_id: int


@dataclass
class StatsSummary:
    """Summary statistics for a timing metric."""
    mean: float
    median: float
    p95: float
    p99: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass
class MetricsSummary:
    """Aggregated metrics matching Table 1 of the paper."""
    response_time: StatsSummary
    waiting_time: StatsSummary
    service_time: StatsSummary
