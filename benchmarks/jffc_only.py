"""JFFC-only benchmark: whole model per server + JFFC load balancing.

Places an entire model instance on each server with sufficient memory,
then uses JFFC for load balancing across single-server chains.
Tests the efficacy of the cache reservation strategy by comparison.

Reference: Table 1 "JFFC only" column in the paper.
"""

from __future__ import annotations

import math
from typing import List

from petals_llm_chain_scheduling.algorithms.jffc import JFFCDispatcher
from petals_llm_chain_scheduling.data_models import (
    AllocatedChain,
    RoutingEdge,
    ServerParams,
)


def build_jffc_only_chains(
    servers: List[ServerParams],
    L: int,
    s_m: float,
    s_c: float,
) -> List[AllocatedChain]:
    """Build single-server chains for servers that can host all L blocks.

    Each chain consists of exactly one server hosting all L blocks.
    Capacity = floor((M_j - s_m * L) / (s_c * L)).

    Args:
        servers: List of server parameters.
        L: Total number of blocks.
        s_m: Block size in GB.
        s_c: Cache size per block per job in GB.

    Returns:
        List of AllocatedChain objects (one per eligible server).
    """
    chains = []
    chain_id = 0

    for s in servers:
        # Check if server can host all L blocks
        if s.memory_gb < s_m * L:
            continue

        # Capacity: residual memory / (L cache slots per job)
        residual = s.memory_gb - s_m * L
        capacity = int(math.floor(residual / (s_c * L)))
        if capacity < 1:
            continue

        service_time = s.tau_c + s.tau_p * L
        service_rate = 1.0 / service_time if service_time > 0 else 0.0

        chains.append(AllocatedChain(
            chain_id=chain_id,
            server_sequence=[s.server_id],
            edges=[],
            capacity=capacity,
            service_time=service_time,
            service_rate=service_rate,
        ))
        chain_id += 1

    return chains
