"""Parameter c optimization for GBP-CR.

Searches c in [1, c_max] to minimize mean response time, using either
Theorem 4 lower bound or the surrogate objective c * K(c).

Reference: Section 3.1.3 (Eq. 9) and Section 3.2.3 of the paper.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from petals_llm_chain_scheduling.data_models import (
    BlockPlacement,
    CacheAllocation,
    InfeasiblePlacementError,
    NoFeasibleParameterError,
    ServerParams,
)
from petals_llm_chain_scheduling.algorithms.gbp_cr import compute_max_blocks, gbp_cr
from petals_llm_chain_scheduling.algorithms.gca import gca
from petals_llm_chain_scheduling.algorithms.theorem4 import compute_stationary_bounds


def compute_c_max(servers: List[ServerParams], s_m: float, s_c: float) -> int:
    """c_max = floor((max_j(M_j) - s_m) / s_c)."""
    if not servers or s_c <= 0:
        return 0
    max_mem = max(s.memory_gb for s in servers)
    if max_mem <= s_m:
        return 0
    return int(math.floor((max_mem - s_m) / s_c))


def optimize_c(
    servers: List[ServerParams],
    L: int,
    s_m: float,
    s_c: float,
    lam: float,
    rho_bar: float,
    method: str = "theorem4_lower",
) -> Tuple[int, BlockPlacement, CacheAllocation]:
    """Search c in [1, c_max] to minimize response time.

    For each c: runs GBP-CR -> GCA -> evaluates objective.

    Args:
        servers: List of server parameters.
        L: Total number of blocks.
        s_m: Block size in GB.
        s_c: Cache size per block per job in GB.
        lam: Arrival rate (requests/s).
        rho_bar: Target maximum system load in (0, 1).
        method: "theorem4_lower" (default) or "surrogate".

    Returns:
        (c_star, best_placement, best_allocation)

    Raises:
        NoFeasibleParameterError: If no feasible c found.
    """
    c_max = compute_c_max(servers, s_m, s_c)
    if c_max < 1:
        raise NoFeasibleParameterError(
            "No feasible c: c_max < 1. Servers lack sufficient memory."
        )

    best_c: Optional[int] = None
    best_obj = float("inf")
    best_placement: Optional[BlockPlacement] = None
    best_allocation: Optional[CacheAllocation] = None

    for c in range(1, c_max + 1):
        try:
            placement = gbp_cr(servers, L, s_m, s_c, c, lam, rho_bar)
        except InfeasiblePlacementError:
            continue

        allocation = gca(placement, servers, L, s_m, s_c)

        if not allocation.chains:
            continue

        if method == "surrogate":
            # Eq. 9: minimize c * K(c)
            K_c = len(placement.chains)
            obj = c * K_c
        else:
            # Theorem 4 lower bound
            rates = sorted(
                [ch.service_rate for ch in allocation.chains],
                reverse=True,
            )
            caps = [ch.capacity for ch in allocation.chains]
            # Re-sort capacities to match rate ordering
            chain_sorted = sorted(
                allocation.chains,
                key=lambda ch: ch.service_rate,
                reverse=True,
            )
            rates = [ch.service_rate for ch in chain_sorted]
            caps = [ch.capacity for ch in chain_sorted]

            total_rate = sum(r * c_k for r, c_k in zip(rates, caps))
            if lam >= total_rate:
                continue  # unstable, skip

            try:
                bounds = compute_stationary_bounds(lam, rates, caps)
                obj = bounds.lower_bound
            except Exception:
                continue

        if obj < best_obj:
            best_obj = obj
            best_c = c
            best_placement = placement
            best_allocation = allocation

    if best_c is None or best_placement is None or best_allocation is None:
        raise NoFeasibleParameterError(
            f"No feasible c found in [1, {c_max}] for lambda={lam}, "
            f"rho_bar={rho_bar}, method={method}."
        )

    return best_c, best_placement, best_allocation
