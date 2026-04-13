"""Algorithm 1: Greedy Block Placement with Cache Reservation (GBP-CR).

Sorts servers by amortized service time and greedily forms disjoint chains
covering all L blocks. Complexity: O(J log J).

Reference: Section 3.1.3 of the paper.
"""

from __future__ import annotations

import math
from typing import List

from petals_llm_chain_scheduling.data_models import (
    BlockPlacement,
    ChainInfo,
    InfeasiblePlacementError,
    ServerBlockAssignment,
    ServerParams,
)


def compute_max_blocks(M_j: float, s_m: float, s_c: float, c: int, L: int) -> int:
    """Eq. 6: m_j(c) = min(floor(M_j / (s_m + s_c * c)), L).

    Returns the maximum number of blocks server j can host given
    cache reservation parameter c.
    """
    if s_m + s_c * c <= 0:
        return 0
    return min(int(math.floor(M_j / (s_m + s_c * c))), L)


def compute_service_time(tau_c_j: float, tau_p_j: float, m_j: int) -> float:
    """t_j(c) = tau_c_j + tau_p_j * m_j(c)."""
    return tau_c_j + tau_p_j * m_j


def compute_amortized_service_time(tau_c_j: float, tau_p_j: float, m_j: int) -> float:
    """Eq. 8: t_tilde_j(c) = t_j(c) / m_j(c).

    Amortized mean service time per block, used to rank servers.
    """
    if m_j <= 0:
        return float("inf")
    return compute_service_time(tau_c_j, tau_p_j, m_j) / m_j


def gbp_cr(
    servers: List[ServerParams],
    L: int,
    s_m: float,
    s_c: float,
    c: int,
    lam: float,
    rho_bar: float,
) -> BlockPlacement:
    """Algorithm 1: Greedy Block Placement with Cache Reservation.

    Args:
        servers: List of server parameters.
        L: Total number of blocks (e.g. 32 for LLaMA-2-7B).
        s_m: Block size in GB.
        s_c: Cache size per block per job in GB.
        c: Cache reservation parameter (>= 1).
        lam: Arrival rate (requests/s).
        rho_bar: Target maximum system load in (0, 1).

    Returns:
        BlockPlacement with server assignments and formed chains.

    Raises:
        ValueError: If no servers provided or invalid parameters.
        InfeasiblePlacementError: If servers cannot cover all L blocks.
    """
    if not servers:
        raise ValueError("No servers provided")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if c < 1:
        raise ValueError(f"c must be >= 1, got {c}")

    # Step 1: Compute m_j(c) and t_tilde_j(c) for each server
    server_info = []
    total_block_capacity = 0
    for s in servers:
        m_j = compute_max_blocks(s.memory_gb, s_m, s_c, c, L)
        total_block_capacity += m_j
        if m_j > 0:
            t_tilde = compute_amortized_service_time(s.tau_c, s.tau_p, m_j)
            t_j = compute_service_time(s.tau_c, s.tau_p, m_j)
            server_info.append((s, m_j, t_tilde, t_j))

    if total_block_capacity < L:
        raise InfeasiblePlacementError(
            f"Total block capacity {total_block_capacity} < L={L}. "
            f"Servers cannot cover all blocks with c={c}."
        )

    # Step 2: Sort by increasing amortized service time (fastest first)
    server_info.sort(key=lambda x: x[2])

    # Step 3: Greedily form chains
    required_rate = lam / (rho_bar * c)
    a = 0  # current block index (0-indexed)
    nu = 0.0  # cumulative scaled service rate
    T = 0.0  # current chain service time
    chain_id = 0
    chains: List[ChainInfo] = []
    assignments: dict = {}
    current_chain_servers: List[str] = []

    for s, m_j, t_tilde, t_j in server_info:
        a_j = min(a, L - m_j)  # Eq: a_j <- min(a, L - m_j(c) + 1), 0-indexed
        assignments[s.server_id] = ServerBlockAssignment(
            server_id=s.server_id,
            start_block=a_j,
            num_blocks=m_j,
            chain_index=chain_id,
        )
        current_chain_servers.append(s.server_id)
        T += t_j
        a = min(a + m_j - 1, L - 1) + 1  # next block index

        if a >= L:
            # Chain complete — covers all L blocks
            service_rate = 1.0 / T if T > 0 else float("inf")
            chains.append(ChainInfo(
                chain_id=chain_id,
                server_ids=list(current_chain_servers),
                service_time=T,
                service_rate=service_rate,
            ))
            nu += service_rate

            if nu >= required_rate:
                break  # sufficient total service rate

            # Start new chain
            chain_id += 1
            a = 0
            T = 0.0
            current_chain_servers = []

    total_rate = sum(ch.service_rate for ch in chains)
    return BlockPlacement(
        server_assignments=assignments,
        chains=chains,
        c=c,
        total_service_rate=total_rate,
    )
