"""Algorithm 2: Greedy Cache Allocation (GCA).

Iteratively finds the fastest chain via Dijkstra shortest path, allocates
maximum capacity from residual memory, and updates the routing topology.
Complexity: O(J^4).

Reference: Section 3.1.4 of the paper.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from petals_llm_chain_scheduling.data_models import (
    AllocatedChain,
    BlockPlacement,
    CacheAllocation,
    RoutingEdge,
    ServerParams,
)

DUMMY_START = "dummy_start"
DUMMY_END = "dummy_end"


def compute_residual_cache_slots(
    M_j: float, s_m: float, m_j: int, s_c: float,
) -> int:
    """Eq. 3: M_tilde_j = floor((M_j - s_m * m_j) / s_c) cache slots."""
    residual = M_j - s_m * m_j
    if residual < 0 or s_c <= 0:
        return 0
    return int(math.floor(residual / s_c))


def build_routing_topology(
    placement: BlockPlacement,
    servers: Dict[str, ServerParams],
    L: int,
) -> Dict[str, List[RoutingEdge]]:
    """Build directed graph G = (J+, E) with dummy start/end nodes.

    An edge (i, j) exists iff a_j <= a_i + m_i <= a_j + m_j - 1.
    Edge cost = tau_c_j + tau_p_j * m_ij where m_ij = a_j + m_j - a_i - m_i.

    The dummy_start node has a_0 = 0, m_0 = 1 (hosts dummy block 0, 1-indexed
    in paper but we use 0-indexed: effectively covers block -1 to 0 boundary).
    The dummy_end node has a_{J+1} = L, m_{J+1} = 1.

    Returns adjacency list: node -> list of RoutingEdge.
    """
    adj: Dict[str, List[RoutingEdge]] = {DUMMY_START: [], DUMMY_END: []}
    for sid in placement.server_assignments:
        adj[sid] = []

    # Collect all nodes with their (a, m) values
    # dummy_start: a=0, m=0 (so a+m=0, meaning next server must start at block 0)
    # We model dummy_start as having a=0, m=0 so a_i + m_i = 0
    # dummy_end: a=L, m=0 so a_j = L, a_j + m_j - 1 = L - 1
    # Edge (i, j) exists iff a_j <= a_i + m_i <= a_j + m_j - 1

    nodes = {}
    # dummy_start: after processing, the "next block" is 0
    nodes[DUMMY_START] = (0, 0)  # a=0, m=0 → a+m = 0
    nodes[DUMMY_END] = (L, 1)    # a=L, m=1 → covers "block L" (dummy)

    for sid, assign in placement.server_assignments.items():
        nodes[sid] = (assign.start_block, assign.num_blocks)

    # Build edges
    for i_id, (a_i, m_i) in nodes.items():
        if i_id == DUMMY_END:
            continue  # no outgoing edges from dummy_end
        next_block = a_i + m_i  # the block index after server i's range

        for j_id, (a_j, m_j) in nodes.items():
            if j_id == DUMMY_START or j_id == i_id:
                continue
            # Edge condition: a_j <= a_i + m_i <= a_j + m_j - 1
            if a_j <= next_block <= a_j + m_j - 1:
                m_ij = a_j + m_j - next_block  # blocks processed at j

                if j_id == DUMMY_END:
                    cost = 0.0  # dummy end has zero cost
                else:
                    s = servers[j_id]
                    cost = s.tau_c + s.tau_p * m_ij

                edge = RoutingEdge(
                    from_server=i_id,
                    to_server=j_id,
                    blocks_at_to=m_ij,
                    cost=cost,
                )
                adj[i_id].append(edge)

    return adj


def _dijkstra_shortest_path(
    adj: Dict[str, List[RoutingEdge]],
    source: str,
    target: str,
) -> Optional[List[RoutingEdge]]:
    """Find shortest path from source to target. Returns list of edges or None."""
    dist: Dict[str, float] = {source: 0.0}
    prev_edge: Dict[str, Optional[RoutingEdge]] = {source: None}
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == target:
            break
        for edge in adj.get(u, []):
            v = edge.to_server
            new_dist = d + edge.cost
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                prev_edge[v] = edge
                heapq.heappush(heap, (new_dist, v))

    if target not in prev_edge:
        return None

    # Reconstruct path
    path_edges = []
    node = target
    while prev_edge[node] is not None:
        path_edges.append(prev_edge[node])
        node = prev_edge[node].from_server
    path_edges.reverse()
    return path_edges


def gca(
    placement: BlockPlacement,
    servers: List[ServerParams],
    L: int,
    s_m: float,
    s_c: float,
) -> CacheAllocation:
    """Algorithm 2: Greedy Cache Allocation.

    Iteratively finds fastest chain via Dijkstra, allocates max capacity,
    updates residual memory. Returns CacheAllocation.

    Args:
        placement: Block placement from GBP-CR.
        servers: List of server parameters.
        L: Total number of blocks.
        s_m: Block size in GB.
        s_c: Cache size per block per job in GB.

    Returns:
        CacheAllocation with chains and their capacities.
    """
    server_map = {s.server_id: s for s in servers}

    # Compute initial residual memory (in cache slots) for each server
    residual: Dict[str, int] = {}
    for sid, assign in placement.server_assignments.items():
        s = server_map[sid]
        residual[sid] = compute_residual_cache_slots(
            s.memory_gb, s_m, assign.num_blocks, s_c,
        )

    # Build initial routing topology
    adj = build_routing_topology(placement, server_map, L)

    # Filter edges where residual memory is insufficient
    def filter_edges(adj_list: Dict[str, List[RoutingEdge]]) -> Dict[str, List[RoutingEdge]]:
        filtered = {}
        for node, edges in adj_list.items():
            filtered[node] = [
                e for e in edges
                if e.to_server == DUMMY_END or residual.get(e.to_server, 0) >= e.blocks_at_to
            ]
        return filtered

    adj = filter_edges(adj)
    allocated_chains: List[AllocatedChain] = []
    chain_id = 0

    while True:
        # Find fastest (shortest cost) path from dummy_start to dummy_end
        path_edges = _dijkstra_shortest_path(adj, DUMMY_START, DUMMY_END)
        if path_edges is None:
            break  # No more feasible chains

        # Compute capacity: min over real servers of floor(residual / m_ij)
        capacity = float("inf")
        for edge in path_edges:
            if edge.to_server == DUMMY_END:
                continue
            m_ij = edge.blocks_at_to
            if m_ij > 0:
                cap = residual.get(edge.to_server, 0) // m_ij
                capacity = min(capacity, cap)

        capacity = int(capacity)
        if capacity <= 0:
            # Remove bottleneck edge and try again
            for edge in path_edges:
                if edge.to_server != DUMMY_END:
                    m_ij = edge.blocks_at_to
                    if m_ij > 0 and residual.get(edge.to_server, 0) // m_ij == 0:
                        adj[edge.from_server] = [
                            e for e in adj[edge.from_server]
                            if not (e.to_server == edge.to_server)
                        ]
            continue

        # Extract server sequence (excluding dummies)
        server_seq = []
        real_edges = []
        for edge in path_edges:
            if edge.to_server != DUMMY_END and edge.from_server != DUMMY_START:
                if edge.from_server not in server_seq:
                    server_seq.append(edge.from_server)
            if edge.to_server != DUMMY_END:
                server_seq.append(edge.to_server)
                real_edges.append(edge)
            else:
                real_edges.append(edge)
        # Deduplicate server_seq preserving order
        seen = set()
        unique_seq = []
        for s in server_seq:
            if s not in seen and s != DUMMY_START:
                seen.add(s)
                unique_seq.append(s)

        service_time = sum(e.cost for e in path_edges)
        service_rate = 1.0 / service_time if service_time > 0 else float("inf")

        allocated_chains.append(AllocatedChain(
            chain_id=chain_id,
            server_sequence=unique_seq,
            edges=path_edges,
            capacity=capacity,
            service_time=service_time,
            service_rate=service_rate,
        ))

        # Update residual memory
        for edge in path_edges:
            if edge.to_server == DUMMY_END:
                continue
            residual[edge.to_server] -= edge.blocks_at_to * capacity

        # Remove edges where residual is now insufficient
        for edge in path_edges:
            if edge.to_server == DUMMY_END:
                continue
            m_ij = edge.blocks_at_to
            if residual.get(edge.to_server, 0) < m_ij:
                adj[edge.from_server] = [
                    e for e in adj.get(edge.from_server, [])
                    if e.to_server != edge.to_server
                ]

        chain_id += 1

    total_rate = sum(ch.capacity * ch.service_rate for ch in allocated_chains)
    total_cap = sum(ch.capacity for ch in allocated_chains)

    return CacheAllocation(
        chains=allocated_chains,
        total_service_rate=total_rate,
        total_capacity=total_cap,
    )
