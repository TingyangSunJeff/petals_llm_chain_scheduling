"""BPRR benchmark: Block Placement and Request Routing (Sun'25).

Two-time-scale algorithm that performs block placement and dynamic request
routing without explicit chain composition or cache pre-allocation.
Requests are routed dynamically based on residual memory at each server.

Reference: Sun'25 Performance, cited as BPRR in Section 4.1.3 of the paper.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional

from petals_llm_chain_scheduling.data_models import RequestMetrics, ServerParams

logger = logging.getLogger(__name__)


class BPRRBlockPlacement:
    """BPRR block placement using coverage-greedy approach.

    Places blocks to maximize minimum throughput across all blocks,
    similar to PETALS' approach but with a different optimization target.
    """

    def __init__(self, servers: List[ServerParams], L: int, s_m: float, s_c: float):
        self.servers = servers
        self.L = L
        self.s_m = s_m
        self.s_c = s_c

    def compute_placement(self) -> Dict[str, tuple]:
        """Compute block placement using coverage-greedy heuristic.

        Returns dict: server_id -> (start_block, end_block).
        """
        assignments = {}
        # Sort servers by memory (largest first) for greedy coverage
        sorted_servers = sorted(self.servers, key=lambda s: s.memory_gb, reverse=True)
        current_block = 0

        for s in sorted_servers:
            if current_block >= self.L:
                break
            max_blocks = min(int(s.memory_gb / self.s_m), self.L - current_block)
            if max_blocks <= 0:
                continue
            assignments[s.server_id] = (current_block, current_block + max_blocks)
            current_block += max_blocks

        return assignments


class BPRRRouter:
    """BPRR dynamic request routing with residual memory tracking.

    Routes requests by finding a feasible path through servers that have
    sufficient residual memory, using weighted shortest-path routing
    (WS-RR from Sun'25).
    """

    def __init__(
        self,
        placement: Dict[str, tuple],
        servers: List[ServerParams],
        s_c: float,
    ):
        self.placement = placement
        self.server_map = {s.server_id: s for s in servers}
        self.s_c = s_c
        # Track residual memory per server (in cache slots)
        self._residual: Dict[str, int] = {}
        for sid, (start, end) in placement.items():
            s = self.server_map[sid]
            num_blocks = end - start
            self._residual[sid] = int((s.memory_gb - 0.21 * num_blocks) / s_c)

    def route_request(self, L: int) -> Optional[List[str]]:
        """Find a feasible route through servers covering all L blocks.

        Returns list of server_ids in order, or None if no route available.
        """
        # Simple greedy: traverse servers in block order
        route = []
        current_block = 0
        for sid, (start, end) in sorted(self.placement.items(), key=lambda x: x[1][0]):
            if start > current_block:
                return None  # gap in coverage
            if self._residual.get(sid, 0) <= 0:
                continue  # no memory left
            route.append(sid)
            current_block = end
            if current_block >= L:
                break
        return route if current_block >= L else None

    def allocate(self, route: List[str]) -> None:
        """Allocate one cache slot per block at each server in the route."""
        for sid in route:
            start, end = self.placement[sid]
            self._residual[sid] -= (end - start)

    def release(self, route: List[str]) -> None:
        """Release cache slots when a request completes."""
        for sid in route:
            start, end = self.placement[sid]
            self._residual[sid] += (end - start)
