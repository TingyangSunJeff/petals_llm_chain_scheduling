"""Algorithm 3: Join-the-Fastest-Free-Chain (JFFC).

Online load balancing policy that assigns each arriving job to the fastest
chain with available capacity, or queues it if all chains are full.

Reference: Section 3.2.1 of the paper.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from petals_llm_chain_scheduling.data_models import (
    AllocatedChain,
    InferenceRequest,
    PendingRequest,
)


class JFFCDispatcher:
    """JFFC load balancing dispatcher with central FIFO queue.

    Chains are sorted by descending service rate (fastest first).
    On arrival: assign to fastest chain with Z_k < c_k, else enqueue.
    On completion: dequeue next request if queue non-empty.
    """

    def __init__(self, chains: List[AllocatedChain]):
        if not chains:
            raise ValueError("At least one chain is required")
        # Sort chains by descending service rate
        self._chains = sorted(chains, key=lambda ch: ch.service_rate, reverse=True)
        self._capacities: Dict[int, int] = {ch.chain_id: ch.capacity for ch in self._chains}
        self._rates: Dict[int, float] = {ch.chain_id: ch.service_rate for ch in self._chains}
        self._ongoing: Dict[int, int] = {ch.chain_id: 0 for ch in self._chains}
        self._queue: Deque[PendingRequest] = deque()

    def on_arrival(self, request: InferenceRequest) -> Optional[int]:
        """Dispatch an arriving request via JFFC.

        Returns the chain_id the request is assigned to, or None if queued.
        """
        # Find fastest chain with available capacity (already sorted desc by rate)
        for ch in self._chains:
            cid = ch.chain_id
            if self._ongoing[cid] < self._capacities[cid]:
                self._ongoing[cid] += 1
                return cid

        # All chains full — enqueue
        self._queue.append(PendingRequest(
            request=request,
            enqueue_time=request.submit_time,
        ))
        return None

    def on_completion(self, chain_id: int) -> Optional[PendingRequest]:
        """Handle job completion on a chain.

        Returns the next PendingRequest to assign to this chain, or None.
        """
        if self._queue:
            pending = self._queue.popleft()
            # Z_k stays the same (one job leaves, one enters)
            return pending
        else:
            self._ongoing[chain_id] -= 1
            return None

    @property
    def queue_length(self) -> int:
        """Number of requests waiting in the central queue."""
        return len(self._queue)

    def get_chain_occupancy(self, chain_id: int) -> int:
        """Number of ongoing jobs on a specific chain."""
        return self._ongoing[chain_id]

    @property
    def total_ongoing(self) -> int:
        """Total number of ongoing jobs across all chains."""
        return sum(self._ongoing.values())

    @property
    def chain_ids_sorted(self) -> List[int]:
        """Chain IDs sorted by descending service rate."""
        return [ch.chain_id for ch in self._chains]
