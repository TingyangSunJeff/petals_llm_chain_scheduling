"""Async dispatcher wrapping JFFC for the orchestrator.

Bridges between HTTP request handling and the JFFC algorithm state,
providing async interfaces for request dispatch and completion handling.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from petals_llm_chain_scheduling.algorithms.jffc import JFFCDispatcher
from petals_llm_chain_scheduling.data_models import (
    AllocatedChain,
    InferenceRequest,
    PendingRequest,
    RequestMetrics,
)


class AsyncDispatcher:
    """Async wrapper around JFFCDispatcher for the orchestrator."""

    def __init__(self, chains: List[AllocatedChain]):
        self._jffc = JFFCDispatcher(chains)
        self._lock = asyncio.Lock()
        self._completion_events: Dict[int, asyncio.Event] = {}
        self._results: Dict[int, RequestMetrics] = {}

    async def dispatch(self, request: InferenceRequest) -> Optional[int]:
        """Dispatch a request via JFFC. Returns chain_id or None (queued)."""
        async with self._lock:
            return self._jffc.on_arrival(request)

    async def on_completion(
        self, chain_id: int, metrics: RequestMetrics,
    ) -> Optional[PendingRequest]:
        """Handle job completion. Returns next pending request or None."""
        async with self._lock:
            self._results[metrics.request_id] = metrics
            evt = self._completion_events.pop(metrics.request_id, None)
            if evt:
                evt.set()
            return self._jffc.on_completion(chain_id)

    @property
    def queue_length(self) -> int:
        return self._jffc.queue_length

    def get_chain_occupancy(self, chain_id: int) -> int:
        return self._jffc.get_chain_occupancy(chain_id)

    @property
    def total_ongoing(self) -> int:
        return self._jffc.total_ongoing
