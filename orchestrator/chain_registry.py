"""Chain registry: maintains state of all server chains and their availability."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from petals_llm_chain_scheduling.data_models import AllocatedChain, CacheAllocation


class ChainRegistry:
    """Thread-safe registry of server chains with service rates and capacities."""

    def __init__(self, allocation: CacheAllocation):
        self._lock = threading.Lock()
        self._chains: Dict[int, AllocatedChain] = {
            ch.chain_id: ch for ch in allocation.chains
        }
        self._available: Dict[int, bool] = {
            ch.chain_id: True for ch in allocation.chains
        }

    @property
    def chains(self) -> List[AllocatedChain]:
        with self._lock:
            return [ch for ch in self._chains.values() if self._available[ch.chain_id]]

    @property
    def all_chains(self) -> List[AllocatedChain]:
        with self._lock:
            return list(self._chains.values())

    def get_chain(self, chain_id: int) -> Optional[AllocatedChain]:
        with self._lock:
            return self._chains.get(chain_id)

    def mark_unavailable(self, chain_id: int) -> None:
        with self._lock:
            self._available[chain_id] = False

    def mark_available(self, chain_id: int) -> None:
        with self._lock:
            self._available[chain_id] = True

    def is_available(self, chain_id: int) -> bool:
        with self._lock:
            return self._available.get(chain_id, False)

    def update_allocation(self, allocation: CacheAllocation) -> None:
        """Replace all chains with a new allocation (after recomposition)."""
        with self._lock:
            self._chains = {ch.chain_id: ch for ch in allocation.chains}
            self._available = {ch.chain_id: True for ch in allocation.chains}
