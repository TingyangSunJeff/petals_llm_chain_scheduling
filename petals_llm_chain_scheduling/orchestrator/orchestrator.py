"""Main orchestrator: offline chain composition + online JFFC dispatch.

Manages the full lifecycle: profile servers, optimize c, compute placement
and allocation, assign blocks to servers, and dispatch inference requests.

Reference: Section 2.2 and Section 3 of the paper.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from petals_llm_chain_scheduling.algorithms.param_optimizer import optimize_c
from petals_llm_chain_scheduling.data_models import (
    AllocatedChain,
    BlockPlacement,
    CacheAllocation,
    InferenceRequest,
    InferenceResult,
    RequestMetrics,
    ServerParams,
)
from petals_llm_chain_scheduling.orchestrator.chain_registry import ChainRegistry
from petals_llm_chain_scheduling.orchestrator.dispatcher import AsyncDispatcher

logger = logging.getLogger(__name__)


class Orchestrator:
    """Centralized orchestrator for server chain composition and load balancing."""

    def __init__(
        self,
        servers: List[ServerParams],
        L: int,
        s_m: float,
        s_c: float,
        lam: float,
        rho_bar: float = 0.7,
        c_method: str = "theorem4_lower",
    ):
        self.servers = servers
        self.L = L
        self.s_m = s_m
        self.s_c = s_c
        self.lam = lam
        self.rho_bar = rho_bar
        self.c_method = c_method

        self.c_star: Optional[int] = None
        self.placement: Optional[BlockPlacement] = None
        self.allocation: Optional[CacheAllocation] = None
        self.registry: Optional[ChainRegistry] = None
        self.dispatcher: Optional[AsyncDispatcher] = None

        self.collected_metrics: List[RequestMetrics] = []
        self._request_counter = 0
        self._inference_fn = None  # Set externally for actual GPU inference

    def setup(self):
        """Run offline phase: optimize c, compute placement + allocation."""
        logger.info("Starting offline server chain composition...")
        logger.info(f"  Servers: {len(self.servers)}, L={self.L}, lambda={self.lam}")

        self.c_star, self.placement, self.allocation = optimize_c(
            servers=self.servers,
            L=self.L,
            s_m=self.s_m,
            s_c=self.s_c,
            lam=self.lam,
            rho_bar=self.rho_bar,
            method=self.c_method,
        )

        logger.info(f"  Optimal c* = {self.c_star}")
        logger.info(f"  Chains: {len(self.allocation.chains)}")
        logger.info(f"  Total capacity: {self.allocation.total_capacity}")
        logger.info(f"  Total service rate: {self.allocation.total_service_rate:.4f}")

        self.registry = ChainRegistry(self.allocation)
        self.dispatcher = AsyncDispatcher(self.registry.chains)

    def set_inference_fn(self, fn):
        """Set the function used to run actual inference on a chain.

        fn(chain: AllocatedChain, prompt: str, max_new_tokens: int, input_tokens: int)
            -> (generated_text: str, service_time: float)
        """
        self._inference_fn = fn

    async def handle_request(
        self,
        prompt: str,
        max_new_tokens: int,
        input_tokens: int = 0,
    ) -> InferenceResult:
        """Accept a request, dispatch via JFFC, run inference, return result."""
        submit_time = time.monotonic()
        self._request_counter += 1
        request = InferenceRequest(
            request_id=self._request_counter,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_tokens=input_tokens,
            submit_time=submit_time,
        )

        # Dispatch via JFFC
        chain_id = await self.dispatcher.dispatch(request)

        if chain_id is None:
            # Queued — wait for a chain to become available
            chain_id = await self._wait_for_chain(request)

        start_time = time.monotonic()

        # Run inference on the assigned chain
        chain = self.registry.get_chain(chain_id)
        if self._inference_fn is not None and chain is not None:
            generated_text, _ = await asyncio.get_event_loop().run_in_executor(
                None, self._inference_fn, chain, prompt, max_new_tokens, input_tokens,
            )
        else:
            # Simulate inference with service time based on chain's rate
            if chain is not None:
                await asyncio.sleep(chain.service_time)
            generated_text = f"[simulated output for request {request.request_id}]"

        end_time = time.monotonic()

        metrics = RequestMetrics(
            request_id=request.request_id,
            submit_time=submit_time,
            start_time=start_time,
            end_time=end_time,
            response_time=end_time - submit_time,
            waiting_time=start_time - submit_time,
            service_time=end_time - start_time,
            chain_id=chain_id,
        )
        self.collected_metrics.append(metrics)

        # Notify dispatcher of completion
        pending = await self.dispatcher.on_completion(chain_id, metrics)
        if pending is not None:
            # Schedule the next queued request on this chain
            asyncio.create_task(self._process_pending(pending, chain_id))

        return InferenceResult(
            request_id=request.request_id,
            generated_text=generated_text,
            metrics=metrics,
        )

    async def _wait_for_chain(self, request: InferenceRequest) -> int:
        """Wait until a chain becomes available for a queued request."""
        while True:
            await asyncio.sleep(0.01)
            for ch in self.registry.chains:
                if self.dispatcher.get_chain_occupancy(ch.chain_id) < ch.capacity:
                    return ch.chain_id

    async def _process_pending(self, pending, chain_id: int):
        """Process a pending request that was dequeued after a completion."""
        await self.handle_request(
            prompt=pending.request.prompt,
            max_new_tokens=pending.request.max_new_tokens,
            input_tokens=pending.request.input_tokens,
        )

    def get_block_assignments(self) -> Dict[str, tuple]:
        """Get block assignments for each server (for launching servers)."""
        if self.placement is None:
            return {}
        return {
            sid: (assign.start_block, assign.start_block + assign.num_blocks)
            for sid, assign in self.placement.server_assignments.items()
        }
