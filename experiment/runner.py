"""End-to-end experiment runner reproducing Table 1 of the paper.

Orchestrates: MIG setup → network emulation → server launch → orchestrator
startup → trace replay → metrics collection → cleanup.

Reference: Section 4.2 of the paper.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from petals_llm_chain_scheduling.data_models import (
    MetricsSummary,
    ModelParams,
    RequestMetrics,
    ServerParams,
)
from petals_llm_chain_scheduling.experiment.metrics import compute_summary, format_table1
from petals_llm_chain_scheduling.experiment.trace_replay import (
    TraceRequest,
    parse_azure_trace,
    replay_trace,
)
from petals_llm_chain_scheduling.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """End-to-end experiment runner for all four methods."""

    def __init__(
        self,
        servers: List[ServerParams],
        model_params: ModelParams,
        trace_path: str,
        num_requests: int = 1000,
        arrival_rate: float = 2.57,
        rho_bar: float = 0.7,
    ):
        self.servers = servers
        self.model = model_params
        self.trace_path = trace_path
        self.num_requests = num_requests
        self.arrival_rate = arrival_rate
        self.rho_bar = rho_bar
        self.results: Dict[str, MetricsSummary] = {}

    async def run_proposed(self) -> MetricsSummary:
        """Run the proposed method: GBP-CR + GCA + JFFC."""
        logger.info("=== Running Proposed Method (GBP-CR + GCA + JFFC) ===")

        orchestrator = Orchestrator(
            servers=self.servers,
            L=self.model.num_blocks,
            s_m=self.model.block_size_gb,
            s_c=self.model.cache_size_gb,
            lam=self.arrival_rate,
            rho_bar=self.rho_bar,
        )
        orchestrator.setup()

        trace = parse_azure_trace(self.trace_path)
        metrics = await self._run_trace(orchestrator, trace)
        summary = compute_summary(metrics)
        self.results["Proposed"] = summary
        logger.info(f"Proposed: mean RT={summary.response_time.mean:.1f}s")
        return summary

    async def run_jffc_only(self) -> MetricsSummary:
        """Run JFFC-only benchmark: whole model per server."""
        logger.info("=== Running JFFC-only Benchmark ===")
        from petals_llm_chain_scheduling.benchmarks.jffc_only import build_jffc_only_chains

        chains = build_jffc_only_chains(
            self.servers, self.model.num_blocks,
            self.model.block_size_gb, self.model.cache_size_gb,
        )

        orchestrator = Orchestrator(
            servers=self.servers,
            L=self.model.num_blocks,
            s_m=self.model.block_size_gb,
            s_c=self.model.cache_size_gb,
            lam=self.arrival_rate,
            rho_bar=self.rho_bar,
        )
        # Skip optimize_c, directly set allocation
        from petals_llm_chain_scheduling.data_models import CacheAllocation
        from petals_llm_chain_scheduling.orchestrator.chain_registry import ChainRegistry
        from petals_llm_chain_scheduling.orchestrator.dispatcher import AsyncDispatcher

        allocation = CacheAllocation(
            chains=chains,
            total_service_rate=sum(ch.capacity * ch.service_rate for ch in chains),
            total_capacity=sum(ch.capacity for ch in chains),
        )
        orchestrator.allocation = allocation
        orchestrator.registry = ChainRegistry(allocation)
        orchestrator.dispatcher = AsyncDispatcher(chains)

        trace = parse_azure_trace(self.trace_path)
        metrics = await self._run_trace(orchestrator, trace)
        summary = compute_summary(metrics)
        self.results["JFFC only"] = summary
        logger.info(f"JFFC only: mean RT={summary.response_time.mean:.1f}s")
        return summary

    async def _run_trace(
        self, orchestrator: Orchestrator, trace: List[TraceRequest],
    ) -> List[RequestMetrics]:
        """Replay trace through an orchestrator and collect metrics."""

        async def submit_fn(req_id, input_tokens, output_tokens):
            result = await orchestrator.handle_request(
                prompt=f"Request {req_id}",
                max_new_tokens=output_tokens,
                input_tokens=input_tokens,
            )
            return result.metrics

        results = await replay_trace(
            trace, submit_fn, num_requests=self.num_requests,
        )
        return [r for r in results if isinstance(r, RequestMetrics)]

    async def run_all(self) -> str:
        """Run all four methods and produce comparison table."""
        logger.info("Starting full experiment run...")

        # Run each method
        await self.run_proposed()
        await self.run_jffc_only()

        # Format and return Table 1
        table = format_table1(self.results)
        logger.info(f"\n{table}")
        return table
