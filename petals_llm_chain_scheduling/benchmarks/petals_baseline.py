"""PETALS baseline benchmark.

Uses original PETALS greedy block placement and heuristic Dijkstra-based
request routing (RemoteSequenceManager). Each request independently finds
its route through the server swarm.

Reference: Section 4.1.3 and 4.2.3 of the paper.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from petals_llm_chain_scheduling.data_models import RequestMetrics

logger = logging.getLogger(__name__)


class PetalsBaselineBenchmark:
    """Benchmark using original PETALS routing.

    Servers use PETALS' default block selection (choose_best_blocks).
    Requests are routed via PETALS' Dijkstra-based sequence manager.
    """

    def __init__(self, model_name: str, initial_peers: List[str]):
        self.model_name = model_name
        self.initial_peers = initial_peers
        self._model = None
        self._tokenizer = None

    def setup(self):
        """Initialize PETALS model and tokenizer."""
        from transformers import AutoTokenizer
        from petals import AutoDistributedModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoDistributedModelForCausalLM.from_pretrained(
            self.model_name,
            initial_peers=self.initial_peers,
        )
        logger.info(f"PETALS baseline model loaded: {self.model_name}")

    def run_inference(
        self,
        request_id: int,
        prompt: str,
        max_new_tokens: int,
    ) -> RequestMetrics:
        """Run a single inference request through PETALS."""
        submit_time = time.monotonic()

        inputs = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
        start_time = time.monotonic()

        outputs = self._model.generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        end_time = time.monotonic()

        return RequestMetrics(
            request_id=request_id,
            submit_time=submit_time,
            start_time=start_time,
            end_time=end_time,
            response_time=end_time - submit_time,
            waiting_time=start_time - submit_time,
            service_time=end_time - start_time,
            chain_id=-1,  # PETALS doesn't use explicit chains
        )
