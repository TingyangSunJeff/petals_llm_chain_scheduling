"""Azure LLM inference trace parser and replayer.

Parses the Azure trace dataset (Patel'24 Splitwise) and replays requests
at the original timestamps with configurable rate scaling.

Reference: Section 4.2.1 of the paper.
"""

from __future__ import annotations

import asyncio
import csv
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TraceRequest:
    """A single request from the Azure trace."""
    timestamp: float        # seconds from trace start
    input_tokens: int
    output_tokens: int


def parse_azure_trace(trace_path: str) -> List[TraceRequest]:
    """Parse Azure LLM inference trace CSV.

    Expected columns: TIMESTAMP, ContextTokens, GeneratedTokens
    (or similar; we handle common column name variants).
    """
    requests = []
    with open(trace_path, "r") as f:
        reader = csv.DictReader(f)
        first_ts = None
        for row in reader:
            # Handle common column name variants
            ts_raw = row.get("TIMESTAMP") or row.get("timestamp") or row.get("Timestamp")
            inp = row.get("ContextTokens") or row.get("input_tokens") or row.get("PromptTokens")
            out = row.get("GeneratedTokens") or row.get("output_tokens") or row.get("CompletionTokens")

            if ts_raw is None or inp is None or out is None:
                continue

            ts = float(ts_raw)
            if first_ts is None:
                first_ts = ts

            requests.append(TraceRequest(
                timestamp=ts - first_ts,
                input_tokens=int(float(inp)),
                output_tokens=int(float(out)),
            ))

    return requests


def scale_trace(
    requests: List[TraceRequest], rate_scale: float,
) -> List[TraceRequest]:
    """Scale inter-arrival times by dividing by rate_scale.

    Preserves relative ordering and token lengths.
    """
    if rate_scale <= 0:
        raise ValueError(f"rate_scale must be positive, got {rate_scale}")
    return [
        TraceRequest(
            timestamp=r.timestamp / rate_scale,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
        )
        for r in requests
    ]


async def replay_trace(
    requests: List[TraceRequest],
    submit_fn,
    num_requests: int = 1000,
    rate_scale: float = 1.0,
) -> List:
    """Submit requests at trace timestamps, collect per-request metrics.

    Args:
        requests: Parsed trace requests.
        submit_fn: Async callable(request_id, input_tokens, output_tokens) -> metrics.
        num_requests: Number of requests to replay (default 1000).
        rate_scale: Scale factor for arrival rate (>1 = faster).

    Returns:
        List of results from submit_fn.
    """
    scaled = scale_trace(requests[:num_requests], rate_scale)
    results = []
    tasks = []
    start_time = time.monotonic()

    for i, req in enumerate(scaled):
        # Wait until the scheduled time
        target_time = start_time + req.timestamp
        now = time.monotonic()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        task = asyncio.create_task(
            submit_fn(i, req.input_tokens, req.output_tokens)
        )
        tasks.append(task)

    # Wait for all requests to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
