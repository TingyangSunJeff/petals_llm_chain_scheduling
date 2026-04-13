"""Hardware profiling for per-block computation time and communication time.

Measures tau_p_j and tau_c_j on actual MIG GPU instances so that algorithm
parameters match the real hardware.

Reference: Section 4.2.2 of the paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from petals_llm_chain_scheduling.data_models import ServerParams


@dataclass
class ProfilingResult:
    """Profiling measurements for a single server."""
    server_id: str
    tau_p: float            # per-block computation time (seconds)
    tau_c: float            # communication time (seconds)
    memory_gb: float        # GPU memory in GB
    mig_profile: str
    num_blocks_tested: int
    computation_times: List[float]  # per-block times for linearity check


def profile_computation_time(
    server_id: str,
    run_inference_fn,
    num_blocks_list: List[int],
    input_tokens: int = 2000,
    output_tokens: int = 20,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> List[float]:
    """Measure per-block computation time by running inference with varying blocks.

    Args:
        server_id: Server identifier.
        run_inference_fn: Callable(num_blocks, input_tokens, output_tokens) -> elapsed_seconds.
        num_blocks_list: List of block counts to test (e.g. [1, 5, 10, 15]).
        input_tokens: Number of input tokens for profiling.
        output_tokens: Number of output tokens for profiling.
        num_warmup: Warmup runs before measurement.
        num_runs: Number of measurement runs to average.

    Returns:
        List of total computation times corresponding to num_blocks_list.
    """
    results = []
    for nb in num_blocks_list:
        # Warmup
        for _ in range(num_warmup):
            run_inference_fn(nb, input_tokens, output_tokens)
        # Measure
        times = []
        for _ in range(num_runs):
            elapsed = run_inference_fn(nb, input_tokens, output_tokens)
            times.append(elapsed)
        results.append(sum(times) / len(times))
    return results


def estimate_tau_p(computation_times: List[float], num_blocks_list: List[int]) -> float:
    """Estimate per-block computation time via linear regression.

    Fits computation_time = tau_c_local + tau_p * num_blocks and returns tau_p.
    Validates linear scaling.
    """
    import numpy as np
    x = np.array(num_blocks_list, dtype=float)
    y = np.array(computation_times, dtype=float)
    # Linear fit: y = a + b*x, tau_p = b
    A = np.vstack([np.ones_like(x), x]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    coeffs = result[0]
    tau_p = coeffs[1]
    return float(tau_p)


def profile_communication_time(
    server_id: str,
    ping_fn,
    num_runs: int = 10,
) -> float:
    """Measure communication time (RTT/2 + overhead) to a server.

    Args:
        server_id: Server identifier.
        ping_fn: Callable() -> rtt_seconds (round-trip time).
        num_runs: Number of pings to average.

    Returns:
        Estimated tau_c in seconds.
    """
    rtts = []
    for _ in range(num_runs):
        rtt = ping_fn()
        rtts.append(rtt)
    avg_rtt = sum(rtts) / len(rtts)
    # tau_c = RTT (full round-trip for relay communication as in PETALS)
    return avg_rtt


def build_server_params(
    profiling_results: List[ProfilingResult],
) -> List[ServerParams]:
    """Convert profiling results to ServerParams for algorithm input."""
    return [
        ServerParams(
            server_id=r.server_id,
            memory_gb=r.memory_gb,
            tau_c=r.tau_c,
            tau_p=r.tau_p,
            mig_profile=r.mig_profile,
        )
        for r in profiling_results
    ]
