"""Per-request timing metrics and Table 1 formatting.

Collects response time, waiting time, and service time for each request,
computes summary statistics, and formats results matching the paper's Table 1.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from petals_llm_chain_scheduling.data_models import (
    MetricsSummary,
    RequestMetrics,
    StatsSummary,
)


def compute_stats(values: List[float], include_minmax: bool = False) -> StatsSummary:
    """Compute summary statistics for a list of timing values."""
    arr = np.array(values)
    return StatsSummary(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min_val=float(np.min(arr)) if include_minmax else None,
        max_val=float(np.max(arr)) if include_minmax else None,
    )


def compute_summary(metrics: List[RequestMetrics]) -> MetricsSummary:
    """Compute Table 1 statistics from per-request metrics."""
    response_times = [m.response_time for m in metrics]
    waiting_times = [m.waiting_time for m in metrics]
    service_times = [m.service_time for m in metrics]

    rt_stats = compute_stats(response_times)
    wt_stats = compute_stats(waiting_times)
    wt_stats.max_val = float(np.max(waiting_times))
    st_stats = compute_stats(service_times, include_minmax=True)

    return MetricsSummary(
        response_time=rt_stats,
        waiting_time=wt_stats,
        service_time=st_stats,
    )


def format_table1(summaries: Dict[str, MetricsSummary]) -> str:
    """Format comparison table matching paper's Table 1.

    Args:
        summaries: Dict mapping method name to MetricsSummary.

    Returns:
        Formatted string table.
    """
    methods = list(summaries.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:>15}" for m in methods)
    sep = "-" * len(header)

    lines = [header, sep]
    lines.append("Response Time (seconds)")

    for label, attr in [("  Mean", "mean"), ("  Median", "median"),
                        ("  P95", "p95"), ("  P99", "p99")]:
        row = f"{label:<30}"
        for m in methods:
            val = getattr(summaries[m].response_time, attr)
            row += f"{val:>15.1f}" if val is not None else f"{'N/A':>15}"
        lines.append(row)

    lines.append("")
    lines.append("Waiting Time (seconds)")
    for label, attr in [("  Mean", "mean"), ("  Median", "median"),
                        ("  P95", "p95"), ("  Max", "max_val")]:
        row = f"{label:<30}"
        for m in methods:
            val = getattr(summaries[m].waiting_time, attr)
            row += f"{val:>15.1f}" if val is not None else f"{'N/A':>15}"
        lines.append(row)

    lines.append("")
    lines.append("Service Time (seconds)")
    for label, attr in [("  Mean", "mean"), ("  Median", "median"),
                        ("  P95", "p95")]:
        row = f"{label:<30}"
        for m in methods:
            val = getattr(summaries[m].service_time, attr)
            row += f"{val:>15.1f}" if val is not None else f"{'N/A':>15}"
        lines.append(row)

    # Min / Max service time
    row = f"{'  Min / Max':<30}"
    for m in methods:
        st = summaries[m].service_time
        mn = f"{st.min_val:.1f}" if st.min_val is not None else "N/A"
        mx = f"{st.max_val:.1f}" if st.max_val is not None else "N/A"
        row += f"{mn + ' / ' + mx:>15}"
    lines.append(row)

    # Improvement vs first method
    if len(methods) >= 2:
        base = methods[0]
        lines.append("")
        lines.append(f"Improvement vs. {base}")
        base_rt = summaries[base].response_time.mean
        base_wt = summaries[base].waiting_time.mean
        base_p95 = summaries[base].response_time.p95

        for label, base_val, getter in [
            ("  Mean Response Time", base_rt, lambda s: s.response_time.mean),
            ("  Mean Waiting Time", base_wt, lambda s: s.waiting_time.mean),
            ("  P95 Response Time", base_p95, lambda s: s.response_time.p95),
        ]:
            row = f"{label:<30}"
            for m in methods:
                if m == base:
                    row += f"{'--':>15}"
                else:
                    val = getter(summaries[m])
                    pct = (1 - val / base_val) * 100 if base_val > 0 else 0
                    row += f"{pct:>14.1f}%"
            lines.append(row)

    return "\n".join(lines)
