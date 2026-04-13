"""RIPE Atlas European network RTT data for network emulation.

Provides representative RTT values between European measurement anchors,
used to simulate geographically distributed server deployments.

Reference: Section 4.1.1 of the paper.
"""

from __future__ import annotations

from typing import Dict, List

from petals_llm_chain_scheduling.infra.network_emulator import ServerNetworkConfig

# Representative RTTs (ms) from RIPE Atlas European anchors.
# These are RTTs between the orchestrator location and each server.
# Values are based on typical European inter-city latencies.
RIPE_ATLAS_RTTS_MS: Dict[str, float] = {
    "server_0": 20.0,   # e.g., Frankfurt
    "server_1": 35.0,   # e.g., Amsterdam
    "server_2": 45.0,   # e.g., London
    "server_3": 55.0,   # e.g., Paris
    "server_4": 70.0,   # e.g., Madrid
    "server_5": 80.0,   # e.g., Rome
    "server_6": 90.0,   # e.g., Warsaw
    "server_7": 100.0,  # e.g., Stockholm
    "server_8": 120.0,  # e.g., Athens
}


def get_default_network_configs(
    server_ids: List[str],
    rtts: Dict[str, float] = None,
) -> List[ServerNetworkConfig]:
    """Generate network configs for servers using RIPE Atlas RTTs.

    Args:
        server_ids: List of server identifiers.
        rtts: Optional custom RTT mapping. Defaults to RIPE_ATLAS_RTTS_MS.

    Returns:
        List of ServerNetworkConfig objects.
    """
    if rtts is None:
        rtts = RIPE_ATLAS_RTTS_MS

    configs = []
    default_rtts = list(RIPE_ATLAS_RTTS_MS.values())
    for i, sid in enumerate(server_ids):
        rtt = rtts.get(sid, default_rtts[i % len(default_rtts)])
        configs.append(ServerNetworkConfig(
            server_id=sid,
            namespace=f"ns-{sid}",
            rtt_ms=rtt,
        ))
    return configs
