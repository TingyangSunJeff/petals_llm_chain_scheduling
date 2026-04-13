"""Network latency emulation using Linux namespaces and traffic control.

Simulates geographically distributed network latencies between servers
using RIPE Atlas European RTT measurements.

Reference: Section 4.1.1 and 4.2.1 of the paper.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

SERIALIZATION_OVERHEAD_MS = 18.0  # Paper's overhead constant


@dataclass
class ServerNetworkConfig:
    """Network configuration for a single emulated server."""
    server_id: str
    namespace: str          # Linux network namespace name
    rtt_ms: float           # RTT to orchestrator in ms
    overhead_ms: float = SERIALIZATION_OVERHEAD_MS

    @property
    def one_way_delay_ms(self) -> float:
        """One-way delay = RTT/2 + overhead (Eq. from paper)."""
        return self.rtt_ms / 2.0 + self.overhead_ms


def _run(cmd: str) -> None:
    """Run a shell command, raise on failure."""
    logger.debug(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")


def setup_network_emulation(server_configs: List[ServerNetworkConfig]) -> None:
    """Create Linux namespaces and tc netem rules for each server.

    Requires root privileges. Creates a veth pair per server connecting
    the default namespace to the server's namespace, then applies netem
    delay rules.
    """
    for cfg in server_configs:
        ns = cfg.namespace
        delay_ms = cfg.one_way_delay_ms

        # Create namespace
        _run(f"sudo ip netns add {ns}")

        # Create veth pair
        veth_host = f"veth-{cfg.server_id}-h"
        veth_ns = f"veth-{cfg.server_id}-n"
        _run(f"sudo ip link add {veth_host} type veth peer name {veth_ns}")

        # Move one end into namespace
        _run(f"sudo ip link set {veth_ns} netns {ns}")

        # Bring up interfaces
        _run(f"sudo ip link set {veth_host} up")
        _run(f"sudo ip netns exec {ns} ip link set {veth_ns} up")
        _run(f"sudo ip netns exec {ns} ip link set lo up")

        # Apply netem delay on the host side (affects traffic to/from namespace)
        _run(f"sudo tc qdisc add dev {veth_host} root netem delay {delay_ms:.1f}ms")

        logger.info(f"Server {cfg.server_id}: namespace={ns}, delay={delay_ms:.1f}ms")


def teardown_network_emulation(server_configs: List[ServerNetworkConfig]) -> None:
    """Remove all network namespaces and tc rules."""
    for cfg in server_configs:
        ns = cfg.namespace
        veth_host = f"veth-{cfg.server_id}-h"

        try:
            _run(f"sudo tc qdisc del dev {veth_host} root")
        except RuntimeError:
            pass
        try:
            _run(f"sudo ip link del {veth_host}")
        except RuntimeError:
            pass
        try:
            _run(f"sudo ip netns del {ns}")
        except RuntimeError:
            pass

    logger.info(f"Network emulation torn down for {len(server_configs)} servers")
