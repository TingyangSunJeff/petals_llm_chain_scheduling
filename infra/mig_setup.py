"""MIG GPU partitioning setup and teardown.

Creates 2×2g.20gb + 1×3g.40gb per A100 GPU using nvidia-smi MIG commands.
Total: 9 GPU instances from 3 A100s.

Reference: Section 4.2.1 of the paper.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# MIG profile IDs for A100 80GB
# 3g.40gb = profile ID 9 (GI), profile ID 0 (CI)
# 2g.20gb = profile ID 14 (GI), profile ID 0 (CI)
MIG_PROFILES = {
    "3g.40gb": {"gi_profile": 9, "ci_profile": 0, "memory_gb": 40.0},
    "2g.20gb": {"gi_profile": 14, "ci_profile": 0, "memory_gb": 20.0},
}


@dataclass
class MIGInstance:
    """A single MIG GPU instance."""
    gpu_index: int
    gi_id: int
    ci_id: int
    profile: str
    uuid: str
    memory_gb: float


def _run_cmd(cmd: str) -> str:
    """Run a shell command and return stdout."""
    logger.debug(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def enable_mig_mode(gpu_indices: List[int]) -> None:
    """Enable MIG mode on specified GPUs (requires root, GPU reset)."""
    for idx in gpu_indices:
        _run_cmd(f"sudo nvidia-smi -i {idx} -mig 1")
    logger.info(f"MIG mode enabled on GPUs {gpu_indices}")


def create_mig_partitions(gpu_indices: List[int] = None) -> List[MIGInstance]:
    """Create 2×2g.20gb + 1×3g.40gb per GPU.

    Args:
        gpu_indices: GPU indices to partition (default [0, 1, 2]).

    Returns:
        List of MIGInstance objects.
    """
    if gpu_indices is None:
        gpu_indices = [0, 1, 2]

    instances = []
    for gpu_idx in gpu_indices:
        # Create 1× 3g.40gb GPU Instance
        gi_profile = MIG_PROFILES["3g.40gb"]["gi_profile"]
        out = _run_cmd(
            f"sudo nvidia-smi mig -i {gpu_idx} -cgi {gi_profile} -C"
        )
        logger.info(f"GPU {gpu_idx}: Created 3g.40gb instance")

        # Create 2× 2g.20gb GPU Instances
        gi_profile = MIG_PROFILES["2g.20gb"]["gi_profile"]
        for _ in range(2):
            out = _run_cmd(
                f"sudo nvidia-smi mig -i {gpu_idx} -cgi {gi_profile} -C"
            )
        logger.info(f"GPU {gpu_idx}: Created 2× 2g.20gb instances")

    # Query created instances
    instances = _list_mig_instances(gpu_indices)
    logger.info(f"Total MIG instances created: {len(instances)}")
    return instances


def _list_mig_instances(gpu_indices: List[int]) -> List[MIGInstance]:
    """List all MIG instances on specified GPUs."""
    instances = []
    try:
        out = _run_cmd("nvidia-smi -L")
        # Parse output for MIG device UUIDs
        gi_id = 0
        for line in out.split("\n"):
            line = line.strip()
            if "MIG" in line and "UUID" in line:
                # Extract profile and UUID
                uuid_start = line.index("UUID: ") + 6
                uuid_end = line.index(")", uuid_start)
                uuid = line[uuid_start:uuid_end]

                if "3g.40gb" in line:
                    profile = "3g.40gb"
                elif "2g.20gb" in line:
                    profile = "2g.20gb"
                else:
                    continue

                instances.append(MIGInstance(
                    gpu_index=gi_id // 3,
                    gi_id=gi_id,
                    ci_id=0,
                    profile=profile,
                    uuid=uuid,
                    memory_gb=MIG_PROFILES[profile]["memory_gb"],
                ))
                gi_id += 1
    except Exception as e:
        logger.warning(f"Could not list MIG instances: {e}")
    return instances


def destroy_mig_partitions(gpu_indices: List[int] = None) -> None:
    """Destroy all MIG instances and restore GPUs.

    Args:
        gpu_indices: GPU indices to clean up (default [0, 1, 2]).
    """
    if gpu_indices is None:
        gpu_indices = [0, 1, 2]

    for gpu_idx in gpu_indices:
        try:
            _run_cmd(f"sudo nvidia-smi mig -i {gpu_idx} -dci")
            _run_cmd(f"sudo nvidia-smi mig -i {gpu_idx} -dgi")
        except RuntimeError as e:
            logger.warning(f"GPU {gpu_idx} cleanup: {e}")

    for gpu_idx in gpu_indices:
        try:
            _run_cmd(f"sudo nvidia-smi -i {gpu_idx} -mig 0")
        except RuntimeError as e:
            logger.warning(f"GPU {gpu_idx} MIG disable: {e}")

    logger.info(f"MIG partitions destroyed on GPUs {gpu_indices}")
