"""Shared fixtures for petals_llm_chain_scheduling tests."""

import pytest

from petals_llm_chain_scheduling.data_models import ModelParams, ServerParams


@pytest.fixture
def llama2_7b_params():
    """LLaMA-2-7B model parameters as used in the paper."""
    return ModelParams(
        num_blocks=32,
        block_size_gb=0.21,  # ~210 MB per block for LLaMA-2-7B
        cache_size_gb=0.065,  # ~65 MB KV cache per block per job (max seq 2048)
        model_name="meta-llama/Llama-2-7b-hf",
    )


@pytest.fixture
def nine_mig_servers():
    """9 MIG server configs matching the paper's experimental setup."""
    servers = []
    # 3x 3g.40gb (high-performance)
    for i in range(3):
        servers.append(ServerParams(
            server_id=f"3g_{i}",
            memory_gb=40.0,
            tau_c=0.1,
            tau_p=0.05,
            mig_profile="3g.40gb",
        ))
    # 6x 2g.20gb (low-performance)
    for i in range(6):
        servers.append(ServerParams(
            server_id=f"2g_{i}",
            memory_gb=20.0,
            tau_c=0.1,
            tau_p=0.08,
            mig_profile="2g.20gb",
        ))
    return servers
