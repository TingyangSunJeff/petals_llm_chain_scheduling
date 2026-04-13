"""Custom Hypothesis strategies for petals_llm_chain_scheduling property tests."""

import string

from hypothesis import strategies as st

from petals_llm_chain_scheduling.data_models import ServerParams


@st.composite
def server_params(draw, min_memory=1.0, max_memory=80.0):
    """Generate a random ServerParams instance."""
    return ServerParams(
        server_id=draw(st.text(min_size=1, max_size=8, alphabet=string.ascii_lowercase)),
        memory_gb=draw(st.floats(min_value=min_memory, max_value=max_memory)),
        tau_c=draw(st.floats(min_value=0.001, max_value=1.0)),
        tau_p=draw(st.floats(min_value=0.001, max_value=0.5)),
        mig_profile=draw(st.sampled_from(["3g.40gb", "2g.20gb"])),
    )


@st.composite
def feasible_server_set(draw, L=32, s_m=0.21, s_c=0.065, c=1):
    """Generate a list of servers guaranteed to cover all L blocks.

    Ensures sum of m_j(c) >= L so GBP-CR can form at least one chain.
    """
    total_blocks = 0
    servers = []
    idx = 0
    while total_blocks < L:
        mem = draw(st.floats(min_value=s_m + s_c * c + 0.01, max_value=80.0))
        m_j = min(int(mem / (s_m + s_c * c)), L)
        if m_j < 1:
            continue
        total_blocks += m_j
        servers.append(ServerParams(
            server_id=f"s{idx}",
            memory_gb=mem,
            tau_c=draw(st.floats(min_value=0.001, max_value=1.0)),
            tau_p=draw(st.floats(min_value=0.001, max_value=0.5)),
            mig_profile=draw(st.sampled_from(["3g.40gb", "2g.20gb"])),
        ))
        idx += 1
    return servers
