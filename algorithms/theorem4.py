"""Theorem 4: Closed-form bounds on steady-state mean response time under JFFC.

Computes upper/lower bounds on mean occupancy via birth-death process analysis,
then converts to response time bounds via Little's law.

Reference: Section 3.2.2 of the paper, Theorem 4.
"""

from __future__ import annotations

import math
from typing import List

from petals_llm_chain_scheduling.data_models import ResponseTimeBounds, UnstableSystemError


def compute_death_rate_upper_bound(
    n: int, rates: List[float], capacities: List[int],
) -> float:
    """Eq. 14: nu_bar_n — max departure rate when jobs occupy fastest chains.

    nu_bar_n = sum_l mu_l * min(c_l, (n - sum_{l'<l} c_{l'})_+)
    """
    result = 0.0
    cumulative_cap = 0
    for l, (mu_l, c_l) in enumerate(zip(rates, capacities)):
        jobs_available = max(0, n - cumulative_cap)
        result += mu_l * min(c_l, jobs_available)
        cumulative_cap += c_l
    return result


def compute_death_rate_lower_bound(
    n: int, rates: List[float], capacities: List[int],
) -> float:
    """Eq. 15: nu_underbar_n — min departure rate when jobs occupy slowest chains.

    nu_underbar_n = sum_l mu_l * min(c_l, (n - sum_{l'>l} c_{l'})_+)
    """
    K = len(rates)
    result = 0.0
    # sum of capacities from l+1 to K
    suffix_cap = [0] * (K + 1)
    for l in range(K - 1, -1, -1):
        suffix_cap[l] = suffix_cap[l + 1] + capacities[l]

    for l in range(K):
        cap_after = suffix_cap[l + 1]
        jobs_available = max(0, n - cap_after)
        result += rates[l] * min(capacities[l], jobs_available)
    return result


def compute_stationary_bounds(
    lam: float,
    rates: List[float],
    capacities: List[int],
) -> ResponseTimeBounds:
    """Theorem 4: closed-form bounds on mean response time.

    Args:
        lam: Arrival rate (lambda).
        rates: Service rates mu_k sorted in descending order.
        capacities: Capacities c_k corresponding to rates.

    Returns:
        ResponseTimeBounds with lower/upper bounds on mean response time.

    Raises:
        UnstableSystemError: If lambda >= sum(c_k * mu_k).
    """
    if not rates or not capacities:
        raise ValueError("rates and capacities must be non-empty")
    if len(rates) != len(capacities):
        raise ValueError("rates and capacities must have same length")

    nu = sum(c * mu for c, mu in zip(capacities, rates))  # total service rate
    C = sum(capacities)  # total capacity
    rho = lam / nu

    if lam >= nu:
        raise UnstableSystemError(
            f"System unstable: lambda={lam} >= nu={nu}. "
            f"Arrival rate must be less than total service rate."
        )

    # Compute phi bounds using log-space to avoid overflow
    # Lower bound on occupancy uses upper death rates (nu_bar)
    # Upper bound on occupancy uses lower death rates (nu_underbar)

    def _compute_phi_and_occupancy(death_rate_fn):
        """Compute stationary distribution and mean occupancy for a birth-death process."""
        # phi_n = phi_0 * prod_{i=1}^{n} (lambda / death_rate(i))  for n <= C
        # For n > C, death_rate = nu (constant), so geometric tail

        # Use log-space: log_ratio[n] = sum_{i=1}^{n} log(lambda / death_rate(i))
        log_ratios = []  # log(phi_n / phi_0) for n = 1..C
        for n in range(1, C + 1):
            dr = death_rate_fn(n)
            if dr <= 0:
                log_ratios.append(float("inf"))
            else:
                prev = log_ratios[-1] if log_ratios else 0.0
                log_ratios.append(prev + math.log(lam / dr))

        # Normalization: phi_0 * (1 + sum_{n=1}^{C-1} exp(log_ratio[n-1])
        #                         + exp(log_ratio[C-1]) * nu / (nu - lambda))
        # = 1

        # Find max log_ratio for numerical stability
        all_log_vals = log_ratios[:C - 1] if C > 1 else []
        if log_ratios:
            tail_log = log_ratios[C - 1] + math.log(nu / (nu - lam))
            all_log_vals.append(tail_log)

        if not all_log_vals:
            # C = 1, only tail term
            tail_log = log_ratios[0] + math.log(nu / (nu - lam))
            max_log = tail_log
        else:
            max_log = max(all_log_vals)

        # Compute sum in log-space
        norm_sum = math.exp(-max_log)  # contribution of phi_0 = exp(0 - max_log)
        for n in range(1, C):
            norm_sum += math.exp(log_ratios[n - 1] - max_log)
        # Tail: exp(log_ratios[C-1]) * nu / (nu - lambda)
        tail_log = log_ratios[C - 1] + math.log(nu / (nu - lam))
        norm_sum += math.exp(tail_log - max_log)

        log_phi_0 = -max_log - math.log(norm_sum)

        # Mean occupancy: sum_{n=0}^{C-1} n * phi_n + phi_C * (rho/(1-rho)^2 + C/(1-rho))
        mean_occ = 0.0
        for n in range(1, C):
            log_phi_n = log_phi_0 + log_ratios[n - 1]
            mean_occ += n * math.exp(log_phi_n)

        log_phi_C = log_phi_0 + log_ratios[C - 1]
        phi_C = math.exp(log_phi_C)
        mean_occ += phi_C * (rho / (1 - rho) ** 2 + C / (1 - rho))

        return mean_occ

    # Lower bound on occupancy → lower bound on response time
    # Uses upper death rates (jobs on fastest chains)
    mean_occ_lower = _compute_phi_and_occupancy(
        lambda n: compute_death_rate_upper_bound(n, rates, capacities)
    )

    # Upper bound on occupancy → upper bound on response time
    # Uses lower death rates (jobs on slowest chains)
    mean_occ_upper = _compute_phi_and_occupancy(
        lambda n: compute_death_rate_lower_bound(n, rates, capacities)
    )

    # Little's law: T_bar = E[sum Z_l] / lambda
    rt_lower = mean_occ_lower / lam
    rt_upper = mean_occ_upper / lam

    return ResponseTimeBounds(
        lower_bound=rt_lower,
        upper_bound=rt_upper,
        mean_occupancy_lower=mean_occ_lower,
        mean_occupancy_upper=mean_occ_upper,
        c=0,  # filled by caller
        num_chains=len(rates),
    )
