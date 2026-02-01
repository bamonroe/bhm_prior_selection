"""
Common functions for RDU risk preference models.

These functions are shared across different RDU model specifications
(power weighting, Prelec weighting, etc.).

The key optimization is precomputing utilities and probability weights
for unique values, then using index lookups in the likelihood calculation.
"""

import jax
import jax.numpy as jnp
import jax.random as random


# =============================================================================
# Utility Functions
# =============================================================================

def crra_ufn(x, r):
    """
    CRRA utility function (vectorized).

    u(x) = x^(1-r) / (1-r)  for r != 1
    u(x) = log(x)           for r = 1

    Args:
        x: Outcomes (scalar or array)
        r: CRRA coefficient (scalar)

    Returns:
        Utility values (same shape as x)
    """
    x = jnp.maximum(x, 1e-10)  # Ensure positive
    r_safe = jnp.clip(r, -10, 10)

    # Standard CRRA
    crra = jnp.power(x, 1 - r_safe) / (1 - r_safe)

    # Log utility (limit as r -> 1)
    log_u = jnp.log(x)

    # Use log when |r - 1| < 0.01
    use_log = jnp.abs(r_safe - 1.0) < 0.01
    return jnp.where(use_log, log_u, crra)


# =============================================================================
# Probability Weighting Functions
# =============================================================================

def power_pwf(p, gamma):
    """
    Power probability weighting function.

    w(p) = p^gamma

    Args:
        p: Probabilities (scalar or array)
        gamma: Curvature parameter (scalar, positive)

    Returns:
        Weighted probabilities (same shape as p)
    """
    gamma = jnp.maximum(gamma, 0.01)
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    return jnp.power(p, gamma)


def prelec_pwf(p, alpha, beta):
    """
    Prelec's 2-parameter probability weighting function.

    w(p) = exp(-beta * (-ln(p))^alpha)

    Args:
        p: Probabilities (scalar or array)
        alpha: Curvature parameter (positive)
        beta: Elevation parameter (positive)

    Returns:
        Weighted probabilities (same shape as p)
    """
    alpha = jnp.maximum(alpha, 0.01)
    beta = jnp.maximum(beta, 0.01)
    p = jnp.clip(p, 1e-10, 1 - 1e-10)

    neg_ln_p = -jnp.log(p)
    return jnp.exp(-beta * jnp.power(neg_ln_p, alpha))


# =============================================================================
# RDU Value Calculation
# =============================================================================

def rdu_value_indexed(
    weighted_probs,  # Precomputed w(p) for unique probs
    utilities,       # Precomputed u(x) for unique outcomes
    idx_high,        # Index of high outcome in utilities
    idx_low,         # Index of low outcome in utilities
    idx_prob,        # Index of probability in weighted_probs
):
    """
    Compute RDU value for a binary lottery using precomputed values.

    For lottery (x_high, x_low) with prob p of x_high:
        RDU = w(p) * u(x_high) + (1 - w(p)) * u(x_low)

    Args:
        weighted_probs: Array of precomputed w(p) values
        utilities: Array of precomputed u(x) values
        idx_high: Index into utilities for high outcome
        idx_low: Index into utilities for low outcome
        idx_prob: Index into weighted_probs for probability

    Returns:
        RDU value (scalar)
    """
    u_high = utilities[idx_high]
    u_low = utilities[idx_low]
    w_p = weighted_probs[idx_prob]

    return w_p * u_high + (1 - w_p) * u_low


# =============================================================================
# Likelihood Functions
# =============================================================================

def risk_likelihood(weighted_probs, utilities, datai, sid, mu):
    """
    Compute log likelihood for a subject's lottery choices.

    This is the core likelihood function used by all RDU models.
    It uses precomputed utilities and weighted probabilities with
    index lookups for efficiency.

    Args:
        weighted_probs: Precomputed w(p) for all unique probabilities
        utilities: Precomputed u(x) for all unique outcomes
        datai: Tuple of integer arrays containing:
            [0]: choices (n_subjects, n_choices) - 1=A, 0=B
            [1]: idx_a_high (n_subjects, n_choices) - index of A's high prize
            [2]: idx_a_low (n_subjects, n_choices) - index of A's low prize
            [3]: idx_b_high (n_subjects, n_choices) - index of B's high prize
            [4]: idx_b_low (n_subjects, n_choices) - index of B's low prize
            [5]: idx_prob (n_subjects, n_choices) - index of probability
        sid: Subject index
        mu: Noise parameter (inverse temperature)

    Returns:
        Log likelihood (scalar)
    """
    choices = datai[0]
    idx_a_high = datai[1]
    idx_a_low = datai[2]
    idx_b_high = datai[3]
    idx_b_low = datai[4]
    idx_prob = datai[5]

    # Get this subject's data
    n_choices = choices.shape[1]
    subj_choices = jax.lax.dynamic_slice(choices, (sid, 0), (1, n_choices))[0]
    subj_idx_a_high = jax.lax.dynamic_slice(idx_a_high, (sid, 0), (1, n_choices))[0]
    subj_idx_a_low = jax.lax.dynamic_slice(idx_a_low, (sid, 0), (1, n_choices))[0]
    subj_idx_b_high = jax.lax.dynamic_slice(idx_b_high, (sid, 0), (1, n_choices))[0]
    subj_idx_b_low = jax.lax.dynamic_slice(idx_b_low, (sid, 0), (1, n_choices))[0]
    subj_idx_prob = jax.lax.dynamic_slice(idx_prob, (sid, 0), (1, n_choices))[0]

    def choice_log_lik(choice, ia_h, ia_l, ib_h, ib_l, ip):
        # RDU values using indexed lookups
        v_a = rdu_value_indexed(weighted_probs, utilities, ia_h, ia_l, ip)
        v_b = rdu_value_indexed(weighted_probs, utilities, ib_h, ib_l, ip)

        # Choice probability (logistic/Fechner noise)
        p_choose_a = jax.nn.sigmoid(mu * (v_a - v_b))
        p_choose_a = jnp.clip(p_choose_a, 1e-10, 1 - 1e-10)

        # Log likelihood
        return choice * jnp.log(p_choose_a) + (1 - choice) * jnp.log(1 - p_choose_a)

    log_liks = jax.vmap(choice_log_lik)(
        subj_choices,
        subj_idx_a_high,
        subj_idx_a_low,
        subj_idx_b_high,
        subj_idx_b_low,
        subj_idx_prob,
    )

    return jnp.sum(log_liks)


# =============================================================================
# Direct Sampler Dispatch (for hyperparameters)
# =============================================================================

def direct_dispatch_4params(key, chain_state, param_indices, data):
    """
    Placeholder direct sampler for hyperparameters.

    In MH-only models, this just returns the current state.
    """
    return chain_state, key
