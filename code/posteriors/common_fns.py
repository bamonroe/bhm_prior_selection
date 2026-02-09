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

def risk_likelihood(weighted_probs, utilities, datai, sid, lam):
    """
    Compute log likelihood for a subject's lottery choices.

    This is the core likelihood function used by all RDU models.
    It uses precomputed utilities and weighted probabilities with
    index lookups for efficiency.

    Implements the Contextual Utility model (Wilcox, 2011) as in eq:likelihood:
        Δ = (U(A) - U(B)) / (λ * (u(x̄) - u(x̲)))
        P(A) = 1 / (1 + exp(-Δ))

    where λ is a noise scaling parameter (larger λ = more noise).

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
        lam: Fechner noise parameter (λ > 0, in denominator)

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

        # Contextual utility: normalize by range of outcomes (Wilcox, 2011)
        # u(x̄) - u(x̲) = max utility - min utility across both lotteries
        u_a_h = utilities[ia_h]
        u_a_l = utilities[ia_l]
        u_b_h = utilities[ib_h]
        u_b_l = utilities[ib_l]
        u_max = jnp.maximum(jnp.maximum(u_a_h, u_a_l), jnp.maximum(u_b_h, u_b_l))
        u_min = jnp.minimum(jnp.minimum(u_a_h, u_a_l), jnp.minimum(u_b_h, u_b_l))
        u_range = jnp.maximum(u_max - u_min, 1e-10)  # Avoid division by zero

        # Fechner model: Δ = (U(A) - U(B)) / (λ * u_range)
        # λ is noise SCALE (in denominator): larger λ = more noise
        delta = (v_a - v_b) / (lam * u_range)

        # Choice probability
        p_choose_a = jax.nn.sigmoid(delta)
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
# Direct Sampler Factories (for hierarchical hyperparameters)
# =============================================================================

def make_direct_sampler_mean(nparams):
    """
    Factory for hierarchical mean direct samplers.

    Uses conjugate Normal-Normal update for population means.

    Args:
        nparams: Number of parameters per subject

    Returns:
        A direct sampler function for population means
    """
    def sampler(key, chain_state, hyper_idx, data):
        nids = data["static"][0]
        first_hyper = nids * nparams

        hyper_offset = hyper_idx - first_hyper
        param_type = hyper_offset // 2  # Which parameter (0, 1, 2, ...)

        # Hyper-Prior: Normal(mu_0, tau_0)
        mu_0 = 0.0
        tau_0_sq = 100.0

        ln_sigma = chain_state[hyper_idx + 1]
        sigma_sq = jnp.exp(ln_sigma)**2

        # Pre-extract all subject parameter arrays using STATIC start indices
        all_subjects = [chain_state[i:first_hyper:nparams] for i in range(nparams)]

        # Use jax.lax.switch to select the appropriate array based on param_type
        subjects = jax.lax.switch(
            param_type,
            [lambda arr=arr: arr for arr in all_subjects]
        )

        n = nids
        sum_theta = jnp.sum(subjects)

        # Conjugate posterior
        post_prec = (1.0 / tau_0_sq) + (n / sigma_sq)
        post_var = 1.0 / post_prec
        post_mean = post_var * ((mu_0 / tau_0_sq) + (sum_theta / sigma_sq))

        new_key, sample_key = random.split(key)
        sample = random.normal(sample_key) * jnp.sqrt(post_var) + post_mean
        return chain_state.at[hyper_idx].set(sample), new_key

    return sampler


def make_direct_sampler_variance(nparams):
    """
    Factory for hierarchical variance direct samplers.

    Uses conjugate Normal-InverseGamma update for population variances.

    Args:
        nparams: Number of parameters per subject

    Returns:
        A direct sampler function for population log-standard-deviations
    """
    def sampler(key, chain_state, hyper_idx, data):
        nids = data["static"][0]
        first_hyper = nids * nparams

        hyper_offset = hyper_idx - first_hyper
        param_type = hyper_offset // 2  # Which parameter (0, 1, 2, ...)

        pop_mean = chain_state[hyper_idx - 1]

        # Pre-extract all subject parameter arrays using STATIC start indices
        all_subjects = [chain_state[i:first_hyper:nparams] for i in range(nparams)]

        # Use jax.lax.switch to select the appropriate array based on param_type
        subjects = jax.lax.switch(
            param_type,
            [lambda arr=arr: arr for arr in all_subjects]
        )

        n = nids

        # Hyper-Prior: InverseGamma(alpha_0, beta_0)
        alpha_0 = 2.0
        beta_0 = 2.0

        alpha_n = alpha_0 + n / 2.0
        sse = jnp.sum((subjects - pop_mean)**2)
        beta_n = beta_0 + 0.5 * sse

        new_key, sample_key = random.split(key)
        gamma_sample = random.gamma(sample_key, alpha_n)
        gamma_sample = jnp.maximum(gamma_sample, 1e-10)

        sigma_sq_new = beta_n / gamma_sample
        ln_sigma_new = 0.5 * jnp.log(sigma_sq_new)

        return chain_state.at[hyper_idx].set(ln_sigma_new), new_key

    return sampler


def make_direct_dispatch(nparams):
    """
    Factory for direct sampler dispatch functions.

    Routes to mean sampler (even offsets) or variance sampler (odd offsets).

    Args:
        nparams: Number of parameters per subject

    Returns:
        A dispatch function that routes to mean or variance sampler
    """
    from functools import partial

    mean_sampler = make_direct_sampler_mean(nparams)
    variance_sampler = make_direct_sampler_variance(nparams)

    def dispatch(key, chain_state, param_indices, data):
        hyper_idx = param_indices[0]
        nids = data["static"][0]
        first_hyper = nids * nparams
        offset = hyper_idx - first_hyper
        is_mean = (offset % 2 == 0)

        true_branch = partial(mean_sampler, data=data)
        false_branch = partial(variance_sampler, data=data)
        return jax.lax.cond(is_mean, true_branch, false_branch, key, chain_state, hyper_idx)

    return dispatch


# Pre-create dispatch functions for common parameter counts
direct_dispatch_2params = make_direct_dispatch(2)
direct_dispatch_3params = make_direct_dispatch(3)
direct_dispatch_4params = make_direct_dispatch(4)
