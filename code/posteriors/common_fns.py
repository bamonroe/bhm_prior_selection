"""
Common functions for RDU risk preference models.

The key optimization is precomputing utilities for unique outcome values,
then using index lookups in the likelihood calculation. Probability
weighting is computed per-choice from the 3-outcome probability arrays.
"""

import jax
import jax.numpy as jnp


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
# RDU Value Calculation (3-outcome)
# =============================================================================

def rdu_value_3outcome(utilities, phi, eta, idx_1, idx_2, idx_3, p1, p2, p3):
    """
    Compute RDU value for a 3-outcome lottery using precomputed utilities.

    Outcomes x1 <= x2 <= x3 with probs p1, p2, p3 (sum=1), sorted ascending.

    Cumulative probabilities (at least as good as):
        C3 = p3                     (best outcome)
        C2 = p2 + p3               (middle or better)
        C1 = 1                     (always)

    Decision weights:
        pi3 = w(C3)               (best)
        pi2 = w(C2) - w(C3)       (middle)
        pi1 = 1 - w(C2)           (worst)

    RDU = pi1*u(x1) + pi2*u(x2) + pi3*u(x3)

    When a slot has prob=0, its cumulative probability contribution vanishes
    and the decision weight naturally reduces, correctly handling binary and
    degenerate lotteries.

    Args:
        utilities: Array of precomputed u(x) for unique outcomes
        phi: Prelec curvature (natural scale, positive)
        eta: Prelec elevation (natural scale, positive)
        idx_1: Index into utilities for worst outcome (x1)
        idx_2: Index into utilities for middle outcome (x2)
        idx_3: Index into utilities for best outcome (x3)
        p1: Probability of worst outcome
        p2: Probability of middle outcome
        p3: Probability of best outcome

    Returns:
        RDU value (scalar)
    """
    u1 = utilities[idx_1]
    u2 = utilities[idx_2]
    u3 = utilities[idx_3]

    # Cumulative probabilities
    C3 = p3
    C2 = p2 + p3

    # Prelec-weighted cumulative probabilities
    w_C3 = prelec_pwf(C3, phi, eta)
    w_C2 = prelec_pwf(C2, phi, eta)

    # Decision weights
    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    return pi1 * u1 + pi2 * u2 + pi3 * u3


# =============================================================================
# Likelihood Functions
# =============================================================================

def risk_likelihood(utilities, phi, eta, datai, dataf, sid, lam):
    """
    Compute log likelihood for a subject's lottery choices (3-outcome format).

    Implements the Contextual Utility model (Wilcox, 2011):
        Delta = (U(L) - U(R)) / (lam * (u_max - u_min))
        P(L) = 1 / (1 + exp(-Delta))

    where lam is a noise scaling parameter (larger = more noise).

    Args:
        utilities: Precomputed u(x) for all unique outcomes
        phi: Prelec curvature (natural scale, positive)
        eta: Prelec elevation (natural scale, positive)
        datai: Tuple of integer arrays:
            [0]: choices (n_subjects, n_choices) - 1=Left, 0=Right
            [1-3]: idx_L_1, idx_L_2, idx_L_3
            [4-6]: idx_R_1, idx_R_2, idx_R_3
        dataf: Tuple of float arrays:
            [0]: unique_outcomes (not used here, but present)
            [1-3]: prob_L_1, prob_L_2, prob_L_3
            [4-6]: prob_R_1, prob_R_2, prob_R_3
        sid: Subject index
        lam: Fechner noise parameter (> 0, in denominator)

    Returns:
        Log likelihood (scalar)
    """
    choices = datai[0]
    idx_L_1 = datai[1]
    idx_L_2 = datai[2]
    idx_L_3 = datai[3]
    idx_R_1 = datai[4]
    idx_R_2 = datai[5]
    idx_R_3 = datai[6]

    prob_L_1 = dataf[1]
    prob_L_2 = dataf[2]
    prob_L_3 = dataf[3]
    prob_R_1 = dataf[4]
    prob_R_2 = dataf[5]
    prob_R_3 = dataf[6]

    # Get this subject's data
    n_choices = choices.shape[1]
    subj_choices = jax.lax.dynamic_slice(choices, (sid, 0), (1, n_choices))[0]
    subj_idx_L_1 = jax.lax.dynamic_slice(idx_L_1, (sid, 0), (1, n_choices))[0]
    subj_idx_L_2 = jax.lax.dynamic_slice(idx_L_2, (sid, 0), (1, n_choices))[0]
    subj_idx_L_3 = jax.lax.dynamic_slice(idx_L_3, (sid, 0), (1, n_choices))[0]
    subj_idx_R_1 = jax.lax.dynamic_slice(idx_R_1, (sid, 0), (1, n_choices))[0]
    subj_idx_R_2 = jax.lax.dynamic_slice(idx_R_2, (sid, 0), (1, n_choices))[0]
    subj_idx_R_3 = jax.lax.dynamic_slice(idx_R_3, (sid, 0), (1, n_choices))[0]

    subj_pL1 = jax.lax.dynamic_slice(prob_L_1, (sid, 0), (1, n_choices))[0]
    subj_pL2 = jax.lax.dynamic_slice(prob_L_2, (sid, 0), (1, n_choices))[0]
    subj_pL3 = jax.lax.dynamic_slice(prob_L_3, (sid, 0), (1, n_choices))[0]
    subj_pR1 = jax.lax.dynamic_slice(prob_R_1, (sid, 0), (1, n_choices))[0]
    subj_pR2 = jax.lax.dynamic_slice(prob_R_2, (sid, 0), (1, n_choices))[0]
    subj_pR3 = jax.lax.dynamic_slice(prob_R_3, (sid, 0), (1, n_choices))[0]

    def choice_log_lik(choice, iL1, iL2, iL3, iR1, iR2, iR3,
                       pL1, pL2, pL3, pR1, pR2, pR3):
        # RDU values using 3-outcome formula
        v_L = rdu_value_3outcome(utilities, phi, eta, iL1, iL2, iL3, pL1, pL2, pL3)
        v_R = rdu_value_3outcome(utilities, phi, eta, iR1, iR2, iR3, pR1, pR2, pR3)

        # Contextual utility: normalize by range of outcomes (Wilcox, 2011)
        u_L1 = utilities[iL1]
        u_L2 = utilities[iL2]
        u_L3 = utilities[iL3]
        u_R1 = utilities[iR1]
        u_R2 = utilities[iR2]
        u_R3 = utilities[iR3]
        u_max = jnp.maximum(
            jnp.maximum(u_L1, jnp.maximum(u_L2, u_L3)),
            jnp.maximum(u_R1, jnp.maximum(u_R2, u_R3)),
        )
        u_min = jnp.minimum(
            jnp.minimum(u_L1, jnp.minimum(u_L2, u_L3)),
            jnp.minimum(u_R1, jnp.minimum(u_R2, u_R3)),
        )
        u_range = jnp.maximum(u_max - u_min, 1e-10)

        # Fechner model: Delta = (U(L) - U(R)) / (lam * u_range)
        delta = (v_L - v_R) / (lam * u_range)

        # Choice probability
        p_choose_L = jax.nn.sigmoid(delta)
        p_choose_L = jnp.clip(p_choose_L, 1e-10, 1 - 1e-10)

        # Log likelihood
        return choice * jnp.log(p_choose_L) + (1 - choice) * jnp.log(1 - p_choose_L)

    log_liks = jax.vmap(choice_log_lik)(
        subj_choices,
        subj_idx_L_1, subj_idx_L_2, subj_idx_L_3,
        subj_idx_R_1, subj_idx_R_2, subj_idx_R_3,
        subj_pL1, subj_pL2, subj_pL3,
        subj_pR1, subj_pR2, subj_pR3,
    )

    return jnp.sum(log_liks)


# =============================================================================
# CRRA Inverse Utility (JAX)
# =============================================================================

def crra_inverse(v, r):
    """
    Inverse of CRRA utility function (JAX).

    Given u(x) = x^(1-r) / (1-r), the inverse is:
        x = ((1-r)*v)^(1/(1-r))

    For r close to 1 (log utility), u(x) = log(x) so x = exp(v).

    Args:
        v: Utility value (scalar)
        r: CRRA coefficient (scalar)

    Returns:
        Certainty equivalent in prize units
    """
    r_safe = jnp.clip(r, -10, 10)
    inner = jnp.maximum((1 - r_safe) * v, 1e-30)
    crra_inv = jnp.power(inner, 1.0 / (1 - r_safe))
    log_inv = jnp.exp(v)
    use_log = jnp.abs(r_safe - 1.0) < 0.01
    return jnp.where(use_log, log_inv, crra_inv)


# =============================================================================
# CE Computation for Reference Lotteries
# =============================================================================

def rdu_ce_3outcome(r, phi, eta, prizes, probs):
    """
    Compute certainty equivalent for a 3-outcome lottery with arbitrary probs.

    Uses RDU with Prelec PWF. Prizes must be sorted ascending.

    Args:
        r: CRRA coefficient (scalar)
        phi: Prelec curvature (natural scale, positive)
        eta: Prelec elevation (natural scale, positive)
        prizes: Array of 3 prizes [x1, x2, x3] sorted ascending
        probs: Array of 3 probabilities [p1, p2, p3] summing to 1

    Returns:
        CE in prize units (scalar)
    """
    x1, x2, x3 = prizes[0], prizes[1], prizes[2]
    p1, p2, p3 = probs[0], probs[1], probs[2]

    u1 = crra_ufn(x1, r)
    u2 = crra_ufn(x2, r)
    u3 = crra_ufn(x3, r)

    # Cumulative probs (at least as good as)
    C3 = p3
    C2 = p2 + p3

    # Decision weights via Prelec PWF
    w_C3 = prelec_pwf(C3, phi, eta)
    w_C2 = prelec_pwf(C2, phi, eta)

    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    rdu_val = pi1 * u1 + pi2 * u2 + pi3 * u3
    return crra_inverse(rdu_val, r)


def ce_mapping_3lotteries(r, phi_raw, eta_raw, ref_lotteries):
    """
    Compute (CE1, CE2, CE3) for three reference lotteries.

    This is the forward map g(theta) from parameter space to CE space.

    Args:
        r: CRRA coefficient (scalar)
        phi_raw: log(phi), Prelec curvature on log scale (scalar)
        eta_raw: log(eta), Prelec elevation on log scale (scalar)
        ref_lotteries: Flat array of shape (18,) encoding 3 lotteries:
            [prizes1(3), probs1(3), prizes2(3), probs2(3), prizes3(3), probs3(3)]

    Returns:
        Array of shape (3,) with [CE1, CE2, CE3]
    """
    phi = jnp.exp(phi_raw)
    eta = jnp.exp(eta_raw)

    ce1 = rdu_ce_3outcome(r, phi, eta, ref_lotteries[0:3], ref_lotteries[3:6])
    ce2 = rdu_ce_3outcome(r, phi, eta, ref_lotteries[6:9], ref_lotteries[9:12])
    ce3 = rdu_ce_3outcome(r, phi, eta, ref_lotteries[12:15], ref_lotteries[15:18])

    return jnp.array([ce1, ce2, ce3])


def ce_jacobian_logabsdet(r, phi_raw, eta_raw, ref_lotteries):
    """
    Compute log|det J| of the CE mapping g: (r, phi_raw, eta_raw) -> (CE1, CE2, CE3).

    Uses jax.jacfwd to compute the 3x3 Jacobian, then jnp.linalg.slogdet.

    Args:
        r: CRRA coefficient (scalar)
        phi_raw: log(phi) (scalar)
        eta_raw: log(eta) (scalar)
        ref_lotteries: Flat array of shape (18,)

    Returns:
        log|det J| (scalar). Returns -inf if det is zero or negative.
    """
    def f(theta):
        return ce_mapping_3lotteries(theta[0], theta[1], theta[2], ref_lotteries)

    theta = jnp.array([r, phi_raw, eta_raw])
    J = jax.jacfwd(f)(theta)  # (3, 3)
    sign, logabsdet = jnp.linalg.slogdet(J)
    # We need log|det J|; sign can be negative (orientation-reversing map is fine)
    # Only reject if det is exactly zero (sign == 0)
    return jnp.where(sign != 0, logabsdet, -jnp.inf)
