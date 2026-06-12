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

def crra_ufn(x, r, x_min=10.0, x_max=50.0):
    """
    Normalized CRRA utility function (vectorized).

    v(x) = x^(1-r) / (1-r)
    u(x) = (v(x) - v(x_min)) / (v(x_max) - v(x_min))

    This normalizes utility to [0, 1] over [x_min, x_max] and is
    continuous at r = 1, where it reduces to ln(x/x_min) / ln(x_max/x_min).

    Args:
        x: Outcomes (scalar or array)
        r: CRRA coefficient (scalar)
        x_min: Minimum outcome (default 10.0)
        x_max: Maximum outcome (default 50.0)

    Returns:
        Normalized utility values (same shape as x)
    """
    x = jnp.maximum(x, 1e-10)
    r_safe = jnp.clip(r, -10, 10)
    s = 1 - r_safe

    # Power form: (x^s - xmin^s) / (xmax^s - xmin^s)
    x_s = jnp.power(x, s)
    xmin_s = jnp.power(x_min, s)
    xmax_s = jnp.power(x_max, s)
    power_form = (x_s - xmin_s) / (xmax_s - xmin_s)

    # Log form (limit as s -> 0): ln(x/xmin) / ln(xmax/xmin)
    log_form = jnp.log(x / x_min) / jnp.log(x_max / x_min)

    use_log = jnp.abs(s) < 1e-10
    return jnp.where(use_log, log_form, power_form)


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
    subj_choices = choices[sid]
    subj_idx_L_1 = idx_L_1[sid]
    subj_idx_L_2 = idx_L_2[sid]
    subj_idx_L_3 = idx_L_3[sid]
    subj_idx_R_1 = idx_R_1[sid]
    subj_idx_R_2 = idx_R_2[sid]
    subj_idx_R_3 = idx_R_3[sid]

    subj_pL1 = prob_L_1[sid]
    subj_pL2 = prob_L_2[sid]
    subj_pL3 = prob_L_3[sid]
    subj_pR1 = prob_R_1[sid]
    subj_pR2 = prob_R_2[sid]
    subj_pR3 = prob_R_3[sid]

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

def crra_inverse(u_val, r, x_min=10.0, x_max=50.0):
    """
    Inverse of normalized CRRA utility function (JAX).

    Given u(x) = (x^s - xmin^s) / (xmax^s - xmin^s) with s = 1-r,
    the inverse is CE = (u_val * (xmax^s - xmin^s) + xmin^s)^(1/s).

    At r = 1: CE = x_min * (x_max / x_min)^u_val.

    Args:
        u_val: Normalized utility value in [0, 1] (scalar)
        r: CRRA coefficient (scalar)
        x_min: Minimum outcome (default 10.0)
        x_max: Maximum outcome (default 50.0)

    Returns:
        Certainty equivalent in prize units
    """
    r_safe = jnp.clip(r, -10, 10)
    s = 1 - r_safe

    # Power form: CE = (u * (xmax^s - xmin^s) + xmin^s)^(1/s)
    xmin_s = jnp.power(x_min, s)
    xmax_s = jnp.power(x_max, s)
    ce_s = u_val * (xmax_s - xmin_s) + xmin_s
    power_form = jnp.power(jnp.maximum(ce_s, 1e-30), 1.0 / s)

    # Log form (limit as s -> 0): CE = xmin * (xmax/xmin)^u_val
    log_form = x_min * jnp.power(x_max / x_min, u_val)

    use_log = jnp.abs(s) < 1e-10
    return jnp.where(use_log, log_form, power_form)


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
        ref_lotteries: Flat array of shape (18,) encoding 3 lotteries in
            3-outcome format (binary lotteries padded with zero-prob middle):
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


# =============================================================================
# Choice Probability for Reference Pair
# =============================================================================

def choice_probability_ref(r, phi_raw, eta_raw, lambda_raw, ref_pair):
    """
    Compute P(choose A | theta) for a reference lottery pair.

    Uses Contextual Utility (Wilcox 2011):
        Delta = (RDU_A - RDU_B) / (lambda * u_range)
        P = sigmoid(Delta)

    Args:
        r: CRRA coefficient (scalar)
        phi_raw: log(phi) (scalar)
        eta_raw: log(eta) (scalar)
        lambda_raw: log(lambda) (scalar)
        ref_pair: Flat array of shape (12,) encoding 2 lotteries in 3-outcome
            padded format:
            [A_prizes(3), A_probs(3), B_prizes(3), B_probs(3)]

    Returns:
        P(choose A) in (0, 1) (scalar)
    """
    phi = jnp.exp(phi_raw)
    eta = jnp.exp(eta_raw)
    lam = jnp.exp(lambda_raw)

    # Compute utilities for all outcomes
    u_A = crra_ufn(ref_pair[0:3], r)
    u_B = crra_ufn(ref_pair[6:9], r)

    # RDU values via decision weights (3-outcome formula)
    p_A = ref_pair[3:6]
    p_B = ref_pair[9:12]

    def rdu_val(u, p):
        C3 = p[2]
        C2 = p[1] + p[2]
        w_C3 = prelec_pwf(C3, phi, eta)
        w_C2 = prelec_pwf(C2, phi, eta)
        pi3 = w_C3
        pi2 = w_C2 - w_C3
        pi1 = 1.0 - w_C2
        return pi1 * u[0] + pi2 * u[1] + pi3 * u[2]

    rdu_A = rdu_val(u_A, p_A)
    rdu_B = rdu_val(u_B, p_B)

    # Contextual utility: normalize by utility range
    all_u = jnp.concatenate([u_A, u_B])
    u_range = jnp.maximum(jnp.max(all_u) - jnp.min(all_u), 1e-10)

    delta = (rdu_A - rdu_B) / (lam * u_range)
    return jax.nn.sigmoid(delta)


# =============================================================================
# Full 4D Forward Map and Jacobian
# =============================================================================

def full_forward_map(r, phi_raw, eta_raw, lambda_raw, ref_lotteries, ref_pair):
    """
    Full 4D forward map: (r, phi_raw, eta_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).

    Args:
        r, phi_raw, eta_raw, lambda_raw: Model parameters (scalars)
        ref_lotteries: Flat (18,) array for 3 reference lotteries
        ref_pair: Flat (12,) array for the reference pair

    Returns:
        Array of shape (4,) with [CE1, CE2, CE3, P_choice]
    """
    ces = ce_mapping_3lotteries(r, phi_raw, eta_raw, ref_lotteries)
    p_choice = choice_probability_ref(r, phi_raw, eta_raw, lambda_raw, ref_pair)
    return jnp.concatenate([ces, jnp.array([p_choice])])


def _ce_and_derivs(w, xlo, xhi, s):
    """Compute CE, dCE/dw, and dCE/dr for a binary lottery.

    Handles s=0 (r=1) analytically via the L'Hopital limits.
    """
    use_log = jnp.abs(s) < 1e-10
    s_safe = jnp.where(use_log, 1.0, s)  # avoid division by zero

    # --- Power form (s != 0) ---
    inner = w * jnp.power(xhi, s_safe) + (1 - w) * jnp.power(xlo, s_safe)
    ce_power = jnp.power(jnp.maximum(inner, 1e-30), 1.0 / s_safe)
    dce_dw_power = (jnp.power(xhi, s_safe) - jnp.power(xlo, s_safe)) / (
        s_safe * jnp.power(jnp.maximum(ce_power, 1e-30), s_safe - 1))
    weighted_lnx = (w * jnp.power(xhi, s_safe) * jnp.log(xhi)
                    + (1 - w) * jnp.power(xlo, s_safe) * jnp.log(xlo))
    dce_dr_power = (ce_power / s_safe) * (jnp.log(jnp.maximum(ce_power, 1e-30))
                                           - weighted_lnx / inner)

    # --- Log form (s = 0) ---
    M = w * jnp.log(xhi) + (1 - w) * jnp.log(xlo)
    ce_log = jnp.exp(M)
    dce_dw_log = ce_log * (jnp.log(xhi) - jnp.log(xlo))
    V = w * jnp.log(xhi)**2 + (1 - w) * jnp.log(xlo)**2
    var_lnx = V - M**2
    dce_dr_log = -ce_log * var_lnx / 2

    ce = jnp.where(use_log, ce_log, ce_power)
    dce_dw = jnp.where(use_log, dce_dw_log, dce_dw_power)
    dce_dr = jnp.where(use_log, dce_dr_log, dce_dr_power)

    return ce, dce_dw, dce_dr


def full_jacobian_logabsdet(r, phi_raw, eta_raw, lambda_raw,
                            ref_lotteries, ref_pair):
    """
    Compute log|det J| of the full 4D forward map — fully analytic.

    Decomposes into three components:
      log|det J| = log|det J_W| + log|det J_CE| + log|dP/dlambda_raw|

    Component 1 (Prelec W map): analytic 2x2 determinant.
    Component 2 (CE map in (r,w1,w2) space): analytic with s=0 limits.
    Component 3 (choice probability): analytic.

    Continuous at r=1 (s=0), unlike the JAX autodiff version.

    Args:
        r, phi_raw, eta_raw, lambda_raw: Model parameters (scalars)
        ref_lotteries: Flat (18,) array
        ref_pair: Flat (12,) array

    Returns:
        log|det J| (scalar)
    """
    phi = jnp.exp(phi_raw)
    eta = jnp.exp(eta_raw)
    lam = jnp.exp(lambda_raw)
    s = 1 - r

    # --- Component 1: Prelec W map ---
    p1 = ref_lotteries[5]   # prob on best outcome for L1
    p2 = ref_lotteries[11]  # prob on best outcome for L2
    t1 = -jnp.log(p1)
    t2 = -jnp.log(p2)
    A1 = eta * jnp.power(t1, phi)
    A2 = eta * jnp.power(t2, phi)
    w1 = jnp.exp(-A1)
    w2 = jnp.exp(-A2)
    log_det_W = (-A1 - A2 + 2 * eta_raw
                 + phi * (jnp.log(t1) + jnp.log(t2))
                 + phi_raw + jnp.log(jnp.abs(jnp.log(t1 / t2))))

    # --- Component 2: CE map in (r, w1, w2) space ---
    w3 = w1  # shared probability p3 = p1
    xlo1, xhi1 = ref_lotteries[0], ref_lotteries[2]
    xlo2, xhi2 = ref_lotteries[6], ref_lotteries[8]
    xlo3, xhi3 = ref_lotteries[12], ref_lotteries[14]

    _, dce1_dw1, dce1_dr = _ce_and_derivs(w1, xlo1, xhi1, s)
    _, dce2_dw2, dce2_dr = _ce_and_derivs(w2, xlo2, xhi2, s)
    _, dce3_dw1, dce3_dr = _ce_and_derivs(w3, xlo3, xhi3, s)

    det_CE = dce2_dw2 * (dce1_dr * dce3_dw1 - dce3_dr * dce1_dw1)
    log_det_CE = jnp.log(jnp.abs(det_CE))

    # --- Component 3: dP/dlambda_raw ---
    u_A = crra_ufn(ref_pair[0:3], r)
    u_B = crra_ufn(ref_pair[6:9], r)
    all_u = jnp.concatenate([u_A, u_B])
    u_range = jnp.maximum(jnp.max(all_u) - jnp.min(all_u), 1e-10)

    p_A = ref_pair[3:6]
    p_B = ref_pair[9:12]

    def rdu_val(u, p):
        C3 = p[2]
        C2 = p[1] + p[2]
        w_C3 = prelec_pwf(C3, phi, eta)
        w_C2 = prelec_pwf(C2, phi, eta)
        pi3 = w_C3
        pi2 = w_C2 - w_C3
        pi1 = 1.0 - w_C2
        return pi1 * u[0] + pi2 * u[1] + pi3 * u[2]

    V_A = rdu_val(u_A, p_A)
    V_B = rdu_val(u_B, p_B)
    delta = (V_A - V_B) / (lam * u_range)
    P_val = jax.nn.sigmoid(delta)
    log_dP = jnp.log(jnp.abs(delta)) + jnp.log(P_val) + jnp.log(1 - P_val)

    return log_det_W + log_det_CE + log_dP


# =============================================================================
# Power Probability Weighting Function
# =============================================================================

def power_pwf(p, delta):
    """
    Power probability weighting function (Quiggin 1982).

    w(p) = p^delta

    Uniformly concave (delta < 1) or convex (delta > 1). NOT inverse-S.
    At delta = 1, w(p) = p (linear, reduces to EUT).

    Args:
        p: Probabilities (scalar or array)
        delta: Curvature parameter (positive scalar)

    Returns:
        Weighted probabilities (same shape as p)
    """
    delta = jnp.maximum(delta, 0.01)
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    return jnp.power(p, delta)


# =============================================================================
# RDU Value with Power Weights (3-outcome)
# =============================================================================

def rdu_value_3outcome_power(utilities, delta, idx_1, idx_2, idx_3, p1, p2, p3):
    """
    Compute RDU value for a 3-outcome lottery using Power decision weights.

    Args:
        utilities: Array of precomputed u(x) for unique outcomes
        delta: Power curvature parameter (natural scale, positive)
        idx_1, idx_2, idx_3: Indices into utilities for worst/middle/best
        p1, p2, p3: Probabilities of worst/middle/best outcomes

    Returns:
        RDU value (scalar)
    """
    u1 = utilities[idx_1]
    u2 = utilities[idx_2]
    u3 = utilities[idx_3]

    C3 = p3
    C2 = p2 + p3

    w_C3 = power_pwf(C3, delta)
    w_C2 = power_pwf(C2, delta)

    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    return pi1 * u1 + pi2 * u2 + pi3 * u3


# =============================================================================
# CRRA+Power Likelihood
# =============================================================================

def risk_likelihood_power(utilities, delta, datai, dataf, sid, lam):
    """
    Log likelihood for a subject's lottery choices using CRRA + Power PWF.

    Args:
        utilities: Precomputed u(x) for all unique outcomes (using crra_ufn)
        delta: Power curvature parameter (natural scale, positive)
        datai, dataf: Data tuples (same format as risk_likelihood)
        sid: Subject index
        lam: Fechner noise parameter (> 0)

    Returns:
        Log likelihood (scalar)
    """
    choices = datai[0]
    idx_L_1, idx_L_2, idx_L_3 = datai[1], datai[2], datai[3]
    idx_R_1, idx_R_2, idx_R_3 = datai[4], datai[5], datai[6]

    prob_L_1, prob_L_2, prob_L_3 = dataf[1], dataf[2], dataf[3]
    prob_R_1, prob_R_2, prob_R_3 = dataf[4], dataf[5], dataf[6]

    subj_choices = choices[sid]
    subj_idx_L_1, subj_idx_L_2, subj_idx_L_3 = idx_L_1[sid], idx_L_2[sid], idx_L_3[sid]
    subj_idx_R_1, subj_idx_R_2, subj_idx_R_3 = idx_R_1[sid], idx_R_2[sid], idx_R_3[sid]
    subj_pL1, subj_pL2, subj_pL3 = prob_L_1[sid], prob_L_2[sid], prob_L_3[sid]
    subj_pR1, subj_pR2, subj_pR3 = prob_R_1[sid], prob_R_2[sid], prob_R_3[sid]

    def choice_log_lik(choice, iL1, iL2, iL3, iR1, iR2, iR3,
                       pL1, pL2, pL3, pR1, pR2, pR3):
        v_L = rdu_value_3outcome_power(utilities, delta, iL1, iL2, iL3, pL1, pL2, pL3)
        v_R = rdu_value_3outcome_power(utilities, delta, iR1, iR2, iR3, pR1, pR2, pR3)

        u_L1, u_L2, u_L3 = utilities[iL1], utilities[iL2], utilities[iL3]
        u_R1, u_R2, u_R3 = utilities[iR1], utilities[iR2], utilities[iR3]
        u_max = jnp.maximum(
            jnp.maximum(u_L1, jnp.maximum(u_L2, u_L3)),
            jnp.maximum(u_R1, jnp.maximum(u_R2, u_R3)),
        )
        u_min = jnp.minimum(
            jnp.minimum(u_L1, jnp.minimum(u_L2, u_L3)),
            jnp.minimum(u_R1, jnp.minimum(u_R2, u_R3)),
        )
        u_range = jnp.maximum(u_max - u_min, 1e-10)

        delta_val = (v_L - v_R) / (lam * u_range)
        p_choose_L = jax.nn.sigmoid(delta_val)
        p_choose_L = jnp.clip(p_choose_L, 1e-10, 1 - 1e-10)

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
# CRRA+Power CE Computation
# =============================================================================

def rdu_ce_3outcome_power(r, delta, prizes, probs):
    """
    Compute certainty equivalent for a 3-outcome lottery under CRRA+Power.

    Uses CRRA utility with Power PWF. Prizes must be sorted ascending.
    CRRA has an analytic inverse, so no Newton iteration is needed.

    Args:
        r: CRRA coefficient (scalar)
        delta: Power curvature parameter (natural scale, positive)
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

    C3 = p3
    C2 = p2 + p3

    w_C3 = power_pwf(C3, delta)
    w_C2 = power_pwf(C2, delta)

    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    rdu_val = pi1 * u1 + pi2 * u2 + pi3 * u3
    return crra_inverse(rdu_val, r)


def ce_mapping_3lotteries_power(r, delta_raw, ref_lotteries):
    """
    Compute (CE1, CE2, CE3) for three reference lotteries under CRRA+Power.

    Forward map g: (r, delta_raw) -> (CE1, CE2, CE3). This is a 2->3 map
    (over-determined: more quantities than preference parameters).

    Args:
        r: CRRA coefficient (scalar)
        delta_raw: log(delta), Power curvature on log scale (scalar)
        ref_lotteries: Flat array of shape (18,) encoding 3 lotteries

    Returns:
        Array of shape (3,) with [CE1, CE2, CE3]
    """
    delta = jnp.exp(delta_raw)

    ce1 = rdu_ce_3outcome_power(r, delta, ref_lotteries[0:3], ref_lotteries[3:6])
    ce2 = rdu_ce_3outcome_power(r, delta, ref_lotteries[6:9], ref_lotteries[9:12])
    ce3 = rdu_ce_3outcome_power(r, delta, ref_lotteries[12:15], ref_lotteries[15:18])

    return jnp.array([ce1, ce2, ce3])


# =============================================================================
# CRRA+Power Choice Probability
# =============================================================================

def choice_probability_ref_power(r, delta_raw, lambda_raw, ref_pair):
    """
    Compute P(choose A | theta) for a reference lottery pair under CRRA+Power.

    Args:
        r: CRRA coefficient (scalar)
        delta_raw: log(delta) (scalar)
        lambda_raw: log(lambda) (scalar)
        ref_pair: Flat array of shape (12,)

    Returns:
        P(choose A) in (0, 1) (scalar)
    """
    delta = jnp.exp(delta_raw)
    lam = jnp.exp(lambda_raw)

    u_A = crra_ufn(ref_pair[0:3], r)
    u_B = crra_ufn(ref_pair[6:9], r)

    p_A = ref_pair[3:6]
    p_B = ref_pair[9:12]

    def rdu_val(u, p):
        C3 = p[2]
        C2 = p[1] + p[2]
        w_C3 = power_pwf(C3, delta)
        w_C2 = power_pwf(C2, delta)
        pi3 = w_C3
        pi2 = w_C2 - w_C3
        pi1 = 1.0 - w_C2
        return pi1 * u[0] + pi2 * u[1] + pi3 * u[2]

    rdu_A = rdu_val(u_A, p_A)
    rdu_B = rdu_val(u_B, p_B)

    all_u = jnp.concatenate([u_A, u_B])
    u_range = jnp.maximum(jnp.max(all_u) - jnp.min(all_u), 1e-10)

    delta_val = (rdu_A - rdu_B) / (lam * u_range)
    return jax.nn.sigmoid(delta_val)


# =============================================================================
# CRRA+Power Full Forward Map and Gram Determinant (k=4 > p=3)
# =============================================================================

def full_forward_map_power(r, delta_raw, lambda_raw, ref_lotteries, ref_pair):
    """
    Full 3D-to-4D forward map for CRRA+Power model.

    Maps (r, delta_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).
    This is the first k > p case: 4 quantities from 3 parameters.

    Args:
        r, delta_raw, lambda_raw: CRRA+Power parameters (scalars)
        ref_lotteries: Flat (18,) array for 3 reference lotteries
        ref_pair: Flat (12,) array for the reference pair

    Returns:
        Array of shape (4,) with [CE1, CE2, CE3, P_choice]
    """
    ces = ce_mapping_3lotteries_power(r, delta_raw, ref_lotteries)
    p_choice = choice_probability_ref_power(r, delta_raw, lambda_raw, ref_pair)
    return jnp.concatenate([ces, jnp.array([p_choice])])


def gram_logdet_power(r, delta_raw, lambda_raw, ref_lotteries, ref_pair):
    """
    Compute log(sqrt(det(J^T J))) for the CRRA+Power forward map.

    The Jacobian J is 4x3 (non-square: 4 quantities, 3 parameters).
    The induced prior uses the Gram determinant sqrt(det(J^T J)) as the
    volume element, generalizing |det J| from the square case.

    Args:
        r, delta_raw, lambda_raw: CRRA+Power parameters (scalars)
        ref_lotteries: Flat (18,) array
        ref_pair: Flat (12,) array

    Returns:
        log(sqrt(det(J^T J))) (scalar). Returns -inf if rank-deficient.
    """
    def f(theta):
        return full_forward_map_power(theta[0], theta[1], theta[2],
                                      ref_lotteries, ref_pair)

    theta = jnp.array([r, delta_raw, lambda_raw])
    J = jax.jacfwd(f)(theta)  # (4, 3)
    gram = J.T @ J             # (3, 3)
    sign, logdet = jnp.linalg.slogdet(gram)
    return jnp.where(sign > 0, 0.5 * logdet, -jnp.inf)


# =============================================================================
# Goldstein-Einhorn (1987) Probability Weighting Function
# =============================================================================

def ge_pwf(p, gamma, delta):
    """
    Goldstein-Einhorn (1987) / Lattimore-Baker-Witte (1992) PWF.

    w(p) = delta * p^gamma / (delta * p^gamma + (1-p)^gamma)

    Two parameters controlling:
        gamma: curvature (discriminability). gamma < 1 gives inverse-S.
        delta: elevation (attractiveness). delta > 1 is optimistic.

    At delta=1 reduces to the Karmarkar/TK form p^g / (p^g + (1-p)^g).

    Args:
        p: Probabilities (scalar or array)
        gamma: Curvature parameter (positive)
        delta: Elevation parameter (positive)

    Returns:
        Weighted probabilities (same shape as p)
    """
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    p_g = jnp.power(p, gamma)
    q_g = jnp.power(1 - p, gamma)
    return delta * p_g / (delta * p_g + q_g)


# =============================================================================
# RDU Value with GE Weights (3-outcome)
# =============================================================================

def rdu_value_3outcome_ge(utilities, gamma, delta, idx_1, idx_2, idx_3,
                          p1, p2, p3):
    """
    Compute RDU value for a 3-outcome lottery using GE decision weights.

    Same structure as rdu_value_3outcome but uses ge_pwf instead of prelec_pwf.

    Args:
        utilities: Array of precomputed u(x) for unique outcomes
        gamma: GE curvature parameter (natural scale, positive)
        delta: GE elevation parameter (natural scale, positive)
        idx_1, idx_2, idx_3: Indices into utilities for worst/middle/best
        p1, p2, p3: Probabilities of worst/middle/best outcomes

    Returns:
        RDU value (scalar)
    """
    u1 = utilities[idx_1]
    u2 = utilities[idx_2]
    u3 = utilities[idx_3]

    C3 = p3
    C2 = p2 + p3

    w_C3 = ge_pwf(C3, gamma, delta)
    w_C2 = ge_pwf(C2, gamma, delta)

    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    return pi1 * u1 + pi2 * u2 + pi3 * u3


# =============================================================================
# Likelihood with GE PWF
# =============================================================================

def risk_likelihood_ge(utilities, gamma, delta, datai, dataf, sid, lam):
    """
    Compute log likelihood for a subject's lottery choices using CRRA + GE PWF.

    Same structure as risk_likelihood but dispatches to rdu_value_3outcome_ge.

    Args:
        utilities: Precomputed u(x) for all unique outcomes
        gamma: GE curvature (natural scale, positive)
        delta: GE elevation (natural scale, positive)
        datai, dataf: Data arrays (same format as risk_likelihood)
        sid: Subject index
        lam: Fechner noise parameter (> 0)

    Returns:
        Log likelihood (scalar)
    """
    choices = datai[0]
    idx_L_1, idx_L_2, idx_L_3 = datai[1], datai[2], datai[3]
    idx_R_1, idx_R_2, idx_R_3 = datai[4], datai[5], datai[6]
    prob_L_1, prob_L_2, prob_L_3 = dataf[1], dataf[2], dataf[3]
    prob_R_1, prob_R_2, prob_R_3 = dataf[4], dataf[5], dataf[6]

    subj_choices = choices[sid]
    subj_idx_L_1, subj_idx_L_2, subj_idx_L_3 = idx_L_1[sid], idx_L_2[sid], idx_L_3[sid]
    subj_idx_R_1, subj_idx_R_2, subj_idx_R_3 = idx_R_1[sid], idx_R_2[sid], idx_R_3[sid]
    subj_pL1, subj_pL2, subj_pL3 = prob_L_1[sid], prob_L_2[sid], prob_L_3[sid]
    subj_pR1, subj_pR2, subj_pR3 = prob_R_1[sid], prob_R_2[sid], prob_R_3[sid]

    def choice_log_lik(choice, iL1, iL2, iL3, iR1, iR2, iR3,
                       pL1, pL2, pL3, pR1, pR2, pR3):
        v_L = rdu_value_3outcome_ge(utilities, gamma, delta, iL1, iL2, iL3, pL1, pL2, pL3)
        v_R = rdu_value_3outcome_ge(utilities, gamma, delta, iR1, iR2, iR3, pR1, pR2, pR3)

        u_L1, u_L2, u_L3 = utilities[iL1], utilities[iL2], utilities[iL3]
        u_R1, u_R2, u_R3 = utilities[iR1], utilities[iR2], utilities[iR3]
        u_max = jnp.maximum(
            jnp.maximum(u_L1, jnp.maximum(u_L2, u_L3)),
            jnp.maximum(u_R1, jnp.maximum(u_R2, u_R3)))
        u_min = jnp.minimum(
            jnp.minimum(u_L1, jnp.minimum(u_L2, u_L3)),
            jnp.minimum(u_R1, jnp.minimum(u_R2, u_R3)))
        u_range = jnp.maximum(u_max - u_min, 1e-10)

        delta_val = (v_L - v_R) / (lam * u_range)
        p_choose_L = jax.nn.sigmoid(delta_val)
        p_choose_L = jnp.clip(p_choose_L, 1e-10, 1 - 1e-10)
        return choice * jnp.log(p_choose_L) + (1 - choice) * jnp.log(1 - p_choose_L)

    log_liks = jax.vmap(choice_log_lik)(
        subj_choices,
        subj_idx_L_1, subj_idx_L_2, subj_idx_L_3,
        subj_idx_R_1, subj_idx_R_2, subj_idx_R_3,
        subj_pL1, subj_pL2, subj_pL3,
        subj_pR1, subj_pR2, subj_pR3)

    return jnp.sum(log_liks)


# =============================================================================
# GE Forward Map and Jacobian
# =============================================================================

def ce_mapping_3lotteries_ge(r, gamma_raw, delta_raw, ref_lotteries):
    """
    Map (r, gamma_raw, delta_raw) -> (CE1, CE2, CE3) for 3 binary reference lotteries.

    Uses CRRA utility with GE probability weighting.

    Args:
        r: CRRA coefficient
        gamma_raw: log(gamma) — GE curvature on log scale
        delta_raw: log(delta) — GE elevation on log scale
        ref_lotteries: Flat (18,) array, 6 values per lottery:
            [xlo, xmid, xhi, plo, pmid, phi] per lottery

    Returns:
        Array of shape (3,) with [CE1, CE2, CE3]
    """
    gamma = jnp.exp(gamma_raw)
    delta = jnp.exp(delta_raw)
    s = 1 - r

    def ce_for_lottery(lot_idx):
        base = lot_idx * 6
        xlo = ref_lotteries[base + 0]
        xhi = ref_lotteries[base + 2]
        p_hi = ref_lotteries[base + 5]

        w = ge_pwf(p_hi, gamma, delta)

        use_log = jnp.abs(s) < 1e-10
        s_safe = jnp.where(use_log, 1.0, s)

        inner = w * jnp.power(xhi, s_safe) + (1 - w) * jnp.power(xlo, s_safe)
        ce_power = jnp.power(jnp.maximum(inner, 1e-30), 1.0 / s_safe)

        M = w * jnp.log(xhi) + (1 - w) * jnp.log(xlo)
        ce_log = jnp.exp(M)

        return jnp.where(use_log, ce_log, ce_power)

    ce1 = ce_for_lottery(0)
    ce2 = ce_for_lottery(1)
    ce3 = ce_for_lottery(2)
    return jnp.array([ce1, ce2, ce3])


def choice_probability_ref_ge(r, gamma_raw, delta_raw, lambda_raw, ref_pair):
    """
    Compute P(choose A) for the reference pair under CRRA + GE.

    Same structure as choice_probability_ref but with GE PWF.

    Args:
        r, gamma_raw, delta_raw, lambda_raw: Parameters
        ref_pair: Flat (12,) array [A_outcomes(3), A_probs(3), B_outcomes(3), B_probs(3)]

    Returns:
        P(choose A) in (0, 1)
    """
    gamma = jnp.exp(gamma_raw)
    delta = jnp.exp(delta_raw)
    lam = jnp.exp(lambda_raw)

    u_A = crra_ufn(ref_pair[0:3], r)
    u_B = crra_ufn(ref_pair[6:9], r)
    all_u = jnp.concatenate([u_A, u_B])
    u_range = jnp.maximum(jnp.max(all_u) - jnp.min(all_u), 1e-10)

    p_A = ref_pair[3:6]
    p_B = ref_pair[9:12]

    def rdu_val(u, p):
        C3 = p[2]
        C2 = p[1] + p[2]
        w_C3 = ge_pwf(C3, gamma, delta)
        w_C2 = ge_pwf(C2, gamma, delta)
        pi3 = w_C3
        pi2 = w_C2 - w_C3
        pi1 = 1.0 - w_C2
        return pi1 * u[0] + pi2 * u[1] + pi3 * u[2]

    V_A = rdu_val(u_A, p_A)
    V_B = rdu_val(u_B, p_B)
    diff = (V_A - V_B) / (lam * u_range)
    return jax.nn.sigmoid(diff)


def full_forward_map_ge(r, gamma_raw, delta_raw, lambda_raw,
                        ref_lotteries, ref_pair):
    """
    Full 4D forward map: (r, gamma_raw, delta_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).

    Args:
        r, gamma_raw, delta_raw, lambda_raw: Model parameters (scalars)
        ref_lotteries: Flat (18,) array
        ref_pair: Flat (12,) array

    Returns:
        Array of shape (4,) with [CE1, CE2, CE3, P_choice]
    """
    ces = ce_mapping_3lotteries_ge(r, gamma_raw, delta_raw, ref_lotteries)
    p_choice = choice_probability_ref_ge(r, gamma_raw, delta_raw, lambda_raw, ref_pair)
    return jnp.concatenate([ces, jnp.array([p_choice])])


def full_jacobian_logabsdet_ge(r, gamma_raw, delta_raw, lambda_raw,
                               ref_lotteries, ref_pair):
    """
    Compute log|det J| of the full 4D forward map — analytic decomposition.

    Decomposes as:
      log|det J| = log|det J_W| + log|det J_CE| + log|dP/dlambda_raw|

    Component 1 (GE W map): 2x2 determinant d(w1,w2)/d(gamma_raw, delta_raw).
    Component 2 (CE map): 3x3 determinant in (r, w1, w2) space.
    Component 3 (choice probability): dP/dlambda_raw.

    Args:
        r, gamma_raw, delta_raw, lambda_raw: Parameters
        ref_lotteries: Flat (18,) array
        ref_pair: Flat (12,) array

    Returns:
        log|det J| (scalar)
    """
    gamma = jnp.exp(gamma_raw)
    delta = jnp.exp(delta_raw)
    lam = jnp.exp(lambda_raw)
    s = 1 - r

    # --- Component 1: GE W map ---
    # w(p) = delta * p^g / (delta * p^g + (1-p)^g)
    # d(w)/d(gamma_raw) = dw/dgamma * gamma  (chain rule through exp)
    # d(w)/d(delta_raw) = dw/ddelta * delta
    #
    # dw/dgamma = delta * p^g * (1-p)^g * [ln(p) * (1-p)^g + ln(1-p) * delta * p^g]
    #            ... actually let's use: w = delta*A / (delta*A + B) where A=p^g, B=(1-p)^g
    # dw/dgamma = delta * (A'*B - A*B') / (delta*A + B)^2 ... with A'=p^g*ln(p), B'=(1-p)^g*ln(1-p)
    # dw/ddelta = A * B / (delta*A + B)^2  (= w*(1-w)/delta)

    p1 = ref_lotteries[5]    # P(best) for L1
    p2 = ref_lotteries[11]   # P(best) for L2

    def ge_w_and_derivs(p):
        """Compute w, dw/d(gamma_raw), dw/d(delta_raw) for GE PWF."""
        A = jnp.power(p, gamma)
        B = jnp.power(1 - p, gamma)
        denom = delta * A + B
        w = delta * A / denom

        # dw/dgamma: derivative of w w.r.t. gamma
        A_prime = A * jnp.log(p)         # d(p^g)/dg = p^g * ln(p)
        B_prime = B * jnp.log(1 - p)     # d((1-p)^g)/dg = (1-p)^g * ln(1-p)
        dw_dgamma = delta * (A_prime * B - A * B_prime) / (denom ** 2)

        # dw/ddelta: derivative of w w.r.t. delta
        dw_ddelta = A * B / (denom ** 2)

        # Chain rule for raw params: d/d(gamma_raw) = dw/dgamma * gamma
        #                             d/d(delta_raw) = dw/ddelta * delta
        dw_dgamma_raw = dw_dgamma * gamma
        dw_ddelta_raw = dw_ddelta * delta

        return w, dw_dgamma_raw, dw_ddelta_raw

    w1, dw1_dg, dw1_dd = ge_w_and_derivs(p1)
    w2, dw2_dg, dw2_dd = ge_w_and_derivs(p2)

    # det of 2x2 W map: |dw1/dg * dw2/dd - dw1/dd * dw2/dg|
    det_W = dw1_dg * dw2_dd - dw1_dd * dw2_dg
    log_det_W = jnp.log(jnp.abs(det_W))

    # --- Component 2: CE map in (r, w1, w2) space ---
    # Same as Prelec: uses _ce_and_derivs which only depends on (w, xlo, xhi, s)
    w3 = w1  # L3 uses same probability as L1 (p_hi = 0.25)
    xlo1, xhi1 = ref_lotteries[0], ref_lotteries[2]
    xlo2, xhi2 = ref_lotteries[6], ref_lotteries[8]
    xlo3, xhi3 = ref_lotteries[12], ref_lotteries[14]

    _, dce1_dw1, dce1_dr = _ce_and_derivs(w1, xlo1, xhi1, s)
    _, dce2_dw2, dce2_dr = _ce_and_derivs(w2, xlo2, xhi2, s)
    _, dce3_dw1, dce3_dr = _ce_and_derivs(w3, xlo3, xhi3, s)

    det_CE = dce2_dw2 * (dce1_dr * dce3_dw1 - dce3_dr * dce1_dw1)
    log_det_CE = jnp.log(jnp.abs(det_CE))

    # --- Component 3: dP/dlambda_raw ---
    u_A = crra_ufn(ref_pair[0:3], r)
    u_B = crra_ufn(ref_pair[6:9], r)
    all_u = jnp.concatenate([u_A, u_B])
    u_range = jnp.maximum(jnp.max(all_u) - jnp.min(all_u), 1e-10)

    p_A = ref_pair[3:6]
    p_B = ref_pair[9:12]

    def rdu_val(u, p):
        C3 = p[2]
        C2 = p[1] + p[2]
        w_C3 = ge_pwf(C3, gamma, delta)
        w_C2 = ge_pwf(C2, gamma, delta)
        pi3 = w_C3
        pi2 = w_C2 - w_C3
        pi1 = 1.0 - w_C2
        return pi1 * u[0] + pi2 * u[1] + pi3 * u[2]

    V_A = rdu_val(u_A, p_A)
    V_B = rdu_val(u_B, p_B)
    diff = (V_A - V_B) / (lam * u_range)
    P_val = jax.nn.sigmoid(diff)
    log_dP = jnp.log(jnp.abs(diff)) + jnp.log(P_val) + jnp.log(1 - P_val)

    return log_det_W + log_det_CE + log_dP


