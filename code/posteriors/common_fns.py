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
# Expo-Power Utility Function
# =============================================================================

def ep_ufn(x, alpha, r, x_min=10.0, x_max=50.0):
    """
    Normalized Expo-Power utility function (Saha 1993).

    V(x) = (1 - exp(-alpha * x^(1-r))) / (alpha * (1-r))
    u(x) = (V(x) - V(x_min)) / (V(x_max) - V(x_min))

    Normalized to [0, 1] over [x_min, x_max]. Continuous at r = 1,
    where it reduces to (1 - exp(-alpha * ln(x))) / (alpha) after
    applying L'Hopital. At alpha -> 0 it reduces to CRRA.

    Args:
        x: Outcomes (scalar or array)
        alpha: Expo-power curvature (positive scalar)
        r: Risk aversion coefficient (scalar)
        x_min: Minimum outcome (default 10.0)
        x_max: Maximum outcome (default 50.0)

    Returns:
        Normalized utility values (same shape as x)
    """
    x = jnp.maximum(x, 1e-10)
    alpha = jnp.maximum(alpha, 1e-10)
    r_safe = jnp.clip(r, -10, 10)
    s = 1 - r_safe

    # Standard form: V(x) = (1 - exp(-alpha * x^s)) / (alpha * s)
    # Normalized: u(x) = (V(x) - V(xmin)) / (V(xmax) - V(xmin))
    # Since alpha*s cancels in the ratio:
    #   u(x) = (exp(-alpha * xmin^s) - exp(-alpha * x^s))
    #        / (exp(-alpha * xmin^s) - exp(-alpha * xmax^s))
    x_s = jnp.power(x, s)
    xmin_s = jnp.power(x_min, s)
    xmax_s = jnp.power(x_max, s)

    exp_x = jnp.exp(-alpha * x_s)
    exp_min = jnp.exp(-alpha * xmin_s)
    exp_max = jnp.exp(-alpha * xmax_s)
    power_form = (exp_min - exp_x) / (exp_min - exp_max)

    # Log form (limit as s -> 0, i.e. r -> 1):
    # V(x) -> (1 - x^(-alpha)) / alpha  (via x^s -> 1 + s*ln(x))
    # Actually: lim_{s->0} exp(-alpha * x^s) = exp(-alpha) * x^{-alpha*... }
    # More carefully: x^s = exp(s ln x) so -alpha * x^s = -alpha * exp(s ln x)
    # At s=0: -alpha * 1 = -alpha, so exp(-alpha * x^s) -> exp(-alpha) for all x.
    # Need first-order expansion: x^s ≈ 1 + s ln(x), so
    #   exp(-alpha * x^s) ≈ exp(-alpha) * exp(-alpha * s * ln(x))
    #                      = exp(-alpha) * x^{-alpha*s}
    # The s factors cancel in the normalized ratio:
    #   u(x) = (x_min^{-alpha*s} - x^{-alpha*s})
    #        / (x_min^{-alpha*s} - x_max^{-alpha*s})
    # At s=0 this is 0/0, but taking the limit via L'Hopital on s:
    #   u(x) -> (ln(x_min) * x_min^0 - ln(x) * x^0)
    #         / (ln(x_min) * x_min^0 - ln(x_max) * x_max^0)
    #         = (ln(x_min) - ln(x)) / (ln(x_min) - ln(x_max))
    #         = ln(x/x_min) / ln(x_max/x_min)
    # This is the same as CRRA at r=1.
    log_form = jnp.log(x / x_min) / jnp.log(x_max / x_min)

    use_log = jnp.abs(s) < 1e-10
    return jnp.where(use_log, log_form, power_form)


def _ep_raw_utility(x, alpha, s):
    """Unnormalized EP utility: exp(-alpha * x_min^s) - exp(-alpha * x^s).

    Returns the numerator of the normalized form. Used by ep_inverse.
    """
    return jnp.exp(-alpha * jnp.power(x, s))


def ep_inverse(u_val, alpha, r, x_min=10.0, x_max=50.0):
    """
    Numerical inverse of normalized Expo-Power utility via Newton iteration.

    Solves ep_ufn(CE; alpha, r) = u_val for CE using Newton's method
    with a CRRA-based initial guess.

    Args:
        u_val: Normalized utility value in [0, 1] (scalar)
        alpha: EP curvature parameter (positive scalar)
        r: Risk aversion coefficient (scalar)
        x_min: Minimum outcome (default 10.0)
        x_max: Maximum outcome (default 50.0)

    Returns:
        Certainty equivalent in prize units (scalar)
    """
    alpha = jnp.maximum(alpha, 1e-10)
    r_safe = jnp.clip(r, -10, 10)
    s = 1 - r_safe

    # Initial guess: CRRA inverse (good because EP nests CRRA as alpha -> 0)
    x0 = crra_inverse(u_val, r, x_min, x_max)
    x0 = jnp.clip(x0, x_min + 1e-8, x_max - 1e-8)

    # Precompute constants for the normalized EP utility
    xmin_s = jnp.power(x_min, s)
    xmax_s = jnp.power(x_max, s)
    exp_min = jnp.exp(-alpha * xmin_s)
    exp_max = jnp.exp(-alpha * xmax_s)
    denom = exp_min - exp_max

    def newton_step(_, x):
        x = jnp.clip(x, x_min + 1e-8, x_max - 1e-8)
        x_s = jnp.power(x, s)
        exp_x = jnp.exp(-alpha * x_s)

        # f(x) = (exp_min - exp_x) / denom - u_val
        fx = (exp_min - exp_x) / denom - u_val

        # f'(x) = alpha * s * x^(s-1) * exp(-alpha * x^s) / denom
        dfx = alpha * s * jnp.power(x, s - 1) * exp_x / denom

        # Guard against zero derivative
        dfx = jnp.where(jnp.abs(dfx) < 1e-30, 1e-30, dfx)

        x_new = x - fx / dfx
        return jnp.clip(x_new, x_min + 1e-8, x_max - 1e-8)

    # Handle s=0 (r=1) case: use CRRA inverse directly
    use_log = jnp.abs(s) < 1e-10
    newton_result = jax.lax.fori_loop(0, 20, newton_step, x0)
    log_result = x_min * jnp.power(x_max / x_min, u_val)

    return jnp.where(use_log, log_result, newton_result)


# =============================================================================
# Tversky-Kahneman (1992) Probability Weighting Function
# =============================================================================

def tk_pwf(p, gamma):
    """
    Tversky-Kahneman (1992) probability weighting function.

    w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)

    Produces inverse-S shape for gamma < 1: overweights small probabilities
    and underweights large probabilities. At gamma = 1, w(p) = p.

    Args:
        p: Probabilities (scalar or array)
        gamma: Curvature parameter (positive scalar, typically 0.3-1.0)

    Returns:
        Weighted probabilities (same shape as p)
    """
    gamma = jnp.maximum(gamma, 0.28)
    p = jnp.clip(p, 1e-10, 1 - 1e-10)

    p_g = jnp.power(p, gamma)
    q_g = jnp.power(1 - p, gamma)
    return p_g / jnp.power(p_g + q_g, 1.0 / gamma)


# =============================================================================
# RDU Value with TK Weights (3-outcome)
# =============================================================================

def rdu_value_3outcome_ep(utilities, gamma, idx_1, idx_2, idx_3, p1, p2, p3):
    """
    Compute RDU value for a 3-outcome lottery using TK decision weights.

    Same structure as rdu_value_3outcome but uses tk_pwf instead of prelec_pwf.

    Args:
        utilities: Array of precomputed u(x) for unique outcomes
        gamma: TK curvature parameter (natural scale, positive)
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

    w_C3 = tk_pwf(C3, gamma)
    w_C2 = tk_pwf(C2, gamma)

    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    return pi1 * u1 + pi2 * u2 + pi3 * u3


# =============================================================================
# EP+TK Likelihood
# =============================================================================

def risk_likelihood_ep(utilities, gamma, datai, dataf, sid, lam):
    """
    Log likelihood for a subject's lottery choices using EP utility + TK PWF.

    Same structure as risk_likelihood but uses tk_pwf for decision weights.

    Args:
        utilities: Precomputed u(x) for all unique outcomes (using ep_ufn)
        gamma: TK curvature parameter (natural scale, positive)
        datai: Tuple of integer arrays (choices, indices)
        dataf: Tuple of float arrays (outcomes, probabilities)
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
        v_L = rdu_value_3outcome_ep(utilities, gamma, iL1, iL2, iL3, pL1, pL2, pL3)
        v_R = rdu_value_3outcome_ep(utilities, gamma, iR1, iR2, iR3, pR1, pR2, pR3)

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

        delta = (v_L - v_R) / (lam * u_range)
        p_choose_L = jax.nn.sigmoid(delta)
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
# EP+TK CE Computation for Reference Lotteries
# =============================================================================

def rdu_ce_3outcome_ep(alpha, r, gamma, prizes, probs):
    """
    Compute certainty equivalent for a 3-outcome lottery under EP+TK.

    Uses Expo-Power utility with TK PWF. Prizes must be sorted ascending.

    Args:
        alpha: EP curvature parameter (natural scale, positive)
        r: Risk aversion coefficient (scalar)
        gamma: TK curvature parameter (natural scale, positive)
        prizes: Array of 3 prizes [x1, x2, x3] sorted ascending
        probs: Array of 3 probabilities [p1, p2, p3] summing to 1

    Returns:
        CE in prize units (scalar)
    """
    x1, x2, x3 = prizes[0], prizes[1], prizes[2]
    p1, p2, p3 = probs[0], probs[1], probs[2]

    u1 = ep_ufn(x1, alpha, r)
    u2 = ep_ufn(x2, alpha, r)
    u3 = ep_ufn(x3, alpha, r)

    C3 = p3
    C2 = p2 + p3

    w_C3 = tk_pwf(C3, gamma)
    w_C2 = tk_pwf(C2, gamma)

    pi3 = w_C3
    pi2 = w_C2 - w_C3
    pi1 = 1.0 - w_C2

    rdu_val = pi1 * u1 + pi2 * u2 + pi3 * u3
    return ep_inverse(rdu_val, alpha, r)


def ce_mapping_3lotteries_ep(alpha_raw, r, gamma_raw, ref_lotteries):
    """
    Compute (CE1, CE2, CE3) for three reference lotteries under EP+TK.

    Forward map g(theta) from parameter space to CE space.

    Args:
        alpha_raw: log(alpha), EP curvature on log scale (scalar)
        r: Risk aversion coefficient (scalar)
        gamma_raw: log(gamma), TK curvature on log scale (scalar)
        ref_lotteries: Flat array of shape (18,) encoding 3 lotteries

    Returns:
        Array of shape (3,) with [CE1, CE2, CE3]
    """
    alpha = jnp.exp(alpha_raw)
    gamma = jnp.exp(gamma_raw)

    ce1 = rdu_ce_3outcome_ep(alpha, r, gamma, ref_lotteries[0:3], ref_lotteries[3:6])
    ce2 = rdu_ce_3outcome_ep(alpha, r, gamma, ref_lotteries[6:9], ref_lotteries[9:12])
    ce3 = rdu_ce_3outcome_ep(alpha, r, gamma, ref_lotteries[12:15], ref_lotteries[15:18])

    return jnp.array([ce1, ce2, ce3])


# =============================================================================
# EP+TK Choice Probability for Reference Pair
# =============================================================================

def choice_probability_ref_ep(alpha_raw, r, gamma_raw, lambda_raw, ref_pair):
    """
    Compute P(choose A | theta) for a reference lottery pair under EP+TK.

    Uses Contextual Utility (Wilcox 2011) with EP utility and TK PWF.

    Args:
        alpha_raw: log(alpha) (scalar)
        r: Risk aversion coefficient (scalar)
        gamma_raw: log(gamma) (scalar)
        lambda_raw: log(lambda) (scalar)
        ref_pair: Flat array of shape (12,) encoding 2 lotteries

    Returns:
        P(choose A) in (0, 1) (scalar)
    """
    alpha = jnp.exp(alpha_raw)
    gamma = jnp.exp(gamma_raw)
    lam = jnp.exp(lambda_raw)

    u_A = ep_ufn(ref_pair[0:3], alpha, r)
    u_B = ep_ufn(ref_pair[6:9], alpha, r)

    p_A = ref_pair[3:6]
    p_B = ref_pair[9:12]

    def rdu_val(u, p):
        C3 = p[2]
        C2 = p[1] + p[2]
        w_C3 = tk_pwf(C3, gamma)
        w_C2 = tk_pwf(C2, gamma)
        pi3 = w_C3
        pi2 = w_C2 - w_C3
        pi1 = 1.0 - w_C2
        return pi1 * u[0] + pi2 * u[1] + pi3 * u[2]

    rdu_A = rdu_val(u_A, p_A)
    rdu_B = rdu_val(u_B, p_B)

    all_u = jnp.concatenate([u_A, u_B])
    u_range = jnp.maximum(jnp.max(all_u) - jnp.min(all_u), 1e-10)

    delta = (rdu_A - rdu_B) / (lam * u_range)
    return jax.nn.sigmoid(delta)


# =============================================================================
# EP+TK Full 4D Forward Map and Jacobian
# =============================================================================

def full_forward_map_ep(alpha_raw, r, gamma_raw, lambda_raw,
                        ref_lotteries, ref_pair):
    """
    Full 4D forward map for EP+TK model.

    Maps (alpha_raw, r, gamma_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).

    Args:
        alpha_raw: log(alpha) (scalar)
        r: Risk aversion coefficient (scalar)
        gamma_raw: log(gamma) (scalar)
        lambda_raw: log(lambda) (scalar)
        ref_lotteries: Flat (18,) array for 3 reference lotteries
        ref_pair: Flat (12,) array for the reference pair

    Returns:
        Array of shape (4,) with [CE1, CE2, CE3, P_choice]
    """
    ces = ce_mapping_3lotteries_ep(alpha_raw, r, gamma_raw, ref_lotteries)
    p_choice = choice_probability_ref_ep(alpha_raw, r, gamma_raw, lambda_raw,
                                         ref_pair)
    return jnp.concatenate([ces, jnp.array([p_choice])])


def full_jacobian_logabsdet_ep(alpha_raw, r, gamma_raw, lambda_raw,
                               ref_lotteries, ref_pair):
    """
    Compute log|det J| of the EP+TK forward map via JAX autodiff.

    Uses jax.jacfwd + jnp.linalg.slogdet. No analytic decomposition.

    Args:
        alpha_raw, r, gamma_raw, lambda_raw: EP+TK parameters (scalars)
        ref_lotteries: Flat (18,) array
        ref_pair: Flat (12,) array

    Returns:
        log|det J| (scalar). Returns -inf if det is zero.
    """
    def f(theta):
        return full_forward_map_ep(theta[0], theta[1], theta[2], theta[3],
                                   ref_lotteries, ref_pair)

    theta = jnp.array([alpha_raw, r, gamma_raw, lambda_raw])
    J = jax.jacfwd(f)(theta)  # (4, 4)
    sign, logabsdet = jnp.linalg.slogdet(J)
    return jnp.where(sign != 0, logabsdet, -jnp.inf)


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
