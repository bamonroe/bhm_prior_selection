"""
Symbolic derivation of the Jacobian for the RDU CE mapping.

The full forward map is:
    g: (r, phi_raw, eta_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice)

where phi_raw = ln(phi), eta_raw = ln(eta), lambda_raw = ln(lambda).

The Jacobian has block-triangular structure since CEs don't depend on lambda:

    J = [[d(CE1,CE2,CE3)/d(r,phi_raw,eta_raw)    0              ],
         [dP/d(r,phi_raw,eta_raw)                  dP/dlambda_raw ]]

So: det(J) = det(J_CE) * dP/dlambda_raw

The CE Jacobian itself factors through the Prelec reparameterization:
    (r, phi_raw, eta_raw) -> (r, w1, w2) -> (CE1, CE2, CE3)

Step 1: Prelec change of variables
    W: (phi_raw, eta_raw) -> (w1, w2) where w_k = exp(-eta * (-ln p_k)^phi)
    This is a 2x2 Jacobian (analytic).

Step 2: CE map in (r, w1, w2) space
    F: (r, w1, w2) -> (CE1, CE2, CE3)
    For binary lotteries: CE_k = [w_k * xhi^s + (1-w_k) * xlo^s]^(1/s)
    where s = 1-r. This is a 3x3 Jacobian (analytic).

Step 3: Choice probability
    P = sigmoid((RDU_A - RDU_B) / (lambda * u_range))
    dP/dlambda_raw involves the chain rule through exp(lambda_raw).

The full Jacobian is assembled from these pieces.
"""

import sympy as sp
from sympy import symbols, exp, ln, log, sqrt, Abs, simplify, Matrix, Function
from sympy import Rational, pi, Symbol


# =============================================================================
# Step 1: Prelec W map — (phi_raw, eta_raw) -> (w1, w2)
# =============================================================================

def prelec_w_jacobian():
    """Derive the 2x2 Jacobian of the Prelec W map analytically.

    W: (phi_raw, eta_raw) -> (w1, w2)
    where phi = exp(phi_raw), eta = exp(eta_raw),
    w_k = exp(-eta * t_k^phi), t_k = -ln(p_k).

    Returns dict with symbolic expressions.
    """
    phi_raw, eta_raw = symbols('phi_raw eta_raw', real=True)
    p1, p2 = symbols('p_1 p_2', positive=True)

    phi = exp(phi_raw)
    eta = exp(eta_raw)

    t1 = -ln(p1)
    t2 = -ln(p2)

    w1 = exp(-eta * t1**phi)
    w2 = exp(-eta * t2**phi)

    # Jacobian dW/d(phi_raw, eta_raw)
    J_W = Matrix([
        [sp.diff(w1, phi_raw), sp.diff(w1, eta_raw)],
        [sp.diff(w2, phi_raw), sp.diff(w2, eta_raw)],
    ])

    # Simplify
    J_W_simplified = J_W.applyfunc(simplify)

    det_W = simplify(J_W_simplified.det())

    return {
        'phi_raw': phi_raw, 'eta_raw': eta_raw,
        'p1': p1, 'p2': p2,
        'w1': w1, 'w2': w2,
        'J_W': J_W_simplified,
        'det_W': det_W,
        'log_abs_det_W': simplify(ln(Abs(det_W))),
    }


# =============================================================================
# Step 2: CE map in (r, w1, w2) space — (r, w1, w2) -> (CE1, CE2, CE3)
# =============================================================================

def ce_jacobian_rw_space():
    """Derive the 3x3 Jacobian of the CE map in (r, w1, w2) space.

    For binary lottery k with outcomes (xlo_k, xhi_k) and decision weight w_k:
        CE_k^s = w_k * xhi_k^s + (1-w_k) * xlo_k^s,  s = 1-r
        CE_k = (w_k * xhi_k^s + (1-w_k) * xlo_k^s)^(1/s)

    The Jacobian is 3x3 in (r, w1, w2). Note:
    - dCE_k/dw_j = 0 for j != k (each CE depends on only one decision weight)
    - dCE_k/dr involves the s-derivative of the power mean

    Returns dict with symbolic expressions.
    """
    r = symbols('r', real=True)
    w1, w2, w3 = symbols('w_1 w_2 w_3', positive=True)
    s = 1 - r

    # Lottery outcomes (positive)
    xlo1, xhi1 = symbols('xlo_1 xhi_1', positive=True)
    xlo2, xhi2 = symbols('xlo_2 xhi_2', positive=True)
    xlo3, xhi3 = symbols('xlo_3 xhi_3', positive=True)

    def ce_expr(w, xlo, xhi):
        """CE = (w * xhi^s + (1-w) * xlo^s)^(1/s)"""
        return (w * xhi**s + (1 - w) * xlo**s)**Rational(1, 1) ** (1/s)

    # More explicit form to help sympy
    def ce_power_mean(w, xlo, xhi):
        inner = w * xhi**s + (1 - w) * xlo**s
        return inner**(1/s)

    ce1 = ce_power_mean(w1, xlo1, xhi1)
    ce2 = ce_power_mean(w2, xlo2, xhi2)
    ce3 = ce_power_mean(w3, xlo3, xhi3)

    # 3x3 Jacobian: d(CE1, CE2, CE3)/d(r, w1, w2)
    # Note: w3 = w1 in the shared-probability design (p3 = p1)
    # But we keep it general here; the shared-prob substitution happens later

    # For the general case (no shared probability), the Jacobian is:
    # Row k: [dCE_k/dr, dCE_k/dw1, dCE_k/dw2]
    # But CE_k depends on w_k only, so most dCE_k/dw_j = 0

    # dCE_k/dw_k (diagonal terms)
    dce1_dw1 = sp.diff(ce1, w1)
    dce2_dw2 = sp.diff(ce2, w2)
    dce3_dw3 = sp.diff(ce3, w3)

    # dCE_k/dr = dCE_k/ds * ds/dr = -dCE_k/ds
    dce1_dr = sp.diff(ce1, r)
    dce2_dr = sp.diff(ce2, r)
    dce3_dr = sp.diff(ce3, r)

    return {
        'r': r, 's': s,
        'w1': w1, 'w2': w2, 'w3': w3,
        'ce1': ce1, 'ce2': ce2, 'ce3': ce3,
        'dce1_dr': simplify(dce1_dr),
        'dce2_dr': simplify(dce2_dr),
        'dce3_dr': simplify(dce3_dr),
        'dce1_dw1': simplify(dce1_dw1),
        'dce2_dw2': simplify(dce2_dw2),
        'dce3_dw3': simplify(dce3_dw3),
        'xlo1': xlo1, 'xhi1': xhi1,
        'xlo2': xlo2, 'xhi2': xhi2,
        'xlo3': xlo3, 'xhi3': xhi3,
    }


# =============================================================================
# Step 3: Choice probability derivative
# =============================================================================

def choice_prob_derivative():
    """Derive dP/dlambda_raw analytically.

    P = sigmoid(Delta / (lambda * u_range))
    lambda = exp(lambda_raw)
    dP/dlambda_raw = dP/dDelta_scaled * dDelta_scaled/dlambda_raw

    Returns dict with symbolic expressions.
    """
    lambda_raw = symbols('lambda_raw', real=True)
    Delta_V, u_range = symbols('Delta_V u_range', real=True)

    lam = exp(lambda_raw)
    delta_scaled = Delta_V / (lam * u_range)
    P = 1 / (1 + exp(-delta_scaled))

    dP_dlambda_raw = simplify(sp.diff(P, lambda_raw))

    return {
        'lambda_raw': lambda_raw,
        'Delta_V': Delta_V,
        'u_range': u_range,
        'P': P,
        'dP_dlambda_raw': dP_dlambda_raw,
    }


# =============================================================================
# Step 4: Assemble full log|det J|
# =============================================================================

def full_log_det_jacobian_formula():
    """Derive the full log|det J| as a sum of components.

    The full 4x4 Jacobian in (r, phi_raw, eta_raw, lambda_raw) space
    factors as:

    det(J_full) = det(J_CE_in_rphieta) * dP/dlambda_raw

    And J_CE_in_rphieta itself factors through the Prelec reparameterization:

    J_CE_in_rphieta = J_CE_in_rw @ diag(1, J_W)

    where J_CE_in_rw is the 3x3 Jacobian in (r, w1, w2) space and
    J_W is the 2x2 Prelec Jacobian.

    So: det(J_CE_in_rphieta) = det(J_CE_in_rw) * det(J_W)

    And: log|det J_full| = log|det J_CE_in_rw| + log|det J_W| + log|dP/dlambda_raw|

    This decomposition allows each piece to be computed analytically.
    """
    print("=" * 70)
    print("Full Jacobian Decomposition")
    print("=" * 70)
    print()
    print("log|det J_full| = log|det J_CE(r,w1,w2)| + log|det J_W(phi,eta)| + log|dP/dlambda|")
    print()

    # Step 1: Prelec W map
    print("--- Step 1: Prelec W map ---")
    w_result = prelec_w_jacobian()
    print(f"J_W =")
    sp.pprint(w_result['J_W'])
    print(f"\ndet(J_W) =")
    sp.pprint(w_result['det_W'])

    # Step 2: CE map in (r, w) space
    print("\n--- Step 2: CE map in (r, w1, w2) space ---")
    ce_result = ce_jacobian_rw_space()
    print("dCE1/dw1 =")
    sp.pprint(ce_result['dce1_dw1'])
    print("\ndCE1/dr =")
    sp.pprint(ce_result['dce1_dr'])

    # Step 3: Choice probability
    print("\n--- Step 3: dP/dlambda_raw ---")
    cp_result = choice_prob_derivative()
    print("dP/dlambda_raw =")
    sp.pprint(cp_result['dP_dlambda_raw'])

    print("\n" + "=" * 70)
    print("Summary: Three additive components in log-space")
    print("=" * 70)

    return w_result, ce_result, cp_result


# =============================================================================
# Export for Stan
# =============================================================================

def generate_stan_expressions():
    """Generate expressions suitable for Stan code generation.

    Stan needs explicit formulas, not matrix operations, because it
    doesn't have symbolic autodiff at the model level.
    """
    # Prelec derivatives
    phi_raw, eta_raw = symbols('phi_raw eta_raw', real=True)
    p_k = symbols('p_k', positive=True)
    phi = exp(phi_raw)
    eta = exp(eta_raw)
    t_k = -ln(p_k)
    w_k = exp(-eta * t_k**phi)

    dw_dphi_raw = simplify(sp.diff(w_k, phi_raw))
    dw_deta_raw = simplify(sp.diff(w_k, eta_raw))

    print("Prelec derivatives (per probability point):")
    print(f"  w_k = exp(-eta * t_k^phi)")
    print(f"  dw_k/dphi_raw = {dw_dphi_raw}")
    print(f"  dw_k/deta_raw = {dw_deta_raw}")

    # CE derivatives for binary lottery
    r = symbols('r', real=True)
    w, xlo, xhi = symbols('w xlo xhi', positive=True)
    s = 1 - r
    inner = w * xhi**s + (1 - w) * xlo**s
    ce = inner**(1/s)

    dce_dw = simplify(sp.diff(ce, w))
    dce_dr = simplify(sp.diff(ce, r))

    print(f"\nCE derivatives (per lottery):")
    print(f"  CE = (w*xhi^s + (1-w)*xlo^s)^(1/s),  s = 1-r")
    print(f"  dCE/dw = {dce_dw}")
    print(f"  dCE/dr = {dce_dr}")

    # Choice prob derivative
    lambda_raw = symbols('lambda_raw', real=True)
    DV, u_range = symbols('DeltaV u_range', real=True)
    lam = exp(lambda_raw)
    delta = DV / (lam * u_range)
    P = 1 / (1 + exp(-delta))
    dP_dlam_raw = simplify(sp.diff(P, lambda_raw))

    print(f"\nChoice probability derivative:")
    print(f"  P = sigmoid(DeltaV / (lambda * u_range))")
    print(f"  dP/dlambda_raw = {dP_dlam_raw}")

    return {
        'dw_dphi_raw': dw_dphi_raw,
        'dw_deta_raw': dw_deta_raw,
        'dce_dw': dce_dw,
        'dce_dr': dce_dr,
        'dP_dlambda_raw': dP_dlam_raw,
    }


if __name__ == "__main__":
    print("Computing symbolic Jacobian decomposition...\n")
    w_res, ce_res, cp_res = full_log_det_jacobian_formula()
    print("\n\n")
    exprs = generate_stan_expressions()
