"""
Numerical analysis of the consistency function H(r) for the RDU-Prelec CE mapping.

Goal: Determine whether H(r) = 0 can have at most one solution, which would
establish global bijectivity of the mapping (r, phi, eta) -> (CE1, CE2, CE3).

Setup:
------
We have 3-outcome lotteries with prizes a < b < c, CRRA utility
u(x; r) = x^(1-r)/(1-r), and Prelec PWF w(p; phi, eta) = exp(-eta*(-ln p)^phi).

For fixed "true" parameters (r0, phi0, eta0), we compute three CEs. Then we ask:
given these CEs, is there any *other* r that is consistent? Specifically:

1. Fix CE1, CE2, CE3 from true parameters.
2. For each candidate r, define:
   - R(r) = Delta_23(r) / Delta_12(r) where Delta_ij = x_j^(1-r) - x_i^(1-r)
   - C_k(r) = [CE_k^(1-r) - a^(1-r)] / [b^(1-r) - a^(1-r)]
3. From lotteries 1 and 2, solve for (phi(r), eta(r)) satisfying:
   C_k(r) = w(s_k; phi, eta) + R(r) * w(q_k; phi, eta),  k = 1, 2
4. Evaluate the consistency function:
   H(r) = C_3(r) - w(s_3; phi(r), eta(r)) - R(r) * w(q_3; phi(r), eta(r))

H(r0) = 0 by construction. The question: does H have any other zeros?

Lotteries:
  L1 (binary):   probs (1/2, 0, 1/2) => q1=1/2, s1=1/2
  L2 (middle):   probs (1/10, 8/10, 1/10) => q2=1/10, s2=9/10
  L3 (skewed):   probs (3/10, 3/10, 4/10) => q3=4/10, s3=7/10
  Prizes: a=10, b=30, c=50

Note on cumulative probabilities:
  q_k = P(X >= c) = prob of best outcome
  s_k = P(X >= b) = prob of middle-or-better
  Decision weights: d3=w(q), d2=w(s)-w(q), d1=1-w(s)
"""

import numpy as np
from scipy.optimize import fsolve, root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# Primitives
# =====================================================================

def prelec_w(p, phi, eta):
    """Prelec 2-parameter PWF: w(p) = exp(-eta * (-ln p)^phi)."""
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0,1), got {p}")
    return np.exp(-eta * (-np.log(p))**phi)


def crra_utility(x, r):
    """CRRA utility: u(x;r) = x^(1-r)/(1-r) for r != 1, ln(x) for r = 1."""
    if abs(r - 1.0) < 1e-10:
        return np.log(x)
    return x**(1 - r) / (1 - r)


def crra_utility_raw(x, r):
    """Just x^(1-r), the unnormalized power. Works for r != 1."""
    if abs(r - 1.0) < 1e-10:
        return np.log(x)
    return x**(1 - r)


def compute_rdu(r, phi, eta, a, b, c, q, s):
    """
    Compute RDU value for a 3-outcome lottery.

    Decision weights:
      d3 = w(q)          [best outcome c]
      d2 = w(s) - w(q)   [middle outcome b]
      d1 = 1 - w(s)      [worst outcome a]

    RDU = d1*u(a) + d2*u(b) + d3*u(c)
    """
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)

    d3 = wq
    d2 = ws - wq
    d1 = 1.0 - ws

    return d1 * crra_utility(a, r) + d2 * crra_utility(b, r) + d3 * crra_utility(c, r)


def compute_ce(r, phi, eta, a, b, c, q, s):
    """
    Compute certainty equivalent: CE such that u(CE; r) = RDU.
    CE = u_inv(RDU; r).
    """
    rdu_val = compute_rdu(r, phi, eta, a, b, c, q, s)

    if abs(r - 1.0) < 1e-10:
        return np.exp(rdu_val)
    else:
        # u(x) = x^(1-r)/(1-r), so x = ((1-r)*u)^(1/(1-r))
        inner = (1 - r) * rdu_val
        if inner <= 0:
            return np.nan
        return inner**(1.0 / (1 - r))


# =====================================================================
# Key functions for the H(r) analysis
# =====================================================================

def R_ratio(r, a, b, c):
    """
    R(r) = Delta_23(r) / Delta_12(r)
    where Delta_12 = b^(1-r) - a^(1-r), Delta_23 = c^(1-r) - b^(1-r).

    For a < b < c and any r, both deltas are positive (since x^(1-r) is
    strictly increasing in x for any r), so R > 0.
    """
    s = 1 - r
    if abs(s) < 1e-12:
        # r ~ 1: use log form
        return np.log(c/b) / np.log(b/a)
    delta_12 = b**s - a**s
    delta_23 = c**s - b**s
    return delta_23 / delta_12


def C_k(r, CE_k, a, b):
    """
    C_k(r) = [CE_k^(1-r) - a^(1-r)] / [b^(1-r) - a^(1-r)]

    This measures "how far CE_k is between a and b" in the utility scale at r.
    """
    s = 1 - r
    if abs(s) < 1e-12:
        return np.log(CE_k / a) / np.log(b / a)
    return (CE_k**s - a**s) / (b**s - a**s)


def rhs_equation(phi, eta, q, s, R_val):
    """
    Right-hand side of the CE equation:
    RHS_k = w(s_k; phi, eta) + R(r) * w(q_k; phi, eta)

    This comes from expanding C_k(r) = [CE^s - a^s]/[b^s - a^s] and using
    the RDU structure:
      CE^s = (1-w(s))a^s + (w(s)-w(q))b^s + w(q)c^s
    Rearranging:
      [CE^s - a^s]/[b^s - a^s] = (w(s) - w(q)) + w(q) * [c^s - a^s]/[b^s - a^s]
                                = w(s) - w(q) + w(q) * (1 + R)
                                = w(s) + R * w(q)

    Wait, let me re-derive more carefully:
      CE^s - a^s = (w(s)-w(q))(b^s - a^s) + w(q)(c^s - a^s)
                 = (w(s)-w(q))(b^s - a^s) + w(q)[(b^s - a^s) + (c^s - b^s)]
                 = [w(s)-w(q)+w(q)](b^s - a^s) + w(q)(c^s - b^s)
                 = w(s)(b^s - a^s) + w(q)(c^s - b^s)
    So:
      C_k = w(s) + w(q) * R

    That's it: C_k(r) = w(s_k) + R(r) * w(q_k).
    """
    ws = prelec_w(s, phi, eta)
    wq = prelec_w(q, phi, eta)
    return ws + R_val * wq


def solve_prelec_params(r_val, CE1, CE2, a, b, c, q1, s1, q2, s2, phi_init=0.7, eta_init=0.9):
    """
    Given r, solve for (phi, eta) from the two CE equations:
      C_1(r) = w(s1; phi, eta) + R(r) * w(q1; phi, eta)
      C_2(r) = w(s2; phi, eta) + R(r) * w(q2; phi, eta)

    Returns (phi, eta) or (nan, nan) if solver fails.
    """
    R_val = R_ratio(r_val, a, b, c)
    C1_val = C_k(r_val, CE1, a, b)
    C2_val = C_k(r_val, CE2, a, b)

    def equations(params):
        phi_try, eta_try = params
        if phi_try <= 0.01 or eta_try <= 0.01:
            return [1e10, 1e10]
        try:
            rhs1 = rhs_equation(phi_try, eta_try, q1, s1, R_val)
            rhs2 = rhs_equation(phi_try, eta_try, q2, s2, R_val)
            return [C1_val - rhs1, C2_val - rhs2]
        except (ValueError, OverflowError, FloatingPointError):
            return [1e10, 1e10]

    # Try multiple initial guesses to improve robustness
    best_sol = None
    best_residual = np.inf

    init_guesses = [
        (phi_init, eta_init),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.3, 1.5),
        (1.5, 0.3),
        (0.8, 0.8),
        (0.6, 1.2),
        (1.2, 0.6),
    ]

    for phi0, eta0 in init_guesses:
        try:
            sol = fsolve(equations, [phi0, eta0], full_output=True)
            x, info, ier, msg = sol
            residual = np.max(np.abs(info['fvec']))
            if ier == 1 and x[0] > 0.01 and x[1] > 0.01 and residual < 1e-8:
                if residual < best_residual:
                    best_residual = residual
                    best_sol = x
        except Exception:
            continue

    if best_sol is not None:
        return best_sol[0], best_sol[1]
    return np.nan, np.nan


def compute_H(r_val, CEs, a, b, c, qs, ss, phi_init=0.7, eta_init=0.9):
    """
    Compute H(r) = C_3(r) - w(s3; phi(r), eta(r)) - R(r) * w(q3; phi(r), eta(r))

    where (phi(r), eta(r)) are determined from the first two CE equations.

    Parameters:
      CEs: (CE1, CE2, CE3)
      qs: (q1, q2, q3)
      ss: (s1, s2, s3)

    Returns: H value, phi(r), eta(r)
    """
    CE1, CE2, CE3 = CEs
    q1_v, q2_v, q3_v = qs
    s1_v, s2_v, s3_v = ss

    # Solve for phi(r), eta(r) from equations 1 and 2
    phi_r, eta_r = solve_prelec_params(
        r_val, CE1, CE2, a, b, c, q1_v, s1_v, q2_v, s2_v,
        phi_init=phi_init, eta_init=eta_init
    )

    if np.isnan(phi_r) or np.isnan(eta_r):
        return np.nan, np.nan, np.nan

    # Evaluate H(r) using the third equation
    R_val = R_ratio(r_val, a, b, c)
    C3_val = C_k(r_val, CE3, a, b)

    try:
        w_s3 = prelec_w(s3_v, phi_r, eta_r)
        w_q3 = prelec_w(q3_v, phi_r, eta_r)
        H_val = C3_val - w_s3 - R_val * w_q3
    except (ValueError, OverflowError):
        return np.nan, phi_r, eta_r

    return H_val, phi_r, eta_r


# =====================================================================
# Main analysis
# =====================================================================

def analyze_single_case(r0, phi0, eta0, a, b, c, qs, ss, label, r_range, ax_H, ax_phi, ax_eta):
    """
    For a single true parameter set, compute CEs, evaluate H(r), and plot.

    Returns: array of (r, H(r)) values.
    """
    q1_v, q2_v, q3_v = qs
    s1_v, s2_v, s3_v = ss

    # Step 1: Compute true CEs
    CE1 = compute_ce(r0, phi0, eta0, a, b, c, q1_v, s1_v)
    CE2 = compute_ce(r0, phi0, eta0, a, b, c, q2_v, s2_v)
    CE3 = compute_ce(r0, phi0, eta0, a, b, c, q3_v, s3_v)

    print(f"\n{'='*60}")
    print(f"Case: {label}")
    print(f"  True params: r={r0}, phi={phi0}, eta={eta0}")
    print(f"  CEs: CE1={CE1:.4f}, CE2={CE2:.4f}, CE3={CE3:.4f}")

    # Verify: at true r, the solved (phi, eta) should match truth
    phi_check, eta_check = solve_prelec_params(
        r0, CE1, CE2, a, b, c, q1_v, s1_v, q2_v, s2_v,
        phi_init=phi0, eta_init=eta0
    )
    print(f"  Verification at r0: phi={phi_check:.6f} (true={phi0}), eta={eta_check:.6f} (true={eta0})")

    H_check, _, _ = compute_H(r0, (CE1, CE2, CE3), a, b, c, qs, ss, phi_init=phi0, eta_init=eta0)
    print(f"  H(r0) = {H_check:.2e} (should be ~0)")

    # Step 2: Evaluate H(r) over a range of r values
    r_vals = r_range
    H_vals = np.full_like(r_vals, np.nan)
    phi_vals = np.full_like(r_vals, np.nan)
    eta_vals = np.full_like(r_vals, np.nan)

    # Use the true params as starting guess, and propagate from there
    phi_prev, eta_prev = phi0, eta0

    # Find index closest to r0 and work outward from there
    idx_r0 = np.argmin(np.abs(r_vals - r0))

    # Forward from r0
    phi_prev, eta_prev = phi0, eta0
    for i in range(idx_r0, len(r_vals)):
        H_v, phi_v, eta_v = compute_H(
            r_vals[i], (CE1, CE2, CE3), a, b, c, qs, ss,
            phi_init=phi_prev, eta_init=eta_prev
        )
        H_vals[i] = H_v
        phi_vals[i] = phi_v
        eta_vals[i] = eta_v
        if not np.isnan(phi_v):
            phi_prev, eta_prev = phi_v, eta_v

    # Backward from r0
    phi_prev, eta_prev = phi0, eta0
    for i in range(idx_r0 - 1, -1, -1):
        H_v, phi_v, eta_v = compute_H(
            r_vals[i], (CE1, CE2, CE3), a, b, c, qs, ss,
            phi_init=phi_prev, eta_init=eta_prev
        )
        H_vals[i] = H_v
        phi_vals[i] = phi_v
        eta_vals[i] = eta_v
        if not np.isnan(phi_v):
            phi_prev, eta_prev = phi_v, eta_v

    # Step 3: Analyze results
    valid = ~np.isnan(H_vals)
    n_valid = np.sum(valid)

    # Count zero crossings
    H_valid = H_vals[valid]
    r_valid = r_vals[valid]
    sign_changes = np.sum(np.diff(np.sign(H_valid)) != 0)

    # Find approximate zeros (sign changes)
    zeros = []
    for i in range(len(H_valid) - 1):
        if H_valid[i] * H_valid[i+1] < 0:
            # Linear interpolation
            r_zero = r_valid[i] - H_valid[i] * (r_valid[i+1] - r_valid[i]) / (H_valid[i+1] - H_valid[i])
            zeros.append(r_zero)

    print(f"\n  Valid evaluations: {n_valid}/{len(r_vals)}")
    print(f"  Sign changes: {sign_changes}")
    print(f"  Approximate zeros: {zeros}")
    print(f"  True r0 = {r0}")

    if len(zeros) == 0:
        # H might not cross zero in the range, or stays very close to 0
        # Check if H is very small everywhere (meaning r0 is at boundary)
        max_abs_H = np.nanmax(np.abs(H_vals))
        print(f"  Max |H|: {max_abs_H:.6e}")
        if max_abs_H < 1e-8:
            print(f"  H is essentially zero everywhere -- degenerate case")
    elif len(zeros) == 1:
        print(f"  SINGLE zero at r ~ {zeros[0]:.6f} (true r0 = {r0})")
        print(f"  ==> CONSISTENT with global bijectivity")
    else:
        print(f"  MULTIPLE zeros: {[f'{z:.6f}' for z in zeros]}")
        print(f"  ==> POTENTIAL non-injectivity!")

    # Check monotonicity
    dH = np.diff(H_valid)
    if np.all(dH > 0):
        print(f"  H(r) is strictly INCREASING over valid range")
    elif np.all(dH < 0):
        print(f"  H(r) is strictly DECREASING over valid range")
    else:
        n_inc = np.sum(dH > 0)
        n_dec = np.sum(dH < 0)
        print(f"  H(r) is NOT monotone: {n_inc} increasing, {n_dec} decreasing steps")

    # Step 4: Plot
    ax_H.plot(r_valid, H_valid, label=label, linewidth=1.5)
    ax_H.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax_H.axvline(x=r0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax_phi.plot(r_valid, phi_vals[valid], label=label, linewidth=1.5)
    ax_phi.axvline(x=r0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax_eta.plot(r_valid, eta_vals[valid], label=label, linewidth=1.5)
    ax_eta.axvline(x=r0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    return r_vals, H_vals, phi_vals, eta_vals, zeros


def main():
    print("=" * 70)
    print("ANALYSIS OF H(r) MONOTONICITY FOR RDU-PRELEC CE MAPPING")
    print("=" * 70)

    # Lottery configuration
    a, b, c = 10.0, 30.0, 50.0

    # Cumulative probabilities: q_k = P(X >= c), s_k = P(X >= b)
    # L1 (binary): probs (1/2, 0, 1/2) => q1 = 1/2, s1 = 1/2
    # L2 (middle): probs (1/10, 8/10, 1/10) => q2 = 1/10, s2 = 9/10
    # L3 (skewed): probs (3/10, 3/10, 4/10) => q3 = 4/10, s3 = 7/10
    q1, s1 = 0.5, 0.5
    q2, s2 = 0.1, 0.9
    q3, s3 = 0.4, 0.7

    qs = (q1, q2, q3)
    ss = (s1, s2, s3)

    print(f"\nLotteries on prizes ({a}, {b}, {c}):")
    print(f"  L1 (binary):  probs (1/2, 0, 1/2),     q1={q1}, s1={s1}")
    print(f"  L2 (middle):  probs (1/10, 8/10, 1/10), q2={q2}, s2={s2}")
    print(f"  L3 (skewed):  probs (3/10, 3/10, 4/10), q3={q3}, s3={s3}")

    # True parameter sets
    true_params = [
        (0.5, 0.7, 0.9, "r=0.5, phi=0.7, eta=0.9"),
        (0.3, 1.2, 0.5, "r=0.3, phi=1.2, eta=0.5"),
        (0.8, 0.5, 1.3, "r=0.8, phi=0.5, eta=1.3"),
        (0.1, 0.9, 0.8, "r=0.1, phi=0.9, eta=0.8"),
        (1.5, 0.6, 1.0, "r=1.5, phi=0.6, eta=1.0"),
    ]

    # r range for evaluation
    r_range = np.linspace(0.01, 2.5, 500)

    # Create figure with 3 panels: H(r), phi(r), eta(r)
    fig, (ax_H, ax_phi, ax_eta) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    all_results = {}

    for r0, phi0, eta0, label in true_params:
        try:
            result = analyze_single_case(
                r0, phi0, eta0, a, b, c, qs, ss, label, r_range,
                ax_H, ax_phi, ax_eta
            )
            all_results[label] = result
        except Exception as e:
            print(f"\n  ERROR in case {label}: {e}")
            import traceback
            traceback.print_exc()

    # Format plots
    ax_H.set_ylabel('H(r)', fontsize=12)
    ax_H.set_title('Consistency function H(r) for different true parameter sets', fontsize=13)
    ax_H.legend(fontsize=9, loc='best')
    ax_H.grid(True, alpha=0.3)
    ax_H.axhline(y=0, color='k', linewidth=0.8)

    ax_phi.set_ylabel('phi(r)', fontsize=12)
    ax_phi.set_title('Implied phi(r) from equations 1 & 2', fontsize=13)
    ax_phi.legend(fontsize=9, loc='best')
    ax_phi.grid(True, alpha=0.3)

    ax_eta.set_ylabel('eta(r)', fontsize=12)
    ax_eta.set_xlabel('r', fontsize=12)
    ax_eta.set_title('Implied eta(r) from equations 1 & 2', fontsize=13)
    ax_eta.legend(fontsize=9, loc='best')
    ax_eta.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/sympy/H_monotonicity.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to /workspace/sympy/H_monotonicity.png")

    # =====================================================================
    # Additional analysis: derivative of H at the true r0
    # =====================================================================
    print("\n" + "=" * 70)
    print("DERIVATIVE OF H AT TRUE r0 (finite difference)")
    print("=" * 70)

    eps = 1e-5
    for r0, phi0, eta0, label in true_params:
        CE1 = compute_ce(r0, phi0, eta0, a, b, c, q1, s1)
        CE2 = compute_ce(r0, phi0, eta0, a, b, c, q2, s2)
        CE3 = compute_ce(r0, phi0, eta0, a, b, c, q3, s3)
        CEs = (CE1, CE2, CE3)

        H_plus, _, _ = compute_H(r0 + eps, CEs, a, b, c, qs, ss, phi_init=phi0, eta_init=eta0)
        H_minus, _, _ = compute_H(r0 - eps, CEs, a, b, c, qs, ss, phi_init=phi0, eta_init=eta0)

        if not np.isnan(H_plus) and not np.isnan(H_minus):
            dH_dr = (H_plus - H_minus) / (2 * eps)
            print(f"  {label}: H'(r0) = {dH_dr:.6e}")
        else:
            print(f"  {label}: could not compute derivative")

    # =====================================================================
    # Detailed behavior near boundaries
    # =====================================================================
    print("\n" + "=" * 70)
    print("BEHAVIOR NEAR BOUNDARIES")
    print("=" * 70)

    for r0, phi0, eta0, label in true_params:
        print(f"\n  Case: {label}")
        CE1 = compute_ce(r0, phi0, eta0, a, b, c, q1, s1)
        CE2 = compute_ce(r0, phi0, eta0, a, b, c, q2, s2)
        CE3 = compute_ce(r0, phi0, eta0, a, b, c, q3, s3)
        CEs = (CE1, CE2, CE3)

        # R(r) behavior
        for r_test in [0.01, 0.5, 1.0, 1.5, 2.0, 2.5]:
            R_val = R_ratio(r_test, a, b, c)
            C1 = C_k(r_test, CE1, a, b)
            C2 = C_k(r_test, CE2, a, b)
            C3 = C_k(r_test, CE3, a, b)
            print(f"    r={r_test:.2f}: R={R_val:.4f}, C1={C1:.4f}, C2={C2:.4f}, C3={C3:.4f}")

    # =====================================================================
    # Extra: very fine grid near true r0 to confirm zero
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINE GRID NEAR TRUE r0")
    print("=" * 70)

    for r0, phi0, eta0, label in true_params:
        CE1 = compute_ce(r0, phi0, eta0, a, b, c, q1, s1)
        CE2 = compute_ce(r0, phi0, eta0, a, b, c, q2, s2)
        CE3 = compute_ce(r0, phi0, eta0, a, b, c, q3, s3)
        CEs = (CE1, CE2, CE3)

        r_fine = np.linspace(max(0.01, r0 - 0.3), r0 + 0.3, 61)
        H_fine = []
        for rv in r_fine:
            Hv, _, _ = compute_H(rv, CEs, a, b, c, qs, ss, phi_init=phi0, eta_init=eta0)
            H_fine.append(Hv)
        H_fine = np.array(H_fine)

        valid = ~np.isnan(H_fine)
        if np.any(valid):
            H_v = H_fine[valid]
            r_v = r_fine[valid]
            sign_changes = np.sum(np.diff(np.sign(H_v)) != 0)
            print(f"  {label}: {sign_changes} sign changes in [{r_v[0]:.2f}, {r_v[-1]:.2f}]")

            # Find zeros
            for i in range(len(H_v) - 1):
                if H_v[i] * H_v[i+1] < 0:
                    rz = r_v[i] - H_v[i] * (r_v[i+1] - r_v[i]) / (H_v[i+1] - H_v[i])
                    print(f"    Zero near r = {rz:.6f}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_cases = len(true_params)
    single_zero_cases = 0
    multi_zero_cases = 0

    for label, result in all_results.items():
        r_vals, H_vals, phi_vals, eta_vals, zeros = result
        if len(zeros) <= 1:
            single_zero_cases += 1
        else:
            multi_zero_cases += 1

    print(f"\n  Total parameter cases tested: {total_cases}")
    print(f"  Cases with at most 1 zero: {single_zero_cases}")
    print(f"  Cases with multiple zeros: {multi_zero_cases}")

    if multi_zero_cases == 0:
        print("\n  RESULT: H(r) appears to have at most one zero in all tested cases.")
        print("  This is consistent with GLOBAL BIJECTIVITY of the CE mapping.")
    else:
        print("\n  WARNING: Some cases show multiple zeros!")
        print("  This would indicate the CE mapping is NOT globally injective.")

    print("\n  Note: This is numerical evidence, not a proof. The analysis covers")
    print("  r in [0.01, 2.5] with 500 grid points per case.")


if __name__ == '__main__':
    main()
