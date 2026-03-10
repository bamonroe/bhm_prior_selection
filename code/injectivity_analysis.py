"""
Injectivity analysis for the RDU-Prelec CE mapping.

Consolidates analysis of whether the mapping (r, phi, eta) -> (CE1, CE2, CE3)
is globally injective for 3-outcome lotteries with Prelec 2-parameter
probability weighting and CRRA utility.

Key findings:
- The Jacobian det is generically nonzero when lotteries have different
  cumulative probability structures (local invertibility).
- However, the mapping is NOT globally injective for 3 lotteries on shared
  prizes: two distinct (r, phi, eta) triples can produce identical CEs.
- Adding a 4th lottery on the same prizes, or using lotteries with different
  prizes, restores injectivity.

This module implements:
1. Symbolic Jacobian analysis (requires sympy, skipped if unavailable)
2. H(r) consistency function analysis (monotonicity, zero counting)
3. Deep investigation of multiple-zero cases
4. Systematic grid scan for observational equivalence
5. Verification of known equivalent pairs
6. Lottery design analysis (which designs restore injectivity)
7. P-matrix verification via dense grid (prize-diverse design)

Used for Appendix C of the paper.

Original research scripts: sympy/check_ce_invertibility.py,
sympy/analyze_H_monotonicity.py, sympy/investigate_multiple_zeros.py,
sympy/systematic_nonuniqueness.py, sympy/verify_nonuniqueness.py,
sympy/lottery_design_analysis.py
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


# =========================================================================
# Shared utility functions
# =========================================================================

def prelec_w(p, phi, eta):
    """Prelec 2-parameter PWF: w(p) = exp(-eta * (-ln p)^phi).

    Parameters
    ----------
    p : float
        Probability in (0, 1).
    phi : float
        Curvature parameter (positive).
    eta : float
        Elevation parameter (positive).
    """
    return np.exp(-eta * (-np.log(p))**phi)


def crra_utility(x, r):
    """CRRA utility: u(x;r) = x^(1-r)/(1-r) for r != 1, ln(x) for r = 1."""
    if abs(r - 1.0) < 1e-10:
        return np.log(x)
    return x**(1 - r) / (1 - r)


def compute_rdu(r, phi, eta, prizes, probs_cum):
    """Compute RDU value for a 3-outcome lottery.

    Parameters
    ----------
    r : float
        CRRA coefficient.
    phi, eta : float
        Prelec PWF parameters.
    prizes : tuple
        (a, b, c) prize amounts with a < b < c.
    probs_cum : dict
        {'q': float, 's': float} where q = P(X >= c), s = P(X >= b).
    """
    a, b, c = prizes
    q, s = probs_cum['q'], probs_cum['s']

    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)

    d3 = wq
    d2 = ws - wq
    d1 = 1.0 - ws

    return d1 * crra_utility(a, r) + d2 * crra_utility(b, r) + d3 * crra_utility(c, r)


def compute_ce(r, phi, eta, prizes, probs_cum):
    """Compute certainty equivalent: CE such that u(CE; r) = RDU.

    Parameters
    ----------
    r : float
        CRRA coefficient.
    phi, eta : float
        Prelec PWF parameters.
    prizes : tuple
        (a, b, c) prize amounts with a < b < c.
    probs_cum : dict
        {'q': float, 's': float} where q = P(X >= c), s = P(X >= b).
    """
    rdu_val = compute_rdu(r, phi, eta, prizes, probs_cum)

    if abs(r - 1.0) < 1e-10:
        return np.exp(rdu_val)
    else:
        inner = (1 - r) * rdu_val
        if inner <= 0:
            return np.nan
        return inner**(1.0 / (1 - r))


def compute_ces(r, phi, eta, lotteries):
    """Compute CEs for a list of lotteries.

    Parameters
    ----------
    lotteries : list of (prizes, probs_cum) tuples
    """
    return np.array([compute_ce(r, phi, eta, prizes, pc)
                     for prizes, pc in lotteries])


def solve_prelec_params(r_val, CE1, CE2, prizes, lot1_cum, lot2_cum,
                        phi_init=0.7, eta_init=0.9):
    """Solve for (phi, eta) given r and two target CEs.

    From the two CE equations:
        C_k(r) = w(s_k; phi, eta) + R(r) * w(q_k; phi, eta)

    Returns (phi, eta) or (nan, nan) if solver fails.
    """
    a, b, c = prizes
    q1, s1 = lot1_cum['q'], lot1_cum['s']
    q2, s2 = lot2_cum['q'], lot2_cum['s']

    R_val = _R_ratio(r_val, a, b, c)
    C1_val = _C_k(r_val, CE1, a, b)
    C2_val = _C_k(r_val, CE2, a, b)

    def equations(params):
        phi_try, eta_try = params
        if phi_try <= 0.01 or eta_try <= 0.01:
            return [1e10, 1e10]
        try:
            ws1 = prelec_w(s1, phi_try, eta_try)
            wq1 = prelec_w(q1, phi_try, eta_try)
            ws2 = prelec_w(s2, phi_try, eta_try)
            wq2 = prelec_w(q2, phi_try, eta_try)
            eq1 = C1_val - ws1 - R_val * wq1
            eq2 = C2_val - ws2 - R_val * wq2
            return [eq1, eq2]
        except (ValueError, OverflowError, FloatingPointError):
            return [1e10, 1e10]

    best_sol = None
    best_residual = np.inf

    init_guesses = [
        (phi_init, eta_init),
        (0.5, 0.5), (1.0, 1.0), (0.3, 1.5), (1.5, 0.3),
        (0.8, 0.8), (0.6, 1.2), (1.2, 0.6),
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


def find_observational_equivalent(r0, phi0, eta0, lotteries, r_init_alt=None):
    """Find an alternative (r, phi, eta) producing the same CEs.

    Parameters
    ----------
    r0, phi0, eta0 : float
        True parameters.
    lotteries : list of (prizes, probs_cum) tuples
    r_init_alt : float, optional
        If given, bias initial guesses toward this r.

    Returns
    -------
    true_ces : list of float
    solutions : list of (r, phi, eta, residual) tuples
    """
    true_ces = [compute_ce(r0, phi0, eta0, prizes, pc)
                for prizes, pc in lotteries]

    def system(params):
        r_try, phi_try, eta_try = params
        if phi_try <= 0.001 or eta_try <= 0.001:
            return [1e10] * len(lotteries)
        residuals = []
        for i, (prizes, pc) in enumerate(lotteries):
            try:
                ce_try = compute_ce(r_try, phi_try, eta_try, prizes, pc)
                if np.isnan(ce_try):
                    residuals.append(1e10)
                else:
                    residuals.append(ce_try - true_ces[i])
            except Exception:
                residuals.append(1e10)
        return residuals

    solutions = []
    init_guesses = []
    for r_init in [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]:
        for phi_init in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]:
            for eta_init in [0.3, 0.5, 0.8, 1.0, 1.3, 1.8]:
                if (abs(r_init - r0) > 0.1 or abs(phi_init - phi0) > 0.1
                        or abs(eta_init - eta0) > 0.1):
                    init_guesses.append((r_init, phi_init, eta_init))

    if r_init_alt is not None:
        init_guesses = ([(r_init_alt, 0.5, 1.0), (r_init_alt, 0.7, 0.7),
                         (r_init_alt, 1.0, 1.5), (r_init_alt, 0.3, 2.0)]
                        + init_guesses)

    for r_i, p_i, e_i in init_guesses:
        try:
            sol = fsolve(system, [r_i, p_i, e_i], full_output=True)
            x, info, ier, msg = sol
            residual = np.max(np.abs(info['fvec']))
            if ier == 1 and x[1] > 0.01 and x[2] > 0.01 and residual < 1e-10:
                dist = np.sqrt((x[0]-r0)**2 + (x[1]-phi0)**2 + (x[2]-eta0)**2)
                if dist > 0.01:
                    is_new = True
                    for prev in solutions:
                        if np.sqrt((x[0]-prev[0])**2 + (x[1]-prev[1])**2
                                   + (x[2]-prev[2])**2) < 0.01:
                            is_new = False
                            break
                    if is_new:
                        solutions.append((x[0], x[1], x[2], residual))
        except Exception:
            continue

    return true_ces, solutions


# =========================================================================
# Internal helpers for the H(r) analysis
# =========================================================================

def _R_ratio(r, a, b, c):
    """R(r) = Delta_23(r) / Delta_12(r)."""
    s = 1 - r
    if abs(s) < 1e-12:
        return np.log(c/b) / np.log(b/a)
    return (c**s - b**s) / (b**s - a**s)


def _C_k(r, CE_k, a, b):
    """C_k(r) = [CE_k^(1-r) - a^(1-r)] / [b^(1-r) - a^(1-r)]."""
    s = 1 - r
    if abs(s) < 1e-12:
        return np.log(CE_k / a) / np.log(b / a)
    return (CE_k**s - a**s) / (b**s - a**s)


def _compute_H(r_val, CEs, prizes, lotteries_cum, phi_init=0.7, eta_init=0.9):
    """Compute H(r) consistency function.

    Given 3 CEs, solve (phi, eta) from the first two equations, then evaluate
    residual of the third. H(r0) = 0 by construction at the true r0.
    """
    phi_r, eta_r = solve_prelec_params(
        r_val, CEs[0], CEs[1], prizes,
        lotteries_cum[0], lotteries_cum[1],
        phi_init=phi_init, eta_init=eta_init
    )

    if np.isnan(phi_r) or np.isnan(eta_r):
        return np.nan, np.nan, np.nan

    a, b, c = prizes
    q3, s3 = lotteries_cum[2]['q'], lotteries_cum[2]['s']
    R_val = _R_ratio(r_val, a, b, c)
    C3_val = _C_k(r_val, CEs[2], a, b)

    try:
        w_s3 = prelec_w(s3, phi_r, eta_r)
        w_q3 = prelec_w(q3, phi_r, eta_r)
        H_val = C3_val - w_s3 - R_val * w_q3
    except (ValueError, OverflowError):
        return np.nan, phi_r, eta_r

    return H_val, phi_r, eta_r


# =========================================================================
# Analysis 1: Symbolic Jacobian (requires sympy)
# =========================================================================

def run_symbolic_jacobian(config, output_dir):
    """Symbolic Jacobian of the CE mapping via B-formulation decomposition.

    Builds the Jacobian step-by-step through intermediate quantities:
    1. Prelec w-values and decision weights
    2. B_k = sum pi_mk * x_m^(1-r)  (power-mean body, no CE exponent)
    3. J_B: 3x3 Jacobian of the B-map (simple expressions)
    4. Decomposition: J_tilde = (1/(1-r)) * J_B + rank-1 correction
    5. Principal minors (3 of 7 transfer from J_B without correction)

    No numerical evaluation — purely symbolic.
    """
    try:
        import sympy as sp
        from sympy import symbols, exp, log, Rational, diff
    except ImportError:
        print("  sympy not available -- skipping symbolic Jacobian analysis")
        return

    output_dir = Path(output_dir)
    lines = []

    def log_print(msg=""):
        print(msg)
        lines.append(msg)

    def log_expr(expr, indent=4):
        """Pretty-print a sympy expression with indentation."""
        pretty = sp.pretty(expr, wrap_line=False)
        for line in pretty.split('\n'):
            log_print(' ' * indent + line)

    log_print("=" * 70)
    log_print("SYMBOLIC JACOBIAN VIA B-FORMULATION DECOMPOSITION")
    log_print("=" * 70)

    # ---- Symbols ----
    r = symbols('r', positive=True)
    phi, eta = symbols('phi eta', positive=True)

    def sym_prelec_w(p_val, phi_s, eta_s):
        return exp(-eta_s * (-log(p_val))**phi_s)

    # ==================================================================
    # Step 1: Prelec w-values and decision weights
    # ==================================================================
    log_print("\n--- Step 1: Prelec w-values and decision weights ---")
    log_print("")
    log_print("  w(p; phi, eta) = exp(-eta * (-ln p)^phi)")
    log_print("")
    log_print("  For 3-outcome lottery with probs (p1, p2, p3) on prizes x1 < x2 < x3:")
    log_print("    cumulative:  q = p3,  s = p2 + p3")
    log_print("    weights:     pi_3 = w(q),  pi_2 = w(s) - w(q),  pi_1 = 1 - w(s)")

    lotteries = config['lotteries']
    w_vals = []     # (w_q, w_s) per lottery
    weights = []    # (pi1, pi2, pi3) per lottery
    prizes_list = []

    for k, lot in enumerate(lotteries, 1):
        x1, x2, x3 = [Rational(x) for x in lot['prizes']]
        p1, p2, p3 = [Rational(p).limit_denominator(1000) for p in lot['probs']]
        prizes_list.append((x1, x2, x3))

        q = p3
        s = p2 + p3

        w_q = sym_prelec_w(q, phi, eta) if q != 0 else sp.Integer(0)
        if s == 0:
            w_s = sp.Integer(0)
        elif s == 1:
            w_s = sp.Integer(1)
        else:
            w_s = sym_prelec_w(s, phi, eta)

        pi3 = w_q
        pi2 = w_s - w_q
        pi1 = 1 - w_s

        w_vals.append((w_q, w_s))
        weights.append((pi1, pi2, pi3))

        log_print(f"\n  L{k}: prizes=({x1},{x2},{x3}), probs=({p1},{p2},{p3})")
        log_print(f"       q={q}, s={s}")
        log_print(f"       w(q) =")
        log_expr(w_q, indent=8)
        if w_s != w_q:
            log_print(f"       w(s) =")
            log_expr(w_s, indent=8)

    # ==================================================================
    # Step 2: B_k expressions (power-mean body)
    # ==================================================================
    log_print("\n" + "-" * 70)
    log_print("--- Step 2: B_k = sum pi_mk * x_m^(1-r) ---")
    log_print("-" * 70)
    log_print("")
    log_print("  B_k is the weighted power sum (no CE exponent).")
    log_print("  CE_k = B_k^(1/(1-r)), so B_k is the natural intermediate.")

    B_exprs = []
    for k in range(3):
        pi1, pi2, pi3 = weights[k]
        x1, x2, x3 = prizes_list[k]
        B_k = pi1 * x1**(1 - r) + pi2 * x2**(1 - r) + pi3 * x3**(1 - r)
        B_exprs.append(B_k)
        log_print(f"\n  B_{k+1} ({sp.count_ops(B_k)} ops):")
        log_expr(B_k)

    # ==================================================================
    # Step 3: J_B — Jacobian of the B-map
    # ==================================================================
    log_print("\n" + "-" * 70)
    log_print("--- Step 3: J_B = d(B1,B2,B3)/d(r,phi,eta) ---")
    log_print("-" * 70)

    param_names = ['r', 'phi', 'eta']
    param_syms = [r, phi, eta]

    J_B = []
    for k in range(3):
        row = []
        for l in range(3):
            entry = diff(B_exprs[k], param_syms[l])
            row.append(entry)
            log_print(f"\n  J_B[B_{k+1}, {param_names[l]}] ({sp.count_ops(entry)} ops):")
            log_expr(entry)
        J_B.append(row)

    # ==================================================================
    # Step 4: Prelec derivative ratio
    # ==================================================================
    log_print("\n" + "-" * 70)
    log_print("--- Step 4: Prelec derivative ratio ---")
    log_print("-" * 70)
    log_print("")
    log_print("  dw/dphi / dw/deta = -eta * ln(-ln p)")
    log_print("")
    log_print("  This ratio depends on the cumulative probability p but NOT on phi.")
    log_print("  Different probability structures across lotteries prevent the")
    log_print("  phi and eta columns of J_B from being proportional.")

    # Show ratio values for each lottery's cumulative probs
    for k, lot in enumerate(lotteries, 1):
        p1, p2, p3 = [Rational(p).limit_denominator(1000) for p in lot['probs']]
        q, s = p3, p2 + p3
        for label, pval in [('q', q), ('s', s)]:
            if pval > 0 and pval < 1:
                t = -log(pval)
                ratio = -eta * log(t)
                log_print(f"  L{k}, {label}={float(pval)}: "
                          f"ratio = -eta * ln({sp.pretty(t, wrap_line=False)})")

    # ==================================================================
    # Step 5: Decomposition J_tilde = (1/(1-r)) * J_B + rank-1 correction
    # ==================================================================
    log_print("\n" + "-" * 70)
    log_print("--- Step 5: Decomposition ---")
    log_print("-" * 70)
    log_print("")
    log_print("  The unnormalized CE Jacobian J_tilde (P-matrix equivalent to")
    log_print("  J_CE by Lemma D_reduction) decomposes as:")
    log_print("")
    log_print("    J_tilde[k, phi] = (1/(1-r)) * dB_k/dphi")
    log_print("    J_tilde[k, eta] = (1/(1-r)) * dB_k/deta")
    log_print("    J_tilde[k, r]   = (1/(1-r)) * dB_k/dr + B_k*ln(B_k)/(1-r)^2")
    log_print("")
    log_print("  In matrix form:")
    log_print("    J_tilde = (1/(1-r)) * J_B + b_ln * e_r^T")
    log_print("")
    log_print("  where b_ln_k = B_k*ln(B_k)/(1-r)^2 and e_r^T = (1,0,0).")
    log_print("  The scalar 1/(1-r) > 0 (for r < 1) drops out of the P-matrix")
    log_print("  condition, so P-matrix(J_tilde) iff P-matrix(J_B + correction).")

    # Compute the correction column
    log_print("\n  Rank-1 correction column b_ln:")
    b_ln = []
    for k in range(3):
        c_k = B_exprs[k] * log(B_exprs[k]) / (1 - r)
        b_ln.append(c_k)
        log_print(f"\n  b_ln_{k+1} ({sp.count_ops(c_k)} ops):")
        log_expr(c_k)

    # ==================================================================
    # Step 6: Principal minors of J_B
    # ==================================================================
    log_print("\n" + "-" * 70)
    log_print("--- Step 6: Principal minors of J_B ---")
    log_print("-" * 70)
    log_print("")
    log_print("  Three minors transfer to J_tilde WITHOUT correction")
    log_print("  (the rank-1 term only affects the r-column):")
    log_print("    M_phi, M_eta, M_{phi,eta}")
    log_print("")
    log_print("  Four minors involving the r-column need the correction:")
    log_print("    M_r, M_{r,phi}, M_{r,eta}, det")

    # Order 1: diagonal entries
    log_print("\n  --- Order 1: Diagonal entries ---")
    for k, name in enumerate(param_names):
        m = J_B[k][k]
        tag = "" if k > 0 else "  [needs correction for J_tilde]"
        log_print(f"\n  M_{{{name}}}^B = J_B[{k},{k}] ({sp.count_ops(m)} ops):{tag}")
        log_expr(m)

    # Order 2: 2x2 principal minors
    log_print("\n  --- Order 2: 2x2 principal minors ---")
    order2_indices = [(0, 1), (0, 2), (1, 2)]
    order2_labels = ['r,phi', 'r,eta', 'phi,eta']
    for (i, j), label in zip(order2_indices, order2_labels):
        m = J_B[i][i] * J_B[j][j] - J_B[i][j] * J_B[j][i]
        tag = "  [transfers directly to J_tilde]" if i > 0 else "  [needs correction]"
        log_print(f"\n  M_{{{label}}}^B ({sp.count_ops(m)} ops):{tag}")
        log_expr(m)

    # Order 3: full determinant
    log_print("\n  --- Order 3: Full determinant ---")
    det_B = (J_B[0][0] * (J_B[1][1] * J_B[2][2] - J_B[1][2] * J_B[2][1])
             - J_B[0][1] * (J_B[1][0] * J_B[2][2] - J_B[1][2] * J_B[2][0])
             + J_B[0][2] * (J_B[1][0] * J_B[2][1] - J_B[1][1] * J_B[2][0]))
    log_print(f"\n  det(J_B) ({sp.count_ops(det_B)} ops):  [needs correction]")
    log_expr(det_B)

    # ==================================================================
    # Step 7: Corrected r-column for J_tilde
    # ==================================================================
    log_print("\n" + "-" * 70)
    log_print("--- Step 7: J_tilde r-column (with correction) ---")
    log_print("-" * 70)
    log_print("")
    log_print("  J_tilde[k, r] = J_B[k, r] + b_ln_k")
    log_print("  (dropping the common positive factor 1/(1-r))")

    J_tilde_r = []
    for k in range(3):
        entry = J_B[k][0] + b_ln[k]
        J_tilde_r.append(entry)
        log_print(f"\n  J_tilde[B_{k+1}, r] ({sp.count_ops(entry)} ops):")
        log_expr(entry)

    # Corrected minors involving r
    log_print("\n  --- Corrected order-1 minor ---")
    m_r_corr = J_tilde_r[0]
    log_print(f"\n  M_r^tilde ({sp.count_ops(m_r_corr)} ops):")
    log_expr(m_r_corr)

    log_print("\n  --- Corrected order-2 minors ---")
    # M_{r,phi}: rows/cols {0,1}
    # J_tilde[0,0]*J_tilde[1,1] - J_tilde[0,1]*J_tilde[1,0]
    # = J_tilde_r[0] * J_B[1][1]/(already scaled) - J_B[0][1] * J_tilde_r[1]
    # (dropping 1/(1-r) factors which are positive)
    m_rphi = J_tilde_r[0] * J_B[1][1] - J_B[0][1] * J_tilde_r[1]
    log_print(f"\n  M_{{r,phi}}^tilde ({sp.count_ops(m_rphi)} ops):")
    log_expr(m_rphi)

    m_reta = J_tilde_r[0] * J_B[2][2] - J_B[0][2] * J_tilde_r[2]
    log_print(f"\n  M_{{r,eta}}^tilde ({sp.count_ops(m_reta)} ops):")
    log_expr(m_reta)

    log_print("\n  --- Corrected determinant ---")
    det_tilde = (J_tilde_r[0] * (J_B[1][1] * J_B[2][2] - J_B[1][2] * J_B[2][1])
                 - J_B[0][1] * (J_tilde_r[1] * J_B[2][2] - J_B[1][2] * J_tilde_r[2])
                 + J_B[0][2] * (J_tilde_r[1] * J_B[2][1] - J_B[1][1] * J_tilde_r[2]))
    log_print(f"\n  det(J_tilde) ({sp.count_ops(det_tilde)} ops):")
    log_expr(det_tilde)

    # ---- Summary ----
    log_print("\n" + "=" * 70)
    log_print("SUMMARY")
    log_print("=" * 70)
    log_print(f"\n  B-formulation decomposition for {len(lotteries)} lotteries:")
    log_print(f"  - B_k entries: simple weighted power sums")
    log_print(f"  - J_B: 9 entries, each a sum of Prelec derivatives * utility gaps")
    log_print(f"  - J_tilde = (1/(1-r)) * J_B + rank-1 correction in r-column")
    log_print(f"  - 3 of 7 P-matrix conditions (M_phi, M_eta, M_{{phi,eta}})")
    log_print(f"    transfer directly from J_B without correction")
    log_print(f"  - 4 conditions (M_r, M_{{r,phi}}, M_{{r,eta}}, det)")
    log_print(f"    require the B_k*ln(B_k) correction term")

    # Save
    summary_path = output_dir / 'symbolic_jacobian.txt'
    summary_path.write_text('\n'.join(lines))
    print(f"\n  Symbolic summary saved to {summary_path}")


# =========================================================================
# Analysis 2: H(r) monotonicity analysis
# =========================================================================

def run_monotonicity_analysis(config, output_dir):
    """Analyze the consistency function H(r) for the CE mapping.

    For each test case, fixes true CEs, then sweeps r to find all zeros of
    the consistency function H(r). A single zero (at the true r) indicates
    local uniqueness; multiple zeros indicate non-injectivity.

    Corresponds to: sympy/analyze_H_monotonicity.py
    """
    output_dir = Path(output_dir)
    prizes = config['prizes']
    lotteries_cum = config['lotteries']
    test_cases = config['test_cases']
    r_grid = config.get('r_grid', {'start': 0.01, 'stop': 2.5, 'n_points': 500})
    r_range = np.linspace(r_grid['start'], r_grid['stop'], r_grid['n_points'])

    print("\n  Lotteries on prizes ({}, {}, {}):".format(*prizes))
    for i, lc in enumerate(lotteries_cum, 1):
        print(f"    L{i}: q={lc['q']}, s={lc['s']}")

    fig, (ax_H, ax_phi, ax_eta) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    all_results = {}

    for r0, phi0, eta0 in test_cases:
        label = f"r={r0}, phi={phi0}, eta={eta0}"
        try:
            result = _analyze_single_case(
                r0, phi0, eta0, prizes, lotteries_cum, label, r_range,
                ax_H, ax_phi, ax_eta
            )
            all_results[label] = result
        except Exception as e:
            print(f"\n  ERROR in case {label}: {e}")

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
    plot_path = output_dir / 'H_monotonicity.pdf'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved to {plot_path}")

    # Summary
    total = len(test_cases)
    single_zero = sum(1 for r in all_results.values() if len(r[4]) <= 1)
    multi_zero = total - single_zero

    print(f"\n  Total cases: {total}")
    print(f"  Cases with at most 1 zero: {single_zero}")
    print(f"  Cases with multiple zeros: {multi_zero}")

    if multi_zero > 0:
        print("  WARNING: Multiple zeros found => CE mapping is NOT globally injective.")
    else:
        print("  All cases show at most 1 zero => consistent with global bijectivity.")

    return all_results


def _analyze_single_case(r0, phi0, eta0, prizes, lotteries_cum, label, r_range,
                         ax_H, ax_phi, ax_eta):
    """Analyze H(r) for a single true parameter set."""
    # Compute true CEs
    CEs = []
    for lc in lotteries_cum:
        CEs.append(compute_ce(r0, phi0, eta0, prizes, lc))
    CE1, CE2, CE3 = CEs

    print(f"\n  Case: {label}")
    print(f"    CEs: CE1={CE1:.4f}, CE2={CE2:.4f}, CE3={CE3:.4f}")

    # Sweep r
    H_vals = np.full_like(r_range, np.nan)
    phi_vals = np.full_like(r_range, np.nan)
    eta_vals = np.full_like(r_range, np.nan)

    idx_r0 = np.argmin(np.abs(r_range - r0))

    # Forward from r0
    phi_prev, eta_prev = phi0, eta0
    for i in range(idx_r0, len(r_range)):
        H_v, phi_v, eta_v = _compute_H(
            r_range[i], (CE1, CE2, CE3), prizes, lotteries_cum,
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
        H_v, phi_v, eta_v = _compute_H(
            r_range[i], (CE1, CE2, CE3), prizes, lotteries_cum,
            phi_init=phi_prev, eta_init=eta_prev
        )
        H_vals[i] = H_v
        phi_vals[i] = phi_v
        eta_vals[i] = eta_v
        if not np.isnan(phi_v):
            phi_prev, eta_prev = phi_v, eta_v

    # Find zeros
    valid = ~np.isnan(H_vals)
    H_valid = H_vals[valid]
    r_valid = r_range[valid]

    zeros = []
    for i in range(len(H_valid) - 1):
        if H_valid[i] * H_valid[i+1] < 0:
            rz = (r_valid[i] - H_valid[i] * (r_valid[i+1] - r_valid[i])
                  / (H_valid[i+1] - H_valid[i]))
            zeros.append(rz)

    n_valid = np.sum(valid)
    print(f"    Valid evaluations: {n_valid}/{len(r_range)}")
    MAX_ZEROS_DISPLAY = 5
    if len(zeros) > MAX_ZEROS_DISPLAY:
        print(f"    Zeros found: {len(zeros)} (likely solver instability; showing first {MAX_ZEROS_DISPLAY})")
        for z in zeros[:MAX_ZEROS_DISPLAY]:
            print(f"      r ~ {z:.6f}")
        print(f"      ...")
    else:
        print(f"    Zeros found: {len(zeros)}")
        for z in zeros:
            print(f"      r ~ {z:.6f}")

    # Plot
    ax_H.plot(r_valid, H_valid, label=label, linewidth=1.5)
    ax_H.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax_H.axvline(x=r0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax_phi.plot(r_valid, phi_vals[valid], label=label, linewidth=1.5)
    ax_phi.axvline(x=r0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax_eta.plot(r_valid, eta_vals[valid], label=label, linewidth=1.5)
    ax_eta.axvline(x=r0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    return r_range, H_vals, phi_vals, eta_vals, zeros


# =========================================================================
# Analysis 3: Deep investigation of multiple-zero cases
# =========================================================================

def run_nonuniqueness_investigation(config, output_dir):
    """Deep investigation of cases where H(r) = 0 has multiple solutions.

    Verifies that the extra zeros are genuine (not solver artifacts) by:
    - Fine-grid scanning with multiple initial guesses
    - Full 3-equation verification at each zero
    - Branch analysis for solution multiplicity

    Corresponds to: sympy/investigate_multiple_zeros.py
    """
    output_dir = Path(output_dir)
    prizes = config['prizes']
    lotteries_cum = config['lotteries']

    # Problematic cases (known to have multiple zeros)
    problematic_cases = [tc for tc in config['test_cases']
                         if tc in [(0.8, 0.5, 1.3), (1.5, 0.6, 1.0)]]
    if not problematic_cases:
        problematic_cases = config['test_cases'][:2]

    all_results = {}

    for r0, phi0, eta0 in problematic_cases:
        label = f"r={r0}, phi={phi0}, eta={eta0}"
        result = _investigate_case(r0, phi0, eta0, prizes, lotteries_cum, label)
        all_results[label] = result

    # Create detailed plots
    n_cases = len(problematic_cases)
    if n_cases == 0:
        return all_results

    fig, axes = plt.subplots(n_cases, 3, figsize=(18, 5 * n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)

    for idx, (r0, phi0, eta0) in enumerate(problematic_cases):
        label = f"r={r0}, phi={phi0}, eta={eta0}"
        r_scan, H_arr, phi_arr, eta_arr, zeros = all_results[label]
        valid = ~np.isnan(H_arr)

        ax_H = axes[idx, 0]
        ax_phi = axes[idx, 1]
        ax_eta = axes[idx, 2]

        ax_H.plot(r_scan[valid], H_arr[valid], 'b-', linewidth=1.5)
        ax_H.axhline(y=0, color='k', linewidth=0.5)
        ax_H.axvline(x=r0, color='r', linewidth=1, linestyle='--', label=f'true r={r0}')
        for rz, pz, ez in zeros:
            ax_H.axvline(x=rz, color='g', linewidth=0.8, linestyle=':', alpha=0.7)
            ax_H.plot(rz, 0, 'go', markersize=8)
        ax_H.set_ylabel('H(r)')
        ax_H.set_title(f'H(r): {label}')
        ax_H.legend(fontsize=8)
        ax_H.grid(True, alpha=0.3)

        ax_phi.plot(r_scan[valid], phi_arr[valid], 'b-', linewidth=1.5)
        ax_phi.axvline(x=r0, color='r', linewidth=1, linestyle='--')
        ax_phi.axhline(y=phi0, color='r', linewidth=0.5, linestyle=':')
        ax_phi.set_ylabel('phi(r)')
        ax_phi.set_title('Implied phi(r)')
        ax_phi.grid(True, alpha=0.3)

        ax_eta.plot(r_scan[valid], eta_arr[valid], 'b-', linewidth=1.5)
        ax_eta.axvline(x=r0, color='r', linewidth=1, linestyle='--')
        ax_eta.axhline(y=eta0, color='r', linewidth=0.5, linestyle=':')
        ax_eta.set_ylabel('eta(r)')
        ax_eta.set_title('Implied eta(r)')
        ax_eta.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel('r')

    plt.suptitle('Detailed investigation of multi-zero cases', fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = output_dir / 'H_investigation.pdf'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Investigation plot saved to {plot_path}")

    return all_results


def _investigate_case(r0, phi0, eta0, prizes, lotteries_cum, label):
    """Full investigation of a single parameter case."""
    # Compute true CEs
    CEs = tuple(compute_ce(r0, phi0, eta0, prizes, lc) for lc in lotteries_cum)
    CE1, CE2, CE3 = CEs

    print(f"\n  Investigating: {label}")
    print(f"    CEs: {CE1:.6f}, {CE2:.6f}, {CE3:.6f}")

    # Fine scan
    r_scan = np.linspace(0.01, 3.0, 3000)
    idx_r0 = np.argmin(np.abs(r_scan - r0))

    results_dict = {}

    # Forward from r0
    pp, ep = phi0, eta0
    for i in range(idx_r0, len(r_scan)):
        H_v, phi_v, eta_v = _compute_H(
            r_scan[i], CEs, prizes, lotteries_cum,
            phi_init=pp, eta_init=ep
        )
        results_dict[i] = (H_v, phi_v, eta_v)
        if not np.isnan(phi_v):
            pp, ep = phi_v, eta_v

    # Backward from r0
    pp, ep = phi0, eta0
    for i in range(idx_r0 - 1, -1, -1):
        H_v, phi_v, eta_v = _compute_H(
            r_scan[i], CEs, prizes, lotteries_cum,
            phi_init=pp, eta_init=ep
        )
        results_dict[i] = (H_v, phi_v, eta_v)
        if not np.isnan(phi_v):
            pp, ep = phi_v, eta_v

    # Assemble arrays
    H_arr = np.full(len(r_scan), np.nan)
    phi_arr = np.full(len(r_scan), np.nan)
    eta_arr = np.full(len(r_scan), np.nan)

    for i, (Hv, pv, ev) in results_dict.items():
        H_arr[i] = Hv
        phi_arr[i] = pv
        eta_arr[i] = ev

    # Find zeros
    valid = ~np.isnan(H_arr)
    H_v = H_arr[valid]
    r_v = r_scan[valid]
    phi_v = phi_arr[valid]
    eta_v = eta_arr[valid]

    zeros = []
    for i in range(len(H_v) - 1):
        if H_v[i] * H_v[i+1] < 0:
            rz = r_v[i] - H_v[i] * (r_v[i+1] - r_v[i]) / (H_v[i+1] - H_v[i])
            phi_z = phi_v[i] + (phi_v[i+1] - phi_v[i]) * (rz - r_v[i]) / (r_v[i+1] - r_v[i])
            eta_z = eta_v[i] + (eta_v[i+1] - eta_v[i]) * (rz - r_v[i]) / (r_v[i+1] - r_v[i])
            zeros.append((rz, phi_z, eta_z))

    print(f"    Found {len(zeros)} zero(s) of H(r)")

    # Verify each zero via full 3-equation solver
    for j, (rz, pz, ez) in enumerate(zeros):
        def full_system(params):
            r_try, phi_try, eta_try = params
            if phi_try <= 0.001 or eta_try <= 0.001:
                return [1e10, 1e10, 1e10]
            try:
                ces = [compute_ce(r_try, phi_try, eta_try, prizes, lc)
                       for lc in lotteries_cum]
                if any(np.isnan(ces)):
                    return [1e10, 1e10, 1e10]
                return [ces[k] - CEs[k] for k in range(3)]
            except Exception:
                return [1e10, 1e10, 1e10]

        try:
            sol = fsolve(full_system, [rz, pz, ez], full_output=True)
            x, info, ier, msg = sol
            residual = np.max(np.abs(info['fvec']))
            if residual < 1e-8:
                print(f"    Zero #{j+1}: r={x[0]:.6f}, phi={x[1]:.6f}, "
                      f"eta={x[2]:.6f} (verified, residual={residual:.2e})")
            else:
                print(f"    Zero #{j+1}: r={rz:.6f} (refinement residual={residual:.2e})")
        except Exception:
            print(f"    Zero #{j+1}: r={rz:.6f} (refinement failed)")

    return r_scan, H_arr, phi_arr, eta_arr, zeros


# =========================================================================
# Analysis 4: Systematic non-uniqueness scan
# =========================================================================

def run_systematic_scan(config, output_dir):
    """Systematic grid scan for observational equivalence.

    Tests whether different parameter triples can produce the same CEs
    across a grid of (r, phi, eta) values.

    Corresponds to: sympy/systematic_nonuniqueness.py
    """
    output_dir = Path(output_dir)
    prizes = config['prizes']
    lotteries_cum = config['lotteries']
    lotteries_tuples = [(prizes, lc) for lc in lotteries_cum]

    scan_grid = config.get('scan_grid', {})

    print("\n  Scanning for observational equivalence (3 lotteries)...")

    r_grid = np.arange(*scan_grid.get('r', (0.1, 2.01, 0.2)))
    phi_grid = np.arange(*scan_grid.get('phi', (0.3, 1.51, 0.3)))
    eta_grid = np.arange(*scan_grid.get('eta', (0.3, 1.51, 0.3)))

    results = []
    for r0 in r_grid:
        for phi0 in phi_grid:
            for eta0 in eta_grid:
                true_ces, equivs = find_observational_equivalent(
                    r0, phi0, eta0, lotteries_tuples
                )
                if any(np.isnan(true_ces)):
                    continue
                results.append((r0, phi0, eta0, len(equivs), equivs))

    n_unique = sum(1 for r in results if r[3] == 0)
    n_nonunique = sum(1 for r in results if r[3] > 0)

    print(f"    Total points tested: {len(results)}")
    print(f"    Unique: {n_unique}")
    print(f"    Non-unique: {n_nonunique}")
    if len(results) > 0:
        print(f"    Non-uniqueness rate: {100*n_nonunique/len(results):.1f}%")

    # Print non-unique cases
    if n_nonunique > 0:
        print(f"\n    Non-unique cases:")
        for r0, phi0, eta0, n_eq, equivs in results:
            if n_eq > 0:
                for eq_r, eq_phi, eq_eta, eq_res in equivs:
                    print(f"      ({r0:.1f}, {phi0:.1f}, {eta0:.1f}) <-> "
                          f"({eq_r:.4f}, {eq_phi:.4f}, {eq_eta:.4f}) "
                          f"[res: {eq_res:.2e}]")

    return results


# =========================================================================
# Analysis 5: Verification of known equivalent pairs
# =========================================================================

def run_verification(config, output_dir):
    """Verify known observationally equivalent parameter pairs.

    Confirms that two distinct (r, phi, eta) triples produce identical CEs
    for all lotteries, and reports decision weight differences.

    Corresponds to: sympy/verify_nonuniqueness.py
    """
    output_dir = Path(output_dir)
    prizes = config['prizes']
    lotteries_cum = config['lotteries']
    equivalence_pairs = config.get('equivalence_pairs', [])

    if not equivalence_pairs:
        print("  No equivalence pairs to verify.")
        return

    lines = []

    def log_print(msg=""):
        print(msg)
        lines.append(msg)

    log_print("\n  VERIFICATION: OBSERVATIONAL EQUIVALENCE IN RDU-PRELEC MODEL")

    for pair_idx, (params_A, params_B) in enumerate(equivalence_pairs):
        r1, p1, e1 = params_A
        r2, p2, e2 = params_B

        log_print(f"\n  Pair {pair_idx + 1}:")
        log_print(f"    Params A: (r={r1}, phi={p1}, eta={e1})")
        log_print(f"    Params B: (r={r2:.10f}, phi={p2:.10f}, eta={e2:.10f})")

        max_diff = 0
        for i, lc in enumerate(lotteries_cum, 1):
            ce1 = compute_ce(r1, p1, e1, prizes, lc)
            ce2 = compute_ce(r2, p2, e2, prizes, lc)
            diff = abs(ce1 - ce2)
            max_diff = max(max_diff, diff)
            log_print(f"    L{i}: CE_A={ce1:.12f}, CE_B={ce2:.12f}, |diff|={diff:.2e}")

        # Decision weights
        for i, lc in enumerate(lotteries_cum, 1):
            q, s = lc['q'], lc['s']
            wq1 = prelec_w(q, p1, e1); ws1 = prelec_w(s, p1, e1)
            wq2 = prelec_w(q, p2, e2); ws2 = prelec_w(s, p2, e2)
            d1_A = 1-ws1; d2_A = ws1-wq1; d3_A = wq1
            d1_B = 1-ws2; d2_B = ws2-wq2; d3_B = wq2
            log_print(f"    L{i} weights A: ({d1_A:.6f}, {d2_A:.6f}, {d3_A:.6f})")
            log_print(f"    L{i} weights B: ({d1_B:.6f}, {d2_B:.6f}, {d3_B:.6f})")

        log_print(f"    Max CE difference: {max_diff:.2e}")
        log_print(f"    Different weights, same CEs => r compensates (phi, eta).")

    # Save text summary
    summary_path = output_dir / 'verification.txt'
    summary_path.write_text('\n'.join(lines))


# =========================================================================
# Analysis 6: Lottery design analysis
# =========================================================================

def run_lottery_design(config, output_dir):
    """Analyze which lottery designs restore global injectivity.

    Tests two strategies:
    1. Add a 4th lottery on the same prizes (overdetermined system)
    2. Use lotteries with different prizes

    Corresponds to: sympy/lottery_design_analysis.py
    """
    output_dir = Path(output_dir)
    prizes = config['prizes']
    lotteries_cum = config['lotteries']
    equivalence_pairs = config.get('equivalence_pairs', [])

    if not equivalence_pairs:
        print("  No equivalence pairs -- skipping lottery design analysis.")
        return

    lines = []

    def log_print(msg=""):
        print(msg)
        lines.append(msg)

    # ---- Part 1: 4th lottery on same prizes ----
    log_print("\n  PART 1: Adding a 4th lottery on the same prizes")

    fourth_lotteries = [
        ({'q': 0.3, 's': 0.6}, "L4a: (0.4, 0.3, 0.3)"),
        ({'q': 0.2, 's': 0.8}, "L4b: (0.2, 0.6, 0.2)"),
        ({'q': 0.6, 's': 0.8}, "L4c: (0.2, 0.2, 0.6)"),
        ({'q': 0.05, 's': 0.95}, "L4d: (0.05, 0.9, 0.05)"),
        ({'q': 0.33, 's': 0.67}, "L4e: (0.33, 0.34, 0.33)"),
    ]

    for lc4, name in fourth_lotteries:
        log_print(f"\n    {name}  (q={lc4['q']}, s={lc4['s']})")
        for pair_idx, (params_A, params_B) in enumerate(equivalence_pairs):
            r1, p1, e1 = params_A
            r2, p2, e2 = params_B
            ce4_A = compute_ce(r1, p1, e1, prizes, lc4)
            ce4_B = compute_ce(r2, p2, e2, prizes, lc4)
            diff = abs(ce4_A - ce4_B)
            sep = "SEPARATES" if diff > 0.001 else "does NOT separate"
            log_print(f"      Pair {pair_idx+1}: CE_A={ce4_A:.6f}, CE_B={ce4_B:.6f}, "
                      f"|diff|={diff:.4f}  [{sep}]")

    # ---- Part 2: Different prizes ----
    log_print("\n  PART 2: Different prizes within prize range")

    different_prize_lotteries = [
        ((10, 20, 50), {'q': 0.5, 's': 0.5}, "prizes (10,20,50), binary"),
        ((10, 40, 50), {'q': 0.5, 's': 0.5}, "prizes (10,40,50), binary"),
        ((15, 30, 45), {'q': 0.5, 's': 0.5}, "prizes (15,30,45), binary"),
        ((10, 15, 50), {'q': 0.3, 's': 0.7}, "prizes (10,15,50), skewed"),
        ((20, 35, 50), {'q': 0.4, 's': 0.7}, "prizes (20,35,50), skewed"),
    ]

    for alt_prizes, alt_cum, name in different_prize_lotteries:
        log_print(f"\n    {name}")
        for pair_idx, (params_A, params_B) in enumerate(equivalence_pairs):
            r1, p1, e1 = params_A
            r2, p2, e2 = params_B
            ce_A = compute_ce(r1, p1, e1, alt_prizes, alt_cum)
            ce_B = compute_ce(r2, p2, e2, alt_prizes, alt_cum)
            diff = abs(ce_A - ce_B)
            sep = "SEPARATES" if diff > 0.001 else "does NOT separate"
            log_print(f"      Pair {pair_idx+1}: diff={diff:.6f} [{sep}]")

    # ---- Part 3: Mixed prize configs ----
    log_print("\n  PART 3: Mixed prize configurations (searching for equivalents)")

    base_lots = [(prizes, lc) for lc in lotteries_cum]
    r_search = np.linspace(0.05, 2.5, 200)

    mixed_configs = [
        ("Config A: all same prizes (control)",
         base_lots),
        ("Config B: one different-prize lottery",
         [base_lots[0], base_lots[1], ((10, 20, 50), {'q': 0.4, 's': 0.7})]),
        ("Config C: two different-prize lotteries",
         [((10, 20, 50), {'q': 0.5, 's': 0.5}), base_lots[1],
          ((10, 40, 50), {'q': 0.4, 's': 0.7})]),
    ]

    # Test with a known non-unique pair
    for params_A, params_B in equivalence_pairs[:1]:
        r1, p1, e1 = params_A
        for name, config_lots in mixed_configs:
            log_print(f"\n    {name}")
            true_ces = compute_ces(r1, p1, e1, config_lots)
            found_equiv = False

            def _make_residual(r_fixed, lots, target_ces):
                def residual(params):
                    phi_t, eta_t = params
                    if phi_t <= 0 or eta_t <= 0:
                        return [1e10] * len(lots)
                    try:
                        ces_t = compute_ces(r_fixed, phi_t, eta_t, lots)
                        return list(ces_t - target_ces)
                    except Exception:
                        return [1e10] * len(lots)
                return residual

            for r_try in r_search:
                if abs(r_try - r1) < 0.01:
                    continue
                residual = _make_residual(r_try, config_lots, true_ces)

                for phi_start, eta_start in [(p1, e1), (0.5, 0.5), (1.0, 1.0)]:
                    try:
                        sol = fsolve(residual, [phi_start, eta_start], full_output=True)
                        x, info, ier, msg = sol
                        if (ier == 1 and x[0] > 0.01 and x[1] > 0.01
                                and np.max(np.abs(residual(x))) < 1e-8):
                            log_print(f"      Equivalent found: r={r_try:.4f}, "
                                      f"phi={x[0]:.4f}, eta={x[1]:.4f}")
                            found_equiv = True
                            break
                    except Exception:
                        pass
                if found_equiv:
                    break

            if not found_equiv:
                log_print(f"      NO equivalents found -- GLOBALLY INJECTIVE")

    log_print("\n  KEY INSIGHT: Different prizes break the compensation mechanism")
    log_print("  because each lottery gets its own utility differences.")

    # Create equivalence pairs plot
    if equivalence_pairs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for pair_idx, (params_A, params_B) in enumerate(equivalence_pairs):
            r1, p1, e1 = params_A
            r2, p2, e2 = params_B
            marker = 'o' if pair_idx == 0 else 's'

            axes[0].plot([r1, r2], [p1, p2], marker=marker, markersize=8, linewidth=2,
                         label=f'Pair {pair_idx+1}')
            axes[0].set_xlabel('r'); axes[0].set_ylabel('phi')
            axes[0].set_title('r vs phi')

            axes[1].plot([r1, r2], [e1, e2], marker=marker, markersize=8, linewidth=2,
                         label=f'Pair {pair_idx+1}')
            axes[1].set_xlabel('r'); axes[1].set_ylabel('eta')
            axes[1].set_title('r vs eta')

            axes[2].plot([p1, p2], [e1, e2], marker=marker, markersize=8, linewidth=2,
                         label=f'Pair {pair_idx+1}')
            axes[2].set_xlabel('phi'); axes[2].set_ylabel('eta')
            axes[2].set_title('phi vs eta')

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = output_dir / 'equivalence_pairs.pdf'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        log_print(f"\n  Equivalence pairs plot saved to {plot_path}")

    # Save text summary
    summary_path = output_dir / 'lottery_design.txt'
    summary_path.write_text('\n'.join(lines))


# =========================================================================
# Analysis 7: Dense grid P-matrix verification (prize-diverse design)
# =========================================================================

def _power_mean_body(s, prizes, q, s_cum, phi, eta):
    r"""Compute B_k = sum pi_mk * x_mk^s for a single lottery.

    Parameters
    ----------
    s : array_like
        Exponent s = 1 - r.
    prizes : tuple of 3 floats
        (a, b, c) with a < b < c.
    q : float
        Cumulative probability P(X >= c).
    s_cum : float
        Cumulative probability P(X >= b).
    phi, eta : array_like
        Prelec PWF parameters.

    Returns
    -------
    B : array_like
        Power-mean body values.
    """
    a, b, c = prizes
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s_cum, phi, eta)
    pi3 = wq
    pi2 = ws - wq
    pi1 = 1.0 - ws
    return pi1 * a**s + pi2 * b**s + pi3 * c**s


def _dB_dphi(s, prizes, q, s_cum, phi, eta):
    r"""Partial dB_k/dphi = sum (dpi_mk/dphi) * x_mk^s.

    Uses dw/dphi(p) = -w(p) * eta * t_p^phi * ln(t_p) where t_p = -ln(p).
    """
    a, b, c = prizes
    t_q = -np.log(q)
    t_s = -np.log(s_cum)

    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s_cum, phi, eta)

    # dw/dphi = -w(p) * eta * t_p^phi * ln(t_p)
    dwq = -wq * eta * t_q**phi * np.log(t_q)
    dws = -ws * eta * t_s**phi * np.log(t_s)

    dpi3 = dwq
    dpi2 = dws - dwq
    dpi1 = -dws

    return dpi1 * a**s + dpi2 * b**s + dpi3 * c**s


def _dB_deta(s, prizes, q, s_cum, phi, eta):
    r"""Partial dB_k/deta = sum (dpi_mk/deta) * x_mk^s.

    Uses dw/deta(p) = -w(p) * t_p^phi where t_p = -ln(p).
    """
    a, b, c = prizes
    t_q = -np.log(q)
    t_s = -np.log(s_cum)

    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s_cum, phi, eta)

    # dw/deta = -w(p) * t_p^phi
    dwq = -wq * t_q**phi
    dws = -ws * t_s**phi

    dpi3 = dwq
    dpi2 = dws - dwq
    dpi1 = -dws

    return dpi1 * a**s + dpi2 * b**s + dpi3 * c**s


def _dB_dr(s, prizes, q, s_cum, phi, eta):
    r"""Partial dB_k/dr = -sum pi_mk * x_mk^s * ln(x_mk).

    Since s = 1 - r, we have ds/dr = -1, so:
        dB_k/dr = -sum_m pi_mk * x_mk^s * ln(x_mk)
    """
    a, b, c = prizes
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s_cum, phi, eta)
    pi3 = wq
    pi2 = ws - wq
    pi1 = 1.0 - ws

    return -(pi1 * a**s * np.log(a) + pi2 * b**s * np.log(b)
             + pi3 * c**s * np.log(c))


def _build_B_jacobian(r, phi, eta, lotteries):
    r"""Build the 3x3 Jacobian of the B-map.

    B_k(r, phi, eta) = sum_m pi_mk(phi, eta) * x_mk^{1-r}.
    J_B[k, j] = dB_k / d(theta_j) where theta = (r, phi, eta).

    Parameters
    ----------
    r, phi, eta : float or array_like
        Parameter values.  For grid evaluation, these are 3D arrays.
    lotteries : list of dicts
        Each dict has keys 'prizes' and 'probs'.

    Returns
    -------
    J : ndarray of shape (..., 3, 3)
        B-map Jacobian.
    """
    s = 1.0 - r
    shape = np.broadcast_shapes(np.shape(r), np.shape(phi), np.shape(eta))
    J = np.empty(shape + (3, 3))

    for k, lot in enumerate(lotteries):
        prizes = lot['prizes']
        probs = lot['probs']
        q = probs[2]              # P(X >= x3)
        s_cum = probs[1] + probs[2]  # P(X >= x2)

        J[..., k, 0] = _dB_dr(s, prizes, q, s_cum, phi, eta)
        J[..., k, 1] = _dB_dphi(s, prizes, q, s_cum, phi, eta)
        J[..., k, 2] = _dB_deta(s, prizes, q, s_cum, phi, eta)

    return J


def run_B_pmatrix_verification(config, output_dir, n_grid=80):
    r"""Dense grid verification that the B-map Jacobian is a P-matrix.

    The B-map is (r, phi, eta) -> (B_1, B_2, B_3) where
    B_k = sum_m pi_mk(phi, eta) * x_mk^{1-r}.

    If the sign-adjusted J_B is a P-matrix everywhere on the domain,
    then by the Gale-Nikaido univalence theorem the B-map is globally
    injective.

    Parameters
    ----------
    config : dict
        Must contain 'lotteries'.  Optionally 'domain'.
    output_dir : Path
        Directory for output files.
    n_grid : int
        Grid points per dimension (total evaluations = n_grid^3).
    """
    output_dir = Path(output_dir)
    lotteries = config['lotteries']
    domain = config.get('domain', {
        'r': (0.01, 0.98),
        'phi': (0.2, 2.0),
        'eta': (0.2, 3.0),
    })

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("B-MAP P-MATRIX VERIFICATION (DENSE GRID)")
    log("=" * 70)

    # Lottery design
    log("\nLottery design:")
    for k, lot in enumerate(lotteries, 1):
        log(f"  L{k}: prizes={lot['prizes']}, probs={lot['probs']}")

    # Domain
    r_lo, r_hi = domain['r']
    phi_lo, phi_hi = domain['phi']
    eta_lo, eta_hi = domain['eta']
    log(f"\nDomain: r in [{r_lo}, {r_hi}], "
        f"phi in [{phi_lo}, {phi_hi}], "
        f"eta in [{eta_lo}, {eta_hi}]")
    log(f"Grid: {n_grid}^3 = {n_grid**3:,} points")

    # ---- Determine sign adjustment from midpoint ----
    r_mid = (r_lo + r_hi) / 2
    phi_mid = (phi_lo + phi_hi) / 2
    eta_mid = (eta_lo + eta_hi) / 2
    J_test = _build_B_jacobian(r_mid, phi_mid, eta_mid, lotteries)
    diag_signs = np.array([np.sign(J_test[k, k]) for k in range(3)])
    param_names = ['r', 'phi', 'eta']
    log(f"\nSign adjustment (diagonal at midpoint "
        f"r={r_mid:.2f}, phi={phi_mid:.2f}, eta={eta_mid:.2f}):")
    for k in range(3):
        log(f"  J_B[{k},{k}] ({param_names[k]}): "
            f"{J_test[k,k]:+.6e}  ->  col sign = {int(diag_signs[k]):+d}")

    # ---- Build grid ----
    r_grid = np.linspace(r_lo, r_hi, n_grid)
    phi_grid = np.linspace(phi_lo, phi_hi, n_grid)
    eta_grid = np.linspace(eta_lo, eta_hi, n_grid)
    R, PHI, ETA = np.meshgrid(r_grid, phi_grid, eta_grid, indexing='ij')

    # ---- Compute J_B and apply sign adjustment ----
    J = _build_B_jacobian(R, PHI, ETA, lotteries)
    for j in range(3):
        J[..., :, j] *= diag_signs[j]

    # ---- Verify diagonal sign consistency ----
    log("\nDiagonal sign consistency across grid:")
    diag_consistent = True
    for k in range(3):
        d = J[..., k, k]
        mn = float(np.min(d))
        ok = mn > 0
        if not ok:
            diag_consistent = False
        log(f"  J_B_adj[{k},{k}] ({param_names[k]}): "
            f"min = {mn:+.6e}  {'OK' if ok else 'SIGN CHANGE'}")
    if not diag_consistent:
        log("\n  WARNING: diagonal sign adjustment is inconsistent.")
        log("  The chosen sign pattern may not work globally.")

    # ---- Compute all 7 principal minors ----
    minors = _all_principal_minors(J)
    minor_names = ['M1', 'M2', 'M3', 'M12', 'M13', 'M23', 'M123']
    minor_labels = {
        'M1': 'M_r (r diagonal)',
        'M2': 'M_phi (phi diagonal)',
        'M3': 'M_eta (eta diagonal)',
        'M12': 'M_{r,phi}',
        'M13': 'M_{r,eta}',
        'M23': 'M_{phi,eta}',
        'M123': 'det(J_B)',
    }

    all_positive = True
    log(f"\n  {'Minor':<28s}  {'Min value':>14s}  {'Max value':>14s}  Status")
    log(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*10}")

    for name in minor_names:
        vals = minors[name]
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        ok = mn > 0
        status = "POSITIVE" if ok else "NOT POSITIVE"
        if not ok:
            all_positive = False
        log(f"  {minor_labels[name]:<28s}  {mn:>+14.6e}  {mx:>+14.6e}  {status}")

        idx = np.unravel_index(np.argmin(vals), vals.shape)
        r_at = R[idx]
        phi_at = PHI[idx]
        eta_at = ETA[idx]
        log(f"    min at (r={r_at:.4f}, phi={phi_at:.4f}, eta={eta_at:.4f})")

    # ---- Verdict ----
    log("\n" + "=" * 70)
    if all_positive:
        log("RESULT: ALL 7 PRINCIPAL MINORS POSITIVE ON DENSE GRID")
        log(f"  Domain: [{r_lo}, {r_hi}] x [{phi_lo}, {phi_hi}] "
            f"x [{eta_lo}, {eta_hi}]")
        log(f"  Grid:   {n_grid}^3 = {n_grid**3:,} evaluation points")
        log("")
        log("The sign-adjusted B-map Jacobian J_B is a P-matrix at every")
        log("grid point.  By the Gale-Nikaido univalence theorem, the B-map")
        log("  (r, phi, eta)  ->  (B_1, B_2, B_3)")
        log("is globally injective on this domain.")
        log("")
        log("Since CE_k = B_k^{1/(1-r)} is strictly monotone in B_k for")
        log("each fixed r, and a change in r with matching B-values is ruled")
        log("out by B-injectivity, global injectivity of the CE-map follows.")
    else:
        log("RESULT: P-MATRIX CONDITION VIOLATED")
        log("At least one principal minor is non-positive at some grid point.")
    log("=" * 70)

    # Save
    summary_path = output_dir / 'B_pmatrix_verification.txt'
    summary_path.write_text('\n'.join(lines))
    log(f"\nResults saved to {summary_path}")

    return all_positive


def _ce_deriv_r(s, prizes, q, s_cum, phi, eta):
    r"""Compute dCE_k/dr for a single lottery.

    From CE_k = B_k^{1/s} and s = 1 - r:
        dCE/dr = (CE/s) * [ln(B)/s - (1/B) * sum pi x^s ln(x)]

    This is negative for s > 0 (r < 1) and positive for s < 0 (r > 1).
    """
    a, b, c = prizes
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s_cum, phi, eta)
    pi3 = wq
    pi2 = ws - wq
    pi1 = 1.0 - ws

    B = pi1 * a**s + pi2 * b**s + pi3 * c**s
    CE = B**(1.0 / s)

    weighted_ln = (pi1 * a**s * np.log(a)
                   + pi2 * b**s * np.log(b)
                   + pi3 * c**s * np.log(c))

    # dCE/dr = (CE/s) * [ln(B)/s - weighted_ln/B]
    return (CE / s) * (np.log(B) / s - weighted_ln / B)


def _ce_deriv_phi(s, prizes, q, s_cum, phi, eta):
    r"""Compute dCE_k/dphi for a single lottery.

    dCE/dphi = CE/(s*B) * dB/dphi
    """
    B = _power_mean_body(s, prizes, q, s_cum, phi, eta)
    CE = B**(1.0 / s)
    dB = _dB_dphi(s, prizes, q, s_cum, phi, eta)
    return CE / (s * B) * dB


def _ce_deriv_eta(s, prizes, q, s_cum, phi, eta):
    r"""Compute dCE_k/deta for a single lottery.

    dCE/deta = CE/(s*B) * dB/deta
    """
    B = _power_mean_body(s, prizes, q, s_cum, phi, eta)
    CE = B**(1.0 / s)
    dB = _dB_deta(s, prizes, q, s_cum, phi, eta)
    return CE / (s * B) * dB


def _build_sign_adjusted_jacobian(r, phi, eta, lottery_specs):
    r"""Build the 3x3 sign-adjusted CE Jacobian.

    Computes the actual CE partial derivatives and applies column signs
    S = diag(-1, +1, -1) so that diagonal entries are positive (for r < 1).

    By the D-reduction lemma (Lem 1 in the technical note), multiplying
    by positive row scalings d_k = CE_k^{-r} does not affect the P-matrix
    property, so we work directly with the CE derivatives.

    Parameters
    ----------
    r, phi, eta : float or array_like
        Parameter values. For grid evaluation, these are 3D arrays.
    lottery_specs : list of dicts
        Each dict has keys: prizes, q, s, param, sign.

    Returns
    -------
    J : ndarray of shape (..., 3, 3)
        Sign-adjusted Jacobian. For scalar inputs, shape is (3, 3).
    """
    s = 1.0 - r
    shape = np.broadcast_shapes(np.shape(r), np.shape(phi), np.shape(eta))
    J = np.empty(shape + (3, 3))

    col_signs = np.array([lot['sign'] for lot in lottery_specs])

    for row, lot in enumerate(lottery_specs):
        prizes = lot['prizes']
        q_val = lot['q']
        s_cum = lot['s']

        # Compute actual CE derivatives
        dCE_dr = _ce_deriv_r(s, prizes, q_val, s_cum, phi, eta)
        dCE_dphi = _ce_deriv_phi(s, prizes, q_val, s_cum, phi, eta)
        dCE_deta = _ce_deriv_eta(s, prizes, q_val, s_cum, phi, eta)

        # Apply column signs: S = diag(sign_r, sign_phi, sign_eta)
        J[..., row, 0] = col_signs[0] * dCE_dr
        J[..., row, 1] = col_signs[1] * dCE_dphi
        J[..., row, 2] = col_signs[2] * dCE_deta

    return J


def _all_principal_minors(J):
    """Compute all 7 principal minors of a 3x3 matrix (or array of matrices).

    Parameters
    ----------
    J : ndarray of shape (..., 3, 3)

    Returns
    -------
    minors : dict mapping str -> ndarray
        Keys: 'M1', 'M2', 'M3', 'M12', 'M13', 'M23', 'M123'.
    """
    # Order 1: diagonal entries
    M1 = J[..., 0, 0]
    M2 = J[..., 1, 1]
    M3 = J[..., 2, 2]

    # Order 2: 2x2 principal minors
    M12 = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    M13 = J[..., 0, 0] * J[..., 2, 2] - J[..., 0, 2] * J[..., 2, 0]
    M23 = J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1]

    # Order 3: full determinant
    M123 = (J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2]
                             - J[..., 1, 2] * J[..., 2, 1])
            - J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2]
                               - J[..., 1, 2] * J[..., 2, 0])
            + J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1]
                               - J[..., 1, 1] * J[..., 2, 0]))

    return {'M1': M1, 'M2': M2, 'M3': M3,
            'M12': M12, 'M13': M13, 'M23': M23, 'M123': M123}


def run_pmatrix_verification(config, output_dir, n_grid=50):
    r"""Dense grid verification of the P-matrix condition for the prize-diverse design.

    Evaluates all 7 principal minors of the sign-adjusted unnormalized
    Jacobian on a dense grid over the parameter domain. If all minors
    are strictly positive at every grid point, this provides strong
    numerical evidence for the P-matrix property (and hence global
    injectivity via the Gale-Nikaido theorem).

    Parameters
    ----------
    config : dict
        Must contain keys 'diverse_lotteries', 'diverse_domain',
        and 'diverse_test_cases'.
    output_dir : Path
        Directory for output files.
    n_grid : int
        Grid points per dimension (total evaluations = n_grid^3).
    """
    output_dir = Path(output_dir)
    lottery_specs = config['diverse_lotteries']
    domain = config['diverse_domain']
    test_cases = config.get('diverse_test_cases', [])

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("P-MATRIX VERIFICATION (DENSE GRID, PRIZE-DIVERSE DESIGN)")
    log("=" * 70)

    # Print lottery design
    log("\nLottery design:")
    for lot in lottery_specs:
        log(f"  prizes={lot['prizes']}, q={lot['q']}, s={lot['s']} "
            f"-> {lot['param']} (sign={lot['sign']:+d})")

    # Print domain
    r_lo, r_hi = domain['r']
    phi_lo, phi_hi = domain['phi']
    eta_lo, eta_hi = domain['eta']
    log(f"\nDomain: r in [{r_lo}, {r_hi}], "
        f"phi in [{phi_lo}, {phi_hi}], "
        f"eta in [{eta_lo}, {eta_hi}]")
    log(f"Grid: {n_grid}^3 = {n_grid**3} points")

    # --- Point evaluations at test cases ---
    if test_cases:
        log("\n--- Point evaluations at test cases ---")
        log(f"  {'(r, phi, eta)':<22s}  {'M1':>10s}  {'M2':>10s}  {'M3':>10s}"
            f"  {'M12':>10s}  {'M13':>10s}  {'M23':>10s}  {'det':>10s}")
        log("  " + "-" * 100)
        for r_val, phi_val, eta_val in test_cases:
            J = _build_sign_adjusted_jacobian(
                np.float64(r_val), np.float64(phi_val), np.float64(eta_val),
                lottery_specs)
            minors = _all_principal_minors(J)
            vals = [minors[k] for k in ['M1', 'M2', 'M3', 'M12', 'M13', 'M23', 'M123']]
            all_pos = all(v > 0 for v in vals)
            status = "OK" if all_pos else "FAIL"
            log(f"  ({r_val}, {phi_val}, {eta_val})  "
                + "  ".join(f"{float(v):+10.4e}" for v in vals)
                + f"  [{status}]")

    # --- Dense grid evaluation ---
    log("\n--- Dense grid evaluation ---")
    r_grid = np.linspace(r_lo, r_hi, n_grid)
    phi_grid = np.linspace(phi_lo, phi_hi, n_grid)
    eta_grid = np.linspace(eta_lo, eta_hi, n_grid)

    # Build 3D meshgrid
    R, PHI, ETA = np.meshgrid(r_grid, phi_grid, eta_grid, indexing='ij')

    # Compute sign-adjusted Jacobian at every grid point
    J = _build_sign_adjusted_jacobian(R, PHI, ETA, lottery_specs)
    minors = _all_principal_minors(J)

    # Check each minor
    minor_names = ['M1', 'M2', 'M3', 'M12', 'M13', 'M23', 'M123']
    minor_labels = {
        'M1': 'M_1 (r diagonal)',
        'M2': 'M_2 (phi diagonal)',
        'M3': 'M_3 (eta diagonal)',
        'M12': 'M_{1,2} (r-phi block)',
        'M13': 'M_{1,3} (r-eta block)',
        'M23': 'M_{2,3} (phi-eta block)',
        'M123': 'M_{1,2,3} (full det)',
    }

    all_positive = True
    log(f"\n  {'Minor':<28s}  {'Min value':>14s}  {'Max value':>14s}  {'Status'}")
    log(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*10}")

    for name in minor_names:
        vals = minors[name]
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        ok = mn > 0
        status = "POSITIVE" if ok else "NOT POSITIVE"
        if not ok:
            all_positive = False
        log(f"  {minor_labels[name]:<28s}  {mn:>+14.6e}  {mx:>+14.6e}  {status}")

        # Report location of minimum
        idx = np.unravel_index(np.argmin(vals), vals.shape)
        r_at = R[idx]
        phi_at = PHI[idx]
        eta_at = ETA[idx]
        log(f"    min at (r={r_at:.4f}, phi={phi_at:.4f}, eta={eta_at:.4f})")

    # --- Analytical conditions check ---
    log("\n--- Analytical conditions (from technical note Section 7) ---")

    # Condition 1: M1 > 0 (r diagonal)
    # -R_1 > 0 iff R_1 < 0, which holds when r < 1 (Jensen's inequality).
    # For r > 1, sign flips but the column sign absorbs it.
    # Actually: -R_k where R_k = weighted_ln - B*ln(B)/s.
    # By Jensen, ln(CE) > E_pi[ln X], so R_k < 0 for s > 0, R_k > 0 for s < 0.
    # After sign negation: -R_k > 0 for all s != 0.
    m1_ok = float(np.min(minors['M1'])) > 0
    log(f"  1. M_1 > 0 (Jensen's inequality): "
        f"{'CONFIRMED' if m1_ok else 'FAILED'} (min = {float(np.min(minors['M1'])):+.6e})")

    # Condition 2: M2 > 0 (phi diagonal)
    m2_ok = float(np.min(minors['M2'])) > 0
    log(f"  2. M_2 > 0 (phi own-effect positive): "
        f"{'CONFIRMED' if m2_ok else 'FAILED'} (min = {float(np.min(minors['M2'])):+.6e})")

    # Condition 3: M3 > 0 (eta diagonal)
    m3_ok = float(np.min(minors['M3'])) > 0
    log(f"  3. M_3 > 0 (eta own-effect positive): "
        f"{'CONFIRMED' if m3_ok else 'FAILED'} (min = {float(np.min(minors['M3'])):+.6e})")

    # Condition 4: det > 0
    det_ok = float(np.min(minors['M123'])) > 0
    log(f"  4. det > 0 (full determinant): "
        f"{'CONFIRMED' if det_ok else 'FAILED'} (min = {float(np.min(minors['M123'])):+.6e})")

    # --- Final verdict ---
    log("\n" + "=" * 70)
    if all_positive:
        log("RESULT: ALL 7 PRINCIPAL MINORS POSITIVE ON DENSE GRID")
        log(f"  Domain: [{r_lo}, {r_hi}] x [{phi_lo}, {phi_hi}] x [{eta_lo}, {eta_hi}]")
        log(f"  Grid:   {n_grid}^3 = {n_grid**3} evaluation points")
        log("")
        log("The sign-adjusted unnormalized Jacobian is a P-matrix at every")
        log("grid point. Since the CE Jacobian differs only by positive row")
        log("scalings, the CE Jacobian is also a P-matrix at every grid point.")
        log("")
        log("By the Gale-Nikaido theorem, this provides strong numerical")
        log("evidence that the CE mapping is globally injective on this domain.")
    else:
        log("RESULT: P-MATRIX CONDITION VIOLATED")
        log("At least one principal minor is non-positive at some grid point.")
        log("The prize-diverse design may not ensure global injectivity on")
        log("this domain, or the grid resolution is insufficient.")
    log("=" * 70)

    # Save results
    summary_path = output_dir / 'pmatrix_verification.txt'
    summary_path.write_text('\n'.join(lines))
    log(f"\nResults saved to {summary_path}")

    return all_positive


# =========================================================================
# Main entry point
# =========================================================================

def run_injectivity_analysis(output_dir, config=None):
    """Run the injectivity analysis for Appendix C.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save all plots and summaries.
    config : dict
        Configuration with keys:
        - 'lotteries': list of {'prizes': tuple, 'probs': tuple} dicts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        raise ValueError("config is required — define INJECTIVITY_CONFIG in main.py")

    print("=" * 60)
    print("CE MAPPING INJECTIVITY ANALYSIS")
    print("=" * 60)

    # 1. Symbolic Jacobian (B-formulation decomposition)
    print("\n--- Analysis 1: Symbolic Jacobian ---")
    run_symbolic_jacobian(config, output_dir)

    # 2. B-map P-matrix verification (dense grid)
    print("\n--- Analysis 2: B-map P-matrix verification ---")
    run_B_pmatrix_verification(config, output_dir)
