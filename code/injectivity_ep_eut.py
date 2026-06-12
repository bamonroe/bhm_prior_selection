"""
Injectivity analysis for the EP+EUT CE mapping (Gram determinant).

The EP+EUT model has 3 parameters (alpha, r, lambda) mapped to 4 quantities
(CE1, CE2, CE3, P_choice). Since k=4 > p=3, the induced prior uses the
Gram determinant sqrt(det(J^T J)) instead of |det J|.

Injectivity requires J to have full column rank everywhere, equivalently
det(J^T J) > 0. Since lambda only enters the P_choice row and the 3 CEs
depend only on (alpha, r), the analysis reduces to checking that the 3x2
submatrix d(CE1,CE2,CE3)/d(alpha,r) has rank 2 for all (alpha, r).

Called from main.py via pipelines.run_injectivity_ep_eut().
"""

import numpy as np
from pathlib import Path


# =============================================================================
# EP utility functions (numpy, no JAX)
# =============================================================================

def _ep_utility_normalized(x, alpha, r, x_min, x_max):
    """Normalized EP utility u(x) in [0,1]."""
    s = 1.0 - r

    if abs(s) < 1e-12:
        return np.log(x / x_min) / np.log(x_max / x_min)

    if abs(alpha) < 1e-12:
        x_s = np.power(x, s)
        xmin_s = np.power(x_min, s)
        xmax_s = np.power(x_max, s)
        return (x_s - xmin_s) / (xmax_s - xmin_s)

    x_s = np.power(x, s)
    xmin_s = np.power(x_min, s)
    xmax_s = np.power(x_max, s)

    exp_x = np.exp(-alpha * x_s)
    exp_min = np.exp(-alpha * xmin_s)
    exp_max = np.exp(-alpha * xmax_s)

    denom = exp_min - exp_max
    if abs(denom) < 1e-300:
        return np.nan

    return (exp_min - exp_x) / denom


def _ep_inverse_newton(u_val, alpha, r, x_min, x_max, n_iter=30):
    """Invert normalized EP utility."""
    s = 1.0 - r

    if abs(s) < 1e-12:
        return x_min * (x_max / x_min) ** u_val

    if abs(alpha) < 1e-12:
        xmin_s = np.power(x_min, s)
        xmax_s = np.power(x_max, s)
        inner = xmin_s + u_val * (xmax_s - xmin_s)
        if inner <= 0:
            return np.nan
        return np.power(inner, 1.0 / s)

    xmin_s = np.power(x_min, s)
    xmax_s = np.power(x_max, s)
    exp_min = np.exp(-alpha * xmin_s)
    exp_max = np.exp(-alpha * xmax_s)
    denom = exp_min - exp_max

    if abs(denom) < 1e-300:
        return np.nan

    # Initial guess from CRRA
    inner = xmin_s + u_val * (xmax_s - xmin_s)
    if inner <= 0:
        x = 0.5 * (x_min + x_max)
    else:
        x = np.power(max(inner, 1e-30), 1.0 / s)
    x = np.clip(x, x_min + 1e-8, x_max - 1e-8)

    for _ in range(n_iter):
        x_s_val = np.power(x, s)
        exp_x = np.exp(-alpha * x_s_val)
        fx = (exp_min - exp_x) / denom - u_val
        dfx = alpha * s * np.power(x, s - 1) * exp_x / denom
        if abs(dfx) < 1e-300:
            break
        x_new = x - fx / dfx
        x_new = np.clip(x_new, x_min + 1e-8, x_max - 1e-8)
        if abs(x_new - x) < 1e-14:
            return x_new
        x = x_new

    return x


def _compute_ce_ep_eut(alpha, r, prizes, probs, x_min, x_max):
    """CE for a binary lottery under EP + EUT (linear PWF)."""
    prizes = np.asarray(prizes, dtype=float)
    probs = np.asarray(probs, dtype=float)

    # EU = sum p_i * u(x_i)
    utilities = np.array([_ep_utility_normalized(x, alpha, r, x_min, x_max)
                          for x in prizes])
    if np.any(np.isnan(utilities)):
        return np.nan

    eu_val = np.dot(probs, utilities)
    if eu_val < 0 or eu_val > 1:
        return np.nan

    return _ep_inverse_newton(eu_val, alpha, r, x_min, x_max)


# =============================================================================
# Jacobian and Gram determinant
# =============================================================================

def _jacobian_ces_fd(alpha, r, lotteries, x_min, x_max, h=1e-7):
    """3x2 Jacobian d(CE1,CE2,CE3)/d(alpha,r) via central differences."""
    J = np.zeros((3, 2))
    params = [alpha, r]

    for j in range(2):
        p_plus = list(params); p_plus[j] += h
        p_minus = list(params); p_minus[j] -= h

        ces_plus = np.array([_compute_ce_ep_eut(*p_plus, lot['prizes'], lot['probs'],
                                                 x_min, x_max) for lot in lotteries])
        ces_minus = np.array([_compute_ce_ep_eut(*p_minus, lot['prizes'], lot['probs'],
                                                  x_min, x_max) for lot in lotteries])
        if np.any(np.isnan(ces_plus)) or np.any(np.isnan(ces_minus)):
            return np.full((3, 2), np.nan)

        J[:, j] = (ces_plus - ces_minus) / (2 * h)

    return J


def _gram_det(J):
    """det(J^T J) for a 3x2 matrix J. Positive iff J has rank 2."""
    gram = J.T @ J  # 2x2
    return np.linalg.det(gram)


# =============================================================================
# Grid evaluation
# =============================================================================

def evaluate_grid(lotteries, domain, n_grid=40, x_min=10.0, x_max=50.0):
    """Evaluate Gram determinant on a 2D (alpha, r) grid.

    Parameters
    ----------
    lotteries : list of dict
        Each dict has 'prizes' and 'probs'.
    domain : dict
        Keys 'alpha' and 'r', each a (lo, hi) tuple.
    n_grid : int
        Grid points per dimension.
    x_min, x_max : float
        Utility normalization bounds.

    Returns
    -------
    dict with 'n_positive', 'n_zero_or_negative', 'n_nan',
              'zero_points', 'gram_values', etc.
    """
    alpha_lo, alpha_hi = domain['alpha']
    r_lo, r_hi = domain['r']

    alpha_grid = np.linspace(alpha_lo, alpha_hi, n_grid)
    r_grid = np.linspace(r_lo, r_hi, n_grid)

    n_positive = 0
    n_zero = 0
    n_nan = 0
    zero_points = []
    gram_values = []

    for alpha in alpha_grid:
        for r in r_grid:
            J = _jacobian_ces_fd(alpha, r, lotteries, x_min, x_max)
            if np.any(np.isnan(J)):
                n_nan += 1
                continue
            gd = _gram_det(J)
            if np.isnan(gd):
                n_nan += 1
                continue
            gram_values.append(gd)
            if gd > 0:
                n_positive += 1
            else:
                n_zero += 1
                zero_points.append((alpha, r, gd))

    return {
        'n_positive': n_positive,
        'n_zero_or_negative': n_zero,
        'n_nan': n_nan,
        'n_total': n_grid ** 2,
        'zero_points': zero_points,
        'gram_values': np.array(gram_values),
    }


# =============================================================================
# Main entry point
# =============================================================================

def run_injectivity_ep_eut(output_dir, reference_lotteries, domain=None,
                           n_grid=40, x_min=10.0, x_max=50.0):
    """Run EP+EUT injectivity analysis.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output report and NPZ files.
    reference_lotteries : list of dict
        Current lottery design to evaluate.
    domain : dict, optional
        Parameter domain. Default: alpha in [-1, 2], r in [0.05, 0.95].
    n_grid : int
        Grid resolution.
    x_min, x_max : float
        Utility normalization bounds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if domain is None:
        domain = {
            'alpha': (-1.0, 2.0),
            'r': (0.05, 0.95),
        }

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("EP+EUT INJECTIVITY ANALYSIS (Gram determinant, k=4 > p=3)")
    log("=" * 70)
    log(f"\nModel: EP utility + linear PWF (EUT)")
    log(f"Parameters: alpha (EP curvature), r (risk aversion), lambda (noise)")
    log(f"Quantities: CE1, CE2, CE3, P_choice")
    log(f"Jacobian: 4x3, checking det(J^T J) > 0")
    log(f"\nSince lambda only enters P_choice, the rank condition reduces to:")
    log(f"  rank(d(CE1,CE2,CE3)/d(alpha,r)) = 2 for all (alpha, r)")
    log(f"\nParameter domain:")
    log(f"  alpha in [{domain['alpha'][0]}, {domain['alpha'][1]}]")
    log(f"  r     in [{domain['r'][0]}, {domain['r'][1]}]")
    log(f"  Grid:  {n_grid}^2 = {n_grid**2} points")
    log(f"  x_min={x_min}, x_max={x_max}")

    # Lottery design
    lotteries = [{'prizes': list(lot['prizes']), 'probs': list(lot['probs'])}
                 for lot in reference_lotteries]

    log(f"\nLottery design:")
    for i, lot in enumerate(lotteries):
        ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))
        log(f"  L{i+1}: prizes={lot['prizes']}, probs={lot['probs']}, EV={ev:.1f}")

    # Grid evaluation
    result = evaluate_grid(lotteries, domain, n_grid=n_grid,
                           x_min=x_min, x_max=x_max)

    log(f"\nResults on {n_grid}^2 grid:")
    log(f"  Gram det > 0 (full rank): {result['n_positive']}")
    log(f"  Gram det <= 0 (rank deficient): {result['n_zero_or_negative']}")
    log(f"  NaN/failed: {result['n_nan']}")
    log(f"  Total: {result['n_total']}")

    valid = result['n_positive'] + result['n_zero_or_negative']
    if valid > 0:
        pct = 100 * result['n_zero_or_negative'] / valid
        log(f"  Rank-deficient rate: {pct:.3f}%")

    if result['n_zero_or_negative'] > 0:
        log(f"\n  Rank-deficient points (first 30):")
        log(f"    {'alpha':>8s}  {'r':>8s}  {'det(J^T J)':>14s}")
        for pt in result['zero_points'][:30]:
            alpha, r, gd = pt
            log(f"    {alpha:8.4f}  {r:8.4f}  {gd:14.6e}")

        pts = np.array(result['zero_points'])
        log(f"\n  Range of rank-deficient points:")
        log(f"    alpha: [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
        log(f"    r:     [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]")

    if len(result['gram_values']) > 0:
        gv = result['gram_values']
        log(f"\n  Gram determinant statistics:")
        log(f"    Min:    {gv.min():.6e}")
        log(f"    Max:    {gv.max():.6e}")
        log(f"    Median: {np.median(gv):.6e}")

    # Optimizer confirmation if grid found no failures
    if result['n_zero_or_negative'] == 0:
        log(f"\n  Grid found no rank-deficient points. Confirming with optimizer...")
        from scipy.optimize import differential_evolution

        def neg_gram(params):
            alpha, r = params
            J = _jacobian_ces_fd(alpha, r, lotteries, x_min, x_max)
            if np.any(np.isnan(J)):
                return 1e10
            gd = _gram_det(J)
            if np.isnan(gd):
                return 1e10
            return gd

        bounds = [domain['alpha'], domain['r']]
        opt = differential_evolution(neg_gram, bounds, seed=42, maxiter=300,
                                     tol=1e-12, polish=True)
        log(f"    Minimum Gram det: {opt.fun:.10e}")
        log(f"    At: alpha={opt.x[0]:.6f}, r={opt.x[1]:.6f}")

        if opt.fun > 0:
            log(f"    CONFIRMED: Gram det > 0 everywhere. Full rank guaranteed.")
        else:
            log(f"    WARNING: Optimizer found non-positive Gram det.")

    # Save results
    save_data = {
        'n_positive': result['n_positive'],
        'n_zero_or_negative': result['n_zero_or_negative'],
        'n_nan': result['n_nan'],
        'n_total': result['n_total'],
        'domain_alpha': domain['alpha'],
        'domain_r': domain['r'],
        'n_grid': n_grid,
    }
    if result['n_zero_or_negative'] > 0:
        pts = np.array(result['zero_points'])
        save_data['zero_alpha'] = pts[:, 0]
        save_data['zero_r'] = pts[:, 1]
        save_data['zero_gram'] = pts[:, 2]

    np.savez(output_dir / 'ep_eut_injectivity.npz', **save_data)

    # Write report
    report_path = output_dir / 'ep_eut_injectivity_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    log(f"\nReport saved to: {report_path}")
    return result
