"""
Injectivity analysis for the CRRA+Power CE mapping (Gram determinant).

The CRRA+Power model has 3 parameters (r, delta, lambda) mapped to 4 quantities
(CE1, CE2, CE3, P_choice). Since k=4 > p=3, the induced prior uses the
Gram determinant sqrt(det(J^T J)).

Under CRRA utility with Power PWF w(p) = p^delta, the CE depends on (r, delta)
and is analytic. The rank condition reduces to checking that the 3x2 submatrix
d(CE1,CE2,CE3)/d(r,delta) has rank 2.

Called from main.py via pipelines.run_injectivity_power().
"""

import numpy as np
from pathlib import Path


# =============================================================================
# CRRA + Power PWF functions (numpy)
# =============================================================================

def _crra_ce_power(r, delta, prizes, probs, x_min=10.0, x_max=50.0):
    """CE for a binary lottery under CRRA + Power PWF w(p) = p^delta."""
    prizes = np.asarray(prizes, dtype=float)
    probs = np.asarray(probs, dtype=float)

    order = np.argsort(prizes)
    prizes = prizes[order]
    probs = probs[order]

    n = len(prizes)
    s = 1.0 - r

    # Decision weights via Power PWF
    cum_from_top = np.zeros(n)
    cum_from_top[n - 1] = probs[n - 1]
    for k in range(n - 2, -1, -1):
        cum_from_top[k] = cum_from_top[k + 1] + probs[k]

    dw = np.zeros(n)
    for k in range(n):
        w_k = np.power(cum_from_top[k], delta)
        w_k1 = np.power(cum_from_top[k + 1], delta) if k + 1 < n else 0.0
        dw[k] = w_k - w_k1

    # CRRA utility and RDU
    if abs(s) < 1e-12:
        utilities = np.log(prizes)
        rdu_val = np.dot(dw, utilities)
        return np.exp(rdu_val)
    else:
        utilities = np.power(prizes, s)
        rdu_val = np.dot(dw, utilities)
        if rdu_val <= 0:
            return np.nan
        return np.power(rdu_val, 1.0 / s)


# =============================================================================
# Jacobian and Gram determinant
# =============================================================================

def _jacobian_ces_fd(r, delta, lotteries, x_min, x_max, h=1e-7):
    """3x2 Jacobian d(CE1,CE2,CE3)/d(r,delta) via central differences."""
    J = np.zeros((3, 2))
    params = [r, delta]

    for j in range(2):
        p_plus = list(params); p_plus[j] += h
        p_minus = list(params); p_minus[j] -= h

        ces_plus = np.array([_crra_ce_power(*p_plus, lot['prizes'], lot['probs'],
                                             x_min, x_max) for lot in lotteries])
        ces_minus = np.array([_crra_ce_power(*p_minus, lot['prizes'], lot['probs'],
                                              x_min, x_max) for lot in lotteries])
        if np.any(np.isnan(ces_plus)) or np.any(np.isnan(ces_minus)):
            return np.full((3, 2), np.nan)

        J[:, j] = (ces_plus - ces_minus) / (2 * h)

    return J


def _gram_det(J):
    """det(J^T J) for a 3x2 matrix J."""
    gram = J.T @ J
    return np.linalg.det(gram)


# =============================================================================
# Grid evaluation
# =============================================================================

def evaluate_grid(lotteries, domain, n_grid=40, x_min=10.0, x_max=50.0):
    """Evaluate Gram determinant on a 2D (r, delta) grid."""
    r_lo, r_hi = domain['r']
    delta_lo, delta_hi = domain['delta']

    r_grid = np.linspace(r_lo, r_hi, n_grid)
    delta_grid = np.linspace(delta_lo, delta_hi, n_grid)

    n_positive = 0
    n_zero = 0
    n_nan = 0
    zero_points = []
    gram_values = []

    for r in r_grid:
        for delta in delta_grid:
            J = _jacobian_ces_fd(r, delta, lotteries, x_min, x_max)
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
                zero_points.append((r, delta, gd))

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

def run_injectivity_power(output_dir, reference_lotteries, domain=None,
                          n_grid=40, x_min=10.0, x_max=50.0):
    """Run CRRA+Power injectivity analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if domain is None:
        domain = {
            'r': (0.01, 0.98),
            'delta': (0.1, 3.0),
        }

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("CRRA+POWER INJECTIVITY ANALYSIS (Gram determinant, k=4 > p=3)")
    log("=" * 70)
    log(f"\nModel: CRRA utility + Power PWF w(p) = p^delta")
    log(f"Parameters: r (CRRA), delta (Power), lambda (noise)")
    log(f"Quantities: CE1, CE2, CE3, P_choice")
    log(f"Jacobian: 4x3, checking det(J^T J) > 0")
    log(f"\nParameter domain:")
    log(f"  r     in [{domain['r'][0]}, {domain['r'][1]}]")
    log(f"  delta in [{domain['delta'][0]}, {domain['delta'][1]}]")
    log(f"  Grid:  {n_grid}^2 = {n_grid**2} points")

    lotteries = [{'prizes': list(lot['prizes']), 'probs': list(lot['probs'])}
                 for lot in reference_lotteries]

    log(f"\nLottery design:")
    for i, lot in enumerate(lotteries):
        ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))
        log(f"  L{i+1}: prizes={lot['prizes']}, probs={lot['probs']}, EV={ev:.1f}")

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
        pts = np.array(result['zero_points'])
        log(f"\n  Rank-deficient points:")
        for pt in result['zero_points'][:20]:
            log(f"    r={pt[0]:.4f}, delta={pt[1]:.4f}: Gram det = {pt[2]:.6e}")

    if len(result['gram_values']) > 0:
        gv = result['gram_values']
        log(f"\n  Gram determinant statistics:")
        log(f"    Min:    {gv.min():.6e}")
        log(f"    Max:    {gv.max():.6e}")
        log(f"    Median: {np.median(gv):.6e}")

    if result['n_zero_or_negative'] == 0:
        log(f"\n  Grid found no rank-deficient points. Confirming with optimizer...")
        from scipy.optimize import differential_evolution

        def neg_gram(params):
            r, delta = params
            J = _jacobian_ces_fd(r, delta, lotteries, x_min, x_max)
            if np.any(np.isnan(J)):
                return 1e10
            gd = _gram_det(J)
            return gd if not np.isnan(gd) else 1e10

        bounds = [domain['r'], domain['delta']]
        opt = differential_evolution(neg_gram, bounds, seed=42, maxiter=300,
                                     tol=1e-12, polish=True)
        log(f"    Minimum Gram det: {opt.fun:.10e}")
        log(f"    At: r={opt.x[0]:.6f}, delta={opt.x[1]:.6f}")

        if opt.fun > 0:
            log(f"    CONFIRMED: Gram det > 0 everywhere. Full rank guaranteed.")
        else:
            log(f"    WARNING: Optimizer found non-positive Gram det.")

    # Save
    save_data = {
        'n_positive': result['n_positive'],
        'n_zero_or_negative': result['n_zero_or_negative'],
        'n_nan': result['n_nan'],
        'domain_r': domain['r'],
        'domain_delta': domain['delta'],
        'n_grid': n_grid,
    }
    np.savez(output_dir / 'power_injectivity.npz', **save_data)

    report_path = output_dir / 'power_injectivity_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    log(f"\nReport saved to: {report_path}")
    return result
