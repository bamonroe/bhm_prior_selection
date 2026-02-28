"""
Deep investigation of the multiple-zero cases found in the H(r) analysis.

For cases where H(r) = 0 has more than one solution, we need to verify:
1. Is the extra zero a genuine solution, or a solver artifact?
2. At the extra zero, do the implied (phi, eta) actually reproduce ALL three CEs?
3. What do the implied (phi, eta) look like at the spurious zero?
4. Are there physical constraints being violated (phi > 0, eta > 0, valid weights)?

This script performs a thorough verification for the two problematic cases:
- (r=0.8, phi=0.5, eta=1.3) with extra zero near r=0.629
- (r=1.5, phi=0.6, eta=1.0) with extra zero near r=0.550
"""

import numpy as np
from scipy.optimize import fsolve, brentq, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# Copy of primitives from main script
# =====================================================================

def prelec_w(p, phi, eta):
    """Prelec 2-parameter PWF: w(p) = exp(-eta * (-ln p)^phi)."""
    return np.exp(-eta * (-np.log(p))**phi)


def crra_utility(x, r):
    """CRRA utility: u(x;r) = x^(1-r)/(1-r) for r != 1, ln(x) for r = 1."""
    if abs(r - 1.0) < 1e-10:
        return np.log(x)
    return x**(1 - r) / (1 - r)


def compute_rdu(r, phi, eta, a, b, c, q, s):
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)
    d3 = wq
    d2 = ws - wq
    d1 = 1.0 - ws
    return d1 * crra_utility(a, r) + d2 * crra_utility(b, r) + d3 * crra_utility(c, r)


def compute_ce(r, phi, eta, a, b, c, q, s):
    rdu_val = compute_rdu(r, phi, eta, a, b, c, q, s)
    if abs(r - 1.0) < 1e-10:
        return np.exp(rdu_val)
    else:
        inner = (1 - r) * rdu_val
        if inner <= 0:
            return np.nan
        return inner**(1.0 / (1 - r))


def R_ratio(r, a, b, c):
    s = 1 - r
    if abs(s) < 1e-12:
        return np.log(c/b) / np.log(b/a)
    return (c**s - b**s) / (b**s - a**s)


def C_k(r, CE_k, a, b):
    s = 1 - r
    if abs(s) < 1e-12:
        return np.log(CE_k / a) / np.log(b / a)
    return (CE_k**s - a**s) / (b**s - a**s)


# =====================================================================
# Lottery configuration
# =====================================================================

a, b, c = 10.0, 30.0, 50.0
q1, s1 = 0.5, 0.5
q2, s2 = 0.1, 0.9
q3, s3 = 0.4, 0.7


def solve_prelec_from_two_eqs(r_val, CE1, CE2, phi_init=0.7, eta_init=0.9):
    """
    Solve for (phi, eta) from two CE equations. Returns all solutions found
    from multiple starting points.
    """
    R_val = R_ratio(r_val, a, b, c)
    C1_val = C_k(r_val, CE1, a, b)
    C2_val = C_k(r_val, CE2, a, b)

    def equations(params):
        phi_try, eta_try = params
        if phi_try <= 0.001 or eta_try <= 0.001:
            return [1e10, 1e10]
        try:
            ws1 = prelec_w(s1, phi_try, eta_try)
            wq1 = prelec_w(q1, phi_try, eta_try)
            ws2 = prelec_w(s2, phi_try, eta_try)
            wq2 = prelec_w(q2, phi_try, eta_try)
            eq1 = C1_val - ws1 - R_val * wq1
            eq2 = C2_val - ws2 - R_val * wq2
            return [eq1, eq2]
        except:
            return [1e10, 1e10]

    solutions = []
    init_guesses = [
        (phi_init, eta_init),
        (0.3, 0.3), (0.3, 0.5), (0.3, 1.0), (0.3, 2.0),
        (0.5, 0.3), (0.5, 0.5), (0.5, 1.0), (0.5, 2.0),
        (0.7, 0.3), (0.7, 0.7), (0.7, 1.0), (0.7, 1.5),
        (1.0, 0.3), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0),
        (1.3, 0.5), (1.3, 1.0), (1.3, 1.5),
        (1.5, 0.5), (1.5, 1.0), (1.5, 2.0),
        (2.0, 0.5), (2.0, 1.0), (2.0, 2.0),
        (0.1, 0.5), (0.1, 3.0), (3.0, 0.1),
    ]

    for phi0, eta0 in init_guesses:
        try:
            sol = fsolve(equations, [phi0, eta0], full_output=True)
            x, info, ier, msg = sol
            residual = np.max(np.abs(info['fvec']))
            if ier == 1 and x[0] > 0.001 and x[1] > 0.001 and residual < 1e-10:
                # Check if this is a new solution (not already found)
                is_new = True
                for prev_sol in solutions:
                    if abs(x[0] - prev_sol[0]) < 1e-6 and abs(x[1] - prev_sol[1]) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    solutions.append((x[0], x[1], residual))
        except:
            continue

    return solutions


def full_verification(r_val, phi_val, eta_val, CEs_true, label=""):
    """
    Given (r, phi, eta), compute all 3 CEs and compare to truth.
    """
    CE1_true, CE2_true, CE3_true = CEs_true

    CE1_comp = compute_ce(r_val, phi_val, eta_val, a, b, c, q1, s1)
    CE2_comp = compute_ce(r_val, phi_val, eta_val, a, b, c, q2, s2)
    CE3_comp = compute_ce(r_val, phi_val, eta_val, a, b, c, q3, s3)

    err1 = abs(CE1_comp - CE1_true)
    err2 = abs(CE2_comp - CE2_true)
    err3 = abs(CE3_comp - CE3_true)

    print(f"  {label}")
    print(f"    (r, phi, eta) = ({r_val:.6f}, {phi_val:.6f}, {eta_val:.6f})")
    print(f"    CE1: computed={CE1_comp:.6f}, true={CE1_true:.6f}, err={err1:.2e}")
    print(f"    CE2: computed={CE2_comp:.6f}, true={CE2_true:.6f}, err={err2:.2e}")
    print(f"    CE3: computed={CE3_comp:.6f}, true={CE3_true:.6f}, err={err3:.2e}")
    print(f"    Max CE error: {max(err1, err2, err3):.2e}")

    # Check decision weight validity
    for k, (qk, sk) in enumerate([(q1,s1),(q2,s2),(q3,s3)], 1):
        wq = prelec_w(qk, phi_val, eta_val)
        ws = prelec_w(sk, phi_val, eta_val)
        d3 = wq
        d2 = ws - wq
        d1 = 1 - ws
        valid = d1 >= 0 and d2 >= 0 and d3 >= 0
        print(f"    L{k} weights: d1={d1:.4f}, d2={d2:.4f}, d3={d3:.4f} {'OK' if valid else 'INVALID!'}")

    return max(err1, err2, err3)


def investigate_case(r0, phi0, eta0, label):
    """
    Full investigation of a single parameter case.
    """
    print(f"\n{'='*70}")
    print(f"INVESTIGATING: {label}")
    print(f"{'='*70}")

    CE1 = compute_ce(r0, phi0, eta0, a, b, c, q1, s1)
    CE2 = compute_ce(r0, phi0, eta0, a, b, c, q2, s2)
    CE3 = compute_ce(r0, phi0, eta0, a, b, c, q3, s3)
    CEs = (CE1, CE2, CE3)
    print(f"  True: r={r0}, phi={phi0}, eta={eta0}")
    print(f"  CEs: {CE1:.6f}, {CE2:.6f}, {CE3:.6f}")

    # Scan r range and find ALL zeros of H(r)
    r_scan = np.linspace(0.01, 3.0, 3000)

    H_vals = []
    phi_sols = []
    eta_sols = []

    phi_prev, eta_prev = phi0, eta0

    # Forward pass from r0
    idx_r0 = np.argmin(np.abs(r_scan - r0))
    results_dict = {}

    # Forward
    pp, ep = phi0, eta0
    for i in range(idx_r0, len(r_scan)):
        rv = r_scan[i]
        R_val = R_ratio(rv, a, b, c)
        C1_val = C_k(rv, CE1, a, b)
        C2_val = C_k(rv, CE2, a, b)
        C3_val = C_k(rv, CE3, a, b)

        sols = solve_prelec_from_two_eqs(rv, CE1, CE2, phi_init=pp, eta_init=ep)
        if sols:
            # Pick solution closest to previous
            best = min(sols, key=lambda s: (s[0]-pp)**2 + (s[1]-ep)**2)
            phi_r, eta_r = best[0], best[1]
            try:
                ws3 = prelec_w(s3, phi_r, eta_r)
                wq3 = prelec_w(q3, phi_r, eta_r)
                H_val = C3_val - ws3 - R_val * wq3
            except:
                H_val = np.nan
            results_dict[i] = (H_val, phi_r, eta_r, len(sols))
            pp, ep = phi_r, eta_r
        else:
            results_dict[i] = (np.nan, np.nan, np.nan, 0)

    # Backward
    pp, ep = phi0, eta0
    for i in range(idx_r0 - 1, -1, -1):
        rv = r_scan[i]
        R_val = R_ratio(rv, a, b, c)
        C1_val = C_k(rv, CE1, a, b)
        C2_val = C_k(rv, CE2, a, b)
        C3_val = C_k(rv, CE3, a, b)

        sols = solve_prelec_from_two_eqs(rv, CE1, CE2, phi_init=pp, eta_init=ep)
        if sols:
            best = min(sols, key=lambda s: (s[0]-pp)**2 + (s[1]-ep)**2)
            phi_r, eta_r = best[0], best[1]
            try:
                ws3 = prelec_w(s3, phi_r, eta_r)
                wq3 = prelec_w(q3, phi_r, eta_r)
                H_val = C3_val - ws3 - R_val * wq3
            except:
                H_val = np.nan
            results_dict[i] = (H_val, phi_r, eta_r, len(sols))
            pp, ep = phi_r, eta_r
        else:
            results_dict[i] = (np.nan, np.nan, np.nan, 0)

    # Assemble arrays
    H_arr = np.full(len(r_scan), np.nan)
    phi_arr = np.full(len(r_scan), np.nan)
    eta_arr = np.full(len(r_scan), np.nan)
    nsols_arr = np.zeros(len(r_scan), dtype=int)

    for i, (Hv, pv, ev, ns) in results_dict.items():
        H_arr[i] = Hv
        phi_arr[i] = pv
        eta_arr[i] = ev
        nsols_arr[i] = ns

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

    print(f"\n  Found {len(zeros)} zero(s) of H(r):")
    for j, (rz, pz, ez) in enumerate(zeros):
        print(f"\n  --- Zero #{j+1}: r = {rz:.6f} ---")
        full_verification(rz, pz, ez, CEs, label=f"Zero #{j+1}")

    # Also refine the zeros using a direct root finder
    print(f"\n  REFINED ZEROS (using direct 3-equation solver):")
    for j, (rz, pz, ez) in enumerate(zeros):
        # Solve the full system: find (r, phi, eta) matching all 3 CEs
        def full_system(params):
            r_try, phi_try, eta_try = params
            if phi_try <= 0.001 or eta_try <= 0.001 or r_try < -5 or r_try > 5:
                return [1e10, 1e10, 1e10]
            try:
                ce1 = compute_ce(r_try, phi_try, eta_try, a, b, c, q1, s1)
                ce2 = compute_ce(r_try, phi_try, eta_try, a, b, c, q2, s2)
                ce3 = compute_ce(r_try, phi_try, eta_try, a, b, c, q3, s3)
                if any(np.isnan([ce1, ce2, ce3])):
                    return [1e10, 1e10, 1e10]
                return [ce1 - CE1, ce2 - CE2, ce3 - CE3]
            except:
                return [1e10, 1e10, 1e10]

        try:
            sol = fsolve(full_system, [rz, pz, ez], full_output=True)
            x, info, ier, msg = sol
            residual = np.max(np.abs(info['fvec']))
            print(f"\n  Refined zero #{j+1}:")
            print(f"    (r, phi, eta) = ({x[0]:.10f}, {x[1]:.10f}, {x[2]:.10f})")
            print(f"    Residual: {residual:.2e}")
            if residual < 1e-8:
                full_verification(x[0], x[1], x[2], CEs, label=f"Refined #{j+1}")
            else:
                print(f"    SOLVER DID NOT CONVERGE (residual = {residual:.2e})")
        except Exception as e:
            print(f"    Refinement failed: {e}")

    # Now check: are there multiple BRANCHES of (phi, eta) solutions?
    print(f"\n  BRANCH ANALYSIS:")
    print(f"  Checking for multiple (phi,eta) solution branches at selected r values")

    test_r_vals = np.linspace(0.1, 2.5, 25)
    for rv in test_r_vals:
        sols = solve_prelec_from_two_eqs(rv, CE1, CE2)
        if len(sols) > 1:
            print(f"\n    r = {rv:.3f}: {len(sols)} solutions found!")
            for k, (p, e, res) in enumerate(sols):
                R_val = R_ratio(rv, a, b, c)
                C3_val = C_k(rv, CE3, a, b)
                ws3 = prelec_w(s3, p, e)
                wq3 = prelec_w(q3, p, e)
                H_val = C3_val - ws3 - R_val * wq3
                print(f"      Branch {k+1}: phi={p:.6f}, eta={e:.6f}, H={H_val:.6e}")

    return r_scan, H_arr, phi_arr, eta_arr, zeros


def main():
    print("=" * 70)
    print("DEEP INVESTIGATION OF MULTIPLE-ZERO CASES")
    print("=" * 70)

    problematic_cases = [
        (0.8, 0.5, 1.3, "r=0.8, phi=0.5, eta=1.3"),
        (1.5, 0.6, 1.0, "r=1.5, phi=0.6, eta=1.0"),
    ]

    # Also check the "good" cases for completeness
    good_cases = [
        (0.5, 0.7, 0.9, "r=0.5, phi=0.7, eta=0.9"),
        (0.3, 1.2, 0.5, "r=0.3, phi=1.2, eta=0.5"),
        (0.1, 0.9, 0.8, "r=0.1, phi=0.9, eta=0.8"),
    ]

    all_results = {}

    for r0, phi0, eta0, label in problematic_cases + good_cases:
        result = investigate_case(r0, phi0, eta0, label)
        all_results[label] = result

    # Create detailed plots for the problematic cases
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (r0, phi0, eta0, label) in enumerate(problematic_cases):
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
        ax_phi.set_title(f'Implied phi(r)')
        ax_phi.grid(True, alpha=0.3)

        ax_eta.plot(r_scan[valid], eta_arr[valid], 'b-', linewidth=1.5)
        ax_eta.axvline(x=r0, color='r', linewidth=1, linestyle='--')
        ax_eta.axhline(y=eta0, color='r', linewidth=0.5, linestyle=':')
        ax_eta.set_ylabel('eta(r)')
        ax_eta.set_title(f'Implied eta(r)')
        ax_eta.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel('r')

    plt.suptitle('Detailed investigation of multi-zero cases', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/workspace/sympy/H_investigation.png', dpi=150, bbox_inches='tight')
    print(f"\nDetailed plot saved to /workspace/sympy/H_investigation.png")

    # =====================================================================
    # Final summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for label, (r_scan, H_arr, phi_arr, eta_arr, zeros) in all_results.items():
        print(f"\n  {label}:")
        print(f"    Number of H(r)=0 zeros found: {len(zeros)}")
        for j, (rz, pz, ez) in enumerate(zeros):
            print(f"    Zero {j+1}: r={rz:.6f}, phi={pz:.6f}, eta={ez:.6f}")


if __name__ == '__main__':
    main()
