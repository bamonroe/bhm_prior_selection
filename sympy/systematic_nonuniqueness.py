"""
Systematic exploration of the non-uniqueness structure of the RDU-Prelec CE mapping.

Key finding from earlier analysis: the mapping (r, phi, eta) -> (CE1, CE2, CE3)
is NOT globally injective. Two distinct parameter triples can produce the same CEs.

This script:
1. Maps out the "observational equivalence curve" in (r, phi, eta) space
2. Investigates what parameter regions give unique vs non-unique inversions
3. Tests whether adding a 4th lottery breaks the non-uniqueness
4. Checks if the non-uniqueness is specific to these lotteries or structural
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# Primitives
# =====================================================================

def prelec_w(p, phi, eta):
    return np.exp(-eta * (-np.log(p))**phi)

def crra_utility(x, r):
    if abs(r - 1.0) < 1e-10:
        return np.log(x)
    return x**(1 - r) / (1 - r)

def compute_ce(r, phi, eta, a, b, c, q, s):
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)
    d3 = wq; d2 = ws - wq; d1 = 1.0 - ws
    rdu_val = d1 * crra_utility(a, r) + d2 * crra_utility(b, r) + d3 * crra_utility(c, r)
    if abs(r - 1.0) < 1e-10:
        return np.exp(rdu_val)
    inner = (1 - r) * rdu_val
    if inner <= 0: return np.nan
    return inner**(1.0 / (1 - r))

# =====================================================================
# Lotteries
# =====================================================================

a, b, c = 10.0, 30.0, 50.0

# Original lotteries
lotteries_3 = [
    (0.5, 0.5, "L1: binary (1/2,0,1/2)"),
    (0.1, 0.9, "L2: middle (1/10,8/10,1/10)"),
    (0.4, 0.7, "L3: skewed (3/10,3/10,4/10)"),
]

# Additional lotteries for testing
extra_lotteries = [
    (0.2, 0.8, "L4: (2/10,6/10,2/10)"),
    (0.3, 0.5, "L5: (5/10,2/10,3/10)"),
    (0.6, 0.6, "L6: (4/10,0,6/10)"),
    (0.05, 0.95, "L7: (5/100,90/100,5/100)"),
    (0.15, 0.45, "L8: (55/100,30/100,15/100)"),
]


def find_observational_equivalent(r0, phi0, eta0, lotteries, r_init_alt=None):
    """
    Given true (r0, phi0, eta0) and a set of lotteries, find an alternative
    (r, phi, eta) that produces the same CEs.
    """
    # Compute true CEs
    true_ces = []
    for q, s, _ in lotteries:
        ce = compute_ce(r0, phi0, eta0, a, b, c, q, s)
        true_ces.append(ce)

    def system(params):
        r_try, phi_try, eta_try = params
        if phi_try <= 0.001 or eta_try <= 0.001:
            return [1e10] * len(lotteries)
        residuals = []
        for i, (q, s, _) in enumerate(lotteries):
            try:
                ce_try = compute_ce(r_try, phi_try, eta_try, a, b, c, q, s)
                if np.isnan(ce_try):
                    residuals.append(1e10)
                else:
                    residuals.append(ce_try - true_ces[i])
            except:
                residuals.append(1e10)
        return residuals

    solutions = []
    # Try many starting points, but avoid the true solution
    init_guesses = []
    for r_init in [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]:
        for phi_init in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]:
            for eta_init in [0.3, 0.5, 0.8, 1.0, 1.3, 1.8]:
                if abs(r_init - r0) > 0.1 or abs(phi_init - phi0) > 0.1 or abs(eta_init - eta0) > 0.1:
                    init_guesses.append((r_init, phi_init, eta_init))

    if r_init_alt is not None:
        init_guesses = [(r_init_alt, 0.5, 1.0), (r_init_alt, 0.7, 0.7),
                        (r_init_alt, 1.0, 1.5), (r_init_alt, 0.3, 2.0)] + init_guesses

    for r_i, p_i, e_i in init_guesses:
        try:
            sol = fsolve(system, [r_i, p_i, e_i], full_output=True)
            x, info, ier, msg = sol
            residual = np.max(np.abs(info['fvec']))
            if ier == 1 and x[1] > 0.01 and x[2] > 0.01 and residual < 1e-10:
                # Check it's genuinely different from the true solution
                dist = np.sqrt((x[0]-r0)**2 + (x[1]-phi0)**2 + (x[2]-eta0)**2)
                if dist > 0.01:
                    # Check it's new
                    is_new = True
                    for prev in solutions:
                        if np.sqrt((x[0]-prev[0])**2 + (x[1]-prev[1])**2 + (x[2]-prev[2])**2) < 0.01:
                            is_new = False
                            break
                    if is_new:
                        solutions.append((x[0], x[1], x[2], residual))
        except:
            continue

    return true_ces, solutions


def main():
    print("=" * 70)
    print("SYSTEMATIC EXPLORATION OF CE MAPPING NON-UNIQUENESS")
    print("=" * 70)

    # =====================================================================
    # Part 1: Map out which parameter regions have observational equivalents
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 1: SCANNING FOR OBSERVATIONAL EQUIVALENCE (3 lotteries)")
    print("=" * 70)

    r_grid = np.arange(0.1, 2.01, 0.1)
    phi_grid = np.arange(0.3, 1.51, 0.2)
    eta_grid = np.arange(0.3, 1.51, 0.2)

    results_3lot = []  # (r0, phi0, eta0, n_equiv, equiv_params)

    for r0 in r_grid:
        for phi0 in phi_grid:
            for eta0 in eta_grid:
                true_ces, equivs = find_observational_equivalent(
                    r0, phi0, eta0, lotteries_3
                )
                if any(np.isnan(true_ces)):
                    continue
                results_3lot.append((r0, phi0, eta0, len(equivs), equivs))

    n_unique = sum(1 for r in results_3lot if r[3] == 0)
    n_nonunique = sum(1 for r in results_3lot if r[3] > 0)

    print(f"\n  Total parameter points tested: {len(results_3lot)}")
    print(f"  Unique (no observational equivalent found): {n_unique}")
    print(f"  Non-unique (at least one equivalent found): {n_nonunique}")
    print(f"  Non-uniqueness rate: {100*n_nonunique/len(results_3lot):.1f}%")

    # Print the non-unique cases
    print(f"\n  Non-unique cases:")
    for r0, phi0, eta0, n_eq, equivs in results_3lot:
        if n_eq > 0:
            for eq_r, eq_phi, eq_eta, eq_res in equivs:
                print(f"    ({r0:.1f}, {phi0:.1f}, {eta0:.1f}) <-> "
                      f"({eq_r:.4f}, {eq_phi:.4f}, {eq_eta:.4f})  "
                      f"[residual: {eq_res:.2e}]")

    # =====================================================================
    # Part 2: Trace the equivalence curve for specific cases
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2: TRACING OBSERVATIONAL EQUIVALENCE CURVES")
    print("=" * 70)

    known_pairs = [
        ((0.8, 0.5, 1.3), (0.6291, 0.5236, 1.4262)),
        ((1.5, 0.6, 1.0), (0.5505, 0.7098, 1.6758)),
    ]

    fig_curve, axes_curve = plt.subplots(1, 3, figsize=(15, 5))

    for pair_idx, ((r1, p1, e1), (r2, p2, e2)) in enumerate(known_pairs):
        print(f"\n  Pair {pair_idx+1}: ({r1},{p1},{e1}) <-> ({r2:.4f},{p2:.4f},{e2:.4f})")

        # Verify the equivalence
        ces1 = [compute_ce(r1, p1, e1, a, b, c, q, s) for q, s, _ in lotteries_3]
        ces2 = [compute_ce(r2, p2, e2, a, b, c, q, s) for q, s, _ in lotteries_3]
        print(f"    CEs from param 1: {[f'{x:.4f}' for x in ces1]}")
        print(f"    CEs from param 2: {[f'{x:.4f}' for x in ces2]}")
        print(f"    Max CE diff: {max(abs(a-b) for a,b in zip(ces1, ces2)):.2e}")

        # Relationship between the pairs
        print(f"    r change: {r1} -> {r2:.4f} (delta = {r2-r1:.4f})")
        print(f"    phi change: {p1} -> {p2:.4f} (delta = {p2-p1:.4f})")
        print(f"    eta change: {e1} -> {e2:.4f} (delta = {e2-e1:.4f})")

        # Plot the pair
        marker = 'o' if pair_idx == 0 else 's'
        color1 = 'blue' if pair_idx == 0 else 'red'
        color2 = 'cyan' if pair_idx == 0 else 'orange'

        axes_curve[0].plot([r1, r2], [p1, p2], f'{color1}', marker=marker, markersize=8, linewidth=2)
        axes_curve[0].set_xlabel('r'); axes_curve[0].set_ylabel('phi')
        axes_curve[0].set_title('r vs phi')

        axes_curve[1].plot([r1, r2], [e1, e2], f'{color1}', marker=marker, markersize=8, linewidth=2)
        axes_curve[1].set_xlabel('r'); axes_curve[1].set_ylabel('eta')
        axes_curve[1].set_title('r vs eta')

        axes_curve[2].plot([p1, p2], [e1, e2], f'{color1}', marker=marker, markersize=8, linewidth=2)
        axes_curve[2].set_xlabel('phi'); axes_curve[2].set_ylabel('eta')
        axes_curve[2].set_title('phi vs eta')

    for ax in axes_curve:
        ax.grid(True, alpha=0.3)
        ax.legend(['Pair 1', 'Pair 2'], fontsize=8)

    plt.tight_layout()
    plt.savefig('/workspace/sympy/equivalence_pairs.png', dpi=150, bbox_inches='tight')
    print(f"\n  Equivalence pairs plot saved.")

    # =====================================================================
    # Part 3: Can adding a 4th lottery break the non-uniqueness?
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 3: TESTING IF A 4TH LOTTERY BREAKS NON-UNIQUENESS")
    print("=" * 70)

    for pair_idx, ((r1, p1, e1), (r2_approx, p2_approx, e2_approx)) in enumerate(known_pairs):
        print(f"\n  Testing pair {pair_idx+1}: ({r1},{p1},{e1})")

        # Refine the second solution
        true_ces = [compute_ce(r1, p1, e1, a, b, c, q, s) for q, s, _ in lotteries_3]
        def system_3(params):
            r, phi, eta = params
            ces = [compute_ce(r, phi, eta, a, b, c, q, s) for q, s, _ in lotteries_3]
            return [c1 - c2 for c1, c2 in zip(ces, true_ces)]
        sol = fsolve(system_3, [r2_approx, p2_approx, e2_approx], full_output=True)
        r2, p2, e2 = sol[0]

        print(f"  Refined equivalent: ({r2:.10f}, {p2:.10f}, {e2:.10f})")

        # Check each extra lottery
        for q_extra, s_extra, lot_name in extra_lotteries:
            ce1_extra = compute_ce(r1, p1, e1, a, b, c, q_extra, s_extra)
            ce2_extra = compute_ce(r2, p2, e2, a, b, c, q_extra, s_extra)
            diff = abs(ce1_extra - ce2_extra)
            separates = "YES - SEPARATES" if diff > 0.01 else "no"
            print(f"    {lot_name}: CE1={ce1_extra:.6f}, CE2={ce2_extra:.6f}, "
                  f"diff={diff:.6e}  {separates}")

    # =====================================================================
    # Part 4: Check structure -- is there always a compensating trade-off?
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 4: STRUCTURE OF THE COMPENSATING TRADE-OFF")
    print("=" * 70)

    print("\n  For each non-unique case, check if higher r always compensates")
    print("  with lower phi (more curvature) or higher eta (more pessimism).\n")

    for pair_idx, ((r1, p1, e1), (r2_approx, p2_approx, e2_approx)) in enumerate(known_pairs):
        # Refine
        true_ces = [compute_ce(r1, p1, e1, a, b, c, q, s) for q, s, _ in lotteries_3]
        def system_3(params):
            r, phi, eta = params
            ces = [compute_ce(r, phi, eta, a, b, c, q, s) for q, s, _ in lotteries_3]
            return [c1 - c2 for c1, c2 in zip(ces, true_ces)]
        sol = fsolve(system_3, [r2_approx, p2_approx, e2_approx])
        r2, p2, e2 = sol

        # Sort so r1 < r2
        if r1 > r2:
            r1, p1, e1, r2, p2, e2 = r2, p2, e2, r1, p1, e1

        print(f"  Pair {pair_idx+1}:")
        print(f"    Low-r:  r={r1:.6f}, phi={p1:.6f}, eta={e1:.6f}")
        print(f"    High-r: r={r2:.6f}, phi={p2:.6f}, eta={e2:.6f}")
        print(f"    Delta r = {r2-r1:+.4f}")
        print(f"    Delta phi = {p2-p1:+.4f} ({'higher' if p2>p1 else 'lower'} curvature)")
        print(f"    Delta eta = {e2-e1:+.4f} ({'higher' if e2>e1 else 'lower'} elevation)")

        # Compute decision weights for each lottery at both parameter sets
        print(f"\n    Decision weights comparison:")
        for q, s, name in lotteries_3:
            wq1 = prelec_w(q, p1, e1); ws1 = prelec_w(s, p1, e1)
            wq2 = prelec_w(q, p2, e2); ws2 = prelec_w(s, p2, e2)
            d31 = wq1; d21 = ws1 - wq1; d11 = 1 - ws1
            d32 = wq2; d22 = ws2 - wq2; d12 = 1 - ws2
            print(f"      {name}:")
            print(f"        Set 1: d1={d11:.4f}, d2={d21:.4f}, d3={d31:.4f}")
            print(f"        Set 2: d1={d12:.4f}, d2={d22:.4f}, d3={d32:.4f}")
            print(f"        Delta: d1={d12-d11:+.4f}, d2={d22-d21:+.4f}, d3={d32-d31:+.4f}")

    # =====================================================================
    # Part 5: Dense scan to find the boundary of uniqueness
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 5: WHICH r VALUES ARE MOST PRONE TO NON-UNIQUENESS?")
    print("=" * 70)

    # For each r0, pick "typical" phi, eta and check if there's an equivalent
    r_dense = np.arange(0.05, 2.51, 0.05)
    uniqueness_by_r = {}

    for r0 in r_dense:
        n_tested = 0
        n_nonunique = 0
        for phi0 in [0.5, 0.7, 0.9, 1.1]:
            for eta0 in [0.5, 0.8, 1.0, 1.3]:
                n_tested += 1
                _, equivs = find_observational_equivalent(r0, phi0, eta0, lotteries_3)
                if len(equivs) > 0:
                    n_nonunique += 1
        uniqueness_by_r[r0] = (n_nonunique, n_tested)

    print(f"\n  r     non-unique/tested")
    for r0 in sorted(uniqueness_by_r.keys()):
        nu, nt = uniqueness_by_r[r0]
        bar = '#' * nu + '.' * (nt - nu)
        print(f"  {r0:.2f}    {nu}/{nt}  {bar}")

    # =====================================================================
    # Part 6: Different lottery configurations
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 6: LOTTERY DESIGN -- WHICH CONFIGURATIONS GIVE UNIQUENESS?")
    print("=" * 70)

    # Test several 3-lottery configurations for a known non-unique case
    r0_test, phi0_test, eta0_test = 0.8, 0.5, 1.3

    lottery_configs = [
        # Original
        [lotteries_3[0], lotteries_3[1], lotteries_3[2]],
        # Replace L3 with various alternatives
        [lotteries_3[0], lotteries_3[1], (0.2, 0.8, "L_alt1")],
        [lotteries_3[0], lotteries_3[1], (0.3, 0.5, "L_alt2")],
        [lotteries_3[0], lotteries_3[1], (0.6, 0.6, "L_alt3")],
        [lotteries_3[0], lotteries_3[1], (0.05, 0.95, "L_alt4")],
        [lotteries_3[0], lotteries_3[1], (0.15, 0.45, "L_alt5")],
        # Use completely different sets
        [(0.2, 0.3, "La"), (0.1, 0.8, "Lb"), (0.5, 0.7, "Lc")],
        [(0.3, 0.6, "Ld"), (0.05, 0.95, "Le"), (0.4, 0.5, "Lf")],
        [(0.1, 0.5, "Lg"), (0.3, 0.9, "Lh"), (0.6, 0.8, "Li")],
    ]

    for config_idx, config in enumerate(lottery_configs):
        names = [name for _, _, name in config]
        _, equivs = find_observational_equivalent(r0_test, phi0_test, eta0_test, config)
        status = "UNIQUE" if len(equivs) == 0 else f"NON-UNIQUE ({len(equivs)} equiv)"
        if len(equivs) > 0:
            eq = equivs[0]
            print(f"  Config {config_idx+1} [{', '.join(names)}]: {status}")
            print(f"    Equivalent: ({eq[0]:.4f}, {eq[1]:.4f}, {eq[2]:.4f})")
        else:
            print(f"  Config {config_idx+1} [{', '.join(names)}]: {status}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print("""
    1. The mapping (r, phi, eta) -> (CE1, CE2, CE3) is NOT globally injective
       for 3 lotteries on the same prize set {a, b, c}.

    2. Observational equivalence occurs: two distinct parameter triples can
       produce identical certainty equivalents for all 3 lotteries.

    3. Verified examples:
       - (0.8, 0.5, 1.3) and (0.6291, 0.5236, 1.4262) give same 3 CEs
       - (1.5, 0.6, 1.0) and (0.5505, 0.7098, 1.6758) give same 3 CEs

    4. The compensating mechanism: higher risk aversion (r) is compensated
       by changes in probability weighting (phi, eta) that adjust decision
       weights to maintain the same CE values.

    5. This is a fundamental limitation: 3 CEs from 3 lotteries on a
       common 3-outcome prize set provide 3 equations in 3 unknowns,
       but the system is not globally injective due to the nonlinear
       interaction between CRRA curvature and Prelec probability weighting.
    """)


if __name__ == '__main__':
    main()
