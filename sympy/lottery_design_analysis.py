"""
Analyze what lottery designs achieve global injectivity for the CE mapping.

Two strategies:
1. Add a 4th lottery on the same prizes (10, 30, 50) — overdetermined system
2. Use lotteries with different prizes (still within 10-50 range) — different
   utility inputs break the r-vs-(phi,eta) compensation

We test against the two confirmed observationally equivalent pairs.
"""

import numpy as np
from scipy.optimize import fsolve
from itertools import product


def prelec_w(p, phi, eta):
    return np.exp(-eta * (-np.log(p))**phi)


def compute_ce(r, phi, eta, prizes, q, s):
    """CE for a 3-outcome lottery with prizes (a, b, c) and cum probs (q, s)."""
    a, b, c = prizes
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)
    d3 = wq; d2 = ws - wq; d1 = 1.0 - ws
    if abs(r - 1.0) < 1e-10:
        rdu = d1*np.log(a) + d2*np.log(b) + d3*np.log(c)
        return np.exp(rdu)
    rdu = d1 * a**(1-r)/(1-r) + d2 * b**(1-r)/(1-r) + d3 * c**(1-r)/(1-r)
    return ((1-r)*rdu)**(1/(1-r))


def compute_ces(r, phi, eta, lotteries):
    """Compute CEs for a list of (prizes, q, s) lotteries."""
    return np.array([compute_ce(r, phi, eta, *lot) for lot in lotteries])


# The two confirmed equivalent pairs
pairs = [
    ((0.8, 0.5, 1.3), (0.6291210768, 0.5236235288, 1.4261708352)),
    ((1.5, 0.6, 1.0), (0.5504549172, 0.7098362338, 1.6758117684)),
]

# Original 3 lotteries (same prizes)
base_prizes = (10.0, 30.0, 50.0)
L_original = [
    (base_prizes, 0.5, 0.5),   # L1: binary
    (base_prizes, 0.1, 0.9),   # L2: concentrated middle
    (base_prizes, 0.4, 0.7),   # L3: skewed
]

print("=" * 70)
print("PART 1: ADDING A 4th LOTTERY ON THE SAME PRIZES (10, 30, 50)")
print("=" * 70)

# Candidate 4th lotteries on same prizes
fourth_lotteries_same_prizes = [
    ((10, 30, 50), 0.3, 0.6, "L4a: (0.4, 0.3, 0.3)"),   # q=0.3, s=0.6
    ((10, 30, 50), 0.2, 0.8, "L4b: (0.2, 0.6, 0.2)"),   # q=0.2, s=0.8
    ((10, 30, 50), 0.6, 0.8, "L4c: (0.2, 0.2, 0.6)"),   # q=0.6, s=0.8
    ((10, 30, 50), 0.1, 0.5, "L4d: (0.5, 0.4, 0.1)"),   # q=0.1, s=0.5
    ((10, 30, 50), 0.8, 0.9, "L4e: (0.1, 0.1, 0.8)"),   # q=0.8, s=0.9
    ((10, 30, 50), 0.05, 0.95, "L4f: (0.05, 0.9, 0.05)"),
    ((10, 30, 50), 0.33, 0.67, "L4g: (0.33, 0.34, 0.33)"),
]

for prizes_4, q4, s4, name in fourth_lotteries_same_prizes:
    print(f"\n  {name}  (q={q4}, s={s4})")
    for pair_idx, ((r1, p1, e1), (r2, p2, e2)) in enumerate(pairs):
        ce4_A = compute_ce(r1, p1, e1, prizes_4, q4, s4)
        ce4_B = compute_ce(r2, p2, e2, prizes_4, q4, s4)
        diff = abs(ce4_A - ce4_B)
        sep = "SEPARATES" if diff > 0.001 else "does NOT separate"
        print(f"    Pair {pair_idx+1}: CE_A={ce4_A:.6f}, CE_B={ce4_B:.6f}, "
              f"|diff|={diff:.4f}  [{sep}]")


print("\n\n" + "=" * 70)
print("PART 2: DIFFERENT PRIZES WITHIN 10-50 RANGE")
print("=" * 70)
print("\nKey idea: if lotteries have different prizes, r enters through")
print("different utility differences, breaking the compensation mechanism.")

# Strategy: replace one of the 3 lotteries with one using different prizes
# All prizes stay within [10, 50]
different_prize_lotteries = [
    ((10, 20, 50), 0.5, 0.5, "L_alt1: prizes (10,20,50), binary"),
    ((10, 40, 50), 0.5, 0.5, "L_alt2: prizes (10,40,50), binary"),
    ((10, 20, 40), 0.5, 0.5, "L_alt3: prizes (10,20,40), binary"),
    ((15, 30, 45), 0.5, 0.5, "L_alt4: prizes (15,30,45), binary"),
    ((10, 15, 50), 0.3, 0.7, "L_alt5: prizes (10,15,50), skewed"),
    ((20, 35, 50), 0.4, 0.7, "L_alt6: prizes (20,35,50), skewed"),
    ((10, 25, 35), 0.2, 0.8, "L_alt7: prizes (10,25,35), middle"),
]

print("\n--- Testing: replace L3 with a different-prize lottery ---")
for prizes_alt, q_alt, s_alt, name in different_prize_lotteries:
    print(f"\n  {name}  (q={q_alt}, s={s_alt})")
    for pair_idx, ((r1, p1, e1), (r2, p2, e2)) in enumerate(pairs):
        ce_A = compute_ce(r1, p1, e1, prizes_alt, q_alt, s_alt)
        ce_B = compute_ce(r2, p2, e2, prizes_alt, q_alt, s_alt)
        diff = abs(ce_A - ce_B)
        sep = "SEPARATES" if diff > 0.001 else "does NOT separate"
        print(f"    Pair {pair_idx+1}: CE_A={ce_A:.6f}, CE_B={ce_B:.6f}, "
              f"|diff|={diff:.4f}  [{sep}]")


print("\n\n" + "=" * 70)
print("PART 3: SYSTEMATIC TEST — DO DIFFERENT PRIZES GUARANTEE SEPARATION?")
print("=" * 70)
print("\nFor the worst-case pair (r=1.5 vs r=0.55), test whether ANY")
print("3-lottery set with at least one different-prize lottery fails.\n")

r1, p1, e1 = 1.5, 0.6, 1.0
r2, p2, e2 = 0.5504549172, 0.7098362338, 1.6758117684

# Try to find a 3-lottery set with at least 2 different prize vectors
# that still has an observationally equivalent pair.
# Strategy: for each r-candidate, solve for (phi, eta) from first 2 lotteries,
# check consistency with 3rd.

def find_equivalent(true_params, lotteries, r_grid):
    """Search for an r that produces the same CEs via different (phi, eta)."""
    r0, phi0, eta0 = true_params
    true_ces = compute_ces(r0, phi0, eta0, lotteries)

    found = []
    for r_try in r_grid:
        if abs(r_try - r0) < 0.01:
            continue

        def residual(params):
            phi_t, eta_t = params
            if phi_t <= 0 or eta_t <= 0:
                return [1e10, 1e10, 1e10]
            try:
                ces_t = compute_ces(r_try, phi_t, eta_t, lotteries)
                return list(ces_t - true_ces)
            except:
                return [1e10, 1e10, 1e10]

        # Multiple starting points
        for phi_start, eta_start in [(phi0, eta0), (0.5, 0.5), (1.0, 1.0),
                                      (0.3, 1.5), (1.5, 0.3)]:
            try:
                sol = fsolve(residual, [phi_start, eta_start], full_output=True)
                x, info, ier, msg = sol
                if ier == 1 and x[0] > 0.01 and x[1] > 0.01:
                    res = np.max(np.abs(residual(x)))
                    if res < 1e-8:
                        found.append((r_try, x[0], x[1], res))
                        break
            except:
                pass
    return found


r_grid = np.linspace(0.05, 2.5, 200)

# Test 1: all same prizes (should find equivalents)
print("Config A: All same prizes (10,30,50)")
equivs = find_equivalent((r1, p1, e1), L_original, r_grid)
print(f"  Found {len(equivs)} equivalent(s)")
for r_eq, phi_eq, eta_eq, res in equivs[:3]:
    print(f"    r={r_eq:.4f}, phi={phi_eq:.4f}, eta={eta_eq:.4f} (residual={res:.2e})")

# Test 2: one lottery with different prizes
mixed_configs = [
    ("Config B: (10,30,50) + (10,30,50) + (10,20,50)",
     [L_original[0], L_original[1], ((10, 20, 50), 0.4, 0.7)]),
    ("Config C: (10,30,50) + (10,20,40) + (10,30,50)",
     [L_original[0], ((10, 20, 40), 0.1, 0.9), L_original[2]]),
    ("Config D: (10,30,50) + (10,30,50) + (15,30,45)",
     [L_original[0], L_original[1], ((15, 30, 45), 0.4, 0.7)]),
    ("Config E: (10,20,50) + (10,30,50) + (10,40,50)",
     [((10, 20, 50), 0.5, 0.5), L_original[1], ((10, 40, 50), 0.4, 0.7)]),
    ("Config F: (10,15,50) + (10,30,50) + (10,45,50)",
     [((10, 15, 50), 0.5, 0.5), L_original[1], ((10, 45, 50), 0.4, 0.7)]),
    ("Config G: (10,20,30) + (10,30,50) + (30,40,50)",
     [((10, 20, 30), 0.5, 0.5), (base_prizes, 0.1, 0.9), ((30, 40, 50), 0.4, 0.7)]),
]

for name, config in mixed_configs:
    print(f"\n{name}")
    equivs = find_equivalent((r1, p1, e1), config, r_grid)
    if len(equivs) == 0:
        print(f"  NO equivalents found — GLOBALLY INJECTIVE ✓")
    else:
        print(f"  Found {len(equivs)} equivalent(s)")
        for r_eq, phi_eq, eta_eq, res in equivs[:3]:
            print(f"    r={r_eq:.4f}, phi={phi_eq:.4f}, eta={eta_eq:.4f} (res={res:.2e})")

# Also test pair 1
print("\n\n--- Repeat for Pair 1 true params (r=0.8, phi=0.5, eta=1.3) ---")
r1b, p1b, e1b = 0.8, 0.5, 1.3

for name, config in mixed_configs:
    print(f"\n{name}")
    equivs = find_equivalent((r1b, p1b, e1b), config, r_grid)
    if len(equivs) == 0:
        print(f"  NO equivalents found — GLOBALLY INJECTIVE ✓")
    else:
        print(f"  Found {len(equivs)} equivalent(s)")
        for r_eq, phi_eq, eta_eq, res in equivs[:3]:
            print(f"    r={r_eq:.4f}, phi={phi_eq:.4f}, eta={eta_eq:.4f} (res={res:.2e})")


print("\n\n" + "=" * 70)
print("PART 4: WHY DIFFERENT PRIZES BREAK THE COMPENSATION")
print("=" * 70)
print("""
The compensation mechanism for same-prize lotteries:

  B_k = w(s_k) * Delta_12(r) + w(q_k) * Delta_23(r)

The key: Delta_12 = b^(1-r) - a^(1-r) and Delta_23 = c^(1-r) - b^(1-r)
are the SAME for all lotteries when prizes are shared. So the ratio
R(r) = Delta_23/Delta_12 is a SINGLE function of r.

With different prizes:
  B_k = w(s_k) * Delta_12^(k)(r) + w(q_k) * Delta_23^(k)(r)

Now each lottery has its OWN utility differences. The ratio R_k(r) differs
across lotteries, and moreover R_k(r) and R_j(r) are NOT related by a
simple transformation. This breaks the 1D manifold of compensation.

Formally: with shared prizes, the system has the form
  C_k = w(s_k) + R * w(q_k),  k=1,...,K
where R is a single scalar (function of r). With K=3 equations and
3 unknowns (R, phi, eta), the system is exactly determined, allowing
multiple solutions.

With different prizes:
  C_k = w(s_k) * f_k(r) + w(q_k) * g_k(r),  k=1,...,K
where f_k and g_k are different functions for each k. Now r enters
through multiple independent channels, and the system is effectively
overdetermined in r.
""")


print("=" * 70)
print("PART 5: WHAT ABOUT 2-OUTCOME LOTTERIES?")
print("=" * 70)
print("\nWith 2-outcome lotteries, we have prizes (a, c) with prob (1-p, p).")
print("Only one cumulative probability per lottery (q = p).")
print("B_k = (1-w(q_k)) * a_k^(1-r) + w(q_k) * c_k^(1-r)")
print("If prizes differ: a_k^(1-r) and c_k^(1-r) are different for each k.")

two_outcome_lotteries = [
    ((10, 10, 50), 0.5, 1.0, "2-out A: ($10 w.p. 0.5, $50 w.p. 0.5)"),
    ((10, 10, 30), 0.3, 1.0, "2-out B: ($10 w.p. 0.7, $30 w.p. 0.3)"),
    ((20, 20, 50), 0.7, 1.0, "2-out C: ($20 w.p. 0.3, $50 w.p. 0.7)"),
]

print("\nSame test with 2-outcome lotteries (different prizes within 10-50):")
for pair_idx, ((r1, p1, e1), (r2, p2, e2)) in enumerate(pairs):
    print(f"\n  Pair {pair_idx+1}:")
    for prizes, q, s, name in two_outcome_lotteries:
        ce_A = compute_ce(r1, p1, e1, prizes, q, s)
        ce_B = compute_ce(r2, p2, e2, prizes, q, s)
        diff = abs(ce_A - ce_B)
        sep = "SEPARATES" if diff > 0.001 else "does NOT"
        print(f"    {name}: diff={diff:.6f} [{sep}]")

config_2out = [
    (two_outcome_lotteries[0][:3], two_outcome_lotteries[1][:3],
     two_outcome_lotteries[2][:3])
]

print("\n  Searching for equivalents with 3 two-outcome lotteries (diff prizes):")
lots_2out = [(prizes, q, s) for prizes, q, s, _ in two_outcome_lotteries]
for true_params in [(1.5, 0.6, 1.0), (0.8, 0.5, 1.3)]:
    equivs = find_equivalent(true_params, lots_2out, r_grid)
    rr, pp, ee = true_params
    if len(equivs) == 0:
        print(f"    true=({rr},{pp},{ee}): NO equivalents — INJECTIVE ✓")
    else:
        print(f"    true=({rr},{pp},{ee}): Found {len(equivs)} equivalent(s)")
        for r_eq, phi_eq, eta_eq, res in equivs[:2]:
            print(f"      r={r_eq:.4f}, phi={phi_eq:.4f}, eta={eta_eq:.4f}")
