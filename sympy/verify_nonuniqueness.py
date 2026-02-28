"""
Standalone verification that two distinct (r, phi, eta) triples produce
identical certainty equivalents for 3 lotteries.

This is the definitive confirmation that the CE mapping is NOT globally injective.
"""

import numpy as np


def prelec_w(p, phi, eta):
    return np.exp(-eta * (-np.log(p))**phi)


def compute_ce(r, phi, eta, a, b, c, q, s):
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)
    d3 = wq; d2 = ws - wq; d1 = 1.0 - ws
    if abs(r - 1.0) < 1e-10:
        rdu = d1*np.log(a) + d2*np.log(b) + d3*np.log(c)
        return np.exp(rdu)
    rdu = d1 * a**(1-r)/(1-r) + d2 * b**(1-r)/(1-r) + d3 * c**(1-r)/(1-r)
    return ((1-r)*rdu)**(1/(1-r))


a, b, c = 10.0, 30.0, 50.0
lotteries = [
    (0.5, 0.5, "L1 (binary)"),
    (0.1, 0.9, "L2 (middle)"),
    (0.4, 0.7, "L3 (skewed)"),
]

# Two confirmed observationally equivalent parameter triples
pairs = [
    ((0.8, 0.5, 1.3), (0.6291210768, 0.5236235288, 1.4261708352)),
    ((1.5, 0.6, 1.0), (0.5504549172, 0.7098362338, 1.6758117684)),
]

print("=" * 70)
print("VERIFICATION: OBSERVATIONAL EQUIVALENCE IN RDU-PRELEC MODEL")
print("=" * 70)

for pair_idx, ((r1, p1, e1), (r2, p2, e2)) in enumerate(pairs):
    print(f"\n  Pair {pair_idx + 1}:")
    print(f"    Params A: (r={r1}, phi={p1}, eta={e1})")
    print(f"    Params B: (r={r2:.10f}, phi={p2:.10f}, eta={e2:.10f})")
    print()

    for q, s, name in lotteries:
        ce1 = compute_ce(r1, p1, e1, a, b, c, q, s)
        ce2 = compute_ce(r2, p2, e2, a, b, c, q, s)
        diff = abs(ce1 - ce2)
        print(f"    {name}: CE_A = {ce1:.12f}, CE_B = {ce2:.12f}, |diff| = {diff:.2e}")

    # Also check decision weights
    print()
    for q, s, name in lotteries:
        wq1 = prelec_w(q, p1, e1); ws1 = prelec_w(s, p1, e1)
        wq2 = prelec_w(q, p2, e2); ws2 = prelec_w(s, p2, e2)
        d1_A = 1-ws1; d2_A = ws1-wq1; d3_A = wq1
        d1_B = 1-ws2; d2_B = ws2-wq2; d3_B = wq2
        print(f"    {name} weights A: ({d1_A:.6f}, {d2_A:.6f}, {d3_A:.6f})")
        print(f"    {name} weights B: ({d1_B:.6f}, {d2_B:.6f}, {d3_B:.6f})")

    print()
    print(f"    The two parameter sets give DIFFERENT decision weights")
    print(f"    but the SAME certainty equivalents, because the change in")
    print(f"    utility curvature (r) exactly compensates the change in")
    print(f"    probability weighting (phi, eta).")

print("\n" + "=" * 70)
print("CONCLUSION: The 3-lottery CE mapping is NOT globally injective.")
print("HOWEVER: Adding a 4th lottery on the SAME prizes DOES break the")
print("equivalence (as shown in the systematic analysis).")
print("=" * 70)
