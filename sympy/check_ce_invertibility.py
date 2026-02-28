"""
Symbolic analysis of whether 3 certainty equivalents from 3 distinct
lotteries yield an invertible mapping from (r, phi, eta) to (CE1, CE2, CE3).

We work with B_k = (1-r)*RDU_k = sum_j d_jk * x_j^(1-r), since
CE_k = B_k^(1/(1-r)) is a monotone transform (invertibility equivalent).

Uses SYMBOLIC prizes (a, b, c) to keep expressions compact.
Uses SYMBOLIC cumulative probabilities to show the structural result.

Key result: det(J) = 0 iff the phi and eta columns are proportional,
which happens iff all lotteries share the same cumulative probability
structure (same Prelec inputs).
"""

import sympy as sp
from sympy import (
    symbols, exp, log, Rational, Matrix, simplify,
    pprint, diff, factor, Function, Symbol
)

# =====================================================================
# Symbols
# =====================================================================

r = symbols('r', positive=True)
phi, eta = symbols('phi eta', positive=True)

# Symbolic prizes (shared across lotteries): a < b < c
a, b, c = symbols('a b c', positive=True)

# Symbolic cumulative probabilities for each lottery
# For lottery k with probs (p1, p2, p3) on (a, b, c):
#   q_k = P(X >= c) = p3      (cum. from best, used in Prelec)
#   s_k = P(X >= b) = p2 + p3 (cum. from best, used in Prelec)
# Decision weights: d3 = w(q), d2 = w(s) - w(q), d1 = 1 - w(s)
q1, s1 = symbols('q1 s1', positive=True)  # Lottery 1
q2, s2 = symbols('q2 s2', positive=True)  # Lottery 2
q3, s3 = symbols('q3 s3', positive=True)  # Lottery 3

print("=" * 70)
print("SYMBOLIC CE INVERTIBILITY ANALYSIS")
print("=" * 70)

# =====================================================================
# Primitives
# =====================================================================

def prelec_w(p, phi, eta):
    """Prelec 2-param PWF: w(p) = exp(-eta * (-ln p)^phi)."""
    return exp(-eta * (-log(p))**phi)


def B_expr(r, phi, eta, a, b, c, q, s):
    """
    B = (1-r)*RDU = d1*a^(1-r) + d2*b^(1-r) + d3*c^(1-r)

    q = cumulative prob of best outcome (input to Prelec for d3)
    s = cumulative prob of middle-or-better (input to Prelec for d2)
    """
    wq = prelec_w(q, phi, eta)
    ws = prelec_w(s, phi, eta)

    d3 = wq
    d2 = ws - wq
    d1 = 1 - ws

    return d1 * a**(1 - r) + d2 * b**(1 - r) + d3 * c**(1 - r)


# =====================================================================
# Build B expressions with symbolic prizes and probs
# =====================================================================

print("\n" + "=" * 70)
print("SYMBOLIC B EXPRESSIONS (fully general)")
print("=" * 70)

print("""
Prizes: a < b < c (shared across lotteries)
Lottery k characterized by (q_k, s_k):
  q_k = P(X >= c)   [prob of best outcome or better]
  s_k = P(X >= b)   [prob of middle outcome or better]

B_k = (1 - w(s_k)) * a^s + (w(s_k) - w(q_k)) * b^s + w(q_k) * c^s
where s = 1-r and w = Prelec PWF.
""")

B1 = B_expr(r, phi, eta, a, b, c, q1, s1)
B2 = B_expr(r, phi, eta, a, b, c, q2, s2)
B3 = B_expr(r, phi, eta, a, b, c, q3, s3)

for k, B in enumerate([B1, B2, B3], 1):
    print(f"B{k}: {sp.count_ops(B)} operations")


# =====================================================================
# Prelec derivative structure
# =====================================================================

print("\n" + "=" * 70)
print("PRELEC DERIVATIVE STRUCTURE")
print("=" * 70)

p = symbols('p', positive=True)
w = prelec_w(p, phi, eta)
dw_dphi = diff(w, phi)
dw_deta = diff(w, eta)

print(f"\n  w(p) = exp(-eta * (-ln p)^phi)")
print(f"\n  dw/dphi = {dw_dphi}")
print(f"\n  dw/deta = {dw_deta}")

ratio = simplify(dw_dphi / dw_deta)
print(f"\n  (dw/dphi) / (dw/deta) = {ratio}")
print("""
  The ratio = eta * ln(-ln(p)) DEPENDS ON p.
  => dw/dphi and dw/deta are NOT proportional for different p values.
  => Lotteries with different (q_k, s_k) produce linearly independent
     phi and eta columns in the Jacobian.
""")


# =====================================================================
# Jacobian
# =====================================================================

print("=" * 70)
print("JACOBIAN d(B1,B2,B3)/d(r,phi,eta)")
print("=" * 70)

print("\nComputing partial derivatives...")

J = Matrix([
    [diff(B1, r), diff(B1, phi), diff(B1, eta)],
    [diff(B2, r), diff(B2, phi), diff(B2, eta)],
    [diff(B3, r), diff(B3, phi), diff(B3, eta)],
])

print("Done. Entry sizes:")
for i in range(3):
    for j, param_name in enumerate(['r', 'phi', 'eta']):
        print(f"  J[B{i+1}, {param_name}]: {sp.count_ops(J[i, j])} ops")


# =====================================================================
# Structural analysis of the Jacobian
# =====================================================================

print("\n" + "=" * 70)
print("STRUCTURAL FACTORIZATION")
print("=" * 70)

print("""
Each B_k has the form:
  B_k = (1 - w(s_k))*a^s + (w(s_k) - w(q_k))*b^s + w(q_k)*c^s

where s = 1-r. The derivatives factor as follows:

dB_k/dr = -(1 - w(s_k))*a^s*ln(a) - (w(s_k) - w(q_k))*b^s*ln(b)
          - w(q_k)*c^s*ln(c)
  [Depends on r through a^s, b^s, c^s and on (phi,eta) through weights]

dB_k/dphi = -w'_phi(s_k) * a^s + (w'_phi(s_k) - w'_phi(q_k)) * b^s
            + w'_phi(q_k) * c^s
  [w'_phi denotes dw/dphi. Depends on BOTH (phi,eta) AND (q_k,s_k)]

dB_k/deta = same structure with w'_eta instead of w'_phi

KEY: the phi and eta columns differ only by the factor
     w'_phi(p) / w'_eta(p) = eta * ln(-ln(p))
evaluated at p = q_k and p = s_k.
""")


# =====================================================================
# Explicit phi/eta column relationship
# =====================================================================

print("=" * 70)
print("PHI vs ETA COLUMN PROPORTIONALITY TEST")
print("=" * 70)

# For each row k, check if J[k, phi] / J[k, eta] is the same
# If all rows give the same ratio, columns are proportional => det = 0

print("\nComputing J[k,phi] / J[k,eta] for each row...")

for k in range(3):
    ratio_k = simplify(J[k, 1] / J[k, 2])
    print(f"\n  Row {k+1} (J[B{k+1},phi] / J[B{k+1},eta]):")
    print(f"    = {ratio_k}")

print("""
If q1=q2=q3 and s1=s2=s3 (same probabilities), these ratios are all
equal and the columns are proportional => det(J) = 0.

Otherwise, the ratios differ because ln(-ln(q_k)) and ln(-ln(s_k))
vary across lotteries => columns are linearly independent.
""")


# =====================================================================
# Determinant
# =====================================================================

print("=" * 70)
print("DETERMINANT")
print("=" * 70)

print("\nComputing det(J)...")
det_J = J.det()
n_ops = sp.count_ops(det_J)
print(f"det(J): {n_ops} operations")

# Check if identically zero
print("\nIs det(J) identically zero?")
# Substitute concrete values to test
test_subs = {
    a: 10, b: 30, c: 50, r: Rational(1, 2),
    phi: Rational(7, 10), eta: Rational(9, 10),
    q1: Rational(1, 2), s1: Rational(1, 2),   # L1: binary gamble (1/2, 0, 1/2)
    q2: Rational(1, 10), s2: Rational(9, 10),  # L2: concentrated middle
    q3: Rational(4, 10), s3: Rational(7, 10),  # L3: mildly right-skewed
}

det_val = float(det_J.subs(test_subs))
print(f"  Numerical test (different probs): det(J) = {det_val:.6e}")
if abs(det_val) < 1e-15:
    print("  => ZERO (mapping is singular)")
else:
    print("  => NONZERO (mapping is locally invertible)")

# Same probs control
test_same = test_subs.copy()
test_same.update({
    q2: Rational(1, 2), s2: Rational(1, 2),
    q3: Rational(1, 2), s3: Rational(1, 2),
})
det_same = float(det_J.subs(test_same))
print(f"\n  Control (same probs): det(J) = {det_same:.6e}")
if abs(det_same) < 1e-15:
    print("  => ZERO (as expected: same probs => singular)")


# =====================================================================
# Numerical evaluation across test points
# =====================================================================

print("\n" + "=" * 70)
print("NUMERICAL EVALUATION (checking sign consistency)")
print("=" * 70)

# No-FOSD lottery set:
# L1: binary gamble (1/2, 0, 1/2) => q1=1/2, s1=1/2
# L2: concentrated middle (1/10, 8/10, 1/10) => q2=1/10, s2=9/10
# L3: mildly right-skewed (3/10, 3/10, 4/10) => q3=4/10, s3=7/10
lottery_subs = {
    q1: Rational(1, 2), s1: Rational(1, 2),
    q2: Rational(1, 10), s2: Rational(9, 10),
    q3: Rational(4, 10), s3: Rational(7, 10),
    a: 10, b: 30, c: 50,
}

# Verify no FOSD
print("\nLottery CDFs (F(a), F(b)):")
print("  L1 (binary):  (0.5, 0.5)")
print("  L2 (middle):  (0.1, 0.9)")
print("  L3 (skewed):  (0.3, 0.6)")
print("  All pairs: CDFs cross => no FOSD ✓")

test_params = [
    (Rational(1, 2), 1, 1),
    (Rational(1, 2), Rational(7, 10), Rational(9, 10)),
    (Rational(1, 10), Rational(3, 2), Rational(1, 2)),
    (Rational(3, 4), Rational(1, 2), Rational(3, 2)),
    (Rational(1, 4), Rational(8, 10), Rational(8, 10)),
    (Rational(9, 10), Rational(6, 10), Rational(12, 10)),
    (Rational(1, 5), Rational(4, 10), Rational(4, 10)),
    (Rational(7, 10), Rational(15, 10), Rational(8, 10)),
]

signs = []
for r_val, phi_val, eta_val in test_params:
    subs_dict = {r: r_val, phi: phi_val, eta: eta_val}
    subs_dict.update(lottery_subs)
    det_val = float(det_J.subs(subs_dict))

    sign = "+" if det_val > 0 else "-" if det_val < 0 else "0"
    signs.append(sign)

    print(f"  (r={float(r_val):.2f}, phi={float(phi_val):.2f}, "
          f"eta={float(eta_val):.2f}): det(J) = {det_val:+.4e}  [{sign}]")

unique_signs = set(s for s in signs if s != "0")
print(f"\nSigns: {signs}")
if len(unique_signs) == 1:
    print(f"All same sign ({unique_signs.pop()}) => consistent with "
          f"GLOBAL invertibility on this region")
elif len(unique_signs) > 1:
    print("MIXED signs => locally invertible but the mapping FOLDS "
          "(not globally injective)")


# =====================================================================
# Summary
# =====================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
1. B_k = (1-r)*RDU_k is a monotone transform of CE_k (for fixed r),
   so invertibility of (r,phi,eta) -> (CE1,CE2,CE3) is equivalent to
   invertibility of (r,phi,eta) -> (B1,B2,B3).

2. The Prelec derivative ratio dw/dphi / dw/deta = eta * ln(-ln(p))
   DEPENDS ON p. This is the structural reason why lotteries with
   DIFFERENT cumulative probabilities (q_k, s_k) produce linearly
   independent phi and eta Jacobian columns.

3. If all lotteries share the same (q, s), the phi and eta columns
   are proportional and det(J) = 0 identically: you CANNOT separate
   phi from eta without varying the probability structure.

4. With different probability structures AND no pairwise FOSD:
   - det(J) is generically nonzero => LOCAL invertibility
   - If det(J) maintains constant sign => GLOBAL invertibility

5. Practical requirement: choose reference lotteries with
   (a) different probability structures (to separate phi from eta)
   (b) no stochastic dominance between any pair (to allow free CE
       orderings across all preference types)
""")
