"""
Check P-matrix condition for the 2-point evaluation map:
    W: (phi, eta) -> (w(S2), w(S3))

This is the inner map when the same CDF is used for both lotteries.
The full map Phi = L . W where L is an invertible linear map.
So Phi is injective iff W is injective.
"""
import numpy as np
import sympy as sp
import itertools

phi, eta = sp.symbols('phi eta', positive=True)

S2, S3 = sp.Rational(3, 4), sp.Rational(1, 4)
t2, t3 = -sp.ln(S2), -sp.ln(S3)
w2 = sp.exp(-eta * t2**phi)
w3 = sp.exp(-eta * t3**phi)

# Jacobian of (phi, eta) -> (w(S2), w(S3))
J_w = sp.Matrix([
    [sp.diff(w2, phi), sp.diff(w2, eta)],
    [sp.diff(w3, phi), sp.diff(w3, eta)],
])

print("=" * 70)
print("Jacobian of W: (phi, eta) -> (w(0.75), w(0.25))")
print("=" * 70)

for i in range(2):
    for j in range(2):
        J_w[i, j] = sp.simplify(J_w[i, j])

print(f"\nJ_w[1,1] = dw(0.75)/dphi = {J_w[0,0]}")
print(f"J_w[1,2] = dw(0.75)/deta = {J_w[0,1]}")
print(f"J_w[2,1] = dw(0.25)/dphi = {J_w[1,0]}")
print(f"J_w[2,2] = dw(0.25)/deta = {J_w[1,1]}")

# Sign analysis
print("\n" + "=" * 70)
print("Sign analysis")
print("=" * 70)
print(f"\nt2 = -ln(3/4) = {float(t2):.6f}  =>  ln(t2) = {float(sp.ln(t2)):.6f} < 0")
print(f"t3 = -ln(1/4) = {float(t3):.6f}  =>  ln(t3) = {float(sp.ln(t3)):.6f} > 0")
print()
print("dw/dphi = -eta * t^phi * ln(t) * w")
print("dw/deta = -t^phi * w")
print()
print("J_w[1,1] = -eta * t2^phi * ln(t2) * w2 > 0  (ln(t2) < 0 cancels the minus)")
print("J_w[1,2] = -t2^phi * w2                 < 0  (always)")
print("J_w[2,1] = -eta * t3^phi * ln(t3) * w3 < 0  (ln(t3) > 0)")
print("J_w[2,2] = -t3^phi * w3                 < 0  (always)")
print()
print("Sign pattern of J_w:")
print("  [+  -]")
print("  [-  -]")
print()
print("det(J_w) = eta * t2^phi * w2 * t3^phi * w3 * (ln(t2) - ln(t3))")
print(f"         = (positive) * ({float(sp.ln(t2)-sp.ln(t3)):.6f})")
print("         < 0  always")

# Now consider flipping the SECOND ROW (equivalently, map (w2, -w3)):
print("\n" + "=" * 70)
print("Flip row 2: consider g(phi,eta) = (w(S2), -w(S3))")
print("=" * 70)

J_g = sp.Matrix([
    [J_w[0, 0], J_w[0, 1]],
    [-J_w[1, 0], -J_w[1, 1]],
])

print("\nSign pattern of J_g:")
print("  [+  -]")
print("  [+  +]")
print()
print("P-matrix check:")
print("  J_g[1,1] > 0  (confirmed above)")
print("  J_g[2,2] > 0  (confirmed above)")
print("  det(J_g) = -det(J_w) > 0  (confirmed above)")
print()
print("=> J_g IS a P-matrix!")
print("=> By Gale-Nikaido, g is injective on R_{>0} x R_{>0}")
print("=> W = (g_1, -g_2) is also injective (negation is bijective)")

# Numerical verification
print("\n" + "=" * 70)
print("Numerical P-matrix verification of J_g over grid")
print("=" * 70)

Jg11_fn = sp.lambdify((phi, eta), J_g[0, 0], 'numpy')
Jg22_fn = sp.lambdify((phi, eta), J_g[1, 1], 'numpy')
detg_fn = sp.lambdify((phi, eta), J_g.det(), 'numpy')

phi_vals = np.linspace(0.05, 5.0, 50)
eta_vals = np.linspace(0.05, 5.0, 50)

min_j11 = float('inf')
min_j22 = float('inf')
min_det = float('inf')

for pv in phi_vals:
    for ev in eta_vals:
        j11 = float(Jg11_fn(pv, ev))
        j22 = float(Jg22_fn(pv, ev))
        d = float(detg_fn(pv, ev))
        min_j11 = min(min_j11, j11)
        min_j22 = min(min_j22, j22)
        min_det = min(min_det, d)

print(f"min J_g[1,1] = {min_j11:.8f}  (>0? {'YES' if min_j11 > 0 else 'NO'})")
print(f"min J_g[2,2] = {min_j22:.8f}  (>0? {'YES' if min_j22 > 0 else 'NO'})")
print(f"min det(J_g) = {min_det:.8f}  (>0? {'YES' if min_det > 0 else 'NO'})")

if min_j11 > 0 and min_j22 > 0 and min_det > 0:
    print("\n=> ALL P-matrix conditions satisfied across the grid")
    print("=> W is globally injective by Gale-Nikaido")
    print("=> Phi = L . W is globally injective (L invertible, W injective)")

# Summary
print("\n" + "=" * 70)
print("COMPLETE ARGUMENT")
print("=" * 70)
print("""
With same-CDF, different-outcome Block I design:

1. The map Phi factors: Phi = L . W
   where W: (phi,eta) -> (w(S2), w(S3))  [2-point Prelec evaluation]
   and   L: (w2, w3) -> (v1, v2)          [linear, invertible since mu1 != mu2]

2. The map g = (w(S2), -w(S3)) has Jacobian:
     J_g = [+  -]    with det(J_g) > 0 everywhere
           [+  +]

3. J_g is a P-matrix for all (phi, eta) in R_{>0} x R_{>0}
   (which is a Cartesian product of open intervals)

4. By Gale-Nikaido: g is globally injective
   => W is globally injective (negation is bijection)
   => Phi = L . W is globally injective (composition of injectives)

KEY REQUIREMENT: S2 > 1/e and S3 < 1/e
   i.e., one decumulative probability above 1/e ~ 0.368
   and one below. This ensures ln(t2) < 0 and ln(t3) > 0,
   giving the sign structure [+,-; -,-] in J_w.
""")

print(f"Our CDF: S2 = 0.75 > 1/e = {1/np.e:.4f},  S3 = 0.25 < 1/e.  CHECK.")
