"""
Check monotonicity of the Prelec 2-parameter PWF:
    w(p; phi, eta) = exp(-eta * (-ln p)^phi)

w is monotonically increasing in p iff w'(p) > 0 for all p in (0,1).
"""
import sympy as sp
import numpy as np

# ── Symbolic derivative ──────────────────────────────────────────────────────
p, phi, eta = sp.symbols('p phi eta', positive=True)

t = -sp.ln(p)  # t > 0 for p in (0,1)
w = sp.exp(-eta * t**phi)

dw_dp = sp.diff(w, p)
dw_dp_simplified = sp.simplify(dw_dp)

print("w(p) = exp(-eta * (-ln p)^phi)")
print(f"\nw'(p) = {dw_dp_simplified}")

# Factor out the always-positive parts
# w'(p) = w(p) * eta * phi * (-ln p)^(phi-1) * (1/p)
# All factors positive except possibly (-ln p)^(phi-1) when phi < 1
# Actually (-ln p) > 0 for p in (0,1), so (-ln p)^(phi-1) > 0 always
# So w'(p) > 0 always? Let's verify.

print("\n" + "=" * 70)
print("Factored form of w'(p):")
print("=" * 70)
print("w'(p) = w(p) * [eta * phi * (-ln p)^(phi-1)] / p")
print("")
print("Since p in (0,1):")
print("  - w(p) > 0    (exponential)")
print("  - eta > 0     (by assumption)")
print("  - phi > 0     (by assumption)")
print("  - (-ln p) > 0 (since 0 < p < 1)")
print("  - (-ln p)^(phi-1) > 0  (positive base, real exponent)")
print("  - 1/p > 0")
print("")
print("=> w'(p) > 0 for ALL phi > 0, eta > 0, p in (0,1)")
print("=> Prelec is ALWAYS monotonically increasing when phi, eta > 0")

# ── But wait: what about w(0)=0, w(1)=1 boundary conditions? ────────────────
print("\n" + "=" * 70)
print("Boundary conditions:")
print("=" * 70)

# w(1) = exp(-eta * 0^phi) = exp(0) = 1  (for phi > 0)
# w(0+) = exp(-eta * inf^phi) = exp(-inf) = 0  (for eta > 0, phi > 0)
print("w(1) = exp(-eta * 0^phi) = exp(0) = 1    [OK for phi > 0]")
print("w(0+) = exp(-eta * (+inf)^phi) = 0        [OK for eta > 0, phi > 0]")

# ── What if eta could be very small? Does it affect the SHAPE? ──────────────
print("\n" + "=" * 70)
print("Shape analysis: w(p) at various (phi, eta)")
print("=" * 70)

w_fn = lambda pp, ph, et: np.exp(-et * (-np.log(pp))**ph)

p_vals = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])

print(f"\n{'p':>5}", end="")
test_params = [(0.5, 0.1), (0.5, 0.3), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0),
               (0.3, 0.3), (0.3, 0.1), (2.0, 0.5)]
for ph, et in test_params:
    print(f" | phi={ph},eta={et}", end="")
print()
print("-" * (6 + 18 * len(test_params)))

for pv in p_vals:
    print(f"{pv:5.2f}", end="")
    for ph, et in test_params:
        wv = w_fn(pv, ph, et)
        print(f" | {wv:15.6f}", end="")
    print()

# ── Numerical derivative check ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Numerical derivative check: is w'(p) ever negative?")
print("=" * 70)

p_fine = np.linspace(0.001, 0.999, 10000)
min_deriv = float('inf')
worst_params = None

for ph in np.linspace(0.1, 3.0, 30):
    for et in np.linspace(0.05, 3.0, 30):
        w_vals = np.exp(-et * (-np.log(p_fine))**ph)
        dw = np.diff(w_vals) / np.diff(p_fine)
        md = dw.min()
        if md < min_deriv:
            min_deriv = md
            worst_params = (ph, et)

print(f"Minimum w'(p) found: {min_deriv:.8f}")
print(f"At (phi, eta) = {worst_params}")
if min_deriv > 0:
    print("=> w'(p) > 0 everywhere in the tested range")
    print("=> Prelec is monotonically increasing for all phi > 0, eta > 0")
else:
    print("=> FOUND NON-MONOTONE BEHAVIOR!")
