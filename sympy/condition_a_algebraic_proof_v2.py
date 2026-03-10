"""
ALGEBRAIC PROOF of 2-point Prelec evaluation map injectivity.

v1 of this script found "non-uniqueness" that turned out to be numerical
artifacts: at extreme parameters, w(p) is so close to 0 or 1 that
different parameter values give abs(w1 - w2) < 1e-10 trivially.

This version uses LOG-SPACE comparison to avoid this issue.
"""
import numpy as np
from scipy.optimize import fsolve

def w(p, phi, eta):
    t = -np.log(p)
    return np.exp(-eta * t**phi)

def log_w(p, phi, eta):
    """Return -eta * t^phi (the log of w, negated)."""
    t = -np.log(p)
    return eta * t**phi

print("=" * 70)
print("ALGEBRAIC PROOF: 2-point Prelec evaluation map")
print("=" * 70)
print("""
Claim: W(phi,eta) = (w(p1), w(p2)) is injective for any p1 != p2.

Proof:
  w(p) = exp(-eta * t^phi)  where t = -ln(p) > 0

  If W(phi1,eta1) = W(phi2,eta2), taking logs:
    eta1 * t1^phi1 = eta2 * t1^phi2    ...(i)
    eta1 * t2^phi1 = eta2 * t2^phi2    ...(ii)

  Dividing (i)/(ii):
    (t1/t2)^phi1 = (t1/t2)^phi2

  Since p1 != p2, t1 != t2, so t1/t2 != 1.
  a^x is injective for a > 0, a != 1.
  Therefore phi1 = phi2.
  Then eta1 = eta2 from (i).  QED.
""")

# ── Verification in LOG SPACE ──────────────────────────────────────────────
print("=" * 70)
print("LOG-SPACE verification (no numerical precision issues)")
print("=" * 70)

test_pairs = [
    (0.9, 0.1, "wide straddle"),
    (0.6, 0.25, "standard straddle"),
    (0.75, 0.50, "both > 1/e"),
    (0.25, 0.10, "both < 1/e"),
    (0.95, 0.90, "both close to 1"),
    (0.10, 0.05, "both close to 0"),
    (0.40, 0.35, "barely different"),
    (0.99, 0.01, "extreme straddle"),
]

for p1, p2, label in test_pairs:
    t1 = -np.log(p1)
    t2 = -np.log(p2)
    found = False

    for phi_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for eta_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            target_log1 = log_w(p1, phi_0, eta_0)  # eta0 * t1^phi0
            target_log2 = log_w(p2, phi_0, eta_0)  # eta0 * t2^phi0

            for pi in np.linspace(0.05, 5.0, 50):
                for ei in np.linspace(0.05, 5.0, 50):
                    if abs(pi-phi_0)<0.1 and abs(ei-eta_0)<0.1:
                        continue
                    # Solve in log space: eta * t^phi = target
                    def res_log(x, tgt1=target_log1, tgt2=target_log2):
                        if x[0] <= 0 or x[1] <= 0:
                            return [1e10, 1e10]
                        return [x[1] * t1**x[0] - tgt1,
                                x[1] * t2**x[0] - tgt2]
                    try:
                        sol = fsolve(res_log, [pi, ei], full_output=True)
                        if (sol[2] == 1 and sol[0][0] > 0.01 and sol[0][1] > 0.01):
                            r = np.linalg.norm(res_log(sol[0]))
                            # Use RELATIVE error
                            rel_err = r / max(abs(target_log1), abs(target_log2), 1e-15)
                            dist = np.sqrt((sol[0][0]-phi_0)**2 + (sol[0][1]-eta_0)**2)
                            if rel_err < 1e-10 and dist > 0.05:
                                found = True
                                print(f"  ({p1}, {p2}) [{label}]: NON-UNIQUE!")
                                print(f"    ({phi_0:.3f},{eta_0:.3f}) & ({sol[0][0]:.6f},{sol[0][1]:.6f})")
                                print(f"    log target: ({target_log1:.6f}, {target_log2:.6f})")
                                print(f"    log result: ({log_w(p1,sol[0][0],sol[0][1]):.6f}, "
                                      f"{log_w(p2,sol[0][0],sol[0][1]):.6f})")
                                break
                    except:
                        pass
                if found:
                    break
            if found:
                break
        if found:
            break

    if not found:
        print(f"  ({p1}, {p2}) [{label}]: INJECTIVE (confirmed)")

# ── Algebraic verification ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Direct algebraic check of the ratio argument")
print("=" * 70)

for p1, p2, label in test_pairs:
    t1 = -np.log(p1)
    t2 = -np.log(p2)
    ratio = t1 / t2

    # If (t1/t2)^phi1 = (t1/t2)^phi2 and t1/t2 != 1, then phi1 = phi2
    # Check: is the function f(x) = ratio^x injective?
    # It is iff ratio != 1 (since f'(x) = ratio^x * ln(ratio) != 0)
    print(f"  ({p1}, {p2}): t1/t2 = {ratio:.6f}, ln(t1/t2) = {np.log(ratio):.6f}"
          f"  {'!= 0 => injective' if abs(np.log(ratio)) > 1e-15 else '= 0 => DEGENERATE'}")

# ── Consequence for same-CDF 3-outcome design ─────────────────────────────
print("\n" + "=" * 70)
print("SAME-CDF 3-OUTCOME DESIGN: Full verification")
print("=" * 70)

# Same CDF: S2=0.75, S3=0.25 (standard design)
S2, S3 = 0.75, 0.25

print(f"\nDesign: same CDF with S2={S2}, S3={S3}")
print(f"  W: (phi,eta) -> (w({S2}), w({S3}))  [2-point eval, INJECTIVE]")
print(f"  L(mu) = [[mu1, 1-mu1], [mu2, 1-mu2]]  [det = mu1 - mu2]")

def Phi_same_cdf(ph, et, m1, m2):
    ww2 = w(S2, ph, et)
    ww3 = w(S3, ph, et)
    return np.array([ww2*m1 + ww3*(1-m1), ww2*m2 + ww3*(1-m2)])

# Test various mu pairs
mu_pairs = [
    (0.5, 0.375, "x2=30 vs y2=25, r=0"),
    (0.3, 0.7, "reversed order"),
    (0.8, 0.2, "wide gap"),
    (0.51, 0.49, "nearly equal"),
]

for m1, m2, label in mu_pairs:
    found = False
    for phi_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for eta_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            target = Phi_same_cdf(phi_0, eta_0, m1, m2)
            for pi in np.linspace(0.05, 5.0, 40):
                for ei in np.linspace(0.05, 5.0, 40):
                    if abs(pi-phi_0)<0.15 and abs(ei-eta_0)<0.15:
                        continue
                    def res(x, tgt=target):
                        if x[0]<=0 or x[1]<=0: return [1e10, 1e10]
                        v = Phi_same_cdf(x[0], x[1], m1, m2)
                        return [v[0]-tgt[0], v[1]-tgt[1]]
                    try:
                        sol = fsolve(res, [pi, ei], full_output=True)
                        if sol[2]==1 and sol[0][0]>0.01 and sol[0][1]>0.01:
                            # Check in w-space for robustness
                            w2_0 = w(S2, phi_0, eta_0)
                            w3_0 = w(S3, phi_0, eta_0)
                            w2_1 = w(S2, sol[0][0], sol[0][1])
                            w3_1 = w(S3, sol[0][0], sol[0][1])
                            w_err = np.sqrt((w2_0-w2_1)**2 + (w3_0-w3_1)**2)
                            dist = np.sqrt((sol[0][0]-phi_0)**2+(sol[0][1]-eta_0)**2)
                            if w_err < 1e-8 and dist > 0.05:
                                found = True
                                print(f"  mu=({m1},{m2}) [{label}]: NON-UNIQUE!")
                                break
                    except:
                        pass
                if found:
                    break
            if found:
                break
        if found:
            break

    if not found:
        print(f"  mu=({m1},{m2}) [{label}]: INJECTIVE (confirmed)")

# ── Also verify with S2, S3 both above 1/e ────────────────────────────────
print("\n--- Also with S2=0.90, S3=0.50 (both > 1/e) ---")
S2, S3 = 0.90, 0.50
for m1, m2, label in mu_pairs[:2]:
    found = False
    for phi_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for eta_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            target = Phi_same_cdf(phi_0, eta_0, m1, m2)
            for pi in np.linspace(0.05, 5.0, 40):
                for ei in np.linspace(0.05, 5.0, 40):
                    if abs(pi-phi_0)<0.15 and abs(ei-eta_0)<0.15:
                        continue
                    def res(x, tgt=target):
                        if x[0]<=0 or x[1]<=0: return [1e10, 1e10]
                        v = Phi_same_cdf(x[0], x[1], m1, m2)
                        return [v[0]-tgt[0], v[1]-tgt[1]]
                    try:
                        sol = fsolve(res, [pi, ei], full_output=True)
                        if sol[2]==1 and sol[0][0]>0.01 and sol[0][1]>0.01:
                            w2_0 = w(S2, phi_0, eta_0)
                            w3_0 = w(S3, phi_0, eta_0)
                            w2_1 = w(S2, sol[0][0], sol[0][1])
                            w3_1 = w(S3, sol[0][0], sol[0][1])
                            w_err = np.sqrt((w2_0-w2_1)**2 + (w3_0-w3_1)**2)
                            dist = np.sqrt((sol[0][0]-phi_0)**2+(sol[0][1]-eta_0)**2)
                            if w_err < 1e-8 and dist > 0.05:
                                found = True
                    except:
                        pass
    if not found:
        print(f"  mu=({m1},{m2}) [{label}]: INJECTIVE (confirmed)")
    else:
        print(f"  mu=({m1},{m2}) [{label}]: NON-UNIQUE!")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPLETE SUMMARY")
print("=" * 70)
print("""
1. The 2-point Prelec evaluation map W(phi,eta) = (w(p1), w(p2)) is
   GLOBALLY INJECTIVE for any p1 != p2 in (0,1).

   Proof: algebraic, using the structure w(p) = exp(-eta * t^phi).
   The ratio equation (t1/t2)^phi1 = (t1/t2)^phi2 forces phi1 = phi2
   whenever t1 != t2 (i.e., p1 != p2). Then eta1 = eta2 follows.

2. Same-CDF 3-outcome design: Phi = L(mu) . W
   - W is injective (by 1.)
   - L is invertible when mu1 != mu2
   - Therefore Phi is injective.

3. Different-CDF designs (3 or 4 outcome): generally FAIL
   - The map factors through higher-dimensional w-space
   - The contraction L: R^n -> R^2 (n > 2) can destroy injectivity
   - Non-uniqueness confirmed numerically.

4. The P-matrix / Gale-Nikaido approach was a red herring for this case.
   The algebraic proof is cleaner and more general:
   - No 1/e condition needed
   - No sign-flip needed
   - Works for ANY two distinct evaluation points

5. The 1/e condition from the P-matrix analysis was SUFFICIENT but
   unnecessarily restrictive. The actual condition is just p1 != p2.

6. FOSD between same-CDF Block I lotteries is unavoidable but benign
   for MPL-based CE measurement.

DESIGN RECOMMENDATION for 3 CEs identifying (r, phi, eta):
  Block I: Two 3-outcome lotteries with same CDF, different middle outcome
           => 2 CEs identify (phi, eta)
  Block II: One lottery with different endpoint outcomes
           => 1 CE identifies r
""")
