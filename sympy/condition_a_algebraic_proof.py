"""
ALGEBRAIC PROOF that the 2-point Prelec evaluation map is ALWAYS injective.

Key observation: the binary lottery P-matrix check showed no non-uniqueness
even when the P-matrix condition FAILED (both probs < 1/e). This suggests
injectivity follows from the algebraic structure of the Prelec function,
not from the Gale-Nikaido sufficient condition.

Claim: W: (phi, eta) -> (w(p1; phi, eta), w(p2; phi, eta))
       is injective on R^2_{>0} for ANY p1 != p2 in (0,1).

Proof:
  w(p; phi, eta) = exp(-eta * t^phi)  where t = -ln(p) > 0

  Suppose W(phi1, eta1) = W(phi2, eta2):
    eta1 * t1^phi1 = eta2 * t1^phi2      ... (i)
    eta1 * t2^phi1 = eta2 * t2^phi2      ... (ii)

  Divide (i) by (ii):
    (t1/t2)^phi1 = (t1/t2)^phi2

  Since p1 != p2, t1 != t2, so t1/t2 != 1.
  Therefore phi1 = phi2.  (a^x = a^y with a != 1 implies x = y)

  Substitute back: eta1 * t1^phi1 = eta2 * t1^phi1
  Since t1^phi1 > 0, we get eta1 = eta2.

  QED.

Consequence for Condition A (same-CDF, different-outcome design):
  The map factors as Phi_mu = L(mu) . W
  where W: (phi,eta) -> (w(S2), w(S3)) is the 2-point evaluation
  and L(mu) = [[mu1, 1-mu1], [mu2, 1-mu2]] is invertible (det = mu1-mu2 != 0).
  So Phi_mu = L . W is injective (invertible linear o injective = injective).
"""
import numpy as np
from scipy.optimize import fsolve

def w(p, phi, eta):
    t = -np.log(p)
    return np.exp(-eta * t**phi)

print("=" * 70)
print("VERIFICATION: 2-point evaluation map is always injective")
print("=" * 70)

# Test many probability pairs, including pathological ones
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
    found = False
    for phi_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for eta_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            target = np.array([w(p1, phi_0, eta_0), w(p2, phi_0, eta_0)])
            for pi in np.linspace(0.05, 5.0, 50):
                for ei in np.linspace(0.05, 5.0, 50):
                    if abs(pi-phi_0)<0.1 and abs(ei-eta_0)<0.1:
                        continue
                    def res(x, tgt=target):
                        if x[0]<=0 or x[1]<=0: return [1e10, 1e10]
                        return [w(p1, x[0], x[1])-tgt[0], w(p2, x[0], x[1])-tgt[1]]
                    try:
                        sol = fsolve(res, [pi, ei], full_output=True)
                        if sol[2]==1 and sol[0][0]>0.01 and sol[0][1]>0.01:
                            r = np.linalg.norm(res(sol[0]))
                            dist = np.sqrt((sol[0][0]-phi_0)**2+(sol[0][1]-eta_0)**2)
                            if r < 1e-10 and dist > 0.05:
                                if not found:
                                    print(f"  ({p1}, {p2}) [{label}]: NON-UNIQUE!")
                                    found = True
                                print(f"    ({phi_0:.2f},{eta_0:.2f}) & ({sol[0][0]:.4f},{sol[0][1]:.4f})")
                    except:
                        pass
    if not found:
        print(f"  ({p1}, {p2}) [{label}]: INJECTIVE (confirmed)")

# Now verify the algebraic argument directly
print("\n" + "=" * 70)
print("ALGEBRAIC VERIFICATION")
print("=" * 70)

print("""
For w(p; phi, eta) = exp(-eta * (-ln p)^phi):

If w(p1; phi1, eta1) = w(p1; phi2, eta2)
   w(p2; phi1, eta1) = w(p2; phi2, eta2)

Taking logs:
   eta1 * t1^phi1 = eta2 * t1^phi2   ...(i)
   eta1 * t2^phi1 = eta2 * t2^phi2   ...(ii)

where t1 = -ln(p1), t2 = -ln(p2).

Dividing: (t1/t2)^phi1 = (t1/t2)^phi2

Since p1 != p2, t1 != t2, so t1/t2 != 1.
The function f(x) = a^x is injective for a != 1.
Therefore phi1 = phi2.

Then from (i): eta1 = eta2 (since t1^phi1 > 0).

QED: The 2-point evaluation map is injective for ANY p1 != p2.
""")

# Verify: check that (t1/t2)^phi really is injective in phi
print("Check: a^x is injective for various a != 1:")
for a in [0.1, 0.5, 0.9, 1.1, 2.0, 5.0]:
    x_vals = np.linspace(0.1, 5.0, 1000)
    y_vals = a**x_vals
    is_injective = len(np.unique(np.round(y_vals, 15))) == len(y_vals)
    print(f"  a = {a}: monotonic (injective)? {is_injective}")

# ── Now check same-CDF, different-outcome 3-outcome lotteries ──────────────
print("\n" + "=" * 70)
print("CONSEQUENCE: Same-CDF, different-outcome 3-outcome lotteries")
print("=" * 70)

print("""
Design:
  L1: (10, x2, 50) with probs (p1, p2, p3)  =>  S = (1, S2, S3)
  L2: (10, y2, 50) with probs (p1, p2, p3)  =>  S = (1, S2, S3) [SAME!]

  mu_1 = u_tilde(x2) = [u(x2)-u(10)]/[u(50)-u(10)]   depends on r
  mu_2 = u_tilde(y2) = [u(y2)-u(10)]/[u(50)-u(10)]   depends on r

  v_k = w(S2)*mu_k + w(S3)*(1-mu_k)    k = 1, 2

  This factors as:
    Phi_mu = L(mu) . W
  where:
    W: (phi,eta) -> (w(S2), w(S3))     [2-point eval, INJECTIVE by above]
    L = [[mu1, 1-mu1],                 [invertible iff mu1 != mu2]
         [mu2, 1-mu2]]
    det(L) = mu1 - mu2

  Since W is injective and L is invertible (when x2 != y2, giving mu1 != mu2):
    Phi_mu = L . W is injective.  QED.

No 1/e condition needed! No P-matrix needed! Just algebra.

The 1/e condition was an artifact of the P-matrix sufficient condition.
The actual injectivity is more general.
""")

# Verify: same-CDF 3-outcome with both S values above 1/e
print("Verification: same-CDF 3-outcome with S2=0.75, S3=0.50 (both > 1/e)")
S2, S3 = 0.75, 0.50

def Phi_same_cdf(ph, et, m1, m2):
    ww2 = w(S2, ph, et)
    ww3 = w(S3, ph, et)
    return np.array([ww2*m1 + ww3*(1-m1), ww2*m2 + ww3*(1-m2)])

m1, m2 = 0.5, 0.375  # as before
found = False
for phi_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    for eta_0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        target = Phi_same_cdf(phi_0, eta_0, m1, m2)
        for pi in np.linspace(0.05, 5.0, 40):
            for ei in np.linspace(0.05, 5.0, 40):
                if abs(pi-phi_0)<0.15 and abs(ei-eta_0)<0.15:
                    continue
                def res(x, tgt=target):
                    if x[0]<=0 or x[1]<=0: return [1e10, 1e10]
                    return list(Phi_same_cdf(x[0], x[1], m1, m2) - tgt)
                try:
                    sol = fsolve(res, [pi, ei], full_output=True)
                    if sol[2]==1 and sol[0][0]>0.01 and sol[0][1]>0.01:
                        r = np.linalg.norm(res(sol[0]))
                        dist = np.sqrt((sol[0][0]-phi_0)**2+(sol[0][1]-eta_0)**2)
                        if r < 1e-10 and dist > 0.05:
                            found = True
                            print(f"  NON-UNIQUE: ({phi_0:.2f},{eta_0:.2f}) & ({sol[0][0]:.4f},{sol[0][1]:.4f})")
                except:
                    pass

if not found:
    print(f"  INJECTIVE (confirmed) — no 1/e condition needed!")

# ── Why different CDFs fail ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WHY DIFFERENT CDFs FAIL")
print("=" * 70)
print("""
With different CDFs, the map is:
  v1 = w(a1)*mu1 + w(b1)*(mu2-mu1) + ... + w(c1)*(1-mu_K)
  v2 = w(a2)*mu1 + w(b2)*(mu2-mu1) + ... + w(c2)*(1-mu_K)

where a_k, b_k, c_k are DIFFERENT evaluation points per lottery.
This is NOT a composition of the 2-point eval with a linear map,
because the w-values are at different points.

Equivalently: the map factors as Phi = L . W where W maps to
high dimension (one entry per distinct evaluation point), and L
contracts back to 2D. The contraction can create "folds" that
destroy injectivity.

CONCLUSION:
  Same CDF + different outcomes = 2-point eval + invertible L = INJECTIVE
  Different CDFs = high-dim eval + non-invertible L = may fail

The same-CDF design is the structural requirement, not the 1/e condition.
""")

# ── The FOSD issue: is it really a problem? ────────────────────────────────
print("=" * 70)
print("THE FOSD ISSUE")
print("=" * 70)
print("""
With same CDF and different outcomes (x2 != y2 in 3-outcome lotteries),
one lottery FOSD the other. Is this a problem?

In the experiment:
  - CEs are measured via multiple price lists (MPL)
  - Subjects choose between L_k and a certain amount
  - They DO NOT choose between L1 and L2 directly
  - FOSD between L1 and L2 just constrains CE(L1) > CE(L2) (or <)
  - This constraint is always satisfied by the model

FOSD between Block I lotteries is:
  - NOT a problem for identification
  - NOT a problem for MPL-based CE measurement
  - A potential concern only for between-lottery choice tasks

For MPL-based designs: FOSD is benign.

ALTERNATIVE (no FOSD): Use 4+ outcomes with same CDF but cross
the middle outcomes: L1 = (10, 20, 45, 50), L2 = (10, 30, 35, 50).
But this requires a 3-point evaluation (2->3->2), which breaks the
clean factoring.

The clean solution: accept FOSD in Block I (MPL measurement) and
use the algebraic proof for injectivity.
""")
