"""
Analytical verification of Condition B via shared-probability design (π₃ = π₁).

Key insight: when π₃ = π₁, both lotteries share the same decision weight
w₁ = w(π₁). The CE equations for L₁ and L₃ are:
    c₁^s = w₁·x̄₁^s + (1-w₁)·x̲₁^s    (defines w₁ given c₁, r)
    c₃^s = w₁·x̄₃^s + (1-w₁)·x̲₃^s    (then c₃ = F(c₁, r))

where s = 1 - r. Condition B reduces to dc₃/dr|_{c₁} ≠ 0.

This script computes dc₃/dr ANALYTICALLY (no finite differences) using:
    dc₃/dr = α₃·(R₁ - R₃)
where R_k = h(ρ_k) / [s·(1-ρ_k)] with h(ρ) = f(ρ)/(1-ρ) from Lemma
gamma_monotone. Since h is strictly increasing and ρ₁ ≠ ρ₃ (different
outcome ratios), R₁ ≠ R₃ everywhere.

Also handles the s = 0 (r = 1) limit separately via Taylor expansion.

Previous scripts (shared_probability_condB.py, _v2.py) found apparent sign
changes using finite differences — this script confirms those were FD artifacts
near the s = 0 singularity.
"""

import numpy as np


def compute_R_diff(r, w, xlo1, xhi1, xlo3, xhi3):
    """
    Compute R₁ - R₃ analytically for r ≠ 1.

    dc₃/dr = α₃·(R₁ - R₃) where α₃ > 0 always.
    So sign(dc₃/dr) = sign(R₁ - R₃).

    R_k = -G_k/α_k = -f(ρ_k)/[s·(1-ρ_k)] = -h(ρ_k)/s
    where h(ρ) = f(ρ)/(1-ρ), f(ρ) = P·ln(P) - (1-w)·ρ·ln(ρ),
    P = w + (1-w)·ρ.

    Returns (R1_minus_R3, R1, R3, rho1, rho3).
    """
    s = 1.0 - r

    # ρ_k = (x̲_k/x̄_k)^s
    log_ratio1 = np.log(xlo1 / xhi1)  # negative
    log_ratio3 = np.log(xlo3 / xhi3)  # negative

    # Use log-space to avoid overflow at extreme s
    rho1 = np.exp(s * log_ratio1)
    rho3 = np.exp(s * log_ratio3)

    # Guard against overflow/underflow
    if not (np.isfinite(rho1) and np.isfinite(rho3)):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if rho1 <= 0 or rho3 <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # P_k = w + (1-w)·ρ_k
    P1 = w + (1.0 - w) * rho1
    P3 = w + (1.0 - w) * rho3

    if P1 <= 0 or P3 <= 0:
        return np.nan, np.nan, np.nan, rho1, rho3

    # f(ρ_k) = P_k·ln(P_k) - (1-w)·ρ_k·ln(ρ_k)
    f1 = P1 * np.log(P1) - (1.0 - w) * rho1 * np.log(rho1)
    f3 = P3 * np.log(P3) - (1.0 - w) * rho3 * np.log(rho3)

    if not (np.isfinite(f1) and np.isfinite(f3)):
        return np.nan, np.nan, np.nan, rho1, rho3

    # R_k = -f(ρ_k) / [s·(1-ρ_k)]
    denom1 = s * (1.0 - rho1)
    denom3 = s * (1.0 - rho3)

    if abs(denom1) < 1e-30 or abs(denom3) < 1e-30:
        return np.nan, np.nan, np.nan, rho1, rho3

    R1 = -f1 / denom1
    R3 = -f3 / denom3

    if not (np.isfinite(R1) and np.isfinite(R3)):
        return np.nan, np.nan, np.nan, rho1, rho3

    return R1 - R3, R1, R3, rho1, rho3


def compute_log_utility(w, xlo1, xhi1, xlo3, xhi3):
    """
    Compute dc₃/dr at r = 1 (s = 0, log utility) via Taylor expansion.

    At r = 1: CE = x̲^(1-w)·x̄^w (geometric mean).

    Taylor expansion gives:
        dc₃/dr|_{r=1} = (1-w)·w · ½ · [ln²(x̄₃/x̲₃) - ln²(x̄₁/x̲₁)]
    times a positive prefactor.

    More precisely, the analytical limit is:
        dc₃/dr = CE₃ · w·(1-w) · [½(ln²ρ₃ - ln²ρ₁)]
    where ρ_k = ln(x̄_k/x̲_k).

    Returns dc3_dr.
    """
    # At r=1: CE₃ = x̲₃^(1-w) · x̄₃^w
    ce3 = xlo3**(1.0 - w) * xhi3**w

    ln_ratio1 = np.log(xhi1 / xlo1)
    ln_ratio3 = np.log(xhi3 / xlo3)

    # The Taylor expansion of R_k around s = 0:
    # h(ρ_k)/s where ρ_k = (x̲_k/x̄_k)^s → 1 as s → 0
    # Using L'Hôpital or direct expansion:
    # f(ρ) → 0 as ρ → 1, and the ratio f(ρ)/[s(1-ρ)] has a finite limit.
    #
    # Let ρ = e^{-s·L} where L = ln(x̄/x̲). As s → 0:
    #   1 - ρ ≈ s·L
    #   P = w + (1-w)ρ ≈ 1 - (1-w)·s·L
    #   P·ln(P) ≈ -(1-w)·s·L + ½(1-w)²·s²·L²
    #   (1-w)ρ·ln(ρ) ≈ -(1-w)·s·L + (1-w)·s²·L²
    #   f(ρ) = P·ln(P) - (1-w)ρ·ln(ρ) ≈ -½(1-w)w·s²L² + ...  (wait, wrong)
    #
    # Actually let me compute R_k = -G_k/α_k directly at s → 0.
    # G_k/α_k = f(ρ)/[s(1-ρ)].
    # With ρ = e^{-sL} ≈ 1 - sL + s²L²/2:
    #   1 - ρ ≈ sL
    #   P = 1 - (1-w)sL  (to first order)
    #   ln P ≈ -(1-w)sL - (1-w)²s²L²/2
    #   P·ln P ≈ -(1-w)sL - (1-w)²s²L²/2 + (1-w)²s²L²
    #          = -(1-w)sL + (1-w)²s²L²/2
    #   (1-w)ρ·ln ρ ≈ (1-w)·(1-sL)·(-sL) = -(1-w)sL + (1-w)s²L²
    #   f ≈ (1-w)²s²L²/2 - (1-w)s²L² = -(1-w)ws²L²/2
    #   f/[s·sL] = -(1-w)wL/2
    #   R_k = -f/[s(1-ρ)] ≈ (1-w)wL/2 = w(1-w)ln(x̄_k/x̲_k)/2
    #
    # So R₁ - R₃ → w(1-w)/2 · [ln(x̄₁/x̲₁) - ln(x̄₃/x̲₃)]
    # and dc₃/dr → α₃·w(1-w)/2·[L₁ - L₃]
    #
    # At r=1, α₃ → CE₃·(ln x̄₃ - ln x̲₃) = CE₃·L₃
    # (limit of CE^r·(x̄^s - x̲^s)/s as s→0: CE^1·(ln x̄ - ln x̲))
    #
    # dc₃/dr → CE₃·L₃·w(1-w)/2·(L₁ - L₃)

    alpha3_lim = ce3 * ln_ratio3
    R_diff_lim = w * (1.0 - w) / 2.0 * (ln_ratio1 - ln_ratio3)
    dc3_dr = alpha3_lim * R_diff_lim

    return dc3_dr


def compute_h(rho, w):
    """Compute h(ρ) = f(ρ)/(1-ρ) where f(ρ) = P·ln(P) - (1-w)·ρ·ln(ρ)."""
    if abs(rho - 1.0) < 1e-15 or rho <= 0:
        return np.nan
    P = w + (1.0 - w) * rho
    f = P * np.log(P) - (1.0 - w) * rho * np.log(rho)
    return f / (1.0 - rho)


def verify_h_monotonicity(w_vals, rho_range=(0.001, 0.999, 1000)):
    """Verify h(ρ) is strictly increasing on (0,1) for various w."""
    print("=" * 70)
    print("VERIFICATION: h(ρ) strictly increasing on (0,1)")
    print("=" * 70)

    rho_vals = np.linspace(rho_range[0], rho_range[1], int(rho_range[2]))
    for w in w_vals:
        h_vals = [compute_h(rho, w) for rho in rho_vals]
        diffs = np.diff(h_vals)
        min_diff = np.nanmin(diffs)
        print(f"  w = {w:.2f}: min(Δh) = {min_diff:.2e} "
              f"{'(increasing ✓)' if min_diff > 0 else '(VIOLATION ✗)'}")


def verify_h_monotonicity_rho_gt_1(w_vals, rho_range=(1.001, 10.0, 1000)):
    """
    Verify h(ρ) is strictly increasing for ρ > 1 (i.e. r > 1).

    For ρ > 1: n'(ρ) = f''(ρ)(1-ρ), with f'' < 0 and (1-ρ) < 0,
    so n'(ρ) > 0. Since n(1) = 0 and n increasing for ρ > 1,
    n(ρ) > 0, so h'(ρ) > 0.
    """
    print()
    print("=" * 70)
    print("VERIFICATION: h(ρ) strictly increasing for ρ > 1")
    print("=" * 70)

    rho_vals = np.linspace(rho_range[0], rho_range[1], int(rho_range[2]))
    for w in w_vals:
        h_vals = [compute_h(rho, w) for rho in rho_vals]
        diffs = np.diff(h_vals)
        min_diff = np.nanmin(diffs)
        print(f"  w = {w:.2f}: min(Δh) = {min_diff:.2e} "
              f"{'(increasing ✓)' if min_diff > 0 else '(VIOLATION ✗)'}")


def main():
    xlo1, xhi1 = 10.0, 50.0  # Block I: L₁

    print("=" * 70)
    print("ANALYTICAL VERIFICATION: SHARED-PROBABILITY CONDITION B")
    print("π₃ = π₁ ⟹ w₃ = w₁ ⟹ dc₃/dr = α₃·(R₁ - R₃)")
    print("=" * 70)
    print()
    print("R_k = -G_k/α_k = h(ρ_k)/s  where h strictly increasing (proved)")
    print("ρ_k = (x̲_k/x̄_k)^s,  different outcome ratios ⟹ ρ₁ ≠ ρ₃ ⟹ R₁ ≠ R₃")
    print()

    # --- 1. Verify h monotonicity ---
    w_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    verify_h_monotonicity(w_vals)
    verify_h_monotonicity_rho_gt_1(w_vals)

    # --- 2. Analytical dc₃/dr across dense grid ---
    print()
    print("=" * 70)
    print("ANALYTICAL dc₃/dr SIGN CHECK")
    print("=" * 70)

    l3_designs = [
        (15, 35, "L₃=(15,35), ratio=2.33"),
        (20, 40, "L₃=(20,40), ratio=2.0"),
        (10, 30, "L₃=(10,30), ratio=3.0"),
        (25, 45, "L₃=(25,45), ratio=1.8"),
        (12, 48, "L₃=(12,48), ratio=4.0"),
        (30, 40, "L₃=(30,40), ratio=1.33"),
    ]

    # r values: avoid r = 1 exactly (handled separately)
    r_vals = np.concatenate([
        np.linspace(0.01, 0.99, 100),
        np.linspace(1.01, 5.0, 100),
        np.linspace(5.01, 50.0, 50),
        np.linspace(50.01, 500.0, 30),
    ])

    # w values: the shared decision weight (represents all possible (φ, η))
    w_vals_grid = np.linspace(0.01, 0.99, 100)

    for xlo3, xhi3, label in l3_designs:
        ratio1 = xhi1 / xlo1
        ratio3 = xhi3 / xlo3

        total = 0
        positive = 0
        negative = 0
        min_val = np.inf
        max_val = -np.inf

        for r in r_vals:
            for w in w_vals_grid:
                R_diff, R1, R3, rho1, rho3 = compute_R_diff(
                    r, w, xlo1, xhi1, xlo3, xhi3)
                if np.isnan(R_diff):
                    continue
                total += 1
                # sign(dc₃/dr) = sign(R₁ - R₃) since α₃ > 0
                if R_diff > 1e-15:
                    positive += 1
                elif R_diff < -1e-15:
                    negative += 1
                min_val = min(min_val, R_diff)
                max_val = max(max_val, R_diff)

        if positive > 0 and negative == 0:
            sign_str = "ALL POSITIVE ✓ (consistent sign)"
        elif negative > 0 and positive == 0:
            sign_str = "ALL NEGATIVE ✓ (consistent sign)"
        elif positive > 0 and negative > 0:
            sign_str = f"MIXED ✗ (+{positive}, -{negative})"
        else:
            sign_str = "ALL NEAR ZERO"

        note = ""
        if abs(ratio3 - ratio1) < 1e-10:
            note = " [same ratio — expect R₁=R₃]"
        elif ratio3 < ratio1:
            note = f" [ratio₃ < ratio₁]"
        else:
            note = f" [ratio₃ > ratio₁]"

        print(f"  {label}{note}: {sign_str}")
        print(f"    R₁-R₃ range: [{min_val:.6e}, {max_val:.6e}], {total} pts")

    # --- 3. Log utility limit (r = 1) ---
    print()
    print("=" * 70)
    print("LOG UTILITY LIMIT (r = 1, s = 0)")
    print("=" * 70)
    print("dc₃/dr → CE₃·L₃·w(1-w)/2·(L₁ - L₃)")
    print(f"L₁ = ln({xhi1}/{xlo1}) = {np.log(xhi1/xlo1):.4f}")
    print()

    for xlo3, xhi3, label in l3_designs:
        L1 = np.log(xhi1 / xlo1)
        L3 = np.log(xhi3 / xlo3)
        print(f"  {label}: L₃ = {L3:.4f}, L₁-L₃ = {L1-L3:.4f}")

        for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dc3 = compute_log_utility(w, xlo1, xhi1, xlo3, xhi3)
            sign = "+" if dc3 > 0 else ("-" if dc3 < 0 else "0")
            print(f"    w={w:.1f}: dc₃/dr = {dc3:.6e} ({sign})")
        print()

    # --- 4. Comparison with finite differences near r = 1 ---
    print("=" * 70)
    print("FD vs ANALYTICAL COMPARISON NEAR r = 1")
    print("(demonstrates FD artifacts)")
    print("=" * 70)

    xlo3, xhi3 = 20.0, 40.0

    # For a specific w = 0.3, compute analytical and FD
    w_test = 0.3
    c1_from_w = lambda w, s: (w * xhi1**s + (1 - w) * xlo1**s) ** (1.0 / s) \
        if abs(s) > 1e-10 else xlo1**(1-w) * xhi1**w

    print(f"\nL₃=({xlo3},{xhi3}), w={w_test}")
    print(f"{'r':>8s}  {'analytical':>14s}  {'FD (h=1e-7)':>14s}  {'FD (h=1e-5)':>14s}  {'ratio FD/anal':>14s}")

    for r in [0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.001, 1.01, 1.02, 1.05, 1.1, 1.2]:
        # Analytical: R₁ - R₃ (sign of dc₃/dr)
        if abs(r - 1.0) < 1e-10:
            dc3_anal = compute_log_utility(w_test, xlo1, xhi1, xlo3, xhi3)
        else:
            R_diff, _, _, _, _ = compute_R_diff(r, w_test, xlo1, xhi1, xlo3, xhi3)
            dc3_anal = R_diff  # proportional to dc₃/dr (α₃ > 0 factor omitted)

        # Finite difference
        s = 1.0 - r
        c1 = c1_from_w(w_test, s)
        dc3_fd7 = _fd_dc3_dr(c1, r, xlo1, xhi1, xlo3, xhi3, h=1e-7)
        dc3_fd5 = _fd_dc3_dr(c1, r, xlo1, xhi1, xlo3, xhi3, h=1e-5)

        ratio_str = ""
        if not np.isnan(dc3_fd7) and abs(dc3_anal) > 1e-20:
            ratio_str = f"{dc3_fd7/dc3_anal:.4f}"

        print(f"  {r:6.3f}  {dc3_anal:14.6e}  {dc3_fd7:14.6e}  {dc3_fd5:14.6e}  {ratio_str:>14s}")

    # --- 5. Sanity check: same outcome ratio gives dc₃/dr = 0 ---
    print()
    print("=" * 70)
    print("SANITY CHECK: same outcome ratio → R₁ = R₃ → dc₃/dr = 0")
    print("=" * 70)

    for xlo3_t, xhi3_t in [(20, 100), (5, 25), (10, 50)]:
        ratio = xhi3_t / xlo3_t
        max_R_diff = 0
        for r in [0.5, 1.5, 3.0]:
            for w in [0.2, 0.5, 0.8]:
                R_diff, R1, R3, _, _ = compute_R_diff(
                    r, w, xlo1, xhi1, xlo3_t, xhi3_t)
                if not np.isnan(R_diff):
                    max_R_diff = max(max_R_diff, abs(R_diff))
        print(f"  L₃=({xlo3_t},{xhi3_t}), ratio={ratio:.1f} "
              f"(same as L₁={xhi1/xlo1:.1f}): max|R₁-R₃| = {max_R_diff:.2e}")

    # --- 6. Verify sign(R₁ - R₃) is determined by h monotonicity ---
    print()
    print("=" * 70)
    print("R₁ - R₃ SIGN ANALYSIS")
    print("=" * 70)
    print("h strictly increasing ⟹ sign(R₁-R₃) = sign(h(ρ₁)-h(ρ₃))")
    print("  = sign(ρ₁ - ρ₃)  (since h increasing)")
    print()

    xlo3, xhi3 = 15.0, 35.0
    print(f"L₁=({xlo1},{xhi1}), L₃=({xlo3},{xhi3})")
    print(f"ratio₁ = {xhi1/xlo1:.1f}, ratio₃ = {xhi3/xlo3:.2f}")
    print()

    # For s > 0 (r < 1): ρ_k = (x̲/x̄)^s.
    # ratio₁ > ratio₃ ⟹ x̲₁/x̄₁ < x̲₃/x̄₃ ⟹ ρ₁ < ρ₃ ⟹ h(ρ₁) < h(ρ₃)
    # R_k = -h(ρ_k)/s. For s > 0, h < 0 on (0,1), so -h > 0, R_k > 0.
    # Since h(ρ₁) < h(ρ₃) and both negative: |h(ρ₁)| > |h(ρ₃)|
    # ⟹ R₁ = -h(ρ₁)/s > -h(ρ₃)/s = R₃ ⟹ R₁ - R₃ > 0.
    #
    # For s < 0 (r > 1): ρ_k = (x̲/x̄)^s > 1.
    # ratio₁ > ratio₃ ⟹ x̲₁/x̄₁ < x̲₃/x̄₃ ⟹ (x̲₁/x̄₁)^s > (x̲₃/x̄₃)^s (s<0 flips)
    # ⟹ ρ₁ > ρ₃ > 1.
    # h increasing ⟹ h(ρ₁) > h(ρ₃). For ρ > 1, h(ρ) > 0 (since h(1) = 0
    # from f(1) = 0, and h increasing).
    # R_k = -h(ρ_k)/s. s < 0, h > 0 ⟹ R_k = -h/s > 0.
    # R₁ - R₃ = -(h(ρ₁) - h(ρ₃))/s. h(ρ₁) > h(ρ₃), s < 0 ⟹ R₁ - R₃ > 0.
    # Same sign!

    print("  For all r ≠ 1:")
    print("    ratio₁ > ratio₃ ⟹ R₁ > R₃ ⟹ dc₃/dr > 0")
    print("    ratio₁ < ratio₃ ⟹ R₁ < R₃ ⟹ dc₃/dr < 0")
    print("    (sign is CONSTANT across all (r,w) — no bounded domain needed!)")
    print()

    # Detailed R₁, R₃ values
    print(f"  {'r':>6s}  {'w':>5s}  {'ρ₁':>10s}  {'ρ₃':>10s}  {'R₁':>12s}  {'R₃':>12s}  {'R₁-R₃':>12s}  {'sign':>5s}")
    for r in [0.1, 0.5, 0.9, 1.5, 3.0, 10.0]:
        for w in [0.1, 0.5, 0.9]:
            R_diff, R1, R3, rho1, rho3 = compute_R_diff(
                r, w, xlo1, xhi1, xlo3, xhi3)
            if np.isnan(R_diff):
                continue
            sign = "+" if R_diff > 0 else ("-" if R_diff < 0 else "0")
            print(f"  {r:6.1f}  {w:5.2f}  {rho1:10.6f}  {rho3:10.6f}  "
                  f"{R1:12.6e}  {R3:12.6e}  {R_diff:12.6e}  {sign:>5s}")


def _fd_dc3_dr(c1, r, xlo1, xhi1, xlo3, xhi3, h=1e-7):
    """Finite-difference dc₃/dr (for comparison with analytical)."""
    s_plus = 1 - (r + h)
    s_minus = 1 - (r - h)

    # w₁ from CE at r±h
    def w1_from_ce(c1, xlo1, xhi1, s):
        if abs(s) < 1e-12:
            return np.log(c1 / xlo1) / np.log(xhi1 / xlo1)
        num = c1**s - xlo1**s
        den = xhi1**s - xlo1**s
        if abs(den) < 1e-30:
            return np.nan
        return num / den

    def ce3_from_w1(w1, xlo3, xhi3, s):
        if abs(s) < 1e-12:
            return xlo3**(1 - w1) * xhi3**w1
        val = w1 * xhi3**s + (1 - w1) * xlo3**s
        if val <= 0:
            return np.nan
        return val**(1.0 / s)

    w1p = w1_from_ce(c1, xlo1, xhi1, s_plus)
    w1m = w1_from_ce(c1, xlo1, xhi1, s_minus)
    if np.isnan(w1p) or np.isnan(w1m):
        return np.nan
    if w1p < 0 or w1p > 1 or w1m < 0 or w1m > 1:
        return np.nan

    ce3p = ce3_from_w1(w1p, xlo3, xhi3, s_plus)
    ce3m = ce3_from_w1(w1m, xlo3, xhi3, s_minus)
    if np.isnan(ce3p) or np.isnan(ce3m):
        return np.nan

    return (ce3p - ce3m) / (2 * h)


if __name__ == "__main__":
    main()
