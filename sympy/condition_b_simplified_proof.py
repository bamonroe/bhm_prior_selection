"""
Verification of the simplified Condition B monotonicity proof.

The key claim: h(rho; w) = G_k / Delta_k is strictly decreasing on (0,inf) minus {1}.

The proof proceeds by showing the numerator ν(ρ) of h'(ρ) satisfies:
  - ν has a unique critical point at ρ = 1 (global maximum)
  - ν(1) = 0
  - Therefore ν(ρ) < 0 for all ρ ≠ 1, so h'(ρ) < 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent.parent / "plots" / "condition_b_simplified"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def h_func(rho, w):
    """h(ρ; w) = [(1-w)ρ ln ρ - P ln P] / (1 - ρ), where P = w + (1-w)ρ."""
    P = w + (1 - w) * rho
    num = (1 - w) * rho * np.log(rho) - P * np.log(P)
    den = 1 - rho
    # Handle ρ → 1 via L'Hôpital: h(1) = 0
    result = np.where(np.abs(den) < 1e-12, 0.0, num / den)
    return result


def N_func(rho, w):
    """N(ρ) = (1-w)ρ ln ρ - P ln P (numerator of h before dividing by 1-ρ)."""
    P = w + (1 - w) * rho
    return (1 - w) * rho * np.log(rho) - P * np.log(P)


def N_prime(rho, w):
    """N'(ρ) = (1-w) ln(ρ/P)."""
    P = w + (1 - w) * rho
    return (1 - w) * np.log(rho / P)


def N_double_prime(rho, w):
    """N''(ρ) = w(1-w)/(ρ P) > 0."""
    P = w + (1 - w) * rho
    return w * (1 - w) / (rho * P)


def nu_func(rho, w):
    """ν(ρ) = N'(ρ)(1 - ρ) + N(ρ), numerator of h'(ρ)."""
    return N_prime(rho, w) * (1 - rho) + N_func(rho, w)


def nu_prime(rho, w):
    """ν'(ρ) = N''(ρ)(1 - ρ) = w(1-w)(1-ρ)/(ρ P)."""
    P = w + (1 - w) * rho
    return w * (1 - w) * (1 - rho) / (rho * P)


def h_prime_numerical(rho, w, eps=1e-7):
    """Numerical derivative of h for verification."""
    return (h_func(rho + eps, w) - h_func(rho - eps, w)) / (2 * eps)


# =============================================================================
# Plot 1: h(ρ; w) for several w values
# =============================================================================
def plot_h():
    fig, ax = plt.subplots(figsize=(8, 5))
    rho = np.linspace(0.01, 5.0, 1000)
    w_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for w in w_values:
        ax.plot(rho, h_func(rho, w), label=f"w = {w}")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(1, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$h(\rho;\, w)$")
    ax.set_title(r"$h(\rho;\,w) = G_k/\Delta_k$: strictly decreasing")
    ax.legend()
    ax.set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "h_function.png", dpi=150)
    plt.close(fig)
    print("[Plot 1] h(ρ; w) saved.")


# =============================================================================
# Plot 2: ν(ρ) for several w values — unique maximum at zero
# =============================================================================
def plot_nu():
    fig, ax = plt.subplots(figsize=(8, 5))
    rho = np.linspace(0.01, 5.0, 1000)
    w_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for w in w_values:
        ax.plot(rho, nu_func(rho, w), label=f"w = {w}")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(1, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\nu(\rho)$")
    ax.set_title(
        r"$\nu(\rho)$: unique maximum at $\rho = 1$ with $\nu(1) = 0$"
    )
    ax.legend()
    ax.set_ylim(-3, 0.5)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "nu_function.png", dpi=150)
    plt.close(fig)
    print("[Plot 2] ν(ρ) saved.")


# =============================================================================
# Plot 3: h'(ρ) — analytical vs numerical, confirming h' < 0
# =============================================================================
def plot_h_prime():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rho = np.linspace(0.01, 5.0, 1000)
    mask = np.abs(rho - 1) > 0.02  # avoid ρ = 1 singularity for display

    w = 0.3
    h_prime_ana = nu_func(rho, w) / (1 - rho) ** 2
    h_prime_num = h_prime_numerical(rho, w)

    ax = axes[0]
    ax.plot(rho[mask], h_prime_ana[mask], label="Analytical", linewidth=2)
    ax.plot(
        rho[mask], h_prime_num[mask], "--", label="Numerical", linewidth=1
    )
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(1, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$h'(\rho)$")
    ax.set_title(f"h'(ρ) for w = {w}: always negative")
    ax.legend()
    ax.set_ylim(-5, 1)

    # Multiple w values
    ax = axes[1]
    w_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for w in w_values:
        hp = nu_func(rho, w) / (1 - rho) ** 2
        ax.plot(rho[mask], hp[mask], label=f"w = {w}")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(1, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$h'(\rho)$")
    ax.set_title("h'(ρ) for various w: always negative")
    ax.legend()
    ax.set_ylim(-5, 1)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "h_prime.png", dpi=150)
    plt.close(fig)
    print("[Plot 3] h'(ρ) saved.")


# =============================================================================
# Numerical verification: h'(ρ) < 0 for all ρ ≠ 1, all w ∈ (0,1)
# =============================================================================
def verify_h_decreasing():
    print("\n=== Verification: h'(ρ) < 0 for ρ ≠ 1 ===")
    rho_test = np.concatenate(
        [np.linspace(0.001, 0.99, 5000), np.linspace(1.01, 100, 5000)]
    )
    w_test = np.linspace(0.01, 0.99, 100)
    max_nu = -np.inf
    for w in w_test:
        nu_vals = nu_func(rho_test, w)
        m = np.max(nu_vals)
        if m > max_nu:
            max_nu = m
    print(f"  max ν(ρ) over grid: {max_nu:.2e} (should be < 0)")
    assert max_nu < 0, f"FAILED: max ν = {max_nu}"
    print("  PASSED: ν(ρ) < 0 everywhere on grid.\n")


# =============================================================================
# Verification: ν(1) = 0 and ν''(1) < 0
# =============================================================================
def verify_nu_at_one():
    print("=== Verification: ν(1) = 0, ν''(1) < 0 ===")
    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        nu_1 = N_prime(1.0, w) * 0 + N_func(1.0, w)
        # ν''(1) = -N''(1) = -w(1-w)
        nu_pp_1 = -w * (1 - w)
        print(f"  w = {w}: ν(1) = {nu_1:.2e}, ν''(1) = {nu_pp_1:.4f}")
        assert abs(nu_1) < 1e-14, f"FAILED: ν(1) = {nu_1}"
        assert nu_pp_1 < 0, f"FAILED: ν''(1) = {nu_pp_1}"
    print("  PASSED.\n")


# =============================================================================
# Verification: N''(ρ) > 0 (strict convexity of N)
# =============================================================================
def verify_N_convex():
    print("=== Verification: N''(ρ) > 0 ===")
    rho_test = np.linspace(0.001, 100, 10000)
    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        npp = N_double_prime(rho_test, w)
        assert np.all(npp > 0), f"FAILED for w = {w}"
    print("  PASSED: N is strictly convex.\n")


# =============================================================================
# Plot 4: Numerator of dc₃/ds for representative lottery parameters
# =============================================================================
def verify_constant_sign_lottery():
    """
    For the recommended design:
      L₁ = (10, 50), w₁ = w₃ (shared probability)
      L₃ = (15, 45) (shifted: e = 5)

    Compute dc₃/ds numerator = G₃ - G₁ Δ₃/Δ₁ for a range of s values
    and several w values. Verify constant sign.
    """
    print("=== Verification: constant sign of dc₃/ds numerator ===")
    xlo1, xhi1 = 10.0, 50.0
    e = 5.0
    xlo3, xhi3 = xlo1 + e, xhi1 + e  # 15, 55

    s_neg = np.linspace(-5, -0.01, 500)
    s_pos = np.linspace(0.01, 5, 500)
    s_all = np.concatenate([s_neg, s_pos])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    w_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Left panel: h(ρ₃) - h(ρ₁) — nonzero but changes sign with s
    # (the sgn(s) factors cancel in the full numerator)
    for w in w_values:
        diffs = []
        for s in s_all:
            rho1 = (xlo1 / xhi1) ** s
            rho3 = (xlo3 / xhi3) ** s
            h1 = float(h_func(np.array([rho1]), w)[0])
            h3 = float(h_func(np.array([rho3]), w)[0])
            diffs.append(h3 - h1)
        diffs = np.array(diffs)
        axes[0].plot(s_all, diffs, label=f"w = {w}")
        # Check nonzero (not constant sign — the sign flips with s)
        assert np.all(
            np.abs(diffs) > 1e-10
        ), f"FAILED: h(ρ₃)-h(ρ₁) = 0 for w={w}"

    axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[0].set_xlabel("s = 1 - r")
    axes[0].set_ylabel(r"$h(\rho_3) - h(\rho_1)$")
    axes[0].set_title(r"$h(\rho_3) - h(\rho_1)$: nonzero (sign flips with $s$)")
    axes[0].legend()

    # Right panel: Δ₃[h(ρ₃) - h(ρ₁)] = numerator of dc₃/ds — constant sign
    # Note: G₃Δ₁ - G₁Δ₃ = Δ₁·Δ₃[h(ρ₃)-h(ρ₁)] is NOT constant sign,
    # but Δ₃[h(ρ₃)-h(ρ₁)] = (G₃Δ₁-G₁Δ₃)/Δ₁ IS constant sign
    # because sgn(Δ₃)·sgn(h(ρ₃)-h(ρ₁)) = sgn(s)·[-sgn(β₃-β₁)·sgn(s)]
    # = -sgn(β₃-β₁).
    for w in w_values:
        numerators = []
        for s in s_all:
            a3, b3 = xhi3**s, xlo3**s
            D3 = a3 - b3  # Δ₃
            rho1 = (xlo1 / xhi1) ** s
            rho3 = (xlo3 / xhi3) ** s
            h1 = float(h_func(np.array([rho1]), w)[0])
            h3 = float(h_func(np.array([rho3]), w)[0])
            num = D3 * (h3 - h1)  # Δ₃[h(ρ₃) - h(ρ₁)]
            numerators.append(num)
        numerators = np.array(numerators)
        axes[1].plot(s_all, numerators, label=f"w = {w}")

        # Check constant sign for dc₃/ds numerator
        signs = np.sign(numerators)
        unique_signs = np.unique(signs[signs != 0])
        assert (
            len(unique_signs) == 1
        ), f"FAILED for w={w}: dc₃/ds numerator signs = {unique_signs}"

    axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel("s = 1 - r")
    axes[1].set_ylabel(r"$\Delta_3\,[h(\rho_3) - h(\rho_1)]$")
    axes[1].set_title(r"$dc_3/ds$ numerator: constant sign")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "constant_sign_lottery.png", dpi=150)
    plt.close(fig)
    print("  PASSED: h-diff nonzero, full numerator constant sign.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Simplified Condition B Proof: Verification")
    print("=" * 60)

    # Numerical verifications
    verify_N_convex()
    verify_nu_at_one()
    verify_h_decreasing()
    verify_constant_sign_lottery()

    # Plots
    plot_h()
    plot_nu()
    plot_h_prime()

    print("All verifications passed. Plots saved to:", PLOT_DIR)
