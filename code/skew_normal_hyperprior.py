"""
Skew-Normal Hyperprior Selection Example
=========================================

This module demonstrates the prior selection methodology for a 3-parameter
Skew-Normal distribution, as described in Appendix B of the paper.

Overview
--------
The methodology finds hyperprior parameters such that the unconditional
(marginal) distribution of θ matches a specified "direct prior". For the
Skew-Normal, this requires matching 3 cumulants using the law of total cumulance.

The Skew-Normal Distribution
----------------------------
Density: f(x; ξ, ω, α) = (2/ω) φ((x-ξ)/ω) Φ(α(x-ξ)/ω)

Parameters:
    - ξ (xi): location parameter (real)
    - ω (omega): scale parameter (positive)
    - α (alpha): shape parameter (real, controls skewness)

Cumulants:
    - κ₁ = ξ + ω·δ·√(2/π)           where δ = α/√(1+α²)
    - κ₂ = ω²(1 - 2δ²/π)
    - κ₃ = (4-π)/2 · (δ√(2/π))³ · ω³

Hierarchical Structure
----------------------
    θ ~ Skew-Normal(ξ, ω, α)    [individual-level]

    ξ ~ Normal(μ_ξ, τ_ξ²)       [location hyperprior]
    ω² ~ Inverse-Gamma(α_ω, β_ω) [scale hyperprior]
    α ~ Normal(μ_α, τ_α²)       [shape hyperprior]

This gives 6 hyperprior parameters. The 3 cumulant matching conditions
determine 3 of them, leaving 3 degrees of freedom for researcher choice.

Researcher-Specified Parameters (3 DOF)
---------------------------------------
    - ρ (rho): Variance partition ratio, τ_ξ² = ρ × κ₂*
    - α_ω (alpha_omega): Shape parameter for ω² Inverse-Gamma (concentration)
    - τ_α² (tau_alpha_sq): Variance of α hyperprior (shape uncertainty)

Determined Parameters (solved numerically)
------------------------------------------
    - μ_ξ: Mean of ξ hyperprior
    - β_ω: Scale parameter for ω² Inverse-Gamma
    - μ_α: Mean of α hyperprior

Usage
-----
    from skew_normal_hyperprior import run_skew_normal_example

    direct_prior = {'xi': 0.0, 'omega': 1.0, 'alpha': 2.0}
    researcher_choices = {'rho': 0.3, 'alpha_omega': 6.0, 'tau_alpha_sq': 0.25}

    hyperpriors = run_skew_normal_example(
        direct_prior, researcher_choices, output_dir='plots/skew_normal'
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy import stats


# =============================================================================
# Type Definitions
# =============================================================================

class DirectPrior(TypedDict):
    """Direct prior parameters for Skew-Normal distribution."""
    xi: float       # Location parameter (ξ*)
    omega: float    # Scale parameter (ω* > 0)
    alpha: float    # Shape parameter (α*, controls skewness)


class ResearcherChoices(TypedDict):
    """Researcher-specified degrees of freedom for hyperprior selection."""
    rho: float           # Variance partition ratio (0 < ρ < 1)
    alpha_omega: float   # IG shape for ω² (α_ω > 2 for finite variance)
    tau_alpha_sq: float  # Variance of α hyperprior (τ_α² > 0)


class Hyperpriors(TypedDict):
    """Complete hyperprior specification (6 parameters)."""
    mu_xi: float        # Mean of ξ ~ Normal(μ_ξ, τ_ξ²)
    tau_xi_sq: float    # Variance of ξ hyperprior
    alpha_omega: float  # Shape of ω² ~ IG(α_ω, β_ω)
    beta_omega: float   # Scale of ω² hyperprior
    mu_alpha: float     # Mean of α ~ Normal(μ_α, τ_α²)
    tau_alpha_sq: float # Variance of α hyperprior


# Type alias for numpy float arrays
FloatArray = NDArray[np.floating]


# =============================================================================
# Skew-Normal Distribution Functions
# =============================================================================

def delta_fn(alpha: float | FloatArray) -> float | FloatArray:
    """
    Compute δ(α) = α / √(1 + α²).

    This function appears in all Skew-Normal moment formulas. It maps
    the shape parameter α ∈ ℝ to δ ∈ (-1, 1).

    Properties:
        - δ(0) = 0 (symmetric case)
        - δ → ±1 as α → ±∞
        - δ(-α) = -δ(α)
    """
    return alpha / np.sqrt(1 + alpha**2)


def skew_normal_pdf(
    x: float | FloatArray,
    xi: float,
    omega: float,
    alpha: float
) -> float | FloatArray:
    """
    Skew-Normal probability density function.

    f(x; ξ, ω, α) = (2/ω) φ((x-ξ)/ω) Φ(α(x-ξ)/ω)

    where φ is standard normal PDF and Φ is standard normal CDF.

    Args:
        x: Evaluation point(s)
        xi: Location parameter
        omega: Scale parameter (> 0)
        alpha: Shape parameter

    Returns:
        Density value(s) at x
    """
    z = (x - xi) / omega
    return (2 / omega) * stats.norm.pdf(z) * stats.norm.cdf(alpha * z)


def skew_normal_cumulants(
    xi: float,
    omega: float,
    alpha: float
) -> tuple[float, float, float]:
    """
    Compute the first 3 cumulants of a Skew-Normal distribution.

    For SN(ξ, ω, α) with δ = α/√(1+α²):
        κ₁ = ξ + ω·δ·√(2/π)           [mean]
        κ₂ = ω²(1 - 2δ²/π)            [variance]
        κ₃ = (4-π)/2 · (δ√(2/π))³ω³   [third central moment]

    Note: For n ≤ 3, cumulants equal central moments.

    Args:
        xi: Location parameter
        omega: Scale parameter
        alpha: Shape parameter

    Returns:
        Tuple (κ₁, κ₂, κ₃) of first three cumulants
    """
    delta = delta_fn(alpha)

    # Mean (first cumulant)
    kappa1 = xi + omega * delta * np.sqrt(2 / np.pi)

    # Variance (second cumulant)
    kappa2 = omega**2 * (1 - 2 * delta**2 / np.pi)

    # Third central moment (third cumulant)
    kappa3 = ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi))**3 * omega**3

    return kappa1, kappa2, kappa3


# =============================================================================
# Monte Carlo Sampling from Hyperpriors
# =============================================================================

def sample_hyperpriors(
    n_samples: int,
    mu_xi: float,
    tau_xi_sq: float,
    alpha_omega: float,
    beta_omega: float,
    mu_alpha: float,
    tau_alpha_sq: float,
    seed: int | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Sample from the hyperprior distributions.

    Hyperprior structure:
        ξ ~ Normal(μ_ξ, τ_ξ²)
        ω² ~ Inverse-Gamma(α_ω, β_ω)
        α ~ Normal(μ_α, τ_α²)

    All distributions allow direct sampling (no MCMC needed).

    Args:
        n_samples: Number of samples to draw
        mu_xi: Mean of ξ hyperprior
        tau_xi_sq: Variance of ξ hyperprior
        alpha_omega: Shape parameter for ω² Inverse-Gamma
        beta_omega: Scale parameter for ω² Inverse-Gamma
        mu_alpha: Mean of α hyperprior
        tau_alpha_sq: Variance of α hyperprior
        seed: Random seed for reproducibility

    Returns:
        Tuple (ξ_samples, ω_samples, α_samples), each of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # ξ ~ Normal(μ_ξ, τ_ξ²)
    xi_samples = rng.normal(mu_xi, np.sqrt(tau_xi_sq), n_samples)

    # ω² ~ Inverse-Gamma(α_ω, β_ω)
    # Sampling: if X ~ Gamma(α, 1/β), then 1/X ~ IG(α, β)
    omega_sq_samples = 1.0 / rng.gamma(alpha_omega, 1.0 / beta_omega, n_samples)
    omega_samples = np.sqrt(omega_sq_samples)

    # α ~ Normal(μ_α, τ_α²)
    alpha_samples = rng.normal(mu_alpha, np.sqrt(tau_alpha_sq), n_samples)

    return xi_samples, omega_samples, alpha_samples


# =============================================================================
# Law of Total Cumulance - Monte Carlo Integration
# =============================================================================

def compute_implied_cumulants(
    mu_xi: float,
    tau_xi_sq: float,
    alpha_omega: float,
    beta_omega: float,
    mu_alpha: float,
    tau_alpha_sq: float,
    n_samples: int = 50000,
    seed: int = 42
) -> tuple[float, float, float]:
    """
    Compute implied cumulants of the unconditional distribution via Monte Carlo.

    Uses the law of total cumulance:
        κ₁(Y) = E[κ₁(Y|X)]
        κ₂(Y) = E[κ₂(Y|X)] + Var[κ₁(Y|X)]
        κ₃(Y) = E[κ₃(Y|X)] + 3·Cov[κ₁(Y|X), κ₂(Y|X)] + κ₃(κ₁(Y|X))

    Here Y = θ and X = (ξ, ω, α).

    The conditional cumulants of θ|ξ,ω,α ~ SN(ξ,ω,α) are:
        κ₁(θ|X) = ξ + ω·δ(α)·√(2/π)
        κ₂(θ|X) = ω²(1 - 2δ(α)²/π)
        κ₃(θ|X) = (4-π)/2 · (δ(α)√(2/π))³ · ω³

    Args:
        mu_xi: Mean of ξ hyperprior
        tau_xi_sq: Variance of ξ hyperprior
        alpha_omega: Shape of ω² Inverse-Gamma
        beta_omega: Scale of ω² Inverse-Gamma
        mu_alpha: Mean of α hyperprior
        tau_alpha_sq: Variance of α hyperprior
        n_samples: Number of Monte Carlo samples
        seed: Random seed

    Returns:
        Tuple (κ₁_implied, κ₂_implied, κ₃_implied)
    """
    # Sample from hyperpriors: (ξ, ω, α)
    xi, omega, alpha = sample_hyperpriors(
        n_samples, mu_xi, tau_xi_sq, alpha_omega, beta_omega,
        mu_alpha, tau_alpha_sq, seed=seed
    )

    # Compute δ(α) for each sample
    delta = delta_fn(alpha)

    # Conditional cumulants (vectorized over samples)
    cond_mean = xi + omega * delta * np.sqrt(2 / np.pi)
    cond_var = omega**2 * (1 - 2 * delta**2 / np.pi)
    cond_kappa3 = ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi))**3 * omega**3

    # Law of total cumulance
    # κ₁ = E[conditional mean]
    kappa1_implied = float(np.mean(cond_mean))

    # κ₂ = E[conditional variance] + Var[conditional mean]
    kappa2_implied = float(np.mean(cond_var) + np.var(cond_mean))

    # κ₃ = E[cond_κ₃] + 3·Cov[cond_mean, cond_var] + κ₃(cond_mean)
    cov_mean_var = np.cov(cond_mean, cond_var)[0, 1]
    kappa3_cond_mean = np.mean((cond_mean - np.mean(cond_mean))**3)
    kappa3_implied = float(np.mean(cond_kappa3) + 3 * cov_mean_var + kappa3_cond_mean)

    return kappa1_implied, kappa2_implied, kappa3_implied


# =============================================================================
# Numerical Root Finding for Hyperprior Parameters
# =============================================================================

def solve_hyperpriors(
    direct_prior: DirectPrior,
    researcher_choices: ResearcherChoices,
    n_samples: int = 50000,
    seed: int = 42
) -> Hyperpriors:
    """
    Solve for determined hyperprior parameters given researcher choices.

    Given:
        - Direct prior SN(ξ*, ω*, α*) → target cumulants (κ₁*, κ₂*, κ₃*)
        - Researcher choices (ρ, α_ω, τ_α²)

    Solves for:
        - μ_ξ, β_ω, μ_α (3 equations, 3 unknowns)

    The variance partition determines τ_ξ² = ρ × κ₂*.

    Uses scipy.optimize.least_squares for numerical root finding.

    Args:
        direct_prior: Dict with 'xi', 'omega', 'alpha'
        researcher_choices: Dict with 'rho', 'alpha_omega', 'tau_alpha_sq'
        n_samples: Monte Carlo samples for cumulant estimation
        seed: Random seed for reproducibility

    Returns:
        Dict with all 6 hyperprior parameters

    Example:
        >>> direct = {'xi': 0.0, 'omega': 1.0, 'alpha': 2.0}
        >>> choices = {'rho': 0.3, 'alpha_omega': 6.0, 'tau_alpha_sq': 0.25}
        >>> hyperpriors = solve_hyperpriors(direct, choices)
    """
    # Extract inputs
    xi_star = direct_prior['xi']
    omega_star = direct_prior['omega']
    alpha_star = direct_prior['alpha']

    rho = researcher_choices['rho']
    alpha_omega = researcher_choices['alpha_omega']
    tau_alpha_sq = researcher_choices['tau_alpha_sq']

    # Compute target cumulants from direct prior
    kappa1_target, kappa2_target, kappa3_target = skew_normal_cumulants(
        xi_star, omega_star, alpha_star
    )

    print(f"Target cumulants from direct prior SN({xi_star}, {omega_star}, {alpha_star}):")
    print(f"  κ₁* = {kappa1_target:.6f}")
    print(f"  κ₂* = {kappa2_target:.6f}")
    print(f"  κ₃* = {kappa3_target:.6f}")

    # Researcher choices determine τ_ξ² via variance partition
    tau_xi_sq = rho * kappa2_target
    print(f"\nResearcher choices:")
    print(f"  ρ = {rho} → τ_ξ² = {tau_xi_sq:.6f}")
    print(f"  α_ω = {alpha_omega}")
    print(f"  τ_α² = {tau_alpha_sq}")

    # Initial guesses for (μ_ξ, β_ω, μ_α)
    x0 = np.array([
        xi_star,                           # μ_ξ ≈ ξ*
        omega_star**2 * (alpha_omega - 1), # β_ω from E[ω²] = β/(α-1) ≈ ω*²
        alpha_star                         # μ_α ≈ α*
    ])

    def objective(x: FloatArray) -> FloatArray:
        """Residual: implied cumulants - target cumulants."""
        mu_xi, beta_omega, mu_alpha = x

        if beta_omega <= 0:
            return np.array([1e10, 1e10, 1e10])

        try:
            k1, k2, k3 = compute_implied_cumulants(
                mu_xi, tau_xi_sq, alpha_omega, beta_omega,
                mu_alpha, tau_alpha_sq, n_samples=n_samples, seed=seed
            )
            return np.array([
                k1 - kappa1_target,
                k2 - kappa2_target,
                k3 - kappa3_target
            ])
        except Exception:
            return np.array([1e10, 1e10, 1e10])

    print("\nSolving for hyperprior parameters...")

    result = optimize.least_squares(
        objective, x0,
        bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
        verbose=1
    )

    mu_xi, beta_omega, mu_alpha = result.x

    # Verify solution
    k1_check, k2_check, k3_check = compute_implied_cumulants(
        mu_xi, tau_xi_sq, alpha_omega, beta_omega,
        mu_alpha, tau_alpha_sq, n_samples=n_samples, seed=seed
    )

    print(f"\nSolution found:")
    print(f"  μ_ξ = {mu_xi:.6f}")
    print(f"  τ_ξ² = {tau_xi_sq:.6f}")
    print(f"  α_ω = {alpha_omega:.6f}")
    print(f"  β_ω = {beta_omega:.6f}")
    print(f"  μ_α = {mu_alpha:.6f}")
    print(f"  τ_α² = {tau_alpha_sq:.6f}")

    print(f"\nVerification (implied vs target):")
    print(f"  κ₁: {k1_check:.6f} vs {kappa1_target:.6f} (error: {abs(k1_check-kappa1_target):.2e})")
    print(f"  κ₂: {k2_check:.6f} vs {kappa2_target:.6f} (error: {abs(k2_check-kappa2_target):.2e})")
    print(f"  κ₃: {k3_check:.6f} vs {kappa3_target:.6f} (error: {abs(k3_check-kappa3_target):.2e})")

    return {
        'mu_xi': mu_xi,
        'tau_xi_sq': tau_xi_sq,
        'alpha_omega': alpha_omega,
        'beta_omega': beta_omega,
        'mu_alpha': mu_alpha,
        'tau_alpha_sq': tau_alpha_sq,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(
    direct_prior: DirectPrior,
    hyperpriors: Hyperpriors,
    output_path: str | Path,
    n_samples: int = 100000,
    seed: int = 42
) -> None:
    """
    Generate comparison plots: direct prior vs fitted unconditional distribution.

    Creates a 2x2 figure:
        - Top-left: Direct prior density overlaid with MC histogram of unconditional
        - Top-right: ξ hyperprior density with ξ* marked
        - Bottom-left: ω² hyperprior (Inverse-Gamma) with ω*² marked
        - Bottom-right: α hyperprior density with α* marked

    Args:
        direct_prior: Dict with 'xi', 'omega', 'alpha'
        hyperpriors: Dict with all 6 hyperprior parameters
        output_path: Where to save the PDF
        n_samples: Number of MC samples for unconditional histogram
        seed: Random seed
    """
    import matplotlib.pyplot as plt

    # Extract parameters
    xi_star = direct_prior['xi']
    omega_star = direct_prior['omega']
    alpha_star = direct_prior['alpha']

    mu_xi = hyperpriors['mu_xi']
    tau_xi_sq = hyperpriors['tau_xi_sq']
    alpha_omega = hyperpriors['alpha_omega']
    beta_omega = hyperpriors['beta_omega']
    mu_alpha = hyperpriors['mu_alpha']
    tau_alpha_sq = hyperpriors['tau_alpha_sq']

    # Sample θ from the hierarchical model
    rng = np.random.default_rng(seed)
    xi_samples, omega_samples, alpha_samples = sample_hyperpriors(
        n_samples, mu_xi, tau_xi_sq, alpha_omega, beta_omega,
        mu_alpha, tau_alpha_sq, seed=seed
    )

    # Sample θ ~ SN(ξ, ω, α) using stochastic representation:
    # θ = ξ + ω(δZ + √(1-δ²)U) where Z~|N(0,1)|, U~N(0,1)
    delta = delta_fn(alpha_samples)
    Z = np.abs(rng.normal(0, 1, n_samples))
    U = rng.normal(0, 1, n_samples)
    theta_samples = xi_samples + omega_samples * (
        delta * Z + np.sqrt(1 - delta**2) * U
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Direct prior vs unconditional
    ax1 = axes[0, 0]
    x_min, x_max = np.percentile(theta_samples, [0.5, 99.5])
    x = np.linspace(x_min, x_max, 500)
    direct_pdf = skew_normal_pdf(x, xi_star, omega_star, alpha_star)
    ax1.plot(x, direct_pdf, 'b-', lw=2, label='Direct prior SN(ξ*, ω*, α*)')
    ax1.hist(theta_samples, bins=100, density=True, alpha=0.5, color='orange',
             label='Unconditional (MC)')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('Density')
    ax1.set_title('Direct Prior vs Unconditional Distribution')
    ax1.legend()

    # Plot 2: ξ hyperprior
    ax2 = axes[0, 1]
    xi_x = np.linspace(mu_xi - 4*np.sqrt(tau_xi_sq), mu_xi + 4*np.sqrt(tau_xi_sq), 200)
    ax2.plot(xi_x, stats.norm.pdf(xi_x, mu_xi, np.sqrt(tau_xi_sq)), 'g-', lw=2)
    ax2.axvline(xi_star, color='b', ls='--', label=f'ξ* = {xi_star:.2f}')
    ax2.axvline(mu_xi, color='g', ls='-', label=f'μ_ξ = {mu_xi:.2f}')
    ax2.set_xlabel('ξ')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Hyperprior: ξ ~ N({mu_xi:.2f}, {tau_xi_sq:.4f})')
    ax2.legend()

    # Plot 3: ω² hyperprior
    ax3 = axes[1, 0]
    omega_sq_mean = beta_omega / (alpha_omega - 1)
    omega_sq_x = np.linspace(0.01, omega_sq_mean * 3, 200)
    ax3.plot(omega_sq_x, stats.invgamma.pdf(omega_sq_x, alpha_omega, scale=beta_omega),
             'r-', lw=2)
    ax3.axvline(omega_star**2, color='b', ls='--', label=f'ω*² = {omega_star**2:.2f}')
    ax3.axvline(omega_sq_mean, color='r', ls='-', label=f'E[ω²] = {omega_sq_mean:.2f}')
    ax3.set_xlabel('ω²')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Hyperprior: ω² ~ IG({alpha_omega:.1f}, {beta_omega:.2f})')
    ax3.legend()

    # Plot 4: α hyperprior
    ax4 = axes[1, 1]
    alpha_x = np.linspace(mu_alpha - 4*np.sqrt(tau_alpha_sq),
                          mu_alpha + 4*np.sqrt(tau_alpha_sq), 200)
    ax4.plot(alpha_x, stats.norm.pdf(alpha_x, mu_alpha, np.sqrt(tau_alpha_sq)),
             'm-', lw=2)
    ax4.axvline(alpha_star, color='b', ls='--', label=f'α* = {alpha_star:.2f}')
    ax4.axvline(mu_alpha, color='m', ls='-', label=f'μ_α = {mu_alpha:.2f}')
    ax4.set_xlabel('α')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Hyperprior: α ~ N({mu_alpha:.2f}, {tau_alpha_sq:.4f})')
    ax4.legend()

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_skew_normal_example(
    direct_prior: DirectPrior,
    researcher_choices: ResearcherChoices,
    output_dir: str | Path,
    n_mc_samples: int = 50000,
    seed: int = 42
) -> Hyperpriors:
    """
    Run the complete Skew-Normal hyperprior selection example.

    This is the main entry point that:
    1. Computes target cumulants from the direct prior
    2. Solves for hyperprior parameters matching those cumulants
    3. Saves results to .npz file
    4. Generates comparison plots

    Args:
        direct_prior: Target distribution parameters
            - 'xi': location (float)
            - 'omega': scale (float > 0)
            - 'alpha': shape (float)
        researcher_choices: Degrees of freedom
            - 'rho': variance partition ratio (0 < float < 1)
            - 'alpha_omega': IG shape for ω² (float > 2)
            - 'tau_alpha_sq': variance of α hyperprior (float > 0)
        output_dir: Directory for output files
        n_mc_samples: Monte Carlo samples for cumulant estimation
        seed: Random seed for reproducibility

    Returns:
        Dict containing all 6 hyperprior parameters

    Output files:
        - {output_dir}/skew_normal_hyperpriors.npz
        - {output_dir}/skew_normal_comparison.pdf
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Skew-Normal Hyperprior Selection Example")
    print("=" * 60)
    print(f"\nDirect prior: SN(ξ={direct_prior['xi']}, ω={direct_prior['omega']}, "
          f"α={direct_prior['alpha']})")

    # Solve for hyperpriors
    hyperpriors = solve_hyperpriors(
        direct_prior, researcher_choices,
        n_samples=n_mc_samples, seed=seed
    )

    # Save results
    results_path = output_dir / 'skew_normal_hyperpriors.npz'
    np.savez(results_path,
             direct_prior=direct_prior,
             researcher_choices=researcher_choices,
             hyperpriors=hyperpriors)
    print(f"\nResults saved to: {results_path}")

    # Generate comparison plot
    plot_path = output_dir / 'skew_normal_comparison.pdf'
    plot_comparison(direct_prior, hyperpriors, plot_path,
                    n_samples=100000, seed=seed)

    return hyperpriors


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    # Default example configuration
    direct_prior: DirectPrior = {
        'xi': 0.0,      # location
        'omega': 1.0,   # scale
        'alpha': 2.0,   # shape (positive = right skew)
    }

    researcher_choices: ResearcherChoices = {
        'rho': 0.3,           # variance partition
        'alpha_omega': 6.0,   # scale concentration
        'tau_alpha_sq': 0.25, # shape uncertainty
    }

    output_dir = Path(__file__).parent.parent / 'plots' / 'skew_normal'

    run_skew_normal_example(direct_prior, researcher_choices, output_dir)
