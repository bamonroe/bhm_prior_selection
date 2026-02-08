"""
Skew-Normal Hyperprior Selection Example

This module demonstrates the prior selection methodology for a 3-parameter
Skew-Normal distribution, as described in Appendix B of the paper.

The Skew-Normal has density:
    f(x; ξ, ω, α) = (2/ω) φ((x-ξ)/ω) Φ(α(x-ξ)/ω)

where ξ is location, ω > 0 is scale, α is shape (skewness).

Given a direct prior SN(ξ*, ω*, α*), we find hyperprior parameters such that
the unconditional distribution matches the target cumulants.

Hyperprior structure:
    ξ ~ Normal(μ_ξ, τ_ξ²)
    ω² ~ Inverse-Gamma(α_ω, β_ω)
    α ~ Normal(μ_α, τ_α²)

Researcher-specified (3 degrees of freedom):
    - ρ: variance partition ratio
    - α_ω: scale concentration
    - τ_α²: shape uncertainty

Determined by matching (3 parameters):
    - μ_ξ: location hyperprior mean
    - β_ω: scale hyperprior parameter
    - μ_α: shape hyperprior mean
"""

import numpy as np
from scipy import optimize
from scipy import stats
from pathlib import Path


# =============================================================================
# Skew-Normal Distribution Functions
# =============================================================================

def skew_normal_pdf(x, xi, omega, alpha):
    """Skew-Normal probability density function."""
    z = (x - xi) / omega
    return (2 / omega) * stats.norm.pdf(z) * stats.norm.cdf(alpha * z)


def skew_normal_cumulants(xi, omega, alpha):
    """
    Compute the first 3 cumulants of a Skew-Normal distribution.

    Returns:
        (κ₁, κ₂, κ₃): mean, variance, third central moment
    """
    delta = alpha / np.sqrt(1 + alpha**2)

    # Mean (first cumulant)
    kappa1 = xi + omega * delta * np.sqrt(2 / np.pi)

    # Variance (second cumulant)
    kappa2 = omega**2 * (1 - 2 * delta**2 / np.pi)

    # Third central moment (third cumulant)
    kappa3 = ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi))**3 * omega**3

    return kappa1, kappa2, kappa3


def delta_fn(alpha):
    """The δ(α) function: α / √(1 + α²)."""
    return alpha / np.sqrt(1 + alpha**2)


# =============================================================================
# Monte Carlo Integration via Simple MH Sampler
# =============================================================================

def sample_hyperpriors(n_samples, mu_xi, tau_xi_sq, alpha_omega, beta_omega,
                       mu_alpha, tau_alpha_sq, seed=None):
    """
    Sample from the hyperprior distributions using simple MH.

    Returns:
        xi_samples, omega_samples, alpha_samples: arrays of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # ξ ~ Normal(μ_ξ, τ_ξ²) - direct sampling
    xi_samples = rng.normal(mu_xi, np.sqrt(tau_xi_sq), n_samples)

    # ω² ~ Inverse-Gamma(α_ω, β_ω) - direct sampling
    # If X ~ Gamma(α, 1/β), then 1/X ~ Inverse-Gamma(α, β)
    omega_sq_samples = 1.0 / rng.gamma(alpha_omega, 1.0 / beta_omega, n_samples)
    omega_samples = np.sqrt(omega_sq_samples)

    # α ~ Normal(μ_α, τ_α²) - direct sampling
    alpha_samples = rng.normal(mu_alpha, np.sqrt(tau_alpha_sq), n_samples)

    return xi_samples, omega_samples, alpha_samples


def compute_implied_cumulants(mu_xi, tau_xi_sq, alpha_omega, beta_omega,
                               mu_alpha, tau_alpha_sq, n_samples=50000, seed=42):
    """
    Compute implied cumulants of the unconditional distribution via Monte Carlo.

    Uses the law of total cumulance:
        κ₁ = E[κ₁(θ|ξ,ω,α)]
        κ₂ = E[κ₂(θ|ξ,ω,α)] + Var[κ₁(θ|ξ,ω,α)]
        κ₃ = E[κ₃(θ|ξ,ω,α)] + 3·Cov[κ₁,κ₂] + κ₃(E[θ|ξ,ω,α])

    Returns:
        (κ₁_implied, κ₂_implied, κ₃_implied)
    """
    # Sample from hyperpriors
    xi, omega, alpha = sample_hyperpriors(
        n_samples, mu_xi, tau_xi_sq, alpha_omega, beta_omega,
        mu_alpha, tau_alpha_sq, seed=seed
    )

    # Compute conditional cumulants for each sample
    delta = delta_fn(alpha)

    # Conditional mean: E[θ|ξ,ω,α] = ξ + ω·δ·√(2/π)
    cond_mean = xi + omega * delta * np.sqrt(2 / np.pi)

    # Conditional variance: Var[θ|ξ,ω,α] = ω²(1 - 2δ²/π)
    cond_var = omega**2 * (1 - 2 * delta**2 / np.pi)

    # Conditional third cumulant
    cond_kappa3 = ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi))**3 * omega**3

    # Apply law of total cumulance
    # κ₁ = E[conditional mean]
    kappa1_implied = np.mean(cond_mean)

    # κ₂ = E[conditional variance] + Var[conditional mean]
    kappa2_implied = np.mean(cond_var) + np.var(cond_mean)

    # κ₃ = E[conditional κ₃] + 3·Cov[cond_mean, cond_var] + κ₃(cond_mean)
    cov_mean_var = np.cov(cond_mean, cond_var)[0, 1]
    kappa3_cond_mean = np.mean((cond_mean - np.mean(cond_mean))**3)
    kappa3_implied = np.mean(cond_kappa3) + 3 * cov_mean_var + kappa3_cond_mean

    return kappa1_implied, kappa2_implied, kappa3_implied


# =============================================================================
# Root Finding for Hyperprior Parameters
# =============================================================================

def solve_hyperpriors(direct_prior, researcher_choices, n_samples=50000, seed=42):
    """
    Solve for the determined hyperprior parameters given researcher choices.

    Args:
        direct_prior: dict with keys 'xi', 'omega', 'alpha'
        researcher_choices: dict with keys 'rho', 'alpha_omega', 'tau_alpha_sq'
        n_samples: Monte Carlo samples for cumulant estimation
        seed: random seed

    Returns:
        dict with all 6 hyperprior parameters
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

    # Researcher choices determine τ_ξ²
    tau_xi_sq = rho * kappa2_target
    print(f"\nResearcher choices:")
    print(f"  ρ = {rho} → τ_ξ² = {tau_xi_sq:.6f}")
    print(f"  α_ω = {alpha_omega}")
    print(f"  τ_α² = {tau_alpha_sq}")

    # Initial guesses for (μ_ξ, β_ω, μ_α)
    # μ_ξ ≈ ξ* (location)
    # β_ω: E[ω²] = β_ω/(α_ω-1), so β_ω ≈ ω*² × (α_ω-1)
    # μ_α ≈ α* (shape)
    x0 = np.array([
        xi_star,
        omega_star**2 * (alpha_omega - 1),
        alpha_star
    ])

    def objective(x):
        """Residual function for root finding."""
        mu_xi, beta_omega, mu_alpha = x

        # Ensure valid parameters
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

    # Use least squares optimization
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
# Plotting
# =============================================================================

def plot_comparison(direct_prior, hyperpriors, output_path, n_samples=100000, seed=42):
    """
    Generate plots comparing direct prior to fitted unconditional distribution.

    Creates two subplots:
    1. Direct prior density vs unconditional prior density
    2. Hyperprior distributions for ξ, ω², α
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

    # Sample from hyperpriors and compute θ samples
    rng = np.random.default_rng(seed)
    xi_samples, omega_samples, alpha_samples = sample_hyperpriors(
        n_samples, mu_xi, tau_xi_sq, alpha_omega, beta_omega,
        mu_alpha, tau_alpha_sq, seed=seed
    )

    # For each (ξ, ω, α), sample a θ from the conditional Skew-Normal
    # Use rejection sampling or approximate method
    # For simplicity, use the stochastic representation:
    # If Z ~ N(0,1) truncated to Z > 0, and U ~ N(0,1) independent,
    # then θ = ξ + ω(δZ + √(1-δ²)U) ~ SN(ξ, ω, α)
    delta = delta_fn(alpha_samples)

    # Sample from truncated normal (Z > 0)
    Z = np.abs(rng.normal(0, 1, n_samples))  # folded normal ≈ truncated normal
    U = rng.normal(0, 1, n_samples)

    theta_samples = xi_samples + omega_samples * (
        delta * Z + np.sqrt(1 - delta**2) * U
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Direct prior vs unconditional density
    ax1 = axes[0, 0]

    # Determine x range
    x_min = np.percentile(theta_samples, 0.5)
    x_max = np.percentile(theta_samples, 99.5)
    x = np.linspace(x_min, x_max, 500)

    # Direct prior density
    direct_pdf = skew_normal_pdf(x, xi_star, omega_star, alpha_star)
    ax1.plot(x, direct_pdf, 'b-', linewidth=2, label='Direct prior SN(ξ*, ω*, α*)')

    # Unconditional density (histogram)
    ax1.hist(theta_samples, bins=100, density=True, alpha=0.5, color='orange',
             label='Unconditional (MC samples)')

    ax1.set_xlabel('θ')
    ax1.set_ylabel('Density')
    ax1.set_title('Direct Prior vs Unconditional Distribution')
    ax1.legend()

    # Plot 2: ξ hyperprior
    ax2 = axes[0, 1]
    xi_x = np.linspace(mu_xi - 4*np.sqrt(tau_xi_sq), mu_xi + 4*np.sqrt(tau_xi_sq), 200)
    xi_pdf = stats.norm.pdf(xi_x, mu_xi, np.sqrt(tau_xi_sq))
    ax2.plot(xi_x, xi_pdf, 'g-', linewidth=2)
    ax2.axvline(xi_star, color='b', linestyle='--', label=f'ξ* = {xi_star:.2f}')
    ax2.axvline(mu_xi, color='g', linestyle='-', label=f'μ_ξ = {mu_xi:.2f}')
    ax2.set_xlabel('ξ')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Hyperprior: ξ ~ N({mu_xi:.2f}, {tau_xi_sq:.4f})')
    ax2.legend()

    # Plot 3: ω² hyperprior (Inverse-Gamma)
    ax3 = axes[1, 0]
    omega_sq_mean = beta_omega / (alpha_omega - 1)
    omega_sq_x = np.linspace(0.01, omega_sq_mean * 3, 200)
    # Inverse-Gamma PDF: (β^α / Γ(α)) x^(-α-1) exp(-β/x)
    omega_sq_pdf = stats.invgamma.pdf(omega_sq_x, alpha_omega, scale=beta_omega)
    ax3.plot(omega_sq_x, omega_sq_pdf, 'r-', linewidth=2)
    ax3.axvline(omega_star**2, color='b', linestyle='--', label=f'ω*² = {omega_star**2:.2f}')
    ax3.axvline(omega_sq_mean, color='r', linestyle='-', label=f'E[ω²] = {omega_sq_mean:.2f}')
    ax3.set_xlabel('ω²')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Hyperprior: ω² ~ IG({alpha_omega:.1f}, {beta_omega:.2f})')
    ax3.legend()

    # Plot 4: α hyperprior
    ax4 = axes[1, 1]
    alpha_x = np.linspace(mu_alpha - 4*np.sqrt(tau_alpha_sq),
                          mu_alpha + 4*np.sqrt(tau_alpha_sq), 200)
    alpha_pdf = stats.norm.pdf(alpha_x, mu_alpha, np.sqrt(tau_alpha_sq))
    ax4.plot(alpha_x, alpha_pdf, 'm-', linewidth=2)
    ax4.axvline(alpha_star, color='b', linestyle='--', label=f'α* = {alpha_star:.2f}')
    ax4.axvline(mu_alpha, color='m', linestyle='-', label=f'μ_α = {mu_alpha:.2f}')
    ax4.set_xlabel('α')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Hyperprior: α ~ N({mu_alpha:.2f}, {tau_alpha_sq:.4f})')
    ax4.legend()

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_skew_normal_example(direct_prior, researcher_choices, output_dir,
                             n_mc_samples=50000, seed=42):
    """
    Run the full Skew-Normal hyperprior selection example.

    Args:
        direct_prior: dict with 'xi', 'omega', 'alpha'
        researcher_choices: dict with 'rho', 'alpha_omega', 'tau_alpha_sq'
        output_dir: directory for output files
        n_mc_samples: number of Monte Carlo samples
        seed: random seed

    Returns:
        dict with solved hyperprior parameters
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


if __name__ == '__main__':
    # Example usage
    direct_prior = {
        'xi': 0.0,      # location
        'omega': 1.0,   # scale
        'alpha': 2.0,   # shape (positive = right skew)
    }

    researcher_choices = {
        'rho': 0.3,           # variance partition
        'alpha_omega': 6.0,   # scale concentration
        'tau_alpha_sq': 0.25, # shape uncertainty
    }

    output_dir = Path(__file__).parent.parent / 'plots' / 'skew_normal'

    run_skew_normal_example(direct_prior, researcher_choices, output_dir)
