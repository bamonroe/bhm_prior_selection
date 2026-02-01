"""
Prior Selection and Hyperprior Derivation for RDU BHM.

This module implements the methodology from technical_notes/bhm_hyperprior_derivation:

1. Prior Selection: Define what reasonable preferences look like for a single individual
2. Hyperprior Derivation: Derive BHM hyperpriors such that the unconditional prior
   matches the individual-level prior selection

The key insight is that specifying priors for a single random individual is cognitively
tractable - we can visualize utility functions and check if implied preferences are
reasonable. The BHM then decomposes this into population-level structure.

For RDU with power probability weighting, we have 3 parameters:
- r: CRRA risk aversion coefficient
- gamma_raw: log of probability weighting parameter (gamma = exp(gamma_raw))
- mu_raw: log of Fechner noise parameter (mu = exp(mu_raw))
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class PriorSelectionResult:
    """Results from prior selection exercise for a single individual."""
    # Parameter names
    param_names: Tuple[str, ...]

    # Prior selection mean vector (mu_ps)
    mean: np.ndarray

    # Prior selection variance vector (diagonal of Sigma_ps)
    # Using independent structure, so only diagonal elements
    variance: np.ndarray

    # Description of the prior selection reasoning
    description: str


@dataclass
class DerivedHyperpriors:
    """Hyperprior parameters derived from prior selection."""
    # Parameter names
    param_names: Tuple[str, ...]

    # Variance partition ratio (rho)
    rho: float

    # Concentration parameter (alpha_0 for Inverse-Gamma)
    alpha_0: float

    # Derived hyperparameters
    mu_0: np.ndarray      # Prior mean of population mean
    tau_0_sq: np.ndarray  # Prior variance of population mean
    beta_0: np.ndarray    # Scale parameter for Inverse-Gamma on sigma^2

    # Original prior selection for reference
    ps_mean: np.ndarray
    ps_variance: np.ndarray


def rdu_power_prior_selection() -> PriorSelectionResult:
    """
    Prior selection for RDU model with power probability weighting.

    This answers: "What do reasonable risk preferences look like for a
    single random individual drawn from some population?"

    Parameters:
    -----------
    r : CRRA coefficient (untransformed)
        - r < 0: risk loving
        - r = 0: risk neutral
        - r = 1: log utility
        - r > 1: very risk averse
        Empirical literature suggests most people are mildly risk averse.
        Prior: N(0.5, 0.5^2) => 95% CI roughly (-0.5, 1.5)

    gamma_raw : log(gamma) where gamma is power probability weighting
        - gamma = 1 (gamma_raw = 0): no distortion (EUT)
        - gamma < 1 (gamma_raw < 0): overweighting small probs (typical finding)
        - gamma > 1 (gamma_raw > 0): underweighting small probs
        Empirical: gamma typically 0.6-1.0
        Prior on log scale: N(0, 0.3^2) => gamma 95% CI roughly (0.55, 1.8)

    mu_raw : log(mu) where mu is Fechner noise (inverse temperature)
        - Higher mu = more deterministic choices
        - mu ~ 0.1-1.0 is typical
        Prior on log scale: N(-2, 1^2) => mu 95% CI roughly (0.02, 1.0)

    Returns:
    --------
    PriorSelectionResult with mean and variance for each parameter
    """
    param_names = ('r', 'gamma_raw', 'mu_raw')

    # Prior selection means
    mean = np.array([
        0.5,   # r: mild risk aversion
        0.0,   # gamma_raw: no distortion on average (gamma=1)
        -2.0,  # mu_raw: moderate noise (mu ~ 0.14)
    ])

    # Prior selection variances (independent structure)
    variance = np.array([
        0.25,  # r: sd=0.5, allows range from risk loving to very risk averse
        0.09,  # gamma_raw: sd=0.3, gamma roughly 0.55 to 1.8
        1.0,   # mu_raw: sd=1.0, wide range of noise levels
    ])

    description = """
    Prior selection for RDU with power probability weighting (3 parameters):

    1. r (CRRA coefficient):
       - Prior: N(0.5, 0.25)
       - Rationale: Mild risk aversion is modal in the literature
       - 95% CI: (-0.5, 1.5) covers risk loving to very risk averse

    2. gamma_raw = log(gamma) (probability weighting):
       - Prior: N(0, 0.09)
       - Rationale: gamma=1 (no distortion) as center, allows both under/overweighting
       - 95% CI for gamma: (0.55, 1.8)

    3. mu_raw = log(mu) (Fechner noise):
       - Prior: N(-2, 1)
       - Rationale: Moderate noise (mu~0.14) with wide uncertainty
       - 95% CI for mu: (0.02, 1.0)
    """

    return PriorSelectionResult(
        param_names=param_names,
        mean=mean,
        variance=variance,
        description=description
    )


def derive_hyperpriors(
    prior_selection: PriorSelectionResult,
    rho: float = 0.3,
    alpha_0: float = 6.0
) -> DerivedHyperpriors:
    """
    Derive BHM hyperpriors from prior selection using variance decomposition.

    The key constraint is that the unconditional BHM prior must equal the
    prior selection distribution for a single random individual.

    Variance decomposition:
        Var[theta] = E[sigma^2] + tau_0^2
                   = (1 - rho) * V + rho * V
                   = V  (matches prior selection)

    Parameters:
    -----------
    prior_selection : PriorSelectionResult
        Results from the prior selection exercise

    rho : float in (0, 1)
        Variance partition ratio. Fraction of total variance attributed to
        uncertainty about the population mean.
        - rho -> 0: Most variance from population heterogeneity
        - rho -> 1: Most variance from uncertainty about population mean
        Default: 0.3 (30% mean uncertainty, 70% heterogeneity)

    alpha_0 : float > 2
        Concentration parameter for Inverse-Gamma prior on sigma^2.
        Controls how confident we are about the prior selection variance.
        - alpha_0 = 6: CV[sigma^2] = 0.5 (moderate uncertainty)
        - alpha_0 = 11: CV[sigma^2] = 0.33 (more confident)
        Default: 6.0

    Returns:
    --------
    DerivedHyperpriors with all hyperprior parameters
    """
    if not 0 < rho < 1:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    if alpha_0 <= 2:
        raise ValueError(f"alpha_0 must be > 2, got {alpha_0}")

    ps_mean = prior_selection.mean
    ps_var = prior_selection.variance
    n_params = len(ps_mean)

    # Step 1: Mean hyperprior location (directly from prior selection)
    mu_0 = ps_mean.copy()

    # Step 2: Mean hyperprior variance (from rho)
    # tau_0^2 = rho * V
    tau_0_sq = rho * ps_var

    # Step 3: Expected population variance (from rho)
    # E[sigma^2] = (1 - rho) * V
    expected_sigma_sq = (1 - rho) * ps_var

    # Step 4: Inverse-Gamma scale parameter (from alpha_0 and E[sigma^2])
    # E[sigma^2] = beta_0 / (alpha_0 - 1)
    # => beta_0 = E[sigma^2] * (alpha_0 - 1)
    beta_0 = expected_sigma_sq * (alpha_0 - 1)

    return DerivedHyperpriors(
        param_names=prior_selection.param_names,
        rho=rho,
        alpha_0=alpha_0,
        mu_0=mu_0,
        tau_0_sq=tau_0_sq,
        beta_0=beta_0,
        ps_mean=ps_mean,
        ps_variance=ps_var
    )


def verify_hyperpriors(hyperpriors: DerivedHyperpriors) -> bool:
    """
    Verify that derived hyperpriors satisfy the unconditional variance constraint.

    The unconditional variance should equal the prior selection variance:
        E[sigma^2] + tau_0^2 = beta_0/(alpha_0-1) + tau_0^2 = V
    """
    expected_sigma_sq = hyperpriors.beta_0 / (hyperpriors.alpha_0 - 1)
    unconditional_var = expected_sigma_sq + hyperpriors.tau_0_sq

    print("Hyperprior Verification:")
    print("=" * 60)

    for i, name in enumerate(hyperpriors.param_names):
        target = hyperpriors.ps_variance[i]
        actual = unconditional_var[i]
        match = np.isclose(target, actual)
        status = "OK" if match else "FAIL"

        print(f"\n{name}:")
        print(f"  Target variance (prior selection): {target:.4f}")
        print(f"  E[sigma^2] = beta_0/(alpha_0-1):   {expected_sigma_sq[i]:.4f}")
        print(f"  tau_0^2:                           {hyperpriors.tau_0_sq[i]:.4f}")
        print(f"  Sum (unconditional variance):      {actual:.4f}")
        print(f"  Status: [{status}]")

    all_match = np.allclose(unconditional_var, hyperpriors.ps_variance)
    print(f"\n{'=' * 60}")
    print(f"Overall: {'All constraints satisfied' if all_match else 'CONSTRAINT VIOLATION'}")

    return all_match


def print_hyperprior_summary(hyperpriors: DerivedHyperpriors):
    """Print a summary of the derived hyperpriors."""
    print("\nDerived BHM Hyperpriors")
    print("=" * 60)
    print(f"Variance partition ratio (rho): {hyperpriors.rho}")
    print(f"Concentration parameter (alpha_0): {hyperpriors.alpha_0}")
    print(f"CV[sigma^2] = 1/sqrt(alpha_0-2) = {1/np.sqrt(hyperpriors.alpha_0-2):.3f}")

    print(f"\n{'Parameter':<12} {'mu_0':>8} {'tau_0':>8} {'alpha_0':>8} {'beta_0':>8}")
    print("-" * 60)

    for i, name in enumerate(hyperpriors.param_names):
        print(f"{name:<12} {hyperpriors.mu_0[i]:>8.3f} "
              f"{np.sqrt(hyperpriors.tau_0_sq[i]):>8.3f} "
              f"{hyperpriors.alpha_0:>8.1f} "
              f"{hyperpriors.beta_0[i]:>8.3f}")

    print("\nInterpretation:")
    print(f"  - Population mean prior: mu_j ~ N(mu_0[j], tau_0[j]^2)")
    print(f"  - Population variance prior: sigma_j^2 ~ InvGamma(alpha_0, beta_0[j])")
    print(f"  - Individual prior: theta_ij ~ N(mu_j, sigma_j^2)")


def get_hyperpriors_for_rdu_bhm(rho: float = 0.3, alpha_0: float = 6.0) -> Dict:
    """
    Get hyperprior parameters formatted for rdu_bhm.py.

    Returns a dict with the hyperprior values in the format expected by
    the log_posterior function.
    """
    ps = rdu_power_prior_selection()
    hp = derive_hyperpriors(ps, rho=rho, alpha_0=alpha_0)

    return {
        'r_mean': float(hp.mu_0[0]),
        'ln_r_sdev': float(np.log(np.sqrt(hp.tau_0_sq[0]))),  # log of std dev
        'gamma_mean': float(hp.mu_0[1]),
        'ln_gamma_sdev': float(np.log(np.sqrt(hp.tau_0_sq[1]))),
        'mu_mean': float(hp.mu_0[2]),
        'ln_mu_sdev': float(np.log(np.sqrt(hp.tau_0_sq[2]))),
        # Inverse-Gamma parameters for hyperprior on population variance
        'alpha_0': float(alpha_0),
        'beta_0_r': float(hp.beta_0[0]),
        'beta_0_gamma': float(hp.beta_0[1]),
        'beta_0_mu': float(hp.beta_0[2]),
        # Original prior selection for reference
        'ps_mean': hp.ps_mean.tolist(),
        'ps_variance': hp.ps_variance.tolist(),
        'rho': rho,
    }


if __name__ == "__main__":
    # Run prior selection and derive hyperpriors
    print("RDU Power Probability Weighting - Prior Selection")
    print("=" * 60)

    ps = rdu_power_prior_selection()
    print(ps.description)

    print("\nPrior Selection Summary:")
    print(f"{'Parameter':<12} {'Mean':>8} {'Std Dev':>8} {'Variance':>10}")
    print("-" * 40)
    for i, name in enumerate(ps.param_names):
        print(f"{name:<12} {ps.mean[i]:>8.3f} {np.sqrt(ps.variance[i]):>8.3f} {ps.variance[i]:>10.4f}")

    # Derive hyperpriors with default settings
    hp = derive_hyperpriors(ps, rho=0.3, alpha_0=6.0)
    print_hyperprior_summary(hp)

    # Verify constraints
    print()
    verify_hyperpriors(hp)

    # Show formatted output for rdu_bhm
    print("\n" + "=" * 60)
    print("Hyperprior values for rdu_bhm.py:")
    print("=" * 60)
    params = get_hyperpriors_for_rdu_bhm()
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
