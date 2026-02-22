"""
Prior Selection and Hyperprior Derivation for RDU BHM.

This module implements the methodology from technical_notes/bhm_hyperprior_derivation:

1. Prior Selection: Define what reasonable preferences look like for a single individual
2. Hyperprior Derivation: Derive BHM hyperpriors such that the unconditional prior
   matches the individual-level prior selection

The key insight is that specifying priors for a single random individual is cognitively
tractable - we can visualize utility functions and check if implied preferences are
reasonable. The BHM then decomposes this into population-level structure.

For RDU with 2-parameter Prelec probability weighting, we have 4 parameters:
- r: CRRA risk aversion coefficient
- phi_raw: log of Prelec curvature parameter (phi = exp(phi_raw))
- eta_raw: log of Prelec elevation parameter (eta = exp(eta_raw))
- lambda_raw: log of Fechner noise parameter (lambda = exp(lambda_raw))
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

    # Coefficient of variation of heterogeneity (kappa)
    # kappa = SD[sigma^2] / E[sigma^2] = 1/sqrt(alpha_0 - 2)
    kappa: float

    # Derived hyperparameters
    mu_0: np.ndarray      # Prior mean of population mean
    tau_0_sq: np.ndarray  # Prior variance of population mean
    beta_0: np.ndarray    # Scale parameter for Inverse-Gamma on sigma^2

    # Original prior selection for reference
    ps_mean: np.ndarray
    ps_variance: np.ndarray


@dataclass
class DerivedHyperpriorsBeta:
    """Hyperprior parameters for Beta-Beta hierarchy on bounded CEs.

    For each CE k, the hierarchy is:
        z_i^k | mu_k, psi_k ~ Beta(mu_k(1-psi_k)/psi_k, (1-mu_k)(1-psi_k)/psi_k)
        mu_k ~ Beta(a_mu[k], b_mu[k])
        psi_k ~ Beta(a_psi[k], b_psi[k])

    where z_i^k = (CE_i^k - a_k) / (b_k - a_k) is the normalized CE.
    """
    # CE identification
    ce_names: Tuple[str, ...]

    # Lottery bounds (dollar scale)
    bounds_lo: np.ndarray  # a_k: min prize per CE
    bounds_hi: np.ndarray  # b_k: max prize per CE

    # Researcher choices
    rho: float    # variance partition ratio
    kappa: float  # CV of heterogeneity

    # Mean hyperprior: mu_k ~ Beta(a_mu, b_mu)
    a_mu: np.ndarray
    b_mu: np.ndarray
    nu_mu: np.ndarray  # effective sample size a_mu + b_mu

    # Dispersion hyperprior: psi_k ~ Beta(a_psi, b_psi)
    a_psi: np.ndarray
    b_psi: np.ndarray
    nu_psi: np.ndarray  # effective sample size a_psi + b_psi

    # Intermediate quantities (normalized)
    mu_star: np.ndarray   # normalized mean per CE
    v_star: np.ndarray    # normalized variance per CE
    psi_star: np.ndarray  # target variance fraction per CE

    # Lambda NIG hyperpriors (lambda is unbounded, stays on NIG path)
    lambda_mu_0: float
    lambda_tau_0_sq: float
    lambda_alpha_0: float
    lambda_beta_0: float

    # Original dollar-scale beliefs
    ps_mean_dollars: np.ndarray
    ps_variance_dollars: np.ndarray


def kappa_to_alpha0(kappa: float) -> float:
    """
    Convert coefficient of variation of heterogeneity (κ) to IG shape parameter (α₀).

    For σ² ~ InvGamma(α₀, β₀):
        CV[σ²] = κ = 1/√(α₀ - 2)
        ⟹ α₀ = 2 + 1/κ²

    Parameters:
        kappa: CV of σ², must be > 0. Smaller = more confident about heterogeneity.
               κ ≈ 0.24 corresponds to α₀ = 20 (tight); κ = 1.0 corresponds to α₀ = 3 (diffuse).

    Returns:
        alpha_0: IG shape parameter (always > 2)
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    return 2.0 + 1.0 / (kappa ** 2)


def alpha0_to_kappa(alpha_0: float) -> float:
    """
    Convert IG shape parameter (α₀) to coefficient of variation of heterogeneity (κ).

    For σ² ~ InvGamma(α₀, β₀):
        κ = SD[σ²] / E[σ²] = 1/√(α₀ - 2)

    Parameters:
        alpha_0: IG shape parameter, must be > 2.

    Returns:
        kappa: CV of σ²
    """
    if alpha_0 <= 2:
        raise ValueError(f"alpha_0 must be > 2, got {alpha_0}")
    return 1.0 / np.sqrt(alpha_0 - 2.0)


def ce_space_prior_selection(
    ce_direct_prior: Dict,
    lambda_direct_prior: Optional[Dict] = None,
) -> PriorSelectionResult:
    """
    Prior selection in CE space for 3 reference lotteries.

    Instead of specifying priors on (r, phi_raw, eta_raw), the researcher
    specifies priors on (CE1, CE2, CE3) — the certainty equivalents of three
    reference lotteries. The Jacobian correction transforms this to parameter
    space.

    Parameters:
    -----------
    ce_direct_prior : dict
        Dictionary with keys 'ce1', 'ce2', 'ce3', each having 'mu' and 'sigma2'.
        Example:
            {
                'ce1': {'mu': 27.0, 'sigma2': 25.0},
                'ce2': {'mu': 22.0, 'sigma2': 25.0},
                'ce3': {'mu': 25.0, 'sigma2': 25.0},
            }

    lambda_direct_prior : dict, optional
        Dict with 'mu' and 'sigma2' for log(lambda).
        If None, uses default: {'mu': -2.0, 'sigma2': 1.0}.

    Returns:
    --------
    PriorSelectionResult with 4 components: [CE1, CE2, CE3, lambda_raw]
    """
    if lambda_direct_prior is None:
        lambda_direct_prior = {'mu': -2.0, 'sigma2': 1.0}

    param_names = ('ce1', 'ce2', 'ce3', 'lambda_raw')

    mean = np.array([
        ce_direct_prior['ce1']['mu'],
        ce_direct_prior['ce2']['mu'],
        ce_direct_prior['ce3']['mu'],
        lambda_direct_prior['mu'],
    ])

    variance = np.array([
        ce_direct_prior['ce1']['sigma2'],
        ce_direct_prior['ce2']['sigma2'],
        ce_direct_prior['ce3']['sigma2'],
        lambda_direct_prior['sigma2'],
    ])

    description = f"""
    CE-space prior selection for RDU with 2-parameter Prelec PWF:

    1. CE1 (reference lottery 1):
       - Prior: N({ce_direct_prior['ce1']['mu']}, {ce_direct_prior['ce1']['sigma2']})
       - sd = {np.sqrt(ce_direct_prior['ce1']['sigma2']):.2f}

    2. CE2 (reference lottery 2):
       - Prior: N({ce_direct_prior['ce2']['mu']}, {ce_direct_prior['ce2']['sigma2']})
       - sd = {np.sqrt(ce_direct_prior['ce2']['sigma2']):.2f}

    3. CE3 (reference lottery 3):
       - Prior: N({ce_direct_prior['ce3']['mu']}, {ce_direct_prior['ce3']['sigma2']})
       - sd = {np.sqrt(ce_direct_prior['ce3']['sigma2']):.2f}

    4. lambda_raw = log(lambda) (Fechner noise):
       - Prior: N({lambda_direct_prior['mu']}, {lambda_direct_prior['sigma2']})
       - sd = {np.sqrt(lambda_direct_prior['sigma2']):.2f}
    """

    return PriorSelectionResult(
        param_names=param_names,
        mean=mean,
        variance=variance,
        description=description
    )


def rdu_prelec_prior_selection(direct_prior: Optional[Dict] = None) -> PriorSelectionResult:
    """
    Prior selection for RDU model with 2-parameter Prelec probability weighting.

    This answers: "What do reasonable risk preferences look like for a
    single random individual drawn from some population?"

    Parameters:
    -----------
    direct_prior : dict, optional
        Dictionary specifying the direct prior for each parameter.
        Expected structure:
        {
            'r': {'mu': float, 'sigma2': float},
            'phi_raw': {'mu': float, 'sigma2': float},
            'eta_raw': {'mu': float, 'sigma2': float},
            'lambda_raw': {'mu': float, 'sigma2': float},
        }
        If None, uses default values.

    Parameter interpretations:
    --------------------------
    r : CRRA coefficient (untransformed)
        - r < 0: risk loving
        - r = 0: risk neutral
        - r = 1: log utility
        - r > 1: very risk averse

    phi_raw : log(phi) where phi is Prelec curvature parameter
        - phi = 1 (phi_raw = 0): linear in log-odds
        - phi < 1: S-shaped (overweight small, underweight large probs)

    eta_raw : log(eta) where eta is Prelec elevation parameter
        - eta = 1 (eta_raw = 0): no elevation shift
        - eta < 1: pessimistic (lower weights overall)

    lambda_raw : log(lambda) where lambda is Fechner noise
        - Higher lambda = more deterministic choices

    Returns:
    --------
    PriorSelectionResult with mean and variance for each parameter
    """
    param_names = ('r', 'phi_raw', 'eta_raw', 'lambda_raw')

    # Use provided direct_prior or defaults
    if direct_prior is None:
        direct_prior = {
            'r': {'mu': 0.5, 'sigma2': 0.25},
            'phi_raw': {'mu': 0.0, 'sigma2': 0.09},
            'eta_raw': {'mu': 0.0, 'sigma2': 0.09},
            'lambda_raw': {'mu': -2.0, 'sigma2': 1.0},
        }

    # Extract means and variances
    mean = np.array([
        direct_prior['r']['mu'],
        direct_prior['phi_raw']['mu'],
        direct_prior['eta_raw']['mu'],
        direct_prior['lambda_raw']['mu'],
    ])

    variance = np.array([
        direct_prior['r']['sigma2'],
        direct_prior['phi_raw']['sigma2'],
        direct_prior['eta_raw']['sigma2'],
        direct_prior['lambda_raw']['sigma2'],
    ])

    description = f"""
    Prior selection for RDU with 2-parameter Prelec probability weighting (4 parameters):

    1. r (CRRA coefficient):
       - Prior: N({direct_prior['r']['mu']}, {direct_prior['r']['sigma2']})
       - sd = {np.sqrt(direct_prior['r']['sigma2']):.2f}

    2. phi_raw = log(phi) (Prelec curvature):
       - Prior: N({direct_prior['phi_raw']['mu']}, {direct_prior['phi_raw']['sigma2']})
       - sd = {np.sqrt(direct_prior['phi_raw']['sigma2']):.2f}

    3. eta_raw = log(eta) (Prelec elevation):
       - Prior: N({direct_prior['eta_raw']['mu']}, {direct_prior['eta_raw']['sigma2']})
       - sd = {np.sqrt(direct_prior['eta_raw']['sigma2']):.2f}

    4. lambda_raw = log(lambda) (Fechner noise):
       - Prior: N({direct_prior['lambda_raw']['mu']}, {direct_prior['lambda_raw']['sigma2']})
       - sd = {np.sqrt(direct_prior['lambda_raw']['sigma2']):.2f}
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
    alpha_0: Optional[float] = None,
    kappa: Optional[float] = None,
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

    alpha_0 : float > 2, optional
        Concentration parameter for Inverse-Gamma prior on sigma^2.
        Exactly one of alpha_0 or kappa must be provided.

    kappa : float > 0, optional
        Coefficient of variation of heterogeneity: κ = SD[σ²]/E[σ²].
        Distribution-free measure of confidence in heterogeneity specification.
        - κ ≈ 0.24: tight (α₀ = 20)
        - κ = 0.5: moderate (α₀ = 6)
        - κ = 1.0: diffuse (α₀ = 3)
        Exactly one of alpha_0 or kappa must be provided.

    Returns:
    --------
    DerivedHyperpriors with all hyperprior parameters
    """
    if not 0 < rho < 1:
        raise ValueError(f"rho must be in (0, 1), got {rho}")

    # Resolve kappa / alpha_0
    if kappa is not None and alpha_0 is not None:
        raise ValueError("Specify exactly one of alpha_0 or kappa, not both")
    if kappa is not None:
        alpha_0 = kappa_to_alpha0(kappa)
    elif alpha_0 is not None:
        if alpha_0 <= 2:
            raise ValueError(f"alpha_0 must be > 2, got {alpha_0}")
        kappa = alpha0_to_kappa(alpha_0)
    else:
        # Default: kappa = 0.5 (moderate), i.e. alpha_0 = 6
        kappa = 0.5
        alpha_0 = kappa_to_alpha0(kappa)

    ps_mean = prior_selection.mean
    ps_var = prior_selection.variance

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
        kappa=kappa,
        mu_0=mu_0,
        tau_0_sq=tau_0_sq,
        beta_0=beta_0,
        ps_mean=ps_mean,
        ps_variance=ps_var
    )


def derive_hyperpriors_beta(
    ce_prior: PriorSelectionResult,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    rho: float = 0.2,
    kappa: Optional[float] = None,
    alpha_0: Optional[float] = None,
) -> DerivedHyperpriorsBeta:
    """
    Derive Beta-Beta hyperpriors for bounded CE-space hierarchy.

    The researcher provides per-CE beliefs (mean and variance in dollars) plus
    shared (rho, kappa). This function:
    1. Normalizes to [0,1] using lottery bounds
    2. Moment-matches Beta hyperpriors on mu_k (population mean)
    3. Moment-matches Beta hyperpriors on psi_k (variance fraction)
    4. Derives NIG hyperpriors for lambda (unbounded, uses existing path)

    Parameters:
    -----------
    ce_prior : PriorSelectionResult
        From ce_space_prior_selection(). Last element is lambda_raw.
    bounds_lo : np.ndarray
        Min prize for each reference lottery (dollar scale), length n_ce.
    bounds_hi : np.ndarray
        Max prize for each reference lottery (dollar scale), length n_ce.
    rho : float in (0, 1)
        Variance partition ratio (fraction from between-population uncertainty).
    kappa : float > 0, optional
        CV of heterogeneity. Enters nu_psi formula directly for Beta-Beta.
        For lambda's NIG, converted via alpha_0 = 2 + 1/kappa^2.
        Exactly one of kappa or alpha_0 must be provided.
    alpha_0 : float > 2, optional
        IG shape parameter. Converted to kappa for Beta-Beta formulas.
        Used directly for lambda's NIG path.

    Returns:
    --------
    DerivedHyperpriorsBeta
    """
    if not 0 < rho < 1:
        raise ValueError(f"rho must be in (0, 1), got {rho}")

    # Resolve kappa / alpha_0
    if kappa is not None and alpha_0 is not None:
        raise ValueError("Specify exactly one of alpha_0 or kappa, not both")
    if kappa is not None:
        alpha_0_for_lambda = kappa_to_alpha0(kappa)
    elif alpha_0 is not None:
        if alpha_0 <= 2:
            raise ValueError(f"alpha_0 must be > 2, got {alpha_0}")
        kappa = alpha0_to_kappa(alpha_0)
        alpha_0_for_lambda = alpha_0
    else:
        kappa = 0.5
        alpha_0_for_lambda = kappa_to_alpha0(kappa)

    bounds_lo = np.asarray(bounds_lo, dtype=float)
    bounds_hi = np.asarray(bounds_hi, dtype=float)

    # Separate CE params (first n_ce) from lambda (last element)
    n_ce = len(ce_prior.param_names) - 1  # last is lambda_raw
    ce_names = ce_prior.param_names[:n_ce]
    mu_dollars = ce_prior.mean[:n_ce]
    var_dollars = ce_prior.variance[:n_ce]
    lambda_mean = ce_prior.mean[-1]
    lambda_var = ce_prior.variance[-1]

    if len(bounds_lo) != n_ce or len(bounds_hi) != n_ce:
        raise ValueError(
            f"bounds must have length {n_ce} (one per CE), "
            f"got bounds_lo={len(bounds_lo)}, bounds_hi={len(bounds_hi)}"
        )

    # --- Step 1: Normalize to [0, 1] ---
    span = bounds_hi - bounds_lo
    mu_tilde = (mu_dollars - bounds_lo) / span  # normalized mean
    v_tilde = var_dollars / span**2              # normalized variance

    # --- Step 2: Validate Bernoulli bound ---
    bernoulli_bound = mu_tilde * (1 - mu_tilde)
    for k in range(n_ce):
        if v_tilde[k] >= bernoulli_bound[k]:
            raise ValueError(
                f"CE '{ce_names[k]}': normalized variance {v_tilde[k]:.6f} >= "
                f"Bernoulli bound mu*(1-mu) = {bernoulli_bound[k]:.6f}. "
                f"Reduce sigma2 or adjust mu to be further from bounds."
            )
        if not (0 < mu_tilde[k] < 1):
            raise ValueError(
                f"CE '{ce_names[k]}': normalized mean {mu_tilde[k]:.6f} not in (0,1). "
                f"Mean {mu_dollars[k]} must be between bounds [{bounds_lo[k]}, {bounds_hi[k]}]."
            )

    # --- Step 3: Mean hyperprior mu_k ~ Beta(a_mu, b_mu) ---
    # nu_mu = mu*(1-mu) / (rho * v) - 1
    nu_mu = bernoulli_bound / (rho * v_tilde) - 1
    a_mu = mu_tilde * nu_mu
    b_mu = (1 - mu_tilde) * nu_mu

    for k in range(n_ce):
        if nu_mu[k] <= 0:
            raise ValueError(
                f"CE '{ce_names[k]}': nu_mu = {nu_mu[k]:.4f} <= 0. "
                f"rho * v_tilde = {rho * v_tilde[k]:.6f} >= Bernoulli bound "
                f"{bernoulli_bound[k]:.6f}. Reduce rho or sigma2."
            )

    # --- Step 4: Variance fraction target psi* ---
    # psi* = (1-rho) * v / (mu*(1-mu) - rho*v)
    denom = bernoulli_bound - rho * v_tilde
    psi_star = (1 - rho) * v_tilde / denom

    for k in range(n_ce):
        if not (0 < psi_star[k] < 1):
            raise ValueError(
                f"CE '{ce_names[k]}': psi_star = {psi_star[k]:.6f} not in (0,1). "
                f"This means the within-population variance target is infeasible."
            )

    # --- Step 5: Dispersion hyperprior psi_k ~ Beta(a_psi, b_psi) ---
    # nu_psi = (1 - psi*) / (psi* * kappa^2) - 1
    nu_psi = (1 - psi_star) / (psi_star * kappa**2) - 1
    a_psi = psi_star * nu_psi
    b_psi = (1 - psi_star) * nu_psi

    for k in range(n_ce):
        if nu_psi[k] <= 0:
            raise ValueError(
                f"CE '{ce_names[k]}': nu_psi = {nu_psi[k]:.4f} <= 0. "
                f"psi_star={psi_star[k]:.4f}, kappa={kappa:.4f}. "
                f"Either psi_star is too large or kappa is too large. "
                f"Try reducing kappa (more confidence in heterogeneity) or "
                f"reducing the prior variance."
            )

    # --- Step 6: Lambda NIG via existing derive_hyperpriors() ---
    lambda_ps = PriorSelectionResult(
        param_names=('lambda_raw',),
        mean=np.array([lambda_mean]),
        variance=np.array([lambda_var]),
        description="Lambda prior for Beta-Beta hierarchy"
    )
    lambda_hp = derive_hyperpriors(lambda_ps, rho=rho, alpha_0=alpha_0_for_lambda)

    return DerivedHyperpriorsBeta(
        ce_names=ce_names,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        rho=rho,
        kappa=kappa,
        a_mu=a_mu,
        b_mu=b_mu,
        nu_mu=nu_mu,
        a_psi=a_psi,
        b_psi=b_psi,
        nu_psi=nu_psi,
        mu_star=mu_tilde,
        v_star=v_tilde,
        psi_star=psi_star,
        lambda_mu_0=float(lambda_hp.mu_0[0]),
        lambda_tau_0_sq=float(lambda_hp.tau_0_sq[0]),
        lambda_alpha_0=float(lambda_hp.alpha_0),
        lambda_beta_0=float(lambda_hp.beta_0[0]),
        ps_mean_dollars=mu_dollars,
        ps_variance_dollars=var_dollars,
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
    print(f"Heterogeneity CV (kappa): {hyperpriors.kappa:.4f}")
    print(f"  => IG shape (alpha_0): {hyperpriors.alpha_0:.2f}")

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


def verify_hyperpriors_beta(hp: DerivedHyperpriorsBeta, n_mc: int = 100_000,
                            seed: int = 42) -> bool:
    """
    Monte Carlo verification that Beta-Beta hyperpriors reproduce targets.

    Draws from the full hierarchy and checks:
    1. E[z_k] ≈ mu_star[k]  (unconditional mean)
    2. Var[z_k] ≈ v_star[k]  (unconditional variance)
    3. Var[mu_k] ≈ rho * v_star[k]  (variance decomposition)
    """
    rng = np.random.default_rng(seed)
    n_ce = len(hp.ce_names)
    all_ok = True

    print("Beta-Beta Hyperprior Verification (Monte Carlo)")
    print("=" * 70)
    print(f"  MC draws: {n_mc:,}")
    print(f"  rho = {hp.rho}, kappa = {hp.kappa:.4f}")

    for k in range(n_ce):
        # Draw mu_k ~ Beta(a_mu, b_mu)
        mu_draws = rng.beta(hp.a_mu[k], hp.b_mu[k], size=n_mc)

        # Draw psi_k ~ Beta(a_psi, b_psi)
        psi_draws = rng.beta(hp.a_psi[k], hp.b_psi[k], size=n_mc)

        # Draw z_k ~ Beta(mu*(1-psi)/psi, (1-mu)*(1-psi)/psi)
        alpha_pop = mu_draws * (1 - psi_draws) / psi_draws
        beta_pop = (1 - mu_draws) * (1 - psi_draws) / psi_draws
        z_draws = rng.beta(alpha_pop, beta_pop)

        # Check targets
        mc_mean = np.mean(z_draws)
        mc_var = np.var(z_draws)
        mc_var_mu = np.var(mu_draws)

        target_mean = hp.mu_star[k]
        target_var = hp.v_star[k]
        target_var_mu = hp.rho * hp.v_star[k]

        # Tolerances: relative for mean, absolute for variance
        mean_ok = np.abs(mc_mean - target_mean) < 0.01
        var_ok = np.abs(mc_var - target_var) / target_var < 0.05
        var_mu_ok = np.abs(mc_var_mu - target_var_mu) / target_var_mu < 0.1

        ok = mean_ok and var_ok and var_mu_ok
        all_ok = all_ok and ok

        print(f"\n  {hp.ce_names[k]}:")
        print(f"    E[z]:      MC={mc_mean:.5f}  target={target_mean:.5f}  "
              f"{'OK' if mean_ok else 'FAIL'}")
        print(f"    Var[z]:    MC={mc_var:.6f}  target={target_var:.6f}  "
              f"{'OK' if var_ok else 'FAIL'}")
        print(f"    Var[mu]:   MC={mc_var_mu:.6f}  target={target_var_mu:.6f}  "
              f"{'OK' if var_mu_ok else 'FAIL'}")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'All checks passed' if all_ok else 'SOME CHECKS FAILED'}")

    return all_ok


def print_hyperprior_summary_beta(hp: DerivedHyperpriorsBeta):
    """Print a summary table of the Beta-Beta hyperprior derivation."""
    n_ce = len(hp.ce_names)

    print("\nBeta-Beta Hyperprior Summary")
    print("=" * 78)
    print(f"Researcher choices: rho = {hp.rho}, kappa = {hp.kappa:.4f}")

    # Dollar-scale beliefs
    print(f"\n{'CE':<6} {'mu($)':>8} {'sd($)':>8} {'bound_lo':>9} {'bound_hi':>9} "
          f"{'mu_tilde':>9} {'v_tilde':>10}")
    print("-" * 78)
    for k in range(n_ce):
        print(f"{hp.ce_names[k]:<6} {hp.ps_mean_dollars[k]:>8.2f} "
              f"{np.sqrt(hp.ps_variance_dollars[k]):>8.2f} "
              f"{hp.bounds_lo[k]:>9.2f} {hp.bounds_hi[k]:>9.2f} "
              f"{hp.mu_star[k]:>9.4f} {hp.v_star[k]:>10.6f}")

    # Mean hyperprior
    print(f"\nMean hyperprior: mu_k ~ Beta(a_mu, b_mu)")
    print(f"{'CE':<6} {'nu_mu':>8} {'a_mu':>8} {'b_mu':>8}")
    print("-" * 40)
    for k in range(n_ce):
        print(f"{hp.ce_names[k]:<6} {hp.nu_mu[k]:>8.2f} "
              f"{hp.a_mu[k]:>8.3f} {hp.b_mu[k]:>8.3f}")

    # Dispersion hyperprior
    print(f"\nDispersion hyperprior: psi_k ~ Beta(a_psi, b_psi)")
    print(f"{'CE':<6} {'psi*':>8} {'nu_psi':>8} {'a_psi':>8} {'b_psi':>8}")
    print("-" * 50)
    for k in range(n_ce):
        print(f"{hp.ce_names[k]:<6} {hp.psi_star[k]:>8.4f} "
              f"{hp.nu_psi[k]:>8.2f} {hp.a_psi[k]:>8.3f} {hp.b_psi[k]:>8.3f}")

    # Lambda NIG
    print(f"\nLambda (unbounded, NIG path):")
    print(f"  mu_0 = {hp.lambda_mu_0:.4f}")
    print(f"  tau_0 = {np.sqrt(hp.lambda_tau_0_sq):.4f}")
    print(f"  alpha_0 = {hp.lambda_alpha_0:.2f}")
    print(f"  beta_0 = {hp.lambda_beta_0:.4f}")

    print(f"\nPopulation: z_i^k | mu_k, psi_k ~ "
          f"Beta(mu*(1-psi)/psi, (1-mu)*(1-psi)/psi)")


def get_hyperpriors_for_rdu_bhm(
    rho: float = 0.3,
    alpha_0: Optional[float] = None,
    kappa: Optional[float] = None,
    direct_prior: Optional[Dict] = None
) -> Dict:
    """
    Get hyperprior parameters formatted for rdu_bhm.py.

    Parameters:
    -----------
    rho : float
        Variance partition ratio
    alpha_0 : float, optional
        IG concentration parameter (exactly one of alpha_0 or kappa required)
    kappa : float, optional
        CV of heterogeneity (exactly one of alpha_0 or kappa required)
    direct_prior : dict, optional
        Dictionary specifying the direct prior for each parameter.
        If None, uses default values from rdu_prelec_prior_selection().

    Returns a dict with the hyperprior values in the format expected by
    the log_posterior function.
    """
    ps = rdu_prelec_prior_selection(direct_prior)
    hp = derive_hyperpriors(ps, rho=rho, alpha_0=alpha_0, kappa=kappa)

    return {
        # Population mean hyperpriors (mu_0, tau_0)
        'r_mean': float(hp.mu_0[0]),
        'ln_r_sdev': float(np.log(np.sqrt(hp.tau_0_sq[0]))),  # log of std dev
        'phi_mean': float(hp.mu_0[1]),
        'ln_phi_sdev': float(np.log(np.sqrt(hp.tau_0_sq[1]))),
        'eta_mean': float(hp.mu_0[2]),
        'ln_eta_sdev': float(np.log(np.sqrt(hp.tau_0_sq[2]))),
        'lambda_mean': float(hp.mu_0[3]),
        'ln_lambda_sdev': float(np.log(np.sqrt(hp.tau_0_sq[3]))),
        # Inverse-Gamma parameters for hyperprior on population variance
        'alpha_0': float(hp.alpha_0),
        'beta_0_r': float(hp.beta_0[0]),
        'beta_0_phi': float(hp.beta_0[1]),
        'beta_0_eta': float(hp.beta_0[2]),
        'beta_0_lambda': float(hp.beta_0[3]),
        # Original prior selection for reference
        'ps_mean': hp.ps_mean.tolist(),
        'ps_variance': hp.ps_variance.tolist(),
        'rho': rho,
    }


def get_hyperprior_specs_for_plotting(
    direct_prior: Dict,
    rho: float = 0.3,
    alpha_0: Optional[float] = None,
    kappa: Optional[float] = None,
) -> Dict:
    """
    Get hyperprior specs in the format expected by plotting functions.

    Parameters:
    -----------
    direct_prior : dict
        Dictionary specifying the direct prior for each parameter.
    rho : float
        Variance partition ratio
    alpha_0 : float, optional
        IG concentration parameter (exactly one of alpha_0 or kappa required)
    kappa : float, optional
        CV of heterogeneity (exactly one of alpha_0 or kappa required)

    Returns:
    --------
    Dict with format:
        {
            'r': {'mu_0': ..., 'tau_0_sq': ..., 'alpha_0': ..., 'beta_0': ...},
            ...
        }
    """
    ps = rdu_prelec_prior_selection(direct_prior)
    hp = derive_hyperpriors(ps, rho=rho, alpha_0=alpha_0, kappa=kappa)

    param_names = ['r', 'phi_raw', 'eta_raw', 'lambda_raw']

    return {
        param_names[i]: {
            'mu_0': float(hp.mu_0[i]),
            'tau_0_sq': float(hp.tau_0_sq[i]),
            'alpha_0': float(hp.alpha_0),
            'beta_0': float(hp.beta_0[i]),
        }
        for i in range(len(param_names))
    }


def run_prior_selection(rho=0.3, alpha_0=None, kappa=None, output_path=None, direct_prior=None):
    """
    Run prior selection, derive hyperpriors, and save to file.

    Args:
        rho: Variance partition ratio (default 0.3)
        alpha_0: IG concentration parameter (exactly one of alpha_0 or kappa required)
        kappa: CV of heterogeneity (exactly one of alpha_0 or kappa required)
        output_path: Path to save NPZ file (default: data/output/prior_params.npz)
        direct_prior: Dict specifying direct prior for each parameter (optional)
    """
    from pathlib import Path

    print("Running prior selection for RDU model with Prelec PWF...")
    ps = rdu_prelec_prior_selection(direct_prior)
    print(ps.description)

    # Derive hyperpriors
    hp = derive_hyperpriors(ps, rho=rho, alpha_0=alpha_0, kappa=kappa)
    print_hyperprior_summary(hp)

    # Verify constraints
    print()
    verify_hyperpriors(hp)

    # Get formatted hyperpriors
    params = get_hyperpriors_for_rdu_bhm(rho=rho, alpha_0=hp.alpha_0, direct_prior=direct_prior)

    # Save to file
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'prior_params.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **params)
    print(f"\nSaved hyperpriors to: {output_path}")

    return params


def save_direct_hyperprior(hyperprior_specs: Dict, output_path) -> Dict:
    """
    Save directly-specified hyperprior parameters to file.

    This bypasses the variance decomposition and saves hyperpriors as-is.
    Use this for uninformative/diffuse priors where you want to specify
    the hyperprior parameters directly.

    Args:
        hyperprior_specs: Dict with format:
            {
                'r': {'mu_0': ..., 'tau_0_sq': ..., 'alpha_0': ..., 'beta_0': ...},
                'phi_raw': {...},
                'eta_raw': {...},
                'lambda_raw': {...},
            }
        output_path: Path to save NPZ file

    Returns:
        Dict with the hyperprior values in format expected by rdu_bhm.py
    """
    from pathlib import Path

    print("Saving direct hyperprior specification...")

    param_names = ['r', 'phi_raw', 'eta_raw', 'lambda_raw']

    # Convert to format expected by rdu_bhm.py
    params = {}
    for name in param_names:
        spec = hyperprior_specs[name]
        base = name.replace('_raw', '')

        # Population mean hyperpriors
        params[f'{base}_mean'] = float(spec['mu_0'])
        params[f'ln_{base}_sdev'] = float(np.log(np.sqrt(spec['tau_0_sq'])))

        # Inverse-Gamma parameters for population variance
        params[f'beta_0_{base}'] = float(spec['beta_0'])

    # Use alpha_0 from first parameter (should be same for all in this case)
    params['alpha_0'] = float(hyperprior_specs['r']['alpha_0'])

    # Add metadata
    params['rho'] = 0.0  # Not applicable for direct specification

    # Print summary
    print("\nDirect Hyperprior Specification:")
    print("=" * 60)
    print(f"{'Parameter':<12} {'mu_0':>8} {'tau_0':>8} {'alpha_0':>8} {'beta_0':>8}")
    print("-" * 60)
    for name in param_names:
        spec = hyperprior_specs[name]
        tau_0 = np.sqrt(spec['tau_0_sq'])
        print(f"{name:<12} {spec['mu_0']:>8.2f} {tau_0:>8.2f} {spec['alpha_0']:>8.2f} {spec['beta_0']:>8.2f}")

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **params)
    print(f"\nSaved hyperpriors to: {output_path}")

    return params


def run_prior_selection_ce(
    ce_direct_prior: Dict,
    rho: float = 0.2,
    alpha_0: Optional[float] = None,
    kappa: Optional[float] = None,
    lambda_direct_prior: Optional[Dict] = None,
    output_path=None,
    ref_lotteries=None,
    distribution: str = 'normal',
):
    """
    Run CE-space prior selection, derive hyperpriors, and save to file.

    Args:
        ce_direct_prior: Dict with keys 'ce1', 'ce2', 'ce3', each having 'mu' and 'sigma2'
        rho: Variance partition ratio
        alpha_0: IG concentration parameter (one of alpha_0 or kappa required)
        kappa: CV of heterogeneity (one of alpha_0 or kappa required)
        lambda_direct_prior: Dict with 'mu' and 'sigma2' for log(lambda)
        output_path: Path to save NPZ (default: data/output/prior_params_ce.npz)
        ref_lotteries: List of reference lottery dicts (required for distribution='beta')
            Each dict has 'prizes' (list of floats) and 'probs'.
        distribution: 'normal' (NIG hierarchy) or 'beta' (Beta-Beta hierarchy)

    Returns:
        dict with hyperprior parameters
    """
    from pathlib import Path

    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'prior_params_ce.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running CE-space prior selection...")
    ps = ce_space_prior_selection(ce_direct_prior, lambda_direct_prior)
    print(ps.description)

    if distribution == 'beta':
        # --- Beta-Beta path (bounded CEs) ---
        if ref_lotteries is None:
            raise ValueError("ref_lotteries required for distribution='beta'")

        # Extract bounds from reference lotteries
        bounds_lo = np.array([min(lot['prizes']) for lot in ref_lotteries])
        bounds_hi = np.array([max(lot['prizes']) for lot in ref_lotteries])

        hp = derive_hyperpriors_beta(
            ps, bounds_lo, bounds_hi, rho=rho, kappa=kappa, alpha_0=alpha_0,
        )
        print_hyperprior_summary_beta(hp)

        print()
        verify_hyperpriors_beta(hp)

        # Build params dict for saving
        n_ce = len(hp.ce_names)
        params = {'distribution': 'beta'}
        for k in range(n_ce):
            name = hp.ce_names[k]
            params[f'{name}_a_mu'] = float(hp.a_mu[k])
            params[f'{name}_b_mu'] = float(hp.b_mu[k])
            params[f'{name}_a_psi'] = float(hp.a_psi[k])
            params[f'{name}_b_psi'] = float(hp.b_psi[k])
            params[f'{name}_bound_lo'] = float(hp.bounds_lo[k])
            params[f'{name}_bound_hi'] = float(hp.bounds_hi[k])

        params['lambda_raw_mu_0'] = hp.lambda_mu_0
        params['lambda_raw_tau_0_sq'] = hp.lambda_tau_0_sq
        params['lambda_raw_alpha_0'] = hp.lambda_alpha_0
        params['lambda_raw_beta_0'] = hp.lambda_beta_0
        params['rho'] = rho
        params['kappa'] = float(hp.kappa)

    else:
        # --- Normal (NIG) path (existing behavior) ---
        hp = derive_hyperpriors(ps, rho=rho, alpha_0=alpha_0, kappa=kappa)
        print_hyperprior_summary(hp)

        print()
        verify_hyperpriors(hp)

        params = {'distribution': 'normal'}
        for i, name in enumerate(hp.param_names):
            params[f'{name}_mu_0'] = float(hp.mu_0[i])
            params[f'{name}_tau_0_sq'] = float(hp.tau_0_sq[i])
            params[f'{name}_alpha_0'] = float(hp.alpha_0)
            params[f'{name}_beta_0'] = float(hp.beta_0[i])

        params['rho'] = rho
        params['kappa'] = float(hp.kappa)

    np.savez(output_path, **params)
    print(f"\nSaved CE hyperpriors to: {output_path}")

    return params


if __name__ == "__main__":
    # Run prior selection and derive hyperpriors
    print("RDU with 2-Parameter Prelec PWF - Prior Selection")
    print("=" * 60)

    ps = rdu_prelec_prior_selection()
    print(ps.description)

    print("\nPrior Selection Summary:")
    print(f"{'Parameter':<12} {'Mean':>8} {'Std Dev':>8} {'Variance':>10}")
    print("-" * 40)
    for i, name in enumerate(ps.param_names):
        print(f"{name:<12} {ps.mean[i]:>8.3f} {np.sqrt(ps.variance[i]):>8.3f} {ps.variance[i]:>10.4f}")

    # Derive hyperpriors with default settings
    hp = derive_hyperpriors(ps, rho=0.3, kappa=0.5)
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
