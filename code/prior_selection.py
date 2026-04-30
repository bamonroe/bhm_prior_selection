"""
Prior Selection and Hyperprior Derivation for RDU BHM.

This module implements quantity-space prior specification with Beta-Beta hierarchy:

1. Prior Selection: Specify beliefs about certainty equivalents (CEs) and choice
   probabilities for reference lotteries — cognitively tractable quantities.
2. Hyperprior Derivation: Derive Beta-Beta hyperpriors via variance decomposition
   with (alpha, kappa) researcher choices.

The hierarchy for each quantity k (3 CEs + 1 choice probability):
    z_i^k | mu_k, psi_k ~ Beta(mu_k(1-psi_k)/psi_k, (1-mu_k)(1-psi_k)/psi_k)
    mu_k ~ Beta(a_mu[k], b_mu[k])
    psi_k ~ Beta(a_psi[k], b_psi[k])

where z_i^k = (Y_i^k - lo_k) / (hi_k - lo_k) is the normalized quantity.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Dict, Tuple


def variance_from_tail_probability(mu, p_tail, reference_point, lo, hi):
    """
    Compute the variance of a scaled Beta distribution given a mean and
    a tail probability at a reference point.

    For a Beta distribution on [lo, hi] with mean mu, find the unique
    variance such that P(X > reference_point) = p_tail.

    Parameters
    ----------
    mu : float
        Mean of the distribution on the natural scale (lo < mu < hi).
    p_tail : float
        Target probability P(X > reference_point), in (0, 1).
    reference_point : float
        The point at which the tail probability is evaluated (e.g., the EV).
    lo, hi : float
        Bounds of the distribution.

    Returns
    -------
    sigma2 : float
        Variance on the natural scale.
    """
    m = (mu - lo) / (hi - lo)
    z0 = (reference_point - lo) / (hi - lo)

    def target(nu):
        a = m * nu
        b = (1 - m) * nu
        return 1 - beta_dist.cdf(z0, a, b) - p_tail

    nu = brentq(target, 1e-6, 1e6)
    var_norm = m * (1 - m) / (nu + 1)
    return var_norm * (hi - lo) ** 2


def build_ce_direct_prior(reference_lotteries, mean_frac, p_risk_seeking,
                          pchoice_mu):
    """
    Build DIRECT_PRIOR_CE from elicitation inputs.

    All four quantities use the same elicitation logic: a mean and a tail
    probability at a reference point that separates risk-averse from
    risk-seeking behavior.

    For CEs: reference point = EV, P(CE > EV) = p_risk_seeking.
    For choice probability: reference point = 0.5, P(P > 0.5) = p_risk_seeking.

    Parameters
    ----------
    reference_lotteries : list of dict
        Each with 'prizes' and 'probs' keys.
    mean_frac : float
        Mean CE as a fraction of the distance from lo to EV.
    p_risk_seeking : float
        Tail probability for risk-seeking behavior, shared across all quantities.
    pchoice_mu : float
        Mean choice probability P(choose riskier lottery).

    Returns
    -------
    dict with keys 'ce1', 'ce2', 'ce3', 'pchoice', each containing
    'mu', 'sigma2', 'lo', 'hi', and for CEs also 'mean_frac',
    'p_risk_seeking', 'ev'.
    """
    result = {}
    for k, lot in enumerate(reference_lotteries):
        lo = min(lot['prizes'])
        hi = max(lot['prizes'])
        ev = lot['prizes'][0] * lot['probs'][0] + lot['prizes'][1] * lot['probs'][1]
        mu = lo + mean_frac * (ev - lo)
        sigma2 = variance_from_tail_probability(mu, p_risk_seeking, ev, lo, hi)
        result[f'ce{k+1}'] = {
            'mu': mu, 'sigma2': sigma2, 'lo': lo, 'hi': hi,
            'mean_frac': mean_frac, 'p_risk_seeking': p_risk_seeking, 'ev': ev,
        }
    # Choice probability: P(choose riskier lottery), reference point = 0.5
    pchoice_sigma2 = variance_from_tail_probability(
        pchoice_mu, p_risk_seeking, 0.5, 0.0, 1.0)
    result['pchoice'] = {
        'mu': pchoice_mu, 'sigma2': pchoice_sigma2, 'lo': 0.0, 'hi': 1.0,
        'p_risk_seeking': p_risk_seeking,
    }
    return result


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
class DerivedHyperpriorsBeta:
    """Hyperprior parameters for Beta-Beta hierarchy on bounded quantities.

    For each quantity k (3 CEs + 1 choice probability), the hierarchy is:
        z_i^k | mu_k, psi_k ~ Beta(mu_k(1-psi_k)/psi_k, (1-mu_k)(1-psi_k)/psi_k)
        mu_k ~ Beta(a_mu[k], b_mu[k])
        psi_k ~ Beta(a_psi[k], b_psi[k])

    where z_i^k = (Y_i^k - lo_k) / (hi_k - lo_k) is the normalized quantity.
    """
    # Quantity identification (length 4: ce1, ce2, ce3, pchoice)
    quantity_names: Tuple[str, ...]

    # Bounds for each quantity
    bounds_lo: np.ndarray  # lo_k per quantity
    bounds_hi: np.ndarray  # hi_k per quantity

    # Researcher choices
    alpha: float    # variance partition ratio
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
    mu_star: np.ndarray   # normalized mean per quantity
    v_star: np.ndarray    # normalized variance per quantity
    psi_star: np.ndarray  # target variance fraction per quantity

    # Original beliefs (on quantity's own scale)
    ps_mean: np.ndarray
    ps_variance: np.ndarray


def ce_space_prior_selection(
    ce_direct_prior: Dict,
) -> PriorSelectionResult:
    """
    Prior selection in quantity space for 3 CEs + 1 choice probability.

    The researcher specifies Beta beliefs about:
    - CE1, CE2, CE3: certainty equivalents of reference lotteries
    - pchoice: probability of choosing L2 over L3 (for lambda identification)

    Each entry provides (mu, sigma2) on the quantity's natural scale, plus
    bounds (lo, hi) defining the Beta support.

    Parameters:
    -----------
    ce_direct_prior : dict
        Dictionary with keys 'ce1', 'ce2', 'ce3', 'pchoice', each having
        'mu', 'sigma2', 'lo', 'hi'.

    Returns:
    --------
    PriorSelectionResult with 4 components: [CE1, CE2, CE3, pchoice]
    """
    param_names = ('ce1', 'ce2', 'ce3', 'pchoice')

    mean = np.array([
        ce_direct_prior['ce1']['mu'],
        ce_direct_prior['ce2']['mu'],
        ce_direct_prior['ce3']['mu'],
        ce_direct_prior['pchoice']['mu'],
    ])

    variance = np.array([
        ce_direct_prior['ce1']['sigma2'],
        ce_direct_prior['ce2']['sigma2'],
        ce_direct_prior['ce3']['sigma2'],
        ce_direct_prior['pchoice']['sigma2'],
    ])

    description = (
        "    Quantity-space prior selection (Beta-Beta hierarchy):\n"
        + "".join(
            f"\n    {i+1}. {name}: mu={ce_direct_prior[name]['mu']}, "
            f"sd={np.sqrt(ce_direct_prior[name]['sigma2']):.4f}, "
            f"bounds=[{ce_direct_prior[name]['lo']}, {ce_direct_prior[name]['hi']}]"
            for i, name in enumerate(param_names)
        )
    )

    return PriorSelectionResult(
        param_names=param_names,
        mean=mean,
        variance=variance,
        description=description
    )


def derive_hyperpriors_beta(
    quantity_prior: PriorSelectionResult,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    alpha: float = 0.2,
    kappa: float = 0.25,
) -> DerivedHyperpriorsBeta:
    """
    Derive Beta-Beta hyperpriors for bounded quantity-space hierarchy.

    All 4 quantities (3 CEs + choice probability) are treated uniformly:
    1. Normalize to [0,1] using bounds
    2. Moment-match Beta hyperpriors on mu_k (population mean)
    3. Moment-match Beta hyperpriors on psi_k (variance fraction)

    Parameters:
    -----------
    quantity_prior : PriorSelectionResult
        From ce_space_prior_selection(). 4 quantities: (ce1, ce2, ce3, pchoice).
    bounds_lo : np.ndarray, shape (4,)
        Lower bound for each quantity.
    bounds_hi : np.ndarray, shape (4,)
        Upper bound for each quantity.
    alpha : float in (0, 1)
        Variance partition ratio.
    kappa : float > 0
        Coefficient of variation of heterogeneity. Controls how confident
        the researcher is about the level of within-population variance.

    Returns:
    --------
    DerivedHyperpriorsBeta
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa}")

    bounds_lo = np.asarray(bounds_lo, dtype=float)
    bounds_hi = np.asarray(bounds_hi, dtype=float)

    n_q = len(quantity_prior.param_names)
    q_names = quantity_prior.param_names
    mu_raw = quantity_prior.mean
    var_raw = quantity_prior.variance

    if len(bounds_lo) != n_q or len(bounds_hi) != n_q:
        raise ValueError(
            f"bounds must have length {n_q}, "
            f"got bounds_lo={len(bounds_lo)}, bounds_hi={len(bounds_hi)}"
        )

    # --- Step 1: Normalize to [0, 1] ---
    span = bounds_hi - bounds_lo
    mu_tilde = (mu_raw - bounds_lo) / span
    v_tilde = var_raw / span**2

    # --- Step 2: Validate Bernoulli bound ---
    bernoulli_bound = mu_tilde * (1 - mu_tilde)
    for k in range(n_q):
        if v_tilde[k] >= bernoulli_bound[k]:
            raise ValueError(
                f"'{q_names[k]}': normalized variance {v_tilde[k]:.6f} >= "
                f"Bernoulli bound mu*(1-mu) = {bernoulli_bound[k]:.6f}. "
                f"Reduce sigma2 or adjust mu to be further from bounds."
            )
        if not (0 < mu_tilde[k] < 1):
            raise ValueError(
                f"'{q_names[k]}': normalized mean {mu_tilde[k]:.6f} not in (0,1). "
                f"Mean {mu_raw[k]} must be between bounds [{bounds_lo[k]}, {bounds_hi[k]}]."
            )

    # --- Step 3: Mean hyperprior mu_k ~ Beta(a_mu, b_mu) ---
    nu_mu = bernoulli_bound / (alpha * v_tilde) - 1
    a_mu = mu_tilde * nu_mu
    b_mu = (1 - mu_tilde) * nu_mu

    for k in range(n_q):
        if nu_mu[k] <= 0:
            raise ValueError(
                f"'{q_names[k]}': nu_mu = {nu_mu[k]:.4f} <= 0. "
                f"alpha * v_tilde = {alpha * v_tilde[k]:.6f} >= Bernoulli bound "
                f"{bernoulli_bound[k]:.6f}. Reduce alpha or sigma2."
            )

    # --- Step 4: Variance fraction target psi* ---
    denom = bernoulli_bound - alpha * v_tilde
    psi_star = (1 - alpha) * v_tilde / denom

    for k in range(n_q):
        if not (0 < psi_star[k] < 1):
            raise ValueError(
                f"'{q_names[k]}': psi_star = {psi_star[k]:.6f} not in (0,1). "
                f"The within-population variance target is infeasible."
            )

    # --- Step 5: Dispersion hyperprior psi_k ~ Beta(a_psi, b_psi) ---
    # κ is the CV of the conditional variance μ(1-μ)ψ, not the CV of ψ.
    # Under independence of μ and ψ:
    #   Var[μ(1-μ)ψ] = E[μ²(1-μ)²]·Var[ψ] + Var[μ(1-μ)]·E[ψ]²
    # Setting the LHS to κ²·((1-ρ)v)² and solving for Var[ψ]:
    S_mu = a_mu + b_mu
    e_mu = (a_mu * (a_mu + 1) * b_mu * (b_mu + 1)
            / (S_mu * (S_mu + 1) * (S_mu + 2) * (S_mu + 3)))
    c_mu = denom  # = E[μ(1-μ)] = μ̃(1-μ̃) - ρṽ
    d_mu = e_mu - c_mu**2  # Var[μ(1-μ)]

    target_var_cond = kappa**2 * ((1 - alpha) * v_tilde)**2
    var_psi = (target_var_cond - d_mu * psi_star**2) / e_mu

    for k in range(n_q):
        kappa_lb = np.sqrt(d_mu[k]) / c_mu[k]
        if var_psi[k] <= 0:
            raise ValueError(
                f"'{q_names[k]}': kappa={kappa:.4f} is below the lower bound "
                f"{kappa_lb:.4f} for this family. The variation in μ(1-μ) "
                f"alone exceeds the target CV of conditional variance."
            )

    nu_psi = psi_star * (1 - psi_star) / var_psi - 1
    a_psi = psi_star * nu_psi
    b_psi = (1 - psi_star) * nu_psi

    for k in range(n_q):
        if nu_psi[k] <= 0:
            raise ValueError(
                f"'{q_names[k]}': nu_psi = {nu_psi[k]:.4f} <= 0. "
                f"psi_star={psi_star[k]:.4f}, kappa={kappa:.4f}. "
                f"Var[psi] exceeds the Beta feasibility bound."
            )

    return DerivedHyperpriorsBeta(
        quantity_names=q_names,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        alpha=alpha,
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
        ps_mean=mu_raw,
        ps_variance=var_raw,
    )


def verify_hyperpriors_beta(hp: DerivedHyperpriorsBeta, n_mc: int = 100_000,
                            seed: int = 42) -> bool:
    """
    Monte Carlo verification that Beta-Beta hyperpriors reproduce targets.

    Draws from the full hierarchy and checks:
    1. E[z_k] ≈ mu_star[k]  (unconditional mean)
    2. Var[z_k] ≈ v_star[k]  (unconditional variance)
    3. Var[mu_k] ≈ alpha * v_star[k]  (variance decomposition)
    """
    rng = np.random.default_rng(seed)
    n_q = len(hp.quantity_names)
    all_ok = True

    print("Beta-Beta Hyperprior Verification (Monte Carlo)")
    print("=" * 70)
    print(f"  MC draws: {n_mc:,}")
    print(f"  alpha = {hp.alpha}, kappa = {hp.kappa:.4f}")

    for k in range(n_q):
        mu_draws = rng.beta(hp.a_mu[k], hp.b_mu[k], size=n_mc)
        psi_draws = rng.beta(hp.a_psi[k], hp.b_psi[k], size=n_mc)

        alpha_pop = mu_draws * (1 - psi_draws) / psi_draws
        beta_pop = (1 - mu_draws) * (1 - psi_draws) / psi_draws
        z_draws = rng.beta(alpha_pop, beta_pop)

        mc_mean = np.mean(z_draws)
        mc_var = np.var(z_draws)
        mc_var_mu = np.var(mu_draws)

        # CV of conditional variance μ(1-μ)ψ — this is what κ should match
        cond_var_draws = mu_draws * (1 - mu_draws) * psi_draws
        mc_cv_cond_var = np.std(cond_var_draws) / np.mean(cond_var_draws)

        target_mean = hp.mu_star[k]
        target_var = hp.v_star[k]
        target_var_mu = hp.alpha * hp.v_star[k]
        target_kappa = hp.kappa

        mean_ok = np.abs(mc_mean - target_mean) < 0.01
        var_ok = np.abs(mc_var - target_var) / target_var < 0.05
        var_mu_ok = np.abs(mc_var_mu - target_var_mu) / target_var_mu < 0.1
        kappa_ok = np.abs(mc_cv_cond_var - target_kappa) / target_kappa < 0.1

        ok = mean_ok and var_ok and var_mu_ok and kappa_ok
        all_ok = all_ok and ok

        print(f"\n  {hp.quantity_names[k]}:")
        print(f"    E[z]:      MC={mc_mean:.5f}  target={target_mean:.5f}  "
              f"{'OK' if mean_ok else 'FAIL'}")
        print(f"    Var[z]:    MC={mc_var:.6f}  target={target_var:.6f}  "
              f"{'OK' if var_ok else 'FAIL'}")
        print(f"    Var[mu]:   MC={mc_var_mu:.6f}  target={target_var_mu:.6f}  "
              f"{'OK' if var_mu_ok else 'FAIL'}")
        print(f"    kappa:     MC={mc_cv_cond_var:.5f}  target={target_kappa:.5f}  "
              f"{'OK' if kappa_ok else 'FAIL'}")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'All checks passed' if all_ok else 'SOME CHECKS FAILED'}")

    return all_ok


def print_hyperprior_summary_beta(hp: DerivedHyperpriorsBeta):
    """Print a summary table of the Beta-Beta hyperprior derivation."""
    n_q = len(hp.quantity_names)

    print("\nBeta-Beta Hyperprior Summary")
    print("=" * 78)
    print(f"Researcher choices: alpha = {hp.alpha}, kappa = {hp.kappa:.4f}")

    # Raw-scale beliefs
    print(f"\n{'Qty':<9} {'mu':>8} {'sd':>8} {'lo':>9} {'hi':>9} "
          f"{'mu_tilde':>9} {'v_tilde':>10}")
    print("-" * 78)
    for k in range(n_q):
        print(f"{hp.quantity_names[k]:<9} {hp.ps_mean[k]:>8.4f} "
              f"{np.sqrt(hp.ps_variance[k]):>8.4f} "
              f"{hp.bounds_lo[k]:>9.2f} {hp.bounds_hi[k]:>9.2f} "
              f"{hp.mu_star[k]:>9.4f} {hp.v_star[k]:>10.6f}")

    # Mean hyperprior
    print(f"\nMean hyperprior: mu_k ~ Beta(a_mu, b_mu)")
    print(f"{'Qty':<9} {'nu_mu':>8} {'a_mu':>8} {'b_mu':>8}")
    print("-" * 42)
    for k in range(n_q):
        print(f"{hp.quantity_names[k]:<9} {hp.nu_mu[k]:>8.2f} "
              f"{hp.a_mu[k]:>8.3f} {hp.b_mu[k]:>8.3f}")

    # Dispersion hyperprior
    print(f"\nDispersion hyperprior: psi_k ~ Beta(a_psi, b_psi)")
    print(f"{'Qty':<9} {'psi*':>8} {'nu_psi':>8} {'a_psi':>8} {'b_psi':>8}")
    print("-" * 52)
    for k in range(n_q):
        print(f"{hp.quantity_names[k]:<9} {hp.psi_star[k]:>8.4f} "
              f"{hp.nu_psi[k]:>8.2f} {hp.a_psi[k]:>8.3f} {hp.b_psi[k]:>8.3f}")

    print(f"\nPopulation: z_i^k | mu_k, psi_k ~ "
          f"Beta(mu*(1-psi)/psi, (1-mu)*(1-psi)/psi)")


def run_prior_selection_ce(
    ce_direct_prior: Dict,
    alpha: float = 0.2,
    kappa: float = 0.25,
    output_path=None,
):
    """
    Run quantity-space prior selection (Beta-Beta), derive hyperpriors, save.

    Args:
        ce_direct_prior: Dict with keys 'ce1', 'ce2', 'ce3', 'pchoice',
            each having 'mu', 'sigma2', 'lo', 'hi'.
        alpha: Variance partition ratio
        kappa: CV of heterogeneity
        output_path: Path to save NPZ (default: data/output/prior_params_ce.npz)

    Returns:
        dict with hyperprior parameters
    """
    from pathlib import Path

    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'prior_params_ce.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running quantity-space prior selection (Beta-Beta)...")
    ps = ce_space_prior_selection(ce_direct_prior)
    print(ps.description)

    # Extract bounds from direct prior config
    quantity_keys = ['ce1', 'ce2', 'ce3', 'pchoice']
    bounds_lo = np.array([ce_direct_prior[k]['lo'] for k in quantity_keys])
    bounds_hi = np.array([ce_direct_prior[k]['hi'] for k in quantity_keys])

    hp = derive_hyperpriors_beta(
        ps, bounds_lo, bounds_hi, alpha=alpha, kappa=kappa,
    )
    print_hyperprior_summary_beta(hp)

    print()
    verify_hyperpriors_beta(hp)

    # Build params dict for saving: 4 quantities x 6 values each = 24 values
    n_q = len(hp.quantity_names)
    params = {'distribution': 'beta'}
    for k in range(n_q):
        name = hp.quantity_names[k]
        params[f'{name}_a_mu'] = float(hp.a_mu[k])
        params[f'{name}_b_mu'] = float(hp.b_mu[k])
        params[f'{name}_a_psi'] = float(hp.a_psi[k])
        params[f'{name}_b_psi'] = float(hp.b_psi[k])
        params[f'{name}_bound_lo'] = float(hp.bounds_lo[k])
        params[f'{name}_bound_hi'] = float(hp.bounds_hi[k])

    params['alpha'] = alpha
    params['kappa'] = float(hp.kappa)

    np.savez(output_path, **params)
    print(f"\nSaved CE hyperpriors to: {output_path}")

    return params


