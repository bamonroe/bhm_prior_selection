"""
Prior Predictive Certainty Equivalent Distribution.

This module translates abstract prior parameter distributions into an
economically interpretable quantity: the certainty equivalent (CE) of a
concrete lottery. This makes it easy to assess whether the prior implies
reasonable preference behavior.

The lottery uses 3 equally likely prizes at the dataset's global minimum,
midpoint, and maximum prize values.

For each MC draw:
    1. Draw hyperparameters from the hyperprior
    2. For each subject, draw individual parameters from the population
    3. Compute the RDU certainty equivalent of the reference lottery
    4. Collect CEs across all MC draws and subjects

The output is a distribution of CEs that can be compared against the
expected value and prize range to assess prior plausibility.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from posteriors.common_fns import crra_ufn, prelec_pwf


# =============================================================================
# CRRA Inverse Utility
# =============================================================================

def _crra_inverse(v, r):
    """
    Inverse of CRRA utility function (JAX).

    Given u(x) = x^(1-r) / (1-r), the inverse is:
        x = ((1-r)*v)^(1/(1-r))

    For r close to 1 (log utility), u(x) = log(x) so x = exp(v).

    Args:
        v: Utility value (scalar)
        r: CRRA coefficient (scalar)

    Returns:
        Certainty equivalent in normalized units
    """
    # Standard CRRA inverse: x = ((1-r)*v)^(1/(1-r))
    inner = jnp.maximum((1 - r) * v, 1e-30)
    crra_inv = jnp.power(inner, 1.0 / (1 - r))

    # Log utility inverse: x = exp(v)
    log_inv = jnp.exp(v)

    # Use log when |r - 1| < 0.01 (matching crra_ufn threshold)
    use_log = jnp.abs(r - 1.0) < 0.01
    return jnp.where(use_log, log_inv, crra_inv)


# =============================================================================
# RDU for 3-Outcome Lottery
# =============================================================================

def _rdu_3outcome(r, phi, eta, prizes):
    """
    Compute RDU value for a 3-outcome equally-likely lottery.

    For prizes x1 <= x2 <= x3 with equal probability 1/3:
        Decision weights (Quiggin 1982 / Diecidue-Wakker):
            pi3 = w(1/3)
            pi2 = w(2/3) - w(1/3)
            pi1 = 1 - w(2/3)
        RDU = pi1*u(x1) + pi2*u(x2) + pi3*u(x3)

    Note: RDU ranks outcomes best-to-worst and applies w to cumulative
    probability of getting outcome "at least as good". For a risk-averse
    decision maker who overweights bad outcomes, pi1 > 1/3.

    Args:
        r: CRRA coefficient
        phi: Prelec curvature (natural scale, positive)
        eta: Prelec elevation (natural scale, positive)
        prizes: Array of 3 prizes [x1, x2, x3] sorted ascending

    Returns:
        RDU utility value (scalar)
    """
    x1, x2, x3 = prizes[0], prizes[1], prizes[2]

    # Utilities
    u1 = crra_ufn(x1, r)
    u2 = crra_ufn(x2, r)
    u3 = crra_ufn(x3, r)

    # Probability weights using Prelec PWF
    # w(p) = exp(-eta * (-ln(p))^phi)
    # Cumulative probability of getting at least as good as x_k (rank from best)
    # P(X >= x3) = 1/3, P(X >= x2) = 2/3, P(X >= x1) = 1
    w_one_third = prelec_pwf(1.0 / 3.0, phi, eta)
    w_two_thirds = prelec_pwf(2.0 / 3.0, phi, eta)

    # Decision weights
    pi3 = w_one_third                  # best outcome
    pi2 = w_two_thirds - w_one_third   # middle outcome
    pi1 = 1.0 - w_two_thirds          # worst outcome

    return pi1 * u1 + pi2 * u2 + pi3 * u3


# =============================================================================
# JAX Monte Carlo Computation
# =============================================================================

def compute_prior_predictive_ce(
    hyperprior_params,
    prizes,
    n_subjects,
    n_mc_draws,
    seed=42,
):
    """
    Compute prior predictive CE distribution via Monte Carlo.

    For each MC draw:
        1. Draw hyperparams: mu_j ~ N(mu_0_j, tau_0_sq_j),
                             sigma2_j ~ InvGamma(alpha_0, beta_0_j)
        2. For each subject: draw (r, phi_raw, eta_raw, lambda_raw) ~ N(mu_j, sigma2_j)
        3. Transform: phi = exp(phi_raw), eta = exp(eta_raw)
        4. Compute RDU of the 3-outcome lottery, then CE = u_inverse(RDU)

    Args:
        hyperprior_params: Dict with keys mu_0, tau_0_sq, alpha_0, beta_0
                           (each except alpha_0 is array of length 4)
        prizes: Array of 3 prizes [min, mid, max], sorted ascending
        n_subjects: Number of subjects per MC draw
        n_mc_draws: Number of MC draws
        seed: Random seed

    Returns:
        Array of shape (n_mc_draws, n_subjects) with CEs
    """
    mu_0 = jnp.array(hyperprior_params['mu_0'])          # (4,)
    tau_0_sq = jnp.array(hyperprior_params['tau_0_sq'])   # (4,)
    alpha_0 = hyperprior_params['alpha_0']                # scalar
    beta_0 = jnp.array(hyperprior_params['beta_0'])       # (4,)

    prizes_jax = jnp.array(prizes)

    key = random.PRNGKey(seed)

    def single_mc_draw(key):
        """One MC draw: sample hyperparams, then subjects, then compute CEs."""
        k1, k2, k3 = random.split(key, 3)

        # --- Draw population-level parameters ---
        # mu_j ~ N(mu_0_j, tau_0_sq_j)
        mu_j = random.normal(k1, shape=(4,)) * jnp.sqrt(tau_0_sq) + mu_0

        # sigma2_j ~ InvGamma(alpha_0, beta_0_j)
        # If X ~ Gamma(alpha, 1), then beta/X ~ InvGamma(alpha, beta)
        gamma_draws = random.gamma(k2, alpha_0, shape=(4,))
        gamma_draws = jnp.maximum(gamma_draws, 1e-10)
        sigma2_j = beta_0 / gamma_draws

        # --- Draw subject-level parameters ---
        # theta_ij ~ N(mu_j, sigma2_j) for each subject
        z = random.normal(k3, shape=(n_subjects, 4))
        theta = z * jnp.sqrt(sigma2_j)[None, :] + mu_j[None, :]

        # Extract and transform
        r_vals = theta[:, 0]             # CRRA (untransformed)
        phi_vals = jnp.exp(theta[:, 1])  # Prelec curvature (exp of raw)
        eta_vals = jnp.exp(theta[:, 2])  # Prelec elevation (exp of raw)
        # lambda (theta[:, 3]) not needed for CE computation

        def single_subject_ce(r, phi, eta):
            """Compute CE for one subject."""
            rdu_val = _rdu_3outcome(r, phi, eta, prizes_jax)
            return _crra_inverse(rdu_val, r)

        # Vectorize over subjects
        ces = jax.vmap(single_subject_ce)(r_vals, phi_vals, eta_vals)
        return ces

    # Vectorize over MC draws
    keys = random.split(key, n_mc_draws)
    all_ces = jax.vmap(single_mc_draw)(keys)  # (n_mc_draws, n_subjects)

    return all_ces


# =============================================================================
# Plotting
# =============================================================================

def plot_prior_predictive_ce(ce_flat, prizes, output_path):
    """
    Plot histogram of prior predictive CE distribution.

    Args:
        ce_flat: 1D array of CE values
        prizes: Array of 3 prizes [min, mid, max]
        output_path: Path to save PDF
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ce_flat = np.asarray(ce_flat)
    prizes = np.asarray(prizes)

    # Expected value under risk neutrality
    ev = np.mean(prizes)

    # Summary statistics
    median_ce = np.median(ce_flat)
    ci_low, ci_high = np.percentile(ce_flat, [2.5, 97.5])
    mean_ce = np.mean(ce_flat)

    # Clip display range to 1st-99th percentile for readability
    x_lo, x_hi = np.percentile(ce_flat, [1, 99])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(ce_flat, bins=150, density=True, alpha=0.6, color='steelblue',
            range=(0, 50))

    # Reference lines
    ax.axvline(ev, color='black', linestyle=':', linewidth=1.5)
    for p in prizes:
        ax.axvline(p, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(median_ce, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(ci_low, color='red', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axvline(ci_high, color='red', linestyle=':', linewidth=1.0, alpha=0.7)

    # Annotation
    text = (f'Mean = {mean_ce:.0f}\n'
            f'Median = {median_ce:.0f}\n'
            f'95% CI = [{ci_low:.0f}, {ci_high:.0f}]')
    ax.text(0.97, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Certainty Equivalent')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 50)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_prior_predictive_ce(
    direct_prior,
    prior_selection,
    data_dir,
    output_dir,
    n_mc_draws=10000,
    seed=42,
):
    """
    Run prior predictive CE analysis.

    Args:
        direct_prior: Dict with per-param mu/sigma2 (same as DIRECT_PRIOR in main.py)
        prior_selection: Dict with 'rho' and 'alpha_0'
        data_dir: Path to data directory (to load clean data for prize range)
        output_dir: Path for output plot
        n_mc_draws: Number of Monte Carlo draws (default 10000)
        seed: Random seed
    """
    from clean_data import load_clean_data
    from prior_selection import rdu_prelec_prior_selection, derive_hyperpriors

    # --- Load data to find prize range ---
    print("  Loading clean data for prize range...")
    clean = load_clean_data()

    all_prizes = np.concatenate([
        np.asarray(clean['prizes_L']).ravel(),
        np.asarray(clean['prizes_R']).ravel(),
    ])
    prize_min = float(np.min(all_prizes))
    prize_max = float(np.max(all_prizes))
    prize_mid = (prize_min + prize_max) / 2.0
    prizes_ref = np.array([prize_min, prize_mid, prize_max])

    n_subjects = int(clean['n_subjects'])

    print(f"  Prizes: min={prize_min:.0f}, mid={prize_mid:.0f}, max={prize_max:.0f}")
    print(f"  EV = {np.mean(prizes_ref):.0f}")
    print(f"  N subjects: {n_subjects}")

    # --- Derive hyperpriors ---
    print("  Deriving hyperpriors from direct prior...")
    ps = rdu_prelec_prior_selection(direct_prior)
    hp = derive_hyperpriors(ps, rho=prior_selection['rho'],
                            alpha_0=prior_selection['alpha_0'])

    hyperprior_params = {
        'mu_0': hp.mu_0,
        'tau_0_sq': hp.tau_0_sq,
        'alpha_0': hp.alpha_0,
        'beta_0': hp.beta_0,
    }

    # --- Compute prior predictive CEs ---
    print(f"  Computing prior predictive CEs ({n_mc_draws} MC draws x {n_subjects} subjects)...")
    all_ces = compute_prior_predictive_ce(
        hyperprior_params=hyperprior_params,
        prizes=prizes_ref,
        n_subjects=n_subjects,
        n_mc_draws=n_mc_draws,
        seed=seed,
    )
    all_ces_np = np.asarray(all_ces)
    ce_flat = all_ces_np.ravel()

    # Filter out non-finite values
    finite_mask = np.isfinite(ce_flat)
    n_nonfinite = np.sum(~finite_mask)
    if n_nonfinite > 0:
        print(f"  Warning: {n_nonfinite} non-finite CEs ({n_nonfinite/len(ce_flat)*100:.1f}%) removed")
    ce_flat = ce_flat[finite_mask]

    # --- Summary statistics ---
    print(f"\n  Prior Predictive CE Summary:")
    print(f"    Mean   = {np.mean(ce_flat):.1f}")
    print(f"    Median = {np.median(ce_flat):.1f}")
    print(f"    Std    = {np.std(ce_flat):.1f}")
    print(f"    95% CI = [{np.percentile(ce_flat, 2.5):.1f}, {np.percentile(ce_flat, 97.5):.1f}]")
    print(f"    EV     = {np.mean(prizes_ref):.1f} (risk-neutral reference)")

    # --- Save NPZ ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(data_dir) / 'output' / 'prior_predictive_ce.npz'
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        ce_samples=all_ces_np,
        prizes=prizes_ref,
        n_mc_draws=n_mc_draws,
        n_subjects=n_subjects,
        mean=np.mean(ce_flat),
        median=np.median(ce_flat),
        std=np.std(ce_flat),
        ci_low=np.percentile(ce_flat, 2.5),
        ci_high=np.percentile(ce_flat, 97.5),
        ev=np.mean(prizes_ref),
    )
    print(f"  Saved NPZ to: {npz_path}")

    # --- Plot ---
    plot_path = output_dir / 'prior_predictive_ce.pdf'
    plot_prior_predictive_ce(ce_flat, prizes_ref, plot_path)


# =============================================================================
# Posterior Predictive CE
# =============================================================================

def compute_posterior_predictive_ce(r_all, phi_all, eta_all, prizes):
    """
    Compute posterior predictive CE from MCMC posterior samples.

    Args:
        r_all: Array of shape (n_draws, n_subjects) - CRRA coefficients
        phi_all: Array of shape (n_draws, n_subjects) - Prelec curvature (natural)
        eta_all: Array of shape (n_draws, n_subjects) - Prelec elevation (natural)
        prizes: Array of 3 prizes [min, mid, max]

    Returns:
        Array of shape (n_draws, n_subjects) with CEs
    """
    prizes_jax = jnp.array(prizes)

    def single_ce(r, phi, eta):
        rdu_val = _rdu_3outcome(r, phi, eta, prizes_jax)
        return _crra_inverse(rdu_val, r)

    # vmap over (draw, subject) — flatten then reshape
    r_flat = r_all.ravel()
    phi_flat = phi_all.ravel()
    eta_flat = eta_all.ravel()

    ces_flat = jax.vmap(single_ce)(r_flat, phi_flat, eta_flat)
    return ces_flat.reshape(r_all.shape)


def plot_posterior_predictive_ce(ce_flat, prizes, output_path, label,
                                  prior_ce_flat=None):
    """
    Plot histogram of posterior predictive CE distribution.

    Args:
        ce_flat: 1D array of posterior CE values
        prizes: Array of 3 prizes [min, mid, max]
        output_path: Path to save PDF
        label: String label ('informative' or 'uninformative')
        prior_ce_flat: Optional 1D array of prior predictive CEs for overlay
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ce_flat = np.asarray(ce_flat)
    prizes = np.asarray(prizes)

    ev = np.mean(prizes)

    median_ce = np.median(ce_flat)
    ci_low, ci_high = np.percentile(ce_flat, [2.5, 97.5])
    mean_ce = np.mean(ce_flat)

    # Display range from 1st-99th percentile
    x_lo, x_hi = np.percentile(ce_flat, [1, 99])

    # Widen range if prior overlay extends further
    if prior_ce_flat is not None:
        prior_ce_flat = np.asarray(prior_ce_flat)
        p_lo, p_hi = np.percentile(prior_ce_flat, [1, 99])
        x_lo = min(x_lo, p_lo)
        x_hi = max(x_hi, p_hi)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Prior overlay (if provided)
    if prior_ce_flat is not None:
        ax.hist(prior_ce_flat, bins=150, density=True, alpha=0.3,
                color='steelblue', range=(x_lo, x_hi),
                label='Prior predictive CE')

    # Posterior histogram
    ax.hist(ce_flat, bins=150, density=True, alpha=0.6, color='coral',
            range=(x_lo, x_hi), label='Posterior predictive CE')

    # Reference lines
    ax.axvline(ev, color='black', linestyle='-', linewidth=1.5,
               label=f'EV = {ev:.0f}')
    for p in prizes:
        ax.axvline(p, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(median_ce, color='darkred', linestyle='--', linewidth=1.5,
               label=f'Posterior median = {median_ce:.0f}')
    ax.axvline(ci_low, color='darkred', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axvline(ci_high, color='darkred', linestyle=':', linewidth=1.0, alpha=0.7)

    # Annotation
    text = (f'Mean = {mean_ce:.0f}\n'
            f'Median = {median_ce:.0f}\n'
            f'95% CI = [{ci_low:.0f}, {ci_high:.0f}]')
    ax.text(0.97, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Prize labels
    prize_labels = ['Min', 'Mid', 'Max']
    for i, p in enumerate(prizes):
        if x_lo <= p <= x_hi:
            ax.text(p, ax.get_ylim()[1] * 0.02, f'{prize_labels[i]}\n{p:.0f}',
                    ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_xlabel('Certainty Equivalent')
    ax.set_ylabel('Density')
    label_title = label.capitalize()
    ax.set_title(f'Posterior Predictive CE Distribution ({label_title} Prior)\n'
                 '(3 equally-likely prizes: min, mid, max)')
    ax.set_xlim(x_lo, x_hi)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {output_path}")


def plot_subject_ce_comparison(ce_info, ce_uninf, prizes, sid,
                               output_path):
    """
    Plot overlaid histograms comparing informative vs uninformative posterior
    predictive CE distributions for a single subject.

    Args:
        ce_info: 1D array of CE draws (informative prior) for one subject
        ce_uninf: 1D array of CE draws (uninformative prior) for one subject
        prizes: Array of 3 prizes [min, mid, max]
        sid: Subject index (0-based)
        output_path: Path to save PDF
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ce_info = np.asarray(ce_info)
    ce_uninf = np.asarray(ce_uninf)
    prizes = np.asarray(prizes)

    ev = np.mean(prizes)

    # Filter non-finite
    ce_info = ce_info[np.isfinite(ce_info)]
    ce_uninf = ce_uninf[np.isfinite(ce_uninf)]

    med_info = np.median(ce_info)
    med_uninf = np.median(ce_uninf)
    med_diff = abs(med_info - med_uninf)

    # Display range: 1st-99th percentile of combined data
    combined = np.concatenate([ce_info, ce_uninf])
    x_lo, x_hi = np.percentile(combined, [1, 99])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(ce_uninf, bins=150, density=True, alpha=0.5, color='steelblue',
            range=(x_lo, x_hi), label='Uninformative prior')
    ax.hist(ce_info, bins=150, density=True, alpha=0.5, color='coral',
            range=(x_lo, x_hi), label='Informative prior')

    # Median lines
    ax.axvline(med_uninf, color='steelblue', linestyle='-', linewidth=1.5,
               label=f'Uninf. median = {med_uninf:.0f}')
    ax.axvline(med_info, color='coral', linestyle='-', linewidth=1.5,
               label=f'Info. median = {med_info:.0f}')

    # EV reference
    ax.axvline(ev, color='black', linestyle='-', linewidth=1.5,
               label=f'EV = {ev:.0f}')

    # Prize lines
    for p in prizes:
        ax.axvline(p, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Text box
    text = (f'Median diff = {med_diff:.0f}\n'
            f'Info. median = {med_info:.0f}\n'
            f'Uninf. median = {med_uninf:.0f}')
    ax.text(0.97, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Certainty Equivalent')
    ax.set_ylabel('Density')
    ax.set_title(f'Subject {sid}: Posterior Predictive CE\n'
                 '(Informative vs Uninformative Prior)')
    ax.set_xlim(x_lo, x_hi)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {output_path}")


def run_posterior_ce_comparison(data_dir, output_dir):
    """
    Compare posterior predictive CE distributions per subject between
    informative and uninformative priors.

    Loads existing NPZ files, computes per-subject divergence (absolute
    median difference), ranks subjects, and generates overlaid histogram
    plots for the most- and least-affected subjects.

    Args:
        data_dir: Path to data directory (contains output/ with NPZ files)
        output_dir: Path for output plots
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    info_path = data_dir / 'output' / 'posterior_predictive_ce_informative.npz'
    uninf_path = data_dir / 'output' / 'posterior_predictive_ce_uninformative.npz'

    print("  Loading posterior predictive CE files...")
    info_data = np.load(info_path)
    uninf_data = np.load(uninf_path)

    ce_info = info_data['ce_samples']    # (n_draws, n_subjects)
    ce_uninf = uninf_data['ce_samples']  # (n_draws, n_subjects)
    prizes = info_data['prizes']
    n_subjects = ce_info.shape[1]

    print(f"  Informative:   {ce_info.shape[0]} draws x {n_subjects} subjects")
    print(f"  Uninformative: {ce_uninf.shape[0]} draws x {n_subjects} subjects")

    # --- Per-subject statistics ---
    print("  Computing per-subject statistics...")

    med_info = np.nanmedian(ce_info, axis=0)
    med_uninf = np.nanmedian(ce_uninf, axis=0)
    mean_info = np.nanmean(ce_info, axis=0)
    mean_uninf = np.nanmean(ce_uninf, axis=0)
    std_info = np.nanstd(ce_info, axis=0)
    std_uninf = np.nanstd(ce_uninf, axis=0)
    p5_info, p95_info = np.nanpercentile(ce_info, [5, 95], axis=0)
    p5_uninf, p95_uninf = np.nanpercentile(ce_uninf, [5, 95], axis=0)

    # Divergence: absolute difference in medians
    divergence = np.abs(med_info - med_uninf)

    # Rank subjects (descending divergence)
    ranking = np.argsort(-divergence)

    most_affected_sid = int(ranking[0])
    least_affected_sid = int(ranking[-1])

    # --- Print rankings ---
    print(f"\n  Top 5 most affected subjects (by |median diff|):")
    print(f"  {'Rank':>4}  {'SID':>5}  {'Med Info':>10}  {'Med Uninf':>10}  {'|Diff|':>10}")
    for i in range(5):
        sid = ranking[i]
        print(f"  {i+1:>4}  {sid:>5}  {med_info[sid]:>10.1f}  "
              f"{med_uninf[sid]:>10.1f}  {divergence[sid]:>10.1f}")

    print(f"\n  Top 5 least affected subjects (by |median diff|):")
    for i in range(5):
        sid = ranking[-(i+1)]
        print(f"  {i+1:>4}  {sid:>5}  {med_info[sid]:>10.1f}  "
              f"{med_uninf[sid]:>10.1f}  {divergence[sid]:>10.1f}")

    print(f"\n  Most affected:  subject {most_affected_sid} "
          f"(|diff| = {divergence[most_affected_sid]:.1f} )")
    print(f"  Least affected: subject {least_affected_sid} "
          f"(|diff| = {divergence[least_affected_sid]:.1f} )")

    # --- Save comparison NPZ ---
    npz_path = data_dir / 'output' / 'posterior_ce_comparison.npz'
    np.savez(
        npz_path,
        med_info=med_info,
        med_uninf=med_uninf,
        mean_info=mean_info,
        mean_uninf=mean_uninf,
        std_info=std_info,
        std_uninf=std_uninf,
        p5_info=p5_info,
        p95_info=p95_info,
        p5_uninf=p5_uninf,
        p95_uninf=p95_uninf,
        divergence=divergence,
        ranking=ranking,
        most_affected_sid=most_affected_sid,
        least_affected_sid=least_affected_sid,
        prizes=prizes,
        n_subjects=n_subjects,
    )
    print(f"  Saved comparison NPZ to: {npz_path}")

    # --- Plot most and least affected subjects ---
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_subject_ce_comparison(
        ce_info[:, most_affected_sid],
        ce_uninf[:, most_affected_sid],
        prizes, most_affected_sid,
        output_dir / 'subject_ce_most_affected.pdf',
    )
    plot_subject_ce_comparison(
        ce_info[:, least_affected_sid],
        ce_uninf[:, least_affected_sid],
        prizes, least_affected_sid,
        output_dir / 'subject_ce_least_affected.pdf',
    )


def run_posterior_predictive_ce(history_dir, data_dir, output_dir, label,
                                 prior_ce_path=None):
    """
    Run posterior predictive CE analysis from MCMC history files.

    Args:
        history_dir: Path to directory containing history_*.npz files
        data_dir: Path to data directory (to load clean data for prize range)
        output_dir: Path for output plot
        label: String label ('informative' or 'uninformative')
        prior_ce_path: Optional path to prior predictive CE NPZ for overlay
    """
    from clean_data import load_clean_data
    from helpers import load_and_filter_history

    history_dir = Path(history_dir)
    output_dir = Path(output_dir)

    # --- Load data for prize range ---
    print("  Loading clean data for prize range...")
    clean = load_clean_data()

    all_prizes = np.concatenate([
        np.asarray(clean['prizes_L']).ravel(),
        np.asarray(clean['prizes_R']).ravel(),
    ])
    prize_min = float(np.min(all_prizes))
    prize_max = float(np.max(all_prizes))
    prize_mid = (prize_min + prize_max) / 2.0
    prizes_ref = np.array([prize_min, prize_mid, prize_max])
    n_subjects = int(clean['n_subjects'])

    print(f"  Prizes: min={prize_min:.0f}, mid={prize_mid:.0f}, max={prize_max:.0f}")
    print(f"  EV = {np.mean(prizes_ref):.0f}")
    print(f"  N subjects: {n_subjects}")

    # --- Load last history file ---
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    last_file = history_files[-1:]
    print(f"  Loading last history file: {last_file[0].name}")
    history, iterations, _, metadata = load_and_filter_history(last_file)
    print(f"  History shape: {history.shape}")

    # --- Apply burn-in (drop first half) ---
    n_samples = history.shape[0]
    burn = n_samples // 2
    history_post = history[burn:]
    n_post = history_post.shape[0]
    n_chains = history_post.shape[1]
    print(f"  After burn-in: {n_post} samples x {n_chains} chains")

    # --- Extract generated quantities ---
    NPARAMS = 4
    N_HYPER = 8
    first_gq = n_subjects * NPARAMS + N_HYPER

    r_all = history_post[:, :, first_gq : first_gq + n_subjects]
    phi_all = history_post[:, :, first_gq + n_subjects : first_gq + 2 * n_subjects]
    eta_all = history_post[:, :, first_gq + 2 * n_subjects : first_gq + 3 * n_subjects]
    # Each is (n_post, n_chains, n_subjects)

    print(f"  r range: [{r_all.min():.3f}, {r_all.max():.3f}]")
    print(f"  phi range: [{phi_all.min():.3f}, {phi_all.max():.3f}]")
    print(f"  eta range: [{eta_all.min():.3f}, {eta_all.max():.3f}]")

    # Reshape to (n_draws, n_subjects) where n_draws = n_post * n_chains
    r_2d = r_all.reshape(-1, n_subjects)
    phi_2d = phi_all.reshape(-1, n_subjects)
    eta_2d = eta_all.reshape(-1, n_subjects)
    n_draws = r_2d.shape[0]
    print(f"  Computing CEs for {n_draws} draws x {n_subjects} subjects...")

    # --- Compute CEs ---
    ce_all = compute_posterior_predictive_ce(
        jnp.array(r_2d), jnp.array(phi_2d), jnp.array(eta_2d),
        prizes_ref,
    )
    ce_all_np = np.asarray(ce_all)
    ce_flat = ce_all_np.ravel()

    # Filter non-finite
    finite_mask = np.isfinite(ce_flat)
    n_nonfinite = np.sum(~finite_mask)
    if n_nonfinite > 0:
        print(f"  Warning: {n_nonfinite} non-finite CEs "
              f"({n_nonfinite / len(ce_flat) * 100:.1f}%) removed")
    ce_flat = ce_flat[finite_mask]

    # --- Summary statistics ---
    print(f"\n  Posterior Predictive CE Summary ({label}):")
    print(f"    Mean   = {np.mean(ce_flat):.1f}")
    print(f"    Median = {np.median(ce_flat):.1f}")
    print(f"    Std    = {np.std(ce_flat):.1f}")
    print(f"    95% CI = [{np.percentile(ce_flat, 2.5):.1f}, "
          f"{np.percentile(ce_flat, 97.5):.1f}]")
    print(f"    EV     = {np.mean(prizes_ref):.1f} (risk-neutral reference)")

    # --- Save NPZ ---
    npz_path = Path(data_dir) / 'output' / f'posterior_predictive_ce_{label}.npz'
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        ce_samples=ce_all_np,
        prizes=prizes_ref,
        n_draws=n_draws,
        n_subjects=n_subjects,
        mean=np.mean(ce_flat),
        median=np.median(ce_flat),
        std=np.std(ce_flat),
        ci_low=np.percentile(ce_flat, 2.5),
        ci_high=np.percentile(ce_flat, 97.5),
        ev=np.mean(prizes_ref),
    )
    print(f"  Saved NPZ to: {npz_path}")

    # --- Load prior CE for overlay if available ---
    prior_ce_flat = None
    if prior_ce_path is not None:
        prior_ce_path = Path(prior_ce_path)
        if prior_ce_path.exists():
            print(f"  Loading prior predictive CE from: {prior_ce_path}")
            prior_data = np.load(prior_ce_path)
            prior_ce_flat = prior_data['ce_samples'].ravel()
            prior_ce_flat = prior_ce_flat[np.isfinite(prior_ce_flat)]
        else:
            print(f"  Prior CE file not found: {prior_ce_path} (skipping overlay)")

    # --- Plot ---
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f'posterior_predictive_ce_{label}.pdf'
    plot_posterior_predictive_ce(ce_flat, prizes_ref, plot_path, label,
                                 prior_ce_flat=prior_ce_flat)


# =============================================================================
# Population-Level Posterior Predictive CE (from hyperparameter draws)
# =============================================================================

def compute_posterior_predictive_ce_from_hyperparams(
    hyper_samples,
    prizes,
    n_new,
    seed=123,
):
    """
    Compute posterior predictive CE for *new* individuals from hyperparameter draws.

    For each posterior draw of the 8 hyperparameters:
        1. Extract mu_j (4 population means) and sigma_j = exp(logsd_j)
        2. Draw n_new individuals: theta ~ N(mu_j, sigma_j^2)
        3. Transform phi, eta via exp; compute CE

    This integrates over posterior uncertainty in population parameters to give
    the predictive CE distribution for a random new individual.

    Args:
        hyper_samples: Array of shape (n_draws, 8) — posterior draws of
            (mu_r, logsd_r, mu_phi, logsd_phi, mu_eta, logsd_eta,
             mu_lambda, logsd_lambda)
        prizes: Array of 3 prizes [min, mid, max], sorted ascending
        n_new: Number of new individuals to draw per posterior sample
        seed: Random seed

    Returns:
        1D array of CEs (flattened over draws × new individuals)
    """
    hyper_samples = jnp.array(hyper_samples)
    prizes_jax = jnp.array(prizes)
    key = random.PRNGKey(seed)

    def single_draw(carry):
        """Draw n_new individuals from one posterior hyperparameter sample."""
        hyper, rng = carry

        # Extract population means and SDs
        mu_r, logsd_r = hyper[0], hyper[1]
        mu_phi, logsd_phi = hyper[2], hyper[3]
        mu_eta, logsd_eta = hyper[4], hyper[5]
        # mu_lambda, logsd_lambda = hyper[6], hyper[7]  # not needed for CE

        sd_r = jnp.exp(logsd_r)
        sd_phi = jnp.exp(logsd_phi)
        sd_eta = jnp.exp(logsd_eta)

        # Draw individual parameters
        k1, k2, k3 = random.split(rng, 3)
        r_vals = mu_r + sd_r * random.normal(k1, shape=(n_new,))
        phi_raw = mu_phi + sd_phi * random.normal(k2, shape=(n_new,))
        eta_raw = mu_eta + sd_eta * random.normal(k3, shape=(n_new,))

        phi_vals = jnp.exp(phi_raw)
        eta_vals = jnp.exp(eta_raw)

        def single_ce(r, phi, eta):
            rdu_val = _rdu_3outcome(r, phi, eta, prizes_jax)
            return _crra_inverse(rdu_val, r)

        return jax.vmap(single_ce)(r_vals, phi_vals, eta_vals)

    # Generate per-draw RNG keys
    keys = random.split(key, hyper_samples.shape[0])

    def draw_with_key(hyper_key_pair):
        hyper, rng = hyper_key_pair
        return single_draw((hyper, rng))

    all_ces = jax.vmap(draw_with_key)((hyper_samples, keys))  # (n_draws, n_new)
    return jnp.ravel(all_ces)


def draw_params_from_hyper_samples(hyper_samples, n_new, seed=123):
    """
    Draw individual-level parameters from posterior hyperparameter samples.

    For each posterior draw of (mu_j, logsd_j), draw n_new individuals on the
    natural scale: r untransformed, phi/eta/lambda via exp.

    Args:
        hyper_samples: Array of shape (n_draws, 8)
        n_new: Number of new individuals per draw
        seed: Random seed

    Returns:
        dict with keys 'r', 'phi', 'eta', 'lambda', each a 1D numpy array
    """
    hyper_samples = jnp.array(hyper_samples)
    key = random.PRNGKey(seed)

    def single_draw(hyper, rng):
        k1, k2, k3, k4 = random.split(rng, 4)
        r = hyper[0] + jnp.exp(hyper[1]) * random.normal(k1, (n_new,))
        phi = jnp.exp(hyper[2] + jnp.exp(hyper[3]) * random.normal(k2, (n_new,)))
        eta = jnp.exp(hyper[4] + jnp.exp(hyper[5]) * random.normal(k3, (n_new,)))
        lam = jnp.exp(hyper[6] + jnp.exp(hyper[7]) * random.normal(k4, (n_new,)))
        return r, phi, eta, lam

    keys = random.split(key, hyper_samples.shape[0])
    r, phi, eta, lam = jax.vmap(single_draw)(hyper_samples, keys)

    return {
        'r': np.asarray(r.ravel()),
        'phi': np.asarray(phi.ravel()),
        'eta': np.asarray(eta.ravel()),
        'lambda': np.asarray(lam.ravel()),
    }


def draw_params_from_hyperprior(hyperprior_params, n_mc_draws, n_new, seed=42):
    """
    Draw individual-level parameters from the hyperprior.

    For each MC draw: sample population params from hyperprior, then draw
    n_new individuals. Returns natural-scale parameters.

    Args:
        hyperprior_params: Dict with keys mu_0, tau_0_sq, alpha_0, beta_0
        n_mc_draws: Number of MC draws from hyperprior
        n_new: Number of individuals per draw
        seed: Random seed

    Returns:
        dict with keys 'r', 'phi', 'eta', 'lambda', each a 1D numpy array
    """
    mu_0 = jnp.array(hyperprior_params['mu_0'])
    tau_0_sq = jnp.array(hyperprior_params['tau_0_sq'])
    alpha_0 = hyperprior_params['alpha_0']
    beta_0 = jnp.array(hyperprior_params['beta_0'])

    key = random.PRNGKey(seed)

    def single_draw(rng):
        k1, k2, k3 = random.split(rng, 3)

        # Draw population params from hyperprior
        mu_j = mu_0 + jnp.sqrt(tau_0_sq) * random.normal(k1, (4,))
        gamma_draws = jnp.maximum(random.gamma(k2, alpha_0, (4,)), 1e-10)
        sigma_j = jnp.sqrt(beta_0 / gamma_draws)

        # Draw individuals
        z = random.normal(k3, (n_new, 4))
        theta = mu_j + sigma_j * z

        r = theta[:, 0]
        phi = jnp.exp(theta[:, 1])
        eta = jnp.exp(theta[:, 2])
        lam = jnp.exp(theta[:, 3])
        return r, phi, eta, lam

    keys = random.split(key, n_mc_draws)
    r, phi, eta, lam = jax.vmap(single_draw)(keys)

    return {
        'r': np.asarray(r.ravel()),
        'phi': np.asarray(phi.ravel()),
        'eta': np.asarray(eta.ravel()),
        'lambda': np.asarray(lam.ravel()),
    }


def plot_predictive_params_overlay(prior_params, info_params, uninf_params,
                                    output_path, figsize=(6.3, 7.2)):
    """
    Plot 4-panel overlay of predictive parameter distributions.

    Each panel shows three KDE curves (prior, informative posterior,
    uninformative posterior) for one parameter on its natural scale.

    Args:
        prior_params: dict with 'r', 'phi', 'eta', 'lambda' arrays
        info_params: dict (informative posterior)
        uninf_params: dict (uninformative posterior)
        output_path: Path to save PDF
        figsize: Figure dimensions in inches (width, height)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.stats import gaussian_kde

    # Configure LaTeX fonts
    code_dir = Path(__file__).parent
    for fname in ('lmroman12-italic.otf', 'latinmodern-math.otf'):
        fpath = code_dir / fname
        if fpath.exists():
            font_manager.fontManager.addfont(str(fpath))
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
    })

    param_keys = ['r', 'phi', 'eta', 'lambda']
    param_labels = [r'$r$ (CRRA)', r'$\phi$ (Prelec curvature)',
                    r'$\eta$ (Prelec elevation)', r'$\lambda$ (noise)']

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)

    for ax, key, label in zip(axes, param_keys, param_labels):
        prior = prior_params[key]
        info = info_params[key]
        uninf = uninf_params[key]

        # Determine x range from combined data (1st-99th percentile)
        combined = np.concatenate([prior, info, uninf])
        combined = combined[np.isfinite(combined)]
        x_lo, x_hi = np.percentile(combined, [1, 99])

        x_grid = np.linspace(x_lo, x_hi, 500)

        def compute_kde(data):
            d = data[(data >= x_lo) & (data <= x_hi) & np.isfinite(data)]
            if len(d) < 10:
                return np.zeros_like(x_grid)
            return gaussian_kde(d, bw_method='scott')(x_grid)

        kde_prior = compute_kde(prior)
        kde_info = compute_kde(info)
        kde_uninf = compute_kde(uninf)

        ax.plot(x_grid, kde_prior, color='0.5', linewidth=1.5,
                label='Prior predictive')
        ax.fill_between(x_grid, kde_prior, alpha=0.15, color='0.5')

        ax.plot(x_grid, kde_info, color='steelblue', linewidth=1.8,
                label='Posterior (informative)')
        ax.fill_between(x_grid, kde_info, alpha=0.2, color='steelblue')

        ax.plot(x_grid, kde_uninf, color='coral', linewidth=1.8,
                label='Posterior (uninformative)')
        ax.fill_between(x_grid, kde_uninf, alpha=0.2, color='coral')

        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_xlim(x_lo, x_hi)

    # Single legend at top
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {output_path}")


def extract_hyper_posterior(preprocessed_path, max_draws=2000, seed=0):
    """
    Extract posterior draws of the 8 hyperparameters from preprocessed file.

    Uses only the last batch (final converged state) and subsamples to
    max_draws for efficient KDE computation.

    Args:
        preprocessed_path: Path to hyperparams_preprocessed.npz
        max_draws: Maximum number of draws to return (default 2000)
        seed: RNG seed for subsampling

    Returns:
        Array of shape (n_draws, 8) with posterior hyperparameter draws.
        Order: (mu_r, logsd_r, mu_phi, logsd_phi, mu_eta, logsd_eta,
                mu_lambda, logsd_lambda)
    """
    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed hyperparameters not found: {preprocessed_path}\n"
            "Run the preprocess_hyper step first."
        )

    print(f"  Loading preprocessed hyperparameters from {preprocessed_path.name}...")
    d = np.load(preprocessed_path)

    # Use only the last batch (final history file = converged state)
    batch_boundaries = d['batch_boundaries']
    last_start = int(batch_boundaries[-2]) if len(batch_boundaries) > 1 else 0
    n_chains = d['r_mean'].shape[1]

    param_names = ['r_mean', 'r_logsd', 'phi_mean', 'phi_logsd',
                   'eta_mean', 'eta_logsd', 'lambda_mean', 'lambda_logsd']
    hyper = np.stack([d[name][last_start:] for name in param_names], axis=-1)
    hyper_flat = hyper.reshape(-1, 8)
    print(f"  Last batch: {hyper.shape[0]} samples x {n_chains} chains "
          f"= {hyper_flat.shape[0]} draws")

    # Subsample if needed
    if hyper_flat.shape[0] > max_draws:
        rng = np.random.default_rng(seed)
        idx = rng.choice(hyper_flat.shape[0], max_draws, replace=False)
        hyper_flat = hyper_flat[idx]
        print(f"  Subsampled to {max_draws} draws")

    return hyper_flat


def plot_predictive_ce_overlay(prior_ce, post_info_ce, post_uninf_ce, prizes,
                                output_path, figsize=(6.3, 3.5)):
    """
    Plot three overlaid KDE curves of predictive CE distributions.

    Shows how data updates beliefs about the CE distribution for a new
    random individual under both prior specifications.

    Args:
        prior_ce: 1D array of prior predictive CEs
        post_info_ce: 1D array of posterior predictive CEs (informative prior)
        post_uninf_ce: 1D array of posterior predictive CEs (uninformative prior)
        prizes: Array of 3 prizes [min, mid, max]
        output_path: Path to save PDF
        figsize: Figure dimensions in inches (width, height)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.stats import gaussian_kde

    # Configure LaTeX fonts (Latin Modern)
    code_dir = Path(__file__).parent
    for fname in ('lmroman12-italic.otf', 'latinmodern-math.otf'):
        fpath = code_dir / fname
        if fpath.exists():
            font_manager.fontManager.addfont(str(fpath))
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
    })

    prior_ce = np.asarray(prior_ce)
    post_info_ce = np.asarray(post_info_ce)
    post_uninf_ce = np.asarray(post_uninf_ce)
    prizes = np.asarray(prizes)

    ev = np.mean(prizes)

    prize_min = float(prizes[0])
    prize_max = float(prizes[-1])
    x_grid = np.linspace(prize_min, prize_max, 500)

    def compute_kde(data, x):
        """Compute KDE on data clipped to display range."""
        d = data[(data >= prize_min) & (data <= prize_max) & np.isfinite(data)]
        if len(d) < 10:
            return np.zeros_like(x)
        kde = gaussian_kde(d, bw_method='scott')
        return kde(x)

    kde_prior = compute_kde(prior_ce, x_grid)
    kde_info = compute_kde(post_info_ce, x_grid)
    kde_uninf = compute_kde(post_uninf_ce, x_grid)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # KDE curves
    ax.plot(x_grid, kde_prior, color='0.5', linewidth=1.5, label='Prior predictive')
    ax.fill_between(x_grid, kde_prior, alpha=0.15, color='0.5')

    ax.plot(x_grid, kde_info, color='steelblue', linewidth=1.8,
            label='Posterior (informative)')
    ax.fill_between(x_grid, kde_info, alpha=0.2, color='steelblue')

    ax.plot(x_grid, kde_uninf, color='coral', linewidth=1.8,
            label='Posterior (uninformative)')
    ax.fill_between(x_grid, kde_uninf, alpha=0.2, color='coral')

    # Reference lines
    ax.axvline(ev, color='black', linestyle=':', linewidth=1.5)
    for p in prizes:
        ax.axvline(p, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Certainty Equivalent')
    ax.set_ylabel('Density')
    ax.set_xlim(prize_min, prize_max)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {output_path}")


def run_predictive_ce_overlay(
    hyper_preprocessed_info,
    hyper_preprocessed_uninf,
    data_dir,
    direct_prior,
    prior_selection,
    output_dir,
    seed=123,
    config=None,
):
    """
    Run the population-level predictive CE and parameter overlay analysis.

    Loads prior predictive CE from existing NPZ, extracts hyperparameter
    posteriors from preprocessed files, draws new individuals, computes CEs,
    and generates overlay plots for both CE and individual parameters.

    Args:
        hyper_preprocessed_info: Path to preprocessed hyperparams (informative)
        hyper_preprocessed_uninf: Path to preprocessed hyperparams (uninformative)
        data_dir: Path to data directory (to find prior_predictive_ce.npz)
        direct_prior: Direct prior dict (for drawing prior predictive params)
        prior_selection: Prior selection dict with rho, alpha_0
        output_dir: Path for output plots and NPZ
        seed: Random seed
        config: Dict with optional keys:
            max_hyper_draws (int): Max posterior hyper draws (default 2000)
            n_new (int): New individuals per hyper draw (default 50)
            n_mc_prior (int): MC draws from hyperprior (default 2000)
            subsample_seed (int): Seed for subsampling (default 99)
            ce_figsize (tuple): CE overlay figure size (default (6.3, 3.5))
            params_figsize (tuple): Param overlay figure size (default (6.3, 7.2))
    """
    if config is None:
        config = {}
    max_hyper_draws = config.get('max_hyper_draws', 2000)
    n_new = config.get('n_new', 50)
    n_mc_prior = config.get('n_mc_prior', 2000)
    subsample_seed = config.get('subsample_seed', 99)
    ce_figsize = tuple(config.get('ce_figsize', (6.3, 3.5)))
    params_figsize = tuple(config.get('params_figsize', (6.3, 7.2)))

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # --- Load prior predictive CE ---
    prior_ce_path = data_dir / 'output' / 'prior_predictive_ce.npz'
    if not prior_ce_path.exists():
        raise FileNotFoundError(
            f"Prior predictive CE not found: {prior_ce_path}\n"
            "Run do_prior_predictive_ce first."
        )
    print("  Loading prior predictive CE...")
    prior_data = np.load(prior_ce_path)
    prior_ce_flat = prior_data['ce_samples'].ravel()
    prior_ce_flat = prior_ce_flat[np.isfinite(prior_ce_flat)]
    prizes = prior_data['prizes']
    print(f"  Prior CE: {len(prior_ce_flat)} samples, "
          f"median={np.median(prior_ce_flat):.1f}")

    # --- Extract hyperparameter posteriors from preprocessed files ---
    print("\n  Loading informative hyperparameter posterior...")
    hyper_info = extract_hyper_posterior(
        hyper_preprocessed_info, max_draws=max_hyper_draws, seed=subsample_seed)

    print("\n  Loading uninformative hyperparameter posterior...")
    hyper_uninf = extract_hyper_posterior(
        hyper_preprocessed_uninf, max_draws=max_hyper_draws, seed=subsample_seed + 1)

    # --- Draw individual parameters from posterior hyperparameters ---
    print(f"\n  Drawing individual parameters (informative, "
          f"{hyper_info.shape[0]} draws x {n_new} new)...")
    info_params = draw_params_from_hyper_samples(hyper_info, n_new, seed=seed)
    print(f"  Drawing individual parameters (uninformative, "
          f"{hyper_uninf.shape[0]} draws x {n_new} new)...")
    uninf_params = draw_params_from_hyper_samples(hyper_uninf, n_new, seed=seed + 1)

    # --- Draw prior predictive parameters from hyperprior ---
    print("  Drawing prior predictive parameters from hyperprior...")
    from prior_selection import rdu_prelec_prior_selection, derive_hyperpriors
    ps = rdu_prelec_prior_selection(direct_prior)
    hp = derive_hyperpriors(ps, rho=prior_selection['rho'],
                            alpha_0=prior_selection['alpha_0'])
    hyperprior_params = {
        'mu_0': hp.mu_0, 'tau_0_sq': hp.tau_0_sq,
        'alpha_0': hp.alpha_0, 'beta_0': hp.beta_0,
    }
    prior_params = draw_params_from_hyperprior(
        hyperprior_params, n_mc_draws=n_mc_prior, n_new=n_new, seed=seed + 2,
    )

    # --- Compute posterior predictive CEs for new individuals ---
    print(f"\n  Computing posterior predictive CEs (informative)...")
    post_info_ce = np.asarray(
        compute_posterior_predictive_ce_from_hyperparams(
            hyper_info, prizes, n_new, seed=seed,
        )
    )
    post_info_ce = post_info_ce[np.isfinite(post_info_ce)]
    print(f"  Informative posterior CE: {len(post_info_ce)} samples, "
          f"median={np.median(post_info_ce):.1f}")

    print(f"  Computing posterior predictive CEs (uninformative)...")
    post_uninf_ce = np.asarray(
        compute_posterior_predictive_ce_from_hyperparams(
            hyper_uninf, prizes, n_new, seed=seed + 1,
        )
    )
    post_uninf_ce = post_uninf_ce[np.isfinite(post_uninf_ce)]
    print(f"  Uninformative posterior CE: {len(post_uninf_ce)} samples, "
          f"median={np.median(post_uninf_ce):.1f}")

    # --- Save combined NPZ ---
    npz_path = data_dir / 'output' / 'predictive_ce_overlay.npz'
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        prior_ce=prior_ce_flat,
        post_info_ce=post_info_ce,
        post_uninf_ce=post_uninf_ce,
        prizes=prizes,
        n_new=n_new,
    )
    print(f"  Saved NPZ to: {npz_path}")

    # --- Plots ---
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / 'predictive_ce_overlay.pdf'
    plot_predictive_ce_overlay(
        prior_ce_flat, post_info_ce, post_uninf_ce, prizes, plot_path,
        figsize=ce_figsize,
    )

    params_plot_path = output_dir / 'predictive_params_overlay.pdf'
    plot_predictive_params_overlay(
        prior_params, info_params, uninf_params, params_plot_path,
        figsize=params_figsize,
    )
