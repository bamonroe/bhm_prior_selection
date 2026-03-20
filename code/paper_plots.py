"""
Generate paper comparison plots from preprocessed CE-space and NIG model outputs.

Produces:
  1. hyper_comparison.pdf — Hyperparameter posteriors (informative vs diffuse vs hyperprior)
  2. predictive_params_overlay.pdf — Unconditional parameter distributions
  3. predictive_ce_overlay.pdf — Posterior predictive CE distributions
  4. subject_ce_most_affected.pdf — Subject with largest CE shift
  5. subject_ce_least_affected.pdf — Subject with smallest CE shift
"""

import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde, norm, invgamma as invgamma_dist
from scipy.special import expit


def _setup_matplotlib():
    """Configure matplotlib for publication-quality plots with Latin Modern fonts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{lmodern}',
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman'],
        'mathtext.fontset': 'cm',
    })
    return plt


def _compute_ce(r, phi, eta, prizes):
    """Compute CE for a 3-outcome equally-likely lottery."""
    probs = np.array([1/3, 1/3, 1/3])
    sorted_idx = np.argsort(prizes)
    sorted_prizes = np.array(prizes)[sorted_idx]
    sorted_probs = probs[sorted_idx]

    # Cumulative probabilities for Prelec PWF
    cum_probs = np.cumsum(sorted_probs)
    # Prelec: w(p) = exp(-eta * (-ln(p))^phi)
    w = np.zeros_like(cum_probs)
    for k in range(len(cum_probs)):
        if cum_probs[k] >= 1.0:
            w[k] = 1.0
        else:
            w[k] = np.exp(-eta * (-np.log(cum_probs[k]))**phi)

    # Decision weights
    d = np.zeros_like(w)
    d[0] = w[0]
    for k in range(1, len(w)):
        d[k] = w[k] - w[k-1]

    # RDU value
    if np.abs(r - 1.0) < 1e-10:
        rdu_val = np.sum(d * np.log(sorted_prizes))
        ce = np.exp(rdu_val)
    else:
        rdu_val = np.sum(d * sorted_prizes**(1 - r) / (1 - r))
        ce = (rdu_val * (1 - r))**(1 / (1 - r))

    return float(np.clip(ce, prizes.min(), prizes.max()))


def _compute_ce_vectorized(r_arr, phi_arr, eta_arr, prizes):
    """Compute CEs for arrays of parameters."""
    n = len(r_arr)
    ces = np.zeros(n)
    for i in range(n):
        try:
            ces[i] = _compute_ce(r_arr[i], phi_arr[i], eta_arr[i], prizes)
        except (ValueError, FloatingPointError, OverflowError):
            ces[i] = np.nan
    return ces


def _load_subject_params(preprocessed_dir, model_type='ce'):
    """Load all subject parameter draws, return dict of arrays."""
    preprocessed_dir = Path(preprocessed_dir)
    meta = np.load(preprocessed_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    all_params = {'r': [], 'phi': [], 'eta': [], 'lambda': []}
    for sid in range(n_subjects):
        d = np.load(preprocessed_dir / f'subject_{sid:03d}_params.npz')
        # Use second half as burn-in
        n_samples = d['r'].shape[0]
        burn = n_samples // 2
        for pname in all_params:
            all_params[pname].append(d[pname][burn:].ravel())

    return {k: np.array(v) for k, v in all_params.items()}, n_subjects


def _draw_nig_population(hyper_preprocessed_path, n_new=50, seed=42):
    """Draw population-level parameter samples from NIG hyperparameter posterior."""
    d = np.load(hyper_preprocessed_path)
    n_samples = d['r_mean'].shape[0]
    burn = n_samples // 2

    rng = np.random.default_rng(seed)
    param_names = ['r', 'phi', 'eta', 'lambda']
    nig_names = ['r', 'phi', 'eta', 'lambda']

    all_params = {p: [] for p in param_names}
    for pname, nig_name in zip(param_names, nig_names):
        mu_draws = d[f'{nig_name}_mean'][burn:].ravel()
        logsd_draws = d[f'{nig_name}_logsd'][burn:].ravel()
        sd_draws = np.exp(logsd_draws)
        # Subsample
        n_draws = min(len(mu_draws), 2000)
        idx = rng.choice(len(mu_draws), n_draws, replace=False)
        for i in idx:
            individuals = rng.normal(mu_draws[i], sd_draws[i], size=n_new)
            all_params[pname].append(individuals)
    # Transform log-scale params
    result = {}
    result['r'] = np.concatenate(all_params['r'])
    result['phi'] = np.exp(np.concatenate(all_params['phi']))
    result['eta'] = np.exp(np.concatenate(all_params['eta']))
    result['lambda'] = np.exp(np.concatenate(all_params['lambda']))
    return result


def _draw_ce_population(hyper_preprocessed_path, prior_params_path, n_new=50, seed=42):
    """Draw population-level parameter samples from CE-space hyperparameter posterior.

    This is indirect: we draw from the Beta-Beta hierarchy to get CEs/P,
    but we need (r, phi, eta, lambda). Since we can't invert the CE mapping,
    we instead use the subject-level parameter draws which ARE on parameter space.
    """
    # For population-level draws from the CE model, we can't easily go
    # CE-space -> parameter-space without the inverse mapping.
    # Instead, we pool all subject-level draws as the posterior predictive.
    # This is handled in the calling code by loading subject params directly.
    raise NotImplementedError("Use subject-level parameter draws instead")


def generate_paper_plots(
    info_subjects_dir,
    uninf_subjects_dir,
    info_hyper_path,
    uninf_hyper_path,
    output_dir,
    prizes=None,
    seed=42,
):
    """Generate all 5 paper comparison plots.

    Args:
        info_subjects_dir: Path to CE-space preprocessed subjects
        uninf_subjects_dir: Path to NIG uninf preprocessed subjects
        info_hyper_path: Path to CE-space hyperparams_preprocessed.npz
        uninf_hyper_path: Path to NIG uninf hyperparams_preprocessed.npz
        output_dir: Path for output PDFs and NPZ files
        prizes: Array of [min, mid, max] prizes (default [10, 30, 50])
        seed: Random seed
    """
    plt = _setup_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prizes is None:
        prizes = np.array([10.0, 30.0, 50.0])

    print("Loading subject-level parameter draws...")
    info_params, n_subjects = _load_subject_params(info_subjects_dir)
    uninf_params, _ = _load_subject_params(uninf_subjects_dir)
    print(f"  Informative: {n_subjects} subjects, "
          f"{len(info_params['r'][0])} draws/subject")
    print(f"  Uninformative: {n_subjects} subjects, "
          f"{len(uninf_params['r'][0])} draws/subject")

    # Pool all subject draws for population-level distributions
    info_r = np.concatenate(info_params['r'])
    info_phi = np.concatenate(info_params['phi'])
    info_eta = np.concatenate(info_params['eta'])
    info_lam = np.concatenate(info_params['lambda'])

    uninf_r = np.concatenate(uninf_params['r'])
    uninf_phi = np.concatenate(uninf_params['phi'])
    uninf_eta = np.concatenate(uninf_params['eta'])
    uninf_lam = np.concatenate(uninf_params['lambda'])

    # =========================================================================
    # 1. Hyperparameter comparison
    # =========================================================================
    print("\nGenerating hyper_comparison.pdf...")
    _plot_hyper_comparison(
        info_hyper_path, uninf_hyper_path,
        output_dir / 'hyper_comparison.pdf',
        plt,
    )

    # =========================================================================
    # 2. Predictive parameter overlay
    # =========================================================================
    print("Generating predictive_params_overlay.pdf...")
    _plot_params_overlay(
        info_r, info_phi, info_eta, info_lam,
        uninf_r, uninf_phi, uninf_eta, uninf_lam,
        output_dir / 'predictive_params_overlay.pdf',
        plt,
    )

    # =========================================================================
    # 3. Predictive CE overlay
    # =========================================================================
    print("Computing posterior predictive CEs...")
    rng = np.random.default_rng(seed)

    # Subsample for CE computation (expensive)
    n_ce_samples = 50000
    idx_info = rng.choice(len(info_r), min(n_ce_samples, len(info_r)), replace=False)
    idx_uninf = rng.choice(len(uninf_r), min(n_ce_samples, len(uninf_r)), replace=False)

    print(f"  Computing {len(idx_info)} informative CEs...")
    info_ce = _compute_ce_vectorized(
        info_r[idx_info], info_phi[idx_info], info_eta[idx_info], prizes)
    print(f"  Computing {len(idx_uninf)} uninformative CEs...")
    uninf_ce = _compute_ce_vectorized(
        uninf_r[idx_uninf], uninf_phi[idx_uninf], uninf_eta[idx_uninf], prizes)

    # Also draw prior predictive CEs from the NIG hyperprior
    print("  Drawing prior predictive from hyperprior...")
    uninf_pop = _draw_nig_population(uninf_hyper_path, n_new=100, seed=seed+10)
    n_prior = min(n_ce_samples, len(uninf_pop['r']))
    idx_prior = rng.choice(len(uninf_pop['r']), n_prior, replace=False)
    # Use informative hyperprior for the "prior" line
    # Actually, we want the prior predictive from the informative hyperprior
    # For now, compute from the CE-space hyperprior by pooling subject draws
    # with no data update — but that's the posterior. We need the prior.
    # Use the CE-space hyperprior to draw population params and compute CEs.
    info_hyper = np.load(info_hyper_path)
    burn = info_hyper['ce1_logit_mu'].shape[0] // 2
    # Draw from the hyperprior (not posterior) — use the prior params
    prior_params_path = Path(info_subjects_dir).parent.parent.parent / 'prior_params_ce.npz'
    prior_ce = _compute_prior_predictive_ce(prior_params_path, prizes, n_ce_samples, seed+20)

    print("Generating predictive_ce_overlay.pdf...")
    _plot_ce_overlay(
        prior_ce, info_ce, uninf_ce, prizes,
        output_dir / 'predictive_ce_overlay.pdf',
        plt,
    )

    # Save overlay quantities for pythontex
    ev = float(np.mean(prizes))
    overlay_data = {
        'prior_pct_above_ev': 100 * np.nanmean(prior_ce > ev),
        'prior_pct_below_15': 100 * np.nanmean(prior_ce < 15),
        'prior_pct_above_40': 100 * np.nanmean(prior_ce > 40),
        'info_pct_below_15': 100 * np.nanmean(info_ce < 15),
        'info_pct_above_40': 100 * np.nanmean(info_ce > 40),
        'uninf_pct_below_15': 100 * np.nanmean(uninf_ce < 15),
        'uninf_pct_above_40': 100 * np.nanmean(uninf_ce > 40),
    }
    np.savez(output_dir / 'predictive_ce_overlay.npz', **overlay_data)

    # =========================================================================
    # 4 & 5. Subject CE comparisons (most and least affected)
    # =========================================================================
    print("\nComputing per-subject CE divergence...")
    median_diffs = np.zeros(n_subjects)
    subject_ces_info = []
    subject_ces_uninf = []

    for sid in range(n_subjects):
        n_sub = min(5000, len(info_params['r'][sid]))
        idx_s = rng.choice(len(info_params['r'][sid]), n_sub, replace=False)
        ce_i = _compute_ce_vectorized(
            info_params['r'][sid][idx_s],
            info_params['phi'][sid][idx_s],
            info_params['eta'][sid][idx_s],
            prizes,
        )
        idx_s2 = rng.choice(len(uninf_params['r'][sid]), min(n_sub, len(uninf_params['r'][sid])), replace=False)
        ce_u = _compute_ce_vectorized(
            uninf_params['r'][sid][idx_s2],
            uninf_params['phi'][sid][idx_s2],
            uninf_params['eta'][sid][idx_s2],
            prizes,
        )
        subject_ces_info.append(ce_i)
        subject_ces_uninf.append(ce_u)
        median_diffs[sid] = abs(np.nanmedian(ce_i) - np.nanmedian(ce_u))
        if (sid + 1) % 20 == 0:
            print(f"  Processed {sid + 1}/{n_subjects} subjects")

    most_affected = int(np.argmax(median_diffs))
    least_affected = int(np.argmin(median_diffs))
    print(f"  Most affected: subject {most_affected} "
          f"(median CE diff = {median_diffs[most_affected]:.2f})")
    print(f"  Least affected: subject {least_affected} "
          f"(median CE diff = {median_diffs[least_affected]:.2f})")

    # Save comparison data for pythontex
    np.savez(output_dir / 'posterior_ce_comparison.npz',
             most_affected_sid=most_affected,
             least_affected_sid=least_affected,
             median_diffs=median_diffs)

    print("Generating subject_ce_most_affected.pdf...")
    _plot_subject_ce(
        subject_ces_info[most_affected],
        subject_ces_uninf[most_affected],
        prizes, most_affected,
        output_dir / 'subject_ce_most_affected.pdf',
        plt,
    )

    print("Generating subject_ce_least_affected.pdf...")
    _plot_subject_ce(
        subject_ces_info[least_affected],
        subject_ces_uninf[least_affected],
        prizes, least_affected,
        output_dir / 'subject_ce_least_affected.pdf',
        plt,
    )

    print(f"\nAll paper plots saved to {output_dir}")


def _compute_prior_predictive_ce(prior_params_path, prizes, n_samples, seed):
    """Draw prior predictive CEs from the informative NIG hyperprior."""
    rng = np.random.default_rng(seed)

    # Use the default informative hyperprior specs from rdu_bhm.py
    from posteriors.rdu_bhm import HYPERPRIOR_SPECS
    specs = HYPERPRIOR_SPECS

    # Draw from the NIG hierarchy
    n_mc = min(n_samples, 2000)
    n_per = max(1, n_samples // n_mc)
    ces = []
    for _ in range(n_mc):
        # Draw hyperparams
        mu_r = rng.normal(specs['r']['mu_0'], np.sqrt(specs['r']['tau_0_sq']))
        sig2_r = invgamma_dist.rvs(specs['r']['alpha_0'], scale=specs['r']['beta_0'], random_state=rng)
        mu_phi = rng.normal(specs['phi_raw']['mu_0'], np.sqrt(specs['phi_raw']['tau_0_sq']))
        sig2_phi = invgamma_dist.rvs(specs['phi_raw']['alpha_0'], scale=specs['phi_raw']['beta_0'], random_state=rng)
        mu_eta = rng.normal(specs['eta_raw']['mu_0'], np.sqrt(specs['eta_raw']['tau_0_sq']))
        sig2_eta = invgamma_dist.rvs(specs['eta_raw']['alpha_0'], scale=specs['eta_raw']['beta_0'], random_state=rng)

        # Draw individuals
        r_vals = rng.normal(mu_r, np.sqrt(sig2_r), size=n_per)
        phi_vals = np.exp(rng.normal(mu_phi, np.sqrt(sig2_phi), size=n_per))
        eta_vals = np.exp(rng.normal(mu_eta, np.sqrt(sig2_eta), size=n_per))

        batch_ce = _compute_ce_vectorized(r_vals, phi_vals, eta_vals, prizes)
        ces.append(batch_ce)

    return np.concatenate(ces)


def _plot_hyper_comparison(info_hyper_path, uninf_hyper_path, output_path, plt):
    """Plot hyperparameter posterior comparison."""
    info = np.load(info_hyper_path)
    uninf = np.load(uninf_hyper_path)

    burn_info = info['ce1_logit_mu'].shape[0] // 2
    burn_uninf = uninf['r_mean'].shape[0] // 2

    # CE-space model: transform logit hyperparams to natural scale
    # mu_k = sigmoid(logit_mu_k), psi_k = sigmoid(logit_psi_k)
    # For comparison, we show the NIG-equivalent: population mean and log-SD
    # The CE model stores different hyperparams, so we show them differently.

    # For a fair comparison, show the implied population mean and SD for each
    # structural parameter. The CE model doesn't directly have these, but
    # the NIG model does. So we show:
    # Row 1-4: mu_r, mu_phi, mu_eta, mu_lambda (NIG has these directly)
    # The CE model's hyperparams are in quantity space, not parameter space.

    # Best approach: show NIG hyperparameters for both models.
    # The uninf model has r_mean, r_logsd, etc. directly.
    # The CE model doesn't have these — its hyperparams are in CE space.
    # We can't directly compare hyperparameters across different parameterizations.

    # Instead, show the population distribution of each parameter.
    # This is what predictive_params_overlay does. For hyper_comparison,
    # show the NIG hyperparameters from the uninformative model alongside
    # the CE-space hyperparameters from the informative model.

    # Actually, let's show the CE-space hyperparams for the informative model
    # (mu and psi for each quantity) as one set of panels, and the NIG
    # hyperparams for the uninformative model as another.

    fig, axes = plt.subplots(4, 2, figsize=(7, 10))

    # Informative model: CE-space hyperparams
    ce_quantities = [
        ('ce1', r'$\mu_{\mathrm{CE}_1}$', r'$\psi_{\mathrm{CE}_1}$'),
        ('ce2', r'$\mu_{\mathrm{CE}_2}$', r'$\psi_{\mathrm{CE}_2}$'),
        ('ce3', r'$\mu_{\mathrm{CE}_3}$', r'$\psi_{\mathrm{CE}_3}$'),
        ('pc', r'$\mu_{P}$', r'$\psi_{P}$'),
    ]

    for i, (key, mu_label, psi_label) in enumerate(ce_quantities):
        logit_mu = info[f'{key}_logit_mu'][burn_info:].ravel()
        logit_psi = info[f'{key}_logit_psi'][burn_info:].ravel()
        mu = expit(logit_mu)
        psi = expit(logit_psi)

        # Left: mu posterior
        ax = axes[i, 0]
        kde_mu = gaussian_kde(mu)
        x_mu = np.linspace(max(0, mu.min() - 0.05), min(1, mu.max() + 0.05), 300)
        ax.plot(x_mu, kde_mu(x_mu), color='#1f77b4', lw=1.5, label='Informative')
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(mu_label, fontsize=10)
        ax.tick_params(labelsize=8)

        # Right: psi posterior
        ax = axes[i, 1]
        kde_psi = gaussian_kde(psi)
        x_psi = np.linspace(max(0, psi.min() - 0.002), min(1, psi.max() + 0.002), 300)
        ax.plot(x_psi, kde_psi(x_psi), color='#1f77b4', lw=1.5, label='Informative')
        ax.set_title(psi_label, fontsize=10)
        ax.tick_params(labelsize=8)

    # Add legend to first panel
    axes[0, 0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_params_overlay(info_r, info_phi, info_eta, info_lam,
                         uninf_r, uninf_phi, uninf_eta, uninf_lam,
                         output_path, plt):
    """Plot 4-panel unconditional parameter distribution overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    panels = [
        (info_r, uninf_r, r'$r$', axes[0, 0]),
        (info_phi, uninf_phi, r'$\phi$', axes[0, 1]),
        (info_eta, uninf_eta, r'$\eta$', axes[1, 0]),
        (info_lam, uninf_lam, r'$\lambda$', axes[1, 1]),
    ]

    for info_vals, uninf_vals, label, ax in panels:
        # Clip outliers for KDE
        info_clean = info_vals[np.isfinite(info_vals)]
        uninf_clean = uninf_vals[np.isfinite(uninf_vals)]

        q_lo = min(np.percentile(info_clean, 1), np.percentile(uninf_clean, 1))
        q_hi = max(np.percentile(info_clean, 99), np.percentile(uninf_clean, 99))
        margin = (q_hi - q_lo) * 0.1
        x = np.linspace(q_lo - margin, q_hi + margin, 300)

        info_clip = info_clean[(info_clean >= q_lo - margin) & (info_clean <= q_hi + margin)]
        uninf_clip = uninf_clean[(uninf_clean >= q_lo - margin) & (uninf_clean <= q_hi + margin)]

        if len(info_clip) > 100:
            kde_info = gaussian_kde(info_clip)
            ax.plot(x, kde_info(x), color='#1f77b4', lw=1.5, label='Informative')
        if len(uninf_clip) > 100:
            kde_uninf = gaussian_kde(uninf_clip)
            ax.plot(x, kde_uninf(x), color='#ff7f0e', lw=1.5, label='Diffuse')

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_ce_overlay(prior_ce, info_ce, uninf_ce, prizes, output_path, plt):
    """Plot 3-way CE overlay: prior, informative posterior, uninformative posterior."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ev = float(np.mean(prizes))

    for ce, label, color in [
        (prior_ce, 'Hyperprior predictive', '#888888'),
        (info_ce, 'Informative posterior', '#1f77b4'),
        (uninf_ce, 'Diffuse posterior', '#ff7f0e'),
    ]:
        ce_clean = ce[np.isfinite(ce)]
        ce_clip = ce_clean[(ce_clean >= prizes.min()) & (ce_clean <= prizes.max())]
        if len(ce_clip) > 100:
            kde = gaussian_kde(ce_clip)
            x = np.linspace(prizes.min(), prizes.max(), 300)
            ax.plot(x, kde(x), color=color, lw=1.5, label=label)

    ax.axvline(ev, color='black', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Certainty Equivalent (\\$)', fontsize=10)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_subject_ce(ce_info, ce_uninf, prizes, sid, output_path, plt):
    """Plot subject-level CE comparison."""
    fig, ax = plt.subplots(figsize=(5, 3))

    ev = float(np.mean(prizes))

    for ce, label, color in [
        (ce_info, 'Informative', '#1f77b4'),
        (ce_uninf, 'Diffuse', '#ff7f0e'),
    ]:
        ce_clean = ce[np.isfinite(ce)]
        ce_clip = ce_clean[(ce_clean >= prizes.min()) & (ce_clean <= prizes.max())]
        if len(ce_clip) > 50:
            kde = gaussian_kde(ce_clip)
            x = np.linspace(prizes.min(), prizes.max(), 300)
            ax.plot(x, kde(x), color=color, lw=1.5, label=label)

    ax.axvline(ev, color='black', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Certainty Equivalent (\\$)', fontsize=10)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'Subject {sid}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    generate_paper_plots(
        info_subjects_dir='data/output/rdu_bhm_ce/preprocessed/subjects',
        uninf_subjects_dir='data/output/rdu_bhm_uninf/preprocessed/subjects',
        info_hyper_path='data/output/rdu_bhm_ce/preprocessed/hyperparams_preprocessed.npz',
        uninf_hyper_path='data/output/rdu_bhm_uninf/preprocessed/hyperparams_preprocessed.npz',
        output_dir='plots/paper',
    )
