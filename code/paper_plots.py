"""
Generate paper comparison plots from preprocessed CE-space and NIG model outputs.

Produces:
  1. hyper_comparison.pdf — CE-space hyperparameter posteriors vs hyperprior
  2. predictive_params_overlay.pdf — Hyper-posterior predictive parameter distributions
  3. predictive_ce_overlay.pdf — Hyper-posterior predictive CE distributions (3 panels)
  4. subject_ce_most_affected.pdf — Subject with largest CE shift
  5. subject_ce_least_affected.pdf — Subject with smallest CE shift

Key distinction: the CE overlay and parameter overlay show the hyper-posterior
predictive distribution for a NEW random individual, not the pooled posterior
draws from the observed subjects. For the CE-space model, CEs are drawn
directly from the Beta-Beta hierarchy. For the NIG model, parameters are drawn
from the NIG hierarchy and CEs computed via the forward map.
"""

import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde, beta as beta_dist, invgamma as invgamma_dist
from scipy.special import expit


# Reference lotteries (must match main.py config)
REFERENCE_LOTTERIES = [
    {'prizes': [10, 50], 'probs': [0.6, 0.4], 'label': r'$L_1$'},
    {'prizes': [20, 40], 'probs': [0.5, 0.5], 'label': r'$L_2$'},
    {'prizes': [14, 44], 'probs': [0.6, 0.4], 'label': r'$L_3$'},
]

QUANTITY_NAMES = ['ce1', 'ce2', 'ce3', 'pchoice']


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


# =============================================================================
# CE computation (for NIG model draws)
# =============================================================================

def _compute_ce_2outcome(r, phi, eta, prizes, probs):
    """Compute CE for a 2-outcome lottery with given prizes and probabilities."""
    sorted_idx = np.argsort(prizes)
    sp = np.array(prizes)[sorted_idx]
    sq = np.array(probs)[sorted_idx]

    cum = np.cumsum(sq)
    w = np.zeros_like(cum)
    for k in range(len(cum)):
        if cum[k] >= 1.0:
            w[k] = 1.0
        else:
            w[k] = np.exp(-eta * (-np.log(cum[k]))**phi)

    d = np.zeros_like(w)
    d[0] = w[0]
    for k in range(1, len(w)):
        d[k] = w[k] - w[k-1]

    if np.abs(r - 1.0) < 1e-10:
        rdu_val = np.sum(d * np.log(sp))
        ce = np.exp(rdu_val)
    else:
        rdu_val = np.sum(d * sp**(1 - r) / (1 - r))
        ce = (rdu_val * (1 - r))**(1 / (1 - r))

    return float(np.clip(ce, min(prizes), max(prizes)))


def _compute_ces_for_lottery(r_arr, phi_arr, eta_arr, lottery):
    """Compute CEs for arrays of parameters and a specific lottery."""
    prizes = lottery['prizes']
    probs = lottery['probs']
    n = len(r_arr)
    ces = np.zeros(n)
    for i in range(n):
        try:
            ces[i] = _compute_ce_2outcome(
                r_arr[i], phi_arr[i], eta_arr[i], prizes, probs)
        except (ValueError, FloatingPointError, OverflowError):
            ces[i] = np.nan
    return ces


# =============================================================================
# Hyper-posterior predictive draws
# =============================================================================

def _draw_ce_from_beta_hierarchy(hyper_path, prior_params_path, quantity_idx,
                                  n_draws=50000, seed=42, use_posterior=True):
    """Draw CEs directly from the Beta-Beta hierarchy.

    If use_posterior=True, draws from the hyperparameter posterior.
    If use_posterior=False, draws from the hyperprior.

    Returns CE values on natural scale (dollars).
    """
    rng = np.random.default_rng(seed)
    pp = np.load(prior_params_path)
    qname = QUANTITY_NAMES[quantity_idx]

    lo = float(pp[f'{qname}_bound_lo'])
    hi = float(pp[f'{qname}_bound_hi'])

    if use_posterior:
        # Draw from hyperparameter posterior
        hyper = np.load(hyper_path)
        key = qname if qname != 'pchoice' else 'pc'
        logit_mu_all = hyper[f'{key}_logit_mu']
        logit_psi_all = hyper[f'{key}_logit_psi']
        burn = logit_mu_all.shape[0] // 2
        logit_mu_flat = logit_mu_all[burn:].ravel()
        logit_psi_flat = logit_psi_all[burn:].ravel()

        idx = rng.choice(len(logit_mu_flat), n_draws, replace=True)
        mu_draws = expit(logit_mu_flat[idx])
        psi_draws = expit(logit_psi_flat[idx])
    else:
        # Draw from hyperprior
        a_mu = float(pp[f'{qname}_a_mu'])
        b_mu = float(pp[f'{qname}_b_mu'])
        a_psi = float(pp[f'{qname}_a_psi'])
        b_psi = float(pp[f'{qname}_b_psi'])

        mu_draws = rng.beta(a_mu, b_mu, size=n_draws)
        psi_draws = rng.beta(a_psi, b_psi, size=n_draws)

    # Draw individual z from Beta(mu*(1-psi)/psi, (1-mu)*(1-psi)/psi)
    ratio = (1 - psi_draws) / psi_draws
    a_pop = mu_draws * ratio
    b_pop = (1 - mu_draws) * ratio
    a_pop = np.maximum(a_pop, 1e-6)
    b_pop = np.maximum(b_pop, 1e-6)

    z = rng.beta(a_pop, b_pop)
    ce = lo + z * (hi - lo)
    return ce


def _draw_nig_predictive_ce(hyper_path, lottery, n_draws=50000, seed=42):
    """Draw hyper-posterior predictive CEs from NIG model.

    Draws (mu, sigma) from hyperparameter posterior, then draws a new
    individual's parameters, then computes CE via forward map.
    """
    rng = np.random.default_rng(seed)
    d = np.load(hyper_path)
    burn = d['r_mean'].shape[0] // 2

    param_specs = [
        ('r', False),        # not log-transformed
        ('phi', True),       # exp-transformed
        ('eta', True),       # exp-transformed
    ]

    # Flatten and subsample hyper draws
    n_avail = d['r_mean'][burn:].size
    n_hyper = min(n_draws, n_avail)
    idx = rng.choice(n_avail, n_hyper, replace=n_hyper > n_avail)

    r_all = np.zeros(n_hyper)
    phi_all = np.zeros(n_hyper)
    eta_all = np.zeros(n_hyper)

    for i in range(n_hyper):
        mu_r = d['r_mean'][burn:].ravel()[idx[i]]
        sd_r = np.exp(d['r_logsd'][burn:].ravel()[idx[i]])
        r_all[i] = rng.normal(mu_r, sd_r)

        mu_phi = d['phi_mean'][burn:].ravel()[idx[i]]
        sd_phi = np.exp(d['phi_logsd'][burn:].ravel()[idx[i]])
        phi_all[i] = np.exp(rng.normal(mu_phi, sd_phi))

        mu_eta = d['eta_mean'][burn:].ravel()[idx[i]]
        sd_eta = np.exp(d['eta_logsd'][burn:].ravel()[idx[i]])
        eta_all[i] = np.exp(rng.normal(mu_eta, sd_eta))

    ces = _compute_ces_for_lottery(r_all, phi_all, eta_all, lottery)
    return ces


def _draw_nig_predictive_params(hyper_path, n_draws=50000, seed=42):
    """Draw hyper-posterior predictive parameters from NIG model."""
    rng = np.random.default_rng(seed)
    d = np.load(hyper_path)
    burn = d['r_mean'].shape[0] // 2

    n_avail = d['r_mean'][burn:].size
    n_hyper = min(n_draws, n_avail)
    idx = rng.choice(n_avail, n_hyper, replace=n_hyper > n_avail)

    params = {}
    for pname, is_log in [('r', False), ('phi', True), ('eta', True), ('lambda', True)]:
        mu_all = d[f'{pname}_mean'][burn:].ravel()[idx]
        sd_all = np.exp(d[f'{pname}_logsd'][burn:].ravel()[idx])
        raw = rng.normal(mu_all, sd_all)
        params[pname] = np.exp(raw) if is_log else raw

    return params


def _draw_nig_prior_predictive_params(n_draws=50000, seed=42):
    """Draw prior predictive parameters from the informative NIG hyperprior."""
    from posteriors.rdu_bhm import HYPERPRIOR_SPECS
    rng = np.random.default_rng(seed)
    specs = HYPERPRIOR_SPECS

    params = {}
    for pname, spec_key, is_log in [
        ('r', 'r', False),
        ('phi', 'phi_raw', True),
        ('eta', 'eta_raw', True),
        ('lambda', 'lambda_raw', True),
    ]:
        s = specs[spec_key]
        mu = rng.normal(s['mu_0'], np.sqrt(s['tau_0_sq']), size=n_draws)
        sig2 = invgamma_dist.rvs(s['alpha_0'], scale=s['beta_0'],
                                  size=n_draws, random_state=rng)
        raw = rng.normal(mu, np.sqrt(sig2))
        params[pname] = np.exp(raw) if is_log else raw

    return params


# =============================================================================
# Subject-level parameter loading
# =============================================================================

def _load_subject_params(preprocessed_dir):
    """Load all subject parameter draws, return dict of arrays."""
    preprocessed_dir = Path(preprocessed_dir)
    meta = np.load(preprocessed_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    all_params = {'r': [], 'phi': [], 'eta': [], 'lambda': []}
    for sid in range(n_subjects):
        d = np.load(preprocessed_dir / f'subject_{sid:03d}_params.npz')
        burn = d['r'].shape[0] // 2
        for pname in all_params:
            all_params[pname].append(d[pname][burn:].ravel())

    return {k: np.array(v) for k, v in all_params.items()}, n_subjects


# =============================================================================
# Main entry point
# =============================================================================

def generate_paper_plots(
    info_subjects_dir,
    uninf_subjects_dir,
    info_hyper_path,
    uninf_hyper_path,
    prior_params_path,
    output_dir,
    data_output_dir=None,
    seed=42,
):
    """Generate all paper comparison plots.

    Args:
        info_subjects_dir: Path to CE-space preprocessed subjects
        uninf_subjects_dir: Path to NIG uninf preprocessed subjects
        info_hyper_path: Path to CE-space hyperparams_preprocessed.npz
        uninf_hyper_path: Path to NIG uninf hyperparams_preprocessed.npz
        prior_params_path: Path to prior_params_ce.npz (Beta-Beta hyperprior specs)
        output_dir: Path for output PDFs and NPZ files
        seed: Random seed
    """
    plt = _setup_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_params_path = Path(prior_params_path)

    # NPZ data files go to data_output_dir (for pythontex), plots go to output_dir
    if data_output_dir is None:
        data_output_dir = output_dir
    else:
        data_output_dir = Path(data_output_dir)
        data_output_dir.mkdir(parents=True, exist_ok=True)

    n_ce_draws = 50000

    # =========================================================================
    # 1. CE-space hyperparameter posteriors vs hyperprior
    # =========================================================================
    print("\nGenerating hyper_comparison.pdf...")
    _plot_hyper_comparison(
        info_hyper_path, prior_params_path,
        output_dir / 'hyper_comparison.pdf',
        plt,
    )

    # =========================================================================
    # 2. Hyper-posterior predictive parameter overlay
    # =========================================================================
    print("\nGenerating predictive_params_overlay.pdf...")
    print("  Drawing informative hyper-posterior predictive params (NIG hyperprior)...")
    info_pred_params = _draw_nig_prior_predictive_params(n_draws=n_ce_draws, seed=seed)
    print("  Drawing diffuse hyper-posterior predictive params...")
    uninf_pred_params = _draw_nig_predictive_params(
        uninf_hyper_path, n_draws=n_ce_draws, seed=seed+1)

    _plot_params_overlay(
        info_pred_params, uninf_pred_params,
        output_dir / 'predictive_params_overlay.pdf',
        plt,
    )

    # =========================================================================
    # 3. Hyper-posterior predictive CE overlay (3 panels, one per ref lottery)
    # =========================================================================
    print("\nGenerating predictive_ce_overlay.pdf...")
    ce_data_all = {}

    for k, lottery in enumerate(REFERENCE_LOTTERIES):
        qname = QUANTITY_NAMES[k]
        ev = sum(p * x for p, x in zip(lottery['probs'], lottery['prizes']))
        print(f"  {lottery['label']}: EV=${ev:.0f}")

        # Informative: draw CEs directly from Beta-Beta hierarchy (posterior)
        info_ce = _draw_ce_from_beta_hierarchy(
            info_hyper_path, prior_params_path, k,
            n_draws=n_ce_draws, seed=seed+10+k, use_posterior=True)

        # Hyperprior predictive: draw CEs from Beta-Beta hierarchy (prior)
        prior_ce = _draw_ce_from_beta_hierarchy(
            info_hyper_path, prior_params_path, k,
            n_draws=n_ce_draws, seed=seed+20+k, use_posterior=False)

        # Diffuse: draw from NIG hyper-posterior, compute CEs via forward map
        uninf_ce = _draw_nig_predictive_ce(
            uninf_hyper_path, lottery,
            n_draws=n_ce_draws, seed=seed+30+k)

        ce_data_all[k] = {
            'prior': prior_ce, 'info': info_ce, 'uninf': uninf_ce,
            'lottery': lottery, 'ev': ev,
        }

        print(f"    Prior:  median={np.nanmedian(prior_ce):.1f}, "
              f">{ev:.0f}: {100*np.nanmean(prior_ce > ev):.1f}%")
        print(f"    Info:   median={np.nanmedian(info_ce):.1f}, "
              f">{ev:.0f}: {100*np.nanmean(info_ce > ev):.1f}%")
        print(f"    Uninf:  median={np.nanmedian(uninf_ce):.1f}, "
              f">{ev:.0f}: {100*np.nanmean(uninf_ce > ev):.1f}%")

    _plot_ce_overlay_3panel(
        ce_data_all,
        output_dir / 'predictive_ce_overlay.pdf',
        plt,
    )

    # Save overlay quantities for pythontex (using L1 as reference)
    d0 = ce_data_all[0]
    ev0 = d0['ev']
    overlay_data = {
        'prior_pct_above_ev': 100 * np.nanmean(d0['prior'] > ev0),
    }
    lo0 = min(d0['lottery']['prizes'])
    hi0 = max(d0['lottery']['prizes'])
    for name, ce in [('prior', d0['prior']), ('info', d0['info']), ('uninf', d0['uninf'])]:
        overlay_data[f'{name}_pct_below_{lo0+5:g}'] = 100 * np.nanmean(ce < lo0 + 5)
        overlay_data[f'{name}_pct_above_{hi0-10:g}'] = 100 * np.nanmean(ce > hi0 - 10)
    np.savez(data_output_dir / 'predictive_ce_overlay.npz', **overlay_data)

    # =========================================================================
    # 4 & 5. Subject CE comparisons (most and least affected)
    # =========================================================================
    print("\nComputing per-subject CE divergence...")
    rng = np.random.default_rng(seed + 100)

    info_params, n_subjects = _load_subject_params(info_subjects_dir)
    uninf_params, _ = _load_subject_params(uninf_subjects_dir)

    lottery_0 = REFERENCE_LOTTERIES[0]
    median_diffs = np.zeros(n_subjects)
    subject_ces_info = []
    subject_ces_uninf = []

    for sid in range(n_subjects):
        n_sub = min(5000, len(info_params['r'][sid]))
        idx_i = rng.choice(len(info_params['r'][sid]), n_sub, replace=False)
        ce_i = _compute_ces_for_lottery(
            info_params['r'][sid][idx_i],
            info_params['phi'][sid][idx_i],
            info_params['eta'][sid][idx_i],
            lottery_0,
        )
        n_sub_u = min(n_sub, len(uninf_params['r'][sid]))
        idx_u = rng.choice(len(uninf_params['r'][sid]), n_sub_u, replace=False)
        ce_u = _compute_ces_for_lottery(
            uninf_params['r'][sid][idx_u],
            uninf_params['phi'][sid][idx_u],
            uninf_params['eta'][sid][idx_u],
            lottery_0,
        )
        subject_ces_info.append(ce_i)
        subject_ces_uninf.append(ce_u)
        median_diffs[sid] = abs(np.nanmedian(ce_i) - np.nanmedian(ce_u))
        if (sid + 1) % 20 == 0:
            print(f"  Processed {sid + 1}/{n_subjects} subjects")

    most_affected = int(np.argmax(median_diffs))
    least_affected = int(np.argmin(median_diffs))
    print(f"  Most affected: subject {most_affected} "
          f"(median CE diff = ${median_diffs[most_affected]:.2f})")
    print(f"  Least affected: subject {least_affected} "
          f"(median CE diff = ${median_diffs[least_affected]:.2f})")

    np.savez(data_output_dir / 'posterior_ce_comparison.npz',
             most_affected_sid=most_affected,
             least_affected_sid=least_affected,
             median_diffs=median_diffs)

    lo0 = min(lottery_0['prizes'])
    hi0 = max(lottery_0['prizes'])
    ev0 = sum(p * x for p, x in zip(lottery_0['probs'], lottery_0['prizes']))

    print("Generating subject_ce_most_affected.pdf...")
    _plot_subject_ce(
        subject_ces_info[most_affected],
        subject_ces_uninf[most_affected],
        lo0, hi0, ev0, most_affected,
        output_dir / 'subject_ce_most_affected.pdf',
        plt,
    )

    print("Generating subject_ce_least_affected.pdf...")
    _plot_subject_ce(
        subject_ces_info[least_affected],
        subject_ces_uninf[least_affected],
        lo0, hi0, ev0, least_affected,
        output_dir / 'subject_ce_least_affected.pdf',
        plt,
    )

    print(f"\nAll paper plots saved to {output_dir}")


# =============================================================================
# Plot functions
# =============================================================================

def _plot_hyper_comparison(info_hyper_path, prior_params_path, output_path, plt):
    """Plot CE-space hyperparameter posteriors vs hyperprior.

    Shows mu_k and psi_k for each quantity, with the Beta hyperprior overlaid.
    """
    info = np.load(info_hyper_path)
    pp = np.load(prior_params_path)
    burn = info['ce1_logit_mu'].shape[0] // 2

    ce_quantities = [
        ('ce1', r'$\mu_{\mathrm{CE}_1}$', r'$\psi_{\mathrm{CE}_1}$'),
        ('ce2', r'$\mu_{\mathrm{CE}_2}$', r'$\psi_{\mathrm{CE}_2}$'),
        ('ce3', r'$\mu_{\mathrm{CE}_3}$', r'$\psi_{\mathrm{CE}_3}$'),
        ('pc', r'$\mu_{P}$', r'$\psi_{P}$'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(7, 10))

    for i, (key, mu_label, psi_label) in enumerate(ce_quantities):
        qname = QUANTITY_NAMES[i]
        logit_mu = info[f'{key}_logit_mu'][burn:].ravel()
        logit_psi = info[f'{key}_logit_psi'][burn:].ravel()
        mu_post = expit(logit_mu)
        psi_post = expit(logit_psi)

        a_mu = float(pp[f'{qname}_a_mu'])
        b_mu = float(pp[f'{qname}_b_mu'])
        a_psi = float(pp[f'{qname}_a_psi'])
        b_psi = float(pp[f'{qname}_b_psi'])

        # Left: mu posterior vs hyperprior
        ax = axes[i, 0]
        x_mu = np.linspace(max(0.001, mu_post.min() - 0.02),
                           min(0.999, mu_post.max() + 0.02), 300)
        kde_mu = gaussian_kde(mu_post)
        ax.plot(x_mu, kde_mu(x_mu), color='#1f77b4', lw=1.5, label='Posterior')
        ax.plot(x_mu, beta_dist.pdf(x_mu, a_mu, b_mu),
                color='#888888', lw=1.2, ls='--', label='Hyperprior')
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(mu_label, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7)

        # Right: psi posterior vs hyperprior
        ax = axes[i, 1]
        x_psi = np.linspace(max(0.0001, psi_post.min() - 0.002),
                            min(0.999, psi_post.max() + 0.002), 300)
        kde_psi = gaussian_kde(psi_post)
        ax.plot(x_psi, kde_psi(x_psi), color='#1f77b4', lw=1.5, label='Posterior')
        ax.plot(x_psi, beta_dist.pdf(x_psi, a_psi, b_psi),
                color='#888888', lw=1.2, ls='--', label='Hyperprior')
        ax.set_title(psi_label, fontsize=10)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_params_overlay(info_params, uninf_params, output_path, plt):
    """Plot 4-panel hyper-posterior predictive parameter distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    panels = [
        ('r', r'$r$', axes[0, 0]),
        ('phi', r'$\phi$', axes[0, 1]),
        ('eta', r'$\eta$', axes[1, 0]),
        ('lambda', r'$\lambda$', axes[1, 1]),
    ]

    for pname, label, ax in panels:
        info_vals = info_params[pname]
        uninf_vals = uninf_params[pname]

        info_clean = info_vals[np.isfinite(info_vals)]
        uninf_clean = uninf_vals[np.isfinite(uninf_vals)]

        q_lo = min(np.percentile(info_clean, 1), np.percentile(uninf_clean, 1))
        q_hi = max(np.percentile(info_clean, 99), np.percentile(uninf_clean, 99))
        margin = (q_hi - q_lo) * 0.1
        x = np.linspace(q_lo - margin, q_hi + margin, 300)

        info_clip = info_clean[(info_clean >= q_lo - margin) & (info_clean <= q_hi + margin)]
        uninf_clip = uninf_clean[(uninf_clean >= q_lo - margin) & (uninf_clean <= q_hi + margin)]

        if len(info_clip) > 100:
            ax.plot(x, gaussian_kde(info_clip)(x),
                    color='#888888', lw=1.5, label='Hyperprior predictive')
        if len(uninf_clip) > 100:
            ax.plot(x, gaussian_kde(uninf_clip)(x),
                    color='#ff7f0e', lw=1.5, label='Diffuse posterior')

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_ce_overlay_3panel(ce_data_all, output_path, plt):
    """Plot 3-panel CE overlay: one panel per reference lottery."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 8))

    for k in range(3):
        ax = axes[k]
        d = ce_data_all[k]
        lottery = d['lottery']
        ev = d['ev']
        lo = min(lottery['prizes'])
        hi = max(lottery['prizes'])

        for ce, label, color, ls in [
            (d['prior'], 'Hyperprior predictive', '#888888', '--'),
            (d['info'], 'Informative posterior', '#1f77b4', '-'),
            (d['uninf'], 'Diffuse posterior', '#ff7f0e', '-'),
        ]:
            ce_clean = ce[np.isfinite(ce)]
            ce_clip = ce_clean[(ce_clean >= lo) & (ce_clean <= hi)]
            if len(ce_clip) > 100:
                kde = gaussian_kde(ce_clip)
                x = np.linspace(lo, hi, 300)
                ax.plot(x, kde(x), color=color, lw=1.5, ls=ls, label=label)

        ax.axvline(ev, color='black', ls=':', lw=0.8, alpha=0.5)
        p_str = '/'.join(f'{p:.1f}' for p in lottery['probs'])
        ax.set_title(
            f'{lottery["label"]}: '
            f'\\${lo}/\\${hi}, $p = ({p_str})$, '
            f'EV = \\${ev:.0f}',
            fontsize=9,
        )
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7)

    axes[-1].set_xlabel('Certainty Equivalent (\\$)', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_subject_ce(ce_info, ce_uninf, lo, hi, ev, sid, output_path, plt):
    """Plot subject-level CE comparison."""
    fig, ax = plt.subplots(figsize=(5, 3))

    for ce, label, color in [
        (ce_info, 'Informative', '#1f77b4'),
        (ce_uninf, 'Diffuse', '#ff7f0e'),
    ]:
        ce_clean = ce[np.isfinite(ce)]
        ce_clip = ce_clean[(ce_clean >= lo) & (ce_clean <= hi)]
        if len(ce_clip) > 50:
            kde = gaussian_kde(ce_clip)
            x = np.linspace(lo, hi, 300)
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
        prior_params_path='data/output/prior_params_ce.npz',
        output_dir='plots/paper',
    )
