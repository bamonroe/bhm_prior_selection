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
    {'prizes': [10, 50], 'probs': [0.75, 0.25], 'label': r'$L_1$'},
    {'prizes': [20, 40], 'probs': [0.5, 0.5], 'label': r'$L_2$'},
    {'prizes': [12, 44], 'probs': [0.75, 0.25], 'label': r'$L_3$'},
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

def _compute_ces_for_lottery(r_arr, phi_arr, eta_arr, lottery):
    """Compute CEs for arrays of parameters and a specific lottery (vectorized)."""
    prizes = np.array(lottery['prizes'], dtype=np.float64)
    probs = np.array(lottery['probs'], dtype=np.float64)

    sorted_idx = np.argsort(prizes)
    sp = prizes[sorted_idx]  # (n_outcomes,)
    sq = probs[sorted_idx]

    lo, hi = float(sp[0]), float(sp[-1])

    # Cumulative probabilities: shape (n_outcomes,)
    cum = np.cumsum(sq)

    # Prelec weights: w(p) = exp(-eta * (-ln p)^phi)
    # For the last outcome, cum=1.0 so w=1.0
    # Shape: (n_draws, n_outcomes)
    r_arr = np.asarray(r_arr, dtype=np.float64)
    phi_arr = np.asarray(phi_arr, dtype=np.float64)
    eta_arr = np.asarray(eta_arr, dtype=np.float64)
    n = len(r_arr)

    neg_ln_cum = -np.log(np.clip(cum[:-1], 1e-30, None))  # (n_outcomes-1,)
    # phi_arr[:, None] broadcasts with neg_ln_cum[None, :]
    powered = neg_ln_cum[None, :] ** phi_arr[:, None]  # (n, n_outcomes-1)
    w_inner = np.exp(-eta_arr[:, None] * powered)       # (n, n_outcomes-1)
    w = np.ones((n, len(cum)), dtype=np.float64)
    w[:, :-1] = w_inner

    # Decision weights
    d = np.empty_like(w)
    d[:, 0] = w[:, 0]
    d[:, 1:] = w[:, 1:] - w[:, :-1]

    # Normalized CRRA utility: u(x) = (x^s - xmin^s) / (xmax^s - xmin^s)
    # where s = 1-r, xmin = lo, xmax = hi
    # At s=0: u(x) = ln(x/xmin) / ln(xmax/xmin)
    x_min, x_max = lo, hi
    s_arr = 1.0 - r_arr
    use_log = np.abs(s_arr) < 1e-10
    s_safe = np.where(use_log, 0.5, s_arr)

    # Normalized utilities: shape (n, n_outcomes)
    sp_s = sp[None, :] ** s_safe[:, None]
    xmin_s = x_min ** s_safe
    xmax_s = x_max ** s_safe
    denom = (xmax_s - xmin_s)[:, None]
    utilities = (sp_s - xmin_s[:, None]) / denom

    log_utilities = np.log(sp / x_min)[None, :] / np.log(x_max / x_min)
    log_utilities = log_utilities * np.ones((n, 1))

    # RDU value (normalized)
    rdu = np.sum(d * np.where(use_log[:, None], log_utilities, utilities), axis=1)

    # CE = (rdu * (xmax^s - xmin^s) + xmin^s)^(1/s), or xmin*(xmax/xmin)^rdu
    with np.errstate(invalid='ignore', over='ignore'):
        ce_s = rdu * (xmax_s - xmin_s) + xmin_s
        ce_normal = np.power(np.maximum(ce_s, 1e-30), 1.0 / s_safe)
        ce_log = x_min * np.power(x_max / x_min, rdu)

    ces = np.where(use_log, ce_log, ce_normal)
    ces = np.clip(ces, lo, hi)
    ces[~np.isfinite(ces)] = np.nan
    return ces


def _compute_ces_for_lottery_ep(alpha_arr, r_arr, gamma_arr, lottery):
    """Compute CEs for arrays of EP+TK parameters and a specific lottery (vectorized).

    Uses numpy for batch computation. The EP inverse is computed via
    Newton iteration (same approach as the JAX version in common_fns).
    """
    prizes = np.array(lottery['prizes'], dtype=np.float64)
    probs = np.array(lottery['probs'], dtype=np.float64)

    sorted_idx = np.argsort(prizes)
    sp = prizes[sorted_idx]
    sq = probs[sorted_idx]

    lo, hi = float(sp[0]), float(sp[-1])

    cum = np.cumsum(sq)

    alpha_arr = np.asarray(alpha_arr, dtype=np.float64)
    r_arr = np.asarray(r_arr, dtype=np.float64)
    gamma_arr = np.asarray(gamma_arr, dtype=np.float64)
    n = len(r_arr)

    # TK weights: w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)
    gamma_safe = np.maximum(gamma_arr, 0.28)
    cum_clipped = np.clip(cum[:-1], 1e-10, 1 - 1e-10)  # (n_outcomes-1,)
    p_g = cum_clipped[None, :] ** gamma_safe[:, None]    # (n, n_outcomes-1)
    q_g = (1 - cum_clipped[None, :]) ** gamma_safe[:, None]
    w_inner = p_g / np.power(p_g + q_g, 1.0 / gamma_safe[:, None])
    w = np.ones((n, len(cum)), dtype=np.float64)
    w[:, :-1] = w_inner

    # Decision weights
    d = np.empty_like(w)
    d[:, 0] = w[:, 0]
    d[:, 1:] = w[:, 1:] - w[:, :-1]

    # EP utility: u(x) = (exp(-alpha*xmin^s) - exp(-alpha*x^s))
    #                   / (exp(-alpha*xmin^s) - exp(-alpha*xmax^s))
    # where s = 1-r. At s=0: reduces to log form.
    x_min, x_max = lo, hi
    s_arr = 1.0 - r_arr
    use_log = np.abs(s_arr) < 1e-10
    s_safe = np.where(use_log, 0.5, s_arr)
    alpha_safe = np.maximum(alpha_arr, 1e-10)

    sp_s = sp[None, :] ** s_safe[:, None]              # (n, n_outcomes)
    xmin_s = x_min ** s_safe                            # (n,)
    xmax_s = x_max ** s_safe                            # (n,)

    exp_x = np.exp(-alpha_safe[:, None] * sp_s)         # (n, n_outcomes)
    exp_min = np.exp(-alpha_safe * xmin_s)              # (n,)
    exp_max = np.exp(-alpha_safe * xmax_s)              # (n,)
    ep_denom = exp_min - exp_max                        # (n,)
    utilities = (exp_min[:, None] - exp_x) / ep_denom[:, None]

    log_utilities = np.log(sp / x_min)[None, :] / np.log(x_max / x_min)
    log_utilities = log_utilities * np.ones((n, 1))

    # RDU value (normalized)
    rdu = np.sum(d * np.where(use_log[:, None], log_utilities, utilities), axis=1)
    rdu = np.clip(rdu, 0.0, 1.0)

    # EP inverse via Newton iteration (numpy batch version)
    # Initial guess: CRRA inverse
    with np.errstate(invalid='ignore', over='ignore'):
        ce_s_crra = rdu * (xmax_s - xmin_s) + xmin_s
        x0_normal = np.power(np.maximum(ce_s_crra, 1e-30), 1.0 / s_safe)
        x0_log = x_min * np.power(x_max / x_min, rdu)
    x = np.where(use_log, x0_log, x0_normal)
    x = np.clip(x, lo + 1e-8, hi - 1e-8)

    # Newton steps
    for _ in range(20):
        x = np.clip(x, lo + 1e-8, hi - 1e-8)
        x_s = np.power(x, s_safe)
        exp_x_newton = np.exp(-alpha_safe * x_s)
        fx = (exp_min - exp_x_newton) / ep_denom - rdu
        dfx = alpha_safe * s_safe * np.power(x, s_safe - 1) * exp_x_newton / ep_denom
        dfx = np.where(np.abs(dfx) < 1e-30, 1e-30, dfx)
        x_new = x - fx / dfx
        # Only update for non-log cases
        x = np.where(use_log, x, np.clip(x_new, lo + 1e-8, hi - 1e-8))

    ces = np.clip(x, lo, hi)
    ces[~np.isfinite(ces)] = np.nan
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

    mu_r = d['r_mean'][burn:].ravel()[idx]
    sd_r = np.exp(d['r_logsd'][burn:].ravel()[idx])
    r_all = rng.normal(mu_r, sd_r)

    mu_phi = d['phi_mean'][burn:].ravel()[idx]
    sd_phi = np.exp(d['phi_logsd'][burn:].ravel()[idx])
    phi_all = np.exp(rng.normal(mu_phi, sd_phi))

    mu_eta = d['eta_mean'][burn:].ravel()[idx]
    sd_eta = np.exp(d['eta_logsd'][burn:].ravel()[idx])
    eta_all = np.exp(rng.normal(mu_eta, sd_eta))

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
    print("\nComputing per-subject CE divergence (averaged across 3 lotteries)...")
    rng = np.random.default_rng(seed + 100)

    info_params, n_subjects = _load_subject_params(info_subjects_dir)
    uninf_params, _ = _load_subject_params(uninf_subjects_dir)

    mean_median_diffs = np.zeros(n_subjects)

    for sid in range(n_subjects):
        n_sub = min(5000, len(info_params['r'][sid]))
        idx_i = rng.choice(len(info_params['r'][sid]), n_sub, replace=False)
        n_sub_u = min(n_sub, len(uninf_params['r'][sid]))
        idx_u = rng.choice(len(uninf_params['r'][sid]), n_sub_u, replace=False)

        diffs_k = []
        for lottery in REFERENCE_LOTTERIES:
            ce_i = _compute_ces_for_lottery(
                info_params['r'][sid][idx_i],
                info_params['phi'][sid][idx_i],
                info_params['eta'][sid][idx_i],
                lottery,
            )
            ce_u = _compute_ces_for_lottery(
                uninf_params['r'][sid][idx_u],
                uninf_params['phi'][sid][idx_u],
                uninf_params['eta'][sid][idx_u],
                lottery,
            )
            diffs_k.append(abs(np.nanmedian(ce_i) - np.nanmedian(ce_u)))

        mean_median_diffs[sid] = np.mean(diffs_k)
        if (sid + 1) % 20 == 0:
            print(f"  Processed {sid + 1}/{n_subjects} subjects")

    most_affected = int(np.argmax(mean_median_diffs))
    least_affected = int(np.argmin(mean_median_diffs))
    print(f"  Most affected: subject {most_affected} "
          f"(mean median CE diff = ${mean_median_diffs[most_affected]:.2f})")
    print(f"  Least affected: subject {least_affected} "
          f"(mean median CE diff = ${mean_median_diffs[least_affected]:.2f})")

    # Group counts using same thresholds as CDF difference plot
    n_well = int(np.sum(mean_median_diffs < 2))
    n_moderate = int(np.sum((mean_median_diffs >= 2) & (mean_median_diffs < 10)))
    n_dominated = int(np.sum(mean_median_diffs >= 10))

    np.savez(data_output_dir / 'posterior_ce_comparison.npz',
             most_affected_sid=most_affected,
             least_affected_sid=least_affected,
             median_diffs=mean_median_diffs,
             n_well=n_well,
             n_moderate=n_moderate,
             n_dominated=n_dominated)

    # Compute prior predictive CEs for each lottery (for subject-level plots)
    print("  Drawing hyperprior predictive CEs for subject plots...")
    prior_ces_by_lottery = []
    for k in range(len(REFERENCE_LOTTERIES)):
        prior_ce_k = _draw_ce_from_beta_hierarchy(
            info_hyper_path, prior_params_path, k,
            n_draws=20000, seed=seed + 400 + k, use_posterior=False)
        prior_ces_by_lottery.append(prior_ce_k)

    # Generate 3-panel plots for most and least affected
    info_dir = Path(info_subjects_dir)
    uninf_dir = Path(uninf_subjects_dir)

    for label, sid in [('most', most_affected), ('least', least_affected)]:
        print(f"Generating subject_ce_{label}_affected.pdf...")
        args = (
            sid,
            str(info_dir / f'subject_{sid:03d}_params.npz'),
            str(uninf_dir / f'subject_{sid:03d}_params.npz'),
            REFERENCE_LOTTERIES,
            str(output_dir / f'subject_ce_{label}_affected.pdf'),
            5000,
            seed + 200,
            False,  # no suptitle for paper figures
            prior_ces_by_lottery,
        )
        _plot_one_subject_ce_all_lotteries(args)
        print(f"  Saved: {output_dir / f'subject_ce_{label}_affected.pdf'}")

    # =========================================================================
    # 6. Subject-level parameter posteriors (most and least affected)
    # =========================================================================
    print("\nGenerating subject parameter comparison plots...")
    info_sub_dir = Path(info_subjects_dir)
    uninf_sub_dir = Path(uninf_subjects_dir)
    for label, sid in [('most', most_affected), ('least', least_affected)]:
        _plot_subject_params(
            sid=sid,
            info_path=str(info_sub_dir / f'subject_{sid:03d}_params.npz'),
            uninf_path=str(uninf_sub_dir / f'subject_{sid:03d}_params.npz'),
            output_path=str(output_dir / f'subject_params_{label}_affected.pdf'),
            plt=plt,
        )
        print(f"  Saved: {output_dir / f'subject_params_{label}_affected.pdf'}")

    # =========================================================================
    # 7. CDF difference plot (all subjects, all lotteries)
    # =========================================================================
    print("\nGenerating cdf_difference_by_subject.pdf...")
    cdf_counts = generate_cdf_difference_plot(
        info_subjects_dir=info_subjects_dir,
        uninf_subjects_dir=uninf_subjects_dir,
        output_path=output_dir / 'cdf_difference_by_subject.pdf',
        lotteries=REFERENCE_LOTTERIES,
        n_sub=3000,
        seed=seed + 300,
        save_dir=output_dir / 'cdf_diff_by_subject',
    )
    print(f"  Groups: {cdf_counts['n_well_identified']} well-identified, "
          f"{cdf_counts['n_moderate']} moderate, "
          f"{cdf_counts['n_prior_dominated']} prior-dominated")

    print(f"\nAll paper plots saved to {output_dir}")


# =============================================================================
# Plot functions
# =============================================================================

def _filter_stuck_chains(draws_2d):
    """Drop chains with near-zero variance (stuck chains) before pooling.

    Args:
        draws_2d: Array of shape (n_samples, n_chains)

    Returns:
        1D array of draws from non-stuck chains
    """
    chain_vars = np.var(draws_2d, axis=0)
    good = chain_vars > 1e-10
    return draws_2d[:, good].ravel()


def _plot_subject_params(sid, info_path, uninf_path, output_path, plt):
    """Plot posterior parameter distributions for a single subject.

    Shows 4 panels (r, phi, eta, lambda) comparing informative vs diffuse
    posterior draws as overlaid KDEs.
    """
    info = np.load(info_path)
    uninf = np.load(uninf_path)
    burn_i = info['r'].shape[0] // 2
    burn_u = uninf['r'].shape[0] // 2

    params = [
        ('r', r'$r$ (CRRA)', False),
        ('phi', r'$\phi$ (Prelec curvature)', True),
        ('eta', r'$\eta$ (Prelec elevation)', True),
        ('lambda', r'$\lambda$ (noise)', True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5))

    for idx, (pname, label, clip_positive) in enumerate(params):
        ax = axes[idx // 2, idx % 2]

        draws_info = _filter_stuck_chains(info[pname][burn_i:])
        draws_uninf = _filter_stuck_chains(uninf[pname][burn_u:])

        # Clip outliers for KDE stability
        for draws in [draws_info, draws_uninf]:
            q01, q99 = np.percentile(draws, [1, 99])
            draws[:] = np.clip(draws, q01, q99)

        lo = min(draws_info.min(), draws_uninf.min())
        hi = max(draws_info.max(), draws_uninf.max())
        if clip_positive:
            lo = max(lo, 1e-4)
        pad = (hi - lo) * 0.1
        x_grid = np.linspace(lo - pad, hi + pad, 300)
        if clip_positive:
            x_grid = x_grid[x_grid > 0]

        try:
            kde_info = gaussian_kde(draws_info)
            kde_uninf = gaussian_kde(draws_uninf)
            ax.plot(x_grid, kde_info(x_grid), color='#1f77b4',
                    linewidth=1.5, label='Informative')
            ax.plot(x_grid, kde_uninf(x_grid), color='#ff7f0e',
                    linewidth=1.5, label='Diffuse')
        except Exception:
            ax.hist(draws_info, bins=50, density=True, alpha=0.5,
                    color='#1f77b4', label='Informative')
            ax.hist(draws_uninf, bins=50, density=True, alpha=0.5,
                    color='#ff7f0e', label='Diffuse')

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)


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

    for row in axes:
        for ax in row:
            ax.legend(fontsize=8)
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
        p_str = '/'.join(f'{p:.2f}' for p in lottery['probs'])
        ax.set_title(
            f'{lottery["label"]}: '
            f'\\${lo}/\\${hi}, $p = ({p_str})$, '
            f'EV = \\${ev:.0f}',
            fontsize=9,
        )
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=8)
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


# =============================================================================
# CDF difference plot (all subjects, all 3 lotteries)
# =============================================================================

def _compute_subject_cdf_diffs(args):
    """Compute CDF differences for a single subject across all lotteries.

    Designed for use with ProcessPoolExecutor or sequential dispatch.
    """
    sid, info_path, uninf_path, lotteries, n_sub, seed = args
    rng = np.random.default_rng(seed + sid)

    info_d = np.load(info_path)
    uninf_d = np.load(uninf_path)
    bi = info_d['r'].shape[0] // 2
    bu = uninf_d['r'].shape[0] // 2

    n_i = min(n_sub, info_d['r'][bi:].size)
    n_u = min(n_sub, uninf_d['r'][bu:].size)
    idx_i = rng.choice(info_d['r'][bi:].size, n_i, replace=False)
    idx_u = rng.choice(uninf_d['r'][bu:].size, n_u, replace=False)

    info_r = info_d['r'][bi:].ravel()[idx_i]
    info_phi = info_d['phi'][bi:].ravel()[idx_i]
    info_eta = info_d['eta'][bi:].ravel()[idx_i]
    uninf_r = uninf_d['r'][bu:].ravel()[idx_u]
    uninf_phi = uninf_d['phi'][bu:].ravel()[idx_u]
    uninf_eta = uninf_d['eta'][bu:].ravel()[idx_u]

    lot_results = []
    lot_diffs = []
    for lot in lotteries:
        lo, hi = min(lot['prizes']), max(lot['prizes'])
        ce_i = _compute_ces_for_lottery(info_r, info_phi, info_eta, lot)
        ce_u = _compute_ces_for_lottery(uninf_r, uninf_phi, uninf_eta, lot)

        x_grid = np.linspace(lo, hi, 200)
        ce_i_clean = np.sort(ce_i[np.isfinite(ce_i)])
        ce_u_clean = np.sort(ce_u[np.isfinite(ce_u)])
        cdf_i = np.searchsorted(ce_i_clean, x_grid) / len(ce_i_clean)
        cdf_u = np.searchsorted(ce_u_clean, x_grid) / len(ce_u_clean)

        lot_results.append((x_grid, cdf_i - cdf_u))
        lot_diffs.append(abs(np.nanmedian(ce_i) - np.nanmedian(ce_u)))

    return sid, lot_results, np.mean(lot_diffs)


def generate_cdf_difference_plot(
    info_subjects_dir,
    uninf_subjects_dir,
    output_path,
    lotteries=None,
    n_sub=3000,
    seed=42,
    n_workers=None,
    save_dir=None,
):
    """Generate 3-panel CDF difference plot: F_info(x) - F_diffuse(x) per subject.

    Each subject is a line, color-coded by how much the prior affects them.
    Computation is parallelized across subjects.

    If save_dir is provided, per-subject CDF difference data is saved as NPZ
    files for later reference.
    """
    from concurrent.futures import ProcessPoolExecutor
    plt = _setup_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if lotteries is None:
        lotteries = REFERENCE_LOTTERIES

    info_subjects_dir = Path(info_subjects_dir)
    uninf_subjects_dir = Path(uninf_subjects_dir)
    meta = np.load(info_subjects_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    # Build work items
    work = []
    for sid in range(n_subjects):
        work.append((
            sid,
            str(info_subjects_dir / f'subject_{sid:03d}_params.npz'),
            str(uninf_subjects_dir / f'subject_{sid:03d}_params.npz'),
            lotteries, n_sub, seed,
        ))

    # Dispatch
    if n_workers and n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_compute_subject_cdf_diffs, work))
    else:
        results = [_compute_subject_cdf_diffs(w) for w in work]

    # Unpack
    avg_abs_diffs = np.zeros(n_subjects)
    all_cdf_diffs = {k: [] for k in range(len(lotteries))}

    for sid, lot_results, avg_diff in results:
        avg_abs_diffs[sid] = avg_diff
        for k, (x_grid, cdf_diff) in enumerate(lot_results):
            all_cdf_diffs[k].append((x_grid, cdf_diff))

    # Save per-subject data if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for sid, lot_results, avg_diff in results:
            save_data = {'avg_abs_diff': avg_diff}
            for k, (x_grid, cdf_diff) in enumerate(lot_results):
                save_data[f'x_grid_{k}'] = x_grid
                save_data[f'cdf_diff_{k}'] = cdf_diff
            np.savez_compressed(
                save_dir / f'subject_{sid:03d}_cdf_diff.npz', **save_data)
        print(f"  Saved CDF diff data for {n_subjects} subjects to {save_dir}")

    # Groups
    well_id = avg_abs_diffs < 2
    moderate = (avg_abs_diffs >= 2) & (avg_abs_diffs < 10)
    prior_dom = avg_abs_diffs >= 10

    groups = [
        (well_id, 'Well-identified', '#2ca02c', 0.3),
        (moderate, 'Moderately affected', '#1f77b4', 0.3),
        (prior_dom, 'Prior-dominated', '#d62728', 0.3),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(5, 7))

    for k, lot in enumerate(lotteries):
        ax = axes[k]
        lo, hi = min(lot['prizes']), max(lot['prizes'])
        ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))

        for mask, label, color, alpha in groups:
            for sid_idx in np.where(mask)[0]:
                x_grid, cdf_diff = all_cdf_diffs[k][sid_idx]
                ax.plot(x_grid, cdf_diff, color=color, alpha=alpha, lw=0.6)

        ax.axhline(0, color='black', lw=0.5, ls='-')
        ax.axvline(ev, color='black', lw=0.5, ls=':')

        lot_label = lot.get('label', f'$L_{{{k+1}}}$')
        ax.set_title(
            f'{lot_label}: \\${lo}/\\${hi}, EV = \\${ev:.0f}',
            fontsize=9,
        )
        ax.set_ylabel(
            r'$F_{\mathrm{info}}(x) - F_{\mathrm{diffuse}}(x)$',
            fontsize=8,
        )
        ax.tick_params(labelsize=8)
        ax.set_xlim(lo, hi)

    # Legend on all panels
    for ax in axes:
        for mask, label, color, _ in groups:
            ax.plot([], [], color=color, lw=1.5, label=f'{label} ({mask.sum()})')
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Certainty Equivalent (\\$)', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return {
        'n_well_identified': int(well_id.sum()),
        'n_moderate': int(moderate.sum()),
        'n_prior_dominated': int(prior_dom.sum()),
    }


# =============================================================================
# Per-subject CE comparison plots (all subjects, all 3 lotteries)
# =============================================================================

def _plot_one_subject_ce_all_lotteries(args):
    """Worker function: generate a 3-panel CE comparison PDF for one subject.

    Called from ProcessPoolExecutor. Receives a single tuple of arguments.
    """
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

    # Parse args tuple (supports 7, 8, or 9 elements)
    sid, info_path, uninf_path, lotteries, output_path, n_sub, seed = args[:7]
    show_title = args[7] if len(args) > 7 else True
    prior_ces = args[8] if len(args) > 8 else None

    rng = np.random.default_rng(seed + sid)

    # Load subject params from both models
    info_d = np.load(info_path)
    uninf_d = np.load(uninf_path)

    # Use second half as burn-in
    burn_i = info_d['r'].shape[0] // 2
    burn_u = uninf_d['r'].shape[0] // 2

    info_r = info_d['r'][burn_i:].ravel()
    info_phi = info_d['phi'][burn_i:].ravel()
    info_eta = info_d['eta'][burn_i:].ravel()

    uninf_r = uninf_d['r'][burn_u:].ravel()
    uninf_phi = uninf_d['phi'][burn_u:].ravel()
    uninf_eta = uninf_d['eta'][burn_u:].ravel()

    # Subsample
    n_i = min(n_sub, len(info_r))
    idx_i = rng.choice(len(info_r), n_i, replace=False)
    n_u = min(n_sub, len(uninf_r))
    idx_u = rng.choice(len(uninf_r), n_u, replace=False)

    fig, axes = plt.subplots(3, 1, figsize=(5, 7))

    for k, lottery in enumerate(lotteries):
        ax = axes[k]
        lo = min(lottery['prizes'])
        hi = max(lottery['prizes'])
        ev = sum(p * x for p, x in zip(lottery['probs'], lottery['prizes']))

        ce_info = _compute_ces_for_lottery(
            info_r[idx_i], info_phi[idx_i], info_eta[idx_i], lottery)
        ce_uninf = _compute_ces_for_lottery(
            uninf_r[idx_u], uninf_phi[idx_u], uninf_eta[idx_u], lottery)

        curves = [
            (ce_info, 'Informative', '#1f77b4', '-'),
            (ce_uninf, 'Diffuse', '#ff7f0e', '-'),
        ]
        if prior_ces is not None:
            curves.insert(0, (prior_ces[k], 'Hyperprior predictive', '#888888', '--'))

        for ce, label, color, ls in curves:
            ce_clean = ce[np.isfinite(ce)]
            ce_clip = ce_clean[(ce_clean >= lo) & (ce_clean <= hi)]
            if len(ce_clip) > 50:
                kde = gaussian_kde(ce_clip)
                x = np.linspace(lo, hi, 300)
                ax.plot(x, kde(x), color=color, lw=1.5, ls=ls, label=label)

        ax.axvline(ev, color='black', ls=':', lw=0.8, alpha=0.5)
        p_str = '/'.join(f'{p:.2f}' for p in lottery['probs'])
        lot_label = lottery.get('label', f'$L_{{{k+1}}}$')
        ax.set_title(
            f'{lot_label}: '
            f'\\${lo}/\\${hi}, $p = ({p_str})$, '
            f'EV = \\${ev:.0f}',
            fontsize=9,
        )
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Certainty Equivalent (\\$)', fontsize=10)
    if show_title:
        fig.suptitle(f'Subject {sid}', fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    return sid


def generate_subject_ce_plots(
    info_subjects_dir,
    uninf_subjects_dir,
    output_dir,
    lotteries=None,
    n_sub=5000,
    n_workers=25,
    seed=42,
):
    """Generate per-subject CE comparison plots for all subjects.

    Each subject gets a single PDF with 3 panels (one per reference lottery),
    showing informative vs diffuse posterior CE distributions.

    Args:
        info_subjects_dir: Path to informative model preprocessed subjects dir
        uninf_subjects_dir: Path to diffuse model preprocessed subjects dir
        output_dir: Output directory for PDFs (subject_NNN.pdf)
        lotteries: List of reference lottery dicts (default: REFERENCE_LOTTERIES)
        n_sub: Number of draws to subsample per subject (default: 5000)
        n_workers: Number of parallel workers (default: 25)
        seed: Random seed
    """
    # Setup matplotlib in main process (validates LaTeX availability)
    _setup_matplotlib()

    info_subjects_dir = Path(info_subjects_dir)
    uninf_subjects_dir = Path(uninf_subjects_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lotteries is None:
        lotteries = REFERENCE_LOTTERIES

    # Load metadata to get subject count
    meta = np.load(info_subjects_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    print(f"  Generating CE comparison plots for {n_subjects} subjects...")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {n_workers}, Subsample: {n_sub}")

    # Build task list
    tasks = []
    for sid in range(n_subjects):
        info_path = str(info_subjects_dir / f'subject_{sid:03d}_params.npz')
        uninf_path = str(uninf_subjects_dir / f'subject_{sid:03d}_params.npz')
        out_path = str(output_dir / f'subject_{sid:03d}.pdf')
        tasks.append((sid, info_path, uninf_path, lotteries, out_path, n_sub, seed))

    from concurrent.futures import ProcessPoolExecutor, as_completed

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_plot_one_subject_ce_all_lotteries, t): t[0]
                   for t in tasks}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == n_subjects:
                    print(f"  Completed {completed}/{n_subjects} subjects")
            except Exception as e:
                print(f"  ERROR on subject {sid}: {e}")

    print(f"  Done: {completed}/{n_subjects} subject PDFs saved to {output_dir}")


def generate_prior_predictive_params_plot(prior_subjects_dir, output_path,
                                          n_subjects_step=10):
    """Generate 2x2 KDE plot of prior predictive parameter distributions.

    Loads preprocessed subject draws from the prior-only model (no data
    likelihood), pools draws across a subsample of subjects, and plots
    kernel density estimates for each structural parameter.

    Args:
        prior_subjects_dir: Path to preprocessed subjects directory
            (e.g. data/output/rdu_bhm_ce_prior/preprocessed/subjects/)
        output_path: Path to save the output PDF
        n_subjects_step: Stride for subsampling subjects (default 10)
    """
    plt = _setup_matplotlib()

    prior_dir = Path(prior_subjects_dir)
    meta = np.load(prior_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    params = ['r', 'phi', 'eta', 'lambda']
    labels = [r'$r$ (CRRA)', r'$\phi$ (Prelec curvature)',
              r'$\eta$ (Prelec elevation)', r'$\lambda$ (noise)']

    # Load all draws once, compute medians, then plot
    all_param_draws = {}
    medians = {}
    for pname in params:
        pdraws = []
        for sid in range(0, n_subjects, n_subjects_step):
            d = np.load(prior_dir / f'subject_{sid:03d}_params.npz')
            burn = d[pname].shape[0] // 2
            pdraws.append(d[pname][burn:].ravel())
        all_param_draws[pname] = np.concatenate(pdraws)
        medians[pname] = float(np.median(all_param_draws[pname]))

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5))
    for idx, (pname, label) in enumerate(zip(params, labels)):
        ax = axes[idx // 2, idx % 2]
        draws = all_param_draws[pname]
        q01, q99 = np.percentile(draws, [1, 99])
        draws_clip = draws[(draws >= q01) & (draws <= q99)]
        x = np.linspace(q01, q99, 300)
        if pname in ('phi', 'eta', 'lambda'):
            x = x[x > 0]
            draws_clip = draws_clip[draws_clip > 0]
        kde = gaussian_kde(draws_clip)
        ax.plot(x, kde(x), color='#1f77b4', lw=1.5)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f'median = {medians[pname]:.2f}', fontsize=8)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output}")

    # Save medians to NPZ for pythontex
    medians_path = output.parent / 'prior_predictive_medians.npz'
    np.savez(medians_path, **{f'pp_median_{k}': v for k, v in medians.items()})
    print(f"  Saved: {medians_path}")


# =============================================================================
# EP+TK Subject Parameter Loading
# =============================================================================

def _load_subject_params_ep(preprocessed_dir):
    """Load all EP+TK subject parameter draws, return dict of arrays.

    EP model stores (alpha, r, gamma, lambda) instead of (r, phi, eta, lambda).
    The param names in the NPZ files match the generated_quantities ordering.
    """
    preprocessed_dir = Path(preprocessed_dir)
    meta = np.load(preprocessed_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    # EP generated_quantities stores: alpha, r, gamma, lambda
    # But the preprocessing extracts them as the 4 param columns, named
    # based on position. Let's check what keys are available.
    d0 = np.load(preprocessed_dir / 'subject_000_params.npz')
    param_keys = [k for k in d0.keys() if k not in ('iterations',)]

    # The EP model has params in order: alpha_raw, r, gamma_raw, lambda_raw
    # generated_quantities returns: alpha, r, gamma, lambda (natural scale)
    # The preprocessing saves them as the same positional keys as CRRA:
    # position 0 -> 'r' (but it's actually alpha for EP)
    # position 1 -> 'phi' (but it's actually r for EP)
    # position 2 -> 'eta' (but it's actually gamma for EP)
    # position 3 -> 'lambda'
    # We need to remap.
    all_params = {'alpha': [], 'r': [], 'gamma': [], 'lambda': []}
    ep_key_map = {'alpha': 'r', 'r': 'phi', 'gamma': 'eta', 'lambda': 'lambda'}

    for sid in range(n_subjects):
        d = np.load(preprocessed_dir / f'subject_{sid:03d}_params.npz')
        burn = d[ep_key_map['alpha']].shape[0] // 2
        for ep_name, file_key in ep_key_map.items():
            all_params[ep_name].append(d[file_key][burn:].ravel())

    return {k: np.array(v) for k, v in all_params.items()}, n_subjects


# =============================================================================
# Cross-Model CE Overlay
# =============================================================================

def generate_cross_model_ce_overlay(
    crra_hyper_path,
    ep_hyper_path,
    uninf_hyper_path,
    prior_params_path,
    ep_subjects_dir,
    output_dir,
    data_output_dir=None,
    seed=42,
):
    """Generate cross-model CE overlay comparing CRRA, EP+TK, and diffuse models.

    For each reference lottery, plots the hyper-posterior predictive CE
    distribution under three specifications:
    1. CRRA+Prelec informative (Beta-Beta hierarchy)
    2. EP+TK informative (Beta-Beta hierarchy, same prior)
    3. Diffuse NIG (Gao 2022)

    The key visual: both informative models share the same quantity-space
    prior, so their hyper-posterior predictive CEs should be comparable
    despite having different structural parameters.
    """
    plt = _setup_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_output_dir is None:
        data_output_dir = output_dir
    else:
        data_output_dir = Path(data_output_dir)

    n_ce_draws = 50000

    print("\nGenerating cross-model CE overlay...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for k, lottery in enumerate(REFERENCE_LOTTERIES):
        ax = axes[k]
        qname = QUANTITY_NAMES[k]
        ev = sum(p * x for p, x in zip(lottery['probs'], lottery['prizes']))
        lo = min(lottery['prizes'])
        hi = max(lottery['prizes'])

        # CRRA informative: draw from Beta-Beta posterior
        crra_ce = _draw_ce_from_beta_hierarchy(
            crra_hyper_path, prior_params_path, k,
            n_draws=n_ce_draws, seed=seed+10+k, use_posterior=True)

        # EP informative: draw from Beta-Beta posterior (same hierarchy!)
        ep_ce = _draw_ce_from_beta_hierarchy(
            ep_hyper_path, prior_params_path, k,
            n_draws=n_ce_draws, seed=seed+40+k, use_posterior=True)

        # Diffuse NIG: draw from NIG hyper-posterior
        uninf_ce = _draw_nig_predictive_ce(
            uninf_hyper_path, lottery,
            n_draws=n_ce_draws, seed=seed+30+k)

        # Hyperprior predictive (shared prior)
        prior_ce = _draw_ce_from_beta_hierarchy(
            crra_hyper_path, prior_params_path, k,
            n_draws=n_ce_draws, seed=seed+20+k, use_posterior=False)

        bins = np.linspace(lo, hi, 80)
        ax.hist(prior_ce, bins=bins, density=True, alpha=0.3, color='gray',
                label='Hyperprior predictive')
        ax.hist(crra_ce, bins=bins, density=True, alpha=0.5, color='C0',
                histtype='step', linewidth=1.5,
                label='CRRA+Prelec posterior')
        ax.hist(ep_ce, bins=bins, density=True, alpha=0.5, color='C2',
                histtype='step', linewidth=1.5, linestyle='--',
                label='EP+TK posterior')
        ax.hist(uninf_ce, bins=bins, density=True, alpha=0.5, color='C1',
                histtype='step', linewidth=1.5, linestyle=':',
                label='Diffuse posterior')
        ax.axvline(ev, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'CE (\\$)')
        ax.set_title(lottery['label'])

        print(f"  {lottery['label']}: EV=${ev:.0f}")
        print(f"    CRRA:   median={np.nanmedian(crra_ce):.1f}")
        print(f"    EP+TK:  median={np.nanmedian(ep_ce):.1f}")
        print(f"    Uninf:  median={np.nanmedian(uninf_ce):.1f}")
        print(f"    Prior:  median={np.nanmedian(prior_ce):.1f}")

    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    outpath = output_dir / 'cross_model_ce_overlay.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


if __name__ == '__main__':
    generate_paper_plots(
        info_subjects_dir='data/output/rdu_bhm_ce/preprocessed/subjects',
        uninf_subjects_dir='data/output/rdu_bhm_uninf/preprocessed/subjects',
        info_hyper_path='data/output/rdu_bhm_ce/preprocessed/hyperparams_preprocessed.npz',
        uninf_hyper_path='data/output/rdu_bhm_uninf/preprocessed/hyperparams_preprocessed.npz',
        prior_params_path='data/output/prior_params_ce.npz',
        output_dir='plots/paper',
    )
