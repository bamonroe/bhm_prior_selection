"""
Generate summary statistics for the paper.

This module creates NPZ files with data summaries that can be loaded
in the paper's LaTeX (via pythontex or similar).
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scipy.integrate import quad as _quad
from scipy.special import gamma as gammafn
from scipy.stats import invgamma, norm

from clean_data import load_clean_data


def _plot_sensitivity_heatmaps(alpha_vals, kappa_vals, between_sd, within_sd,
                               within_iqr, output_path, prefix='nig'):
    """Generate three-panel heatmap of hierarchical sensitivity analysis."""
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (between_sd, f'{prefix}_sensitivity_between.pdf', None),
        (within_sd, f'{prefix}_sensitivity_within.pdf', None),
        (within_iqr, f'{prefix}_sensitivity_iqr.pdf', None),
    ]

    for data, filename, cbar_ticks in panels:
        fig, ax = plt.subplots(figsize=(2.8, 1.9))
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[kappa_vals[0], kappa_vals[-1],
                    alpha_vals[0], alpha_vals[-1]],
        )
        ax.set_xlabel(r'$\kappa$', fontsize=10)
        ax.set_ylabel(r'$\alpha$', fontsize=10)
        ax.set_xticks([0.2, 0.5, 0.8, 1.1, 1.4, 1.7])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=8)
        if cbar_ticks:
            cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks)
        else:
            cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fpath = output_path.parent / filename
        fig.savefig(fpath, bbox_inches='tight', backend='pdf')
        plt.close(fig)
        print(f"  Saved: {fpath}")


def _nig_compound_cdf_array(x_grid, tau0_sq, alpha0, beta0):
    """Evaluate CDF of NIG compound distribution at all points in x_grid.

    For each x, integrates Phi(x / sqrt(s + tau0_sq)) * f_IG(s) over s,
    where v | sigma^2 ~ N(0, sigma^2 + tau0^2).
    """
    result = np.empty(len(x_grid))
    for k, x in enumerate(x_grid):
        def integrand(s, x=x):
            return (norm.cdf(x, scale=np.sqrt(s + tau0_sq))
                    * invgamma.pdf(s, alpha0, scale=beta0))
        result[k], _ = _quad(integrand, 0, np.inf, limit=100)
    return result


def _ks_for_point(args):
    """Compute KS distance for a single (alpha, kappa) pair.

    Must be at module level for ProcessPoolExecutor pickling.
    """
    alpha, kappa, v_star, x_grid, F_direct = args
    tau0_sq = alpha * v_star
    alpha0 = 2.0 + 1.0 / kappa**2
    beta0 = (1.0 - alpha) * v_star * (alpha0 - 1.0)
    F_comp = _nig_compound_cdf_array(x_grid, tau0_sq, alpha0, beta0)
    diff = F_comp - F_direct
    idx = np.argmax(np.abs(diff))
    return np.max(np.abs(diff)), x_grid[idx], diff[idx]


def _mc_compound_cdf(x_grid, sample_fn, n_mc=200_000, seed=42):
    """Compute compound CDF at x_grid points via Monte Carlo.

    Args:
        x_grid: 1D array of evaluation points.
        sample_fn: callable(rng, n) -> 1D array of n compound draws.
        n_mc: number of MC draws.
        seed: random seed.

    Returns:
        1D array of CDF values at x_grid.
    """
    rng = np.random.default_rng(seed)
    draws = sample_fn(rng, n_mc)
    # For each x, CDF = fraction of draws <= x
    return np.array([np.mean(draws <= x) for x in x_grid])


def _ks_for_point_bb(args):
    """KS distance for Beta-Beta compound vs direct Beta prior."""
    alpha, kappa, m_star, v_star, x_grid, F_direct, seed = args
    from scipy.stats import beta as beta_dist

    # Mean hyperprior
    bern = m_star * (1 - m_star)
    S_mu = bern / (alpha * v_star) - 1
    a_mu = m_star * S_mu
    b_mu = (1 - m_star) * S_mu

    # Dispersion hyperprior (corrected kappa)
    c_mu = bern - alpha * v_star
    e_mu = (a_mu * (a_mu + 1) * b_mu * (b_mu + 1)
            / (S_mu * (S_mu + 1) * (S_mu + 2) * (S_mu + 3)))
    d_mu = e_mu - c_mu**2
    psi_star = (1 - alpha) * v_star / c_mu
    target_var = kappa**2 * ((1 - alpha) * v_star)**2
    var_psi = (target_var - d_mu * psi_star**2) / e_mu
    if var_psi <= 0 or psi_star <= 0 or psi_star >= 1:
        return 0.0, 0.0, 0.0
    nu_psi = psi_star * (1 - psi_star) / var_psi - 1
    if nu_psi <= 0:
        return 0.0, 0.0, 0.0
    a_psi = psi_star * nu_psi
    b_psi = (1 - psi_star) * nu_psi

    def sample_fn(rng, n):
        mu = rng.beta(a_mu, b_mu, size=n)
        psi = rng.beta(a_psi, b_psi, size=n)
        ratio = (1 - psi) / psi
        a_z = mu * ratio
        b_z = (1 - mu) * ratio
        return rng.beta(a_z, b_z)

    F_comp = _mc_compound_cdf(x_grid, sample_fn, n_mc=200_000, seed=seed)
    diff = F_comp - F_direct
    idx = np.argmax(np.abs(diff))
    return np.max(np.abs(diff)), x_grid[idx], diff[idx]


def _ks_for_point_gamma(args):
    """KS distance for Gamma-IG compound vs direct Gamma prior."""
    alpha, kappa, m_star, v_star, x_grid, F_direct, seed = args

    # Mean hyperprior: mu ~ Gamma(a_mu, b_mu)
    a_mu = m_star**2 / (alpha * v_star)
    b_mu = m_star / (alpha * v_star)

    # Variance hyperprior: sigma^2 ~ IG(alpha0, beta0)
    alpha0 = 2.0 + 1.0 / kappa**2
    beta0 = (1.0 - alpha) * v_star * (alpha0 - 1.0)

    def sample_fn(rng, n):
        mu = rng.gamma(a_mu, 1.0 / b_mu, size=n)
        sig2 = invgamma.rvs(alpha0, scale=beta0, size=n, random_state=rng)
        # Gamma(shape=mu^2/sig2, rate=mu/sig2) => scale=sig2/mu
        shape = mu**2 / sig2
        scale = sig2 / mu
        return rng.gamma(shape, scale)

    F_comp = _mc_compound_cdf(x_grid, sample_fn, n_mc=200_000, seed=seed)
    diff = F_comp - F_direct
    idx = np.argmax(np.abs(diff))
    return np.max(np.abs(diff)), x_grid[idx], diff[idx]


def _plot_ks_heatmap(alpha_vals, kappa_vals, ks_grid, filename, output_dir,
                     cbar_ticks=None):
    """Generic KS distance heatmap plot."""
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    im = ax.imshow(
        ks_grid, aspect='auto', origin='lower', cmap='viridis',
        extent=[kappa_vals[0], kappa_vals[-1],
                alpha_vals[0], alpha_vals[-1]],
    )
    ax.set_xlabel(r'$\kappa$', fontsize=10)
    ax.set_ylabel(r'$\alpha$', fontsize=10)
    ax.set_xticks([0.2, 0.5, 0.8, 1.1, 1.4, 1.7])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8)
    if cbar_ticks:
        cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks)
    else:
        cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fpath = output_dir / filename
    fig.savefig(fpath, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _plot_cdf_diff_curves(x_grid, F_direct, points_and_params, filename,
                           output_dir, sample_fn_factory, xlabel=r'$v$'):
    """Generic CDF difference curves plot."""
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    for alpha, kappa in points_and_params:
        sample_fn = sample_fn_factory(alpha, kappa)
        F_comp = _mc_compound_cdf(x_grid, sample_fn, n_mc=500_000, seed=42)
        diff = F_comp - F_direct
        label = rf'$\alpha={alpha},\, \kappa={kappa}$'
        ax.plot(x_grid, diff, label=label, linewidth=1.2)
        print(f"  alpha={alpha}, kappa={kappa}: KS={np.max(np.abs(diff)):.4f}")

    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r'$\Delta F(v)$', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fpath = output_dir / filename
    fig.savefig(fpath, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _plot_nig_ks_distance(alpha_vals, kappa_vals, ks_grid, output_path):
    """Generate heatmap of KS distance between NIG compound and direct prior."""
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    im = ax.imshow(
        ks_grid,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[kappa_vals[0], kappa_vals[-1],
                alpha_vals[0], alpha_vals[-1]],
    )
    ax.set_xlabel(r'$\kappa$', fontsize=10)
    ax.set_ylabel(r'$\alpha$', fontsize=10)
    ax.set_xticks([0.2, 0.5, 0.8, 1.1, 1.4, 1.7])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=[0.01, 0.02, 0.03, 0.04])
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fpath = output_path.parent / 'nig_ks_distance.pdf'
    fig.savefig(fpath, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _plot_nig_cdf_diff_curves(v_star, cdf_diff_points, output_path):
    """Plot CDF difference curves for representative (alpha, kappa) points."""
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(-4.0, 4.0, 400)
    F_direct = norm.cdf(x_grid)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    curves = {}
    for alpha, kappa in cdf_diff_points:
        tau0_sq = alpha * v_star
        alpha0 = 2.0 + 1.0 / kappa**2
        beta0 = (1.0 - alpha) * v_star * (alpha0 - 1.0)
        F_comp = _nig_compound_cdf_array(x_grid, tau0_sq, alpha0, beta0)
        diff = F_comp - F_direct
        curves[(alpha, kappa)] = diff
        label = rf'$\alpha={alpha},\, \kappa={kappa}$'
        ax.plot(x_grid, diff, label=label, linewidth=1.2)
        print(f"  alpha={alpha}, kappa={kappa}: KS={np.max(np.abs(diff)):.4f}")

    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel(r'$v$', fontsize=10)
    ax.set_ylabel(r'$\Delta F(v)$', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fpath = output_path.parent / 'nig_cdf_diff_curves.pdf'
    fig.savefig(fpath, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {fpath}")

    return x_grid, curves


def gen_paper_data(output_path=None,
                   ce_direct_prior=None, reference_lotteries=None,
                   reference_pair=None, prior_selection_ce=None,
                   ce_mean_frac=None, ce_p_risk_seeking=None):
    """
    Generate summary statistics about the experimental data.

    Saves to NPZ file that can be loaded in LaTeX via pythontex.

    Args:
        output_path: Path to save NPZ file (default: data/output/paper_data.npz)
        ce_direct_prior: Dict with CE-space direct prior (ce1, ce2, ce3, pchoice)
        reference_lotteries: List of 3 reference lottery dicts (prizes, probs)
        reference_pair: Dict with lottery_A and lottery_B for choice probability
        prior_selection_ce: Dict with CE-space researcher choices (alpha, kappa)
        ce_mean_frac: Mean CE as fraction of distance from lo to EV
        ce_p_risk_seeking: P(risk-seeking) for each quantity
    """
    print("Loading data...")
    clean = load_clean_data()

    # Summary statistics
    n_subjects = int(clean['n_subjects'])
    n_choices_per_subject = int(clean['n_choices'])
    total_choices = n_subjects * n_choices_per_subject

    # Prize statistics (from unified 3-outcome format)
    prizes_L = np.asarray(clean['prizes_L'])
    prizes_R = np.asarray(clean['prizes_R'])
    all_prizes = np.concatenate([prizes_L.ravel(), prizes_R.ravel()])

    # Probability statistics
    probs_L = np.asarray(clean['probs_L'])
    probs_R = np.asarray(clean['probs_R'])
    # Non-zero probabilities for summary
    all_probs = np.concatenate([probs_L.ravel(), probs_R.ravel()])
    nonzero_probs = all_probs[all_probs > 0]

    print(f"\nData Summary:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices_per_subject}")
    print(f"  Total choices: {total_choices}")
    print(f"  Prize range: [{all_prizes.min():.0f}, {all_prizes.max():.0f}]")
    print(f"  Probability range (nonzero): [{nonzero_probs.min():.2f}, {nonzero_probs.max():.2f}]")

    # Default output path
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'paper_data.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build save dict
    save_dict = {
        'n_subjects': n_subjects,
        'n_choices_per_subject': n_choices_per_subject,
        'total_choices': total_choices,
        'prize_min': all_prizes.min(),
        'prize_max': all_prizes.max(),
        'prob_min': nonzero_probs.min(),
        'prob_max': nonzero_probs.max(),
    }

    # Add CE-space direct prior if provided
    # Key mapping: ce1->1, ce2->2, ce3->3 (for pythontex compatibility)
    if ce_direct_prior is not None:
        _ce_key_map = {'ce1': '1', 'ce2': '2', 'ce3': '3'}
        print("\nCE-Space Direct Prior:")
        for qty, spec in ce_direct_prior.items():
            print(f"  {qty}: mu={spec['mu']}, sigma2={spec['sigma2']}, "
                  f"lo={spec['lo']}, hi={spec['hi']}")
            if qty in _ce_key_map:
                suffix = _ce_key_map[qty]
                save_dict[f'ce_mu_{suffix}'] = spec['mu']
                save_dict[f'ce_sigma2_{suffix}'] = spec['sigma2']
                save_dict[f'ce_lo_{suffix}'] = spec['lo']
                save_dict[f'ce_hi_{suffix}'] = spec['hi']
            elif qty == 'pchoice':
                save_dict['pchoice_mu'] = spec['mu']
                save_dict['pchoice_sigma2'] = spec['sigma2']
                save_dict['pchoice_lo'] = spec['lo']
                save_dict['pchoice_hi'] = spec['hi']

    # Add reference lotteries if provided
    if reference_lotteries is not None:
        print("\nReference Lotteries:")
        for i, lot in enumerate(reference_lotteries):
            ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))
            print(f"  L{i+1}: prizes={lot['prizes']}, probs={lot['probs']}, EV={ev:.0f}")
            for j, prize in enumerate(lot['prizes']):
                save_dict[f'ref_lot_{i+1}_prize_{j+1}'] = prize
            for j, prob in enumerate(lot['probs']):
                save_dict[f'ref_lot_{i+1}_prob_{j+1}'] = prob
            save_dict[f'ref_lot_{i+1}_n_outcomes'] = len(lot['prizes'])
            save_dict[f'ref_lot_{i+1}_ev'] = ev

    # Add reference pair if provided
    if reference_pair is not None:
        print("\nReference Pair:")
        for side, key in [('A', 'lottery_A'), ('B', 'lottery_B')]:
            lot = reference_pair[key]
            print(f"  {side}: prizes={lot['prizes']}, probs={lot['probs']}")
            for j, prize in enumerate(lot['prizes']):
                save_dict[f'ref_pair_{side}_prize_{j+1}'] = prize
            for j, prob in enumerate(lot['probs']):
                save_dict[f'ref_pair_{side}_prob_{j+1}'] = prob

    # Add CE-space prior selection choices if provided
    if prior_selection_ce is not None:
        print("\nCE-Space Prior Selection:")
        for key, val in prior_selection_ce.items():
            print(f"  {key}: {val}")
            save_dict[f'ce_ps_{key}'] = val

    # Pre-compute display values for pythontex (avoid inline math in LaTeX)
    if prior_selection_ce is not None:
        alpha = prior_selection_ce['alpha']
        kappa = prior_selection_ce['kappa']
        save_dict['alpha_pct'] = alpha * 100
        save_dict['one_minus_alpha_pct'] = (1 - alpha) * 100
        save_dict['kappa_pct'] = kappa * 100

    # Add CE elicitation inputs and derived quantities
    if ce_mean_frac is not None and ce_p_risk_seeking is not None:
        save_dict['ce_mean_frac'] = ce_mean_frac
        save_dict['ce_p_risk_seeking'] = ce_p_risk_seeking
        save_dict['ce_mean_frac_pct'] = ce_mean_frac * 100
        save_dict['ce_p_risk_seeking_pct'] = ce_p_risk_seeking * 100
        print(f"\nCE Elicitation Inputs:")
        print(f"  mean_frac: {ce_mean_frac}")
        print(f"  p_risk_seeking: {ce_p_risk_seeking}")
        # Save per-lottery derived quantities (EV, mean CE, implied var/sd)
        if ce_direct_prior is not None and reference_lotteries is not None:
            for k, lot in enumerate(reference_lotteries):
                suffix = str(k + 1)
                spec = ce_direct_prior[f'ce{k+1}']
                ev = spec.get('ev', sum(p * x for p, x in zip(lot['probs'], lot['prizes'])))
                sd = np.sqrt(spec['sigma2'])
                save_dict[f'ce_ev_{suffix}'] = ev
                save_dict[f'ce_mean_{suffix}'] = spec['mu']
                save_dict[f'ce_var_{suffix}'] = spec['sigma2']
                save_dict[f'ce_sd_{suffix}'] = sd
                print(f"  L{suffix}: EV={ev:.1f}, mean={spec['mu']:.2f}, "
                      f"var={spec['sigma2']:.2f}, sd={sd:.2f}")
        # Save pchoice derived quantities
        if ce_direct_prior is not None and 'pchoice' in ce_direct_prior:
            pc = ce_direct_prior['pchoice']
            pc_sd = np.sqrt(pc['sigma2'])
            save_dict['pchoice_var'] = pc['sigma2']
            save_dict['pchoice_sd'] = pc_sd
            print(f"  P: mu={pc['mu']:.2f}, var={pc['sigma2']:.4f}, sd={pc_sd:.4f}")

    # MCMC summary stats (from preprocessed hyperparameters if available)
    hyper_ce_path = output_path.parent / 'rdu_bhm_ce' / 'preprocessed' / 'hyperparams_preprocessed.npz'
    if hyper_ce_path.exists():
        hd = np.load(hyper_ce_path)
        n_samples, n_chains = hd['ce1_logit_mu'].shape
        burn = n_samples // 2
        save_dict['n_chains'] = n_chains
        save_dict['n_samples_post_burnin'] = n_samples - burn
        save_dict['n_total_params'] = int(n_subjects) * 4 + 8
        print(f"\nMCMC Summary (from preprocessed):")
        print(f"  Chains: {n_chains}, samples post burn-in: {n_samples - burn}")
        print(f"  Total posterior samples: {(n_samples - burn) * n_chains:,}")

    # NIG sensitivity analysis for Section 3 (analytical, no external data needed)
    nig_alpha_vals = np.linspace(0.01, 0.99, 50)
    nig_kappa_vals = np.linspace(0.10, 1.70, 50)
    nig_v_star = 1.0
    n_alpha, n_kappa = len(nig_alpha_vals), len(nig_kappa_vals)

    nig_between_sd = np.zeros((n_alpha, n_kappa))
    nig_within_sd = np.zeros((n_alpha, n_kappa))
    nig_within_iqr = np.zeros((n_alpha, n_kappa))

    for i, alpha in enumerate(nig_alpha_vals):
        for j, kappa in enumerate(nig_kappa_vals):
            tau0 = np.sqrt(alpha * nig_v_star)
            nig_between_sd[i, j] = tau0

            alpha0 = 2.0 + 1.0 / kappa**2
            beta0 = (1.0 - alpha) * nig_v_star * (alpha0 - 1.0)

            e_sigma = np.sqrt(beta0) * gammafn(alpha0 - 0.5) / gammafn(alpha0)
            nig_within_sd[i, j] = e_sigma

            q25 = invgamma.ppf(0.25, alpha0, scale=beta0)
            q75 = invgamma.ppf(0.75, alpha0, scale=beta0)
            nig_within_iqr[i, j] = np.sqrt(q75) - np.sqrt(q25)

    save_dict['nig_alpha_values'] = nig_alpha_vals
    save_dict['nig_kappa_values'] = nig_kappa_vals
    save_dict['nig_between_sd'] = nig_between_sd
    save_dict['nig_within_sd'] = nig_within_sd
    save_dict['nig_within_iqr'] = nig_within_iqr

    # Generate heatmap figure
    _plot_sensitivity_heatmaps(
        nig_alpha_vals, nig_kappa_vals,
        nig_between_sd, nig_within_sd, nig_within_iqr,
        output_path.parent.parent.parent / 'plots' / 'paper' / 'nig_sensitivity.pdf',
        prefix='nig',
    )

    print("\nNIG Sensitivity (Section 3):")
    print(f"  alpha: [{nig_alpha_vals[0]:.2f}, ..., {nig_alpha_vals[-1]:.2f}] ({n_alpha} values)")
    print(f"  kappa: [{nig_kappa_vals[0]:.2f}, ..., {nig_kappa_vals[-1]:.2f}] ({n_kappa} values)")
    print(f"  v* = {nig_v_star}")

    # NIG KS distance: sup|F_compound - F_direct| over (alpha, kappa) grid
    print("\nComputing NIG KS distance grid (parallelized)...")
    ks_x_grid = np.linspace(-5.0, 0.0, 200)  # negative half by symmetry
    ks_F_direct = norm.cdf(ks_x_grid)

    work = []
    for alpha in nig_alpha_vals:
        for kappa in nig_kappa_vals:
            work.append((alpha, kappa, nig_v_star, ks_x_grid, ks_F_direct))

    with ProcessPoolExecutor(max_workers=8) as pool:
        ks_results = list(pool.map(_ks_for_point, work, chunksize=50))

    nig_ks_grid = np.zeros((n_alpha, n_kappa))
    for idx, (ks, xp, dp) in enumerate(ks_results):
        i = idx // n_kappa
        j = idx % n_kappa
        nig_ks_grid[i, j] = ks

    save_dict['nig_ks_grid'] = nig_ks_grid
    print(f"  KS range: [{nig_ks_grid.min():.6f}, {nig_ks_grid.max():.6f}]")

    plot_dir = output_path.parent.parent.parent / 'plots' / 'paper'
    _plot_nig_ks_distance(
        nig_alpha_vals, nig_kappa_vals, nig_ks_grid, plot_dir / 'dummy',
    )

    # CDF difference curves at representative (alpha, kappa) points
    print("\nComputing NIG CDF difference curves...")
    cdf_diff_points = [
        (0.20, 0.25),
        (0.10, 0.75),
        (0.05, 1.00),
        (0.05, 1.50),
    ]
    cdf_x_grid, cdf_curves = _plot_nig_cdf_diff_curves(
        nig_v_star, cdf_diff_points, plot_dir / 'dummy',
    )
    save_dict['cdf_diff_x_grid'] = cdf_x_grid
    for k, (alpha, kappa) in enumerate(cdf_diff_points):
        save_dict[f'cdf_diff_{k}_alpha'] = alpha
        save_dict[f'cdf_diff_{k}_kappa'] = kappa
        save_dict[f'cdf_diff_{k}_diff'] = cdf_curves[(alpha, kappa)]

    # ==================================================================
    # Beta-Beta sensitivity (Appendix B.2)
    # ==================================================================
    from scipy.stats import beta as beta_dist

    bb_m_star = 0.5
    bb_v_star = 0.04  # canonical Beta on [0,1]
    bb_alpha_vals = nig_alpha_vals
    bb_kappa_vals = nig_kappa_vals

    # Sensitivity heatmaps (between SD, within SD, within IQR via MC)
    print("\nComputing Beta-Beta sensitivity heatmaps (MC)...")
    bb_between_sd = np.zeros((n_alpha, n_kappa))
    bb_within_sd = np.zeros((n_alpha, n_kappa))
    bb_within_iqr = np.zeros((n_alpha, n_kappa))
    n_mc_bb = 50_000
    rng_bb = np.random.default_rng(42)

    for i, alpha in enumerate(bb_alpha_vals):
        bern = bb_m_star * (1 - bb_m_star)
        if alpha * bb_v_star >= bern:
            continue
        S_mu = bern / (alpha * bb_v_star) - 1
        a_mu = bb_m_star * S_mu
        b_mu = (1 - bb_m_star) * S_mu
        c_mu = bern - alpha * bb_v_star
        e_mu = (a_mu * (a_mu + 1) * b_mu * (b_mu + 1)
                / (S_mu * (S_mu + 1) * (S_mu + 2) * (S_mu + 3)))
        d_mu = e_mu - c_mu**2
        psi_star = (1 - alpha) * bb_v_star / c_mu

        for j, kappa in enumerate(bb_kappa_vals):
            target_var = kappa**2 * ((1 - alpha) * bb_v_star)**2
            var_psi = (target_var - d_mu * psi_star**2) / e_mu
            if var_psi <= 0 or psi_star >= 1:
                bb_between_sd[i, j] = np.nan
                bb_within_sd[i, j] = np.nan
                bb_within_iqr[i, j] = np.nan
                continue
            nu_psi = psi_star * (1 - psi_star) / var_psi - 1
            if nu_psi <= 0:
                bb_between_sd[i, j] = np.nan
                bb_within_sd[i, j] = np.nan
                bb_within_iqr[i, j] = np.nan
                continue
            a_psi = psi_star * nu_psi
            b_psi = (1 - psi_star) * nu_psi

            mu = rng_bb.beta(a_mu, b_mu, size=n_mc_bb)
            psi = rng_bb.beta(a_psi, b_psi, size=n_mc_bb)

            bb_between_sd[i, j] = np.std(mu)
            cond_var = mu * (1 - mu) * psi
            draw_sds = np.sqrt(cond_var)
            bb_within_sd[i, j] = np.median(draw_sds)
            q75, q25 = np.percentile(draw_sds, [75, 25])
            bb_within_iqr[i, j] = q75 - q25
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_alpha} alpha values done")

    save_dict['bb_between_sd'] = bb_between_sd
    save_dict['bb_within_sd'] = bb_within_sd
    save_dict['bb_within_iqr'] = bb_within_iqr

    _plot_sensitivity_heatmaps(
        bb_alpha_vals, bb_kappa_vals,
        bb_between_sd, bb_within_sd, bb_within_iqr,
        plot_dir / 'bb_sensitivity.pdf',
        prefix='bb',
    )

    # Beta-Beta KS distance
    print("\nComputing Beta-Beta KS distance grid (parallelized)...")
    bb_x_grid = np.linspace(0.001, 0.999, 200)
    bb_F_direct = beta_dist.cdf(bb_x_grid,
                                 bb_m_star * (bb_m_star * (1 - bb_m_star) / bb_v_star - 1),
                                 (1 - bb_m_star) * (bb_m_star * (1 - bb_m_star) / bb_v_star - 1))

    bb_work = []
    for i, alpha in enumerate(bb_alpha_vals):
        for j, kappa in enumerate(bb_kappa_vals):
            bb_work.append((alpha, kappa, bb_m_star, bb_v_star, bb_x_grid,
                           bb_F_direct, 42 + i * n_kappa + j))

    with ProcessPoolExecutor(max_workers=8) as pool:
        bb_ks_results = list(pool.map(_ks_for_point_bb, bb_work, chunksize=50))

    bb_ks_grid = np.zeros((n_alpha, n_kappa))
    for idx, (ks, xp, dp) in enumerate(bb_ks_results):
        i = idx // n_kappa
        j = idx % n_kappa
        bb_ks_grid[i, j] = ks

    save_dict['bb_ks_grid'] = bb_ks_grid
    print(f"  KS range: [{bb_ks_grid.min():.6f}, {bb_ks_grid.max():.6f}]")

    _plot_ks_heatmap(bb_alpha_vals, bb_kappa_vals, bb_ks_grid,
                     'bb_ks_distance.pdf', plot_dir)

    # Beta-Beta CDF difference curves
    print("\nComputing Beta-Beta CDF difference curves...")
    bb_cdf_x = np.linspace(0.001, 0.999, 400)
    bb_cdf_F_direct = beta_dist.cdf(bb_cdf_x,
                                     bb_m_star * (bb_m_star * (1 - bb_m_star) / bb_v_star - 1),
                                     (1 - bb_m_star) * (bb_m_star * (1 - bb_m_star) / bb_v_star - 1))

    def _bb_sample_factory(alpha, kappa):
        bern = bb_m_star * (1 - bb_m_star)
        S_mu = bern / (alpha * bb_v_star) - 1
        a_mu = bb_m_star * S_mu
        b_mu = (1 - bb_m_star) * S_mu
        c_mu = bern - alpha * bb_v_star
        e_mu = (a_mu * (a_mu + 1) * b_mu * (b_mu + 1)
                / (S_mu * (S_mu + 1) * (S_mu + 2) * (S_mu + 3)))
        d_mu = e_mu - c_mu**2
        psi_star = (1 - alpha) * bb_v_star / c_mu
        target_var = kappa**2 * ((1 - alpha) * bb_v_star)**2
        var_psi = (target_var - d_mu * psi_star**2) / e_mu
        nu_psi = psi_star * (1 - psi_star) / var_psi - 1
        a_psi = psi_star * nu_psi
        b_psi = (1 - psi_star) * nu_psi
        def sample_fn(rng, n):
            mu = rng.beta(a_mu, b_mu, size=n)
            psi = rng.beta(a_psi, b_psi, size=n)
            ratio = (1 - psi) / psi
            return rng.beta(mu * ratio, (1 - mu) * ratio)
        return sample_fn

    _plot_cdf_diff_curves(bb_cdf_x, bb_cdf_F_direct, cdf_diff_points,
                          'bb_cdf_diff_curves.pdf', plot_dir,
                          _bb_sample_factory, xlabel=r'$z$')

    # ==================================================================
    # Gamma-IG sensitivity (Appendix B.3)
    # ==================================================================
    from scipy.stats import gamma as gamma_dist

    gam_m_star = 1.0
    gam_v_star = 1.0
    gam_alpha_vals = nig_alpha_vals
    gam_kappa_vals = nig_kappa_vals

    # Note: sensitivity heatmaps are identical to NIG (same IG on sigma^2,
    # same between-SD formula). We only compute KS distance + CDF diff curves.
    print("\nComputing Gamma-IG KS distance grid (parallelized)...")
    gam_x_grid = np.linspace(0.001, 6.0, 200)
    # Direct prior: Gamma with mean=m_star, var=v_star => shape=1, rate=1
    gam_shape_direct = gam_m_star**2 / gam_v_star
    gam_rate_direct = gam_m_star / gam_v_star
    gam_F_direct = gamma_dist.cdf(gam_x_grid, gam_shape_direct,
                                   scale=1.0 / gam_rate_direct)

    gam_work = []
    for i, alpha in enumerate(gam_alpha_vals):
        for j, kappa in enumerate(gam_kappa_vals):
            gam_work.append((alpha, kappa, gam_m_star, gam_v_star,
                            gam_x_grid, gam_F_direct,
                            1000 + i * n_kappa + j))

    with ProcessPoolExecutor(max_workers=8) as pool:
        gam_ks_results = list(pool.map(_ks_for_point_gamma, gam_work,
                                       chunksize=50))

    gam_ks_grid = np.zeros((n_alpha, n_kappa))
    for idx, (ks, xp, dp) in enumerate(gam_ks_results):
        i = idx // n_kappa
        j = idx % n_kappa
        gam_ks_grid[i, j] = ks

    save_dict['gam_ks_grid'] = gam_ks_grid
    print(f"  KS range: [{gam_ks_grid.min():.6f}, {gam_ks_grid.max():.6f}]")

    _plot_ks_heatmap(gam_alpha_vals, gam_kappa_vals, gam_ks_grid,
                     'gamma_ks_distance.pdf', plot_dir)

    # Gamma-IG CDF difference curves
    print("\nComputing Gamma-IG CDF difference curves...")
    gam_cdf_x = np.linspace(0.001, 6.0, 400)
    gam_cdf_F_direct = gamma_dist.cdf(gam_cdf_x, gam_shape_direct,
                                       scale=1.0 / gam_rate_direct)

    def _gam_sample_factory(alpha, kappa):
        a_mu = gam_m_star**2 / (alpha * gam_v_star)
        b_mu = gam_m_star / (alpha * gam_v_star)
        alpha0 = 2.0 + 1.0 / kappa**2
        beta0 = (1.0 - alpha) * gam_v_star * (alpha0 - 1.0)
        def sample_fn(rng, n):
            mu = rng.gamma(a_mu, 1.0 / b_mu, size=n)
            sig2 = invgamma.rvs(alpha0, scale=beta0, size=n,
                                random_state=rng)
            shape = mu**2 / sig2
            scale = sig2 / mu
            return rng.gamma(shape, scale)
        return sample_fn

    _plot_cdf_diff_curves(gam_cdf_x, gam_cdf_F_direct, cdf_diff_points,
                          'gamma_cdf_diff_curves.pdf', plot_dir,
                          _gam_sample_factory, xlabel=r'$v$')

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    gen_paper_data()
