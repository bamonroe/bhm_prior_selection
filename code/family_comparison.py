"""
Hyperprior family comparison: sensitivity heatmaps, KS distances, and CDF
difference curves for NIG, Beta-Beta, and Gamma-IG hierarchies.

These computations support Appendix A (Hyperprior Families). They are
computationally expensive (parallelized MC) and should be run via
do_family_comparison in main.py, not from gen_paper_data.py.

Output: data/output/family_comparison.npz + plots in plots/paper/
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scipy.integrate import quad as _quad
from scipy.special import gamma as gammafn
from scipy.stats import invgamma, norm, beta as beta_dist, gamma as gamma_dist


# =============================================================================
# NIG compound distribution
# =============================================================================

def _nig_compound_cdf_array(x_grid, tau0_sq, alpha0, beta0):
    """Evaluate CDF of NIG compound distribution at all points in x_grid."""
    result = np.empty(len(x_grid))
    for k, x in enumerate(x_grid):
        def integrand(s, x=x):
            return (norm.cdf(x, scale=np.sqrt(s + tau0_sq))
                    * invgamma.pdf(s, alpha0, scale=beta0))
        result[k], _ = _quad(integrand, 0, np.inf, limit=100)
    return result


def _ks_for_point_nig(args):
    """Compute KS distance for a single NIG (alpha, kappa) pair."""
    alpha, kappa, v_star, x_grid, F_direct = args
    tau0_sq = alpha * v_star
    alpha0 = 2.0 + 1.0 / kappa**2
    beta0 = (1.0 - alpha) * v_star * (alpha0 - 1.0)
    F_comp = _nig_compound_cdf_array(x_grid, tau0_sq, alpha0, beta0)
    diff = F_comp - F_direct
    idx = np.argmax(np.abs(diff))
    return np.max(np.abs(diff)), x_grid[idx], diff[idx]


def _mc_compound_cdf(x_grid, sample_fn, n_mc=200_000, seed=42):
    """Compute compound CDF at x_grid points via Monte Carlo."""
    rng = np.random.default_rng(seed)
    draws = sample_fn(rng, n_mc)
    return np.array([np.mean(draws <= x) for x in x_grid])


def _ks_for_point_bb(args):
    """KS distance for Beta-Beta compound vs direct Beta prior."""
    alpha, kappa, m_star, v_star, x_grid, F_direct, seed = args

    bern = m_star * (1 - m_star)
    S_mu = bern / (alpha * v_star) - 1
    a_mu = m_star * S_mu
    b_mu = (1 - m_star) * S_mu
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

    a_mu = m_star**2 / (alpha * v_star)
    b_mu = m_star / (alpha * v_star)
    alpha0 = 2.0 + 1.0 / kappa**2
    beta0 = (1.0 - alpha) * v_star * (alpha0 - 1.0)

    def sample_fn(rng, n):
        mu = rng.gamma(a_mu, 1.0 / b_mu, size=n)
        sig2 = invgamma.rvs(alpha0, scale=beta0, size=n, random_state=rng)
        shape = mu**2 / sig2
        scale = sig2 / mu
        return rng.gamma(shape, scale)

    F_comp = _mc_compound_cdf(x_grid, sample_fn, n_mc=200_000, seed=seed)
    diff = F_comp - F_direct
    idx = np.argmax(np.abs(diff))
    return np.max(np.abs(diff)), x_grid[idx], diff[idx]


# =============================================================================
# Plotting helpers
# =============================================================================

def _setup_plt():
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


def _plot_sensitivity_heatmaps(alpha_vals, kappa_vals, between_sd, within_sd,
                               within_iqr, output_dir, prefix='nig'):
    plt = _setup_plt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        (between_sd, f'{prefix}_sensitivity_between.pdf'),
        (within_sd, f'{prefix}_sensitivity_within.pdf'),
        (within_iqr, f'{prefix}_sensitivity_iqr.pdf'),
    ]

    for data, filename in panels:
        fig, ax = plt.subplots(figsize=(2.8, 1.9))
        im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis',
                       extent=[kappa_vals[0], kappa_vals[-1],
                               alpha_vals[0], alpha_vals[-1]])
        ax.set_xlabel(r'$\kappa$', fontsize=10)
        ax.set_ylabel(r'$\alpha$', fontsize=10)
        ax.set_xticks([0.2, 0.5, 0.8, 1.1, 1.4, 1.7])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fpath = output_dir / filename
        fig.savefig(fpath, bbox_inches='tight', backend='pdf')
        plt.close(fig)
        print(f"  Saved: {fpath}")


def _plot_ks_heatmap(alpha_vals, kappa_vals, ks_grid, filename, output_dir):
    plt = _setup_plt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    im = ax.imshow(ks_grid, aspect='auto', origin='lower', cmap='viridis',
                   extent=[kappa_vals[0], kappa_vals[-1],
                           alpha_vals[0], alpha_vals[-1]])
    ax.set_xlabel(r'$\kappa$', fontsize=10)
    ax.set_ylabel(r'$\alpha$', fontsize=10)
    ax.set_xticks([0.2, 0.5, 0.8, 1.1, 1.4, 1.7])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fpath = output_dir / filename
    fig.savefig(fpath, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _plot_cdf_diff_curves(x_grid, F_direct, points_and_params, filename,
                          output_dir, sample_fn_factory, xlabel=r'$v$'):
    plt = _setup_plt()
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


# =============================================================================
# Main entry point
# =============================================================================

def run_family_comparison(output_dir, plot_dir, n_workers=8):
    """Run all hyperprior family comparison analyses.

    Computes NIG, Beta-Beta, and Gamma-IG sensitivity heatmaps, KS distance
    grids, and CDF difference curves. Saves results to NPZ and plots to PDF.

    Args:
        output_dir: Path for NPZ output (e.g., data/output/)
        plot_dir: Path for PDF plots (e.g., plots/paper/)
        n_workers: Number of parallel workers for MC computations
    """
    output_dir = Path(output_dir)
    plot_dir = Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {}

    alpha_vals = np.linspace(0.01, 0.99, 50)
    kappa_vals = np.linspace(0.10, 1.70, 50)
    n_alpha, n_kappa = len(alpha_vals), len(kappa_vals)

    cdf_diff_points = [
        (0.20, 0.25),
        (0.10, 0.75),
        (0.05, 1.00),
        (0.05, 1.50),
    ]

    # ==================================================================
    # NIG
    # ==================================================================
    print("Computing NIG sensitivity heatmaps...")
    nig_v_star = 1.0
    nig_between_sd = np.zeros((n_alpha, n_kappa))
    nig_within_sd = np.zeros((n_alpha, n_kappa))
    nig_within_iqr = np.zeros((n_alpha, n_kappa))

    for i, alpha in enumerate(alpha_vals):
        for j, kappa in enumerate(kappa_vals):
            tau0 = np.sqrt(alpha * nig_v_star)
            nig_between_sd[i, j] = tau0
            alpha0 = 2.0 + 1.0 / kappa**2
            beta0 = (1.0 - alpha) * nig_v_star * (alpha0 - 1.0)
            e_sigma = np.sqrt(beta0) * gammafn(alpha0 - 0.5) / gammafn(alpha0)
            nig_within_sd[i, j] = e_sigma
            q25 = invgamma.ppf(0.25, alpha0, scale=beta0)
            q75 = invgamma.ppf(0.75, alpha0, scale=beta0)
            nig_within_iqr[i, j] = np.sqrt(q75) - np.sqrt(q25)

    save_dict['nig_alpha_values'] = alpha_vals
    save_dict['nig_kappa_values'] = kappa_vals
    save_dict['nig_between_sd'] = nig_between_sd
    save_dict['nig_within_sd'] = nig_within_sd
    save_dict['nig_within_iqr'] = nig_within_iqr

    _plot_sensitivity_heatmaps(alpha_vals, kappa_vals,
                               nig_between_sd, nig_within_sd, nig_within_iqr,
                               plot_dir, prefix='nig')

    print("Computing NIG KS distance grid...")
    ks_x_grid = np.linspace(-5.0, 0.0, 200)
    ks_F_direct = norm.cdf(ks_x_grid)
    work = [(alpha, kappa, nig_v_star, ks_x_grid, ks_F_direct)
            for alpha in alpha_vals for kappa in kappa_vals]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        ks_results = list(pool.map(_ks_for_point_nig, work, chunksize=50))

    nig_ks_grid = np.zeros((n_alpha, n_kappa))
    for idx, (ks, xp, dp) in enumerate(ks_results):
        nig_ks_grid[idx // n_kappa, idx % n_kappa] = ks
    save_dict['nig_ks_grid'] = nig_ks_grid
    print(f"  KS range: [{nig_ks_grid.min():.6f}, {nig_ks_grid.max():.6f}]")
    _plot_ks_heatmap(alpha_vals, kappa_vals, nig_ks_grid,
                     'nig_ks_distance.pdf', plot_dir)

    print("Computing NIG CDF difference curves...")
    cdf_x_grid = np.linspace(-4.0, 4.0, 400)
    F_direct_nig = norm.cdf(cdf_x_grid)

    def _nig_sample_factory(alpha, kappa):
        tau0_sq = alpha * nig_v_star
        alpha0 = 2.0 + 1.0 / kappa**2
        beta0 = (1.0 - alpha) * nig_v_star * (alpha0 - 1.0)
        def sample_fn(rng, n):
            sig2 = invgamma.rvs(alpha0, scale=beta0, size=n, random_state=rng)
            return rng.normal(0, np.sqrt(sig2 + tau0_sq))
        return sample_fn

    _plot_cdf_diff_curves(cdf_x_grid, F_direct_nig, cdf_diff_points,
                          'nig_cdf_diff_curves.pdf', plot_dir,
                          _nig_sample_factory, xlabel=r'$v$')

    # ==================================================================
    # Beta-Beta
    # ==================================================================
    bb_m_star = 0.5
    bb_v_star = 0.04

    print("Computing Beta-Beta sensitivity heatmaps (MC)...")
    bb_between_sd = np.zeros((n_alpha, n_kappa))
    bb_within_sd = np.zeros((n_alpha, n_kappa))
    bb_within_iqr = np.zeros((n_alpha, n_kappa))
    n_mc_bb = 50_000
    rng_bb = np.random.default_rng(42)

    for i, alpha in enumerate(alpha_vals):
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

        for j, kappa in enumerate(kappa_vals):
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
    _plot_sensitivity_heatmaps(alpha_vals, kappa_vals,
                               bb_between_sd, bb_within_sd, bb_within_iqr,
                               plot_dir, prefix='bb')

    print("Computing Beta-Beta KS distance grid...")
    bb_x_grid = np.linspace(0.001, 0.999, 200)
    bb_F_direct = beta_dist.cdf(bb_x_grid,
                                bb_m_star * (bb_m_star * (1 - bb_m_star) / bb_v_star - 1),
                                (1 - bb_m_star) * (bb_m_star * (1 - bb_m_star) / bb_v_star - 1))
    bb_work = [(alpha, kappa, bb_m_star, bb_v_star, bb_x_grid, bb_F_direct,
                42 + i * n_kappa + j)
               for i, alpha in enumerate(alpha_vals)
               for j, kappa in enumerate(kappa_vals)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        bb_ks_results = list(pool.map(_ks_for_point_bb, bb_work, chunksize=50))

    bb_ks_grid = np.zeros((n_alpha, n_kappa))
    for idx, (ks, xp, dp) in enumerate(bb_ks_results):
        bb_ks_grid[idx // n_kappa, idx % n_kappa] = ks
    save_dict['bb_ks_grid'] = bb_ks_grid
    print(f"  KS range: [{bb_ks_grid.min():.6f}, {bb_ks_grid.max():.6f}]")
    _plot_ks_heatmap(alpha_vals, kappa_vals, bb_ks_grid,
                     'bb_ks_distance.pdf', plot_dir)

    print("Computing Beta-Beta CDF difference curves...")
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
    # Gamma-IG
    # ==================================================================
    gam_m_star = 1.0
    gam_v_star = 1.0

    print("Computing Gamma-IG KS distance grid...")
    gam_x_grid = np.linspace(0.001, 6.0, 200)
    gam_shape_direct = gam_m_star**2 / gam_v_star
    gam_rate_direct = gam_m_star / gam_v_star
    gam_F_direct = gamma_dist.cdf(gam_x_grid, gam_shape_direct,
                                  scale=1.0 / gam_rate_direct)
    gam_work = [(alpha, kappa, gam_m_star, gam_v_star,
                 gam_x_grid, gam_F_direct, 1000 + i * n_kappa + j)
                for i, alpha in enumerate(alpha_vals)
                for j, kappa in enumerate(kappa_vals)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        gam_ks_results = list(pool.map(_ks_for_point_gamma, gam_work,
                                       chunksize=50))

    gam_ks_grid = np.zeros((n_alpha, n_kappa))
    for idx, (ks, xp, dp) in enumerate(gam_ks_results):
        gam_ks_grid[idx // n_kappa, idx % n_kappa] = ks
    save_dict['gam_ks_grid'] = gam_ks_grid
    print(f"  KS range: [{gam_ks_grid.min():.6f}, {gam_ks_grid.max():.6f}]")
    _plot_ks_heatmap(alpha_vals, kappa_vals, gam_ks_grid,
                     'gamma_ks_distance.pdf', plot_dir)

    print("Computing Gamma-IG CDF difference curves...")
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
            sig2 = invgamma.rvs(alpha0, scale=beta0, size=n, random_state=rng)
            shape = mu**2 / sig2
            scale = sig2 / mu
            return rng.gamma(shape, scale)
        return sample_fn

    _plot_cdf_diff_curves(gam_cdf_x, gam_cdf_F_direct, cdf_diff_points,
                          'gamma_cdf_diff_curves.pdf', plot_dir,
                          _gam_sample_factory, xlabel=r'$v$')

    # Save all results
    npz_path = output_dir / 'family_comparison.npz'
    np.savez(npz_path, **save_dict)
    print(f"\nSaved to: {npz_path}")
