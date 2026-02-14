"""
Plotting functions for diagnostics and paper figures.

Two-stage approach for memory efficiency:
1. Preprocess: Extract per-subject data from full history, save to individual files
2. Plot: Pool of workers load individual subject files and generate PDFs

Plot styles follow the project template conventions:
- Hyperparameter plots: 2-column (traces left, densities right), 5 params per page
- Subject plots: Multi-page PDF with traces, densities, and utility spaghetti
- Color scheme: steelblue for data, red for median, tab20 for chains
- Output: Multi-page PDFs in organized directory structure
"""

# Limit BLAS/OpenMP threads to avoid oversubscription with multiprocessing
# Must be set BEFORE importing numpy
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from scipy import stats
from pathlib import Path
from multiprocessing import Pool

from bamcmc import combine_batch_histories
from posteriors.rdu_bhm import NPARAMS, N_HYPER

# Configure LaTeX fonts (Latin Modern)
_CODE_DIR = Path(__file__).parent
_FONT_PATH_ROMAN = _CODE_DIR / 'lmroman12-italic.otf'
_FONT_PATH_MATH = _CODE_DIR / 'latinmodern-math.otf'

if _FONT_PATH_ROMAN.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH_ROMAN))
if _FONT_PATH_MATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH_MATH))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm',  # Computer Modern for math
    'axes.unicode_minus': False,
})


# Hyperparameter display info: (state_name, latex_label)
HYPERPARAM_INFO = [
    ('r_mean', r'$\mu_r$'),
    ('r_logsd', r'$\ln\sigma_r$'),
    ('phi_mean', r'$\mu_{\phi}$ (log scale)'),
    ('phi_logsd', r'$\ln\sigma_{\phi}$'),
    ('eta_mean', r'$\mu_{\eta}$ (log scale)'),
    ('eta_logsd', r'$\ln\sigma_{\eta}$'),
    ('lambda_mean', r'$\mu_{\lambda}$ (log scale)'),
    ('lambda_logsd', r'$\ln\sigma_{\lambda}$'),
]

# Subject parameter display info: (name, latex_label)
SUBJECT_PARAM_INFO = [
    ('r', r'$r$ (CRRA)'),
    ('phi', r'$\phi$ (Prelec curvature)'),
    ('eta', r'$\eta$ (Prelec elevation)'),
    ('lambda', r'$\lambda$ (noise)'),
]

# Color schemes
COLOR_DATA = 'steelblue'
COLOR_MEDIAN = 'red'
COLOR_REFERENCE = 'black'
COLOR_PRIOR = 'green'
COLOR_SPAGHETTI = 'steelblue'
RUN_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

# Number of posterior draws for spaghetti plots
N_SPAGHETTI_SAMPLES = 100

# Dataset outcome range (DKK from Danish experiment)
X_MIN = 50.0
X_MAX = 4500.0

# CE plot constants
CE_HIST_BINS = 100


def compute_hyperprior_quantiles(hyperprior_specs):
    """
    Compute median and 95% CI for each hyperparameter's prior distribution.

    Args:
        hyperprior_specs: Dict with keys 'r', 'phi_raw', 'eta_raw', 'lambda_raw',
                         each containing 'mu_0', 'tau_0_sq', 'alpha_0', 'beta_0'

    Returns:
        Dict mapping hyperparameter name to (median, lower_95, upper_95)
    """
    quantiles = {}

    for param_name in ['r', 'phi_raw', 'eta_raw', 'lambda_raw']:
        spec = hyperprior_specs[param_name]
        mu_0 = spec['mu_0']
        tau_0 = np.sqrt(spec['tau_0_sq'])
        alpha_0 = spec['alpha_0']
        beta_0 = spec['beta_0']

        # Mean parameter: mu ~ N(mu_0, tau_0^2)
        mean_median = mu_0
        mean_lower = mu_0 - 1.96 * tau_0
        mean_upper = mu_0 + 1.96 * tau_0

        # Map to state names used in HYPERPARAM_INFO
        if param_name == 'r':
            mean_key = 'r_mean'
            logsd_key = 'r_logsd'
        elif param_name == 'phi_raw':
            mean_key = 'phi_mean'
            logsd_key = 'phi_logsd'
        elif param_name == 'eta_raw':
            mean_key = 'eta_mean'
            logsd_key = 'eta_logsd'
        else:  # lambda_raw
            mean_key = 'lambda_mean'
            logsd_key = 'lambda_logsd'

        quantiles[mean_key] = (mean_median, mean_lower, mean_upper)

        # Logsd parameter: sigma^2 ~ InvGamma(alpha_0, beta_0)
        # logsd = log(sigma) = 0.5 * log(sigma^2)
        # Use scipy to compute quantiles of InvGamma, then transform
        from scipy.stats import invgamma
        # InvGamma in scipy uses shape=alpha, scale=beta
        sigma_sq_median = invgamma.ppf(0.5, alpha_0, scale=beta_0)
        sigma_sq_lower = invgamma.ppf(0.025, alpha_0, scale=beta_0)
        sigma_sq_upper = invgamma.ppf(0.975, alpha_0, scale=beta_0)

        logsd_median = 0.5 * np.log(sigma_sq_median)
        logsd_lower = 0.5 * np.log(sigma_sq_lower)
        logsd_upper = 0.5 * np.log(sigma_sq_upper)

        quantiles[logsd_key] = (logsd_median, logsd_lower, logsd_upper)

    return quantiles


def crra_utility(x, r, x_min=X_MIN, x_max=X_MAX):
    """
    Normalized CRRA utility function.

    v(x) = x^(1-r) / (1-r)  for r ≠ 1
    u(x) = (v(x) - v(x_min)) / (v(x_max) - v(x_min))

    This normalizes utility to [0, 1] over the outcome range.
    """
    if np.abs(r - 1) < 1e-8:
        v = np.log(x)
        v_min = np.log(x_min)
        v_max = np.log(x_max)
    else:
        v = np.power(x, 1 - r) / (1 - r)
        v_min = np.power(x_min, 1 - r) / (1 - r)
        v_max = np.power(x_max, 1 - r) / (1 - r)
    return (v - v_min) / (v_max - v_min)


# =============================================================================
# STAGE 1: PREPROCESSING - Extract and save per-subject data
# =============================================================================

def preprocess_subjects(history_dir, output_dir, n_subjects=None, force=False):
    """
    Load full history and save per-subject parameter files.

    Subject data is preserved between runs unless force=True or history is newer.

    Args:
        history_dir: Directory with history files
        output_dir: Directory to save per-subject files
        n_subjects: Number of subjects (auto-detected if None)
        force: If True, overwrite existing subject data

    Returns:
        tuple: (output_dir, iterations, batch_boundaries) for use in plotting
    """
    history_dir = Path(history_dir)
    output_dir = Path(output_dir)

    # Load history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files in {history_dir}")

    # Check if subject data already exists and is up-to-date
    existing_files = list(output_dir.glob('subject_*_params.npz')) if output_dir.exists() else []

    if existing_files and not force:
        newest_history = max(f.stat().st_mtime for f in history_files)
        oldest_preprocessed = min(f.stat().st_mtime for f in existing_files)

        if newest_history <= oldest_preprocessed:
            print(f"  Subject data up-to-date ({len(existing_files)} files)")
            # Load metadata for return values
            meta_path = output_dir / 'metadata.npz'
            if meta_path.exists():
                meta = np.load(meta_path)
                return output_dir, meta['iterations'], meta['batch_boundaries']
            # Fall through to regenerate if metadata missing

    # Clear old files
    if existing_files:
        for old_file in existing_files:
            old_file.unlink()
        meta_path = output_dir / 'metadata.npz'
        if meta_path.exists():
            meta_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {len(history_files)} history files...")
    history, iterations, _, metadata = combine_batch_histories(history_files)
    print(f"  Combined shape: {history.shape}")

    n_samples, n_chains, n_params = history.shape

    # Auto-detect n_subjects
    if n_subjects is None:
        # History contains:
        # - Raw state: N_HYPER hyperparams + n_subjects * NPARAMS subject epsilon params
        # - Generated quantities: n_subjects * 4 (r, phi, eta, lambda) + 8 (population params)
        # Total = N_HYPER + n_subjects * NPARAMS + n_subjects * 4 + 8
        #       = N_HYPER + 8 + n_subjects * (NPARAMS + 4)
        n_subjects = (n_params - N_HYPER - 8) // (NPARAMS + 4)
        print(f"  Auto-detected {n_subjects} subjects")

    # Calculate batch boundaries from metadata
    batch_boundaries = [0]
    if metadata and 'batch_sizes' in metadata:
        cumsum = 0
        for size in metadata['batch_sizes']:
            cumsum += size
            batch_boundaries.append(cumsum)

    # First generated quantity index (after raw params)
    # Raw params: N_HYPER hyperparams + n_subjects * NPARAMS subject params
    first_gq = N_HYPER + n_subjects * NPARAMS

    print(f"  Extracting {n_subjects} subjects...")
    for sid in range(n_subjects):
        if (sid + 1) % 100 == 0 or sid == 0:
            print(f"    Subject {sid}...", flush=True)

        # Generated quantities layout: r[0:n], phi[n:2n], eta[2n:3n], lambda[3n:4n], then pop params
        subject_data = {
            'r': history[:, :, first_gq + sid],
            'phi': history[:, :, first_gq + n_subjects + sid],
            'eta': history[:, :, first_gq + 2 * n_subjects + sid],
            'lambda': history[:, :, first_gq + 3 * n_subjects + sid],
        }

        out_path = output_dir / f'subject_{sid:03d}_params.npz'
        np.savez_compressed(out_path, **subject_data)

    # Save metadata
    np.savez(output_dir / 'metadata.npz',
             iterations=iterations,
             batch_boundaries=batch_boundaries,
             n_subjects=n_subjects)

    print(f"  Saved {n_subjects} subject files to {output_dir}")
    return output_dir, iterations, batch_boundaries


def preprocess_hyperparams(history_dir, output_dir, force=False):
    """
    Load full history and save hyperparameter data to a single file.

    Args:
        history_dir: Directory with history files
        output_dir: Directory to save preprocessed hyperparameter file
        force: If True, overwrite existing data

    Returns:
        Path to the preprocessed hyperparameter file
    """
    history_dir = Path(history_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'hyperparams_preprocessed.npz'

    # Load history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files in {history_dir}")

    # Check if preprocessed data already exists and is up-to-date
    if output_path.exists() and not force:
        newest_history = max(f.stat().st_mtime for f in history_files)
        preprocessed_time = output_path.stat().st_mtime

        if newest_history <= preprocessed_time:
            print(f"  Hyperparameter data up-to-date: {output_path}")
            return output_path

    print(f"  Loading {len(history_files)} history files...")
    history, iterations, _, metadata = combine_batch_histories(history_files)
    print(f"  Combined shape: {history.shape}")

    n_samples, n_chains, n_total_params = history.shape

    # Calculate hyperparameter indices
    # History contains raw chain state (2012 params) + generated quantities (2012 params)
    # Raw chain state: [subject epsilons (n_subjects * NPARAMS), hyperparams (N_HYPER)]
    n_raw_params = n_total_params // 2
    n_subjects = (n_raw_params - N_HYPER) // NPARAMS
    hyper_start = n_subjects * NPARAMS

    print(f"  Extracting {N_HYPER} hyperparameters (subjects={n_subjects})...")

    # Extract hyperparameter samples
    hyper_data = {}
    for param_idx, (state_name, latex_label) in enumerate(HYPERPARAM_INFO):
        hyper_data[state_name] = history[:, :, hyper_start + param_idx]

    # Calculate batch boundaries
    batch_boundaries = [0]
    if metadata and 'batch_sizes' in metadata:
        cumsum = 0
        for size in metadata['batch_sizes']:
            cumsum += size
            batch_boundaries.append(cumsum)

    # Save preprocessed data
    np.savez_compressed(
        output_path,
        iterations=iterations,
        batch_boundaries=np.array(batch_boundaries),
        n_subjects=n_subjects,
        **hyper_data
    )

    print(f"  Saved hyperparameter data to {output_path}")
    return output_path


# =============================================================================
# STAGE 2: PARALLEL PLOTTING - Workers load from files
# =============================================================================

def _plot_single_subject(args):
    """
    Worker function to plot a single subject.

    Args:
        args: tuple of (subject_id, params_dir, output_dir, iterations,
              batch_boundaries, prizes, prize_scale)
              prizes and prize_scale may be None (CE page skipped).
    """
    subject_id, params_dir, output_dir, iterations, batch_boundaries, \
        prizes, prize_scale = args

    params_dir = Path(params_dir)
    output_dir = Path(output_dir)

    # Load subject data from file
    params_path = params_dir / f'subject_{subject_id:03d}_params.npz'
    data = np.load(params_path)
    params = {k: data[k] for k in ['r', 'phi', 'eta', 'lambda']}

    # Generate PDF
    pdf_path = output_dir / f'subject_{subject_id:03d}.pdf'

    with PdfPages(pdf_path) as pdf:
        # Page 1: Trace plots
        _plot_subject_traces(params, iterations, batch_boundaries, subject_id, pdf)

        # Page 2: Density plots (last half)
        _plot_subject_densities(params, subject_id, pdf)

        # Page 3: Utility spaghetti (last half)
        _plot_subject_utility(params, subject_id, pdf)

        # Page 4: CE distribution (last half)
        if prizes is not None and prize_scale is not None:
            _plot_subject_ce(params, subject_id, pdf,
                             np.asarray(prizes), prize_scale)

    return subject_id


def _compute_subject_ces(r_samples, phi_samples, eta_samples, prizes_norm,
                          prize_scale):
    """
    Compute CE of a 3-outcome equally-likely reference lottery for each
    posterior draw.  Pure numpy (no JAX) for multiprocessing compatibility.

    Args:
        r_samples: 1D array of CRRA coefficients
        phi_samples: 1D array of Prelec curvature (natural scale)
        eta_samples: 1D array of Prelec elevation (natural scale)
        prizes_norm: Array [x1, x2, x3] normalized by prize_scale, sorted ascending
        prize_scale: Denominator used for normalization

    Returns:
        1D array of CEs in original currency units
    """
    x1, x2, x3 = prizes_norm

    # CRRA utility (vectorized, with log-utility branch)
    use_log = np.abs(r_samples - 1.0) < 0.01
    onemr = 1.0 - r_samples
    # Guard against division by zero in non-log branch
    onemr_safe = np.where(use_log, 1.0, onemr)

    def _crra(x):
        return np.where(use_log, np.log(x), np.power(x, onemr) / onemr_safe)

    u1, u2, u3 = _crra(x1), _crra(x2), _crra(x3)

    # Prelec PWF: w(p) = exp(-eta * (-ln p)^phi)
    neg_ln_one_third = -np.log(1.0 / 3.0)   # scalar
    neg_ln_two_thirds = -np.log(2.0 / 3.0)   # scalar
    w13 = np.exp(-eta_samples * np.power(neg_ln_one_third, phi_samples))
    w23 = np.exp(-eta_samples * np.power(neg_ln_two_thirds, phi_samples))

    # Decision weights
    pi3 = w13
    pi2 = w23 - w13
    pi1 = 1.0 - w23

    # RDU value
    rdu = pi1 * u1 + pi2 * u2 + pi3 * u3

    # Inverse CRRA
    inner = np.maximum(onemr_safe * rdu, 1e-30)
    ce_norm = np.where(use_log, np.exp(rdu),
                       np.power(inner, 1.0 / onemr_safe))

    return ce_norm * prize_scale


def _plot_subject_ce(params, subject_id, pdf, prizes, prize_scale):
    """
    Plot posterior predictive CE distribution for a single subject.

    Uses the last half of samples (same burn-in as density/utility pages).

    Args:
        params: dict with 'r', 'phi', 'eta' arrays of shape (n_samples, n_chains)
        subject_id: subject index
        pdf: PdfPages object
        prizes: array of 3 reference prizes in original currency [min, mid, max]
        prize_scale: normalization denominator
    """
    n_samples = params['r'].shape[0]
    burn_idx = n_samples // 2

    r_flat = params['r'][burn_idx:, :].flatten()
    phi_flat = params['phi'][burn_idx:, :].flatten()
    eta_flat = params['eta'][burn_idx:, :].flatten()

    prizes_norm = np.asarray(prizes) / prize_scale

    ces = _compute_subject_ces(r_flat, phi_flat, eta_flat, prizes_norm,
                               prize_scale)

    # Filter non-finite
    finite = np.isfinite(ces)
    ces = ces[finite]
    if len(ces) == 0:
        return

    ev = np.mean(prizes)
    median_ce = np.median(ces)
    ci_low, ci_high = np.percentile(ces, [2.5, 97.5])
    x_lo = float(prizes[0])
    x_hi = float(prizes[-1])

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Subject {subject_id} - Posterior Predictive CE', fontsize=14)

    ax.hist(ces, bins=CE_HIST_BINS, density=True, alpha=0.6, color=COLOR_DATA,
            range=(x_lo, x_hi))

    # Reference lines
    ax.axvline(ev, color=COLOR_REFERENCE, linestyle='-', linewidth=1.5,
               label=f'EV = {ev:.1f}')
    for p in prizes:
        ax.axvline(p, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(median_ce, color=COLOR_MEDIAN, linestyle='--', linewidth=1.5,
               label=f'Median CE = {median_ce:.1f}')
    ax.axvline(ci_low, color=COLOR_MEDIAN, linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axvline(ci_high, color=COLOR_MEDIAN, linestyle=':', linewidth=1.0, alpha=0.7)

    text = (f'Median = {median_ce:.1f}\n'
            f'95% CI = [{ci_low:.1f}, {ci_high:.1f}]\n'
            f'N draws = {len(ces)}')
    ax.text(0.97, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Certainty Equivalent')
    ax.set_ylabel('Density')
    ax.set_title('3 equally-likely prizes: '
                 f'{prizes[0]:.0f}, {prizes[1]:.0f}, {prizes[2]:.0f}')
    ax.set_xlim(x_lo, x_hi)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_subject_traces(params, iterations, batch_boundaries, subject_id, pdf):
    """Plot trace plots for a subject."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Subject {subject_id} - Trace Plots', fontsize=14)

    n_samples, n_chains = params['r'].shape

    # Subsample chains for visibility
    chain_subsample = min(50, n_chains)
    chain_indices = np.linspace(0, n_chains - 1, chain_subsample, dtype=int)
    chain_colors = plt.cm.tab20(np.linspace(0, 1, min(20, chain_subsample)))

    for i, (name, label) in enumerate(SUBJECT_PARAM_INFO):
        ax = axes[i]
        param_samples = params[name]

        for j, chain_idx in enumerate(chain_indices):
            color = chain_colors[j % len(chain_colors)]
            ax.plot(iterations, param_samples[:, chain_idx],
                    color=color, alpha=0.4, lw=0.3)

        # Run boundaries
        for boundary in batch_boundaries[1:]:
            if boundary < len(iterations):
                ax.axvline(iterations[boundary], color='gray',
                           linestyle=':', alpha=0.7, lw=1)

        # Median line (from last batch only)
        if batch_boundaries and len(batch_boundaries) > 1:
            last_batch_start = batch_boundaries[-2]
            last_batch_samples = param_samples[last_batch_start:, :]
        else:
            n_samples = param_samples.shape[0]
            last_batch_samples = param_samples[n_samples // 2:, :]
        median_val = np.median(last_batch_samples)
        ax.axhline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=1.5, alpha=0.8)

        ax.set_ylabel(label)
        ax.set_xlabel('Iteration')
        ax.set_title(f'{name}')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_subject_densities(params, subject_id, pdf):
    """Plot density plots for a subject (using last half of samples)."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Subject {subject_id} - Posterior Densities', fontsize=14)

    n_samples = params['r'].shape[0]
    burn_idx = n_samples // 2

    for i, (name, label) in enumerate(SUBJECT_PARAM_INFO):
        ax = axes[i]
        flat_samples = params[name][burn_idx:, :].flatten()

        # KDE
        try:
            kde = stats.gaussian_kde(flat_samples)
            x_range = np.linspace(flat_samples.min(), flat_samples.max(), 200)
            ax.fill_between(x_range, kde(x_range), color=COLOR_DATA, alpha=0.6)
            ax.plot(x_range, kde(x_range), color=COLOR_DATA, lw=1.5)
        except Exception:
            ax.hist(flat_samples, bins=50, density=True, color=COLOR_DATA, alpha=0.6)

        median_val = np.median(flat_samples)
        ax.axvline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=2,
                   label=f'Median: {median_val:.3f}')
        ax.legend(loc='upper right', fontsize=9)

        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{name}')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_subject_utility(params, subject_id, pdf):
    """Plot utility function spaghetti plot (using last half of samples)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Subject {subject_id} - CRRA Utility Function', fontsize=14)

    n_samples = params['r'].shape[0]
    burn_idx = n_samples // 2

    r_samples = params['r'][burn_idx:, :].flatten()

    # Subsample for spaghetti plot
    n_draws = min(N_SPAGHETTI_SAMPLES, len(r_samples))
    np.random.seed(subject_id)  # Reproducible
    draw_indices = np.random.choice(len(r_samples), n_draws, replace=False)

    # Domain for utility function
    x = np.linspace(X_MIN, X_MAX, 200)

    # Plot spaghetti (posterior draws)
    for idx in draw_indices:
        r = r_samples[idx]
        u = crra_utility(x, r)
        ax.plot(x, u, color=COLOR_SPAGHETTI, alpha=0.15, lw=0.8)

    # Median utility curve
    r_median = np.median(r_samples)
    u_median = crra_utility(x, r_median)
    ax.plot(x, u_median, color=COLOR_MEDIAN, lw=2.5, zorder=10,
            label=f'Median (r={r_median:.2f})')

    # Risk-neutral reference
    u_neutral = crra_utility(x, 0.0)
    ax.plot(x, u_neutral, color=COLOR_REFERENCE, linestyle='--', lw=1.5,
            label='Risk neutral (r=0)')

    ax.set_xlabel('Outcome (DKK)')
    ax.set_ylabel('Normalized Utility')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def parallel_plot_subjects(params_dir, output_dir, iterations, batch_boundaries,
                           n_workers=1, max_subjects=None,
                           prizes=None, prize_scale=None):
    """
    Run parallel plotting with a pool of workers.

    Args:
        params_dir: Directory with per-subject param files
        output_dir: Directory for output PDFs
        iterations: Iteration numbers array
        batch_boundaries: MCMC run boundary indices
        n_workers: Number of parallel workers
        max_subjects: Limit number of subjects (for testing)
        prizes: Array of 3 reference prizes [min, mid, max] for CE page (None to skip)
        prize_scale: Prize normalization denominator for CE page (None to skip)
    """
    params_dir = Path(params_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect n_subjects from files
    subject_files = sorted(params_dir.glob('subject_*_params.npz'))
    n_subjects = len(subject_files)

    if max_subjects:
        n_subjects = min(n_subjects, max_subjects)

    print(f"  Plotting {n_subjects} subjects with {n_workers} workers...")

    # Convert to lists for pickling
    iterations_list = iterations.tolist() if hasattr(iterations, 'tolist') else list(iterations)
    boundaries_list = list(batch_boundaries)
    prizes_list = prizes.tolist() if hasattr(prizes, 'tolist') else prizes

    # Build task list - workers receive paths, not data
    tasks = [
        (sid, str(params_dir), str(output_dir), iterations_list, boundaries_list,
         prizes_list, prize_scale)
        for sid in range(n_subjects)
    ]

    completed = 0
    with Pool(n_workers) as pool:
        for sid in pool.imap_unordered(_plot_single_subject, tasks):
            completed += 1
            if completed % 50 == 0 or completed == n_subjects:
                print(f"    Completed {completed}/{n_subjects}", flush=True)

    print(f"  Done! PDFs saved to {output_dir}")


# =============================================================================
# HYPERPARAMETER PLOTS (not parallelized - single PDF)
# =============================================================================

def plot_hyperparameters(history, iterations, batch_boundaries, output_path,
                         hyperprior_specs=None):
    """
    Generate hyperparameter diagnostic PDF.

    Layout: 2-column (traces left, densities right), up to 5 params per page.

    Args:
        history: MCMC history array
        iterations: Iteration numbers
        batch_boundaries: Run boundaries
        output_path: Path to save PDF
        hyperprior_specs: Optional dict with hyperprior specs for prior reference lines
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_chains, n_total_params = history.shape
    burn_idx = n_samples // 2

    # Compute hyperparameter offset
    # History contains raw chain state (2012 params) + generated quantities (2012 params)
    # Raw chain state: [subject epsilons (n_subjects * NPARAMS), hyperparams (N_HYPER)]
    # n_subjects = (n_total_params / 2 - N_HYPER) / NPARAMS
    n_raw_params = n_total_params // 2  # 2012
    n_subjects = (n_raw_params - N_HYPER) // NPARAMS
    hyper_start = n_subjects * NPARAMS  # Index where hyperparameters start

    # Compute prior quantiles if specs provided
    prior_quantiles = None
    if hyperprior_specs is not None:
        prior_quantiles = compute_hyperprior_quantiles(hyperprior_specs)

    # Subsample chains for traces
    n_chains_plot = min(100, n_chains)
    chain_indices = np.linspace(0, n_chains - 1, n_chains_plot, dtype=int)
    chain_colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_chains_plot)))

    with PdfPages(output_path) as pdf:
        n_params = len(HYPERPARAM_INFO)
        params_per_page = 5

        for page_start in range(0, n_params, params_per_page):
            page_end = min(page_start + params_per_page, n_params)
            n_on_page = page_end - page_start

            fig, axes = plt.subplots(n_on_page, 2, figsize=(14, 2.8 * n_on_page))
            if n_on_page == 1:
                axes = axes.reshape(1, 2)

            for i, param_idx in enumerate(range(page_start, page_end)):
                state_name, latex_label = HYPERPARAM_INFO[param_idx]
                ax_trace = axes[i, 0]
                ax_density = axes[i, 1]

                # Use correct index: hyper_start + param_idx
                param_samples = history[:, :, hyper_start + param_idx]

                # Trace plot
                for j, chain_idx in enumerate(chain_indices):
                    color = chain_colors[j % len(chain_colors)]
                    ax_trace.plot(iterations, param_samples[:, chain_idx],
                                  color=color, alpha=0.4, lw=0.3)

                # Run boundaries
                for boundary in batch_boundaries[1:]:
                    if boundary < len(iterations):
                        ax_trace.axvline(iterations[boundary], color='gray',
                                         linestyle=':', alpha=0.7, lw=1)

                # Compute median from last batch only (final converged distribution)
                if batch_boundaries and len(batch_boundaries) > 1:
                    last_batch_start = batch_boundaries[-2]
                    last_batch_samples = param_samples[last_batch_start:, :]
                else:
                    last_batch_samples = param_samples[burn_idx:, :]
                median_val = np.median(last_batch_samples)
                ax_trace.axhline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=1.5,
                                label='Posterior median (last run)')

                # Prior reference lines on trace plot
                if prior_quantiles and state_name in prior_quantiles:
                    prior_med, prior_lo, prior_hi = prior_quantiles[state_name]
                    ax_trace.axhline(prior_med, color=COLOR_PRIOR, linestyle='--', lw=1.5,
                                    label='Prior median')
                    ax_trace.axhline(prior_lo, color=COLOR_PRIOR, linestyle=':', lw=1.0,
                                    alpha=0.7)
                    ax_trace.axhline(prior_hi, color=COLOR_PRIOR, linestyle=':', lw=1.0,
                                    alpha=0.7, label='Prior 95% CI')

                ax_trace.set_ylabel(latex_label)
                ax_trace.set_xlabel('Iteration')
                ax_trace.set_title(f'{state_name} - Trace')
                ax_trace.legend(loc='upper right', fontsize=7)

                # Density plot (per run)
                if batch_boundaries and len(batch_boundaries) > 1:
                    for run_idx in range(len(batch_boundaries) - 1):
                        start = max(batch_boundaries[run_idx], burn_idx)
                        end = batch_boundaries[run_idx + 1]
                        if start < end:
                            run_samples = param_samples[start:end, :].flatten()
                            try:
                                kde = stats.gaussian_kde(run_samples)
                                x_range = np.linspace(run_samples.min(), run_samples.max(), 200)
                                color = RUN_COLORS[run_idx % len(RUN_COLORS)]
                                ax_density.fill_between(x_range, kde(x_range),
                                                       alpha=0.3, color=color,
                                                       label=f'Run {run_idx + 1}')
                                ax_density.plot(x_range, kde(x_range), color=color, lw=1)
                            except Exception:
                                pass
                else:
                    flat_samples = param_samples[burn_idx:, :].flatten()
                    try:
                        kde = stats.gaussian_kde(flat_samples)
                        x_range = np.linspace(flat_samples.min(), flat_samples.max(), 200)
                        ax_density.fill_between(x_range, kde(x_range),
                                               color=COLOR_DATA, alpha=0.6)
                        ax_density.plot(x_range, kde(x_range), color=COLOR_DATA, lw=1.5)
                    except Exception:
                        ax_density.hist(flat_samples, bins=50, density=True,
                                       color=COLOR_DATA, alpha=0.6)

                ax_density.axvline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=2,
                                  label='Posterior median (last run)')

                # Prior reference lines on density plot
                if prior_quantiles and state_name in prior_quantiles:
                    prior_med, prior_lo, prior_hi = prior_quantiles[state_name]
                    ax_density.axvline(prior_med, color=COLOR_PRIOR, linestyle='--', lw=2,
                                      label='Prior median')
                    ax_density.axvline(prior_lo, color=COLOR_PRIOR, linestyle=':', lw=1.5,
                                      alpha=0.7)
                    ax_density.axvline(prior_hi, color=COLOR_PRIOR, linestyle=':', lw=1.5,
                                      alpha=0.7, label='Prior 95% CI')
                ax_density.legend(loc='upper right', fontsize=8)
                ax_density.set_xlabel(latex_label)
                ax_density.set_ylabel('Density')
                ax_density.set_title(f'{state_name} - Posterior')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: {output_path}")


# =============================================================================
# STANDALONE PLOT FUNCTIONS
# =============================================================================

def plot_hyperparams_from_preprocessed(preprocessed_path, plots_dir, hyperprior_specs=None):
    """
    Generate hyperparameter plots from preprocessed data file.

    Args:
        preprocessed_path: Path to hyperparams_preprocessed.npz file
        plots_dir: Directory to save plots
        hyperprior_specs: Optional dict with hyperprior specs for prior reference lines
    """
    preprocessed_path = Path(preprocessed_path)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")

    print(f"  Loading preprocessed hyperparameters from {preprocessed_path}...")
    data = np.load(preprocessed_path)

    iterations = data['iterations']
    batch_boundaries = list(data['batch_boundaries'])

    # Reconstruct hyperparameter array in format expected by plot_hyperparameters
    # Get sample shape from first hyperparameter
    first_state = HYPERPARAM_INFO[0][0]
    n_samples, n_chains = data[first_state].shape
    n_hyper = len(HYPERPARAM_INFO)

    # Build a "fake" history array with just hyperparameters at the expected indices
    # plot_hyperparameters expects history[:, :, hyper_start + param_idx]
    # We'll set hyper_start = 0 and put hyperparams at indices 0, 1, 2, ...
    hyper_history = np.zeros((n_samples, n_chains, n_hyper), dtype=np.float64)
    for param_idx, (state_name, _) in enumerate(HYPERPARAM_INFO):
        hyper_history[:, :, param_idx] = data[state_name]

    print(f"  Shape: ({n_samples}, {n_chains}, {n_hyper})")

    # Plot hyperparameters
    print("  Generating hyperparameter plots...")
    hyper_path = plots_dir / 'rdu_bhm_hyperparams.pdf'
    _plot_hyperparameters_direct(hyper_history, iterations, batch_boundaries, hyper_path,
                                  hyperprior_specs=hyperprior_specs)
    print(f"  Saved: {hyper_path}")


def _plot_hyperparameters_direct(hyper_history, iterations, batch_boundaries, output_path,
                                  hyperprior_specs=None):
    """
    Plot hyperparameters from a direct hyperparameter array (no offset calculation needed).

    Args:
        hyper_history: Array of shape (n_samples, n_chains, n_hyper) with hyperparams at indices 0, 1, ...
        iterations: Iteration numbers
        batch_boundaries: Run boundaries
        output_path: Path to save PDF
        hyperprior_specs: Optional dict with hyperprior specs for prior reference lines
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_chains, n_hyper = hyper_history.shape
    burn_idx = n_samples // 2

    # Compute prior quantiles if specs provided
    prior_quantiles = None
    if hyperprior_specs is not None:
        prior_quantiles = compute_hyperprior_quantiles(hyperprior_specs)

    # Subsample chains for traces
    n_chains_plot = min(100, n_chains)
    chain_indices = np.linspace(0, n_chains - 1, n_chains_plot, dtype=int)
    chain_colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_chains_plot)))

    with PdfPages(output_path) as pdf:
        n_params = len(HYPERPARAM_INFO)
        params_per_page = 5

        for page_start in range(0, n_params, params_per_page):
            page_end = min(page_start + params_per_page, n_params)
            n_on_page = page_end - page_start

            fig, axes = plt.subplots(n_on_page, 2, figsize=(14, 2.8 * n_on_page))
            if n_on_page == 1:
                axes = axes.reshape(1, 2)

            for i, param_idx in enumerate(range(page_start, page_end)):
                state_name, latex_label = HYPERPARAM_INFO[param_idx]
                ax_trace = axes[i, 0]
                ax_density = axes[i, 1]

                # Hyperparams are at indices 0, 1, 2, ... in hyper_history
                param_samples = hyper_history[:, :, param_idx]

                # Trace plot
                for j, chain_idx in enumerate(chain_indices):
                    color = chain_colors[j % len(chain_colors)]
                    ax_trace.plot(iterations, param_samples[:, chain_idx],
                                  color=color, alpha=0.4, lw=0.3)

                # Run boundaries
                for boundary in batch_boundaries[1:]:
                    if boundary < len(iterations):
                        ax_trace.axvline(iterations[boundary], color='gray',
                                         linestyle=':', alpha=0.7, lw=1)

                # Compute median from last batch only (final converged distribution)
                if batch_boundaries and len(batch_boundaries) > 1:
                    last_batch_start = batch_boundaries[-2]
                    last_batch_samples = param_samples[last_batch_start:, :]
                else:
                    last_batch_samples = param_samples[burn_idx:, :]
                median_val = np.median(last_batch_samples)
                ax_trace.axhline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=1.5,
                                label='Posterior median (last run)')

                # Prior reference lines on trace plot
                if prior_quantiles and state_name in prior_quantiles:
                    prior_med, prior_lo, prior_hi = prior_quantiles[state_name]
                    ax_trace.axhline(prior_med, color=COLOR_PRIOR, linestyle='--', lw=1.5,
                                    label='Prior median')
                    ax_trace.axhline(prior_lo, color=COLOR_PRIOR, linestyle=':', lw=1.0,
                                    alpha=0.7)
                    ax_trace.axhline(prior_hi, color=COLOR_PRIOR, linestyle=':', lw=1.0,
                                    alpha=0.7, label='Prior 95% CI')

                ax_trace.set_ylabel(latex_label)
                ax_trace.set_xlabel('Iteration')
                ax_trace.set_title(f'{state_name} - Trace')
                ax_trace.legend(loc='upper right', fontsize=7)

                # Density plot (per run)
                if batch_boundaries and len(batch_boundaries) > 1:
                    for run_idx in range(len(batch_boundaries) - 1):
                        start = max(batch_boundaries[run_idx], burn_idx)
                        end = batch_boundaries[run_idx + 1]
                        if start < end:
                            run_samples = param_samples[start:end, :].flatten()
                            try:
                                kde = stats.gaussian_kde(run_samples)
                                x_range = np.linspace(run_samples.min(), run_samples.max(), 200)
                                color = RUN_COLORS[run_idx % len(RUN_COLORS)]
                                ax_density.fill_between(x_range, kde(x_range),
                                                       alpha=0.3, color=color,
                                                       label=f'Run {run_idx + 1}')
                                ax_density.plot(x_range, kde(x_range), color=color, lw=1)
                            except Exception:
                                pass
                else:
                    flat_samples = param_samples[burn_idx:, :].flatten()
                    try:
                        kde = stats.gaussian_kde(flat_samples)
                        x_range = np.linspace(flat_samples.min(), flat_samples.max(), 200)
                        ax_density.fill_between(x_range, kde(x_range),
                                               color=COLOR_DATA, alpha=0.6)
                        ax_density.plot(x_range, kde(x_range), color=COLOR_DATA, lw=1.5)
                    except Exception:
                        ax_density.hist(flat_samples, bins=50, density=True,
                                       color=COLOR_DATA, alpha=0.6)

                ax_density.axvline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=2,
                                  label='Posterior median (last run)')

                # Prior reference lines on density plot
                if prior_quantiles and state_name in prior_quantiles:
                    prior_med, prior_lo, prior_hi = prior_quantiles[state_name]
                    ax_density.axvline(prior_med, color=COLOR_PRIOR, linestyle='--', lw=2,
                                      label='Prior median')
                    ax_density.axvline(prior_lo, color=COLOR_PRIOR, linestyle=':', lw=1.5,
                                      alpha=0.7)
                    ax_density.axvline(prior_hi, color=COLOR_PRIOR, linestyle=':', lw=1.5,
                                      alpha=0.7, label='Prior 95% CI')

                ax_density.set_xlabel(latex_label)
                ax_density.set_title(f'{state_name} - Density')
                ax_density.legend(loc='upper right', fontsize=7)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_hyperparams_only(history_dir, plots_dir, hyperprior_specs=None):
    """
    Generate hyperparameter plots directly from history files (no preprocessing).

    For large datasets, prefer using preprocess_hyperparams() followed by
    plot_hyperparams_from_preprocessed() to avoid reloading history.

    Args:
        history_dir: Directory containing history_*.npz files
        plots_dir: Directory to save plots
        hyperprior_specs: Optional dict with hyperprior specs for prior reference lines
    """
    history_dir = Path(history_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Check for history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"  Found {len(history_files)} history files")

    # Load history
    print("  Loading history for hyperparameters...")
    history, iterations, _, metadata = combine_batch_histories(history_files)
    print(f"  Shape: {history.shape}")

    # Calculate batch boundaries
    batch_boundaries = [0]
    if metadata and 'batch_sizes' in metadata:
        cumsum = 0
        for size in metadata['batch_sizes']:
            cumsum += size
            batch_boundaries.append(cumsum)

    # Plot hyperparameters
    print("  Generating hyperparameter plots...")
    hyper_path = plots_dir / 'rdu_bhm_hyperparams.pdf'
    plot_hyperparameters(history, iterations, batch_boundaries, hyper_path,
                        hyperprior_specs=hyperprior_specs)
    print(f"  Saved: {hyper_path}")


def plot_subjects_only(history_dir, plots_dir, max_subjects=None, n_workers=1,
                       prizes=None, prize_scale=None):
    """
    Generate only per-subject diagnostic plots.

    Args:
        history_dir: Directory containing history_*.npz files
        plots_dir: Directory to save plots
        max_subjects: Limit subjects for testing
        n_workers: Number of parallel workers
        prizes: Array of 3 reference prizes for CE page (None to skip)
        prize_scale: Prize normalization for CE page (None to skip)
    """
    history_dir = Path(history_dir)
    plots_dir = Path(plots_dir)
    subjects_dir = plots_dir / 'subjects'
    params_dir = history_dir.parent / 'subject_params'

    # Stage 1: Preprocess
    print("  Stage 1: Preprocessing subject data...")
    params_dir, iterations, batch_boundaries = preprocess_subjects(
        history_dir, params_dir, force=False
    )

    # Stage 2: Parallel plot
    print("  Stage 2: Parallel plotting...")
    parallel_plot_subjects(
        params_dir, subjects_dir, iterations, batch_boundaries,
        n_workers=n_workers, max_subjects=max_subjects,
        prizes=prizes, prize_scale=prize_scale
    )


def paper_plots(quantities_path, plots_dir):
    """
    Generate publication-ready plots for the paper.

    Args:
        quantities_path: Path to paper_quantities.npz
        plots_dir: Directory to save plots
    """
    quantities_path = Path(quantities_path)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not quantities_path.exists():
        raise FileNotFoundError(f"Run gen_paper_quantities first: {quantities_path}")

    data = np.load(quantities_path)

    print("Paper quantities loaded:")
    print(f"  n_subjects: {data['n_subjects']}")
    print(f"  n_samples (post burn-in): {data['n_samples_post_burnin']}")
    print(f"  R-hat max: {data['rhat_max']:.4f}")

    # TODO: Add specific paper plots as needed
    print("  Paper plots not yet implemented")


# =============================================================================
# PRIOR DISTRIBUTION PLOTS
# =============================================================================

# Parameter display info for prior plots: (key, label, transform_exp)
# transform_exp=True means samples are on log scale and should be exp-transformed for display
PRIOR_PARAM_INFO = [
    ('r', r'$r$ (CRRA)', False),
    ('phi_raw', r'$\phi$ (Prelec curvature)', True),
    ('eta_raw', r'$\eta$ (Prelec elevation)', True),
    ('lambda_raw', r'$\lambda$ (noise)', True),
]


def plot_direct_prior(direct_prior, output_path, n_samples=10000, seed=42):
    """
    Generate 4-panel plot of the direct prior distribution.

    Samples from N(mu, sigma2) for each parameter and plots the distribution.
    For log-scale parameters (phi, eta, lambda), displays on natural scale via exp().

    Args:
        direct_prior: Dict with {param: {'mu': float, 'sigma2': float}}
        output_path: Path to save the figure
        n_samples: Number of samples to draw
        seed: Random seed for reproducibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    fig.suptitle('Direct Prior Distribution', fontsize=14, fontweight='bold')

    for i, (key, label, transform_exp) in enumerate(PRIOR_PARAM_INFO):
        ax = axes[i]
        mu = direct_prior[key]['mu']
        sigma = np.sqrt(direct_prior[key]['sigma2'])

        # Sample from direct prior (on raw/log scale)
        raw_samples = np.random.normal(mu, sigma, n_samples)

        # Transform to natural scale if needed
        if transform_exp:
            samples = np.exp(raw_samples)
            median = np.exp(mu)  # Median of lognormal
        else:
            samples = raw_samples
            median = mu

        # Plot histogram and KDE
        ax.hist(samples, bins=50, density=True, alpha=0.6, color=COLOR_DATA,
                edgecolor='white', linewidth=0.5)

        try:
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(np.percentile(samples, 0.5), np.percentile(samples, 99.5), 200)
            ax.plot(x_range, kde(x_range), color=COLOR_DATA, lw=2)
        except Exception:
            pass

        # Mark median (which equals exp(mu) for lognormal)
        ax.axvline(median, color=COLOR_MEDIAN, linestyle='--', lw=2,
                   label=f'Median: {median:.2f}')

        # Add 95% CI
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        ax.axvline(ci_low, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(ci_high, color='gray', linestyle=':', alpha=0.7)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)

        # Add text annotation with raw scale params and natural scale stats
        if transform_exp:
            ax.text(0.02, 0.98,
                    f'Log scale: $\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$\n'
                    f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.02, 0.98,
                    f'$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$\n'
                    f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_unconditional_prior(direct_prior, prior_selection, output_path,
                              n_samples=10000, seed=42):
    """
    Generate 4-panel plot of the unconditional BHM prior distribution.

    Samples from the full BHM structure:
    1. Sample hyperparams: mu ~ N(mu_0, tau_0^2), sigma^2 ~ IG(alpha_0, beta_0)
    2. Sample individual: theta ~ N(mu, sigma^2)

    For log-scale parameters (phi, eta, lambda), displays on natural scale via exp().

    Args:
        direct_prior: Dict with {param: {'mu': float, 'sigma2': float}}
        prior_selection: Dict with {'rho': float, 'alpha_0': float}
        output_path: Path to save the figure
        n_samples: Number of samples to draw
        seed: Random seed for reproducibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    rho = prior_selection['rho']
    alpha_0 = prior_selection['alpha_0']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    fig.suptitle('Unconditional Prior Distribution (BHM)', fontsize=14, fontweight='bold')

    for i, (key, label, transform_exp) in enumerate(PRIOR_PARAM_INFO):
        ax = axes[i]

        # Get direct prior params
        mu_ps = direct_prior[key]['mu']
        var_ps = direct_prior[key]['sigma2']
        sigma_ps = np.sqrt(var_ps)

        # Derive hyperpriors
        tau_0_sq = rho * var_ps
        expected_sigma_sq = (1 - rho) * var_ps
        beta_0 = expected_sigma_sq * (alpha_0 - 1)

        # Sample from BHM (on raw/log scale)
        raw_samples = np.zeros(n_samples)
        for j in range(n_samples):
            # Sample population mean
            mu = np.random.normal(mu_ps, np.sqrt(tau_0_sq))

            # Sample population variance (Inverse-Gamma)
            sigma_sq = 1.0 / np.random.gamma(alpha_0, 1.0 / beta_0)

            # Sample individual
            raw_samples[j] = np.random.normal(mu, np.sqrt(sigma_sq))

        # Transform to natural scale if needed
        if transform_exp:
            samples = np.exp(raw_samples)
            median = np.exp(mu_ps)
        else:
            samples = raw_samples
            median = mu_ps

        # Plot histogram and KDE
        ax.hist(samples, bins=50, density=True, alpha=0.6, color=COLOR_DATA,
                edgecolor='white', linewidth=0.5)

        try:
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(np.percentile(samples, 0.5), np.percentile(samples, 99.5), 200)
            ax.plot(x_range, kde(x_range), color=COLOR_DATA, lw=2)
        except Exception:
            pass

        # Mark median
        ax.axvline(median, color=COLOR_MEDIAN, linestyle='--', lw=2,
                   label=f'Median: {median:.2f}')

        # Add 95% CI
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        ax.axvline(ci_low, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(ci_high, color='gray', linestyle=':', alpha=0.7)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)

        # Add text with sample stats
        if transform_exp:
            ax.text(0.02, 0.98,
                    f'Log scale: $\\mu={mu_ps:.2f}$, $\\sigma={sigma_ps:.2f}$\n'
                    f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            sample_mean = np.mean(samples)
            sample_std = np.std(samples)
            ax.text(0.02, 0.98,
                    f'Sample: $\\mu={sample_mean:.2f}$, $\\sigma={sample_std:.2f}$\n'
                    f'Target: $\\mu={mu_ps:.2f}$, $\\sigma={sigma_ps:.2f}$',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_prior_comparison(direct_prior, prior_selection, output_path,
                           n_samples=10000, seed=42):
    """
    Generate comparison plot showing direct prior vs unconditional BHM prior.

    Overlays both distributions to verify they match.

    Args:
        direct_prior: Dict with {param: {'mu': float, 'sigma2': float}}
        prior_selection: Dict with {'rho': float, 'alpha_0': float}
        output_path: Path to save the figure
        n_samples: Number of samples to draw
        seed: Random seed for reproducibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    rho = prior_selection['rho']
    alpha_0 = prior_selection['alpha_0']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    fig.suptitle('Prior Comparison: Direct vs Unconditional BHM', fontsize=14, fontweight='bold')

    for i, (key, label, transform_exp) in enumerate(PRIOR_PARAM_INFO):
        ax = axes[i]

        # Get direct prior params
        mu_ps = direct_prior[key]['mu']
        var_ps = direct_prior[key]['sigma2']
        sigma_ps = np.sqrt(var_ps)

        # Sample from direct prior
        direct_samples = np.random.normal(mu_ps, sigma_ps, n_samples)

        # Derive hyperpriors and sample from BHM
        tau_0_sq = rho * var_ps
        expected_sigma_sq = (1 - rho) * var_ps
        beta_0 = expected_sigma_sq * (alpha_0 - 1)

        bhm_samples = np.zeros(n_samples)
        for j in range(n_samples):
            mu = np.random.normal(mu_ps, np.sqrt(tau_0_sq))
            sigma_sq = 1.0 / np.random.gamma(alpha_0, 1.0 / beta_0)
            bhm_samples[j] = np.random.normal(mu, np.sqrt(sigma_sq))

        # Transform to natural scale if needed (exp for log-transformed params)
        if transform_exp:
            direct_samples = np.exp(direct_samples)
            bhm_samples = np.exp(bhm_samples)

        # Plot both distributions
        x_min = min(direct_samples.min(), bhm_samples.min())
        x_max = max(direct_samples.max(), bhm_samples.max())
        x_range = np.linspace(x_min, x_max, 200)

        try:
            kde_direct = stats.gaussian_kde(direct_samples)
            ax.plot(x_range, kde_direct(x_range), color='blue', lw=2,
                    label='Direct Prior', alpha=0.8)
            ax.fill_between(x_range, kde_direct(x_range), color='blue', alpha=0.2)
        except Exception:
            pass

        try:
            kde_bhm = stats.gaussian_kde(bhm_samples)
            ax.plot(x_range, kde_bhm(x_range), color='red', lw=2,
                    label='Unconditional BHM', linestyle='--', alpha=0.8)
        except Exception:
            pass

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
