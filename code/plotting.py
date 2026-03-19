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
from helpers import load_and_filter_history
from posteriors.rdu_bhm_ce import (
    NPARAMS as NPARAMS_CE,
    N_HYPER as N_HYPER_CE,
)

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


# CE-model hyperparameter display info: (state_name, latex_label)
# State names match the keys saved in preprocess_hyperparams_ce (logit scale).
HYPERPARAM_INFO_CE = [
    ('ce1_logit_mu',  r'$\mu_1$ (CE1, \$)'),
    ('ce1_logit_psi', r'$\psi_1$ (CE1)'),
    ('ce2_logit_mu',  r'$\mu_2$ (CE2, \$)'),
    ('ce2_logit_psi', r'$\psi_2$ (CE2)'),
    ('ce3_logit_mu',  r'$\mu_3$ (CE3, \$)'),
    ('ce3_logit_psi', r'$\psi_3$ (CE3)'),
    ('pc_logit_mu',   r'$\mu_P$ (Pchoice)'),
    ('pc_logit_psi',  r'$\psi_P$ (Pchoice)'),
]

# Transform specs for CE hyperparameters: map logit-scale chain state to natural scale.
# For mu_k: natural = lo + (hi - lo) * sigmoid(logit)  (CE in dollars / probability)
# For psi_k: natural = sigmoid(logit)                  (dispersion on [0,1])
CE_HYPER_TRANSFORMS = [
    {'lo': 10, 'hi': 50},   # ce1_logit_mu  -> CE1 in $
    {'lo': 0,  'hi': 1},    # ce1_logit_psi -> psi on [0,1]
    {'lo': 20, 'hi': 40},   # ce2_logit_mu  -> CE2 in $
    {'lo': 0,  'hi': 1},    # ce2_logit_psi -> psi on [0,1]
    {'lo': 15, 'hi': 45},   # ce3_logit_mu  -> CE3 in $
    {'lo': 0,  'hi': 1},    # ce3_logit_psi -> psi on [0,1]
    {'lo': 0.0, 'hi': 1.0}, # pc_logit_mu   -> P(choose A)
    {'lo': 0,  'hi': 1},    # pc_logit_psi  -> psi on [0,1]
]

# Subject parameter display info: (name, latex_label)
SUBJECT_PARAM_INFO = [
    ('r', r'$r$'),
    ('phi', r'$\phi$'),
    ('eta', r'$\eta$'),
    ('lambda', r'$\lambda$'),
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

# Default outcome range (used if prizes not provided)
X_MIN = 0.0
X_MAX = 70.0

# CE plot constants
CE_HIST_BINS = 100


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

def preprocess_subjects(history_dir, output_dir, n_subjects=None, force=False,
                        nparams=None, n_hyper=None):
    """
    Load full history and save per-subject parameter files.

    Subject data is preserved between runs unless force=True or history is newer.

    Args:
        history_dir: Directory with history files
        output_dir: Directory to save per-subject files
        n_subjects: Number of subjects (auto-detected if None)
        force: If True, overwrite existing subject data
        nparams: Number of params per subject (default: NPARAMS from rdu_bhm_ce)
        n_hyper: Number of hyperparameters (default: N_HYPER from rdu_bhm_ce)

    Returns:
        tuple: (output_dir, iterations, batch_boundaries) for use in plotting
    """
    if nparams is None:
        nparams = NPARAMS_CE
    if n_hyper is None:
        n_hyper = N_HYPER_CE
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
    history, iterations, _, metadata = load_and_filter_history(history_files)
    print(f"  Combined shape: {history.shape}")

    n_samples, n_chains, n_params = history.shape

    # Auto-detect n_subjects
    if n_subjects is None:
        # History contains:
        # - Raw state: n_hyper hyperparams + n_subjects * nparams subject epsilon params
        # - Generated quantities: n_subjects * 4 (r, phi, eta, lambda) + 8 (population params)
        # Total = n_hyper + n_subjects * nparams + n_subjects * 4 + 8
        #       = n_hyper + 8 + n_subjects * (nparams + 4)
        n_subjects = (n_params - n_hyper - 8) // (nparams + 4)
        print(f"  Auto-detected {n_subjects} subjects")

    # Calculate batch boundaries from metadata
    batch_boundaries = [0]
    if metadata and 'batch_sizes' in metadata:
        cumsum = 0
        for size in metadata['batch_sizes']:
            cumsum += size
            batch_boundaries.append(cumsum)

    # First generated quantity index (after raw params)
    # Raw params: n_hyper hyperparams + n_subjects * nparams subject params
    first_gq = n_hyper + n_subjects * nparams

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


def preprocess_hyperparams_ce(history_dir, output_dir, force=False):
    """
    Load full history and save CE-model hyperparameter data to a single file.

    Uses CE-model constants (NPARAMS_CE, N_HYPER_CE) and HYPERPARAM_INFO_CE
    for state names.

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
    history, iterations, _, metadata = load_and_filter_history(history_files)
    print(f"  Combined shape: {history.shape}")

    n_samples, n_chains, n_total_params = history.shape

    # Calculate hyperparameter indices
    n_raw_params = n_total_params // 2
    n_subjects = (n_raw_params - N_HYPER_CE) // NPARAMS_CE
    hyper_start = n_subjects * NPARAMS_CE

    print(f"  Extracting {N_HYPER_CE} CE hyperparameters (subjects={n_subjects})...")

    # Extract hyperparameter samples using CE-specific state names
    hyper_data = {}
    for param_idx, (state_name, latex_label) in enumerate(HYPERPARAM_INFO_CE):
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

    print(f"  Saved CE hyperparameter data to {output_path}")
    return output_path


# =============================================================================
# STAGE 2: PARALLEL PLOTTING - Workers load from files
# =============================================================================

def _plot_single_subject(args):
    """
    Worker function to plot a single subject.

    Args:
        args: tuple of (subject_id, params_dir, output_dir, iterations,
              batch_boundaries, prizes, config)
              prizes may be None (CE page skipped).
              config may be None or a dict with plotting options.
    """
    subject_id, params_dir, output_dir, iterations, batch_boundaries, \
        prizes, config = args
    if config is None:
        config = {}

    params_dir = Path(params_dir)
    output_dir = Path(output_dir)

    # Load subject data from file
    params_path = params_dir / f'subject_{subject_id:03d}_params.npz'
    data = np.load(params_path)
    params = {k: data[k] for k in ['r', 'phi', 'eta', 'lambda']}

    # Generate PDF
    pdf_path = output_dir / f'subject_{subject_id:03d}.pdf'

    with PdfPages(pdf_path) as pdf:
        # Page 1: Trace plots (all history files)
        _plot_subject_traces(params, iterations, batch_boundaries, subject_id, pdf,
                             chain_subsample=config.get('chain_subsample', 50),
                             figsize=config.get('subject_trace_figsize', (12, 12)))

        # Page 2: Density plots (last history file)
        _plot_subject_densities(params, subject_id, pdf, batch_boundaries)

        # Page 3: Utility spaghetti (last history file)
        _plot_subject_utility(params, subject_id, pdf, batch_boundaries,
                              prizes=np.asarray(prizes) if prizes is not None else None,
                              figsize=config.get('subject_utility_figsize', (10, 7)),
                              n_spaghetti=config.get('n_spaghetti', None))

        # Page 4: CE distribution (last history file)
        if prizes is not None:
            _plot_subject_ce(params, subject_id, pdf,
                             np.asarray(prizes),
                             batch_boundaries,
                             figsize=config.get('subject_ce_figsize', (10, 7)),
                             bins=config.get('ce_hist_bins', None))

    return subject_id


def _compute_subject_ces(r_samples, phi_samples, eta_samples, prizes):
    """
    Compute CE of a 3-outcome equally-likely reference lottery for each
    posterior draw.  Pure numpy (no JAX) for multiprocessing compatibility.

    Args:
        r_samples: 1D array of CRRA coefficients
        phi_samples: 1D array of Prelec curvature (natural scale)
        eta_samples: 1D array of Prelec elevation (natural scale)
        prizes: Array [x1, x2, x3] sorted ascending

    Returns:
        1D array of CEs
    """
    x1, x2, x3 = prizes

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
    ce = np.where(use_log, np.exp(rdu),
                  np.power(inner, 1.0 / onemr_safe))

    return ce


def _plot_subject_ce(params, subject_id, pdf, prizes,
                     batch_boundaries, figsize=(10, 7), bins=None):
    """
    Plot posterior predictive CE distribution for a single subject.

    Uses last history file samples (same window as density/utility pages).

    Args:
        params: dict with 'r', 'phi', 'eta' arrays of shape (n_samples, n_chains)
        subject_id: subject index
        pdf: PdfPages object
        prizes: array of 3 reference prizes [min, mid, max]
        batch_boundaries: list of batch boundary indices
        figsize: Figure dimensions in inches (default (10, 7))
        bins: Number of histogram bins (default uses module-level CE_HIST_BINS)
    """
    if bins is None:
        bins = CE_HIST_BINS
    n_samples = params['r'].shape[0]
    burn_idx = _last_batch_start(batch_boundaries, n_samples)

    r_flat = params['r'][burn_idx:, :].flatten()
    phi_flat = params['phi'][burn_idx:, :].flatten()
    eta_flat = params['eta'][burn_idx:, :].flatten()

    ces = _compute_subject_ces(r_flat, phi_flat, eta_flat, np.asarray(prizes))

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

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f'Subject {subject_id} - Posterior Predictive CE', fontsize=14)

    ax.hist(ces, bins=bins, density=True, alpha=0.6, color=COLOR_DATA,
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

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _last_batch_start(batch_boundaries, n_samples):
    """Return the sample index where the last history file begins."""
    if batch_boundaries and len(batch_boundaries) > 1:
        return batch_boundaries[-2]
    return n_samples // 2


def _plot_subject_traces(params, iterations, batch_boundaries, subject_id, pdf,
                         chain_subsample=50, figsize=(12, 12)):
    """Plot trace plots for a subject."""
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle(f'Subject {subject_id} - Trace Plots', fontsize=14)

    n_samples, n_chains = params['r'].shape

    # Subsample chains for visibility
    chain_subsample = min(chain_subsample, n_chains)
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
        lb = _last_batch_start(batch_boundaries, param_samples.shape[0])
        median_val = np.median(param_samples[lb:, :])
        ax.axhline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=1.5, alpha=0.8)

        ax.set_ylabel(label)
        ax.set_xlabel('Iteration')
        ax.set_title(f'{name}')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_subject_densities(params, subject_id, pdf, batch_boundaries):
    """Plot density plots for a subject (using last history file)."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Subject {subject_id} - Posterior Densities', fontsize=14)

    n_samples = params['r'].shape[0]
    burn_idx = _last_batch_start(batch_boundaries, n_samples)

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
        ax.set_title(f'{name}')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_subject_utility(params, subject_id, pdf, batch_boundaries,
                          prizes=None, figsize=(10, 7), n_spaghetti=None):
    """Plot utility function spaghetti plot (using last history file)."""
    if n_spaghetti is None:
        n_spaghetti = N_SPAGHETTI_SAMPLES
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f'Subject {subject_id} - CRRA Utility Function', fontsize=14)

    n_samples = params['r'].shape[0]
    burn_idx = _last_batch_start(batch_boundaries, n_samples)

    r_samples = params['r'][burn_idx:, :].flatten()

    # Subsample for spaghetti plot
    n_draws = min(n_spaghetti, len(r_samples))
    np.random.seed(subject_id)  # Reproducible
    draw_indices = np.random.choice(len(r_samples), n_draws, replace=False)

    # Domain for utility function from prize range
    x_min = float(prizes[0]) if prizes is not None else X_MIN
    x_max = float(prizes[-1]) if prizes is not None else X_MAX
    x = np.linspace(x_min, x_max, 200)

    # Plot spaghetti (posterior draws)
    for idx in draw_indices:
        r = r_samples[idx]
        u = crra_utility(x, r, x_min=x_min, x_max=x_max)
        ax.plot(x, u, color=COLOR_SPAGHETTI, alpha=0.15, lw=0.8)

    # Median utility curve
    r_median = np.median(r_samples)
    u_median = crra_utility(x, r_median, x_min=x_min, x_max=x_max)
    ax.plot(x, u_median, color=COLOR_MEDIAN, lw=2.5, zorder=10,
            label=f'Median (r={r_median:.2f})')

    # Risk-neutral reference
    u_neutral = crra_utility(x, 0.0, x_min=x_min, x_max=x_max)
    ax.plot(x, u_neutral, color=COLOR_REFERENCE, linestyle='--', lw=1.5,
            label='Risk neutral (r=0)')

    ax.set_xlabel('Outcome')
    ax.set_ylabel('Normalized Utility')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def parallel_plot_subjects(params_dir, output_dir, iterations, batch_boundaries,
                           n_workers=1, max_subjects=None,
                           prizes=None, config=None):
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
        config: Optional dict with plotting config (forwarded to per-subject plot fns)
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
         prizes_list, config)
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
# STANDALONE PLOT FUNCTIONS
# =============================================================================

def plot_hyperparams_ce_from_preprocessed(preprocessed_path, plots_dir,
                                           ce_direct_prior=None, config=None):
    """
    Generate CE-model hyperparameter plots from preprocessed data file.

    Transforms logit-scale chain state to natural scale before plotting:
    - mu_k: lo + (hi-lo)*sigmoid(logit_mu) → CE in dollars (or probability)
    - psi_k: sigmoid(logit_psi) → dispersion on [0,1]

    Args:
        preprocessed_path: Path to hyperparams_preprocessed.npz file
        plots_dir: Directory to save plots
        ce_direct_prior: Optional dict with per-quantity {lo, hi} bounds.
                        If provided, overrides CE_HYPER_TRANSFORMS.
        config: Optional dict with keys hyper_chain_subsample, hyper_params_per_page, hyper_row_height
    """
    if config is None:
        config = {}
    preprocessed_path = Path(preprocessed_path)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")

    # Build transforms from ce_direct_prior if provided, else use defaults
    if ce_direct_prior is not None:
        qty_keys = ['ce1', 'ce2', 'ce3', 'pchoice']
        transforms = []
        for key in qty_keys:
            spec = ce_direct_prior[key]
            transforms.append({'lo': spec['lo'], 'hi': spec['hi']})  # mu
            transforms.append({'lo': 0, 'hi': 1})                    # psi
    else:
        transforms = CE_HYPER_TRANSFORMS

    print(f"  Loading preprocessed CE hyperparameters from {preprocessed_path}...")
    data = np.load(preprocessed_path)

    iterations = data['iterations']
    batch_boundaries = list(data['batch_boundaries'])

    # Reconstruct hyperparameter array from CE-specific state names
    first_state = HYPERPARAM_INFO_CE[0][0]
    n_samples, n_chains = data[first_state].shape
    n_hyper = len(HYPERPARAM_INFO_CE)

    hyper_history = np.zeros((n_samples, n_chains, n_hyper), dtype=np.float64)
    for param_idx, (state_name, _) in enumerate(HYPERPARAM_INFO_CE):
        logit_samples = data[state_name]
        # Transform from logit scale to natural scale: lo + (hi-lo)*sigmoid(logit)
        tfm = transforms[param_idx]
        sigmoid_samples = 1.0 / (1.0 + np.exp(-logit_samples))
        hyper_history[:, :, param_idx] = (
            tfm['lo'] + (tfm['hi'] - tfm['lo']) * sigmoid_samples
        )

    print(f"  Shape: ({n_samples}, {n_chains}, {n_hyper})")

    # Plot using shared function with CE-specific labels (natural scale)
    print("  Generating CE hyperparameter plots...")
    hyper_path = plots_dir / 'rdu_bhm_ce_hyperparams.pdf'
    _plot_hyperparameters_direct(hyper_history, iterations, batch_boundaries, hyper_path,
                                  chain_subsample=config.get('hyper_chain_subsample', 100),
                                  params_per_page=config.get('hyper_params_per_page', 5),
                                  row_height=config.get('hyper_row_height', 2.8),
                                  hyperparam_info=HYPERPARAM_INFO_CE)
    print(f"  Saved: {hyper_path}")


def _plot_hyperparameters_direct(hyper_history, iterations, batch_boundaries, output_path,
                                  chain_subsample=100,
                                  params_per_page=5, row_height=2.8,
                                  hyperparam_info=None):
    """
    Plot hyperparameters from a direct hyperparameter array (no offset calculation needed).

    Args:
        hyper_history: Array of shape (n_samples, n_chains, n_hyper) with hyperparams at indices 0, 1, ...
        iterations: Iteration numbers
        batch_boundaries: Run boundaries
        output_path: Path to save PDF
        chain_subsample: Max chains to show in traces (default 100)
        params_per_page: Hyperparams per PDF page (default 5)
        row_height: Height per row in inches (default 2.8)
        hyperparam_info: List of (state_name, latex_label) tuples (default: HYPERPARAM_INFO_CE)
    """
    if hyperparam_info is None:
        hyperparam_info = HYPERPARAM_INFO_CE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_chains, n_hyper = hyper_history.shape
    burn_idx = n_samples // 2

    # Subsample chains for traces
    n_chains_plot = min(chain_subsample, n_chains)
    chain_indices = np.linspace(0, n_chains - 1, n_chains_plot, dtype=int)
    chain_colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_chains_plot)))

    with PdfPages(output_path) as pdf:
        n_params = len(hyperparam_info)

        for page_start in range(0, n_params, params_per_page):
            page_end = min(page_start + params_per_page, n_params)
            n_on_page = page_end - page_start

            fig, axes = plt.subplots(n_on_page, 2, figsize=(14, row_height * n_on_page))
            if n_on_page == 1:
                axes = axes.reshape(1, 2)

            for i, param_idx in enumerate(range(page_start, page_end)):
                state_name, latex_label = hyperparam_info[param_idx]
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

                ax_density.set_xlabel(latex_label)
                ax_density.set_title(f'{state_name} - Density')
                ax_density.legend(loc='upper right', fontsize=7)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
