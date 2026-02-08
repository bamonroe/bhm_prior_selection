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
from scipy import stats
from pathlib import Path
from multiprocessing import Pool

from bamcmc import combine_batch_histories
from posteriors.rdu_bhm import NPARAMS, N_HYPER


# Hyperparameter display info: (state_name, latex_label)
HYPERPARAM_INFO = [
    ('r_mean', r'$\mu_r$'),
    ('r_logsd', r'$\ln\sigma_r$'),
    ('phi_mean', r'$\mu_{\phi}$ (log scale)'),
    ('phi_logsd', r'$\ln\sigma_{\phi}$'),
    ('eta_mean', r'$\mu_{\eta}$ (log scale)'),
    ('eta_logsd', r'$\ln\sigma_{\eta}$'),
    ('mu_mean', r'$\mu_{\mu}$ (log scale)'),
    ('mu_logsd', r'$\ln\sigma_{\mu}$'),
]

# Subject parameter display info: (name, latex_label)
SUBJECT_PARAM_INFO = [
    ('r', r'$r$ (CRRA)'),
    ('phi', r'$\phi$ (Prelec curvature)'),
    ('eta', r'$\eta$ (Prelec elevation)'),
    ('mu', r'$\mu$ (noise)'),
]

# Color schemes
COLOR_DATA = 'steelblue'
COLOR_MEDIAN = 'red'
COLOR_REFERENCE = 'black'
COLOR_SPAGHETTI = 'steelblue'
RUN_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

# Number of posterior draws for spaghetti plots
N_SPAGHETTI_SAMPLES = 100

# Dataset outcome range (DKK from Danish experiment)
X_MIN = 50.0
X_MAX = 4500.0


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
    history, iterations, likelihoods, metadata = combine_batch_histories(history_files)
    print(f"  Combined shape: {history.shape}")

    n_samples, n_chains, n_params = history.shape

    # Auto-detect n_subjects
    if n_subjects is None:
        # History contains:
        # - Raw state: N_HYPER hyperparams + n_subjects * NPARAMS subject epsilon params
        # - Generated quantities: n_subjects * 4 (r, phi, eta, mu) + 8 (population params)
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

        # Generated quantities layout: r[0:n], phi[n:2n], eta[2n:3n], mu[3n:4n], then pop params
        subject_data = {
            'r': history[:, :, first_gq + sid],
            'phi': history[:, :, first_gq + n_subjects + sid],
            'eta': history[:, :, first_gq + 2 * n_subjects + sid],
            'mu': history[:, :, first_gq + 3 * n_subjects + sid],
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


# =============================================================================
# STAGE 2: PARALLEL PLOTTING - Workers load from files
# =============================================================================

def _plot_single_subject(args):
    """
    Worker function to plot a single subject.

    Args:
        args: tuple of (subject_id, params_dir, output_dir, iterations, batch_boundaries)
    """
    subject_id, params_dir, output_dir, iterations, batch_boundaries = args

    params_dir = Path(params_dir)
    output_dir = Path(output_dir)

    # Load subject data from file
    params_path = params_dir / f'subject_{subject_id:03d}_params.npz'
    data = np.load(params_path)
    params = {k: data[k] for k in ['r', 'phi', 'eta', 'mu']}

    # Generate PDF
    pdf_path = output_dir / f'subject_{subject_id:03d}.pdf'

    with PdfPages(pdf_path) as pdf:
        # Page 1: Trace plots
        _plot_subject_traces(params, iterations, batch_boundaries, subject_id, pdf)

        # Page 2: Density plots (last half)
        _plot_subject_densities(params, subject_id, pdf)

        # Page 3: Utility spaghetti (last half)
        _plot_subject_utility(params, subject_id, pdf)

    return subject_id


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

        # Median line
        median_val = np.median(param_samples)
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
                           n_workers=1, max_subjects=None):
    """
    Run parallel plotting with a pool of workers.

    Args:
        params_dir: Directory with per-subject param files
        output_dir: Directory for output PDFs
        iterations: Iteration numbers array
        batch_boundaries: MCMC run boundary indices
        n_workers: Number of parallel workers
        max_subjects: Limit number of subjects (for testing)
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

    # Build task list - workers receive paths, not data
    tasks = [
        (sid, str(params_dir), str(output_dir), iterations_list, boundaries_list)
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

def plot_hyperparameters(history, iterations, batch_boundaries, output_path):
    """
    Generate hyperparameter diagnostic PDF.

    Layout: 2-column (traces left, densities right), up to 5 params per page.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_chains, _ = history.shape
    burn_idx = n_samples // 2

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

                param_samples = history[:, :, param_idx]

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

                median_val = np.median(param_samples)
                ax_trace.axhline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=1.5)
                ax_trace.set_ylabel(latex_label)
                ax_trace.set_xlabel('Iteration')
                ax_trace.set_title(f'{state_name} - Trace')

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

                ax_density.axvline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=2)
                ax_density.legend(loc='upper right', fontsize=8)
                ax_density.set_xlabel(latex_label)
                ax_density.set_ylabel('Density')
                ax_density.set_title(f'{state_name} - Posterior')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def diagnostic_plots(history_dir, plots_dir, plot_subjects=True, max_subjects=None, n_workers=1):
    """
    Generate diagnostic plots from MCMC output.

    Two-stage approach for memory efficiency:
    1. Preprocess: Extract per-subject data to individual files
    2. Plot: Workers load individual files (not full history)

    Args:
        history_dir: Directory containing history_*.npz files
        plots_dir: Directory to save plots
        plot_subjects: Whether to generate per-subject PDFs
        max_subjects: Limit subjects for testing
        n_workers: Number of parallel workers for subject plots
    """
    history_dir = Path(history_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Check for history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"  Found {len(history_files)} history files")

    # Load history for hyperparameter plots
    print("  Loading history for hyperparameters...")
    history, iterations, likelihoods, metadata = combine_batch_histories(history_files)
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
    plot_hyperparameters(history, iterations, batch_boundaries, hyper_path)

    # Subject plots (two-stage)
    if plot_subjects:
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
            n_workers=n_workers, max_subjects=max_subjects
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
    print(f"  n_samples: {data['n_samples']}")
    print(f"  R-hat max: {data['rhat_max']:.4f}")

    # TODO: Add specific paper plots as needed
    print("  Paper plots not yet implemented")


# =============================================================================
# PRIOR DISTRIBUTION PLOTS
# =============================================================================

# Parameter display info for prior plots
PRIOR_PARAM_INFO = [
    ('r', r'$r$ (CRRA)', False),           # (key, label, log_scale)
    ('phi_raw', r'$\ln\phi$', True),
    ('eta_raw', r'$\ln\eta$', True),
    ('lambda_raw', r'$\ln\lambda$', True),
]


def plot_direct_prior(direct_prior, output_path, n_samples=10000, seed=42):
    """
    Generate 4-panel plot of the direct prior distribution.

    Samples from N(mu, sigma2) for each parameter and plots the distribution.

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

    for i, (key, label, is_log) in enumerate(PRIOR_PARAM_INFO):
        ax = axes[i]
        mu = direct_prior[key]['mu']
        sigma = np.sqrt(direct_prior[key]['sigma2'])

        # Sample from direct prior
        samples = np.random.normal(mu, sigma, n_samples)

        # Plot histogram and KDE
        ax.hist(samples, bins=50, density=True, alpha=0.6, color=COLOR_DATA,
                edgecolor='white', linewidth=0.5)

        try:
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(samples.min(), samples.max(), 200)
            ax.plot(x_range, kde(x_range), color=COLOR_DATA, lw=2)
        except Exception:
            pass

        # Mark mean
        ax.axvline(mu, color=COLOR_MEDIAN, linestyle='--', lw=2,
                   label=f'Mean: {mu:.2f}')

        # Add ±2σ range
        ax.axvline(mu - 2*sigma, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(mu + 2*sigma, color='gray', linestyle=':', alpha=0.7)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label.split("$")[1].split("$")[0] if "$" in label else key}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

        # Add text annotation
        ax.text(0.02, 0.98, f'$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$',
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

    for i, (key, label, is_log) in enumerate(PRIOR_PARAM_INFO):
        ax = axes[i]

        # Get direct prior params
        mu_ps = direct_prior[key]['mu']
        var_ps = direct_prior[key]['sigma2']

        # Derive hyperpriors
        tau_0_sq = rho * var_ps
        expected_sigma_sq = (1 - rho) * var_ps
        beta_0 = expected_sigma_sq * (alpha_0 - 1)

        # Sample from BHM
        samples = np.zeros(n_samples)
        for j in range(n_samples):
            # Sample population mean
            mu = np.random.normal(mu_ps, np.sqrt(tau_0_sq))

            # Sample population variance (Inverse-Gamma)
            sigma_sq = 1.0 / np.random.gamma(alpha_0, 1.0 / beta_0)

            # Sample individual
            samples[j] = np.random.normal(mu, np.sqrt(sigma_sq))

        # Plot histogram and KDE
        ax.hist(samples, bins=50, density=True, alpha=0.6, color=COLOR_DATA,
                edgecolor='white', linewidth=0.5)

        try:
            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(samples.min(), samples.max(), 200)
            ax.plot(x_range, kde(x_range), color=COLOR_DATA, lw=2)
        except Exception:
            pass

        # Mark direct prior mean
        ax.axvline(mu_ps, color=COLOR_MEDIAN, linestyle='--', lw=2,
                   label=f'Prior mean: {mu_ps:.2f}')

        # Add ±2σ range from direct prior
        sigma_ps = np.sqrt(var_ps)
        ax.axvline(mu_ps - 2*sigma_ps, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(mu_ps + 2*sigma_ps, color='gray', linestyle=':', alpha=0.7)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label.split("$")[1].split("$")[0] if "$" in label else key}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

        # Add text with sample stats
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        ax.text(0.02, 0.98, f'Sample: $\\mu={sample_mean:.2f}$, $\\sigma={sample_std:.2f}$\n'
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

    for i, (key, label, is_log) in enumerate(PRIOR_PARAM_INFO):
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
        ax.set_title(f'{label.split("$")[1].split("$")[0] if "$" in label else key}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
