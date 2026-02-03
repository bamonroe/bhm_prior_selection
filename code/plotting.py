"""
Plotting functions for diagnostics and paper figures.

Plot styles follow the project template conventions:
- Hyperparameter plots: 2-column (traces left, densities right), 5 params per page
- Subject plots: Multi-page PDF with traces, densities, and utility spaghetti
- Color scheme: steelblue for data, red for median, tab20 for chains
- Output: Multi-page PDFs in organized directory structure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from pathlib import Path

from bamcmc import combine_batch_histories
from posteriors.rdu_bhm import NPARAMS, N_HYPER
from posteriors.common_fns import crra_ufn


# Hyperparameter display info: (state_name, latex_label)
HYPERPARAM_INFO = [
    ('r_mean', r'$\mu_r$'),
    ('r_logsd', r'$\ln\sigma_r$'),
    ('gamma_mean', r'$\mu_{\gamma}$ (log scale)'),
    ('gamma_logsd', r'$\ln\sigma_{\gamma}$'),
    ('mu_mean', r'$\mu_{\mu}$ (log scale)'),
    ('mu_logsd', r'$\ln\sigma_{\mu}$'),
]

# Subject parameter display info: (name, latex_label)
SUBJECT_PARAM_INFO = [
    ('r', r'$r$ (CRRA)'),
    ('gamma', r'$\gamma$ (prob. weight)'),
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


def diagnostic_plots(history_dir, plots_dir, plot_subjects=True, max_subjects=None):
    """
    Generate diagnostic plots from MCMC output.

    Creates:
    - Hyperparameter diagnostic PDF
    - Per-subject PDFs in subjects/ subdirectory (if plot_subjects=True)

    Args:
        history_dir: Directory containing history_*.npz files
        plots_dir: Directory to save plots
        plot_subjects: Whether to generate per-subject plots
        max_subjects: Maximum number of subjects to plot (None = all)
    """
    history_dir = Path(history_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load combined history
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"Loading {len(history_files)} history files...")
    history_paths = [str(f) for f in history_files]
    history, iterations, likelihoods, metadata = combine_batch_histories(history_paths)

    n_samples, n_chains, n_params = history.shape
    n_subjects = (n_params - N_HYPER - 6) // (NPARAMS + 3)
    first_hyper = n_subjects * NPARAMS
    first_gq = first_hyper + N_HYPER  # Generated quantities start after hyperparams

    # Get run boundaries from metadata
    batch_boundaries = metadata.get('batch_boundaries', [0])

    print(f"  Samples: {n_samples}, Chains: {n_chains}, Params: {n_params}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Hyperparameters at indices {first_hyper}:{first_hyper + N_HYPER}")
    print(f"  Generated quantities start at index {first_gq}")

    # Generate hyperparameter diagnostic PDF
    _plot_hyperparameters(
        history, iterations, first_hyper, batch_boundaries,
        plots_dir / 'rdu_bhm_hyperparams.pdf'
    )

    # Generate per-subject plots
    if plot_subjects:
        subjects_dir = plots_dir / 'subjects'
        subjects_dir.mkdir(parents=True, exist_ok=True)

        n_to_plot = n_subjects if max_subjects is None else min(max_subjects, n_subjects)
        print(f"\nGenerating subject plots ({n_to_plot} subjects)...")

        for subj_idx in range(n_to_plot):
            _plot_subject(
                history, iterations, subj_idx, n_subjects, first_gq,
                batch_boundaries, subjects_dir / f'subject_{subj_idx:03d}.pdf'
            )
            if (subj_idx + 1) % 50 == 0 or subj_idx == n_to_plot - 1:
                print(f"  Completed {subj_idx + 1}/{n_to_plot} subjects")

    print(f"\nDiagnostic plots saved to: {plots_dir}")


def _plot_hyperparameters(history, iterations, first_hyper, batch_boundaries, output_path):
    """
    Generate hyperparameter diagnostic PDF.

    Layout: 2 columns (traces left, densities right), up to 5 params per page.
    """
    n_samples, n_chains, _ = history.shape
    n_hyper = len(HYPERPARAM_INFO)
    params_per_page = 5

    # Subsample chains for visibility (max 100)
    chain_subsample = min(100, n_chains)
    chain_indices = np.linspace(0, n_chains - 1, chain_subsample, dtype=int)

    # Colors for chains
    chain_colors = plt.cm.tab20(np.linspace(0, 1, min(20, chain_subsample)))

    print(f"Generating hyperparameter diagnostics ({n_hyper} params)...")

    with PdfPages(output_path) as pdf:
        # Process in pages of 5
        for page_start in range(0, n_hyper, params_per_page):
            page_end = min(page_start + params_per_page, n_hyper)
            n_on_page = page_end - page_start

            fig, axes = plt.subplots(n_on_page, 2, figsize=(14, 2.8 * n_on_page))
            if n_on_page == 1:
                axes = axes.reshape(1, 2)

            for i, param_idx_on_page in enumerate(range(page_start, page_end)):
                state_name, latex_label = HYPERPARAM_INFO[param_idx_on_page]
                param_idx = first_hyper + param_idx_on_page

                ax_trace = axes[i, 0]
                ax_density = axes[i, 1]

                # Extract samples for this parameter
                param_samples = history[:, :, param_idx]

                # --- Trace plot (left column) ---
                for j, chain_idx in enumerate(chain_indices):
                    color = chain_colors[j % len(chain_colors)]
                    ax_trace.plot(
                        iterations, param_samples[:, chain_idx],
                        color=color, alpha=0.4, lw=0.3
                    )

                # Add run boundary lines
                for boundary in batch_boundaries[1:]:
                    if boundary < len(iterations):
                        ax_trace.axvline(
                            iterations[boundary], color='gray',
                            linestyle=':', alpha=0.7, lw=1
                        )

                # Median line
                median_val = np.median(param_samples)
                ax_trace.axhline(median_val, color=COLOR_MEDIAN, linestyle='--', lw=1.5, alpha=0.8)

                ax_trace.set_ylabel(latex_label)
                ax_trace.set_xlabel('Iteration')
                ax_trace.set_title(f'{state_name} - Trace')

                # --- Density plot (right column) ---
                # Flatten samples for density estimation
                flat_samples = param_samples.flatten()

                # KDE
                try:
                    kde = stats.gaussian_kde(flat_samples)
                    x_range = np.linspace(flat_samples.min(), flat_samples.max(), 200)
                    ax_density.fill_between(
                        x_range, kde(x_range),
                        color=COLOR_DATA, alpha=0.6
                    )
                    ax_density.plot(x_range, kde(x_range), color=COLOR_DATA, lw=1.5)
                except Exception:
                    # Fallback to histogram if KDE fails
                    ax_density.hist(flat_samples, bins=50, density=True, color=COLOR_DATA, alpha=0.6)

                # Median line with value
                ax_density.axvline(
                    median_val, color=COLOR_MEDIAN, linestyle='--', lw=2,
                    label=f'Median: {median_val:.3f}'
                )
                ax_density.legend(loc='upper right', fontsize=9)

                ax_density.set_xlabel(latex_label)
                ax_density.set_ylabel('Density')
                ax_density.set_title(f'{state_name} - Posterior')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved: {output_path}")


def _plot_subject(history, iterations, subj_idx, n_subjects, first_gq, batch_boundaries, output_path):
    """
    Generate per-subject diagnostic PDF.

    Pages:
    1. Trace plots (3 params: r, gamma, mu)
    2. Density plots (3 params)
    3. Utility function spaghetti plot
    """
    n_samples, n_chains, _ = history.shape

    # Generated quantities for this subject: r, gamma, mu
    # Layout: first_gq + subj_idx * 3 + [0, 1, 2]
    gq_base = first_gq + subj_idx * 3

    # Subsample chains for visibility
    chain_subsample = min(50, n_chains)
    chain_indices = np.linspace(0, n_chains - 1, chain_subsample, dtype=int)
    chain_colors = plt.cm.tab20(np.linspace(0, 1, min(20, chain_subsample)))

    # Use second half as post burn-in for densities
    burn_idx = n_samples // 2

    with PdfPages(output_path) as pdf:
        # --- Page 1: Trace plots ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Subject {subj_idx} - Trace Plots', fontsize=14)

        for i, (name, label) in enumerate(SUBJECT_PARAM_INFO):
            ax = axes[i]
            param_idx = gq_base + i
            param_samples = history[:, :, param_idx]

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

        # --- Page 2: Density plots ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Subject {subj_idx} - Posterior Densities', fontsize=14)

        for i, (name, label) in enumerate(SUBJECT_PARAM_INFO):
            ax = axes[i]
            param_idx = gq_base + i
            # Use post burn-in samples
            flat_samples = history[burn_idx:, :, param_idx].flatten()

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

        # --- Page 3: Utility function spaghetti plot ---
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle(f'Subject {subj_idx} - CRRA Utility Function', fontsize=14)

        # Get r samples (post burn-in)
        r_samples = history[burn_idx:, :, gq_base].flatten()

        # Subsample for spaghetti plot
        n_draws = min(N_SPAGHETTI_SAMPLES, len(r_samples))
        draw_indices = np.random.choice(len(r_samples), n_draws, replace=False)

        # Domain for utility function (normalized to [0, 1] for display)
        x = np.linspace(0.01, 1.0, 100)

        # Plot spaghetti (posterior draws)
        for idx in draw_indices:
            r = r_samples[idx]
            # CRRA utility: u(x) = x^(1-r) for display purposes
            # This gives concave curves for r > 0 (risk averse)
            # and convex curves for r < 0 (risk loving)
            u = np.power(x, 1 - r)
            ax.plot(x, u, color=COLOR_SPAGHETTI, alpha=0.15, lw=0.8)

        # Median utility curve
        r_median = np.median(r_samples)
        u_median = np.power(x, 1 - r_median)
        ax.plot(x, u_median, color=COLOR_MEDIAN, lw=2.5, zorder=10,
                label=f'Median (r={r_median:.2f})')

        # Risk-neutral reference (r=0 => u(x) = x)
        ax.plot(x, x, color=COLOR_REFERENCE, linestyle='--', lw=1.5,
                label='Risk neutral (r=0)')

        ax.set_xlabel('Normalized Outcome')
        ax.set_ylabel('Utility')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


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
    # Examples:
    # - Prior vs posterior comparison
    # - Population distribution plots
    # - Individual heterogeneity visualization

    print("\nPaper plots: Add specific visualizations based on paper requirements")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument('--mode', choices=['diagnostic', 'paper'], required=True)
    parser.add_argument('--history-dir', type=str, default='data/output/rdu_bhm/history/full')
    parser.add_argument('--plots-dir', type=str, default='plots/rdu_bhm')
    parser.add_argument('--quantities', type=str, default='data/output/paper_quantities.npz')
    args = parser.parse_args()

    if args.mode == 'diagnostic':
        diagnostic_plots(args.history_dir, args.plots_dir)
    else:
        paper_plots(args.quantities, args.plots_dir)
