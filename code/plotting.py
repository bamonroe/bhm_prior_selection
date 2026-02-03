"""
Plotting functions for diagnostics and paper figures.

This module generates trace plots, posterior histograms, and publication-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bamcmc import combine_batch_histories
from posteriors.rdu_bhm import NPARAMS, N_HYPER


def diagnostic_plots(history_dir, plots_dir):
    """
    Generate diagnostic plots from MCMC output.

    Args:
        history_dir: Directory containing history_*.npz files
        plots_dir: Directory to save plots
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

    n_params = history.shape[2]
    n_subjects = (n_params - N_HYPER - 6) // (NPARAMS + 3)
    first_hyper = n_subjects * NPARAMS

    # Plot 1: Hyperparameter traces
    print("Generating hyperparameter trace plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    hyper_names = ['r_mean', 'r_logsd', 'gamma_mean', 'gamma_logsd', 'mu_mean', 'mu_logsd']

    for i, (ax, name) in enumerate(zip(axes.flat, hyper_names)):
        param_idx = first_hyper + i
        # Plot a subset of chains
        for chain_idx in range(0, min(20, history.shape[1]), 2):
            ax.plot(iterations, history[:, chain_idx, param_idx], alpha=0.5, lw=0.5)
        ax.set_title(name)
        ax.set_xlabel('Iteration')

    plt.tight_layout()
    trace_path = plots_dir / 'hyperparameter_traces.pdf'
    plt.savefig(trace_path)
    plt.close()
    print(f"  Saved: {trace_path}")

    # Plot 2: Hyperparameter posteriors
    print("Generating hyperparameter posterior plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Use second half as post burn-in
    burn_idx = len(iterations) // 2
    post_samples = history[burn_idx:].reshape(-1, n_params)

    for i, (ax, name) in enumerate(zip(axes.flat, hyper_names)):
        param_idx = first_hyper + i
        samples = post_samples[:, param_idx]
        ax.hist(samples, bins=50, density=True, alpha=0.7)
        ax.axvline(samples.mean(), color='r', linestyle='--', label=f'Mean: {samples.mean():.3f}')
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    post_path = plots_dir / 'hyperparameter_posteriors.pdf'
    plt.savefig(post_path)
    plt.close()
    print(f"  Saved: {post_path}")

    print(f"\nDiagnostic plots saved to: {plots_dir}")


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
