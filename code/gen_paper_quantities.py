"""
Extract posterior summaries for the paper.

This module processes MCMC output and generates summary statistics
that can be loaded in the paper's LaTeX.
"""

import numpy as np
from pathlib import Path

from bamcmc import combine_batch_histories, apply_burnin, compute_rhat_from_history
from posteriors.rdu_bhm import NPARAMS, N_HYPER


def gen_paper_quantities(history_dir, output_path=None):
    """
    Extract posterior summaries from MCMC output.

    Args:
        history_dir: Directory containing history_*.npz files
        output_path: Path to save NPZ file (default: data/output/paper_quantities.npz)
    """
    history_dir = Path(history_dir)

    # Find all history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"Found {len(history_files)} history files")

    # Combine histories
    print("Combining histories...")
    history_paths = [str(f) for f in history_files]
    combined = combine_batch_histories(history_paths)

    history = combined['history']
    iterations = combined['iterations']

    print(f"Combined history shape: {history.shape}")
    print(f"Iteration range: [{iterations.min()}, {iterations.max()}]")

    # Apply burn-in (drop first half)
    burn_in_iter = iterations.max() // 2
    print(f"Applying burn-in at iteration {burn_in_iter}...")
    history_post, iter_post, _ = apply_burnin(history, iterations, min_iteration=burn_in_iter)

    print(f"Post burn-in shape: {history_post.shape}")

    # Compute R-hat
    K = 20  # Superchains
    M = history_post.shape[1] // K
    print(f"Computing R-hat with K={K}, M={M}...")
    rhat = compute_rhat_from_history(history_post, K, M)

    # Get number of subjects from parameter count
    n_params = history_post.shape[2]
    # history has params + generated quantities
    # params = n_subjects * NPARAMS + N_HYPER
    # gq = n_subjects * 3 + 6
    # total = params + gq = n_subjects * (NPARAMS + 3) + N_HYPER + 6
    n_subjects = (n_params - N_HYPER - 6) // (NPARAMS + 3)
    first_hyper = n_subjects * NPARAMS

    # Extract posterior means
    flat_samples = history_post.reshape(-1, n_params)
    posterior_means = flat_samples.mean(axis=0)
    posterior_stds = flat_samples.std(axis=0)

    # Hyperparameter posteriors (raw state values)
    hyper_means = posterior_means[first_hyper:first_hyper + N_HYPER]
    hyper_stds = posterior_stds[first_hyper:first_hyper + N_HYPER]
    hyper_rhat = rhat[first_hyper:first_hyper + N_HYPER]

    print(f"\nHyperparameter Posterior Summary:")
    print(f"  r_mean:      {hyper_means[0]:.4f} ± {hyper_stds[0]:.4f} (R-hat: {hyper_rhat[0]:.3f})")
    print(f"  r_logsd:     {hyper_means[1]:.4f} ± {hyper_stds[1]:.4f} (R-hat: {hyper_rhat[1]:.3f})")
    print(f"  gamma_mean:  {hyper_means[2]:.4f} ± {hyper_stds[2]:.4f} (R-hat: {hyper_rhat[2]:.3f})")
    print(f"  gamma_logsd: {hyper_means[3]:.4f} ± {hyper_stds[3]:.4f} (R-hat: {hyper_rhat[3]:.3f})")
    print(f"  mu_mean:     {hyper_means[4]:.4f} ± {hyper_stds[4]:.4f} (R-hat: {hyper_rhat[4]:.3f})")
    print(f"  mu_logsd:    {hyper_means[5]:.4f} ± {hyper_stds[5]:.4f} (R-hat: {hyper_rhat[5]:.3f})")

    # R-hat summary
    print(f"\nR-hat Summary:")
    print(f"  Max: {rhat.max():.4f}")
    print(f"  % > 1.1: {100 * np.mean(rhat > 1.1):.1f}%")

    # Default output path
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'paper_quantities.npz'
    output_path = Path(output_path)

    np.savez(
        output_path,
        hyper_means=hyper_means,
        hyper_stds=hyper_stds,
        hyper_rhat=hyper_rhat,
        rhat_max=rhat.max(),
        rhat_pct_above_1_1=np.mean(rhat > 1.1),
        n_subjects=n_subjects,
        n_samples=flat_samples.shape[0],
    )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper quantities from MCMC output")
    parser.add_argument('--history-dir', type=str, default='data/output/rdu_bhm/history/full')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    gen_paper_quantities(args.history_dir, args.output)
