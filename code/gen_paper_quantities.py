"""
Extract posterior summaries for the paper.

This module processes MCMC output and generates summary statistics
that can be loaded in the paper's LaTeX via pythontex.
"""

import numpy as np
from pathlib import Path

from bamcmc import combine_batch_histories, apply_burnin, compute_rhat_from_history
from helpers import load_and_filter_history
from posteriors.rdu_bhm import NPARAMS, N_HYPER
from plotting import HYPERPARAM_INFO


def gen_paper_quantities(history_dir, output_path=None,
                         burnin_fraction=0.5, rhat_K=20):
    """
    Extract posterior summaries from MCMC output.

    Saves to NPZ file that can be loaded in LaTeX via pythontex.

    Args:
        history_dir: Directory containing history_*.npz files
        output_path: Path to save NPZ file (default: data/output/paper_quantities.npz)
        burnin_fraction: Fraction of iterations to discard as burn-in (default 0.5)
        rhat_K: Number of chain groups for R-hat computation (default 20)
    """
    history_dir = Path(history_dir)

    # Find all history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"Found {len(history_files)} history files")

    # Combine histories (filter to beta=1 if tempering was used)
    print("Combining histories...")
    history_paths = [str(f) for f in history_files]
    history, iterations, _, metadata = load_and_filter_history(history_paths)

    n_samples_total, n_chains, n_params = history.shape
    print(f"Combined history shape: {history.shape}")
    print(f"Iteration range: [{iterations.min()}, {iterations.max()}]")

    # Apply burn-in
    burn_in_iter = int(iterations.max() * burnin_fraction)
    print(f"Applying burn-in at iteration {burn_in_iter}...")
    history_post, iter_post, _ = apply_burnin(history, iterations, min_iteration=burn_in_iter)

    n_samples_post = history_post.shape[0]
    print(f"Post burn-in shape: {history_post.shape}")

    # Compute R-hat (adapt K to chain count after potential tempering filter)
    n_chains_post = history_post.shape[1]
    K = rhat_K
    while n_chains_post % K != 0 and K > 1:
        K -= 1
    M = n_chains_post // K
    print(f"Computing R-hat with K={K}, M={M}...")
    rhat = compute_rhat_from_history(history_post, K, M)

    # Get number of subjects from parameter count
    # history has params + generated quantities
    # params = n_subjects * NPARAMS + N_HYPER
    # gq = n_subjects * 4 + 8 (r, phi, eta, lambda per subject + 8 pop params)
    # total = params + gq = n_subjects * (NPARAMS + 4) + N_HYPER + 8
    n_subjects = (n_params - N_HYPER - 8) // (NPARAMS + 4)
    n_total_params = n_subjects * NPARAMS + N_HYPER
    first_hyper = n_subjects * NPARAMS

    # Extract hyperparameter samples
    hyper_samples = history_post[:, :, first_hyper:first_hyper + N_HYPER].reshape(-1, N_HYPER)

    # Compute statistics for each hyperparameter
    hyper_names = [info[0] for info in HYPERPARAM_INFO]  # e.g., 'r_mean', 'r_logsd', ...

    print(f"\nHyperparameter Posterior Summary:")
    save_dict = {
        'n_chains': n_chains,
        'n_samples_total': n_samples_total,
        'n_samples_post_burnin': n_samples_post,
        'n_subjects': n_subjects,
        'n_total_params': n_total_params,
    }

    for i, name in enumerate(hyper_names):
        samples = hyper_samples[:, i]
        mean = samples.mean()
        std = samples.std()
        ci_lo = np.percentile(samples, 2.5)
        ci_hi = np.percentile(samples, 97.5)
        r_hat = rhat[first_hyper + i]

        save_dict[f'{name}_mean'] = mean
        save_dict[f'{name}_std'] = std
        save_dict[f'{name}_ci_lo'] = ci_lo
        save_dict[f'{name}_ci_hi'] = ci_hi
        save_dict[f'{name}_rhat'] = r_hat

        print(f"  {name:15s}: {mean:7.3f} ± {std:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  R-hat: {r_hat:.3f}")

    # R-hat summary
    rhat_max = rhat.max()
    rhat_pct_above = 100 * np.mean(rhat > 1.1)
    save_dict['rhat_max'] = rhat_max
    save_dict['rhat_pct_above_1_1'] = rhat_pct_above

    print(f"\nR-hat Summary:")
    print(f"  Max: {rhat_max:.3f}")
    print(f"  % > 1.1: {rhat_pct_above:.1f}%")

    # Default output path
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'paper_quantities.npz'
    output_path = Path(output_path)

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper quantities from MCMC output")
    parser.add_argument('--history-dir', type=str, default='data/output/rdu_bhm/history/full')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    gen_paper_quantities(args.history_dir, args.output)
