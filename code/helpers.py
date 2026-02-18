"""
Helper functions for main.py pipeline orchestration.
"""

import sys
from pathlib import Path

import numpy as np

# Setup paths (same as main.py)
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'

# Output file paths
OUTPUTS = {
    'clean_data': DATA_DIR / 'input' / 'dbar_clean.npz',
    'paper_data': DATA_DIR / 'output' / 'paper_data.npz',
    'prior_params': DATA_DIR / 'output' / 'prior_params.npz',
    'paper_quantities': DATA_DIR / 'output' / 'paper_quantities.npz',
    'prior_predictive_ce': DATA_DIR / 'output' / 'prior_predictive_ce.npz',
    'posterior_predictive_ce_info': DATA_DIR / 'output' / 'posterior_predictive_ce_informative.npz',
    'posterior_predictive_ce_uninf': DATA_DIR / 'output' / 'posterior_predictive_ce_uninformative.npz',
    'posterior_ce_comparison': DATA_DIR / 'output' / 'posterior_ce_comparison.npz',
}


def check_dependency(step_name, required_outputs, strict=True):
    """Check that required outputs from previous steps exist."""
    missing = []
    for output_key in required_outputs:
        output_path = OUTPUTS[output_key]
        if not output_path.exists():
            missing.append((output_key, output_path))

    if missing:
        print(f"\nERROR: Missing dependencies for '{step_name}':")
        for key, path in missing:
            print(f"  - {key}: {path}")
        print(f"\nRun the required step(s) first.")
        if strict:
            sys.exit(1)
        return False
    return True


def get_mcmc_dir(model='rdu_bhm'):
    """Get MCMC output directory for a model."""
    return DATA_DIR / 'output' / model


def get_plots_dir(model='rdu_bhm'):
    """Get plots directory for a model."""
    return PLOTS_DIR / model


def load_and_filter_history(history_files):
    """Load history files; filter to beta=1 samples if tempering was used.

    Args:
        history_files: List of history file paths

    Returns:
        tuple: (history, iterations, likelihoods, metadata)
            If tempering was used, history contains only beta=1 samples and
            iterations is reset to a simple arange. Chains with too few beta=1
            samples are dropped to avoid truncating all chains to near-zero.
    """
    from bamcmc import combine_batch_histories, filter_beta1_samples

    history, iterations, likelihoods, metadata = combine_batch_histories(history_files)

    temp_history = metadata.get('temperature_history') if metadata else None
    if temp_history is not None:
        n_samples, n_chains, n_params = history.shape

        # Compute per-chain beta=1 sample counts
        mask = (temp_history == 0)
        counts = np.sum(mask, axis=0)

        # Drop chains with fewer than half the median count
        median_count = np.median(counts[counts > 0]) if np.any(counts > 0) else 0
        threshold = max(1, int(median_count * 0.5))
        keep = counts >= threshold
        n_dropped = np.sum(~keep)

        if n_dropped > 0:
            print(f"  Dropping {n_dropped}/{n_chains} chains with <{threshold} "
                  f"beta=1 samples (median={median_count:.0f})")

        history_keep = history[:, keep, :]
        temp_keep = temp_history[:, keep]
        filtered, kept_counts = filter_beta1_samples(history_keep, temp_keep)

        history = filtered
        iterations = np.arange(history.shape[0])
        print(f"  Filtered to beta=1: {history.shape} "
              f"(min {kept_counts.min()} samples/chain)")

    return history, iterations, likelihoods, metadata
