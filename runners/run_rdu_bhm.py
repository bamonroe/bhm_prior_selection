"""
Runner script for RDU BHM sampling.

This script runs the RDU Bayesian Hierarchical Model sampler using bamcmc.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from code.clean_data import (
    load_raw_data,
    parse_ra_data,
    make_bamcmc_data_indexed,
)
from code.posteriors.rdu_bhm import MODEL_ID, NPARAMS, N_HYPER
from code.prior_selection import (
    rdu_power_prior_selection,
    derive_hyperpriors,
    print_hyperprior_summary,
    verify_hyperpriors,
)


def load_data():
    """Load and prepare data for sampling."""
    print("Loading data...")
    ra_rows = load_raw_data()
    data = parse_ra_data(ra_rows)
    indexed_data = make_bamcmc_data_indexed(data)

    n_subjects = indexed_data['static'][0]
    n_choices = indexed_data['static'][1]
    n_outcomes = indexed_data['static'][2]
    n_probs = indexed_data['static'][3]

    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices}")
    print(f"  Unique outcomes: {n_outcomes}")
    print(f"  Unique cumulative probs: {n_probs}")
    print(f"  Total parameters: {n_subjects * NPARAMS + N_HYPER}")

    return indexed_data


def run_test_sampling(indexed_data, num_iter=100, num_collect=50):
    """
    Run a short test sampling to verify the model works.

    This uses a small number of chains and iterations for quick testing.
    """
    from bamcmc import rmcmc_single

    n_subjects = indexed_data['static'][0]

    # Small test configuration
    mcmc_config = {
        'posterior_id': MODEL_ID,
        'num_chains_a': 8,
        'num_chains_b': 8,
        'burn_iter': num_iter,
        'num_collect': num_collect,
        'thin_iteration': 1,
        'rng_seed': 42,
    }

    print(f"\nTest Sampling Configuration:")
    print(f"  Posterior: {MODEL_ID}")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Burn-in iterations: {mcmc_config['burn_iter']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Parameters: {n_subjects * NPARAMS + N_HYPER}")
    print(f"    - Subject parameters: {n_subjects * NPARAMS}")
    print(f"    - Hyperparameters: {N_HYPER}")

    print("\nRunning test sampling...")
    results, checkpoint = rmcmc_single(
        mcmc_config,
        indexed_data,
        calculate_rhat=True,
    )

    print("\nResults:")
    print(f"  History shape: {results['history'].shape}")
    print(f"  Final iteration: {results['iterations'][-1]}")

    if 'diagnostics' in results and results['diagnostics']:
        diag = results['diagnostics']
        if 'rhat' in diag:
            rhat = diag['rhat']
            print(f"\n  R-hat summary:")
            print(f"    Min: {np.min(rhat):.3f}")
            print(f"    Max: {np.max(rhat):.3f}")
            print(f"    Mean: {np.mean(rhat):.3f}")
            print(f"    % > 1.1: {100 * np.mean(rhat > 1.1):.1f}%")

    return results, checkpoint


def run_full_sampling(indexed_data, output_dir, num_runs=5, num_collect=1000):
    """
    Run full sampling with multiple runs.

    Uses the rmcmc function for automatic checkpoint management.
    """
    from bamcmc import rmcmc

    n_subjects = indexed_data['static'][0]

    # Full sampling configuration following CLAUDE.md guidelines
    # Chains should be at least n*(n+1)/2 for block of size n
    # Largest block is 6 (hyperparameters), so minimum is 21
    # Using 100 chains per group for good mixing

    mcmc_config = {
        'posterior_id': MODEL_ID,
        'num_chains_a': 100,
        'num_chains_b': 100,
        'burn_iter': 0,  # Keep full history, drop burn-in post-hoc
        'num_collect': num_collect,
        'thin_iteration': 10,
        'rng_seed': 42,
    }

    print(f"\nFull Sampling Configuration:")
    print(f"  Posterior: {MODEL_ID}")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Thinning: {mcmc_config['thin_iteration']}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Output directory: {output_dir}")

    # Run schedule: reset first run, then resume
    run_schedule = [("reset", 1), ("resume", num_runs - 1)]

    print(f"\nRunning MCMC with schedule: {run_schedule}")
    summary = rmcmc(
        mcmc_config,
        indexed_data,
        output_dir=output_dir,
        run_schedule=run_schedule,
        calculate_rhat=True,
        burn_in_fresh=True,
    )

    print(f"\nSampling complete!")
    print(f"  Total runs: {summary['total_runs_completed']}")
    print(f"  Final iteration: {summary['final_iteration']}")
    print(f"  History files: {len(summary['history_files'])}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run RDU BHM sampling")
    parser.add_argument(
        '--mode', choices=['test', 'full'], default='test',
        help='Sampling mode: test (quick) or full (production)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/output/rdu_bhm',
        help='Output directory for full sampling'
    )
    parser.add_argument(
        '--num-runs', type=int, default=5,
        help='Number of sampling runs for full mode'
    )
    parser.add_argument(
        '--num-collect', type=int, default=1000,
        help='Collection iterations per run'
    )
    parser.add_argument(
        '--show-priors', action='store_true',
        help='Show prior selection and derived hyperpriors'
    )
    args = parser.parse_args()

    # Show prior selection if requested
    if args.show_priors:
        print("\n" + "=" * 60)
        print("Prior Selection and Hyperprior Derivation")
        print("=" * 60)
        ps = rdu_power_prior_selection()
        print(ps.description)
        hp = derive_hyperpriors(ps, rho=0.3, alpha_0=6.0)
        print_hyperprior_summary(hp)
        verify_hyperpriors(hp)
        print()

    # Load data
    indexed_data = load_data()

    if args.mode == 'test':
        results, checkpoint = run_test_sampling(indexed_data)
    else:
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = run_full_sampling(
            indexed_data,
            str(output_dir),
            num_runs=args.num_runs,
            num_collect=args.num_collect,
        )


if __name__ == "__main__":
    main()
