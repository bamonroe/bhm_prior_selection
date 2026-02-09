"""
MCMC sampling for RDU BHM model.

This module handles running the bamcmc sampler and managing output.
"""

from pathlib import Path

from bamcmc import rmcmc
from clean_data import load_raw_data, parse_ra_data, make_bamcmc_data_indexed
from posteriors.rdu_bhm import MODEL_ID, NPARAMS, N_HYPER


# Default configuration
DEFAULT_CONFIG = {
    'num_chains_a': 100,
    'num_chains_b': 100,
    'num_collect': 1000,
    'thin_iteration': 10,
    'chunk_size': 100,  # Iterations per compiled chunk (1-10000)
    'num_runs': 5,
    'rng_seed': 42,
}


def run_sampling(output_dir, num_runs=None, num_collect=None, config=None):
    """
    Run MCMC sampling with bamcmc.

    Args:
        output_dir: Directory for output files
        num_runs: Number of MCMC runs (overrides config)
        num_collect: Collection iterations per run (overrides config)
        config: Configuration dict (uses DEFAULT_CONFIG if None)

    Returns:
        Summary dict from rmcmc
    """
    cfg = config or DEFAULT_CONFIG

    # Load data
    print("Loading data...")
    ra_rows = load_raw_data()
    data = parse_ra_data(ra_rows)
    indexed_data = make_bamcmc_data_indexed(data)

    n_subjects = indexed_data['static'][0]
    print(f"  Subjects: {n_subjects}")
    print(f"  Total parameters: {n_subjects * NPARAMS + N_HYPER}")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MCMC configuration
    mcmc_config = {
        'posterior_id': MODEL_ID,
        'num_chains_a': cfg['num_chains_a'],
        'num_chains_b': cfg['num_chains_b'],
        'burn_iter': 0,  # Keep full history, drop burn-in post-hoc
        'num_collect': num_collect or cfg['num_collect'],
        'thin_iteration': cfg['thin_iteration'],
        'chunk_size': cfg.get('chunk_size', 100),
        'rng_seed': cfg['rng_seed'],
    }

    actual_runs = num_runs or cfg['num_runs']

    print(f"\nSampling Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Thinning: {mcmc_config['thin_iteration']}")
    print(f"  Number of runs: {actual_runs}")
    print(f"  Output: {output_dir}")

    # Run schedule: first run resets, subsequent runs resume
    if actual_runs > 1:
        run_schedule = [("reset", 1), ("resume", actual_runs - 1)]
    else:
        run_schedule = [("reset", 1)]

    print(f"\nStarting MCMC with schedule: {run_schedule}")
    summary = rmcmc(
        mcmc_config,
        indexed_data,
        output_dir=str(output_dir),
        run_schedule=run_schedule,
        calculate_rhat=True,
        burn_in_fresh=True,
    )

    print(f"\nSampling complete!")
    print(f"  Total runs: {summary['total_runs_completed']}")
    print(f"  Final iteration: {summary['final_iteration']}")
    print(f"  History files: {len(summary['history_files'])}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RDU BHM sampling")
    parser.add_argument('--output-dir', type=str, default='data/output/rdu_bhm')
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--num-collect', type=int, default=1000)
    args = parser.parse_args()

    run_sampling(args.output_dir, args.num_runs, args.num_collect)
