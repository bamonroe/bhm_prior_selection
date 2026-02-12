"""
MCMC sampling for RDU BHM model.

This module handles running the bamcmc sampler and managing output.
"""

from pathlib import Path

from bamcmc import rmcmc
from clean_data import load_clean_data, make_bamcmc_data_indexed
from posteriors.rdu_bhm import MODEL_ID, NPARAMS, N_HYPER, get_hyperprior_array


# Default configuration
DEFAULT_CONFIG = {
    'num_chains_a': 100,
    'num_chains_b': 100,
    'num_collect': 1000,
    'thin_iteration': 10,
    'chunk_size': 100,  # Iterations per compiled chunk (1-10000)
    'rng_seed': 42,
}


def run_sampling(output_dir, reset_runs=None, resume_runs=None, num_collect=None,
                 config=None, prize_scale=1000.0):
    """
    Run MCMC sampling with bamcmc.

    Args:
        output_dir: Directory for output files
        reset_runs: Number of fresh/reset runs (default 1)
        resume_runs: Number of resume runs after reset runs (default 0)
        num_collect: Collection iterations per run (overrides config)
        config: Configuration dict (uses DEFAULT_CONFIG if None)
        prize_scale: Normalization factor for prizes

    Returns:
        Summary dict from rmcmc
    """
    cfg = config or DEFAULT_CONFIG

    # Load data
    print("Loading data...")
    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data, prize_scale=prize_scale)

    # Add hyperprior array to data["float"] so it's passed at runtime (not traced by JAX)
    # This allows different priors to be used without recompiling
    hyperprior_arr = get_hyperprior_array()
    indexed_data['float'] = indexed_data['float'] + (hyperprior_arr,)

    # Debug: print hyperprior values to verify correct prior is loaded
    print(f"  Hyperprior beta_0 values: r={hyperprior_arr[3]:.1f}, phi={hyperprior_arr[7]:.1f}, eta={hyperprior_arr[11]:.1f}, lambda={hyperprior_arr[15]:.1f}")

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

    # Build run schedule from reset_runs and resume_runs
    n_reset = reset_runs if reset_runs is not None else 1
    n_resume = resume_runs if resume_runs is not None else 0
    total_runs = n_reset + n_resume

    print(f"\nSampling Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Thinning: {mcmc_config['thin_iteration']}")
    print(f"  Reset runs: {n_reset}, Resume runs: {n_resume}")
    print(f"  Output: {output_dir}")

    # Build run schedule
    run_schedule = []
    if n_reset > 0:
        run_schedule.append(("reset", n_reset))
    if n_resume > 0:
        run_schedule.append(("resume", n_resume))

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
    parser.add_argument('--reset-runs', type=int, default=1)
    parser.add_argument('--resume-runs', type=int, default=0)
    parser.add_argument('--num-collect', type=int, default=1000)
    args = parser.parse_args()

    run_sampling(args.output_dir, args.reset_runs, args.resume_runs, args.num_collect)
