"""
MCMC sampling for RDU BHM model with CE-space prior specification.

Parallel to sampling.py but uses the rdu_bhm_ce posterior. The key difference
is that reference lottery data is packed into data["float"] alongside the
hyperprior array.
"""

import numpy as np
from pathlib import Path

from bamcmc import rmcmc
from clean_data import load_clean_data, make_bamcmc_data_indexed
from posteriors.rdu_bhm_ce import MODEL_ID, NPARAMS, N_HYPER, get_hyperprior_array


# Default configuration
DEFAULT_CONFIG = {
    'num_chains_a': 100,
    'num_chains_b': 100,
    'num_collect': 1000,
    'thin_iteration': 10,
    'chunk_size': 100,
    'rng_seed': 42,
}


def pack_ref_lotteries(ref_lotteries):
    """
    Pack reference lotteries into a flat numpy array.

    Args:
        ref_lotteries: List of 3 dicts, each with 'prizes' and 'probs' keys.

    Returns:
        Flat array of shape (18,): [prizes1(3), probs1(3), prizes2(3), ...]
    """
    arr = np.zeros(18, dtype=np.float64)
    for i, lot in enumerate(ref_lotteries):
        prizes = np.array(lot['prizes'], dtype=np.float64)
        probs = np.array(lot['probs'], dtype=np.float64)
        arr[i*6 : i*6+3] = prizes
        arr[i*6+3 : i*6+6] = probs
    return arr


def run_sampling(output_dir, ref_lotteries, reset_runs=None, resume_runs=None,
                 num_collect=None, config=None):
    """
    Run MCMC sampling with CE-space BHM.

    Args:
        output_dir: Directory for output files
        ref_lotteries: List of 3 reference lottery dicts
        reset_runs: Number of fresh/reset runs (default 1)
        resume_runs: Number of resume runs after reset runs (default 0)
        num_collect: Collection iterations per run (overrides config)
        config: Configuration dict (uses DEFAULT_CONFIG if None)

    Returns:
        Summary dict from rmcmc
    """
    cfg = config or DEFAULT_CONFIG

    # Load data
    print("Loading data...")
    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)

    # Add hyperprior array to data["float"][7]
    hyperprior_arr = get_hyperprior_array()
    indexed_data['float'] = indexed_data['float'] + (hyperprior_arr,)

    # Add reference lottery data to data["float"][8]
    ref_lot_arr = pack_ref_lotteries(ref_lotteries)
    indexed_data['float'] = indexed_data['float'] + (ref_lot_arr,)

    print(f"  Hyperprior array: {hyperprior_arr[:4]}")
    print(f"  Ref lottery array (first 6): {ref_lot_arr[:6]}")

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
        'burn_iter': 0,
        'num_collect': num_collect or cfg['num_collect'],
        'thin_iteration': cfg['thin_iteration'],
        'chunk_size': cfg.get('chunk_size', 100),
        'rng_seed': cfg['rng_seed'],
        'n_temperatures': max(cfg.get('n_temperatures', 1), 1),
        'beta_min': cfg.get('beta_min', 0.1),
        'per_temp_proposals': cfg.get('per_temp_proposals', True),
        'use_deo': cfg.get('use_deo', True),
    }

    # Build run schedule
    n_reset = reset_runs if reset_runs is not None else 1
    n_resume = resume_runs if resume_runs is not None else 0

    n_temps = mcmc_config['n_temperatures']

    print(f"\nSampling Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Thinning: {mcmc_config['thin_iteration']}")
    if n_temps > 1:
        print(f"  Parallel tempering: {n_temps} temperatures, "
              f"beta_min={mcmc_config['beta_min']}")
    print(f"  Reset runs: {n_reset}, Resume runs: {n_resume}")
    print(f"  Output: {output_dir}")

    run_schedule = []
    if n_reset > 0:
        run_schedule.append(("reset", n_reset))
    if n_resume > 0:
        run_schedule.append(("resume", n_resume))
    mcmc_config['run_schedule'] = run_schedule

    print(f"\nStarting MCMC with schedule: {run_schedule}")
    summary = rmcmc(
        mcmc_config,
        indexed_data,
        output_dir=str(output_dir),
        calculate_rhat=True,
        burn_in_fresh=True,
    )

    print(f"\nSampling complete!")
    print(f"  Total runs: {summary['total_runs_completed']}")
    print(f"  Final iteration: {summary['final_iteration']}")
    print(f"  History files: {len(summary['history_files'])}")

    return summary
