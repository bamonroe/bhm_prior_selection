"""
Sampling module for the prior-only RDU CE-space model.

Runs the same sampler as sampling_ce.py but uses the rdu_bhm_ce_prior
posterior, which omits the data likelihood. The output is the hyperprior
predictive distribution in parameter space.
"""

import numpy as np
from pathlib import Path

from clean_data import load_clean_data, make_bamcmc_data_indexed
from sampling_ce import pack_ref_lotteries, pack_ref_pair, DEFAULT_CONFIG

# Import the prior-only posterior (triggers registration)
from posteriors.rdu_bhm_ce_prior import (
    MODEL_ID, NPARAMS, N_HYPER, get_hyperprior_array,
)


def run_sampling_prior(output_dir, ref_lotteries, ref_pair, reset_runs=None,
                       resume_runs=None, num_collect=None, config=None):
    """
    Run prior-only MCMC sampling with quantity-space Beta-Beta BHM.

    Same interface as sampling_ce.run_sampling but uses the prior-only model.
    """
    from bamcmc import rmcmc

    cfg = config or DEFAULT_CONFIG

    # Load data (needed for the forward map, not for likelihood)
    print("Loading data...")
    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)

    # Add hyperprior array to data["float"][7]
    hyperprior_arr = get_hyperprior_array()
    indexed_data['float'] = indexed_data['float'] + (hyperprior_arr,)

    # Add reference lottery data to data["float"][8]
    ref_lot_arr = pack_ref_lotteries(ref_lotteries)
    indexed_data['float'] = indexed_data['float'] + (ref_lot_arr,)

    # Add reference pair data to data["float"][9]
    ref_pair_arr = pack_ref_pair(ref_pair)
    indexed_data['float'] = indexed_data['float'] + (ref_pair_arr,)

    n_subjects = indexed_data['static'][0]
    print(f"  Subjects: {n_subjects}")
    print(f"  Total parameters: {n_subjects * NPARAMS + N_HYPER}")
    print(f"  Model: {MODEL_ID} (prior only, no likelihood)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcmc_config = {
        'posterior_id': MODEL_ID,
        'num_chains_a': cfg['num_chains_a'],
        'num_chains_b': cfg['num_chains_b'],
        'burn_iter': 0,
        'num_collect': num_collect or cfg['num_collect'],
        'thin_iteration': cfg['thin_iteration'],
        'chunk_size': cfg.get('chunk_size', 100),
        'rng_seed': cfg['rng_seed'],
        'n_temperatures': 1,  # no tempering needed for prior
        'beta_min': 1.0,
    }

    n_reset = reset_runs if reset_runs is not None else 1
    n_resume = resume_runs if resume_runs is not None else 0

    print(f"\nSampling Configuration:")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Thinning: {mcmc_config['thin_iteration']}")
    print(f"  Reset runs: {n_reset}, Resume runs: {n_resume}")
    print(f"  Output: {output_dir}")

    run_schedule = (
        [{'mode': 'reset'}] * n_reset
        + [{'mode': 'resume'}] * n_resume
    )

    return rmcmc(
        mcmc_config=mcmc_config,
        data=indexed_data,
        output_dir=str(output_dir),
        run_schedule=run_schedule,
    )
