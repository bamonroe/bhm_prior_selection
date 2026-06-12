"""
Run prior-only sampling for all three active models (1 subject each).

Samples from the induced prior (no data likelihood) to visualize what the
quantity-space Beta-Beta hierarchy + Jacobian correction implies in parameter
space before seeing data.

Usage: python code/run_prior_only_all.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
PLOTS_DIR = ROOT / 'plots'

# --- Configuration ---
MODELS = [
    {
        'key': 'crra_prior',
        'model_id': 'rdu_bhm_ce_prior',
        'posterior_module': 'posteriors.rdu_bhm_ce_prior',
        'nparams': 4,
        'n_hyper': 8,
        'param_names': ['r', 'phi', 'eta', 'lambda'],
    },
    {
        'key': 'ge_prior',
        'model_id': 'rdu_ge_bhm_ce_prior',
        'posterior_module': 'posteriors.rdu_ge_bhm_ce_prior',
        'nparams': 4,
        'n_hyper': 8,
        'param_names': ['r', 'gamma', 'delta', 'lambda'],
    },
    {
        'key': 'power_prior',
        'model_id': 'rdu_power_bhm_ce_prior',
        'posterior_module': 'posteriors.rdu_power_bhm_ce_prior',
        'nparams': 3,
        'n_hyper': 8,
        'param_names': ['r', 'delta', 'lambda'],
    },
]

REF_LOTTERIES = [
    {'prizes': [10, 50], 'probs': [0.75, 0.25]},
    {'prizes': [20, 40], 'probs': [0.5, 0.5]},
    {'prizes': [12, 44], 'probs': [0.75, 0.25]},
]

REF_PAIR = {
    'lottery_A': {'prizes': [10, 50], 'probs': [0.75, 0.25]},
    'lottery_B': {'prizes': [12, 44], 'probs': [0.75, 0.25]},
}


def build_single_subject_data():
    """Load data but reduce to 1 subject for prior-only sampling."""
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from sampling_ce import pack_ref_lotteries, pack_ref_pair

    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)

    # Reduce to 1 subject
    indexed_data['static'] = (1,) + indexed_data['static'][1:]
    indexed_data['int'] = tuple(arr[:1] for arr in indexed_data['int'])
    indexed_data['float'] = tuple(
        arr[:1] if arr.ndim > 0 and arr.shape[0] > 1 else arr
        for arr in indexed_data['float']
    )

    # Append hyperprior, ref lottery, ref pair arrays
    from posteriors.rdu_bhm_ce import load_hyperpriors_from_file, get_hyperprior_array
    load_hyperpriors_from_file(str(DATA_DIR / 'output' / 'prior_params_ce.npz'))
    hyperprior_arr = get_hyperprior_array()
    indexed_data['float'] = indexed_data['float'] + (hyperprior_arr,)

    ref_lot_arr = pack_ref_lotteries(REF_LOTTERIES)
    indexed_data['float'] = indexed_data['float'] + (ref_lot_arr,)

    ref_pair_arr = pack_ref_pair(REF_PAIR)
    indexed_data['float'] = indexed_data['float'] + (ref_pair_arr,)

    return indexed_data


def run_sampling(model_id, indexed_data):
    """Run prior-only MCMC for one model."""
    from bamcmc import rmcmc, clean_model_files

    out = str(DATA_DIR / 'output')
    clean_model_files(out, model_id, mode='all')

    mcmc_config = {
        'posterior_id': model_id,
        'num_chains_a': 200,
        'num_chains_b': 200,
        'burn_iter': 0,
        'num_collect': 200,
        'thin_iteration': 10,
        'chunk_size': 10,
        'rng_seed': 42,
        'use_double': True,
        'n_temperatures': 1,
        'beta_min': 0.1,
        'per_temp_proposals': True,
        'use_deo': True,
        'run_schedule': [('reset', 1), ('resume', 4)],
    }

    summary = rmcmc(mcmc_config, indexed_data, output_dir=out,
                    calculate_rhat=True, burn_in_fresh=True)
    return summary


def preprocess_and_plot(model_id, nparams, n_hyper, param_names):
    """Preprocess history files and generate diagnostic plots."""
    from plotting import preprocess_hyperparams_ce, preprocess_subjects
    from plotting import parallel_plot_subjects

    data_dir = DATA_DIR / 'output' / model_id
    history_dir = data_dir / 'history' / 'full'
    preproc_dir = data_dir / 'preprocessed'
    subjects_dir = preproc_dir / 'subjects'
    plots_dir = PLOTS_DIR / model_id / 'subjects'

    if not history_dir.exists():
        print(f"  No history for {model_id}, skipping")
        return

    print(f"  Preprocessing {model_id}...")
    preprocess_hyperparams_ce(history_dir, preproc_dir, force=True,
                              nparams=nparams, n_hyper=n_hyper)
    result = preprocess_subjects(history_dir, subjects_dir, force=True,
                                 nparams=nparams, n_hyper=n_hyper,
                                 param_names=param_names)

    if result is not None:
        params_dir, iterations, batch_boundaries = result
        print(f"  Plotting {model_id}...")
        plots_dir.mkdir(parents=True, exist_ok=True)
        parallel_plot_subjects(
            params_dir, plots_dir, iterations, batch_boundaries,
            n_workers=1, prizes=np.array([10, 30, 50]),
            config={'model_key': model_id}
        )
    print(f"  Done with {model_id}")


def main():
    # Import all prior-only posteriors to register them
    import importlib
    for m in MODELS:
        importlib.import_module(m['posterior_module'])

    # Build shared data (1 subject)
    print("Building single-subject data...")
    indexed_data = build_single_subject_data()
    print(f"  Subjects: {indexed_data['static'][0]}")

    # Clear kernel cache between models
    from bamcmc.mcmc.compile import get_compiled_kernel_cache

    for m in MODELS:
        print(f"\n{'='*60}")
        print(f"Prior-only sampling: {m['model_id']}")
        print(f"{'='*60}")

        cache = get_compiled_kernel_cache()
        if cache:
            cache.clear()

        summary = run_sampling(m['model_id'], indexed_data)
        print(f"  Completed {summary['total_runs_completed']} runs")

        preprocess_and_plot(
            m['model_id'], m['nparams'], m['n_hyper'], m['param_names']
        )


if __name__ == '__main__':
    main()
