"""
Thermodynamic integration via independent MCMC runs at fixed temperatures.

Instead of using parallel tempering (which compiles a complex temperature-swap
kernel), this module runs standard MCMC at each temperature level independently.
Each run reuses the n_temperatures=1 compiled kernel, with the temperature
value passed through bamcmc's `fixed_beta` config parameter.

The marginal likelihood is estimated via:
    log p(y|M) = integral_0^1 E_beta[log L(theta)] d_beta

using the trapezoidal rule and the stepping-stone estimator (Xie et al. 2011).

For hierarchical models, the lowest beta level (near-prior) uses samples from
the prior-only MCMC runs rather than fresh power-posterior runs, since at
beta ~ 0 the power posterior is the prior.

Usage: called from main.py via do_ti_sampling / do_ti_compute flags.
"""

import numpy as np
from pathlib import Path
import importlib


def geometric_ladder(n_temps, beta_min):
    """Ascending geometric temperature ladder [beta_min, ..., 1.0].

    Uses the same spacing as bamcmc's parallel tempering:
        beta_k = beta_min^((n_temps - 1 - k) / (n_temps - 1))

    Args:
        n_temps: Number of temperature levels (including endpoints)
        beta_min: Coldest temperature (near-prior)

    Returns:
        Sorted ascending array of length n_temps, ending at 1.0
    """
    if n_temps == 1:
        return np.array([1.0])
    indices = np.arange(n_temps)
    ladder = np.power(beta_min, indices / (n_temps - 1))
    return np.sort(ladder)


# =========================================================================
# Sampling phase: run independent MCMC at each temperature
# =========================================================================

def run_ti_sampling(model_key, sampling_opts, ref_lotteries, ref_pair,
                    prior_params_path, ti_config):
    """Run independent MCMC at each temperature level for one model.

    For each beta in the temperature ladder:
    - beta = 1.0: reuses existing posterior samples
    - beta = beta_min: uses existing prior-only samples
    - intermediate betas: runs standard MCMC with fixed_beta

    Args:
        model_key: Key in MODEL_REGISTRY ('crra', 'ge', 'power')
        sampling_opts: Base sampling config (chain count, thinning, etc.)
        ref_lotteries: List of 3 reference lottery dicts
        ref_pair: Dict with lottery_A, lottery_B
        prior_params_path: Path to prior_params_ce.npz
        ti_config: Dict with n_temps, beta_min, reset_runs, resume_runs, num_collect
    """
    from model_registry import MODEL_REGISTRY
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from posteriors.rdu_bhm_ce import load_hyperpriors_from_file, get_hyperprior_array
    from sampling_ce import pack_ref_lotteries, pack_ref_pair
    from bamcmc import rmcmc

    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg['model_id']

    # Ensure posterior is registered
    importlib.import_module(cfg['posterior_module'])

    # Build data dict (same as standard sampling)
    load_hyperpriors_from_file(str(prior_params_path))
    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)
    indexed_data['float'] = indexed_data['float'] + (get_hyperprior_array(),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_lotteries(ref_lotteries),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_pair(ref_pair),)

    ladder = geometric_ladder(ti_config['n_temps'], ti_config['beta_min'])
    base_dir = Path(sampling_opts['output_dir']) / model_id / 'ti'

    n_reset = ti_config.get('reset_runs', 1)
    n_resume = ti_config.get('resume_runs', 3)
    expected_files = n_reset + n_resume

    # Check for prior-only samples (used for beta_min)
    prior_model_id = cfg.get('prior_model_id')
    if prior_model_id:
        prior_hist = Path(sampling_opts['output_dir']) / prior_model_id / 'history' / 'full'
        has_prior = prior_hist.exists() and list(prior_hist.glob('*.npz'))
    else:
        has_prior = False

    for i, beta in enumerate(ladder):
        # beta = 1.0: use existing posterior samples
        if np.isclose(beta, 1.0):
            std_dir = Path(sampling_opts['output_dir']) / model_id / 'history' / 'full'
            if std_dir.exists() and list(std_dir.glob('*.npz')):
                print(f"  [{i+1}/{len(ladder)}] beta={beta:.6f}: "
                      f"using existing posterior samples")
            else:
                print(f"  [{i+1}/{len(ladder)}] beta={beta:.6f}: "
                      f"WARNING — no existing posterior samples at {std_dir}")
            continue

        # beta = beta_min: use prior-only samples
        if np.isclose(beta, ladder[0]) and has_prior:
            print(f"  [{i+1}/{len(ladder)}] beta={beta:.6f}: "
                  f"using prior-only samples from {prior_model_id}")
            continue

        beta_dir = base_dir / f'beta_{beta:.6f}'
        # rmcmc nests model_id/ under output_dir
        hist_dir = beta_dir / model_id / 'history' / 'full'

        # Skip if already completed
        if hist_dir.exists():
            existing = len(list(hist_dir.glob('*.npz')))
            if existing >= expected_files:
                print(f"  [{i+1}/{len(ladder)}] beta={beta:.6f}: "
                      f"already completed ({existing} history files)")
                continue

        beta_dir.mkdir(parents=True, exist_ok=True)

        mcmc_config = {
            'posterior_id': model_id,
            'num_chains_a': sampling_opts['num_chains'],
            'num_chains_b': sampling_opts['num_chains'],
            'burn_iter': 0,
            'num_collect': ti_config.get('num_collect', 100),
            'thin_iteration': sampling_opts['thin_iteration'],
            'chunk_size': sampling_opts.get('chunk_size', 10),
            'rng_seed': sampling_opts['seed'],
            'use_double': sampling_opts.get('use_double', True),
            'n_temperatures': 1,
            'fixed_beta': float(beta),
        }

        run_schedule = []
        if n_reset > 0:
            run_schedule.append(("reset", n_reset))
        if n_resume > 0:
            run_schedule.append(("resume", n_resume))
        mcmc_config['run_schedule'] = run_schedule

        print(f"\n  [{i+1}/{len(ladder)}] beta={beta:.6f}: "
              f"sampling ({n_reset} reset + {n_resume} resume)")
        summary = rmcmc(mcmc_config, indexed_data, output_dir=str(beta_dir),
                        calculate_rhat=False, burn_in_fresh=True)
        print(f"    Completed {summary['total_runs_completed']} runs, "
              f"iteration {summary['final_iteration']}")


# =========================================================================
# Computation phase: load samples, evaluate log-likelihoods, integrate
# =========================================================================

def _load_history(history_dir, nparams, n_hyper):
    """Load and concatenate history files, return raw state samples.

    Args:
        history_dir: Path to history/full/ directory
        nparams: Per-subject parameter count
        n_hyper: Number of hyperparameters

    Returns:
        (n_samples, n_raw_params) array of post-burn-in samples
    """
    history_files = sorted(Path(history_dir).glob('*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files in {history_dir}")

    first = np.load(history_files[0])
    total_dim = first['history'].shape[2]
    n_chains_expected = first['history'].shape[1]
    n_subjects = (total_dim - n_hyper - 8) // (2 * nparams)
    n_raw = n_subjects * nparams + n_hyper

    all_states = []
    for f in history_files:
        d = np.load(f)
        h = d['history']
        # Skip files with mismatched chain counts (from different configs)
        if h.shape[1] != n_chains_expected or h.shape[2] != total_dim:
            continue
        all_states.append(h[:, :, :n_raw])

    if not all_states:
        raise FileNotFoundError(f"No compatible history files in {history_dir}")

    combined = np.concatenate(all_states, axis=0)

    # Discard first half as burn-in
    burn = combined.shape[0] // 2
    post_burn = combined[burn:]

    # Flatten (iterations, chains) -> samples
    return post_burn.reshape(-1, n_raw)


def _build_indexed_data(ref_lotteries, ref_pair):
    """Build the indexed data dict for log-likelihood evaluation."""
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from posteriors.rdu_bhm_ce import get_hyperprior_array
    from sampling_ce import pack_ref_lotteries, pack_ref_pair

    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)
    indexed_data['float'] = indexed_data['float'] + (get_hyperprior_array(),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_lotteries(ref_lotteries),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_pair(ref_pair),)
    return indexed_data


def _resolve_history_dir(model_key, data_dir, beta, ladder, ti_config):
    """Determine which history directory to use for a given beta level.

    Returns the history directory path. For beta_min, uses prior-only samples
    if available. For beta=1.0, uses the standard posterior samples.
    """
    from model_registry import MODEL_REGISTRY

    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg['model_id']
    data_dir = Path(data_dir)

    if np.isclose(beta, 1.0):
        return data_dir / model_id / 'history' / 'full'

    if np.isclose(beta, ladder[0]):
        prior_model_id = cfg.get('prior_model_id')
        if prior_model_id:
            prior_dir = data_dir / prior_model_id / 'history' / 'full'
            if prior_dir.exists() and list(prior_dir.glob('*.npz')):
                return prior_dir

    return (data_dir / model_id / 'ti' / f'beta_{beta:.6f}'
            / model_id / 'history' / 'full')


def compute_ti_marginal_likelihood(model_key, data_dir, ti_config, indexed_data,
                                   max_samples=50000, batch_size=500, seed=42):
    """Compute log marginal likelihood from independent TI runs.

    Loads posterior samples at each temperature level, evaluates the
    log-likelihood (prior-free) at each sample, and integrates E_beta[log L]
    over the temperature ladder.

    Args:
        model_key: Key in MODEL_REGISTRY
        data_dir: Path to data/output
        ti_config: Dict with n_temps, beta_min
        indexed_data: Data dict for log-likelihood evaluation
        max_samples: Cap on samples per temperature (subsampled if exceeded)
        batch_size: GPU batch size for log-likelihood evaluation
        seed: RNG seed for subsampling

    Returns:
        Dict with log_ml_trap, log_ml_ss, temperature_ladder, mean_log_liks
    """
    import jax
    import jax.numpy as jnp
    from model_registry import MODEL_REGISTRY
    from thermodynamic_integration import (
        LOG_LIK_BUILDERS, _evaluate_log_liks,
        _trapezoidal_integration, _stepping_stone_estimator,
    )

    cfg = MODEL_REGISTRY[model_key]
    nparams = cfg['nparams']
    n_hyper = cfg['n_hyper']

    ladder = geometric_ladder(ti_config['n_temps'], ti_config['beta_min'])
    n_temps = len(ladder)

    # Build log-likelihood evaluator
    print("  Building log-likelihood evaluator...")
    builder = LOG_LIK_BUILDERS[model_key]
    log_lik_fn, _ = builder(indexed_data)

    print(f"\n  {'beta':>8s}  {'source':>10s}  {'n_samples':>10s}  "
          f"{'E[log L]':>12s}  {'SE':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

    mean_log_liks = np.empty(n_temps)
    log_liks_per_temp = []

    for k, beta in enumerate(ladder):
        hist_dir = _resolve_history_dir(
            model_key, data_dir, beta, ladder, ti_config)

        # Identify source for display
        hist_dir_str = str(hist_dir)
        if 'prior' in hist_dir_str:
            source = 'prior'
        elif '/ti/' in hist_dir_str:
            source = 'ti-mcmc'
        else:
            source = 'posterior'

        samples = _load_history(hist_dir, nparams, n_hyper)

        # Subsample if too many
        if len(samples) > max_samples:
            rng = np.random.default_rng(seed + k)
            idx = rng.choice(len(samples), max_samples, replace=False)
            samples = samples[idx]

        # Evaluate log L(theta) at all samples
        lls = _evaluate_log_liks(log_lik_fn, samples, batch_size=batch_size)
        lls = lls[np.isfinite(lls)]

        if len(lls) > 0:
            mean_ll = float(np.mean(lls))
            se_ll = float(np.std(lls) / np.sqrt(len(lls)))
        else:
            mean_ll = np.nan
            se_ll = np.nan

        mean_log_liks[k] = mean_ll
        log_liks_per_temp.append(lls)

        print(f"  {beta:8.6f}  {source:>10s}  {len(lls):10d}  "
              f"{mean_ll:12.2f}  {se_ll:10.4f}")

    # Integrate
    valid = np.isfinite(mean_log_liks)
    if not np.all(valid):
        print(f"  WARNING: {int(np.sum(~valid))} temperature levels with no data")

    log_ml_trap = _trapezoidal_integration(ladder[valid], mean_log_liks[valid])
    log_ml_ss = _stepping_stone_estimator(ladder, log_liks_per_temp)

    return {
        'log_ml_trap': float(log_ml_trap),
        'log_ml_ss': float(log_ml_ss),
        'temperature_ladder': ladder,
        'mean_log_liks': mean_log_liks,
    }


def compute_ti_bayes_factors(model_keys, data_dir, ti_config, output_path,
                             ref_lotteries, ref_pair):
    """Compute Bayes factors from independent TI runs.

    Args:
        model_keys: List of model keys (e.g., ['crra', 'ge', 'power'])
        data_dir: Path to data/output
        ti_config: Dict with n_temps, beta_min
        output_path: Path to save results NPZ
        ref_lotteries: Reference lotteries for data building
        ref_pair: Reference pair for data building
    """
    from posteriors.rdu_bhm_ce import load_hyperpriors_from_file

    data_dir = Path(data_dir)
    load_hyperpriors_from_file(str(data_dir / 'prior_params_ce.npz'))
    indexed_data = _build_indexed_data(ref_lotteries, ref_pair)

    print("=" * 60)
    print("THERMODYNAMIC INTEGRATION (Independent Runs)")
    print("=" * 60)

    ladder = geometric_ladder(ti_config['n_temps'], ti_config['beta_min'])
    print(f"  Temperature ladder ({len(ladder)} levels):")
    print(f"    {[f'{b:.4f}' for b in ladder]}")

    results = {}
    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        result = compute_ti_marginal_likelihood(
            model_key, str(data_dir), ti_config, indexed_data)
        results[model_key] = result

        print(f"\n  Trapezoidal:    log ML = {result['log_ml_trap']:.2f}")
        print(f"  Stepping-stone: log ML = {result['log_ml_ss']:.2f}")

    # Pairwise Bayes factors
    print("\n" + "=" * 60)
    print("BAYES FACTORS (log scale, stepping-stone)")
    print("=" * 60)

    save_dict = {}
    for key, res in results.items():
        save_dict[f'{key}_log_ml_trap'] = res['log_ml_trap']
        save_dict[f'{key}_log_ml_ss'] = res['log_ml_ss']
        save_dict[f'{key}_mean_log_liks'] = res['mean_log_liks']
    save_dict['temperature_ladder'] = ladder

    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            log_bf = results[k1]['log_ml_ss'] - results[k2]['log_ml_ss']
            print(f"  log BF({k1} vs {k2}) = {log_bf:.2f}")
            save_dict[f'log_bf_ti_{k1}_vs_{k2}'] = log_bf

    np.savez(Path(output_path), **save_dict)
    print(f"\nSaved to: {output_path}")
    return results
