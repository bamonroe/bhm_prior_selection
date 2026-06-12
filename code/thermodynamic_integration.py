"""
Thermodynamic integration (TI) for marginal likelihood estimation.

Computes log p(y|M) = integral_0^1 E_beta[log L(theta)] d_beta

where E_beta is the expectation under the power posterior
p_beta(theta) proportional to L(theta)^beta pi(theta).

Post-processes tempered MCMC output from bamcmc. Requires runs with
n_temperatures > 1, which saves history, temperature_history, and
temperature_ladder in the NPZ files.

References:
    Friel & Pettitt (2008). Marginal likelihood estimation via power posteriors.
    Xie et al. (2011). Improving marginal likelihood estimation for Bayesian
        phylogenetic model selection. (Stepping-stone estimator.)
    Lartillot & Philippe (2006). Computing Bayes factors using thermodynamic
        integration.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path


# =============================================================================
# Per-model log-likelihood evaluators (likelihood ONLY, no priors/Jacobians)
# =============================================================================

def _build_full_log_lik_crra(data):
    """JIT function: chain_state -> sum of log p(y_i|theta_i) over all subjects."""
    from posteriors.rdu_bhm_ce import _get_indices, _get_subject_params, NPARAMS, N_HYPER
    from posteriors.common_fns import crra_ufn, risk_likelihood

    n_subjects = data["static"][0]
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in data["float"])
    unique_outcomes = dataf_jax[0]

    @jax.jit
    def full_log_lik(chain_state):
        def subject_body(sid, acc):
            r_val = chain_state[sid * NPARAMS + 0]
            phi_raw = chain_state[sid * NPARAMS + 1]
            eta_raw = chain_state[sid * NPARAMS + 2]
            lambda_raw = chain_state[sid * NPARAMS + 3]

            phi = jnp.exp(phi_raw)
            eta = jnp.exp(eta_raw)
            lam = jnp.exp(lambda_raw)
            utils = crra_ufn(unique_outcomes, r_val)
            log_like = risk_likelihood(utils, phi, eta, datai_jax, dataf_jax, sid, lam)
            return acc + log_like

        return jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)

    return full_log_lik, n_subjects * NPARAMS + N_HYPER


def _build_full_log_lik_ge(data):
    """JIT function: chain_state -> sum of log p(y_i|theta_i) for CRRA+GE."""
    from posteriors.rdu_ge_bhm_ce import NPARAMS, N_HYPER
    from posteriors.common_fns import crra_ufn, risk_likelihood_ge

    n_subjects = data["static"][0]
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in data["float"])
    unique_outcomes = dataf_jax[0]

    @jax.jit
    def full_log_lik(chain_state):
        def subject_body(sid, acc):
            r_val = chain_state[sid * NPARAMS + 0]
            gamma_raw = chain_state[sid * NPARAMS + 1]
            delta_raw = chain_state[sid * NPARAMS + 2]
            lambda_raw = chain_state[sid * NPARAMS + 3]

            gamma = jnp.exp(gamma_raw)
            delta = jnp.exp(delta_raw)
            lam = jnp.exp(lambda_raw)
            utils = crra_ufn(unique_outcomes, r_val)
            log_like = risk_likelihood_ge(utils, gamma, delta, datai_jax, dataf_jax, sid, lam)
            return acc + log_like

        return jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)

    return full_log_lik, n_subjects * NPARAMS + N_HYPER


def _build_full_log_lik_power(data):
    """JIT function: chain_state -> sum of log p(y_i|theta_i) for CRRA+Power."""
    from posteriors.rdu_power_bhm_ce import NPARAMS, N_HYPER
    from posteriors.common_fns import crra_ufn, risk_likelihood_power

    n_subjects = data["static"][0]
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in data["float"])
    unique_outcomes = dataf_jax[0]

    @jax.jit
    def full_log_lik(chain_state):
        def subject_body(sid, acc):
            r_val = chain_state[sid * NPARAMS + 0]
            delta_raw = chain_state[sid * NPARAMS + 1]
            lambda_raw = chain_state[sid * NPARAMS + 2]

            delta = jnp.exp(delta_raw)
            lam = jnp.exp(lambda_raw)
            utils = crra_ufn(unique_outcomes, r_val)
            log_like = risk_likelihood_power(utils, delta, datai_jax, dataf_jax, sid, lam)
            return acc + log_like

        return jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)

    return full_log_lik, n_subjects * NPARAMS + N_HYPER


LOG_LIK_BUILDERS = {
    'crra': _build_full_log_lik_crra,
    'ge': _build_full_log_lik_ge,
    'power': _build_full_log_lik_power,
}


# =============================================================================
# Load tempered MCMC output
# =============================================================================

def _load_tempered_run(model_key, data_dir):
    """Load history, temperature_history, and temperature_ladder from tempered run.

    Returns:
        states: (n_samples, n_chains, n_raw_params) — raw chain states
        temp_history: (n_samples, n_chains) — temperature index per chain
        temperature_ladder: (n_temps,) — beta values
    """
    from model_registry import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg['model_id']
    nparams = cfg['nparams']
    n_hyper = cfg['n_hyper']

    history_dir = Path(data_dir) / model_id / 'history' / 'full'
    history_files = sorted(history_dir.glob('*.npz'))

    if not history_files:
        raise FileNotFoundError(f"No history files for {model_id}")

    # Determine dimensions from first file
    first = np.load(history_files[0], allow_pickle=True)
    total_dim = first['history'].shape[2]
    n_subjects = (total_dim - n_hyper - 8) // (2 * nparams)
    n_raw = n_subjects * nparams + n_hyper

    # Check for temperature data
    if 'temperature_history' not in first:
        raise ValueError(f"No temperature_history in {history_files[0]}. "
                         f"Re-run with n_temperatures > 1.")
    temperature_ladder = first['temperature_ladder']

    # Load and concatenate all history files
    all_states = []
    all_temps = []
    for f in history_files:
        d = np.load(f, allow_pickle=True)
        all_states.append(d['history'][:, :, :n_raw])
        all_temps.append(d['temperature_history'])

    states = np.concatenate(all_states, axis=0)
    temp_history = np.concatenate(all_temps, axis=0)

    # Discard first half as burn-in
    burn = states.shape[0] // 2
    states = states[burn:]
    temp_history = temp_history[burn:]

    return states, temp_history, temperature_ladder


# =============================================================================
# GPU-batched log-likelihood evaluation
# =============================================================================

def _evaluate_log_liks(log_lik_fn, states_2d, batch_size=500):
    """Evaluate log L(theta) at many states, GPU-batched.

    Args:
        log_lik_fn: JIT-compiled function(chain_state) -> scalar
        states_2d: (n_states, n_params) array
        batch_size: vmap batch size

    Returns:
        (n_states,) array of log-likelihoods
    """
    batched_fn = jax.vmap(log_lik_fn)
    n = len(states_2d)
    results = []

    for i in range(0, n, batch_size):
        batch = jnp.array(states_2d[i:i + batch_size])
        vals = batched_fn(batch)
        results.append(np.array(vals))

    return np.concatenate(results)


# =============================================================================
# TI estimators
# =============================================================================

def _trapezoidal_integration(betas, mean_log_liks):
    """Estimate log p(y|M) via trapezoidal rule.

    Args:
        betas: (K,) array of temperature levels, sorted ascending (0 to 1)
        mean_log_liks: (K,) array of E_beta[log L] at each level

    Returns:
        Estimated log marginal likelihood
    """
    # Trapezoidal rule: integral ≈ sum (beta_{k+1} - beta_k) * (f_k + f_{k+1}) / 2
    log_ml = 0.0
    for k in range(len(betas) - 1):
        db = betas[k + 1] - betas[k]
        log_ml += db * (mean_log_liks[k] + mean_log_liks[k + 1]) / 2
    return log_ml


def _stepping_stone_estimator(betas, log_liks_per_temp):
    """Stepping-stone estimator (Xie et al. 2011).

    More robust than trapezoidal, uses the identity:
    log p(y|M) = sum_k log[ (1/N_k) sum_j exp((beta_{k+1} - beta_k) * log_lik_j^(k)) ]

    Args:
        betas: (K,) sorted ascending
        log_liks_per_temp: list of K arrays, each containing log L(theta)
            evaluated at samples from p_beta_k

    Returns:
        Estimated log marginal likelihood
    """
    log_ml = 0.0
    for k in range(len(betas) - 1):
        db = betas[k + 1] - betas[k]
        lls = log_liks_per_temp[k]
        if len(lls) == 0:
            continue
        # log-sum-exp trick
        scaled = db * lls
        max_s = np.max(scaled)
        log_ml += max_s + np.log(np.mean(np.exp(scaled - max_s)))
    return log_ml


# =============================================================================
# Main entry point
# =============================================================================

def compute_bayes_factors_ti(model_keys, data_dir, output_path, seed=42):
    """Compute Bayes factors via thermodynamic integration.

    Requires that each model has been run with n_temperatures > 1.

    Args:
        model_keys: List of model keys
        data_dir: Path to data/output
        output_path: Path to save results NPZ
    """
    from posteriors.rdu_bhm_ce import load_hyperpriors_from_file
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from sampling_ce import pack_ref_lotteries, pack_ref_pair
    from posteriors.rdu_bhm_ce import get_hyperprior_array

    data_dir = Path(data_dir)

    load_hyperpriors_from_file(str(data_dir / 'prior_params_ce.npz'))
    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)
    indexed_data['float'] = indexed_data['float'] + (get_hyperprior_array(),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_lotteries([
        {'prizes': [10, 50], 'probs': [0.75, 0.25]},
        {'prizes': [20, 40], 'probs': [0.5, 0.5]},
        {'prizes': [12, 44], 'probs': [0.75, 0.25]},
    ]),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_pair({
        'lottery_A': {'prizes': [10, 50], 'probs': [0.75, 0.25]},
        'lottery_B': {'prizes': [12, 44], 'probs': [0.75, 0.25]},
    }),)

    print("=" * 60)
    print("BAYES FACTORS (Thermodynamic Integration)")
    print("=" * 60)

    results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        # Load tempered MCMC output
        print("  Loading tempered run...")
        states, temp_history, temperature_ladder = _load_tempered_run(
            model_key, str(data_dir))
        n_samples, n_chains, n_params = states.shape
        n_temps = len(temperature_ladder)
        print(f"  {n_samples} samples x {n_chains} chains, "
              f"{n_temps} temperatures, {n_params} params")
        print(f"  Temperature ladder: {temperature_ladder}")

        # Build log-likelihood evaluator
        print("  Building log-likelihood evaluator...")
        builder = LOG_LIK_BUILDERS[model_key]
        log_lik_fn, expected_params = builder(indexed_data)

        # Evaluate log L(theta) at ALL saved states
        print(f"  Evaluating log L at {n_samples * n_chains} states...")
        all_states_flat = states.reshape(-1, n_params)
        all_log_liks = _evaluate_log_liks(log_lik_fn, all_states_flat)
        all_log_liks = all_log_liks.reshape(n_samples, n_chains)
        print(f"  Log L range: [{np.nanmin(all_log_liks):.1f}, "
              f"{np.nanmax(all_log_liks):.1f}]")

        n_nonfinite = np.sum(~np.isfinite(all_log_liks))
        if n_nonfinite > 0:
            print(f"  WARNING: {n_nonfinite} non-finite log-likelihoods")

        # Group by temperature and compute E_beta[log L]
        # temperature_ladder is sorted descending: [1.0, ..., beta_min]
        # temp_history values are indices into temperature_ladder
        betas_sorted = np.sort(temperature_ladder)  # ascending for integration
        temp_idx_sorted = np.argsort(temperature_ladder)  # map sorted -> original

        mean_log_liks = np.empty(n_temps)
        log_liks_per_temp = []

        print(f"\n  {'beta':>8s}  {'n_samples':>10s}  {'E[log L]':>12s}  {'SE':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}")

        for sort_idx in range(n_temps):
            orig_idx = temp_idx_sorted[sort_idx]
            beta = temperature_ladder[orig_idx]

            # Find all (sample, chain) pairs at this temperature
            mask = (temp_history == orig_idx)
            lls = all_log_liks[mask]
            lls = lls[np.isfinite(lls)]

            if len(lls) > 0:
                mean_ll = np.mean(lls)
                se_ll = np.std(lls) / np.sqrt(len(lls))
            else:
                mean_ll = np.nan
                se_ll = np.nan

            mean_log_liks[sort_idx] = mean_ll
            log_liks_per_temp.append(lls)

            print(f"  {beta:8.4f}  {len(lls):10d}  {mean_ll:12.2f}  {se_ll:10.4f}")

        # Integrate
        valid = np.isfinite(mean_log_liks)
        if not np.all(valid):
            print(f"  WARNING: {np.sum(~valid)} temperature levels with no data")

        log_ml_trap = _trapezoidal_integration(betas_sorted[valid],
                                                mean_log_liks[valid])
        log_ml_ss = _stepping_stone_estimator(betas_sorted, log_liks_per_temp)

        print(f"\n  Trapezoidal:    log ML = {log_ml_trap:.2f}")
        print(f"  Stepping-stone: log ML = {log_ml_ss:.2f}")

        results[model_key] = {
            'log_ml_trap': float(log_ml_trap),
            'log_ml_ss': float(log_ml_ss),
            'temperature_ladder': temperature_ladder,
            'mean_log_liks': mean_log_liks,
        }

    # Bayes factors
    print("\n" + "=" * 60)
    print("BAYES FACTORS (log scale, stepping-stone)")
    print("=" * 60)

    save_dict = {}
    for key, res in results.items():
        save_dict[f'{key}_log_ml_trap'] = res['log_ml_trap']
        save_dict[f'{key}_log_ml_ss'] = res['log_ml_ss']

    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            log_bf = results[k1]['log_ml_ss'] - results[k2]['log_ml_ss']
            print(f"  log BF({k1} vs {k2}) = {log_bf:.2f}")
            save_dict[f'log_bf_ti_{k1}_vs_{k2}'] = log_bf

    np.savez(Path(output_path), **save_dict)
    print(f"\nSaved to: {output_path}")
    return results
