"""
Rao-Blackwellized bridge sampling for marginal likelihood estimation.

Integrates out subject-level parameters via importance sampling, reducing
bridge sampling to the 8-dimensional hyperparameter space.

For each subject i, the integrated likelihood is:
    p(y_i|psi) = integral f_i(theta_i) p(theta_i|psi) d(theta_i)

where f_i(theta_i) = p(y_i|theta_i) |J(theta_i)| contains everything that
doesn't depend on psi. We precompute log f_i at K theta draws per subject,
then for any psi, reweight by p(theta_i|psi) to estimate the integral.

The unnormalized marginal posterior of psi is:
    p_tilde(psi) = pi(psi) * prod_i p(y_i|psi)

Bridge sampling on this 8-dimensional function converges easily.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path


# =============================================================================
# Load MCMC states
# =============================================================================

def _load_mcmc_states(model_key, data_dir, n_samples=None, seed=42):
    """Load MCMC states split into hyper and per-subject parts."""
    from model_registry import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg['model_id']
    nparams = cfg['nparams']
    n_hyper = cfg['n_hyper']

    history_dir = Path(data_dir) / model_id / 'history' / 'full'
    history_files = sorted(history_dir.glob('*.npz'))

    first = np.load(history_files[0], allow_pickle=True)
    total_dim = first['history'].shape[2]
    n_subjects = (total_dim - n_hyper - 8) // (2 * nparams)
    n_raw = n_subjects * nparams + n_hyper

    all_states = []
    for f in history_files:
        d = np.load(f, allow_pickle=True)
        all_states.append(d['history'][:, :, :n_raw])

    combined = np.concatenate(all_states, axis=0)
    burn = combined.shape[0] // 2
    flat = combined[burn:].reshape(-1, n_raw)

    if n_samples is not None and n_samples < len(flat):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), n_samples, replace=False)
        flat = flat[idx]

    hyper_start = n_subjects * nparams
    subject_flat = flat[:, :hyper_start].reshape(-1, n_subjects, nparams)
    hyper_flat = flat[:, hyper_start:hyper_start + n_hyper]

    return hyper_flat, subject_flat, n_subjects


# =============================================================================
# Precompute per-subject f_i(theta_i) = p(y_i|theta_i) |J(theta_i)|
# =============================================================================

def _build_log_f_fn(model_key, data):
    """Build function: log f_i(theta_i) = log p(y_i|theta_i) + log |J(theta_i)|.

    This is the part of the subject contribution that does NOT depend on psi.
    """
    dataf = data["float"]
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
    unique_outcomes = dataf_jax[0]

    if model_key == 'crra':
        from posteriors.common_fns import (
            full_jacobian_logabsdet, crra_ufn, risk_likelihood,
        )

        @jax.jit
        def log_f(theta_i, sid):
            r = theta_i[0]
            phi_raw, eta_raw, lambda_raw = theta_i[1], theta_i[2], theta_i[3]
            log_det_j = full_jacobian_logabsdet(
                r, phi_raw, eta_raw, lambda_raw, ref_lotteries, ref_pair)
            phi = jnp.exp(phi_raw)
            eta = jnp.exp(eta_raw)
            lam = jnp.exp(lambda_raw)
            utils = crra_ufn(unique_outcomes, r)
            log_like = risk_likelihood(utils, phi, eta, datai_jax, dataf_jax, sid, lam)
            return log_det_j + log_like

    elif model_key == 'ge':
        from posteriors.common_fns import (
            full_jacobian_logabsdet_ge, crra_ufn, risk_likelihood_ge,
        )

        @jax.jit
        def log_f(theta_i, sid):
            r = theta_i[0]
            gamma_raw, delta_raw, lambda_raw = theta_i[1], theta_i[2], theta_i[3]
            log_det_j = full_jacobian_logabsdet_ge(
                r, gamma_raw, delta_raw, lambda_raw, ref_lotteries, ref_pair)
            gamma = jnp.exp(gamma_raw)
            delta = jnp.exp(delta_raw)
            lam = jnp.exp(lambda_raw)
            utils = crra_ufn(unique_outcomes, r)
            log_like = risk_likelihood_ge(utils, gamma, delta, datai_jax, dataf_jax, sid, lam)
            return log_det_j + log_like

    elif model_key == 'power':
        from posteriors.common_fns import (
            gram_logdet_power, crra_ufn, risk_likelihood_power,
        )

        @jax.jit
        def log_f(theta_i, sid):
            r = theta_i[0]
            delta_raw, lambda_raw = theta_i[1], theta_i[2]
            log_gram = gram_logdet_power(
                r, delta_raw, lambda_raw, ref_lotteries, ref_pair)
            delta = jnp.exp(delta_raw)
            lam = jnp.exp(lambda_raw)
            utils = crra_ufn(unique_outcomes, r)
            log_like = risk_likelihood_power(utils, delta, datai_jax, dataf_jax, sid, lam)
            return log_gram + log_like

    return log_f


# =============================================================================
# Population prior p(theta_i | psi) — the psi-dependent part
# =============================================================================

def _build_log_pop_prior_fn(data):
    """Build function: log p(theta_i | psi) via the Beta-Beta hierarchy.

    This maps theta_i -> quantities via the forward map, then evaluates
    the Beta-Beta population prior conditional on psi.
    """
    from posteriors.rdu_bhm_ce import (
        _subject_quantity_prior, _get_indices,
        N_QUANTITIES, HYPERPRIOR_STRIDE,
    )

    dataf = data["float"]
    hyperprior_arr = jnp.asarray(dataf[7])
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])

    def log_pop_prior_crra(theta_i, psi):
        from posteriors.common_fns import full_forward_map
        r, phi_raw, eta_raw, lambda_raw = theta_i[0], theta_i[1], theta_i[2], theta_i[3]
        quantities = full_forward_map(r, phi_raw, eta_raw, lambda_raw,
                                      ref_lotteries, ref_pair)
        chain_state = jnp.concatenate([theta_i, psi])
        idx = _get_indices(1)
        return _subject_quantity_prior(quantities, chain_state, idx, hyperprior_arr)

    def log_pop_prior_ge(theta_i, psi):
        from posteriors.common_fns import full_forward_map_ge
        r, gamma_raw, delta_raw, lambda_raw = theta_i[0], theta_i[1], theta_i[2], theta_i[3]
        quantities = full_forward_map_ge(r, gamma_raw, delta_raw, lambda_raw,
                                         ref_lotteries, ref_pair)
        chain_state = jnp.concatenate([theta_i, psi])
        idx = _get_indices(1)
        return _subject_quantity_prior(quantities, chain_state, idx, hyperprior_arr)

    def log_pop_prior_power(theta_i, psi):
        from posteriors.common_fns import full_forward_map_power
        from posteriors.rdu_power_bhm_ce import _get_indices as _get_indices_power
        r, delta_raw, lambda_raw = theta_i[0], theta_i[1], theta_i[2]
        quantities = full_forward_map_power(r, delta_raw, lambda_raw,
                                             ref_lotteries, ref_pair)
        chain_state = jnp.concatenate([theta_i, psi])
        idx = _get_indices_power(1)
        return _subject_quantity_prior(quantities, chain_state, idx, hyperprior_arr)

    return {
        'crra': jax.jit(log_pop_prior_crra),
        'ge': jax.jit(log_pop_prior_ge),
        'power': jax.jit(log_pop_prior_power),
    }


# =============================================================================
# Hyperprior
# =============================================================================

def _build_log_hyperprior(data):
    from posteriors.rdu_bhm_ce import _compute_all_hyperpriors, _get_indices
    dataf = data["float"]
    hyperprior_arr = jnp.asarray(dataf[7])

    @jax.jit
    def log_hyperprior(psi):
        idx = _get_indices(0)
        return _compute_all_hyperpriors(psi, idx, hyperprior_arr)

    return log_hyperprior


# =============================================================================
# Bridge sampling utilities
# =============================================================================

def _logsumexp(x):
    mx = np.max(x)
    return mx + np.log(np.sum(np.exp(x - mx)))


# =============================================================================
# Main
# =============================================================================

def compute_bayes_factors_rb(model_keys, data_dir, output_path,
                              n_psi_samples=5000, n_theta_draws=200,
                              n_iter=200, seed=42):
    """Compute Bayes factors via Rao-Blackwellized bridge sampling.

    For each model:
    1. Precompute log f_i(theta_i^(k)) for K theta draws per subject
    2. For each psi (posterior or proposal), estimate:
       log p(y_i|psi) = log (1/K) sum_k exp[log f_i(theta_i^(k)) + log p(theta_i^(k)|psi)]
       via log-sum-exp over the K draws
    3. Bridge sample on log[pi(psi) * prod_i p(y_i|psi)] in 8 dimensions

    Args:
        n_psi_samples: Number of psi draws for bridge sampling
        n_theta_draws: Number of theta draws per subject for IS integration
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
    print("BAYES FACTORS (Rao-Blackwellized Bridge Sampling)")
    print("=" * 60)

    log_hyperprior_fn = _build_log_hyperprior(indexed_data)
    pop_prior_fns = _build_log_pop_prior_fn(indexed_data)

    results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        # Load MCMC states
        # Need enough to get n_psi_samples psi draws AND n_theta_draws theta draws
        n_total = max(n_psi_samples, n_theta_draws) * 2
        hyper_states, subject_states, n_subjects = _load_mcmc_states(
            model_key, str(data_dir), n_samples=n_total, seed=seed)
        nparams = subject_states.shape[2]
        print(f"  Loaded {len(hyper_states)} MCMC draws, "
              f"{n_subjects} subjects, {nparams} params/subject")

        # Split: first n_theta_draws for precomputing f_i, rest for psi samples
        theta_draws = subject_states[:n_theta_draws]  # (K, n_subjects, nparams)
        psi_draws = hyper_states[n_theta_draws:n_theta_draws + n_psi_samples]
        n_psi = len(psi_draws)
        K = n_theta_draws
        print(f"  Using {K} theta draws for IS, {n_psi} psi draws for bridge")

        # Step 1: Precompute log f_i(theta_i^(k)) for all subjects and draws
        # Build a JIT function that computes log f for ALL subjects at once
        # given a full subject-params vector, using fori_loop
        log_f_fn = _build_log_f_fn(model_key, indexed_data)
        log_pop_fn = pop_prior_fns[model_key]

        @jax.jit
        def all_subjects_log_f(all_theta):
            """Compute log f_i for all subjects. all_theta: (n_subjects, nparams)."""
            def body(sid, carry):
                theta_i = all_theta[sid]
                lf = log_f_fn(theta_i, sid)
                return carry.at[sid].set(lf)
            return jax.lax.fori_loop(0, n_subjects, body,
                                     jnp.zeros(n_subjects))

        print(f"  Precomputing log f_i for {n_subjects} subjects x {K} draws...")
        log_f_cache = np.empty((K, n_subjects))
        for k in range(K):
            all_theta_k = jnp.array(theta_draws[k])  # (n_subjects, nparams)
            log_f_cache[k] = np.array(all_subjects_log_f(all_theta_k))
            if k % 20 == 0:
                print(f"    k={k}/{K}")

        n_nan_f = np.sum(~np.isfinite(log_f_cache))
        print(f"  log f cache: {n_nan_f} non-finite out of {K * n_subjects}")

        theta_cache = theta_draws  # (K, n_subjects, nparams)

        # Step 2: Build a JIT function that computes log p(y|psi) for a given psi
        # using the precomputed log_f_cache and the theta draws.
        #
        # log p(y_i|psi) ≈ -log(K) + LSE_k[log_f_cache[k,i] + log p(theta_i^(k)|psi)]
        # log p(y|psi) = sum_i log p(y_i|psi)
        log_f_cache_jax = jnp.array(log_f_cache)  # (K, n_subjects)
        theta_cache_jax = jnp.array(theta_cache)    # (K, n_subjects, nparams)

        @jax.jit
        def log_p_y_given_psi_jit(psi_jax):
            """Estimate log p(y|psi) via importance sampling."""
            def subject_body(sid, total):
                def draw_body(k, log_w_arr):
                    theta_ik = theta_cache_jax[k, sid]
                    log_pop = log_pop_fn(theta_ik, psi_jax)
                    log_w = log_f_cache_jax[k, sid] + log_pop
                    return log_w_arr.at[k].set(log_w)

                log_w = jax.lax.fori_loop(0, K, draw_body, jnp.full(K, -jnp.inf))
                # LSE over K draws
                max_w = jnp.max(log_w)
                log_p_yi = -jnp.log(K) + max_w + jnp.log(
                    jnp.sum(jnp.exp(log_w - max_w)))
                return total + log_p_yi

            return jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)

        def log_p_y_given_psi(psi):
            return float(log_p_y_given_psi_jit(jnp.array(psi)))

        # Evaluate log p_tilde(psi) = log pi(psi) + log p(y|psi) at posterior psi draws
        print(f"  Evaluating log p_tilde at {n_psi} posterior psi draws...")
        log_ptilde_post = np.empty(n_psi)
        for s in range(n_psi):
            lp_hyper = float(log_hyperprior_fn(jnp.array(psi_draws[s])))
            lp_data = log_p_y_given_psi(psi_draws[s])
            log_ptilde_post[s] = lp_hyper + lp_data
            if s % 500 == 0:
                print(f"    {s}/{n_psi}: log p_tilde = {log_ptilde_post[s]:.1f}")

        n_fin = int(np.sum(np.isfinite(log_ptilde_post)))
        print(f"  Posterior: {n_fin}/{n_psi} finite")

        # Fit 8-dim MVN proposal to psi posterior
        n_hyper = psi_draws.shape[1]
        psi_mean = np.mean(psi_draws, axis=0)
        psi_cov = np.cov(psi_draws, rowvar=False) + 1e-8 * np.eye(n_hyper)
        psi_L = np.linalg.cholesky(psi_cov)
        psi_L_inv = np.linalg.inv(psi_L)
        psi_log_det_L = np.sum(np.log(np.abs(np.diag(psi_L))))
        psi_log_norm = -0.5 * n_hyper * np.log(2 * np.pi) - psi_log_det_L

        # Draw proposal psi samples
        rng = np.random.default_rng(seed + 5000)
        z = rng.standard_normal((n_psi, n_hyper))
        psi_proposals = psi_mean + z @ psi_L.T

        # Evaluate log p_tilde at proposal psi draws
        print(f"  Evaluating log p_tilde at {n_psi} proposal psi draws...")
        log_ptilde_prop = np.empty(n_psi)
        for s in range(n_psi):
            lp_hyper = float(log_hyperprior_fn(jnp.array(psi_proposals[s])))
            lp_data = log_p_y_given_psi(psi_proposals[s])
            log_ptilde_prop[s] = lp_hyper + lp_data
            if s % 500 == 0:
                print(f"    {s}/{n_psi}: log p_tilde = {log_ptilde_prop[s]:.1f}")

        n_fin_prop = int(np.sum(np.isfinite(log_ptilde_prop)))
        print(f"  Proposals: {n_fin_prop}/{n_psi} finite")

        # Evaluate log q(psi) at all points
        def log_q(psi_val):
            diff = psi_val - psi_mean
            z = psi_L_inv @ diff
            return psi_log_norm - 0.5 * np.dot(z, z)

        log_q_post = np.array([log_q(h) for h in psi_draws])
        log_q_prop = np.array([log_q(h) for h in psi_proposals])

        # Filter to finite
        vp = np.isfinite(log_ptilde_post)
        vq = np.isfinite(log_ptilde_prop)
        lp_post = log_ptilde_post[vp]
        lp_prop = log_ptilde_prop[vq]
        lq_post = log_q_post[vp]
        lq_prop = log_q_prop[vq]

        n2, n1 = len(lp_post), len(lp_prop)
        s1 = n1 / (n1 + n2)
        s2 = n2 / (n1 + n2)

        log_r = float(np.median(lp_post - lq_post))
        print(f"  Initial log r = {log_r:.2f}")
        print(f"  Bridge sampling ({n2} posterior + {n1} proposal)...")

        for it in range(n_iter):
            log_r_old = log_r

            log_num = lp_prop - np.logaddexp(
                np.log(s1) + lp_prop, np.log(s2) + lq_prop + log_r)
            log_numerator = -np.log(n1) + _logsumexp(log_num)

            log_den = lq_post - np.logaddexp(
                np.log(s1) + lp_post, np.log(s2) + lq_post + log_r)
            log_denominator = -np.log(n2) + _logsumexp(log_den)

            log_r = log_numerator - log_denominator

            if it < 5 or it % 20 == 0:
                print(f"    iter {it:4d}: log ML = {log_r:.4f} "
                      f"(change = {abs(log_r - log_r_old):.2e})")

            if abs(log_r - log_r_old) < 1e-10 and it > 2:
                print(f"    Converged at iteration {it}")
                break

        results[model_key] = {
            'log_ml': float(log_r),
            'converged': abs(log_r - log_r_old) < 1e-10,
        }
        print(f"  log ML = {results[model_key]['log_ml']:.2f} "
              f"(converged: {results[model_key]['converged']})")

    # Bayes factors
    print("\n" + "=" * 60)
    print("BAYES FACTORS (log scale)")
    print("=" * 60)

    save_dict = {}
    for key, res in results.items():
        save_dict[f'{key}_log_ml_rb'] = res['log_ml']

    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            log_bf = results[k1]['log_ml'] - results[k2]['log_ml']
            print(f"  log BF({k1} vs {k2}) = {log_bf:.2f}")
            save_dict[f'log_bf_rb_{k1}_vs_{k2}'] = log_bf

    np.savez(Path(output_path), **save_dict)
    print(f"\nSaved to: {output_path}")
    return results
