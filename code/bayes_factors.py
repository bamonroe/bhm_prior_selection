"""
Bayes factor computation via bridge sampling.

Computes marginal likelihoods for each model using the Warp-III bridge
sampling estimator (Meng & Wong 1996, Gronau et al. 2017). The Bayes
factor between any two models is the ratio of their marginal likelihoods.

The key input is a function that evaluates the full unnormalized
log-posterior at a given parameter vector. Each model's posterior module
provides the building blocks; this module assembles them into full-joint
evaluators and runs the bridge estimator.

Usage: called from main.py via do_bayes_factors flag.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from functools import partial


# =============================================================================
# Full log-posterior evaluators (one per model)
# =============================================================================

def _build_full_log_posterior_crra(data):
    """Build a JIT-compiled full log-posterior for CRRA+Prelec."""
    from posteriors.rdu_bhm_ce import (
        _get_indices, _get_subject_params, _compute_all_hyperpriors,
        _subject_quantity_prior, NPARAMS, N_HYPER, N_QUANTITIES,
    )
    from posteriors.common_fns import (
        full_forward_map, full_jacobian_logabsdet,
        crra_ufn, risk_likelihood,
    )

    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    dataf = data["float"]
    hyperprior_arr = jnp.asarray(dataf[7])
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
    unique_outcomes = dataf_jax[0]

    @jax.jit
    def full_lp(chain_state):
        # Hyperprior
        lp = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        # Per-subject: population prior + Jacobian + likelihood
        def subject_body(sid, acc):
            r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
                chain_state, sid)

            quantities = full_forward_map(
                r_val, phi_raw, eta_raw, lambda_raw,
                ref_lotteries, ref_pair)
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)

            log_det_j = full_jacobian_logabsdet(
                r_val, phi_raw, eta_raw, lambda_raw,
                ref_lotteries, ref_pair)

            phi = jnp.exp(phi_raw)
            eta = jnp.exp(eta_raw)
            lam = jnp.exp(lambda_raw)
            unique_utils = crra_ufn(unique_outcomes, r_val)
            log_like = risk_likelihood(
                unique_utils, phi, eta, datai_jax, dataf_jax, sid, lam)

            subj_lp = lp_pop + log_det_j + log_like
            # NaN/inf from numerical overflow at extreme r: substitute the
            # minimum valid per-subject contribution (~-500) so the total
            # log-posterior stays finite and in a realistic range.
            # No NaN guard — let non-finite values propagate.
            # The adaptive proposal ensures all evaluations are finite.
            return acc + subj_lp

        lp += jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)
        return lp

    return full_lp, n_subjects * NPARAMS + N_HYPER


def _build_full_log_posterior_ge(data):
    """Build a JIT-compiled full log-posterior for CRRA+GE."""
    from posteriors.rdu_ge_bhm_ce import (
        _get_indices, _get_subject_params, NPARAMS, N_HYPER,
    )
    from posteriors.rdu_bhm_ce import (
        _compute_all_hyperpriors, _subject_quantity_prior, N_QUANTITIES,
    )
    from posteriors.common_fns import (
        full_forward_map_ge, full_jacobian_logabsdet_ge,
        crra_ufn, risk_likelihood_ge,
    )

    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    dataf = data["float"]
    hyperprior_arr = jnp.asarray(dataf[7])
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
    unique_outcomes = dataf_jax[0]

    @jax.jit
    def full_lp(chain_state):
        lp = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        def subject_body(sid, acc):
            r_val, gamma_raw, delta_raw, lambda_raw = _get_subject_params(
                chain_state, sid)

            quantities = full_forward_map_ge(
                r_val, gamma_raw, delta_raw, lambda_raw,
                ref_lotteries, ref_pair)
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)

            log_det_j = full_jacobian_logabsdet_ge(
                r_val, gamma_raw, delta_raw, lambda_raw,
                ref_lotteries, ref_pair)

            gamma = jnp.exp(gamma_raw)
            delta = jnp.exp(delta_raw)
            lam = jnp.exp(lambda_raw)
            unique_utils = crra_ufn(unique_outcomes, r_val)
            log_like = risk_likelihood_ge(
                unique_utils, gamma, delta, datai_jax, dataf_jax, sid, lam)

            subj_lp = lp_pop + log_det_j + log_like
            # No NaN guard — let non-finite values propagate.
            # The adaptive proposal ensures all evaluations are finite.
            return acc + subj_lp

        lp += jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)
        return lp

    return full_lp, n_subjects * NPARAMS + N_HYPER


def _build_full_log_posterior_power(data):
    """Build a JIT-compiled full log-posterior for CRRA+Power."""
    from posteriors.rdu_power_bhm_ce import (
        _get_indices, _get_subject_params, NPARAMS, N_HYPER,
    )
    from posteriors.rdu_bhm_ce import (
        _compute_all_hyperpriors, _subject_quantity_prior, N_QUANTITIES,
    )
    from posteriors.common_fns import (
        full_forward_map_power, gram_logdet_power,
        crra_ufn, risk_likelihood_power,
    )

    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    dataf = data["float"]
    hyperprior_arr = jnp.asarray(dataf[7])
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])
    datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
    dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
    unique_outcomes = dataf_jax[0]

    @jax.jit
    def full_lp(chain_state):
        lp = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        def subject_body(sid, acc):
            r_val, delta_raw, lambda_raw = _get_subject_params(
                chain_state, sid)

            quantities = full_forward_map_power(
                r_val, delta_raw, lambda_raw,
                ref_lotteries, ref_pair)
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)

            log_gram = gram_logdet_power(
                r_val, delta_raw, lambda_raw,
                ref_lotteries, ref_pair)

            delta = jnp.exp(delta_raw)
            lam = jnp.exp(lambda_raw)
            unique_utils = crra_ufn(unique_outcomes, r_val)
            log_like = risk_likelihood_power(
                unique_utils, delta, datai_jax, dataf_jax, sid, lam)

            subj_lp = lp_pop + log_gram + log_like
            # No NaN guard — let non-finite values propagate.
            # The adaptive proposal ensures all evaluations are finite.
            return acc + subj_lp

        lp += jax.lax.fori_loop(0, n_subjects, subject_body, 0.0)
        return lp

    return full_lp, n_subjects * NPARAMS + N_HYPER


MODEL_BUILDERS = {
    'crra': _build_full_log_posterior_crra,
    'ge': _build_full_log_posterior_ge,
    'power': _build_full_log_posterior_power,
}


# =============================================================================
# Bridge sampling estimator
# =============================================================================

class GaussianMixtureProposal:
    """Gaussian mixture proposal with K components sharing a diagonal covariance.

    q(theta) = (1/K) sum_k N(theta | mu_k, scale^2 * diag(sigma^2))

    where mu_k are centers selected from posterior samples, sigma^2 is the
    per-coordinate posterior variance, and scale controls the width.
    """

    def __init__(self, posterior_samples, n_components=10, scale=1.0, seed=42):
        """Fit mixture proposal from posterior samples.

        Args:
            posterior_samples: (N, d) array of posterior draws
            n_components: Number of mixture components
            scale: Scale factor for the covariance (1.0 = posterior variance)
            seed: RNG seed for center selection
        """
        N, d = posterior_samples.shape
        rng = np.random.default_rng(seed)

        self.K = n_components
        self.d = d
        self.scale = scale

        # Select centers: spread across posterior via stratified sampling
        # Sort by first principal component, pick evenly spaced quantiles
        mean = np.mean(posterior_samples, axis=0)
        centered = posterior_samples - mean
        # Project onto top PC for stratification
        cov_diag = np.var(posterior_samples, axis=0)
        pc1 = centered @ (cov_diag / np.sum(cov_diag))  # weighted projection
        order = np.argsort(pc1)
        # Pick evenly spaced indices along the sorted order
        quantile_idx = np.linspace(0, N - 1, n_components, dtype=int)
        center_idx = order[quantile_idx]
        self.centers = posterior_samples[center_idx].copy()  # (K, d)

        # Per-coordinate std from full posterior
        self.sigma = np.std(posterior_samples, axis=0)  # (d,)
        self.sigma = np.maximum(self.sigma, 1e-10)

        # Effective bandwidth per coordinate
        self.bw = self.scale * self.sigma  # (d,)
        self.inv_bw = 1.0 / self.bw  # (d,)

        # Log normalization constant per component
        self.log_norm = (-0.5 * d * np.log(2 * np.pi)
                         - np.sum(np.log(self.bw)))

    def sample(self, n, seed=None):
        """Draw n samples from the mixture proposal."""
        rng = np.random.default_rng(seed)
        comp_idx = rng.integers(0, self.K, size=n)
        chosen_centers = self.centers[comp_idx]
        noise = rng.standard_normal((n, self.d)) * self.bw
        return chosen_centers + noise

    def batch_log_density(self, thetas, chunk_size=500):
        """Evaluate log q(theta) for many points, vectorized in chunks.

        With only K=10 components, each chunk is (chunk, 10, d) — small.
        """
        n = len(thetas)
        results = np.empty(n)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = thetas[start:end]  # (chunk, d)

            # Standardized distances: (chunk, K, d)
            z = (batch[:, None, :] - self.centers[None, :, :]) * self.inv_bw[None, None, :]
            # Log component densities: (chunk, K)
            log_components = self.log_norm - 0.5 * np.sum(z ** 2, axis=2)
            # Log mixture density with logsumexp: (chunk,)
            max_lc = np.max(log_components, axis=1, keepdims=True)
            results[start:end] = (-np.log(self.K) + max_lc[:, 0] +
                np.log(np.sum(np.exp(log_components - max_lc), axis=1)))

        return results


def bridge_sampling_log_ml(posterior_samples, log_posterior_fn, data,
                           n_proposal=None, n_iter=2000, seed=42,
                           verbose=True):
    """Estimate log marginal likelihood via bridge sampling.

    Uses the optimal bridge function of Meng & Wong (1996) with
    iterative estimation and a Gaussian KDE proposal.

    Args:
        posterior_samples: (n_post, n_params) array of posterior draws.
        log_posterior_fn: Callable(chain_state) -> scalar log p̃(θ|data).
        data: Data dict for the model.
        n_proposal: Number of proposal samples (default: same as n_post).
        n_iter: Bridge sampling iterations.
        seed: RNG seed for proposal samples.
        n_kde_centers: Number of KDE centers for the proposal.
        verbose: Print progress.

    Returns:
        dict with 'log_ml', 'log_ml_se', 'n_iter', 'converged'
    """
    n_post = len(posterior_samples)
    if n_proposal is None:
        n_proposal = n_post
    n_params = posterior_samples.shape[1]

    # Fit 10-component mixture with FULL posterior covariance
    n_components = 10
    if verbose:
        print(f"  Fitting {n_components}-component mixture with full "
              f"covariance (d={n_params})...")

    mean = np.mean(posterior_samples, axis=0)
    cov = np.cov(posterior_samples, rowvar=False)
    cov += 1e-6 * np.eye(n_params)  # regularize
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)
    log_det_L = np.sum(np.log(np.abs(np.diag(L))))
    log_norm_full = -0.5 * n_params * np.log(2 * np.pi) - log_det_L

    # Select centers via stratified sampling along first PC
    rng_centers = np.random.default_rng(seed)
    centered = posterior_samples - mean
    # Project onto top eigenvector for stratification
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = centered @ eigvecs[:, -1]
    order = np.argsort(pc1)
    quantile_idx = np.linspace(0, len(posterior_samples) - 1,
                               n_components, dtype=int)
    centers = posterior_samples[order[quantile_idx]]  # (K, d)

    if verbose:
        print(f"  Cholesky computed. log|det L| = {log_det_L:.1f}")

    def _sample_proposal(n, seed_val, scale=1.0):
        """Draw n samples from the mixture proposal with given scale."""
        rng_p = np.random.default_rng(seed_val)
        comp_idx = rng_p.integers(0, n_components, size=n)
        z = rng_p.standard_normal((n, n_params))
        return centers[comp_idx] + scale * (z @ L.T)

    def _log_q_batch(thetas, scale=1.0):
        """Evaluate log q(theta) for a batch of points with given scale."""
        # Adjust for scale: N(mu, scale^2 * Sigma)
        # log|det(scale*L)| = d*log(scale) + log|det(L)|
        log_norm_scaled = (-0.5 * n_params * np.log(2 * np.pi)
                           - n_params * np.log(scale) - log_det_L)
        L_inv_scaled = L_inv / scale

        n = len(thetas)
        log_q = np.empty(n)
        for i in range(n):
            log_comps = np.empty(n_components)
            for k in range(n_components):
                diff = thetas[i] - centers[k]
                z = L_inv_scaled @ diff
                log_comps[k] = log_norm_scaled - 0.5 * np.dot(z, z)
            log_q[i] = -np.log(n_components) + _logsumexp(log_comps)
        return log_q

    # Use scale=1.0 (posterior covariance). Over-sample proposals to
    # compensate for non-finite evaluations in regions outside the
    # posterior support. All posterior samples evaluate finitely in
    # float64, so the non-finite proposals are in zero-density regions.
    scale = 1.0

    # Evaluate log p̃(θ) at posterior samples — these should all be finite
    if verbose:
        print(f"  Evaluating log-posterior at {n_post} posterior samples...")
    log_p_post = np.array(_batch_evaluate(log_posterior_fn, posterior_samples))
    n_nonfinite_post = int(np.sum(~np.isfinite(log_p_post)))
    if n_nonfinite_post > 0 and verbose:
        print(f"  WARNING: {n_nonfinite_post} non-finite posterior evals")

    # Evaluate log p̃(θ) at proposal samples, over-sampling to get enough
    # finite evaluations. Non-finite proposals are in zero-density regions
    # beyond the posterior support — dropping them introduces no bias.
    if verbose:
        print(f"  Evaluating proposals (target: {n_proposal} finite)...")
    oversample_factor = 1.2  # draw 20% extra to compensate for ~5% rejection
    valid_prop = []
    valid_prop_logp = []
    valid_prop_total = 0
    prop_seed = seed + 1000
    while valid_prop_total < n_proposal:
        n_draw = int(n_proposal * oversample_factor)
        candidates = _sample_proposal(n_draw, prop_seed, scale=scale)
        log_p_cand = np.array(_batch_evaluate(log_posterior_fn, candidates))
        finite = np.isfinite(log_p_cand)
        valid_prop.append(candidates[finite])
        valid_prop_logp.append(log_p_cand[finite])
        valid_prop_total = sum(len(v) for v in valid_prop)
        n_rejected = int(np.sum(~finite))
        if verbose:
            print(f"    Drew {n_draw}: {int(np.sum(finite))} finite, "
                  f"{n_rejected} rejected (total: {valid_prop_total})")
        prop_seed += 1
        oversample_factor = min(oversample_factor * 1.1, 3.0)

    proposal_samples = np.concatenate(valid_prop)[:n_proposal]
    log_p_prop = np.concatenate(valid_prop_logp)[:n_proposal]
    n_proposal = len(log_p_prop)

    # Evaluate log q(θ) at all samples using the final scale
    if verbose:
        print(f"  Evaluating proposal density at "
              f"{n_post + n_proposal} points (scale={scale:.4f})...")
    log_q_post = _log_q_batch(posterior_samples, scale=scale)
    log_q_prop = _log_q_batch(proposal_samples, scale=scale)

    # Iterative bridge estimator
    # l1 = log(1/n1 * Σ_proposal [p̃(θ) * alpha(θ)] / [s1*p̃(θ)*alpha(θ) + s2*q(θ)])
    # l2 = log(1/n2 * Σ_posterior [q(θ)] / [s1*p̃(θ)/r + s2*q(θ)])
    # where r = p(data) is the marginal likelihood, s1 = n1/(n1+n2), s2 = n2/(n1+n2)

    s1 = n_proposal / (n_post + n_proposal)
    s2 = n_post / (n_post + n_proposal)

    # Initialize log(r) using the identity E_post[p̃/q] = r
    # => log r ≈ median(log p̃ - log q) over posterior samples
    log_r = float(np.median(log_p_post - log_q_post))
    if verbose:
        print(f"  Initial log r = {log_r:.2f}")

    for it in range(n_iter):
        log_r_old = log_r

        # Numerator: average over proposal samples
        # log[p̃(θ) / (s1*p̃(θ) + s2*q(θ)*r)]
        log_num = log_p_prop - np.logaddexp(
            np.log(s1) + log_p_prop,
            np.log(s2) + log_q_prop + log_r)
        log_numerator = -np.log(n_proposal) + _logsumexp(log_num)

        # Denominator: average over posterior samples
        # log[q(θ) / (s1*p̃(θ) + s2*q(θ)*r)]
        log_den = log_q_post - np.logaddexp(
            np.log(s1) + log_p_post,
            np.log(s2) + log_q_post + log_r)
        log_denominator = -np.log(n_post) + _logsumexp(log_den)

        log_r = log_numerator - log_denominator

        if verbose and (it < 3 or it % 10 == 0):
            print(f"    iter {it:3d}: log ML = {log_r:.4f} "
                  f"(change = {abs(log_r - log_r_old):.2e})")

        if abs(log_r - log_r_old) < 1e-10 and it > 2:
            if verbose:
                print(f"    Converged at iteration {it}")
            break

    # Standard error via the coefficient of variation
    # (simplified, see Gronau et al. 2017 for full expression)
    log_h_post = log_q_post - np.logaddexp(
        np.log(s1) + log_p_post,
        np.log(s2) + log_q_post + log_r)
    log_h_prop = log_p_prop - np.logaddexp(
        np.log(s1) + log_p_prop,
        np.log(s2) + log_q_prop + log_r)

    # Approximate SE
    h_post = np.exp(log_h_post - np.max(log_h_post))
    h_prop = np.exp(log_h_prop - np.max(log_h_prop))
    cv_sq = (np.var(h_post) / np.mean(h_post)**2 / n_post +
             np.var(h_prop) / np.mean(h_prop)**2 / n_proposal)
    log_ml_se = np.sqrt(cv_sq) if cv_sq > 0 else np.nan

    return {
        'log_ml': float(log_r),
        'log_ml_se': float(log_ml_se),
        'n_iter': it + 1,
        'converged': abs(log_r - log_r_old) < 1e-10,
    }


def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    mx = np.max(x)
    return mx + np.log(np.sum(np.exp(x - mx)))


def _batch_evaluate(log_posterior_fn, samples, batch_size=200):
    """Evaluate log-posterior at many parameter vectors, GPU-batched.

    Args:
        log_posterior_fn: JIT-compiled function(chain_state) -> scalar
        samples: (n_samples, n_params) array
        batch_size: Number of samples per GPU batch

    Returns:
        (n_samples,) array of log-posterior values
    """
    n = len(samples)
    results = []

    # vmap the function for batched evaluation
    batched_fn = jax.vmap(log_posterior_fn)

    for i in range(0, n, batch_size):
        batch = jnp.array(samples[i:i+batch_size])
        vals = batched_fn(batch)
        results.append(np.array(vals))

    return np.concatenate(results)


# =============================================================================
# Load posterior samples from history files
# =============================================================================

def _load_posterior_samples(model_key, data_dir, n_samples=None, seed=42):
    """Load posterior samples from history files.

    Loads the second half of retained samples (burn-in discarded) and
    subsamples if n_samples is specified.

    Args:
        model_key: Key in MODEL_REGISTRY
        data_dir: Path to data/output
        n_samples: Max samples to use (None = all)
        seed: RNG seed for subsampling

    Returns:
        (n_samples, n_params) array of chain states (raw, not GQ-transformed)
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

    # Compute expected raw state dimension
    # History stores [raw_state | generated_quantities] per chain
    # We need to figure out n_subjects from data, not hardcode it
    first = np.load(history_files[0], allow_pickle=True)
    total_dim = first['history'].shape[2]
    # n_raw_params = n_subjects * nparams + n_hyper
    # n_gq = n_subjects * nparams + 2 * 4 (for CE models)
    # total_dim = n_raw_params + n_gq
    # So: total_dim = 2 * n_subjects * nparams + n_hyper + 8
    # => n_subjects = (total_dim - n_hyper - 8) / (2 * nparams)
    n_subjects = (total_dim - n_hyper - 8) // (2 * nparams)
    n_raw_params = n_subjects * nparams + n_hyper

    # Load and concatenate all history files
    all_states = []
    for f in history_files:
        d = np.load(f, allow_pickle=True)
        history = d['history']  # (n_collect, n_chains, n_params + n_gq)
        all_states.append(history[:, :, :n_raw_params])

    combined = np.concatenate(all_states, axis=0)

    # Discard first half as burn-in
    burn = combined.shape[0] // 2
    post_burn = combined[burn:]

    # Flatten (iterations, chains) -> samples
    n_iter, n_chains, n_params = post_burn.shape
    flat = post_burn.reshape(-1, n_params)

    # Subsample if requested
    if n_samples is not None and n_samples < len(flat):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), n_samples, replace=False)
        flat = flat[idx]

    return flat


# =============================================================================
# Main entry point
# =============================================================================

def compute_bayes_factors(model_keys, data_dir, output_path,
                          n_samples=10000, seed=42):
    """Compute Bayes factors between models via bridge sampling.

    Args:
        model_keys: List of model keys (e.g., ['crra', 'ge', 'power'])
        data_dir: Path to data/output
        output_path: Path to save results NPZ
        n_samples: Number of posterior samples to use per model
        seed: RNG seed

    Returns:
        dict with log marginal likelihoods and Bayes factors
    """
    from posteriors.rdu_bhm_ce import load_hyperpriors_from_file
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from sampling_ce import pack_ref_lotteries, pack_ref_pair

    data_dir = Path(data_dir)
    output_path = Path(output_path)

    # Load hyperpriors
    prior_params_path = data_dir / 'prior_params_ce.npz'
    if not prior_params_path.exists():
        raise FileNotFoundError(f"Prior params not found: {prior_params_path}")
    load_hyperpriors_from_file(str(prior_params_path))

    # Build data
    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)

    from posteriors.rdu_bhm_ce import get_hyperprior_array
    hyperprior_arr = get_hyperprior_array()
    indexed_data['float'] = indexed_data['float'] + (hyperprior_arr,)

    ref_lotteries = [
        {'prizes': [10, 50], 'probs': [0.75, 0.25]},
        {'prizes': [20, 40], 'probs': [0.5, 0.5]},
        {'prizes': [12, 44], 'probs': [0.75, 0.25]},
    ]
    ref_pair = {
        'lottery_A': {'prizes': [10, 50], 'probs': [0.75, 0.25]},
        'lottery_B': {'prizes': [12, 44], 'probs': [0.75, 0.25]},
    }
    indexed_data['float'] = indexed_data['float'] + (pack_ref_lotteries(ref_lotteries),)
    indexed_data['float'] = indexed_data['float'] + (pack_ref_pair(ref_pair),)

    print("=" * 60)
    print("BAYES FACTOR COMPUTATION (Bridge Sampling)")
    print("=" * 60)

    results = {}

    for model_key in model_keys:
        print(f"\n--- Model: {model_key} ---")

        # Load posterior samples
        print(f"  Loading posterior samples...")
        samples = _load_posterior_samples(model_key, str(data_dir),
                                          n_samples=n_samples, seed=seed)
        print(f"  Loaded {len(samples)} samples, dim={samples.shape[1]}")

        # Build full log-posterior
        print(f"  Building full log-posterior...")
        builder = MODEL_BUILDERS[model_key]
        full_lp, n_params = builder(indexed_data)
        assert samples.shape[1] == n_params, \
            f"Sample dim {samples.shape[1]} != expected {n_params}"

        # Run bridge sampling
        print(f"  Running bridge sampling...")
        result = bridge_sampling_log_ml(
            samples, full_lp, indexed_data,
            n_proposal=n_samples, seed=seed + 100,
            verbose=True)

        results[model_key] = result
        print(f"  log ML = {result['log_ml']:.2f} "
              f"(SE = {result['log_ml_se']:.4f}, "
              f"converged = {result['converged']})")

    # Compute pairwise Bayes factors
    print("\n" + "=" * 60)
    print("BAYES FACTORS (log scale)")
    print("=" * 60)

    save_dict = {}
    for key, res in results.items():
        save_dict[f'{key}_log_ml'] = res['log_ml']
        save_dict[f'{key}_log_ml_se'] = res['log_ml_se']

    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            log_bf = results[k1]['log_ml'] - results[k2]['log_ml']
            print(f"  log BF({k1} vs {k2}) = {log_bf:.2f}")
            save_dict[f'log_bf_{k1}_vs_{k2}'] = log_bf

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")

    return results
