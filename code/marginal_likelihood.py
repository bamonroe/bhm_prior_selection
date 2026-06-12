"""
Corrected Rao-Blackwellized marginal likelihood estimation.

Replaces the integrated-likelihood estimator in bayes_factors_rb.py, whose
per-subject integral used posterior theta draws without dividing by their
sampling density (not a valid importance-sampling estimate).

Decomposition:

    log p(y) = log integral pi(psi) prod_i p(y_i|psi) dpsi

    p(y_i|psi) = integral f_i(theta) p_pop(theta|psi) dtheta
              ~= (1/K) sum_k f_i(theta_ik) p_pop(theta_ik|psi) / q_i(theta_ik)

with theta_ik drawn FRESH from an explicit multivariate Student-t proposal
q_i fitted to subject i's marginal posterior draws. f_i = likelihood x
|Jacobian| (psi-independent), reused from bayes_factors_rb._build_log_f_fn.

The psi-dependence of log p_pop factorizes. For quantity q with bounds
(lo_q, hi_q) and Beta population parameters (a_q(psi), b_q(psi)):

    log p_pop(theta|psi) = sum_q [(a_q - 1) log z_q + (b_q - 1) log(1 - z_q)
                                  - log B(a_q, b_q) - log(hi_q - lo_q)]

where z_q = clip((Y_q(theta) - lo_q)/(hi_q - lo_q)) depends only on theta.
So log z / log(1-z) caches of shape (K, n_subjects, 4) are computed once and
each psi evaluation is a pair of tensor contractions.

The outer 8-D psi integral is computed two independent ways:
    (a) importance sampling from a Student-t fitted to the psi posterior
    (b) Meng-Wong bridge sampling between psi posterior draws and that
        same Student-t proposal
Agreement within standard errors is required before the result is trusted.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path

# =============================================================================
# Per-subject f_i(theta_i) = p(y_i|theta_i) |J(theta_i)| and the psi-dependent
# population prior (originally in bayes_factors_rb.py, retired June 2026)
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

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return log_f


def _build_log_pop_prior_fn(data):
    """Build function: log p(theta_i | psi) via the Beta-Beta hierarchy.

    Reference implementation (forward map + _subject_quantity_prior),
    used by the vectorization unit test.
    """
    from posteriors.rdu_bhm_ce import _subject_quantity_prior, _get_indices

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


def _build_log_hyperprior(data):
    from posteriors.rdu_bhm_ce import _compute_all_hyperpriors, _get_indices
    dataf = data["float"]
    hyperprior_arr = jnp.asarray(dataf[7])

    @jax.jit
    def log_hyperprior(psi):
        idx = _get_indices(0)
        return _compute_all_hyperpriors(psi, idx, hyperprior_arr)

    return log_hyperprior


def _logsumexp(x):
    mx = np.max(x)
    return mx + np.log(np.sum(np.exp(x - mx)))


def load_posterior_states(model_key, data_dir, n_samples=None, seed=42):
    """Load posterior states, excluding parallel-tempering runs.

    History directories can mix plain posterior runs with parallel
    tempering runs made for thermodynamic integration (e.g. the May 27
    rdu_bhm_ce files have n_temperatures=20, so only ~1/20 of their
    saved chain states are beta=1 posterior draws — treating them as
    posterior samples contaminates everything downstream). Each history
    file stores its mcmc_config; keep only files with n_temperatures=1,
    require them to share one shape, and drop the first half of their
    iterations as burn-in.
    """
    from model_registry import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg['model_id']
    nparams = cfg['nparams']
    n_hyper = cfg['n_hyper']

    history_dir = Path(data_dir) / model_id / 'history' / 'full'
    files = sorted(history_dir.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f"No history files in {history_dir}")

    group = []
    shapes = []
    n_skipped = 0
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            run_cfg = d['mcmc_config'].item() if 'mcmc_config' in d else {}
            if run_cfg.get('n_temperatures', 1) > 1:
                n_skipped += 1
                continue
            group.append(f)
            shapes.append(d['history'].shape)
    if n_skipped:
        print(f"  NOTE: skipped {n_skipped} parallel-tempering history "
              f"file(s) (n_temperatures > 1); using {len(group)} plain "
              f"posterior file(s)")
    if not group:
        raise FileNotFoundError(
            f"No untempered history files in {history_dir}")
    if len(set(shapes)) > 1:
        raise RuntimeError(
            f"Untempered history files have mixed shapes: {set(shapes)}")

    # State layout: [theta (n*p), psi (n_hyper)] + GQ [theta_nat (n*p),
    # pop params (2*N_QUANTITIES)], so total = 2*n*p + n_hyper + 2*N_QUANTITIES
    total_dim = shapes[0][2]
    n_subjects = (total_dim - n_hyper - 2 * N_QUANTITIES) // (2 * nparams)
    n_raw = n_subjects * nparams + n_hyper

    states = [np.load(f, allow_pickle=True)['history'][:, :, :n_raw]
              for f in group]
    combined = np.concatenate(states, axis=0)
    burn = combined.shape[0] // 2
    flat = combined[burn:].reshape(-1, n_raw)

    if n_samples is not None and n_samples < len(flat):
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(len(flat), n_samples, replace=False)]

    subject = flat[:, :n_subjects * nparams].reshape(-1, n_subjects, nparams)
    hyper = flat[:, n_subjects * nparams:n_raw]
    return hyper, subject, n_subjects

N_QUANTITIES = 4
HYPERPRIOR_STRIDE = 6

# Same reference lotteries/pair as the sampling runs (cf. sampling_ce.py,
# bayes_factors_rb.py) — these define the quantity-space prior.
REFERENCE_LOTTERIES = [
    {'prizes': [10, 50], 'probs': [0.75, 0.25]},
    {'prizes': [20, 40], 'probs': [0.5, 0.5]},
    {'prizes': [12, 44], 'probs': [0.75, 0.25]},
]
REFERENCE_PAIR = {
    'lottery_A': {'prizes': [10, 50], 'probs': [0.75, 0.25]},
    'lottery_B': {'prizes': [12, 44], 'probs': [0.75, 0.25]},
}


def assemble_data(data_dir):
    """Build the indexed data tuple expected by the log_f / pop-prior builders."""
    from posteriors.rdu_bhm_ce import (
        load_hyperpriors_from_file, get_hyperprior_array)
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from sampling_ce import pack_ref_lotteries, pack_ref_pair

    load_hyperpriors_from_file(str(Path(data_dir) / 'prior_params_ce.npz'))
    clean = load_clean_data()
    data = make_bamcmc_data_indexed(clean)
    data['float'] = data['float'] + (get_hyperprior_array(),)
    data['float'] = data['float'] + (pack_ref_lotteries(REFERENCE_LOTTERIES),)
    data['float'] = data['float'] + (pack_ref_pair(REFERENCE_PAIR),)
    return data


# =============================================================================
# Multivariate Student-t proposal
# =============================================================================

class StudentTProposal:
    """Multivariate Student-t fitted to draws: location = mean, shape =
    sample covariance scaled by `scale**2`, fixed degrees of freedom."""

    def __init__(self, draws, df=5.0, scale=1.5, jitter=1e-9):
        draws = np.asarray(draws, dtype=np.float64)
        cov = np.cov(draws, rowvar=False).reshape(draws.shape[1], draws.shape[1])
        self._init_from_moments(draws.mean(axis=0),
                                scale ** 2 * cov, df, jitter)

    @classmethod
    def from_moments(cls, mean, cov, df=5.0, jitter=1e-9):
        obj = cls.__new__(cls)
        obj._init_from_moments(np.asarray(mean, dtype=np.float64),
                               np.asarray(cov, dtype=np.float64), df, jitter)
        return obj

    def _init_from_moments(self, mean, cov, df, jitter):
        self.df = float(df)
        self.d = len(mean)
        self.mean = mean
        self.L = np.linalg.cholesky(cov + jitter * np.eye(self.d))
        half_logdet = np.sum(np.log(np.diag(self.L)))
        from scipy.special import gammaln
        self._log_norm = (gammaln((self.df + self.d) / 2)
                          - gammaln(self.df / 2)
                          - 0.5 * self.d * np.log(self.df * np.pi)
                          - half_logdet)

    def sample(self, n, rng):
        z = rng.standard_normal((n, self.d))
        g = rng.chisquare(self.df, size=n) / self.df
        return self.mean + (z @ self.L.T) / np.sqrt(g)[:, None]

    def log_density(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        diff = x - self.mean
        from scipy.linalg import solve_triangular
        u = solve_triangular(self.L, diff.T, lower=True).T
        m = np.sum(u * u, axis=1)
        return self._log_norm - 0.5 * (self.df + self.d) * np.log1p(m / self.df)


class MixtureProposal:
    """Finite mixture of proposals with sample/log_density interface."""

    def __init__(self, components, weights=None):
        self.components = list(components)
        k = len(self.components)
        self.weights = (np.full(k, 1.0 / k) if weights is None
                        else np.asarray(weights, dtype=np.float64))
        self.weights = self.weights / self.weights.sum()
        self.d = self.components[0].d

    def sample(self, n, rng):
        counts = rng.multinomial(n, self.weights)
        draws = np.concatenate([c.sample(m, rng)
                                for c, m in zip(self.components, counts)
                                if m > 0])
        rng.shuffle(draws)
        return draws

    def log_density(self, x):
        parts = np.stack([np.log(w) + c.log_density(x)
                          for c, w in zip(self.components, self.weights)])
        mx = np.max(parts, axis=0)
        return mx + np.log(np.sum(np.exp(parts - mx), axis=0))


# =============================================================================
# Per-theta quantity caches
# =============================================================================

def _build_quantity_fn(model_key, data):
    """Return theta_i -> (4,) quantities on the natural scale."""
    dataf = data['float']
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])

    if model_key == 'crra':
        from posteriors.common_fns import full_forward_map

        def qfn(theta_i):
            return full_forward_map(theta_i[0], theta_i[1], theta_i[2],
                                    theta_i[3], ref_lotteries, ref_pair)
    elif model_key == 'ge':
        from posteriors.common_fns import full_forward_map_ge

        def qfn(theta_i):
            return full_forward_map_ge(theta_i[0], theta_i[1], theta_i[2],
                                       theta_i[3], ref_lotteries, ref_pair)
    elif model_key == 'power':
        from posteriors.common_fns import full_forward_map_power

        def qfn(theta_i):
            return full_forward_map_power(theta_i[0], theta_i[1], theta_i[2],
                                          ref_lotteries, ref_pair)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return jax.jit(qfn)


def _bounds_from_hyperprior(hyperprior_arr):
    """Extract (lo, hi) arrays of shape (4,) from the flat hyperprior array."""
    hp = np.asarray(hyperprior_arr).reshape(N_QUANTITIES, HYPERPRIOR_STRIDE)
    return hp[:, 4], hp[:, 5]


def fit_subject_proposal(draws, n_bins=8, df=7.0, scale=1.3,
                         defensive_weight=0.1):
    """Fit a PC1-stratified Student-t mixture to one subject's draws.

    Subject conditional posteriors are thin curved ridges in theta space
    (the population prior at posterior psi is a near-degenerate Beta), so
    a single full-covariance Student-t covers them mostly with off-ridge
    mass (observed ESS < 1%). Binning the draws into quantiles along the
    first principal component and fitting a local Student-t per bin
    piecewise-linearizes the ridge. A wide global component with weight
    `defensive_weight` bounds the importance weights in the tails.
    """
    draws = np.asarray(draws, dtype=np.float64)
    center = draws.mean(axis=0)
    cov = np.cov(draws, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, -1]
    proj = (draws - center) @ pc1

    edges = np.quantile(proj, np.linspace(0, 1, n_bins + 1))
    components = []
    weights = []
    for b in range(n_bins):
        if b == n_bins - 1:
            sel = (proj >= edges[b]) & (proj <= edges[b + 1])
        else:
            sel = (proj >= edges[b]) & (proj < edges[b + 1])
        rows = draws[sel]
        if len(rows) < 20:
            continue
        local_cov = np.cov(rows, rowvar=False) + 1e-10 * np.eye(draws.shape[1])
        components.append(StudentTProposal.from_moments(
            rows.mean(axis=0), scale ** 2 * local_cov, df=df))
        weights.append(len(rows))

    weights = np.asarray(weights, dtype=np.float64)
    weights = (1 - defensive_weight) * weights / weights.sum()
    if defensive_weight > 0:
        components.append(StudentTProposal.from_moments(
            center, 1.5 ** 2 * cov, df=5.0))
        weights = np.append(weights, defensive_weight)
    return MixtureProposal(components, weights)


def fit_subject_proposals(subject_draws, rng, df=7.0, scale=1.3, n_bins=8,
                          fit_max=20000):
    """Fit per-subject stratified mixture proposals to posterior draws."""
    fit_rows = subject_draws if len(subject_draws) <= fit_max \
        else subject_draws[rng.choice(len(subject_draws), fit_max, replace=False)]
    return [fit_subject_proposal(fit_rows[:, i, :], n_bins=n_bins,
                                 df=df, scale=scale)
            for i in range(subject_draws.shape[1])]


def refine_subject_proposals(model_key, data, proposals, psi_probes, rng,
                             K_probe=16384, n_resample=2000,
                             df=7.0, scale=1.3, n_bins=8,
                             defensive_weight=0.15):
    """One round of adaptive proposal refinement.

    The integrand of p(y_i|psi) at posterior psi is the subject's
    conditional posterior, which is much narrower than the marginal
    posterior the initial proposals are fitted to (observed ESS ~ 1%).
    Probe the integrand at SEVERAL posterior psi draws (the conditional
    ball moves with psi, so a single-psi refinement under-covers the
    posterior region), importance-resample per probe, and refit the
    mixture on the pooled resamples. The initial proposal is kept as a
    defensive component so regions lost in refinement stay covered.

    Returns (refined_proposals, probe_ess_per_subject) where probe_ess
    is the minimum over probes.
    """
    caches = build_caches_from_proposals(model_key, data, proposals,
                                         K_probe, rng)
    hyperprior_arr = caches['hyperprior_arr']
    psi_probes = np.atleast_2d(psi_probes)
    a, b, _ = _beta_pop_coeffs(jnp.asarray(psi_probes), hyperprior_arr)
    a, b = np.asarray(a), np.asarray(b)          # (P, 4)
    # log weights per probe: (P, K, n)
    core = (np.einsum('knq,pq->pkn', caches['logz'], a - 1.0)
            + np.einsum('knq,pq->pkn', caches['log1mz'], b - 1.0))
    log_w = caches['log_fq'][None, :, :] + core
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)

    n_subjects = log_w.shape[2]
    refined = []
    probe_ess = np.empty(n_subjects)
    for i in range(n_subjects):
        pooled = []
        ess_i = []
        for p in range(len(psi_probes)):
            lw = log_w[p, :, i]
            w = np.exp(lw - np.max(lw))
            ess_i.append(np.sum(w) ** 2 / np.sum(w * w))
            idx = rng.choice(K_probe, n_resample, replace=True,
                             p=w / w.sum())
            pooled.append(caches['theta'][idx, i, :])
        probe_ess[i] = np.min(ess_i)
        core_mix = fit_subject_proposal(
            np.concatenate(pooled), n_bins=n_bins, df=df, scale=scale,
            defensive_weight=0.0)
        refined.append(MixtureProposal(
            [core_mix, proposals[i]], [1 - defensive_weight,
                                       defensive_weight]))
    return refined, probe_ess


def build_subject_caches(model_key, data, subject_draws, K, rng,
                         df=7.0, scale=1.3, n_bins=8, fit_max=20000,
                         proposals=None):
    """Build importance-sampling caches from per-subject proposals.

    If `proposals` is None, fits stratified mixtures to the posterior
    draws (use refine_subject_proposals for better efficiency).

    Args:
        subject_draws: (N, n_subjects, p) posterior draws (for fitting only)
        K: fresh proposal draws per subject
    """
    if proposals is None:
        proposals = fit_subject_proposals(subject_draws, rng, df=df,
                                          scale=scale, n_bins=n_bins,
                                          fit_max=fit_max)
    return build_caches_from_proposals(model_key, data, proposals, K, rng)


def build_caches_from_proposals(model_key, data, proposals, K, rng):
    """Draw fresh theta from explicit per-subject proposals and cache
    log f, log q, and the quantity-space log z / log(1-z) tensors.

    Returns dict with:
        log_fq: (K, n) = log f - log q (non-finite log f set to -inf)
        logz, log1mz: (K, n, 4)
        nonfinite_frac: fraction of non-finite log f over the cache
        theta: (K, n, p) the proposal draws (kept for reference checks)
    """
    n_subjects = len(proposals)
    nparams = proposals[0].d
    hyperprior_arr = np.asarray(data['float'][7])
    lo, hi = _bounds_from_hyperprior(hyperprior_arr)

    log_f_fn = _build_log_f_fn(model_key, data)
    qfn = _build_quantity_fn(model_key, data)
    log_f_vec = jax.jit(jax.vmap(log_f_fn, in_axes=(0, None)))
    qfn_vec = jax.jit(jax.vmap(qfn))

    theta = np.empty((K, n_subjects, nparams))
    log_q = np.empty((K, n_subjects))
    log_f = np.empty((K, n_subjects))
    quantities = np.empty((K, n_subjects, N_QUANTITIES))

    for i in range(n_subjects):
        prop = proposals[i]
        th = prop.sample(K, rng)
        theta[:, i, :] = th
        log_q[:, i] = prop.log_density(th)
        th_jax = jnp.asarray(th)
        log_f[:, i] = np.asarray(log_f_vec(th_jax, i))
        quantities[:, i, :] = np.asarray(qfn_vec(th_jax))

    # A draw is unusable if log f or any quantity is non-finite. Zero its
    # weight via log_f = -inf AND sanitize the quantities: a NaN in logz
    # would otherwise poison the whole (subject, psi) sweep (-inf + NaN).
    bad = ~np.isfinite(log_f) | ~np.isfinite(quantities).all(axis=2)
    nonfinite_frac = float(bad.mean())
    log_f[bad] = -np.inf
    quantities[bad] = 0.5 * (lo + hi)

    z = (quantities - lo) / (hi - lo)
    z = np.clip(z, 1e-10, 1 - 1e-10)  # match _subject_quantity_prior
    logz = np.log(z)
    log1mz = np.log1p(-z)

    return {
        'log_fq': log_f - log_q,
        'logz': logz,
        'log1mz': log1mz,
        'nonfinite_frac': nonfinite_frac,
        'theta': theta,
        'model_key': model_key,
        'hyperprior_arr': hyperprior_arr,
    }


# =============================================================================
# Vectorized population-prior sweep over psi
# =============================================================================

def _beta_pop_coeffs(psi_batch, hyperprior_arr):
    """Map psi (B, 8) -> Beta population parameters and constants.

    Returns (a, b, const): a, b of shape (B, 4); const of shape (B,) holding
    sum_q [-log B(a_q, b_q) - log(hi_q - lo_q)] (per-subject constant).
    Replicates the clamps in _subject_quantity_prior.
    """
    hp = jnp.asarray(hyperprior_arr).reshape(N_QUANTITIES, HYPERPRIOR_STRIDE)
    lo = hp[:, 4]
    hi = hp[:, 5]

    logit_mu = psi_batch[:, 0::2]   # (B, 4)
    logit_psi = psi_batch[:, 1::2]  # (B, 4)
    # Clip away from {0, 1}: at psi -> 0 the Beta concentration diverges
    # and gammaln cancellation overflows. Posterior draws sit far inside
    # these bounds; only absurd proposal-tail psi values are affected.
    mu = jnp.clip(jax.nn.sigmoid(logit_mu), 1e-7, 1 - 1e-7)
    ps = jnp.clip(jax.nn.sigmoid(logit_psi), 1e-7, 1 - 1e-7)

    a = jnp.maximum(mu * (1 - ps) / ps, 1e-6)
    b = jnp.maximum((1 - mu) * (1 - ps) / ps, 1e-6)

    log_B = (jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)
             - jax.scipy.special.gammaln(a + b))
    const = jnp.sum(-log_B - jnp.log(hi - lo)[None, :], axis=1)
    return a, b, const


def make_psi_sweep_fn(caches):
    """Build a jitted function psi_batch (B, 8) ->
    (log_p_y (B,), log_p_yi (B, n), ess (B, n))."""
    log_fq = jnp.asarray(caches['log_fq'])      # (K, n)
    logz = jnp.asarray(caches['logz'])          # (K, n, 4)
    log1mz = jnp.asarray(caches['log1mz'])      # (K, n, 4)
    hyperprior_arr = jnp.asarray(caches['hyperprior_arr'])
    K = log_fq.shape[0]

    @jax.jit
    def sweep(psi_batch):
        a, b, const = _beta_pop_coeffs(psi_batch, hyperprior_arr)
        # core[b, k, i] = sum_q (a-1) logz + (b-1) log1mz
        core = (jnp.einsum('knq,bq->bkn', logz, a - 1.0)
                + jnp.einsum('knq,bq->bkn', log1mz, b - 1.0))
        log_w = log_fq[None, :, :] + core   # (B, K, n)

        max_w = jnp.max(log_w, axis=1, keepdims=True)
        max_w = jnp.where(jnp.isfinite(max_w), max_w, 0.0)
        w = jnp.exp(log_w - max_w)
        sum_w = jnp.sum(w, axis=1)                       # (B, n)
        sum_w2 = jnp.sum(w * w, axis=1)                  # (B, n)
        ess = sum_w ** 2 / jnp.maximum(sum_w2, 1e-300)   # (B, n)

        log_p_yi = (max_w[:, 0, :] + jnp.log(jnp.maximum(sum_w, 1e-300))
                    - jnp.log(K) + const[:, None])       # (B, n)
        log_p_y = jnp.sum(log_p_yi, axis=1)              # (B,)
        return log_p_y, log_p_yi, ess

    sweep.cache_shape = (int(log_fq.shape[0]), int(log_fq.shape[1]))
    return sweep


def sweep_psi(sweep_fn, psi, chunk=128, want_ess=False):
    """Apply the sweep function over psi rows in chunks (numpy in/out).

    The chunk size is capped so the (chunk, K, n_subjects) float64
    buffers stay within ~1 GB each on the GPU.
    """
    psi = np.asarray(psi, dtype=np.float64)
    K, n_subj = sweep_fn.cache_shape
    chunk = max(8, min(chunk, int(1.2e8 / (K * n_subj))))
    n = len(psi)
    log_p_y = np.empty(n)
    ess_min = np.empty(n) if want_ess else None
    for s in range(0, n, chunk):
        batch = jnp.asarray(psi[s:s + chunk])
        lp, _, ess = sweep_fn(batch)
        log_p_y[s:s + chunk] = np.asarray(lp)
        if want_ess:
            ess_min[s:s + chunk] = np.asarray(jnp.min(ess, axis=1))
    return (log_p_y, ess_min) if want_ess else log_p_y


# =============================================================================
# Hyperprior evaluation (vectorized)
# =============================================================================

def make_log_hyperprior_vec(data):
    log_hyperprior_fn = _build_log_hyperprior(data)
    vec = jax.jit(jax.vmap(log_hyperprior_fn))

    def f(psi):
        return np.asarray(vec(jnp.asarray(psi)))

    return f


# =============================================================================
# Route (a): importance sampling over psi
# =============================================================================

def log_ml_is(sweep_fn, log_pi_vec, psi_proposal, n_psi, rng, n_batches=25):
    """Importance sampling estimate of log p(y) from a Student-t psi proposal.

    Returns dict with log_ml, se (batch means), ess (psi-level).
    """
    psi = psi_proposal.sample(n_psi, rng)
    log_q = psi_proposal.log_density(psi)
    log_pi = log_pi_vec(psi)
    log_py = sweep_psi(sweep_fn, psi)

    log_w = log_pi + log_py - log_q
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)

    log_ml = _logsumexp(log_w) - np.log(n_psi)

    # psi-level ESS
    w = np.exp(log_w - np.max(log_w))
    ess = float(np.sum(w) ** 2 / np.sum(w * w))

    # SE via batch means: log of per-batch mean weight
    splits = np.array_split(log_w, n_batches)
    batch_log_means = np.array([_logsumexp(s) - np.log(len(s)) for s in splits])
    # SD of batch means on the weight scale, propagated to the log
    bm = np.exp(batch_log_means - log_ml)  # batch mean / overall mean
    se = float(np.std(bm, ddof=1) / np.sqrt(n_batches))

    return {'log_ml': float(log_ml), 'se': se, 'ess': ess,
            'n_psi': n_psi}


# =============================================================================
# Route (b): bridge sampling over psi
# =============================================================================

def _bridge_iterate(lp_post, lq_post, lp_prop, lq_prop, n_iter=500, tol=1e-10):
    """Meng-Wong iterative bridge estimator (cf. bayes_factors.py)."""
    n2, n1 = len(lp_post), len(lp_prop)
    if n1 == 0 or n2 == 0:
        raise RuntimeError(
            f"Bridge sampling has no finite evaluations on one side "
            f"(posterior: {n2}, proposal: {n1})")
    s1 = n1 / (n1 + n2)
    s2 = n2 / (n1 + n2)

    log_r = float(np.median(lp_post - lq_post))
    for it in range(n_iter):
        log_r_old = log_r
        log_num = lp_prop - np.logaddexp(np.log(s1) + lp_prop,
                                         np.log(s2) + lq_prop + log_r)
        log_numerator = -np.log(n1) + _logsumexp(log_num)
        log_den = lq_post - np.logaddexp(np.log(s1) + lp_post,
                                         np.log(s2) + lq_post + log_r)
        log_denominator = -np.log(n2) + _logsumexp(log_den)
        log_r = log_numerator - log_denominator
        if abs(log_r - log_r_old) < tol and it > 2:
            break

    # CV-based SE (simplified Gronau et al. 2017)
    log_h_post = lq_post - np.logaddexp(np.log(s1) + lp_post,
                                        np.log(s2) + lq_post + log_r)
    log_h_prop = lp_prop - np.logaddexp(np.log(s1) + lp_prop,
                                        np.log(s2) + lq_prop + log_r)
    h_post = np.exp(log_h_post - np.max(log_h_post))
    h_prop = np.exp(log_h_prop - np.max(log_h_prop))
    cv_sq = (np.var(h_post) / np.mean(h_post) ** 2 / n2
             + np.var(h_prop) / np.mean(h_prop) ** 2 / n1)
    se = float(np.sqrt(cv_sq)) if cv_sq > 0 else np.nan

    return {'log_ml': float(log_r), 'se': se, 'n_iter': it + 1,
            'converged': abs(log_r - log_r_old) < tol}


def log_ml_bridge(sweep_fn, log_pi_vec, psi_post, psi_proposal, n_prop, rng):
    """Bridge sampling between psi posterior draws and the Student-t proposal.

    Returns dict with log_ml, se (CV-based), se_split (split-half spread).
    """
    psi_prop = psi_proposal.sample(n_prop, rng)

    lp_post = log_pi_vec(psi_post) + sweep_psi(sweep_fn, psi_post)
    lp_prop = log_pi_vec(psi_prop) + sweep_psi(sweep_fn, psi_prop)
    lq_post = psi_proposal.log_density(psi_post)
    lq_prop = psi_proposal.log_density(psi_prop)

    fin_post = np.isfinite(lp_post)
    fin_prop = np.isfinite(lp_prop)
    res = _bridge_iterate(lp_post[fin_post], lq_post[fin_post],
                          lp_prop[fin_prop], lq_prop[fin_prop])

    # Split-half check
    hp = np.where(fin_post)[0]
    hq = np.where(fin_prop)[0]
    halves = []
    for sel_p, sel_q in (((hp[0::2]), (hq[0::2])), ((hp[1::2]), (hq[1::2]))):
        r = _bridge_iterate(lp_post[sel_p], lq_post[sel_p],
                            lp_prop[sel_q], lq_prop[sel_q])
        halves.append(r['log_ml'])
    res['se_split'] = float(abs(halves[0] - halves[1]) / 2)
    res['n_finite_post'] = int(fin_post.sum())
    res['n_finite_prop'] = int(fin_prop.sum())
    return res


# =============================================================================
# Unit test: vectorized pop-prior vs reference implementation
# =============================================================================

def verify_pop_prior_vectorization(data, model_keys=('crra', 'ge', 'power'),
                                   n_points=100, seed=0, tol=1e-9):
    """Check the factorized psi-sweep against _subject_quantity_prior."""
    pop_fns = _build_log_pop_prior_fn(data)
    hyperprior_arr = np.asarray(data['float'][7])
    rng = np.random.default_rng(seed)

    print("Pop-prior vectorization check (max |diff|):")
    all_ok = True
    for mk in model_keys:
        nparams = 3 if mk == 'power' else 4
        qfn = _build_quantity_fn(mk, data)
        lo, hi = _bounds_from_hyperprior(hyperprior_arr)

        # Plausible random points
        theta = np.column_stack([
            rng.uniform(0.1, 0.9, n_points),         # r
            rng.normal(0.0, 0.5, (n_points, nparams - 2)).reshape(n_points, -1),
            rng.normal(-2.0, 0.5, n_points),          # lambda_raw
        ])
        psi = rng.normal(0.0, 1.0, (n_points, 8))

        max_diff = 0.0
        for j in range(n_points):
            ref = float(pop_fns[mk](jnp.asarray(theta[j]), jnp.asarray(psi[j])))

            q = np.asarray(qfn(jnp.asarray(theta[j])))
            z = np.clip((q - lo) / (hi - lo), 1e-10, 1 - 1e-10)
            a, b, const = _beta_pop_coeffs(jnp.asarray(psi[j:j + 1]),
                                           hyperprior_arr)
            a, b, const = np.asarray(a)[0], np.asarray(b)[0], float(const[0])
            vec = float(np.sum((a - 1) * np.log(z) + (b - 1) * np.log1p(-z))
                        + const)
            max_diff = max(max_diff, abs(ref - vec))

        ok = max_diff < tol * max(1.0, abs(ref))
        all_ok = all_ok and ok
        print(f"  {mk:6s}: max |diff| = {max_diff:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
    return all_ok


# =============================================================================
# Replicated estimate for one model
# =============================================================================

def estimate_model_log_ml(model_key, data, hyper, subj, rng,
                          K=16384, R=8, n_psi_is=4000, n_psi_bridge=2500,
                          theta_df=7.0, theta_scale=1.3,
                          psi_df=10.0, psi_scale=1.1, ess_floor=100.0,
                          verbose=True):
    """Replicated corrected-RB estimate of log p(y) for one model.

    A single theta cache produces route estimates whose log is biased
    downward by roughly sum_i 1/(2 ESS_i) and whose cache-level noise is
    invisible to the psi-level SE (every psi evaluation shares the same
    inner draws). We therefore build R INDEPENDENT caches, run both psi
    routes on each, and combine each route across replicates on the
    natural scale: per-cache route estimates are (approximately) unbiased
    for p(y) on that scale, so averaging shrinks the log bias by R and
    the replicate spread gives an SE that includes inner-IS noise.
    """
    log_pi_vec = make_log_hyperprior_vec(data)
    psi_proposal = StudentTProposal(hyper, df=psi_df, scale=psi_scale)

    # Fit proposals once, refine once at a spread of posterior psi draws,
    # share the refined proposals across replicates (independence of
    # caches comes from fresh theta draws within each replicate).
    proposals = fit_subject_proposals(subj, rng, df=theta_df,
                                      scale=theta_scale)
    psi_probes = np.vstack([
        hyper.mean(axis=0),
        hyper[rng.choice(len(hyper), 7, replace=False)],
    ])
    proposals, probe_ess = refine_subject_proposals(
        model_key, data, proposals, psi_probes, rng,
        df=theta_df, scale=theta_scale)
    if verbose:
        print(f"  Proposal refinement probe ESS (min over {len(psi_probes)} "
              f"probes): min={probe_ess.min():.0f}, "
              f"median={np.median(probe_ess):.0f}")

    rep_is = np.empty(R)
    rep_br = np.empty(R)
    ess_min = ess_med = nonfinite = None
    psi_ess = []

    for rep in range(R):
        caches = build_subject_caches(model_key, data, subj, K, rng,
                                      proposals=proposals)
        sweep_fn = make_psi_sweep_fn(caches)

        if rep == 0:
            nonfinite = caches['nonfinite_frac']
            idx50 = rng.choice(len(hyper), 50, replace=False)
            _, ess50 = sweep_psi(sweep_fn, hyper[idx50], want_ess=True)
            ess_min = float(np.min(ess50))
            ess_med = float(np.median(ess50))
            if verbose:
                print(f"  Non-finite log f fraction: {nonfinite:.4f}")
                print(f"  Per-subject ESS at 50 posterior psi: "
                      f"min={ess_min:.0f}, median={ess_med:.0f} (K={K})")
                if ess_min < ess_floor:
                    print(f"  WARNING: min ESS below floor {ess_floor}")

        res_is = log_ml_is(sweep_fn, log_pi_vec, psi_proposal, n_psi_is, rng)
        idx_b = rng.choice(len(hyper), n_psi_bridge, replace=False)
        res_br = log_ml_bridge(sweep_fn, log_pi_vec, hyper[idx_b],
                               psi_proposal, n_psi_bridge, rng)
        rep_is[rep] = res_is['log_ml']
        rep_br[rep] = res_br['log_ml']
        psi_ess.append(res_is['ess'])
        if verbose:
            print(f"  replicate {rep}: IS {res_is['log_ml']:.3f}, "
                  f"bridge {res_br['log_ml']:.3f} "
                  f"(psi ESS {res_is['ess']:.0f}, "
                  f"bridge converged={res_br['converged']})")

    def combine(rep_vals):
        log_ml = _logsumexp(rep_vals) - np.log(R)
        bm = np.exp(rep_vals - log_ml)
        se = float(np.std(bm, ddof=1) / np.sqrt(R))
        return float(log_ml), se

    is_ml, is_se = combine(rep_is)
    br_ml, br_se = combine(rep_br)

    res = {
        'log_ml_is': is_ml, 'se_is': is_se,
        'log_ml_bridge': br_ml, 'se_bridge': br_se,
        'rep_is': rep_is, 'rep_bridge': rep_br,
        'rep_spread_is': float(rep_is.max() - rep_is.min()),
        'rep_spread_bridge': float(rep_br.max() - rep_br.min()),
        'ess_min': ess_min, 'ess_med': ess_med,
        'psi_ess_mean': float(np.mean(psi_ess)),
        'nonfinite_frac': nonfinite,
        'K': K, 'R': R,
    }
    agree = abs(is_ml - br_ml)
    comb_se = float(np.sqrt(is_se ** 2 + br_se ** 2))
    res['route_agreement'] = agree
    res['route_tol'] = max(2 * comb_se, 0.1)
    res['routes_agree'] = agree <= res['route_tol']
    if verbose:
        print(f"  IS route:     log ML = {is_ml:.3f} +/- {is_se:.3f} "
              f"(replicate spread {res['rep_spread_is']:.3f})")
        print(f"  Bridge route: log ML = {br_ml:.3f} +/- {br_se:.3f} "
              f"(replicate spread {res['rep_spread_bridge']:.3f})")
        print(f"  Route agreement: |IS - bridge| = {agree:.3f} "
              f"(tol {res['route_tol']:.3f}) "
              f"{'OK' if res['routes_agree'] else 'DISAGREE'}")
    return res


# =============================================================================
# Driver
# =============================================================================

def compute_marginal_likelihoods(model_keys, data_dir, output_path,
                                 K=16384, R=8, n_psi_is=4000,
                                 n_psi_bridge=2500,
                                 n_load=100000, seed=42,
                                 psi_df=10.0, psi_scale=1.1,
                                 theta_df=7.0, theta_scale=1.3,
                                 ess_floor=100.0):
    """Corrected RB marginal likelihoods with two psi-integration routes.

    Saves NPZ with per-model log ML (IS and bridge routes), SEs, diagnostics,
    and pairwise log Bayes factors (IS route).
    """
    data_dir = Path(data_dir)
    data = assemble_data(data_dir)

    print("=" * 60)
    print("MARGINAL LIKELIHOODS (corrected Rao-Blackwellized estimator)")
    print("=" * 60)

    if not verify_pop_prior_vectorization(data, tuple(model_keys)):
        raise RuntimeError("Pop-prior vectorization check failed")

    results = {}
    save = {'K': K, 'R': R, 'n_psi_is': n_psi_is,
            'n_psi_bridge': n_psi_bridge, 'seed': seed}

    for mk in model_keys:
        print(f"\n{'=' * 60}\nModel: {mk}\n{'=' * 60}")
        rng = np.random.default_rng(seed + hash(mk) % 1000)

        hyper, subj, n_subjects = load_posterior_states(
            mk, str(data_dir), n_samples=n_load, seed=seed)
        print(f"  Loaded {len(hyper)} draws, {n_subjects} subjects, "
              f"{subj.shape[2]} params/subject")

        res = estimate_model_log_ml(
            mk, data, hyper, subj, rng, K=K, R=R,
            n_psi_is=n_psi_is, n_psi_bridge=n_psi_bridge,
            theta_df=theta_df, theta_scale=theta_scale,
            psi_df=psi_df, psi_scale=psi_scale, ess_floor=ess_floor)
        results[mk] = res

        save[f'{mk}_log_ml_is'] = res['log_ml_is']
        save[f'{mk}_log_ml_is_se'] = res['se_is']
        save[f'{mk}_log_ml_bridge'] = res['log_ml_bridge']
        save[f'{mk}_log_ml_bridge_se'] = res['se_bridge']
        save[f'{mk}_rep_is'] = res['rep_is']
        save[f'{mk}_rep_bridge'] = res['rep_bridge']
        save[f'{mk}_ess_subject_min'] = res['ess_min']
        save[f'{mk}_ess_subject_med'] = res['ess_med']
        save[f'{mk}_psi_ess_mean'] = res['psi_ess_mean']
        save[f'{mk}_nonfinite_frac'] = res['nonfinite_frac']
        save[f'{mk}_routes_agree'] = res['routes_agree']

    print(f"\n{'=' * 60}\nBAYES FACTORS (log scale, IS route)\n{'=' * 60}")
    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            bf = results[k1]['log_ml_is'] - results[k2]['log_ml_is']
            bf_se = np.sqrt(results[k1]['se_is'] ** 2
                            + results[k2]['se_is'] ** 2)
            print(f"  log BF({k1} vs {k2}) = {bf:.2f} +/- {bf_se:.2f}")
            save[f'log_bf_rb2_{k1}_vs_{k2}'] = bf
            save[f'log_bf_rb2_{k1}_vs_{k2}_se'] = bf_se

    np.savez(Path(output_path), **save)
    print(f"\nSaved to: {output_path}")
    return results


# =============================================================================
# Prior normalization mass C_m(psi) = integral of the unnormalized induced
# prior pop(g(theta)|psi) * vol(J(theta)) over theta.
#
# The induced prior used in sampling and in the marginal likelihood above is
# unnormalized: its theta-integral C_m(psi) need not equal 1, generally
# differs across models (for k > p the quantity-space support is a
# lower-dimensional manifold), and may vary with psi. The properly
# normalized model divides each subject's prior by C_m(psi), so the
# corrected marginal likelihood subtracts n_subjects * log C_m(psi) inside
# the psi-integral. C_m is subject-independent, so it is a single 3- or
# 4-dimensional integral per (model, psi), estimated below by the same
# adaptive importance sampling machinery with the likelihood replaced by 1.
# =============================================================================

def _build_log_jac_fn(model_key, data):
    """Per-model log volume element: log|det J| (k=p) or the Gram
    log-determinant (k>p), for the full 4-quantity map including P."""
    dataf = data['float']
    ref_lotteries = jnp.asarray(dataf[8])
    ref_pair = jnp.asarray(dataf[9])

    if model_key == 'crra':
        from posteriors.common_fns import full_jacobian_logabsdet

        def jac(t):
            return full_jacobian_logabsdet(t[0], t[1], t[2], t[3],
                                           ref_lotteries, ref_pair)
    elif model_key == 'ge':
        from posteriors.common_fns import full_jacobian_logabsdet_ge

        def jac(t):
            return full_jacobian_logabsdet_ge(t[0], t[1], t[2], t[3],
                                              ref_lotteries, ref_pair)
    elif model_key == 'power':
        from posteriors.common_fns import gram_logdet_power

        def jac(t):
            return gram_logdet_power(t[0], t[1], t[2],
                                     ref_lotteries, ref_pair)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return jax.jit(jac)


def load_prior_theta_draws(model_key, data_dir, n_samples=50000, seed=42):
    """Load theta draws from the model's prior-only MCMC run (1 subject)."""
    from model_registry import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_key]
    nparams = cfg['nparams']
    n_hyper = cfg['n_hyper']

    history_dir = (Path(data_dir) / cfg['prior_model_id'] / 'history' / 'full')
    files = sorted(history_dir.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f"No prior-only history in {history_dir}")

    n_raw = nparams + n_hyper
    states = [np.load(f, allow_pickle=True)['history'][:, :, :n_raw]
              for f in files]
    combined = np.concatenate(states, axis=0)
    burn = combined.shape[0] // 2
    flat = combined[burn:].reshape(-1, n_raw)
    if n_samples < len(flat):
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(len(flat), n_samples, replace=False)]
    return flat[:, :nparams]


def _build_prior_mass_cache(model_key, data, proposal, K, rng):
    """Draw K theta from the proposal and cache the C-integrand pieces in
    the (K, 1, 4) layout that make_psi_sweep_fn expects."""
    hyperprior_arr = np.asarray(data['float'][7])
    lo, hi = _bounds_from_hyperprior(hyperprior_arr)
    jac_vec = jax.jit(jax.vmap(_build_log_jac_fn(model_key, data)))
    qfn_vec = jax.jit(jax.vmap(_build_quantity_fn(model_key, data)))

    theta = proposal.sample(K, rng)
    log_q = proposal.log_density(theta)
    th_jax = jnp.asarray(theta)
    log_jac = np.array(jac_vec(th_jax))
    quantities = np.array(qfn_vec(th_jax))

    bad = ~np.isfinite(log_jac) | ~np.isfinite(quantities).all(axis=1)
    log_jac[bad] = -np.inf
    quantities[bad] = 0.5 * (lo + hi)

    z = np.clip((quantities - lo) / (hi - lo), 1e-10, 1 - 1e-10)
    return {
        'log_fq': (log_jac - log_q)[:, None],
        'logz': np.log(z)[:, None, :],
        'log1mz': np.log1p(-z)[:, None, :],
        'nonfinite_frac': float(bad.mean()),
        'theta': theta[:, None, :],
        'hyperprior_arr': hyperprior_arr,
        'model_key': model_key,
    }


def build_prior_mass_estimator(model_key, data, psi_probes, rng,
                               K=262144, K_probe=65536, n_resample=4000,
                               rounds=2, df=7.0, scale=1.3, n_bins=8,
                               init_theta=None, data_dir=None,
                               verbose=True):
    """Adaptively fit proposals to the prior integrand and return a sweep
    function psi (B,8) -> log C_hat (B,), plus the proposal for replicates.

    The integrand at given psi is the conditional prior ball, which moves
    and tightens with psi, so proposals are refined by importance
    resampling against the integrand pooled over several posterior psi
    probes (same scheme as the per-subject likelihood integrals).
    """
    if init_theta is None:
        init_theta = load_prior_theta_draws(model_key, data_dir)
    proposal = fit_subject_proposal(init_theta, n_bins=n_bins, df=df,
                                    scale=scale)
    psi_probes = np.atleast_2d(psi_probes)

    probe_ess = None
    for rnd in range(rounds):
        caches = _build_prior_mass_cache(model_key, data, proposal,
                                         K_probe, rng)
        hyperprior_arr = caches['hyperprior_arr']
        a, b, _ = _beta_pop_coeffs(jnp.asarray(psi_probes), hyperprior_arr)
        a, b = np.asarray(a), np.asarray(b)
        core = (np.einsum('knq,pq->pkn', caches['logz'], a - 1.0)
                + np.einsum('knq,pq->pkn', caches['log1mz'], b - 1.0))
        log_w = caches['log_fq'][None, :, :] + core   # (P, K, 1)
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)[:, :, 0]

        pooled = []
        ess_list = []
        for p in range(len(psi_probes)):
            lw = log_w[p]
            w = np.exp(lw - np.max(lw))
            ess_list.append(np.sum(w) ** 2 / np.sum(w * w))
            idx = rng.choice(K_probe, n_resample, replace=True,
                             p=w / w.sum())
            pooled.append(caches['theta'][idx, 0, :])
        probe_ess = np.array(ess_list)
        if verbose:
            print(f"  [{model_key}] C-proposal round {rnd}: probe ESS "
                  f"min={probe_ess.min():.0f}, "
                  f"median={np.median(probe_ess):.0f} (K_probe={K_probe})")
        core_mix = fit_subject_proposal(np.concatenate(pooled),
                                        n_bins=n_bins, df=df, scale=scale,
                                        defensive_weight=0.0)
        proposal = MixtureProposal([core_mix, proposal], [0.85, 0.15])

    caches = _build_prior_mass_cache(model_key, data, proposal, K, rng)
    sweep_fn = make_psi_sweep_fn(caches)
    if verbose:
        print(f"  [{model_key}] final C-cache: K={K}, non-finite "
              f"{caches['nonfinite_frac']:.5f}")

    def log_C(psi, want_ess=False):
        return sweep_psi(sweep_fn, psi, want_ess=want_ess)

    return log_C, proposal, probe_ess


def measure_prior_mass(model_keys, data_dir, output_path=None, n_psi=200,
                       K=262144, n_replicates=2, seed=42):
    """Phase-A measurement: level and psi-variation of log C_m(psi) at
    posterior-representative psi draws, with replicate agreement."""
    data_dir = Path(data_dir)
    data = assemble_data(data_dir)
    results = {}

    for mk in model_keys:
        print(f"\n{'=' * 60}\nPrior mass: {mk}\n{'=' * 60}")
        rng = np.random.default_rng(seed + hash(mk) % 1000)
        hyper, _subj, n_subjects = load_posterior_states(
            mk, str(data_dir), n_samples=50000, seed=seed)
        psi_bar = hyper.mean(axis=0)
        idx = rng.choice(len(hyper), n_psi, replace=False)
        psi_eval = np.vstack([psi_bar, hyper[idx]])
        psi_probes = np.vstack([psi_bar,
                                hyper[rng.choice(len(hyper), 7,
                                                 replace=False)]])

        reps = []
        for rep in range(n_replicates):
            log_C, proposal, probe_ess = build_prior_mass_estimator(
                mk, data, psi_probes, rng, K=K, data_dir=data_dir)
            lc, ess = log_C(psi_eval, want_ess=True)
            reps.append(lc)
            print(f"  replicate {rep}: log C(psi_bar) = {lc[0]:.5f}, "
                  f"across-psi SD = {np.std(lc[1:]):.5f}, "
                  f"ESS min={ess.min():.0f} med={np.median(ess):.0f}")

        lc_mean = np.mean(reps, axis=0)
        rep_diff = float(np.max(np.abs(reps[0] - reps[1]))) \
            if n_replicates > 1 else np.nan

        # Linear regression of log C on psi to detect systematic variation
        X = np.column_stack([np.ones(len(psi_eval)), psi_eval])
        beta, res, *_ = np.linalg.lstsq(X, lc_mean, rcond=None)
        fitted = X @ beta
        resid_sd = float(np.std(lc_mean - fitted))
        fitted_spread = float(np.std(fitted[1:]))

        results[mk] = {
            'log_C_bar': float(lc_mean[0]),
            'sd_across_psi': float(np.std(lc_mean[1:])),
            'fitted_spread': fitted_spread,
            'resid_sd': resid_sd,
            'rep_max_diff': rep_diff,
            'n_subjects': n_subjects,
            'correction_at_bar': float(-n_subjects * lc_mean[0]),
            'tilt_sd': float(n_subjects * fitted_spread),
            'beta': beta,
            'psi_eval': psi_eval,
            'log_C': lc_mean,
        }
        r = results[mk]
        print(f"  log C(psi_bar) = {r['log_C_bar']:.5f}  "
              f"(replicate max |diff| {rep_diff:.5f})")
        print(f"  across-psi SD = {r['sd_across_psi']:.5f} "
              f"(systematic {fitted_spread:.5f}, noise {resid_sd:.5f})")
        print(f"  ML level correction -n*logC = {r['correction_at_bar']:.2f}; "
              f"psi-tilt SD n*spread = {r['tilt_sd']:.2f}")

    if output_path is not None:
        save = {}
        for mk, r in results.items():
            for k in ('log_C_bar', 'sd_across_psi', 'fitted_spread',
                      'resid_sd', 'rep_max_diff', 'correction_at_bar',
                      'tilt_sd'):
                save[f'{mk}_{k}'] = r[k]
            save[f'{mk}_beta'] = r['beta']
            save[f'{mk}_psi_eval'] = r['psi_eval']
            save[f'{mk}_log_C'] = r['log_C']
        np.savez(Path(output_path), **save)
        print(f"\nSaved to: {output_path}")
    return results


# =============================================================================
# Hyperprior-tilted feasible mass Z_m = E_{psi ~ pi}[ C_m(psi)^N ].
#
# The model as sampled is the quantity-space hierarchy TRUNCATED to each
# model's feasible image: per subject the conditional prior is p_tilde / C,
# and the truncation reweights the hyperprior by the feasible mass,
# pi_eff(psi) proportional to pi(psi) C(psi)^N with normalizer Z_m. The
# posteriors reported in the paper are exactly this model's posteriors, and
# its properly normalized marginal likelihood is the reported value minus
# log Z_m. Because N log C amplifies estimation noise by N, log C is
# evaluated at a moderate number of design psi points with high inner ESS,
# smoothed by a quadratic surface in psi, and the tilted expectation is
# taken over exact hyperprior draws evaluated on the surface.
# =============================================================================

def sample_hyperprior_psi(n, hyperprior_arr, rng):
    """Draw psi (n, 8) exactly from the Beta-Beta hyperprior (logit scale)."""
    hp = np.asarray(hyperprior_arr).reshape(N_QUANTITIES, HYPERPRIOR_STRIDE)
    psi = np.empty((n, 2 * N_QUANTITIES))
    for k in range(N_QUANTITIES):
        mu = rng.beta(hp[k, 0], hp[k, 1], size=n)
        ps = rng.beta(hp[k, 2], hp[k, 3], size=n)
        psi[:, 2 * k] = np.log(mu / (1 - mu))
        psi[:, 2 * k + 1] = np.log(ps / (1 - ps))
    return psi


def _quad_features(psi, center, scale):
    """Standardized quadratic feature matrix (1, z, z_i z_j for i<=j)."""
    z = (psi - center) / scale
    n, d = z.shape
    cols = [np.ones(n)]
    cols.extend(z[:, j] for j in range(d))
    for i in range(d):
        for j in range(i, d):
            cols.append(z[:, i] * z[:, j])
    return np.column_stack(cols)


def estimate_log_Z_tr(model_key, data, n_subjects, rng, data_dir,
                      n_per_round=160, K=262144, max_rounds=8,
                      n_is=300000, verbose=True):
    """Trust-region estimate of log Z = log E_pi[C(psi)^N].

    The tilt C^N concentrates the integral in a small psi region, and any
    surface for log C extrapolates dangerously there (N amplifies fit
    errors and the maximizer of an extrapolated quadratic is an
    artifact). This routine therefore only ever evaluates the fitted
    surface INSIDE the design support: each round fits a quadratic to all
    design evaluations, importance-samples the tilted integrand with
    weights zeroed outside the design's Mahalanobis hull, and adds new
    design points where the tilted mass actually sits. Iteration stops
    when the tilted mass is interior to the hull (negligible boundary
    fraction) and the region is stable. The hull truncation makes the
    estimate a lower bound whose omitted mass is controlled by the
    reported boundary fraction.
    """
    from scipy.linalg import solve_triangular

    hyperprior_arr = np.asarray(data['float'][7])
    log_pi_vec = make_log_hyperprior_vec(data)
    bounded = model_key in ('crra', 'ge')

    # Round-0 design: hyperprior draws
    psi_D = sample_hyperprior_psi(n_per_round, hyperprior_arr, rng)
    probes = psi_D[rng.choice(len(psi_D), 12, replace=False)]
    log_C, _p, _ = build_prior_mass_estimator(
        model_key, data, probes, rng, K=K, data_dir=data_dir,
        verbose=False)
    lc_D = log_C(psi_D)

    history = []
    prev_tilt_pts = None
    prev_ess = 0.0
    for rnd in range(max_rounds):
        # Active design subset: once the tilted mass has localized, fit
        # the surface only on design points near it (a global quadratic
        # cannot track log C over the whole hyperprior for the k > p
        # model, whose feasible-mass surface is much rougher).
        active = np.ones(len(psi_D), bool)
        if prev_tilt_pts is not None and prev_ess >= 10:
            t_center = prev_tilt_pts.mean(axis=0)
            t_cov = np.cov(prev_tilt_pts, rowvar=False) \
                + 1e-10 * np.eye(8)
            Lt = np.linalg.cholesky(t_cov)
            ut = solve_triangular(Lt, (psi_D - t_center).T, lower=True)
            d_t = np.sqrt(np.sum(ut * ut, axis=0))
            loc = d_t <= 6.0
            if loc.sum() >= 60:
                active = loc

        psi_A = psi_D[active]
        lc_A = lc_D[active]
        center = psi_A.mean(axis=0)
        cov_D = np.cov(psi_A, rowvar=False) + 1e-10 * np.eye(8)
        L_D = np.linalg.cholesky(cov_D)
        scale = np.sqrt(np.diag(cov_D))

        X = _quad_features(psi_A, center, scale)
        beta, *_ = np.linalg.lstsq(X, lc_A, rcond=None)
        resid_sd = float(np.std(lc_A - X @ beta))

        def mahal(psi):
            u = solve_triangular(L_D, (psi - center).T, lower=True)
            return np.sqrt(np.sum(u * u, axis=0))

        rho = float(np.max(mahal(psi_A)))

        # Tilted IS within the hull. After round 0 the proposal is fitted
        # to the previous round's tilted resample (far better matched than
        # the design cloud), defensively mixed with the design-cloud fit.
        q_design = StudentTProposal.from_moments(center, 1.5 ** 2 * cov_D,
                                                 df=7.0)
        if prev_tilt_pts is None:
            q = q_design
        else:
            q = MixtureProposal(
                [StudentTProposal(prev_tilt_pts, df=7.0, scale=1.5),
                 q_design], [0.8, 0.2])
        psi_s = q.sample(n_is, rng)
        x = _quad_features(psi_s, center, scale) @ beta
        if bounded:
            x = np.minimum(x, 0.0)
        lw = log_pi_vec(psi_s) + n_subjects * x - q.log_density(psi_s)
        m = mahal(psi_s)
        inside = m <= rho
        lw = np.where(inside, lw, -np.inf)

        log_Z = float(_logsumexp(lw) - np.log(n_is))
        w = np.exp(lw - lw.max())
        ess = float(np.sum(w) ** 2 / np.sum(w * w))
        boundary_frac = float(np.sum(w[m > 0.9 * rho]) / np.sum(w))

        # Where does the tilted mass sit?
        idx = rng.choice(n_is, 2000, replace=True, p=w / w.sum())
        tilt_pts = psi_s[idx]
        prev_tilt_pts = tilt_pts
        prev_ess = ess
        tilt_center = tilt_pts.mean(axis=0)
        shift = float(np.max(np.abs(tilt_center - center) / scale))

        history.append((log_Z, ess, boundary_frac, resid_sd))
        if verbose:
            print(f"  [{model_key}] round {rnd}: log Z = {log_Z:.3f}, "
                  f"IS ESS {ess:.0f}, boundary mass {boundary_frac:.3f}, "
                  f"fit resid {resid_sd:.4f}, design n={len(psi_D)} "
                  f"(active {active.sum()}), shift {shift:.2f}")

        if boundary_frac < 0.02 and ess >= 300 and rnd > 0:
            break

        # New design points at the tilted mass; jitter widens when the
        # tilt resample has collapsed to a few points.
        jit = 0.3 if ess >= 50 else 1.0
        new_pts = tilt_pts[rng.choice(len(tilt_pts), n_per_round,
                                      replace=False)] \
            + jit * scale * rng.standard_normal((n_per_round, 8))
        probes = new_pts[rng.choice(len(new_pts), 12, replace=False)]
        log_C, _p, _ = build_prior_mass_estimator(
            model_key, data, probes, rng, K=K, data_dir=data_dir,
            verbose=False)
        lc_new = log_C(new_pts)
        psi_D = np.vstack([psi_D, new_pts])
        lc_D = np.concatenate([lc_D, lc_new])

    # Jackknife over (active) design blocks for the surface SE
    psi_D = psi_A
    lc_D = lc_A
    n_blocks = 8
    blocks = np.array_split(rng.permutation(len(psi_D)), n_blocks)
    psi_jk = psi_s[:60000]
    lq_jk = q.log_density(psi_jk)
    lpi_jk = log_pi_vec(psi_jk)
    m_jk = mahal(psi_jk)
    F_jk = _quad_features(psi_jk, center, scale)
    jk = []
    for b in blocks:
        mask = np.ones(len(psi_D), bool)
        mask[b] = False
        Xb = _quad_features(psi_D[mask], center, scale)
        bta, *_ = np.linalg.lstsq(Xb, lc_D[mask], rcond=None)
        xj = F_jk @ bta
        if bounded:
            xj = np.minimum(xj, 0.0)
        lwj = np.where(m_jk <= rho,
                       lpi_jk + n_subjects * xj - lq_jk, -np.inf)
        jk.append(_logsumexp(lwj) - np.log(len(psi_jk)))
    jk = np.asarray(jk)
    se = float(np.sqrt((n_blocks - 1) / n_blocks
                       * np.sum((jk - jk.mean()) ** 2)))

    hard_bound = n_subjects * float(np.max(lc_D))
    if verbose:
        print(f"  [{model_key}] log Z = {log_Z:.3f} (jackknife SE {se:.3f}); "
              f"sanity: <= N*max(logC) = {hard_bound:.1f}")

    return {'log_Z': log_Z, 'se_jk': se, 'boundary_frac': boundary_frac,
            'is_ess': ess, 'resid_sd': resid_sd,
            'hard_bound': hard_bound, 'history': history,
            'psi_D': psi_D, 'lc_D': lc_D}


def compute_normalization_corrections_tr(model_keys, data_dir, output_path,
                                         seed=42, n_replicates=2):
    """Trust-region log Z estimates with replicate agreement, applied to
    the reported marginal likelihoods."""
    data_dir = Path(data_dir)
    data = assemble_data(data_dir)
    ml = dict(np.load(data_dir / 'bayes_factors_rb2.npz'))

    results = {}
    for mk in model_keys:
        print(f"\n{'=' * 60}\nFeasible-mass tilt (trust region): {mk}\n"
              f"{'=' * 60}")
        _h, _s, n_subjects = load_posterior_states(
            mk, str(data_dir), n_samples=2000, seed=seed)
        reps = []
        for rep in range(n_replicates):
            rng = np.random.default_rng(seed + 11000 + 100 * rep
                                        + hash(mk) % 1000)
            reps.append(estimate_log_Z_tr(mk, data, n_subjects, rng,
                                          data_dir=data_dir))
        lz = np.array([r['log_Z'] for r in reps])
        print(f"  [{mk}] replicate log Z values: {np.round(lz, 3)} "
              f"(spread {lz.max() - lz.min():.3f})")
        results[mk] = {
            'log_Z': float(np.mean(lz)),
            'se': float(np.sqrt(np.mean([r['se_jk'] ** 2 for r in reps])
                                / n_replicates
                                + (np.var(lz, ddof=1) / n_replicates
                                   if n_replicates > 1 else 0.0))),
            'boundary_frac': float(np.max([r['boundary_frac']
                                           for r in reps])),
            'reps': reps,
        }

    for mk in model_keys:
        r = results[mk]
        ml[f'{mk}_log_Z'] = r['log_Z']
        ml[f'{mk}_log_Z_se'] = r['se']
        ml[f'{mk}_log_Z_boundary_frac'] = r['boundary_frac']
        ml[f'{mk}_log_ml_is_corrected'] = \
            float(ml[f'{mk}_log_ml_is']) - r['log_Z']
        ml[f'{mk}_log_ml_bridge_corrected'] = \
            float(ml[f'{mk}_log_ml_bridge']) - r['log_Z']

    print(f"\n{'=' * 60}\nCORRECTED BAYES FACTORS (log scale, IS route)\n"
          f"{'=' * 60}")
    keys = list(model_keys)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            bf = (float(ml[f'{k1}_log_ml_is_corrected'])
                  - float(ml[f'{k2}_log_ml_is_corrected']))
            se = float(np.sqrt(float(ml[f'{k1}_log_ml_is_se']) ** 2
                               + float(ml[f'{k2}_log_ml_is_se']) ** 2
                               + results[k1]['se'] ** 2
                               + results[k2]['se'] ** 2))
            print(f"  log BF({k1} vs {k2}) = {bf:.2f} +/- {se:.2f}  "
                  f"(uncorrected {float(ml[f'log_bf_rb2_{k1}_vs_{k2}']):.2f})")
            ml[f'log_bf_rb2_{k1}_vs_{k2}_corrected'] = bf
            ml[f'log_bf_rb2_{k1}_vs_{k2}_corrected_se'] = se

    np.savez(Path(output_path), **ml)
    print(f"\nSaved to: {output_path}")
    return results


def estimate_log_Z_anneal(model_key, data, n_subjects, rng, data_dir,
                          n_particles=50000, K=262144, ess_target=0.30,
                          max_stages=40, t_stop=None, verbose=True):
    """Annealed adaptive-IS estimate of log Z = log E_pi[C(psi)^N].

    For the k > p model the tilted distribution pi * C^N concentrates
    along a curved ridge of psi values whose conditional mean sits near
    the feasible manifold, which defeats mode-centered schemes. This
    routine anneals the tilt exponent t from 0 to N adaptively: at each
    stage particles are drawn fresh from a mixture proposal fitted to the
    previous stage's tilted cloud (PC1-stratified Student-t components,
    which follow ridges), the C estimator is rebuilt with probes from the
    current cloud, and the next t is chosen so the incremental effective
    sample size stays above `ess_target`. The reported value is the
    DIRECT importance-sampling estimate at t = N from the final proposal;
    intermediate stages only guide the proposal, so no telescoping error
    accumulates.
    """
    hyperprior_arr = np.asarray(data['float'][7])
    log_pi_vec = make_log_hyperprior_vec(data)
    bounded = model_key in ('crra', 'ge')

    t = 0.0
    stage = 0
    q = None  # None means exact hyperprior sampling
    cloud = sample_hyperprior_psi(2000, hyperprior_arr, rng)
    widen = 1.0          # proposal widening factor, grown on stalls
    target_now = ess_target

    while True:
        # Rebuild the C estimator around the current cloud
        probes = cloud[rng.choice(len(cloud), 12, replace=False)]
        log_C, _p, _ = build_prior_mass_estimator(
            model_key, data, probes, rng, K=K, data_dir=data_dir,
            verbose=False)

        if q is None:
            psi_s = sample_hyperprior_psi(n_particles, hyperprior_arr, rng)
            log_base = np.zeros(n_particles)   # log pi - log q = 0
        else:
            psi_s = q.sample(n_particles, rng)
            log_base = log_pi_vec(psi_s) - q.log_density(psi_s)

        lc, ess_in = log_C(psi_s, want_ess=True)
        if bounded:
            lc = np.minimum(lc, 0.0)
        lc = np.where(np.isfinite(lc), lc, -np.inf)
        finite = np.isfinite(log_base)
        log_base = np.where(finite, log_base, -np.inf)

        def stage_ess(tt):
            lw = log_base + tt * lc
            w = np.exp(lw - lw.max())
            return float(np.sum(w) ** 2 / np.sum(w * w))

        # Largest t with acceptable ESS (bisection)
        t_cap = float(n_subjects if t_stop is None else min(t_stop,
                                                            n_subjects))
        t_lo, t_hi = t, t_cap
        if stage_ess(t_hi) >= target_now * n_particles:
            t_new = t_hi
        else:
            for _ in range(40):
                mid = 0.5 * (t_lo + t_hi)
                if stage_ess(mid) >= target_now * n_particles:
                    t_lo = mid
                else:
                    t_hi = mid
            t_new = t_lo
        lw = log_base + t_new * lc
        log_Z_t = float(_logsumexp(lw) - np.log(n_particles))
        w = np.exp(lw - lw.max())
        ess = float(np.sum(w) ** 2 / np.sum(w * w))

        if verbose:
            print(f"  [{model_key}] stage {stage}: t {t:.1f} -> {t_new:.1f}, "
                  f"log Z(t) = {log_Z_t:.3f}, ESS {ess:.0f}, "
                  f"inner ESS med {np.median(ess_in):.0f}")

        if t_stop is not None and t_new >= t_cap - 1e-9 \
                and ess >= 0.05 * n_particles:
            # Early stop for the Jensen lower bound: return log Z(t_stop)
            # and the tilted expectation of log C at t_stop, so that
            # log Z(N) >= log Z(t_stop) + (N - t_stop) E_{t_stop}[log C]
            # by convexity of the cumulant function t -> log Z(t).
            wn = w / w.sum()
            e_logc = float(np.sum(wn * lc))
            return {'log_Z_t': log_Z_t, 't_stop': t_new,
                    'E_logC_t': e_logc, 'ess': ess,
                    'log_Z_lower': float(log_Z_t + (n_subjects - t_new)
                                         * e_logc),
                    'n_stages': stage + 1}

        idx = rng.choice(n_particles, 4000, replace=True, p=w / w.sum())
        cloud = psi_s[idx]
        advance = t_new - t
        t = t_new
        stage += 1

        # Stall-breaker: when t barely advances, widen the proposal and
        # temporarily relax the ESS target so the anneal can traverse a
        # hard stretch of the ridge; restore both on progress.
        if advance < 0.5:
            widen = min(widen * 1.5, 6.0)
            target_now = max(target_now * 0.5, 0.02)
            if verbose:
                print(f"  [{model_key}]   stall: widen={widen:.2f}, "
                      f"target={target_now:.3f}")
        else:
            widen = max(widen / 1.3, 1.0)
            target_now = min(ess_target, target_now * 1.6)

        if t >= n_subjects - 1e-9:
            # One final fresh pass at t = N from a proposal fitted to the
            # final cloud, with a freshly refined C estimator
            q_fin = MixtureProposal(
                [fit_subject_proposal(cloud, n_bins=8, df=7.0, scale=1.3),
                 StudentTProposal(cloud, df=5.0, scale=2.0)], [0.85, 0.15])
            probes = cloud[rng.choice(len(cloud), 12, replace=False)]
            log_C, _p, _ = build_prior_mass_estimator(
                model_key, data, probes, rng, K=K, data_dir=data_dir,
                verbose=False)
            psi_f = q_fin.sample(n_particles, rng)
            lc_f, ess_in_f = log_C(psi_f, want_ess=True)
            if bounded:
                lc_f = np.minimum(lc_f, 0.0)
            lw_f = (log_pi_vec(psi_f) + n_subjects * lc_f
                    - q_fin.log_density(psi_f))
            lw_f = np.where(np.isfinite(lw_f), lw_f, -np.inf)
            log_Z = float(_logsumexp(lw_f) - np.log(n_particles))
            wf = np.exp(lw_f - lw_f.max())
            ess_f = float(np.sum(wf) ** 2 / np.sum(wf * wf))
            if verbose:
                print(f"  [{model_key}] final: log Z = {log_Z:.3f}, "
                      f"ESS {ess_f:.0f}, inner ESS med "
                      f"{np.median(ess_in_f):.0f}")
            return {'log_Z': log_Z, 'final_ess': ess_f,
                    'n_stages': stage,
                    'final_inner_ess_med': float(np.median(ess_in_f))}

        if stage >= max_stages:
            raise RuntimeError(f"Annealing did not reach t=N in "
                               f"{max_stages} stages (t={t:.1f})")

        q = MixtureProposal(
            [fit_subject_proposal(cloud, n_bins=8, df=7.0,
                                  scale=1.3 * widen),
             StudentTProposal(cloud, df=4.0, scale=2.0 * widen)],
            [0.85, 0.15])
