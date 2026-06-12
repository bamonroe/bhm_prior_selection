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

from bayes_factors_rb import (
    _load_mcmc_states,
    _build_log_f_fn,
    _build_log_pop_prior_fn,
    _build_log_hyperprior,
    _logsumexp,
)

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


def build_subject_caches(model_key, data, subject_draws, K, rng,
                         df=5.0, scale=1.5, fit_max=50000):
    """Fit per-subject Student-t proposals to posterior draws, then build
    the importance-sampling caches via build_caches_from_proposals.

    Args:
        subject_draws: (N, n_subjects, p) posterior draws (for fitting only)
        K: fresh proposal draws per subject
    """
    fit_rows = subject_draws if len(subject_draws) <= fit_max \
        else subject_draws[rng.choice(len(subject_draws), fit_max, replace=False)]

    proposals = [StudentTProposal(fit_rows[:, i, :], df=df, scale=scale)
                 for i in range(subject_draws.shape[1])]
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

    nonfinite = ~np.isfinite(log_f)
    nonfinite_frac = float(nonfinite.mean())
    log_f[nonfinite] = -np.inf

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
    mu = jax.nn.sigmoid(logit_mu)
    ps = jax.nn.sigmoid(logit_psi)

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

    return sweep


def sweep_psi(sweep_fn, psi, chunk=128, want_ess=False):
    """Apply the sweep function over psi rows in chunks (numpy in/out)."""
    psi = np.asarray(psi, dtype=np.float64)
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
# Driver
# =============================================================================

def compute_marginal_likelihoods(model_keys, data_dir, output_path,
                                 K=1024, n_psi_is=50000, n_psi_bridge=20000,
                                 n_load=100000, seed=42,
                                 psi_df=5.0, psi_scale=1.5,
                                 theta_df=5.0, theta_scale=1.5,
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

    log_pi_vec = make_log_hyperprior_vec(data)
    results = {}
    save = {'K': K, 'n_psi_is': n_psi_is, 'n_psi_bridge': n_psi_bridge,
            'seed': seed}

    for mk in model_keys:
        print(f"\n{'=' * 60}\nModel: {mk}\n{'=' * 60}")
        rng = np.random.default_rng(seed + hash(mk) % 1000)

        hyper, subj, n_subjects = _load_mcmc_states(
            mk, str(data_dir), n_samples=n_load, seed=seed)
        print(f"  Loaded {len(hyper)} draws, {n_subjects} subjects, "
              f"{subj.shape[2]} params/subject")

        print(f"  Building subject caches (K={K})...")
        caches = build_subject_caches(mk, data, subj, K, rng,
                                      df=theta_df, scale=theta_scale)
        print(f"  Non-finite log f fraction: {caches['nonfinite_frac']:.4f}")

        sweep_fn = make_psi_sweep_fn(caches)

        # Per-subject ESS at representative posterior psi draws
        idx50 = rng.choice(len(hyper), 50, replace=False)
        _, ess50 = sweep_psi(sweep_fn, hyper[idx50], want_ess=True)
        ess_min = float(np.min(ess50))
        ess_med = float(np.median(ess50))
        print(f"  Per-subject ESS at 50 posterior psi: "
              f"min={ess_min:.1f}, median={ess_med:.1f} (K={K})")
        if ess_min < ess_floor:
            print(f"  WARNING: min ESS {ess_min:.1f} below floor {ess_floor}")

        psi_proposal = StudentTProposal(hyper, df=psi_df, scale=psi_scale)

        # Route (a): importance sampling
        print(f"  Route (a): IS with {n_psi_is} psi draws...")
        res_is = log_ml_is(sweep_fn, log_pi_vec, psi_proposal, n_psi_is, rng)
        print(f"    log ML = {res_is['log_ml']:.3f} +/- {res_is['se']:.3f} "
              f"(psi ESS = {res_is['ess']:.0f})")

        # Route (b): bridge sampling
        print(f"  Route (b): bridge with {n_psi_bridge}+{n_psi_bridge} draws...")
        idx_b = rng.choice(len(hyper), n_psi_bridge, replace=False)
        res_br = log_ml_bridge(sweep_fn, log_pi_vec, hyper[idx_b],
                               psi_proposal, n_psi_bridge, rng)
        print(f"    log ML = {res_br['log_ml']:.3f} +/- {res_br['se']:.3f} "
              f"(split-half {res_br['se_split']:.3f}, "
              f"converged={res_br['converged']})")

        # Inner-noise check: two independent half-K caches
        print("  Inner-noise check (two K/2 half-caches)...")
        half_ml = []
        for h in range(2):
            half = {k: (v[h::2] if isinstance(v, np.ndarray) and
                        v.shape[:1] == (K,) else v)
                    for k, v in caches.items()}
            sweep_h = make_psi_sweep_fn(half)
            r = log_ml_is(sweep_h, log_pi_vec, psi_proposal,
                          n_psi_is // 2, rng)
            half_ml.append(r['log_ml'])
        inner_spread = float(abs(half_ml[0] - half_ml[1]))
        print(f"    half-cache log MLs: {half_ml[0]:.3f}, {half_ml[1]:.3f} "
              f"(spread {inner_spread:.3f})")

        agree = abs(res_is['log_ml'] - res_br['log_ml'])
        comb_se = np.sqrt(res_is['se'] ** 2 + res_br['se'] ** 2)
        print(f"  Route agreement: |IS - bridge| = {agree:.3f} "
              f"(2 x combined SE = {2 * comb_se:.3f}) "
              f"{'OK' if agree <= max(2 * comb_se, 0.1) else 'DISAGREE'}")

        results[mk] = {'is': res_is, 'bridge': res_br,
                       'ess_min': ess_min, 'ess_med': ess_med,
                       'inner_spread': inner_spread,
                       'nonfinite_frac': caches['nonfinite_frac']}

        save[f'{mk}_log_ml_is'] = res_is['log_ml']
        save[f'{mk}_log_ml_is_se'] = res_is['se']
        save[f'{mk}_psi_ess_is'] = res_is['ess']
        save[f'{mk}_log_ml_bridge'] = res_br['log_ml']
        save[f'{mk}_log_ml_bridge_se'] = res_br['se']
        save[f'{mk}_log_ml_bridge_se_split'] = res_br['se_split']
        save[f'{mk}_ess_subject_min'] = ess_min
        save[f'{mk}_ess_subject_med'] = ess_med
        save[f'{mk}_inner_spread'] = inner_spread
        save[f'{mk}_nonfinite_frac'] = caches['nonfinite_frac']

    print(f"\n{'=' * 60}\nBAYES FACTORS (log scale, IS route)\n{'=' * 60}")
    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            bf = results[k1]['is']['log_ml'] - results[k2]['is']['log_ml']
            bf_se = np.sqrt(results[k1]['is']['se'] ** 2
                            + results[k2]['is']['se'] ** 2)
            print(f"  log BF({k1} vs {k2}) = {bf:.2f} +/- {bf_se:.2f}")
            save[f'log_bf_rb2_{k1}_vs_{k2}'] = bf
            save[f'log_bf_rb2_{k1}_vs_{k2}_se'] = bf_se

    np.savez(Path(output_path), **save)
    print(f"\nSaved to: {output_path}")
    return results
