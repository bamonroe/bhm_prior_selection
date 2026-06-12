"""
Synthetic-data validation of the corrected RB marginal likelihood estimator.

Builds a small CRRA+Prelec dataset (default 5 subjects x 20 choices) with
known generating parameters, computes the marginal likelihood by brute
force, runs the standard MCMC on the synthetic data, and checks that the
corrected Rao-Blackwellized estimator (marginal_likelihood.py) recovers
the brute-force value.

The brute-force ground truth exploits the small scale:

    log p(y) = log (1/S) sum_s prod_i p_hat(y_i | psi_s),   psi_s ~ pi(psi)

with psi drawn EXACTLY from the Beta-Beta hyperprior (so the outer
integral has no importance weights), and each p_hat(y_i|psi) computed by
per-subject importance sampling with K0 wide Student-t draws centered at
the generating theta. Two different inner proposals must agree.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from marginal_likelihood import (
    assemble_data,
    StudentTProposal,
    MixtureProposal,
    build_caches_from_proposals,
    build_subject_caches,
    make_psi_sweep_fn,
    make_log_hyperprior_vec,
    sweep_psi,
    log_ml_is,
    log_ml_bridge,
    _beta_pop_coeffs,
    _build_quantity_fn,
    _bounds_from_hyperprior,
    _logsumexp,
    N_QUANTITIES,
    HYPERPRIOR_STRIDE,
)

# Generating parameters: (r, phi_raw, eta_raw, lambda_raw) per subject,
# spanning a plausible range of risk attitudes and PWF shapes.
THETA_TRUE = np.array([
    [0.30, np.log(0.70), np.log(0.95), np.log(0.10)],
    [0.50, np.log(0.90), np.log(1.00), np.log(0.15)],
    [0.65, np.log(1.10), np.log(1.10), np.log(0.08)],
    [0.80, np.log(0.80), np.log(0.90), np.log(0.20)],
    [0.45, np.log(1.00), np.log(1.05), np.log(0.12)],
])


# =============================================================================
# Choice simulation (mirrors risk_likelihood in common_fns.py)
# =============================================================================

def choice_probs_crra(theta_i, sid, data):
    """P(choose Left) for every choice of subject sid (CRRA + Prelec).

    Mirrors the Contextual Utility likelihood in
    posteriors.common_fns.risk_likelihood.
    """
    from posteriors.common_fns import crra_ufn, rdu_value_3outcome

    datai = tuple(jnp.asarray(a) for a in data['int'])
    dataf = tuple(jnp.asarray(a) for a in data['float'])
    utilities = crra_ufn(dataf[0], theta_i[0])
    phi = jnp.exp(theta_i[1])
    eta = jnp.exp(theta_i[2])
    lam = jnp.exp(theta_i[3])

    def one_choice(iL1, iL2, iL3, iR1, iR2, iR3,
                   pL1, pL2, pL3, pR1, pR2, pR3):
        v_L = rdu_value_3outcome(utilities, phi, eta, iL1, iL2, iL3,
                                 pL1, pL2, pL3)
        v_R = rdu_value_3outcome(utilities, phi, eta, iR1, iR2, iR3,
                                 pR1, pR2, pR3)
        us = jnp.stack([utilities[iL1], utilities[iL2], utilities[iL3],
                        utilities[iR1], utilities[iR2], utilities[iR3]])
        u_range = jnp.maximum(jnp.max(us) - jnp.min(us), 1e-10)
        delta = (v_L - v_R) / (lam * u_range)
        return jax.nn.sigmoid(delta)

    return np.asarray(jax.vmap(one_choice)(
        datai[1][sid], datai[2][sid], datai[3][sid],
        datai[4][sid], datai[5][sid], datai[6][sid],
        dataf[1][sid], dataf[2][sid], dataf[3][sid],
        dataf[4][sid], dataf[5][sid], dataf[6][sid],
    ))


def build_synthetic_dataset(data_dir, n_subjects=5, n_choices=20, seed=1234):
    """Slice the real indexed data to a small panel and substitute
    simulated choices generated from THETA_TRUE.

    Only arrays whose leading dimension is n_subjects are sliced;
    unique_outcomes and the hyperprior/reference arrays are kept intact.
    """
    full = assemble_data(data_dir)

    datai = tuple(np.ascontiguousarray(a[:n_subjects, :n_choices])
                  for a in full['int'])
    dataf = (full['float'][0],) + tuple(
        np.ascontiguousarray(a[:n_subjects, :n_choices])
        for a in full['float'][1:7]) + full['float'][7:]

    data = {
        'static': (n_subjects, n_choices, full['static'][2]),
        'int': datai,
        'float': dataf,
    }

    rng = np.random.default_rng(seed)
    choices = np.empty((n_subjects, n_choices), dtype=np.int32)
    for i in range(n_subjects):
        p_left = choice_probs_crra(jnp.asarray(THETA_TRUE[i]), i, data)
        choices[i] = (rng.uniform(size=n_choices) < p_left).astype(np.int32)

    data['int'] = (choices,) + data['int'][1:]
    return data


# =============================================================================
# Exact hyperprior draws
# =============================================================================

def sample_hyperprior(n, hyperprior_arr, rng):
    """Draw psi (n, 8) exactly from the Beta-Beta hyperprior (logit scale)."""
    hp = np.asarray(hyperprior_arr).reshape(N_QUANTITIES, HYPERPRIOR_STRIDE)
    psi = np.empty((n, 2 * N_QUANTITIES))
    for k in range(N_QUANTITIES):
        mu = rng.beta(hp[k, 0], hp[k, 1], size=n)
        ps = rng.beta(hp[k, 2], hp[k, 3], size=n)
        psi[:, 2 * k] = np.log(mu / (1 - mu))
        psi[:, 2 * k + 1] = np.log(ps / (1 - ps))
    return psi


# =============================================================================
# Brute-force ground truth
# =============================================================================

GT_PROPOSALS = {
    'wide': {'df': 5.0, 'sd': np.array([0.35, 0.45, 0.45, 0.70]),
             'scale': 1.3, 'M': 400000, 'S0': 512},
    'wider': {'df': 3.0, 'sd': np.array([0.50, 0.65, 0.65, 1.00]),
              'scale': 1.6, 'M': 400000, 'S0': 512},
}


def fit_prior_theta_proposal(data, rng, sd, df, scale, M=400000, S0=512,
                             n_resample=20000):
    """Fit a Student-t to the MARGINAL prior of theta.

    The Beta-Beta hyperprior is tight, so for any psi the population prior
    concentrates in a small theta ball. A proposal centered at the
    generating theta misses it; this one targets it directly.

    Procedure: draw theta from a wide Student-t, weight by
    E_psi[p_pop(theta|psi)] x |J(theta)| / q_wide(theta) with psi drawn
    exactly from the hyperprior, resample, fit.
    """
    from posteriors.common_fns import full_jacobian_logabsdet

    hyperprior_arr = np.asarray(data['float'][7])
    lo, hi = _bounds_from_hyperprior(hyperprior_arr)
    ref_lotteries = jnp.asarray(data['float'][8])
    ref_pair = jnp.asarray(data['float'][9])

    center = np.array([0.5, 0.0, 0.0, -2.0])
    wide = StudentTProposal.from_moments(center, np.diag(sd ** 2), df=df)
    th = wide.sample(M, rng)

    qfn = _build_quantity_fn('crra', data)
    quantities = np.asarray(jax.jit(jax.vmap(qfn))(jnp.asarray(th)))
    z = np.clip((quantities - lo) / (hi - lo), 1e-10, 1 - 1e-10)
    logz = jnp.asarray(np.log(z))          # (M, 4)
    log1mz = jnp.asarray(np.log1p(-z))     # (M, 4)

    psi = sample_hyperprior(S0, hyperprior_arr, rng)
    a, b, const = _beta_pop_coeffs(jnp.asarray(psi), hyperprior_arr)
    core = (logz @ (a - 1.0).T + log1mz @ (b - 1.0).T
            + const[None, :])              # (M, S0)
    log_pop = np.asarray(jax.scipy.special.logsumexp(core, axis=1)
                         - np.log(S0))

    jac = jax.jit(jax.vmap(lambda t: full_jacobian_logabsdet(
        t[0], t[1], t[2], t[3], ref_lotteries, ref_pair)))
    log_j = np.asarray(jac(jnp.asarray(th)))

    log_w = log_pop + log_j - wide.log_density(th)
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    w = np.exp(log_w - np.max(log_w))
    ess = float(np.sum(w) ** 2 / np.sum(w * w))
    idx = rng.choice(M, n_resample, replace=True, p=w / w.sum())
    print(f"  prior-theta proposal: resampling ESS = {ess:.0f} of {M}")
    return StudentTProposal(th[idx], df=df, scale=scale)


def ground_truth_log_ml(data, theta_center, S=65536, K0=16384,
                        proposal='wide', seed=7, n_replicates=8, chunk=64):
    """Brute-force log p(y): exact-hyperprior outer draws, per-subject
    importance sampling for the inner integrals.

    The estimator mean_s prod_i p_hat(y_i|psi_s) is unbiased for p(y) on
    the natural scale for ANY inner sample size, but reusing one theta
    cache across all psi draws correlates the summands, so a single-cache
    batch SE understates the error. We therefore run independent
    replicates, each with FRESH theta draws and fresh psi draws, and take
    the SE from the replicate-level means.

    Inner proposal per subject: defensive mixture of (i) a Student-t
    fitted to the marginal prior of theta (where the integrand lives when
    data are weak), (ii) one centered at the generating theta, and
    (iii) a wide component bounding the importance weights.
    """
    rng = np.random.default_rng(seed)
    spec = GT_PROPOSALS[proposal]
    hyperprior_arr = np.asarray(data['float'][7])

    prior_prop = fit_prior_theta_proposal(
        data, rng, sd=spec['sd'], df=spec['df'], scale=spec['scale'],
        M=spec['M'], S0=spec['S0'])
    wide_center = np.array([0.5, 0.0, 0.0, -2.0])

    proposals = [
        MixtureProposal([
            prior_prop,
            StudentTProposal.from_moments(theta_center[i],
                                          np.diag(spec['sd'] ** 2),
                                          df=spec['df']),
            StudentTProposal.from_moments(wide_center,
                                          np.diag(spec['sd'] ** 2),
                                          df=spec['df']),
        ], [0.6, 0.2, 0.2])
        for i in range(len(theta_center))
    ]

    S_rep = S // n_replicates
    rep_log_means = np.empty(n_replicates)
    ess_all = []
    nonfinite = []
    for rep in range(n_replicates):
        caches = build_caches_from_proposals('crra', data, proposals, K0, rng)
        nonfinite.append(caches['nonfinite_frac'])
        sweep_fn = make_psi_sweep_fn(caches)
        psi = sample_hyperprior(S_rep, hyperprior_arr, rng)
        log_py, ess = sweep_psi(sweep_fn, psi, chunk=chunk, want_ess=True)
        ess_all.append(ess)
        rep_log_means[rep] = _logsumexp(log_py) - np.log(S_rep)

    ess_all = np.concatenate(ess_all)
    print(f"  [{proposal}] non-finite log f fraction: "
          f"{np.mean(nonfinite):.5f}")
    print(f"  [{proposal}] inner ESS over outer draws: "
          f"min={ess_all.min():.0f}, median={np.median(ess_all):.0f} "
          f"(K0={K0}, {n_replicates} replicates)")

    # Combine on the natural scale; SE from independent replicate means
    log_ml = _logsumexp(rep_log_means) - np.log(n_replicates)
    bm = np.exp(rep_log_means - log_ml)
    se = float(np.std(bm, ddof=1) / np.sqrt(n_replicates))
    print(f"  [{proposal}] replicate log means: "
          f"[{rep_log_means.min():.3f}, {rep_log_means.max():.3f}]")
    print(f"  [{proposal}] log ML = {log_ml:.4f} +/- {se:.4f}")
    return {'log_ml': float(log_ml), 'se': se,
            'rep_spread': float(rep_log_means.max() - rep_log_means.min())}


# =============================================================================
# Synthetic MCMC
# =============================================================================

def run_synthetic_mcmc(data, output_dir, num_collect=200, reset_runs=1,
                       resume_runs=3, seed=42):
    """Run the standard rdu_bhm_ce sampler on the synthetic dataset."""
    from bamcmc import rmcmc
    import posteriors.rdu_bhm_ce  # noqa: F401 (registers the posterior)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcmc_config = {
        'posterior_id': 'rdu_bhm_ce',
        'num_chains_a': 200,
        'num_chains_b': 200,
        'burn_iter': 0,
        'num_collect': num_collect,
        'thin_iteration': 10,
        'chunk_size': 100,
        'rng_seed': seed,
        'use_double': True,
        'n_temperatures': 1,
        'run_schedule': [('reset', reset_runs), ('resume', resume_runs)],
    }
    return rmcmc(mcmc_config, data, output_dir=str(output_dir),
                 calculate_rhat=True, burn_in_fresh=True)


def load_synthetic_states(output_dir, n_subjects, nparams=4, n_hyper=8,
                          n_samples=None, seed=42):
    """Load raw states from the synthetic run's history files.

    rmcmc writes to {output_dir}/{posterior_id}/history/full/.
    """
    history_dir = Path(output_dir) / 'rdu_bhm_ce' / 'history' / 'full'
    files = sorted(history_dir.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f"No history files in {history_dir}")

    n_raw = n_subjects * nparams + n_hyper
    all_states = [np.load(f, allow_pickle=True)['history'][:, :, :n_raw]
                  for f in files]
    combined = np.concatenate(all_states, axis=0)
    burn = combined.shape[0] // 2
    flat = combined[burn:].reshape(-1, n_raw)

    if n_samples is not None and n_samples < len(flat):
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(len(flat), n_samples, replace=False)]

    subject = flat[:, :n_subjects * nparams].reshape(-1, n_subjects, nparams)
    hyper = flat[:, n_subjects * nparams:n_subjects * nparams + n_hyper]
    return hyper, subject


# =============================================================================
# Full validation pipeline
# =============================================================================

def run_ml_validation(data_dir, output_path, n_subjects=5, n_choices=20,
                      skip_mcmc_if_exists=True, K=1024, n_psi_is=50000,
                      n_psi_bridge=20000, seed=42):
    data_dir = Path(data_dir)
    mcmc_dir = data_dir / 'ml_validation' / 'rdu_bhm_ce_synth'

    print("=" * 60)
    print("SYNTHETIC VALIDATION: corrected RB estimator vs brute force")
    print("=" * 60)

    data = build_synthetic_dataset(data_dir, n_subjects, n_choices)
    n_left = int(np.sum(data['int'][0]))
    print(f"Synthetic data: {n_subjects} subjects x {n_choices} choices, "
          f"{n_left} Left choices")

    print("\nBrute-force ground truth:")
    gt = {p: ground_truth_log_ml(data, THETA_TRUE, proposal=p, seed=7 + j,
                                 n_replicates=16)
          for j, p in enumerate(GT_PROPOSALS)}
    gt_diff = abs(gt['wide']['log_ml'] - gt['wider']['log_ml'])
    gt_se = np.sqrt(gt['wide']['se'] ** 2 + gt['wider']['se'] ** 2)
    print(f"  proposal agreement: |diff| = {gt_diff:.4f} "
          f"(2 x combined SE = {2 * gt_se:.4f}) "
          f"{'OK' if gt_diff <= max(2 * gt_se, 0.05) else 'DISAGREE'}")

    hist_dir = mcmc_dir / 'rdu_bhm_ce' / 'history' / 'full'
    have_history = hist_dir.exists() and list(hist_dir.glob('*.npz'))
    if have_history and skip_mcmc_if_exists:
        print(f"\nReusing existing synthetic MCMC in {mcmc_dir}")
    else:
        print(f"\nRunning synthetic MCMC -> {mcmc_dir}")
        run_synthetic_mcmc(data, mcmc_dir)

    hyper, subject = load_synthetic_states(mcmc_dir, n_subjects,
                                           n_samples=200000, seed=seed)
    print(f"Loaded {len(hyper)} posterior draws")

    rng = np.random.default_rng(seed)
    caches = build_subject_caches('crra', data, subject, K, rng)
    print(f"Cache non-finite log f fraction: {caches['nonfinite_frac']:.5f}")
    sweep_fn = make_psi_sweep_fn(caches)
    log_pi_vec = make_log_hyperprior_vec(data)

    idx50 = rng.choice(len(hyper), 50, replace=False)
    _, ess50 = sweep_psi(sweep_fn, hyper[idx50], want_ess=True)
    print(f"Per-subject ESS at posterior psi: min={ess50.min():.0f}, "
          f"median={np.median(ess50):.0f} (K={K})")

    psi_proposal = StudentTProposal(hyper, df=5.0, scale=1.5)

    print("\nCorrected RB estimator:")
    res_is = log_ml_is(sweep_fn, log_pi_vec, psi_proposal, n_psi_is, rng)
    print(f"  IS route:     log ML = {res_is['log_ml']:.4f} "
          f"+/- {res_is['se']:.4f} (psi ESS = {res_is['ess']:.0f})")

    idx_b = rng.choice(len(hyper), n_psi_bridge, replace=False)
    res_br = log_ml_bridge(sweep_fn, log_pi_vec, hyper[idx_b],
                           psi_proposal, n_psi_bridge, rng)
    print(f"  Bridge route: log ML = {res_br['log_ml']:.4f} "
          f"+/- {res_br['se']:.4f} (split-half {res_br['se_split']:.4f})")

    print("\nVerdict:")
    gt_ml = gt['wide']['log_ml']
    ok_all = gt_diff <= max(2 * gt_se, 0.05)
    for label, res in (('IS', res_is), ('bridge', res_br)):
        diff = abs(res['log_ml'] - gt_ml)
        tol = max(0.1, 2 * np.sqrt(res['se'] ** 2 + gt['wide']['se'] ** 2))
        ok = diff <= tol
        ok_all = ok_all and ok
        print(f"  {label:6s} vs ground truth: |diff| = {diff:.4f} "
              f"(tol {tol:.4f}) {'OK' if ok else 'FAIL'}")

    np.savez(Path(output_path),
             gt_wide_log_ml=gt['wide']['log_ml'], gt_wide_se=gt['wide']['se'],
             gt_wider_log_ml=gt['wider']['log_ml'],
             gt_wider_se=gt['wider']['se'],
             rb_is_log_ml=res_is['log_ml'], rb_is_se=res_is['se'],
             rb_bridge_log_ml=res_br['log_ml'], rb_bridge_se=res_br['se'],
             rb_bridge_se_split=res_br['se_split'],
             ess_subject_min=float(ess50.min()),
             n_subjects=n_subjects, n_choices=n_choices, K=K,
             passed=bool(ok_all))
    print(f"\n{'PASSED' if ok_all else 'FAILED'} — saved to {output_path}")
    return ok_all
