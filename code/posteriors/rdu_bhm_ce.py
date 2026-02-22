"""
Rank-Dependent Utility (RDU) BHM with CE-Space Prior Specification.

This posterior implements an RDU model where the prior on individual-level
parameters (r, phi_raw, eta_raw) is specified in certainty-equivalent space
rather than parameter space. The Jacobian correction transforms:

    p(theta) = p_CE(g(theta)) |det J_g(theta)|

where g maps (r, phi_raw, eta_raw) -> (CE1, CE2, CE3) for three reference
lotteries.

Parameterization:
- Subject params (r_i, phi_raw_i, eta_raw_i): CENTERED in parameter space.
  Prior is defined in CE space with Jacobian correction.
- Subject param (lambda_eps_i): NCP as in rdu_bhm.py. Lambda doesn't affect
  CEs so it can stay non-centered.

Hyperparameters (8 total):
- ce1_mean, ce1_logsd: population mean/logsd for CE1
- ce2_mean, ce2_logsd: population mean/logsd for CE2
- ce3_mean, ce3_logsd: population mean/logsd for CE3
- lambda_mean, lambda_logsd: population mean/logsd for log(lambda) (NCP)

Data format: Same as rdu_bhm, plus:
- float[7]: hyperprior_arr (16 values) — NIG hyperprior for 4 dims
- float[8]: ref_lotteries (18 values) — 3 lotteries x (3 prizes + 3 probs)
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

try:
    from bamcmc import (
        register_posterior,
        BlockSpec,
        SamplerType,
        ProposalType,
        create_subject_blocks,
    )
    BAMCMC_AVAILABLE = True
except ImportError:
    BAMCMC_AVAILABLE = False
    print("Warning: bamcmc not available. Posterior registration skipped.")

from .common_fns import (
    crra_ufn,
    risk_likelihood,
    ce_mapping_3lotteries,
    ce_jacobian_logabsdet,
)

MODEL_ID = "rdu_bhm_ce"

# =============================================================================
# Model Constants
# =============================================================================

NPARAMS = 4  # r, phi_raw, eta_raw, lambda_eps per subject
N_HYPER = 8  # (mean, logsd) for CE1, CE2, CE3, lambda
HYPER_BLOCK_SIZE = 2


# =============================================================================
# Index Helpers
# =============================================================================

def _get_indices(nids):
    """Compute index offsets for the CE-space state vector."""
    first_hyper = nids * NPARAMS

    # Hyperparameter layout:
    # [ce1_mean, ce1_logsd, ce2_mean, ce2_logsd, ce3_mean, ce3_logsd,
    #  lambda_mean, lambda_logsd]
    ce1_hyper_start = first_hyper
    ce2_hyper_start = first_hyper + HYPER_BLOCK_SIZE
    ce3_hyper_start = first_hyper + 2 * HYPER_BLOCK_SIZE
    lambda_hyper_start = first_hyper + 3 * HYPER_BLOCK_SIZE

    return {
        'first_hyper': first_hyper,
        'ce1_hyper_start': ce1_hyper_start,
        'ce2_hyper_start': ce2_hyper_start,
        'ce3_hyper_start': ce3_hyper_start,
        'lambda_hyper_start': lambda_hyper_start,
    }


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """
    CE-space BHM block specification.

    Subject blocks: (r, phi_raw, eta_raw, lambda_eps) per subject — MH.
    CE hyper blocks (x3): (mean, logsd) — MH (no coupled transform).
    Lambda hyper block: (mean, logsd) — COUPLED_TRANSFORM (NCP for lambda).
    """
    n_subjects = data["static"][0]

    proposal_type = ProposalType.MCOV_SMOOTH
    settings = {'k_alpha': 3.0, 'k_g': 12}

    # Subject blocks: 4 params per subject, standard MH
    subject_specs = create_subject_blocks(
        n_subjects=n_subjects,
        params_per_subject=NPARAMS,
        sampler_type=SamplerType.METROPOLIS_HASTINGS,
        proposal_type=proposal_type,
        settings=settings,
        label_prefix="Subj"
    )

    # CE hyper blocks: standard MH (centered parameterization — no coupled transform)
    ce_hyper_specs = [
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=proposal_type,
            settings=settings,
            label="Hyper_CE1",
        ),
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=proposal_type,
            settings=settings,
            label="Hyper_CE2",
        ),
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=proposal_type,
            settings=settings,
            label="Hyper_CE3",
        ),
    ]

    # Lambda hyper block: COUPLED_TRANSFORM (NCP for lambda_eps)
    lambda_hyper_spec = [
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=proposal_type,
            settings=settings,
            label="Hyper_lambda",
        ),
    ]

    return subject_specs + ce_hyper_specs + lambda_hyper_spec


# =============================================================================
# Hyperprior Functions
# =============================================================================

def _hyperprior_mean_logsd(mean_val, logsd_val, mu_0, tau_0_sq, alpha_0, beta_0):
    """
    Log hyperprior for a (mean, logsd) pair.

    Same as rdu_bhm._hyperprior_mean_logsd:
        mean ~ Normal(mu_0, tau_0_sq)
        sigma^2 ~ InverseGamma(alpha_0, beta_0)
        logsd = 0.5 * log(sigma^2)
    """
    lp_mean = stats.norm.logpdf(mean_val, loc=mu_0, scale=jnp.sqrt(tau_0_sq))

    sigma_sq = jnp.exp(2 * logsd_val)
    lp_var = (alpha_0 * jnp.log(beta_0) - jax.scipy.special.gammaln(alpha_0)
              - (alpha_0 + 1) * jnp.log(sigma_sq) - beta_0 / sigma_sq)

    lp_jacobian = jnp.log(2.0) + 2 * logsd_val

    return lp_mean + lp_var + lp_jacobian


def _compute_all_hyperpriors(chain_state, idx, hyperprior_arr):
    """
    Compute sum of all hyperprior terms.

    Hyperprior array layout (16 values):
    [ce1_mu_0, ce1_tau_0_sq, ce1_alpha_0, ce1_beta_0,
     ce2_mu_0, ce2_tau_0_sq, ce2_alpha_0, ce2_beta_0,
     ce3_mu_0, ce3_tau_0_sq, ce3_alpha_0, ce3_beta_0,
     lambda_mu_0, lambda_tau_0_sq, lambda_alpha_0, lambda_beta_0]
    """
    lp = 0.0

    hyper_keys = ['ce1_hyper_start', 'ce2_hyper_start',
                  'ce3_hyper_start', 'lambda_hyper_start']
    for i, hyper_key in enumerate(hyper_keys):
        mean_val = chain_state[idx[hyper_key]]
        logsd_val = chain_state[idx[hyper_key] + 1]
        mu_0 = hyperprior_arr[i*4 + 0]
        tau_0_sq = hyperprior_arr[i*4 + 1]
        alpha_0 = hyperprior_arr[i*4 + 2]
        beta_0 = hyperprior_arr[i*4 + 3]
        lp += _hyperprior_mean_logsd(mean_val, logsd_val, mu_0, tau_0_sq, alpha_0, beta_0)

    return lp


# =============================================================================
# Subject Parameter Extraction
# =============================================================================

def _get_subject_params(chain_state, sid, idx):
    """
    Extract actual parameters for a subject.

    For CE-space model:
    - r, phi_raw, eta_raw are stored directly (centered parameterization)
    - lambda_eps is NCP: lambda_raw = lambda_mean + exp(lambda_logsd) * lambda_eps
    """
    subject_start = sid * NPARAMS

    r_val = chain_state[subject_start + 0]
    phi_raw = chain_state[subject_start + 1]
    eta_raw = chain_state[subject_start + 2]
    lambda_eps = chain_state[subject_start + 3]

    # NCP transform for lambda only
    lambda_mean = chain_state[idx['lambda_hyper_start']]
    lambda_sigma = jnp.exp(chain_state[idx['lambda_hyper_start'] + 1])
    lambda_raw = lambda_mean + lambda_sigma * lambda_eps

    return r_val, phi_raw, eta_raw, lambda_raw


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """
    Log posterior for RDU CE-space hierarchical model.
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        """Log posterior for hyperparameter blocks."""
        datai = data["int"]
        dataf = data["float"]

        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])

        lp_hyper = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        datai_jax = tuple(jnp.asarray(arr) for arr in datai)
        dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
        unique_outcomes = dataf_jax[0]

        # CE hyperparameters
        ce1_mean = chain_state[idx['ce1_hyper_start']]
        ce1_sd = jnp.exp(chain_state[idx['ce1_hyper_start'] + 1])
        ce2_mean = chain_state[idx['ce2_hyper_start']]
        ce2_sd = jnp.exp(chain_state[idx['ce2_hyper_start'] + 1])
        ce3_mean = chain_state[idx['ce3_hyper_start']]
        ce3_sd = jnp.exp(chain_state[idx['ce3_hyper_start'] + 1])

        def subject_ll_body(sid, acc):
            r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
                chain_state, sid, idx)

            # CE-space prior on (r, phi_raw, eta_raw):
            # 1. Compute CEs via forward map
            ces = ce_mapping_3lotteries(r_val, phi_raw, eta_raw, ref_lotteries)
            # 2. Normal population prior on each CE
            lp_ce = (stats.norm.logpdf(ces[0], ce1_mean, ce1_sd)
                     + stats.norm.logpdf(ces[1], ce2_mean, ce2_sd)
                     + stats.norm.logpdf(ces[2], ce3_mean, ce3_sd))
            # 3. Jacobian correction
            log_det_j = ce_jacobian_logabsdet(r_val, phi_raw, eta_raw, ref_lotteries)
            lp_ce += log_det_j

            # NCP prior on lambda_eps
            lambda_eps = chain_state[sid * NPARAMS + 3]
            lp_lambda = stats.norm.logpdf(lambda_eps, 0.0, 1.0)

            # Data likelihood
            phi = jnp.exp(phi_raw)
            eta = jnp.exp(eta_raw)
            lam = jnp.exp(lambda_raw)
            unique_utils = crra_ufn(unique_outcomes, r_val)
            ll = risk_likelihood(unique_utils, phi, eta, datai_jax, dataf_jax, sid, lam)

            return acc + lp_ce + lp_lambda + beta * ll

        total = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)

        return lp_hyper + total

    def compute_subject_posterior():
        """Log posterior for subject parameter blocks."""
        sid = first_param // NPARAMS
        r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
            chain_state, sid, idx)

        dataf = data["float"]
        ref_lotteries = jnp.asarray(dataf[8])

        # CE-space prior on (r, phi_raw, eta_raw)
        ce1_mean = chain_state[idx['ce1_hyper_start']]
        ce1_sd = jnp.exp(chain_state[idx['ce1_hyper_start'] + 1])
        ce2_mean = chain_state[idx['ce2_hyper_start']]
        ce2_sd = jnp.exp(chain_state[idx['ce2_hyper_start'] + 1])
        ce3_mean = chain_state[idx['ce3_hyper_start']]
        ce3_sd = jnp.exp(chain_state[idx['ce3_hyper_start'] + 1])

        ces = ce_mapping_3lotteries(r_val, phi_raw, eta_raw, ref_lotteries)
        lp_ce = (stats.norm.logpdf(ces[0], ce1_mean, ce1_sd)
                 + stats.norm.logpdf(ces[1], ce2_mean, ce2_sd)
                 + stats.norm.logpdf(ces[2], ce3_mean, ce3_sd))
        log_det_j = ce_jacobian_logabsdet(r_val, phi_raw, eta_raw, ref_lotteries)
        lp_ce += log_det_j

        # NCP prior on lambda_eps
        lambda_eps = chain_state[sid * NPARAMS + 3]
        lp_lambda = stats.norm.logpdf(lambda_eps, 0.0, 1.0)

        # Data likelihood
        phi = jnp.exp(phi_raw)
        eta = jnp.exp(eta_raw)
        lam = jnp.exp(lambda_raw)

        datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
        dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
        unique_outcomes = dataf_jax[0]
        unique_utils = crra_ufn(unique_outcomes, r_val)
        log_like = risk_likelihood(unique_utils, phi, eta, datai_jax, dataf_jax, sid, lam)

        return lp_ce + lp_lambda + beta * log_like

    return jax.lax.cond(
        is_hyper,
        compute_hyper_posterior,
        compute_subject_posterior
    )


# =============================================================================
# Coupled Transform Dispatch (for lambda NCP only)
# =============================================================================

def coupled_transform_dispatch(key, chain_state, primary_indices, proposed_primary, data):
    """
    Dispatch function for COUPLED_TRANSFORM blocks.

    Only lambda hyperparameters use coupled transform (NCP).
    CE hyper blocks use standard MH — they should never reach here.
    """
    nids = data["static"][0]
    idx = _get_indices(nids)

    # Lambda hyper block: transform lambda_eps to preserve theta
    hyper_start = primary_indices[0]

    mu_old = chain_state[hyper_start]
    logsd_old = chain_state[hyper_start + 1]
    sigma_old = jnp.exp(logsd_old)

    mu_new = proposed_primary[0]
    logsd_new = proposed_primary[1]
    sigma_new = jnp.exp(logsd_new)

    # Lambda epsilon is at offset 3 within each subject's 4 params
    subject_offsets = jnp.arange(nids) * NPARAMS
    coupled_indices = subject_offsets + 3  # lambda_eps position

    old_eps = chain_state[coupled_indices]
    theta = mu_old + sigma_old * old_eps
    new_eps = (theta - mu_new) / sigma_new

    log_jacobian = nids * (jnp.log(sigma_old) - jnp.log(sigma_new))
    log_prior_ratio = -0.5 * (jnp.sum(new_eps**2) - jnp.sum(old_eps**2))

    return coupled_indices, new_eps, log_jacobian, log_prior_ratio, key


# =============================================================================
# Generated Quantities
# =============================================================================

def generated_quantities(chain_state, data):
    """
    Generated quantities: transformed parameters on natural scale.

    Returns:
    - r, phi, eta, lambda for each subject
    - Population means and standard deviations (CE-space for first 3, lambda for 4th)
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    def get_subject_theta(sid):
        r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
            chain_state, sid, idx)
        phi = jnp.exp(phi_raw)
        eta = jnp.exp(eta_raw)
        lam = jnp.exp(lambda_raw)
        return jnp.array([r_val, phi, eta, lam])

    subject_params = jax.vmap(get_subject_theta)(jnp.arange(n_subjects))
    r_subjects = subject_params[:, 0]
    phi_subjects = subject_params[:, 1]
    eta_subjects = subject_params[:, 2]
    lambda_subjects = subject_params[:, 3]

    # Population parameters (CE-space means/SDs for first 3)
    ce1_pop_mean = chain_state[idx['ce1_hyper_start']]
    ce1_pop_sd = jnp.exp(chain_state[idx['ce1_hyper_start'] + 1])
    ce2_pop_mean = chain_state[idx['ce2_hyper_start']]
    ce2_pop_sd = jnp.exp(chain_state[idx['ce2_hyper_start'] + 1])
    ce3_pop_mean = chain_state[idx['ce3_hyper_start']]
    ce3_pop_sd = jnp.exp(chain_state[idx['ce3_hyper_start'] + 1])
    lambda_pop_mean = chain_state[idx['lambda_hyper_start']]
    lambda_pop_sd = jnp.exp(chain_state[idx['lambda_hyper_start'] + 1])

    lambda_pop_mean_natural = jnp.exp(lambda_pop_mean)

    return jnp.concatenate([
        r_subjects,
        phi_subjects,
        eta_subjects,
        lambda_subjects,
        jnp.array([
            ce1_pop_mean, ce1_pop_sd,
            ce2_pop_mean, ce2_pop_sd,
            ce3_pop_mean, ce3_pop_sd,
            lambda_pop_mean_natural, lambda_pop_sd
        ])
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    n_subjects = data["static"][0]
    return 4 * n_subjects + 8


# =============================================================================
# Hyperprior Array (from loaded specs)
# =============================================================================

# These can be overwritten by load_hyperpriors_from_file or set dynamically
HYPERPRIOR_SPECS = {
    'ce1': {
        'mu_0': 27.0, 'tau_0_sq': 5.0,
        'alpha_0': 6.0, 'beta_0': 20.0,
    },
    'ce2': {
        'mu_0': 22.0, 'tau_0_sq': 5.0,
        'alpha_0': 6.0, 'beta_0': 20.0,
    },
    'ce3': {
        'mu_0': 25.0, 'tau_0_sq': 5.0,
        'alpha_0': 6.0, 'beta_0': 20.0,
    },
    'lambda_raw': {
        'mu_0': -2.0, 'tau_0_sq': 0.3,
        'alpha_0': 6.0, 'beta_0': 3.5,
    },
}

PARAM_NAMES = ('ce1', 'ce2', 'ce3', 'lambda_raw')


def load_hyperpriors_from_file(filepath):
    """
    Load hyperprior specifications from a prior_params_ce.npz file.

    Updates the global HYPERPRIOR_SPECS with values from the file.
    """
    global HYPERPRIOR_SPECS

    data = np.load(filepath)

    new_specs = {}
    for i, name in enumerate(PARAM_NAMES):
        new_specs[name] = {
            'mu_0': float(data[f'{name}_mu_0']),
            'tau_0_sq': float(data[f'{name}_tau_0_sq']),
            'alpha_0': float(data[f'{name}_alpha_0']),
            'beta_0': float(data[f'{name}_beta_0']),
        }

    HYPERPRIOR_SPECS.update(new_specs)
    print(f"Loaded CE hyperpriors from {filepath}")
    for name in PARAM_NAMES:
        s = new_specs[name]
        print(f"  {name}: mu_0={s['mu_0']:.2f}, tau_0_sq={s['tau_0_sq']:.4f}, "
              f"alpha_0={s['alpha_0']:.2f}, beta_0={s['beta_0']:.4f}")

    return new_specs


def get_hyperprior_array():
    """
    Convert HYPERPRIOR_SPECS to a flat numpy array for passing through data.

    Returns array of shape (16,) with layout:
    [ce1_mu_0, ce1_tau_0_sq, ce1_alpha_0, ce1_beta_0,
     ce2_mu_0, ce2_tau_0_sq, ce2_alpha_0, ce2_beta_0,
     ce3_mu_0, ce3_tau_0_sq, ce3_alpha_0, ce3_beta_0,
     lambda_mu_0, lambda_tau_0_sq, lambda_alpha_0, lambda_beta_0]
    """
    arr = np.zeros(16, dtype=np.float64)
    for i, param_name in enumerate(PARAM_NAMES):
        spec = HYPERPRIOR_SPECS[param_name]
        arr[i*4 + 0] = spec['mu_0']
        arr[i*4 + 1] = spec['tau_0_sq']
        arr[i*4 + 2] = spec['alpha_0']
        arr[i*4 + 3] = spec['beta_0']
    return arr


# =============================================================================
# Initial Values
# =============================================================================

def initial_vector(mcmc_config, data):
    """
    Generate initial parameter values for all chains.

    Subject params: r near 0.5, phi_raw/eta_raw near 0, lambda_eps near 0.
    CE hypers: near expected CE values.
    Lambda hyper: near typical values.
    """
    n_subjects = data["static"][0]
    total_params = n_subjects * NPARAMS + N_HYPER

    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))

    init = np.zeros((num_chains, total_params))
    first_hyper = n_subjects * NPARAMS

    # Subject parameters: initialize near reasonable values
    INIT_STD = 0.3
    init[:, 0:first_hyper:NPARAMS] = rng.normal(0.5, INIT_STD, (num_chains, n_subjects))   # r
    init[:, 1:first_hyper:NPARAMS] = rng.normal(0.0, INIT_STD, (num_chains, n_subjects))   # phi_raw
    init[:, 2:first_hyper:NPARAMS] = rng.normal(0.0, INIT_STD, (num_chains, n_subjects))   # eta_raw
    init[:, 3:first_hyper:NPARAMS] = rng.normal(0.0, 0.5, (num_chains, n_subjects))        # lambda_eps

    # CE hyperparameters: initialize near prior means from HYPERPRIOR_SPECS
    HYPER_INIT_STD = 0.02
    ce1_init_mean = HYPERPRIOR_SPECS['ce1']['mu_0']
    ce2_init_mean = HYPERPRIOR_SPECS['ce2']['mu_0']
    ce3_init_mean = HYPERPRIOR_SPECS['ce3']['mu_0']

    init[:, first_hyper + 0] = rng.normal(ce1_init_mean, HYPER_INIT_STD, num_chains)  # ce1_mean
    init[:, first_hyper + 1] = rng.normal(0.5, HYPER_INIT_STD, num_chains)             # ce1_logsd
    init[:, first_hyper + 2] = rng.normal(ce2_init_mean, HYPER_INIT_STD, num_chains)  # ce2_mean
    init[:, first_hyper + 3] = rng.normal(0.5, HYPER_INIT_STD, num_chains)             # ce2_logsd
    init[:, first_hyper + 4] = rng.normal(ce3_init_mean, HYPER_INIT_STD, num_chains)  # ce3_mean
    init[:, first_hyper + 5] = rng.normal(0.5, HYPER_INIT_STD, num_chains)             # ce3_logsd

    # Lambda hyperparameters (same as rdu_bhm)
    init[:, first_hyper + 6] = rng.normal(-2.0, HYPER_INIT_STD, num_chains)  # lambda_mean
    init[:, first_hyper + 7] = rng.normal(0.0, HYPER_INIT_STD, num_chains)   # lambda_logsd

    return init.flatten()


# =============================================================================
# Direct Sampler (Placeholder)
# =============================================================================

def direct_sampler(key, chain_state, param_indices, data):
    """Placeholder for MH-only models. Required by bamcmc but not used."""
    return chain_state, key


# =============================================================================
# Registration
# =============================================================================

def register():
    """Register the posterior with bamcmc."""
    if not BAMCMC_AVAILABLE:
        print(f"Cannot register {MODEL_ID}: bamcmc not available")
        return

    register_posterior(MODEL_ID, {
        'log_posterior': log_posterior,
        'batch_type': batch_type,
        'initial_vector': initial_vector,
        'direct_sampler': direct_sampler,
        'generated_quantities': generated_quantities,
        'get_num_gq': get_num_gq,
        'coupled_transform_dispatch': coupled_transform_dispatch,
    })
    print(f"Registered posterior: {MODEL_ID}")


# Register on import if bamcmc is available
if BAMCMC_AVAILABLE:
    register()
