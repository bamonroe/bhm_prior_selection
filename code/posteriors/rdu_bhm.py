"""
Rank-Dependent Utility (RDU) Bayesian Hierarchical Model - Non-Centered Parameterization.

This posterior implements an RDU model for lottery choices with hierarchical
structure across subjects using Non-Centered Parameterization (NCP).

NCP Benefits:
- Breaks the "funnel" coupling between hyperparameters and subject parameters
- Dramatically improves mixing for hierarchical models
- Hyperparameter updates can achieve high acceptance rates via theta-preserving transforms

Parameterization:
- Subject parameters store epsilon ~ N(0,1) standardized deviations
- Actual parameter: theta = mu + sigma * epsilon
- Transform to positive values: gamma = exp(gamma_raw), mu = exp(mu_raw)

Model Parameters (per subject, stored as epsilon):
- r_eps: CRRA risk aversion epsilon
- gamma_eps: probability weighting epsilon
- mu_eps: Fechner noise epsilon

Hyperparameters (6 total):
- Population mean and log-standard deviation for each parameter
- Updated via COUPLED_TRANSFORM for theta-preserving proposals

Data Format (indexed):
- float[0]: unique_outcomes (normalized prizes)
- float[1]: unique_probs (unique probability values)
- int[0]: choices (n_subjects, n_choices)
- int[1-4]: outcome indices for A_high, A_low, B_high, B_low
- int[5]: probability indices
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

from .common_fns import crra_ufn, power_pwf, risk_likelihood

MODEL_ID = "rdu_bhm"

# =============================================================================
# Model Constants
# =============================================================================

NPARAMS = 3  # r_eps, gamma_eps, mu_eps per subject (epsilon values)
N_HYPER = 6  # mean and logsd for each of the 3 parameters
HYPER_BLOCK_SIZE = 2  # (mean, logsd) per hyperparameter block


# =============================================================================
# Hyperprior Specifications
# =============================================================================

# These are derived from prior_selection.py with rho=0.3, alpha_0=6.0
# They define the prior on population-level hyperparameters.
#
# For each parameter j:
#   mu_j ~ N(mu_0[j], tau_0_sq[j])     (population mean)
#   sigma_j^2 ~ IG(alpha_0, beta_0[j]) (population variance)
#
# The variance decomposition ensures: E[sigma^2] + tau_0^2 = V (prior selection variance)

HYPERPRIOR_SPECS = {
    'r': {
        'mu_0': 0.5,              # Prior selection mean for r
        'tau_0_sq': 0.075,        # tau_0^2 = rho * V = 0.3 * 0.25
        'alpha_0': 6.0,           # Concentration parameter
        'beta_0': 0.875,          # beta_0 = (1-rho) * V * (alpha_0-1) = 0.7 * 0.25 * 5
    },
    'gamma_raw': {
        'mu_0': 0.0,              # Prior selection mean for log(gamma)
        'tau_0_sq': 0.027,        # tau_0^2 = 0.3 * 0.09
        'alpha_0': 6.0,
        'beta_0': 0.315,          # beta_0 = 0.7 * 0.09 * 5
    },
    'mu_raw': {
        'mu_0': -2.0,             # Prior selection mean for log(mu)
        'tau_0_sq': 0.3,          # tau_0^2 = 0.3 * 1.0
        'alpha_0': 6.0,
        'beta_0': 3.5,            # beta_0 = 0.7 * 1.0 * 5
    },
}

PARAM_NAMES = ('r', 'gamma_raw', 'mu_raw')


# =============================================================================
# Index Helpers
# =============================================================================

def _get_indices(nids):
    """Compute index offsets for the NCP state vector."""
    first_hyper = nids * NPARAMS

    # Hyperparameter layout: [r_mean, r_logsd, gamma_mean, gamma_logsd, mu_mean, mu_logsd]
    r_hyper_start = first_hyper
    gamma_hyper_start = first_hyper + HYPER_BLOCK_SIZE
    mu_hyper_start = first_hyper + 2 * HYPER_BLOCK_SIZE

    return {
        'first_hyper': first_hyper,
        'r_hyper_start': r_hyper_start,
        'gamma_hyper_start': gamma_hyper_start,
        'mu_hyper_start': mu_hyper_start,
    }


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """
    RDU NCP-BHM block specification.

    Parameters per subject (3 epsilon values):
    - r_eps: standardized CRRA coefficient
    - gamma_eps: standardized log probability weighting
    - mu_eps: standardized log Fechner noise

    Hyperparameters (6 total): mean and logsd for each parameter.
    Hyperparameters use COUPLED_TRANSFORM for theta-preserving updates.
    """
    n_subjects = data["static"][0]

    # Subject epsilon blocks: 3 params per subject using MH
    subject_specs = create_subject_blocks(
        n_subjects=n_subjects,
        params_per_subject=NPARAMS,
        sampler_type=SamplerType.METROPOLIS_HASTINGS,
        proposal_type=ProposalType.MCOV_SMOOTH,
        settings={'cov_mult': 1.0},
        label_prefix="Subj"
    )

    # Hyperparameter blocks: COUPLED_TRANSFORM for theta-preserving updates
    # Each block has 2 params (mean, logsd) for one parameter type
    hyper_specs = [
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.MCOV_SMOOTH,
            settings={'cov_mult': 1.0},
            label="Hyper_r",
        ),
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.MCOV_SMOOTH,
            settings={'cov_mult': 1.0},
            label="Hyper_gamma",
        ),
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.MCOV_SMOOTH,
            settings={'cov_mult': 1.0},
            label="Hyper_mu",
        ),
    ]

    return subject_specs + hyper_specs


# =============================================================================
# Hyperprior Functions
# =============================================================================

def _hyperprior_mean_logsd(mean_val, logsd_val, mu_0, tau_0_sq, alpha_0, beta_0):
    """
    Log hyperprior for a (mean, logsd) pair.

    Prior structure:
        mean ~ Normal(mu_0, tau_0_sq)
        sigma^2 ~ InverseGamma(alpha_0, beta_0)
        logsd = 0.5 * log(sigma^2)

    The Jacobian for logsd -> sigma^2 transform is included.
    """
    # Prior on mean
    lp_mean = stats.norm.logpdf(mean_val, loc=mu_0, scale=jnp.sqrt(tau_0_sq))

    # Prior on sigma^2 (via logsd) - Inverse-Gamma
    sigma_sq = jnp.exp(2 * logsd_val)
    lp_var = (alpha_0 * jnp.log(beta_0) - jax.scipy.special.gammaln(alpha_0)
              - (alpha_0 + 1) * jnp.log(sigma_sq) - beta_0 / sigma_sq)

    # Jacobian for logsd -> sigma^2: d(sigma^2)/d(logsd) = 2 * sigma^2
    lp_jacobian = jnp.log(2.0) + 2 * logsd_val

    return lp_mean + lp_var + lp_jacobian


def _compute_all_hyperpriors(chain_state, idx):
    """Compute sum of all hyperprior terms."""
    lp = 0.0

    # r hyperpriors
    r_mean = chain_state[idx['r_hyper_start']]
    r_logsd = chain_state[idx['r_hyper_start'] + 1]
    spec = HYPERPRIOR_SPECS['r']
    lp += _hyperprior_mean_logsd(
        r_mean, r_logsd,
        spec['mu_0'], spec['tau_0_sq'], spec['alpha_0'], spec['beta_0']
    )

    # gamma hyperpriors
    gamma_mean = chain_state[idx['gamma_hyper_start']]
    gamma_logsd = chain_state[idx['gamma_hyper_start'] + 1]
    spec = HYPERPRIOR_SPECS['gamma_raw']
    lp += _hyperprior_mean_logsd(
        gamma_mean, gamma_logsd,
        spec['mu_0'], spec['tau_0_sq'], spec['alpha_0'], spec['beta_0']
    )

    # mu hyperpriors
    mu_mean = chain_state[idx['mu_hyper_start']]
    mu_logsd = chain_state[idx['mu_hyper_start'] + 1]
    spec = HYPERPRIOR_SPECS['mu_raw']
    lp += _hyperprior_mean_logsd(
        mu_mean, mu_logsd,
        spec['mu_0'], spec['tau_0_sq'], spec['alpha_0'], spec['beta_0']
    )

    return lp


# =============================================================================
# NCP Transform: epsilon -> theta
# =============================================================================

def _transform_eps_to_theta(chain_state, sid, idx):
    """
    Transform epsilon values to actual parameters for a subject.

    theta = mu + sigma * epsilon
    """
    subject_start = sid * NPARAMS

    # Get epsilon values
    r_eps = chain_state[subject_start + 0]
    gamma_eps = chain_state[subject_start + 1]
    mu_eps = chain_state[subject_start + 2]

    # Get hyperparameters
    r_mean = chain_state[idx['r_hyper_start']]
    r_sigma = jnp.exp(chain_state[idx['r_hyper_start'] + 1])

    gamma_mean = chain_state[idx['gamma_hyper_start']]
    gamma_sigma = jnp.exp(chain_state[idx['gamma_hyper_start'] + 1])

    mu_mean = chain_state[idx['mu_hyper_start']]
    mu_sigma = jnp.exp(chain_state[idx['mu_hyper_start'] + 1])

    # Transform: theta = mu + sigma * epsilon
    r_val = r_mean + r_sigma * r_eps
    gamma_raw = gamma_mean + gamma_sigma * gamma_eps
    mu_raw = mu_mean + mu_sigma * mu_eps

    return r_val, gamma_raw, mu_raw


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """
    Log posterior for RDU NCP hierarchical model.

    For subject epsilon blocks:
    - Epsilon prior: N(0, 1) - does NOT depend on hyperparameters!
    - Data likelihood: uses transformed theta values

    For hyperparameter blocks:
    - Hyperprior only (epsilon prior is constant w.r.t. hyperparameters)
    - Data likelihood is computed over all subjects (transformed)

    Note: COUPLED_TRANSFORM blocks don't evaluate this function for acceptance;
    they use the coupled_transform_dispatch. But we still need it for subject blocks.

    Args:
        chain_state: Current state of all parameters
        param_indices: Indices of parameters in this block
        data: Data dictionary with static, int, float arrays
        beta: Inverse temperature for parallel tempering (default 1.0)
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        """
        Log posterior for hyperparameter blocks.

        In NCP with COUPLED_TRANSFORM, this is typically not called
        because the sampler uses coupled_transform_dispatch instead.
        But we include it for completeness.
        """
        # Hyperpriors
        lp_hyper = _compute_all_hyperpriors(chain_state, idx)

        # Data likelihood over all subjects (using transformed theta)
        datai = data["int"]
        dataf = data["float"]
        datai_jax = tuple(jnp.asarray(arr) for arr in datai)

        unique_outcomes = jnp.asarray(dataf[0])
        unique_cumprobs = jnp.asarray(dataf[1])

        def subject_ll_body(sid, acc):
            r_val, gamma_raw, mu_raw = _transform_eps_to_theta(chain_state, sid, idx)

            gamma = jnp.exp(gamma_raw)
            mu = jnp.exp(mu_raw)

            unique_utils = crra_ufn(unique_outcomes, r_val)
            unique_weighted = power_pwf(unique_cumprobs, gamma)

            ll = risk_likelihood(unique_weighted, unique_utils, datai_jax, sid, mu)
            return acc + ll

        total_ll = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)

        return lp_hyper + beta * total_ll

    def compute_subject_posterior():
        """Log posterior for subject epsilon blocks."""
        sid = first_param // NPARAMS
        subject_start = sid * NPARAMS

        # Epsilon prior: N(0, 1) for each epsilon
        r_eps = chain_state[subject_start + 0]
        gamma_eps = chain_state[subject_start + 1]
        mu_eps = chain_state[subject_start + 2]

        lp_eps = (stats.norm.logpdf(r_eps, 0.0, 1.0)
                  + stats.norm.logpdf(gamma_eps, 0.0, 1.0)
                  + stats.norm.logpdf(mu_eps, 0.0, 1.0))

        # Data likelihood using transformed theta
        r_val, gamma_raw, mu_raw = _transform_eps_to_theta(chain_state, sid, idx)

        gamma = jnp.exp(gamma_raw)
        mu = jnp.exp(mu_raw)

        unique_outcomes = jnp.asarray(data["float"][0])
        unique_cumprobs = jnp.asarray(data["float"][1])
        datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])

        unique_utils = crra_ufn(unique_outcomes, r_val)
        unique_weighted = power_pwf(unique_cumprobs, gamma)

        log_like = risk_likelihood(unique_weighted, unique_utils, datai_jax, sid, mu)

        return lp_eps + beta * log_like

    return jax.lax.cond(
        is_hyper,
        compute_hyper_posterior,
        compute_subject_posterior
    )


# =============================================================================
# Coupled Transform Dispatch (for COUPLED_TRANSFORM sampler)
# =============================================================================

def coupled_transform_dispatch(key, chain_state, primary_indices, proposed_primary, data):
    """
    Dispatch function for COUPLED_TRANSFORM blocks.

    Given proposed hyperparameter values, computes:
    - coupled_indices: which epsilon values depend on this hyperparameter
    - new_coupled: transformed epsilon values to keep theta constant
    - log_jacobian: log |det(Jacobian)| = N × (log σ_old - log σ_new)
    - log_prior_ratio: log p(ε') - log p(ε) = -0.5 × (Σε'² - Σε²)

    The likelihood CANCELS because theta = mu + sigma * epsilon is held constant.

    Args:
        key: JAX random key
        chain_state: Full chain state vector
        primary_indices: Indices of the hyperparameter block being updated
        proposed_primary: Proposed values for the hyperparameters (mean, logsd)
        data: Data dict

    Returns:
        (coupled_indices, new_coupled, log_jacobian, log_prior_ratio, new_key)
    """
    nids = data["static"][0]
    idx = _get_indices(nids)
    first_param = primary_indices[0]

    # Identify which hyperparameter block this is
    # 0=r, 1=gamma, 2=mu
    dispatch_idx = jnp.where(
        first_param == idx['r_hyper_start'], 0,
        jnp.where(
            first_param == idx['gamma_hyper_start'], 1,
            jnp.where(
                first_param == idx['mu_hyper_start'], 2,
                3  # default (should never happen)
            )
        )
    )

    def transform_hyper(param_offset):
        """Transform for a single hyperparameter (r, gamma, or mu)."""
        def do_transform(_):
            hyper_start = primary_indices[0]

            # Get old and new hyperparameters
            mu_old = chain_state[hyper_start]
            logsd_old = chain_state[hyper_start + 1]
            sigma_old = jnp.exp(logsd_old)

            mu_new = proposed_primary[0]
            logsd_new = proposed_primary[1]
            sigma_new = jnp.exp(logsd_new)

            # Build coupled indices (all subjects for this parameter)
            subject_offsets = jnp.arange(nids) * NPARAMS
            coupled_indices = subject_offsets + param_offset

            # Get old epsilon values
            old_eps = chain_state[coupled_indices]

            # Compute invariant theta and new epsilon
            theta = mu_old + sigma_old * old_eps
            new_eps = (theta - mu_new) / sigma_new

            # Log Jacobian: n * log(sigma_old / sigma_new)
            log_jacobian = nids * (jnp.log(sigma_old) - jnp.log(sigma_new))

            # Log prior ratio: -0.5 * (||new_eps||^2 - ||old_eps||^2) for N(0,1)
            log_prior_ratio = -0.5 * (jnp.sum(new_eps**2) - jnp.sum(old_eps**2))

            return coupled_indices, new_eps, log_jacobian, log_prior_ratio

        return do_transform

    def do_default(_):
        """Fallback - should never be called."""
        dummy_indices = jnp.zeros(nids, dtype=jnp.int64)
        dummy_values = jnp.zeros(nids, dtype=chain_state.dtype)
        return dummy_indices, dummy_values, 0.0, 0.0

    # Dispatch based on which hyperparameter block
    # param_offset: 0=r_eps, 1=gamma_eps, 2=mu_eps
    coupled_indices, new_coupled, log_jacobian, log_prior_ratio = jax.lax.switch(
        dispatch_idx,
        [
            transform_hyper(0),  # 0: r hyperparameters
            transform_hyper(1),  # 1: gamma hyperparameters
            transform_hyper(2),  # 2: mu hyperparameters
            do_default,          # 3: default (should never happen)
        ],
        None
    )

    return coupled_indices, new_coupled, log_jacobian, log_prior_ratio, key


# =============================================================================
# Generated Quantities
# =============================================================================

def generated_quantities(chain_state, data):
    """
    Generated quantities: transformed parameters on natural scale.

    Returns:
    - r for each subject
    - gamma (power weighting) for each subject
    - mu (noise) for each subject
    - Population means (transformed)
    - Population standard deviations
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    # Subject-level parameters (transform epsilon to theta)
    def get_subject_theta(sid):
        r_val, gamma_raw, mu_raw = _transform_eps_to_theta(chain_state, sid, idx)
        gamma = jnp.exp(gamma_raw)
        mu = jnp.exp(mu_raw)
        return jnp.array([r_val, gamma, mu])

    subject_params = jax.vmap(get_subject_theta)(jnp.arange(n_subjects))
    r_subjects = subject_params[:, 0]
    gamma_subjects = subject_params[:, 1]
    mu_subjects = subject_params[:, 2]

    # Population parameters
    r_pop_mean = chain_state[idx['r_hyper_start']]
    r_pop_sd = jnp.exp(chain_state[idx['r_hyper_start'] + 1])

    gamma_pop_mean = chain_state[idx['gamma_hyper_start']]
    gamma_pop_sd = jnp.exp(chain_state[idx['gamma_hyper_start'] + 1])

    mu_pop_mean = chain_state[idx['mu_hyper_start']]
    mu_pop_sd = jnp.exp(chain_state[idx['mu_hyper_start'] + 1])

    # Transform population means to natural scale for gamma and mu
    gamma_pop_mean_natural = jnp.exp(gamma_pop_mean)
    mu_pop_mean_natural = jnp.exp(mu_pop_mean)

    return jnp.concatenate([
        r_subjects,
        gamma_subjects,
        mu_subjects,
        jnp.array([
            r_pop_mean, r_pop_sd,
            gamma_pop_mean_natural, gamma_pop_sd,
            mu_pop_mean_natural, mu_pop_sd
        ])
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    n_subjects = data["static"][0]
    # r, gamma, mu for each subject + 6 population params
    return 3 * n_subjects + 6


# =============================================================================
# Initial Values
# =============================================================================

def initial_vector(mcmc_config, data):
    """
    Generate initial parameter values for all chains.

    For NCP, subject parameters are epsilon ~ N(0, 1), initialized near 0.
    """
    n_subjects = data["static"][0]
    total_params = n_subjects * NPARAMS + N_HYPER

    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))

    init = np.zeros((num_chains, total_params))

    first_hyper = n_subjects * NPARAMS

    # Subject epsilon values: initialize near 0 (the prior mean for N(0,1))
    # Use small variance to start near prior means
    EPS_INIT_STD = 0.5
    init[:, 0:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # r_eps
    init[:, 1:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # gamma_eps
    init[:, 2:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # mu_eps

    # Hyperparameters: initialize at prior means
    HYPER_INIT_STD = 0.02

    # r hyperparameters
    r_spec = HYPERPRIOR_SPECS['r']
    init[:, first_hyper + 0] = rng.normal(r_spec['mu_0'], HYPER_INIT_STD, num_chains)  # r_mean
    # logsd: mode of IG is beta/(alpha+1), so sigma^2 ~ beta/(alpha+1), logsd ~ 0.5*log(beta/(alpha+1))
    r_sigma_init = np.sqrt(r_spec['beta_0'] / (r_spec['alpha_0'] + 1))
    init[:, first_hyper + 1] = rng.normal(np.log(r_sigma_init), HYPER_INIT_STD, num_chains)

    # gamma hyperparameters
    gamma_spec = HYPERPRIOR_SPECS['gamma_raw']
    init[:, first_hyper + 2] = rng.normal(gamma_spec['mu_0'], HYPER_INIT_STD, num_chains)
    gamma_sigma_init = np.sqrt(gamma_spec['beta_0'] / (gamma_spec['alpha_0'] + 1))
    init[:, first_hyper + 3] = rng.normal(np.log(gamma_sigma_init), HYPER_INIT_STD, num_chains)

    # mu hyperparameters
    mu_spec = HYPERPRIOR_SPECS['mu_raw']
    init[:, first_hyper + 4] = rng.normal(mu_spec['mu_0'], HYPER_INIT_STD, num_chains)
    mu_sigma_init = np.sqrt(mu_spec['beta_0'] / (mu_spec['alpha_0'] + 1))
    init[:, first_hyper + 5] = rng.normal(np.log(mu_sigma_init), HYPER_INIT_STD, num_chains)

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
