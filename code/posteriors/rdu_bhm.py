"""
Rank-Dependent Utility (RDU) Bayesian Hierarchical Model - Non-Centered Parameterization.

This posterior implements an RDU model for lottery choices with hierarchical
structure across subjects using Non-Centered Parameterization (NCP).

Uses Prelec 2-parameter probability weighting function:
    w(p) = exp(-eta * (-ln(p))^phi)

When phi = eta = 1, the PWF is linear (EUT special case).

Model Parameters (per subject, stored as epsilon):
- r_eps: CRRA risk aversion epsilon
- phi_eps: Prelec curvature epsilon (log scale)
- eta_eps: Prelec elevation epsilon (log scale)
- lambda_eps: Fechner noise epsilon (log scale)

Hyperparameters (8 total):
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

from .common_fns import crra_ufn, prelec_pwf, risk_likelihood

MODEL_ID = "rdu_bhm"

# =============================================================================
# Model Constants
# =============================================================================

NPARAMS = 4  # r_eps, phi_eps, eta_eps, lambda_eps per subject (epsilon values)
N_HYPER = 8  # mean and logsd for each of the 4 parameters
HYPER_BLOCK_SIZE = 2  # (mean, logsd) per hyperparameter block


# =============================================================================
# Hyperprior Specifications
# =============================================================================

# These define the prior on population-level hyperparameters.
# For each parameter j:
#   mu_j ~ N(mu_0[j], tau_0_sq[j])     (population mean)
#   sigma_j^2 ~ IG(alpha_0, beta_0[j]) (population variance)

HYPERPRIOR_SPECS = {
    'r': {
        'mu_0': 0.5,              # Prior selection mean for r
        'tau_0_sq': 0.075,        # tau_0^2 = rho * V = 0.3 * 0.25
        'alpha_0': 6.0,           # Concentration parameter
        'beta_0': 0.875,          # beta_0 = (1-rho) * V * (alpha_0-1)
    },
    'phi_raw': {
        # phi (curvature): centered at 0 on log scale => median phi ~ 1
        # phi = 1 is linear, phi < 1 gives inverse-S shape (typical)
        'mu_0': 0.0,              # log(1) = 0
        'tau_0_sq': 0.027,        # tau_0^2 = 0.3 * 0.09
        'alpha_0': 6.0,
        'beta_0': 0.315,          # beta_0 = 0.7 * 0.09 * 5
    },
    'eta_raw': {
        # eta (elevation): centered at 0 on log scale => median eta ~ 1
        # eta = 1 is standard, eta < 1 flattens the curve
        'mu_0': 0.0,              # log(1) = 0
        'tau_0_sq': 0.027,        # tau_0^2 = 0.3 * 0.09
        'alpha_0': 6.0,
        'beta_0': 0.315,          # beta_0 = 0.7 * 0.09 * 5
    },
    'lambda_raw': {
        'mu_0': -2.0,             # Prior selection mean for log(lambda)
        'tau_0_sq': 0.3,          # tau_0^2 = 0.3 * 1.0
        'alpha_0': 6.0,
        'beta_0': 3.5,            # beta_0 = 0.7 * 1.0 * 5
    },
}

PARAM_NAMES = ('r', 'phi_raw', 'eta_raw', 'lambda_raw')


def load_hyperpriors_from_file(filepath):
    """
    Load hyperprior specifications from a prior_params.npz file.

    Updates the global HYPERPRIOR_SPECS with values from the file.
    This should be called BEFORE running sampling to use custom priors.

    Args:
        filepath: Path to prior_params.npz file

    Returns:
        dict: The loaded hyperprior specs
    """
    global HYPERPRIOR_SPECS

    data = np.load(filepath)

    # Build new hyperprior specs from file
    new_specs = {}

    # r
    new_specs['r'] = {
        'mu_0': float(data['r_mean']),
        'tau_0_sq': float(np.exp(data['ln_r_sdev']) ** 2),
        'alpha_0': float(data['alpha_0']),
        'beta_0': float(data['beta_0_r']),
    }

    # phi_raw
    new_specs['phi_raw'] = {
        'mu_0': float(data['phi_mean']),
        'tau_0_sq': float(np.exp(data['ln_phi_sdev']) ** 2),
        'alpha_0': float(data['alpha_0']),
        'beta_0': float(data['beta_0_phi']),
    }

    # eta_raw
    new_specs['eta_raw'] = {
        'mu_0': float(data['eta_mean']),
        'tau_0_sq': float(np.exp(data['ln_eta_sdev']) ** 2),
        'alpha_0': float(data['alpha_0']),
        'beta_0': float(data['beta_0_eta']),
    }

    # lambda_raw
    new_specs['lambda_raw'] = {
        'mu_0': float(data['lambda_mean']),
        'tau_0_sq': float(np.exp(data['ln_lambda_sdev']) ** 2),
        'alpha_0': float(data['alpha_0']),
        'beta_0': float(data['beta_0_lambda']),
    }

    HYPERPRIOR_SPECS.update(new_specs)
    print(f"Loaded hyperpriors from {filepath}")
    print(f"  r: mu_0={new_specs['r']['mu_0']:.4f}, tau_0={np.sqrt(new_specs['r']['tau_0_sq']):.4f}")
    print(f"  phi_raw: mu_0={new_specs['phi_raw']['mu_0']:.4f}")
    print(f"  eta_raw: mu_0={new_specs['eta_raw']['mu_0']:.4f}")
    print(f"  lambda_raw: mu_0={new_specs['lambda_raw']['mu_0']:.4f}")

    return new_specs


# =============================================================================
# Index Helpers
# =============================================================================

def _get_indices(nids):
    """Compute index offsets for the NCP state vector."""
    first_hyper = nids * NPARAMS

    # Hyperparameter layout: [r_mean, r_logsd, phi_mean, phi_logsd, eta_mean, eta_logsd, lambda_mean, lambda_logsd]
    r_hyper_start = first_hyper
    phi_hyper_start = first_hyper + HYPER_BLOCK_SIZE
    eta_hyper_start = first_hyper + 2 * HYPER_BLOCK_SIZE
    lambda_hyper_start = first_hyper + 3 * HYPER_BLOCK_SIZE

    return {
        'first_hyper': first_hyper,
        'r_hyper_start': r_hyper_start,
        'phi_hyper_start': phi_hyper_start,
        'eta_hyper_start': eta_hyper_start,
        'lambda_hyper_start': lambda_hyper_start,
    }


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """
    RDU NCP-BHM block specification with Prelec PWF.

    Parameters per subject (4 epsilon values):
    - r_eps: standardized CRRA coefficient
    - phi_eps: standardized log Prelec curvature
    - eta_eps: standardized log Prelec elevation
    - lambda_eps: standardized log Fechner noise

    Hyperparameters (8 total): mean and logsd for each parameter.
    """
    n_subjects = data["static"][0]

    # Subject epsilon blocks: 4 params per subject using MH
    subject_specs = create_subject_blocks(
        n_subjects=n_subjects,
        params_per_subject=NPARAMS,
        sampler_type=SamplerType.METROPOLIS_HASTINGS,
        proposal_type=ProposalType.MCOV_SMOOTH,
        settings={'cov_mult': 1.0},
        label_prefix="Subj"
    )

    # Hyperparameter blocks: COUPLED_TRANSFORM for theta-preserving updates
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
            label="Hyper_phi",
        ),
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.MCOV_SMOOTH,
            settings={'cov_mult': 1.0},
            label="Hyper_eta",
        ),
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.MCOV_SMOOTH,
            settings={'cov_mult': 1.0},
            label="Hyper_lambda",
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

    for param_name, hyper_key in [('r', 'r_hyper_start'),
                                   ('phi_raw', 'phi_hyper_start'),
                                   ('eta_raw', 'eta_hyper_start'),
                                   ('lambda_raw', 'lambda_hyper_start')]:
        mean_val = chain_state[idx[hyper_key]]
        logsd_val = chain_state[idx[hyper_key] + 1]
        spec = HYPERPRIOR_SPECS[param_name]
        lp += _hyperprior_mean_logsd(
            mean_val, logsd_val,
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
    phi_eps = chain_state[subject_start + 1]
    eta_eps = chain_state[subject_start + 2]
    lambda_eps = chain_state[subject_start + 3]

    # Get hyperparameters
    r_mean = chain_state[idx['r_hyper_start']]
    r_sigma = jnp.exp(chain_state[idx['r_hyper_start'] + 1])

    phi_mean = chain_state[idx['phi_hyper_start']]
    phi_sigma = jnp.exp(chain_state[idx['phi_hyper_start'] + 1])

    eta_mean = chain_state[idx['eta_hyper_start']]
    eta_sigma = jnp.exp(chain_state[idx['eta_hyper_start'] + 1])

    lambda_mean = chain_state[idx['lambda_hyper_start']]
    lambda_sigma = jnp.exp(chain_state[idx['lambda_hyper_start'] + 1])

    # Transform: theta = mu + sigma * epsilon
    r_val = r_mean + r_sigma * r_eps
    phi_raw = phi_mean + phi_sigma * phi_eps
    eta_raw = eta_mean + eta_sigma * eta_eps
    lambda_raw = lambda_mean + lambda_sigma * lambda_eps

    return r_val, phi_raw, eta_raw, lambda_raw


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """
    Log posterior for RDU NCP hierarchical model with Prelec PWF.
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        """Log posterior for hyperparameter blocks."""
        lp_hyper = _compute_all_hyperpriors(chain_state, idx)

        datai = data["int"]
        dataf = data["float"]
        datai_jax = tuple(jnp.asarray(arr) for arr in datai)

        unique_outcomes = jnp.asarray(dataf[0])
        unique_cumprobs = jnp.asarray(dataf[1])

        def subject_ll_body(sid, acc):
            r_val, phi_raw, eta_raw, lambda_raw = _transform_eps_to_theta(chain_state, sid, idx)

            phi = jnp.exp(phi_raw)
            eta = jnp.exp(eta_raw)
            lam = jnp.exp(lambda_raw)

            unique_utils = crra_ufn(unique_outcomes, r_val)
            unique_weighted = prelec_pwf(unique_cumprobs, phi, eta)

            ll = risk_likelihood(unique_weighted, unique_utils, datai_jax, sid, lam)
            return acc + ll

        total_ll = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)

        return lp_hyper + beta * total_ll

    def compute_subject_posterior():
        """Log posterior for subject epsilon blocks."""
        sid = first_param // NPARAMS
        subject_start = sid * NPARAMS

        # Epsilon prior: N(0, 1) for each epsilon
        r_eps = chain_state[subject_start + 0]
        phi_eps = chain_state[subject_start + 1]
        eta_eps = chain_state[subject_start + 2]
        lambda_eps = chain_state[subject_start + 3]

        lp_eps = (stats.norm.logpdf(r_eps, 0.0, 1.0)
                  + stats.norm.logpdf(phi_eps, 0.0, 1.0)
                  + stats.norm.logpdf(eta_eps, 0.0, 1.0)
                  + stats.norm.logpdf(lambda_eps, 0.0, 1.0))

        # Data likelihood using transformed theta
        r_val, phi_raw, eta_raw, lambda_raw = _transform_eps_to_theta(chain_state, sid, idx)

        phi = jnp.exp(phi_raw)
        eta = jnp.exp(eta_raw)
        lam = jnp.exp(lambda_raw)

        unique_outcomes = jnp.asarray(data["float"][0])
        unique_cumprobs = jnp.asarray(data["float"][1])
        datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])

        unique_utils = crra_ufn(unique_outcomes, r_val)
        unique_weighted = prelec_pwf(unique_cumprobs, phi, eta)

        log_like = risk_likelihood(unique_weighted, unique_utils, datai_jax, sid, lam)

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
    """
    nids = data["static"][0]
    idx = _get_indices(nids)
    first_param = primary_indices[0]

    # Identify which hyperparameter block this is: 0=r, 1=phi, 2=eta, 3=mu
    dispatch_idx = jnp.where(
        first_param == idx['r_hyper_start'], 0,
        jnp.where(
            first_param == idx['phi_hyper_start'], 1,
            jnp.where(
                first_param == idx['eta_hyper_start'], 2,
                jnp.where(
                    first_param == idx['lambda_hyper_start'], 3,
                    4  # default (should never happen)
                )
            )
        )
    )

    def transform_hyper(param_offset):
        """Transform for a single hyperparameter."""
        def do_transform(_):
            hyper_start = primary_indices[0]

            mu_old = chain_state[hyper_start]
            logsd_old = chain_state[hyper_start + 1]
            sigma_old = jnp.exp(logsd_old)

            mu_new = proposed_primary[0]
            logsd_new = proposed_primary[1]
            sigma_new = jnp.exp(logsd_new)

            subject_offsets = jnp.arange(nids) * NPARAMS
            coupled_indices = subject_offsets + param_offset

            old_eps = chain_state[coupled_indices]
            theta = mu_old + sigma_old * old_eps
            new_eps = (theta - mu_new) / sigma_new

            log_jacobian = nids * (jnp.log(sigma_old) - jnp.log(sigma_new))
            log_prior_ratio = -0.5 * (jnp.sum(new_eps**2) - jnp.sum(old_eps**2))

            return coupled_indices, new_eps, log_jacobian, log_prior_ratio

        return do_transform

    def do_default(_):
        dummy_indices = jnp.zeros(nids, dtype=jnp.int64)
        dummy_values = jnp.zeros(nids, dtype=chain_state.dtype)
        return dummy_indices, dummy_values, 0.0, 0.0

    coupled_indices, new_coupled, log_jacobian, log_prior_ratio = jax.lax.switch(
        dispatch_idx,
        [
            transform_hyper(0),  # 0: r hyperparameters
            transform_hyper(1),  # 1: phi hyperparameters
            transform_hyper(2),  # 2: eta hyperparameters
            transform_hyper(3),  # 3: lambda hyperparameters
            do_default,          # 4: default (should never happen)
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
    - phi (Prelec curvature) for each subject
    - eta (Prelec elevation) for each subject
    - lambda (noise) for each subject
    - Population means and standard deviations
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    def get_subject_theta(sid):
        r_val, phi_raw, eta_raw, lambda_raw = _transform_eps_to_theta(chain_state, sid, idx)
        phi = jnp.exp(phi_raw)
        eta = jnp.exp(eta_raw)
        lam = jnp.exp(lambda_raw)
        return jnp.array([r_val, phi, eta, lam])

    subject_params = jax.vmap(get_subject_theta)(jnp.arange(n_subjects))
    r_subjects = subject_params[:, 0]
    phi_subjects = subject_params[:, 1]
    eta_subjects = subject_params[:, 2]
    lambda_subjects = subject_params[:, 3]

    # Population parameters
    r_pop_mean = chain_state[idx['r_hyper_start']]
    r_pop_sd = jnp.exp(chain_state[idx['r_hyper_start'] + 1])

    phi_pop_mean = chain_state[idx['phi_hyper_start']]
    phi_pop_sd = jnp.exp(chain_state[idx['phi_hyper_start'] + 1])

    eta_pop_mean = chain_state[idx['eta_hyper_start']]
    eta_pop_sd = jnp.exp(chain_state[idx['eta_hyper_start'] + 1])

    lambda_pop_mean = chain_state[idx['lambda_hyper_start']]
    lambda_pop_sd = jnp.exp(chain_state[idx['lambda_hyper_start'] + 1])

    # Transform population means to natural scale
    phi_pop_mean_natural = jnp.exp(phi_pop_mean)
    eta_pop_mean_natural = jnp.exp(eta_pop_mean)
    lambda_pop_mean_natural = jnp.exp(lambda_pop_mean)

    return jnp.concatenate([
        r_subjects,
        phi_subjects,
        eta_subjects,
        lambda_subjects,
        jnp.array([
            r_pop_mean, r_pop_sd,
            phi_pop_mean_natural, phi_pop_sd,
            eta_pop_mean_natural, eta_pop_sd,
            lambda_pop_mean_natural, lambda_pop_sd
        ])
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    n_subjects = data["static"][0]
    # r, phi, eta, lambda for each subject + 8 population params
    return 4 * n_subjects + 8


# =============================================================================
# Initial Values
# =============================================================================

def initial_vector(mcmc_config, data):
    """
    Generate initial parameter values for all chains.
    """
    n_subjects = data["static"][0]
    total_params = n_subjects * NPARAMS + N_HYPER

    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))

    init = np.zeros((num_chains, total_params))

    first_hyper = n_subjects * NPARAMS

    # Subject epsilon values: initialize near 0 (the prior mean for N(0,1))
    EPS_INIT_STD = 0.5
    init[:, 0:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # r_eps
    init[:, 1:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # phi_eps
    init[:, 2:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # eta_eps
    init[:, 3:first_hyper:NPARAMS] = rng.normal(0.0, EPS_INIT_STD, (num_chains, n_subjects))  # lambda_eps

    # Hyperparameters: initialize at prior means
    HYPER_INIT_STD = 0.02

    # r hyperparameters
    r_spec = HYPERPRIOR_SPECS['r']
    init[:, first_hyper + 0] = rng.normal(r_spec['mu_0'], HYPER_INIT_STD, num_chains)
    r_sigma_init = np.sqrt(r_spec['beta_0'] / (r_spec['alpha_0'] + 1))
    init[:, first_hyper + 1] = rng.normal(np.log(r_sigma_init), HYPER_INIT_STD, num_chains)

    # phi hyperparameters
    phi_spec = HYPERPRIOR_SPECS['phi_raw']
    init[:, first_hyper + 2] = rng.normal(phi_spec['mu_0'], HYPER_INIT_STD, num_chains)
    phi_sigma_init = np.sqrt(phi_spec['beta_0'] / (phi_spec['alpha_0'] + 1))
    init[:, first_hyper + 3] = rng.normal(np.log(phi_sigma_init), HYPER_INIT_STD, num_chains)

    # eta hyperparameters
    eta_spec = HYPERPRIOR_SPECS['eta_raw']
    init[:, first_hyper + 4] = rng.normal(eta_spec['mu_0'], HYPER_INIT_STD, num_chains)
    eta_sigma_init = np.sqrt(eta_spec['beta_0'] / (eta_spec['alpha_0'] + 1))
    init[:, first_hyper + 5] = rng.normal(np.log(eta_sigma_init), HYPER_INIT_STD, num_chains)

    # lambda hyperparameters
    lambda_spec = HYPERPRIOR_SPECS['lambda_raw']
    init[:, first_hyper + 6] = rng.normal(lambda_spec['mu_0'], HYPER_INIT_STD, num_chains)
    lambda_sigma_init = np.sqrt(lambda_spec['beta_0'] / (lambda_spec['alpha_0'] + 1))
    init[:, first_hyper + 7] = rng.normal(np.log(lambda_sigma_init), HYPER_INIT_STD, num_chains)

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
