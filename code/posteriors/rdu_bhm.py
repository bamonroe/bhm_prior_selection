"""
Rank-Dependent Utility (RDU) Bayesian Hierarchical Model.

This posterior implements an RDU model for lottery choices with
hierarchical structure across subjects. Uses power probability weighting.

Model Parameters (per subject):
- r: CRRA risk aversion coefficient
- gamma_raw: log of probability weighting parameter (gamma = exp(gamma_raw))
- mu_raw: log of Fechner noise parameter (mu = exp(mu_raw))

Hierarchical Structure:
- Each parameter has population mean and log-standard deviation
- Subject parameters drawn from population distribution

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
    from bamcmc import register_posterior, BlockSpec, SamplerType, ProposalType
    BAMCMC_AVAILABLE = True
except ImportError:
    BAMCMC_AVAILABLE = False
    print("Warning: bamcmc not available. Posterior registration skipped.")

from .common_fns import crra_ufn, power_pwf, risk_likelihood

MODEL_ID = "rdu_bhm"

# =============================================================================
# Model Constants
# =============================================================================

NPARAMS = 3  # r, gamma_raw, mu_raw per subject
N_HYPER = 6  # mean and logsd for each of the 3 parameters

# =============================================================================
# Prior Configuration
# =============================================================================

# Set to True to use uninformative priors (mean=0, sd=100) for hyperparameters
# This is for comparison with the principled prior selection approach
USE_UNINFORMATIVE_PRIOR = False


def set_prior_mode(uninformative: bool = False):
    """
    Set the prior mode for the model.

    Parameters:
    -----------
    uninformative : bool
        If True, use uninformative priors (mean=0, sd=100) for all hyperparameters.
        If False (default), use the principled prior selection approach.

    Note: Must be called before registering the posterior or before sampling.
    """
    global USE_UNINFORMATIVE_PRIOR
    USE_UNINFORMATIVE_PRIOR = uninformative
    mode_str = "uninformative (N(0, 100^2))" if uninformative else "principled prior selection"
    print(f"RDU-BHM prior mode set to: {mode_str}")


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """
    RDU with power PWF - Bayesian Hierarchical Model block specification.

    Parameters per subject (3):
    - r: CRRA utility curvature
    - gamma_raw: log of power weighting param (gamma = exp(gamma_raw))
    - mu_raw: log noise parameter (mu = exp(mu_raw))

    Hyperparameters (6 total): mean and logsd for each subject parameter
    """
    n_subjects = data["static"][0]

    # Subject parameters: 3 params per subject
    subject_specs = [
        BlockSpec(
            size=NPARAMS,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.MCOV_MODE,
            settings={'cov_mult': 1.0},
            label=f"Subj_{i}"
        )
        for i in range(n_subjects)
    ]

    # Hyperparameters: 6 total (mean + logsd for r, gamma, mu)
    hyper_spec = BlockSpec(
        size=N_HYPER,
        sampler_type=SamplerType.METROPOLIS_HASTINGS,
        proposal_type=ProposalType.MCOV_MODE,
        settings={'cov_mult': 1.0},
        label="Hyperparameters"
    )

    return subject_specs + [hyper_spec]


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """
    Log posterior for RDU with power PWF hierarchical model.

    This function handles both subject blocks and hyperparameter blocks,
    dispatching based on the first index.

    Args:
        chain_state: Current state of all parameters
        param_indices: Indices of parameters in this block
        data: Data dictionary with static, int, float arrays
        beta: Inverse temperature for parallel tempering (default 1.0)
              Only affects likelihood, not priors
    """
    n_subjects = data["static"][0]
    first_hyper = n_subjects * NPARAMS

    # Determine if this is a subject block or hyperparameter block
    first_idx = param_indices[0]

    return jax.lax.cond(
        first_idx < first_hyper,
        lambda _: _subject_log_posterior(chain_state, param_indices, data, beta),
        lambda _: _hyper_log_posterior(chain_state, param_indices, data),
        operand=None
    )


def _subject_log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log posterior for a single subject's parameters.

    Args:
        beta: Inverse temperature for parallel tempering. Applied to likelihood only.
    """
    n_subjects = data["static"][0]
    first_hyper = n_subjects * NPARAMS

    # Subject ID
    sid = param_indices[0] // NPARAMS

    # Extract hyperparameters
    r_mean = chain_state[first_hyper + 0]
    ln_r_sdev = chain_state[first_hyper + 1]
    gamma_mean = chain_state[first_hyper + 2]
    ln_gamma_sdev = chain_state[first_hyper + 3]
    mu_mean = chain_state[first_hyper + 4]
    ln_mu_sdev = chain_state[first_hyper + 5]

    # Extract subject parameters
    r_val = chain_state[param_indices[0]]
    gamma_raw = chain_state[param_indices[1]]
    mu_raw = chain_state[param_indices[2]]

    # Prior: hierarchical normal
    lp = 0.0
    lp += stats.norm.logpdf(r_val, loc=r_mean, scale=jnp.exp(ln_r_sdev))
    lp += stats.norm.logpdf(gamma_raw, loc=gamma_mean, scale=jnp.exp(ln_gamma_sdev))
    lp += stats.norm.logpdf(mu_raw, loc=mu_mean, scale=jnp.exp(ln_mu_sdev))

    # Transform to positive values
    gamma = jnp.exp(gamma_raw)
    mu = jnp.exp(mu_raw)

    # Get lookup tables from data
    unique_outcomes = jnp.asarray(data["float"][0])
    unique_cumprobs = jnp.asarray(data["float"][1])

    # Precompute utilities for all unique outcomes
    unique_utils = crra_ufn(unique_outcomes, r_val)

    # Precompute weighted cumulative probs for all unique probabilities
    unique_weighted = power_pwf(unique_cumprobs, gamma)

    # Convert int arrays to JAX arrays for dynamic slicing
    datai = tuple(jnp.asarray(arr) for arr in data["int"])

    # Compute likelihood using indexed lookups
    log_like = risk_likelihood(unique_weighted, unique_utils, datai, sid, mu)

    # Apply beta tempering to likelihood only (not prior)
    return lp + beta * log_like


def _hyper_log_posterior(chain_state, param_indices, data):
    """
    Log posterior for hyperparameter block.

    If USE_UNINFORMATIVE_PRIOR is False (default):
        Hyperpriors derived from prior selection methodology (see prior_selection.py):
        - Prior selection defines what a single random individual's preferences look like
        - Variance decomposition: Var[theta] = E[sigma^2] + tau_0^2
        - With rho=0.3 (30% mean uncertainty, 70% heterogeneity) and alpha_0=6

        Derived values:
        - r: mu_0=0.5, tau_0=0.274, E[sigma]=0.42
        - gamma_raw: mu_0=0.0, tau_0=0.164, E[sigma]=0.25
        - mu_raw: mu_0=-2.0, tau_0=0.548, E[sigma]=0.84

    If USE_UNINFORMATIVE_PRIOR is True:
        All hyperparameters get N(0, 100^2) priors for comparison purposes.
    """
    n_subjects = data["static"][0]
    first_hyper = n_subjects * NPARAMS

    # Extract hyperparameters
    r_mean = chain_state[first_hyper + 0]
    ln_r_sdev = chain_state[first_hyper + 1]
    gamma_mean = chain_state[first_hyper + 2]
    ln_gamma_sdev = chain_state[first_hyper + 3]
    mu_mean = chain_state[first_hyper + 4]
    ln_mu_sdev = chain_state[first_hyper + 5]

    lp = 0.0

    if USE_UNINFORMATIVE_PRIOR:
        # Uninformative priors: N(0, 100^2) for all hyperparameters
        lp += stats.norm.logpdf(r_mean, loc=0.0, scale=100.0)
        lp += stats.norm.logpdf(ln_r_sdev, loc=0.0, scale=100.0)
        lp += stats.norm.logpdf(gamma_mean, loc=0.0, scale=100.0)
        lp += stats.norm.logpdf(ln_gamma_sdev, loc=0.0, scale=100.0)
        lp += stats.norm.logpdf(mu_mean, loc=0.0, scale=100.0)
        lp += stats.norm.logpdf(ln_mu_sdev, loc=0.0, scale=100.0)
    else:
        # Hyperpriors derived from prior selection (see prior_selection.py)
        # Using variance decomposition with rho=0.3, alpha_0=6

        # r_mean ~ N(0.5, 0.274^2) - from prior selection
        # Using scale=0.5 to be moderately informative
        lp += stats.norm.logpdf(r_mean, loc=0.5, scale=0.5)

        # ln_r_sdev: E[sigma_r] = 0.42, ln(0.42) = -0.87
        # Prior centered on expected log-sdev with moderate uncertainty
        lp += stats.norm.logpdf(ln_r_sdev, loc=-0.87, scale=0.5)

        # gamma_mean ~ N(0, 0.164^2) - from prior selection
        lp += stats.norm.logpdf(gamma_mean, loc=0.0, scale=0.5)

        # ln_gamma_sdev: E[sigma_gamma] = 0.25, ln(0.25) = -1.39
        lp += stats.norm.logpdf(ln_gamma_sdev, loc=-1.39, scale=0.5)

        # mu_mean ~ N(-2, 0.548^2) - from prior selection
        lp += stats.norm.logpdf(mu_mean, loc=-2.0, scale=0.5)

        # ln_mu_sdev: E[sigma_mu] = 0.84, ln(0.84) = -0.18
        lp += stats.norm.logpdf(ln_mu_sdev, loc=-0.18, scale=0.5)

    return lp


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
    - Population means
    """
    n_subjects = data["static"][0]
    first_hyper = n_subjects * NPARAMS

    # Subject-level parameters
    r_subjects = chain_state[0:first_hyper:NPARAMS]
    gamma_raw_subjects = chain_state[1:first_hyper:NPARAMS]
    mu_raw_subjects = chain_state[2:first_hyper:NPARAMS]

    # Transform to natural scale
    gamma_subjects = jnp.exp(gamma_raw_subjects)
    mu_subjects = jnp.exp(mu_raw_subjects)

    # Population means (transformed)
    r_pop_mean = chain_state[first_hyper + 0]
    gamma_pop_mean = jnp.exp(chain_state[first_hyper + 2])
    mu_pop_mean = jnp.exp(chain_state[first_hyper + 4])

    return jnp.concatenate([
        r_subjects,
        gamma_subjects,
        mu_subjects,
        jnp.array([r_pop_mean, gamma_pop_mean, mu_pop_mean])
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    n_subjects = data["static"][0]
    # r, gamma, mu for each subject + 3 population means
    return 3 * n_subjects + 3


# =============================================================================
# Initial Values
# =============================================================================

def initial_vector(mcmc_config, data):
    """Generate initial parameter values for all chains.

    If USE_UNINFORMATIVE_PRIOR is True, uses more dispersed initial values
    since the priors provide less guidance.
    """
    n_subjects = data["static"][0]
    total_params = n_subjects * NPARAMS + N_HYPER

    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))

    init = np.zeros((num_chains, total_params))

    first_hyper = n_subjects * NPARAMS

    if USE_UNINFORMATIVE_PRIOR:
        # With uninformative priors, use wider initialization
        # but still reasonable starting points for numerical stability
        init[:, 0:first_hyper:NPARAMS] = rng.normal(0.0, 1.0, (num_chains, n_subjects))   # r
        init[:, 1:first_hyper:NPARAMS] = rng.normal(0.0, 1.0, (num_chains, n_subjects))   # gamma_raw
        init[:, 2:first_hyper:NPARAMS] = rng.normal(0.0, 1.0, (num_chains, n_subjects))   # mu_raw

        # Hyperparameters: start near 0 with wider spread
        init[:, first_hyper + 0] = rng.normal(0.0, 1.0, num_chains)    # r_mean
        init[:, first_hyper + 1] = rng.normal(0.0, 1.0, num_chains)    # ln_r_sdev
        init[:, first_hyper + 2] = rng.normal(0.0, 1.0, num_chains)    # gamma_mean
        init[:, first_hyper + 3] = rng.normal(0.0, 1.0, num_chains)    # ln_gamma_sdev
        init[:, first_hyper + 4] = rng.normal(0.0, 1.0, num_chains)    # mu_mean
        init[:, first_hyper + 5] = rng.normal(0.0, 1.0, num_chains)    # ln_mu_sdev
    else:
        # Subject parameters
        init[:, 0:first_hyper:NPARAMS] = rng.normal(0.5, 0.3, (num_chains, n_subjects))      # r
        init[:, 1:first_hyper:NPARAMS] = rng.normal(0.0, 0.2, (num_chains, n_subjects))      # gamma_raw (log)
        init[:, 2:first_hyper:NPARAMS] = rng.normal(-2.0, 0.2, (num_chains, n_subjects))     # mu_raw (log)

        # Hyperparameters (values derived from prior selection, see prior_selection.py)
        init[:, first_hyper + 0] = rng.normal(0.5, 0.1, num_chains)    # r_mean
        init[:, first_hyper + 1] = rng.normal(-0.87, 0.1, num_chains)  # ln_r_sdev (E[sigma_r]=0.42)
        init[:, first_hyper + 2] = rng.normal(0.0, 0.1, num_chains)    # gamma_mean
        init[:, first_hyper + 3] = rng.normal(-1.39, 0.1, num_chains)  # ln_gamma_sdev (E[sigma_gamma]=0.25)
        init[:, first_hyper + 4] = rng.normal(-2.0, 0.1, num_chains)   # mu_mean
        init[:, first_hyper + 5] = rng.normal(-0.18, 0.1, num_chains)  # ln_mu_sdev (E[sigma_mu]=0.84)

    return init.flatten()


# =============================================================================
# Direct Sampler (Placeholder)
# =============================================================================

def direct_sampler(key, chain_state, param_indices, data):
    """Placeholder for MH-only models."""
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
    })
    print(f"Registered posterior: {MODEL_ID}")


# Register on import if bamcmc is available
if BAMCMC_AVAILABLE:
    register()
