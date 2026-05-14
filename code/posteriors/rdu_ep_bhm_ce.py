"""
RDU BHM with Expo-Power Utility + TK92 PWF, Quantity-Space Prior (Beta-Beta).

This posterior implements an RDU model where the prior on individual-level
parameters (alpha_raw, r, gamma_raw, lambda_raw) is specified in quantity space
(3 CEs + 1 choice probability) using a Beta-Beta hierarchical model with
Jacobian correction:

    p(theta) = p_Y(g(theta)) |det J_g(theta)|

where g maps (alpha_raw, r, gamma_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).

Utility: Expo-Power (Saha 1993) with 2 parameters (alpha, r)
PWF: Tversky-Kahneman (1992) inverse-S with 1 parameter (gamma)
Noise: Fechner (lambda)

Hyperparameters (8 total, on logit scale):
    [logit_mu_1, logit_psi_1,   # CE1 population mean & dispersion
     logit_mu_2, logit_psi_2,   # CE2
     logit_mu_3, logit_psi_3,   # CE3
     logit_mu_P, logit_psi_P]   # P_choice

Data format: Same as rdu_bhm_ce:
- float[7]: hyperprior_arr (24 values)
- float[8]: ref_lotteries (18 values)
- float[9]: ref_pair (12 values)
"""

import numpy as np
import jax
import jax.numpy as jnp

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
    ep_ufn,
    risk_likelihood_ep,
    full_forward_map_ep,
    full_jacobian_logabsdet_ep,
)

# Import shared Beta-Beta infrastructure from the CRRA model
from .rdu_bhm_ce import (
    N_QUANTITIES,
    HYPER_BLOCK_SIZE,
    HYPERPRIOR_STRIDE,
    QUANTITY_NAMES,
    HYPERPRIOR_SPECS,
    _log_beta_pdf_normalized,
    _hyperprior_logit_beta,
    _compute_all_hyperpriors,
    _subject_quantity_prior,
    load_hyperpriors_from_file,
    get_hyperprior_array,
)

MODEL_ID = "rdu_ep_bhm_ce"

# =============================================================================
# Model Constants
# =============================================================================

NPARAMS = 4          # alpha_raw, r, gamma_raw, lambda_raw per subject
N_HYPER = 8          # (logit_mu, logit_psi) for 4 quantities


# =============================================================================
# Index Helpers
# =============================================================================

def _get_indices(nids):
    """Compute index offsets for the state vector."""
    first_hyper = nids * NPARAMS
    hyper_starts = [first_hyper + k * HYPER_BLOCK_SIZE for k in range(N_QUANTITIES)]
    return {
        'first_hyper': first_hyper,
        'hyper_starts': hyper_starts,
    }


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """
    Beta-Beta BHM block specification for EP+TK model.

    Subject blocks: (alpha_raw, r, gamma_raw, lambda_raw) per subject — MH.
    Hyper blocks (x4): (logit_mu_k, logit_psi_k) per quantity — MH.
    """
    n_subjects = data["static"][0]

    proposal_type = ProposalType.MCOV_SMOOTH
    settings = {'k_alpha': 3.0, 'k_g': 12}

    subject_specs = create_subject_blocks(
        n_subjects=n_subjects,
        params_per_subject=NPARAMS,
        sampler_type=SamplerType.METROPOLIS_HASTINGS,
        proposal_type=proposal_type,
        settings=settings,
        label_prefix="Subj"
    )

    hyper_labels = ["Hyper_CE1", "Hyper_CE2", "Hyper_CE3", "Hyper_Pchoice"]
    hyper_specs = [
        BlockSpec(
            size=HYPER_BLOCK_SIZE,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=proposal_type,
            settings=settings,
            label=label,
        )
        for label in hyper_labels
    ]

    return subject_specs + hyper_specs


# =============================================================================
# Subject Parameter Extraction
# =============================================================================

def _get_subject_params(chain_state, sid):
    """Extract EP+TK parameters for a subject (all centered)."""
    subject_start = sid * NPARAMS
    alpha_raw = chain_state[subject_start + 0]
    r_val = chain_state[subject_start + 1]
    gamma_raw = chain_state[subject_start + 2]
    lambda_raw = chain_state[subject_start + 3]
    return alpha_raw, r_val, gamma_raw, lambda_raw


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log posterior for RDU EP+TK quantity-space Beta-Beta hierarchical model."""
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        """Log posterior for hyperparameter blocks."""
        dataf = data["float"]

        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        lp_hyper = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        def subject_ll_body(sid, acc):
            alpha_raw, r_val, gamma_raw, lambda_raw = _get_subject_params(
                chain_state, sid)

            quantities = full_forward_map_ep(
                alpha_raw, r_val, gamma_raw, lambda_raw,
                ref_lotteries, ref_pair)

            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)

            return acc + lp_pop

        total = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)

        return lp_hyper + total

    def compute_subject_posterior():
        """Log posterior for a single subject block."""
        sid = first_param // NPARAMS
        alpha_raw, r_val, gamma_raw, lambda_raw = _get_subject_params(
            chain_state, sid)

        dataf = data["float"]
        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        # Forward map
        quantities = full_forward_map_ep(
            alpha_raw, r_val, gamma_raw, lambda_raw,
            ref_lotteries, ref_pair)

        # Beta-Beta population prior
        lp_pop = _subject_quantity_prior(
            quantities, chain_state, idx, hyperprior_arr)

        # Jacobian correction (autodiff)
        log_det_j = full_jacobian_logabsdet_ep(
            alpha_raw, r_val, gamma_raw, lambda_raw,
            ref_lotteries, ref_pair)

        # Data likelihood
        alpha = jnp.exp(alpha_raw)
        gamma = jnp.exp(gamma_raw)
        lam = jnp.exp(lambda_raw)

        datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
        dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
        unique_outcomes = dataf_jax[0]
        unique_utils = ep_ufn(unique_outcomes, alpha, r_val)
        log_like = risk_likelihood_ep(
            unique_utils, gamma, datai_jax, dataf_jax, sid, lam)

        return lp_pop + log_det_j + beta * log_like

    return jax.lax.cond(
        is_hyper,
        compute_hyper_posterior,
        compute_subject_posterior
    )


# =============================================================================
# Generated Quantities
# =============================================================================

def generated_quantities(chain_state, data):
    """
    Generated quantities: transformed parameters on natural scale.

    Returns:
    - alpha, r, gamma, lambda for each subject (4 * n_subjects values)
    - Population mu_k, psi_k (natural scale) for 4 quantities (8 values)
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    def get_subject_theta(sid):
        alpha_raw, r_val, gamma_raw, lambda_raw = _get_subject_params(
            chain_state, sid)
        alpha = jnp.exp(alpha_raw)
        gamma = jnp.exp(gamma_raw)
        lam = jnp.exp(lambda_raw)
        return jnp.array([alpha, r_val, gamma, lam])

    subject_params = jax.vmap(get_subject_theta)(jnp.arange(n_subjects))
    alpha_subjects = subject_params[:, 0]
    r_subjects = subject_params[:, 1]
    gamma_subjects = subject_params[:, 2]
    lambda_subjects = subject_params[:, 3]

    # Population hyperparameters on natural scale
    pop_params = []
    for k in range(N_QUANTITIES):
        hs = idx['hyper_starts'][k]
        mu_k = jax.nn.sigmoid(chain_state[hs])
        psi_k = jax.nn.sigmoid(chain_state[hs + 1])
        pop_params.extend([mu_k, psi_k])

    return jnp.concatenate([
        alpha_subjects,
        r_subjects,
        gamma_subjects,
        lambda_subjects,
        jnp.array(pop_params),
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    n_subjects = data["static"][0]
    return 4 * n_subjects + 2 * N_QUANTITIES


# =============================================================================
# Initial Values
# =============================================================================

def initial_vector(mcmc_config, data):
    """
    Generate initial parameter values for all chains.

    Subject params: alpha_raw near log(0.1)=-2.3, r near 0.5,
                    gamma_raw near log(0.65)=-0.43, lambda_raw near -2.
    Hyper params: logit_mu near 0 (mu=0.5), logit_psi near -2 (psi~0.12).
    """
    n_subjects = data["static"][0]
    total_params = n_subjects * NPARAMS + N_HYPER

    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))

    init = np.zeros((num_chains, total_params))
    first_hyper = n_subjects * NPARAMS

    # Subject parameters
    INIT_STD = 0.3
    init[:, 0:first_hyper:NPARAMS] = rng.normal(np.log(0.1), INIT_STD, (num_chains, n_subjects))   # alpha_raw
    init[:, 1:first_hyper:NPARAMS] = rng.normal(0.5, INIT_STD, (num_chains, n_subjects))            # r
    init[:, 2:first_hyper:NPARAMS] = rng.normal(np.log(0.65), INIT_STD, (num_chains, n_subjects))   # gamma_raw
    init[:, 3:first_hyper:NPARAMS] = rng.normal(-2.0, 0.5, (num_chains, n_subjects))                # lambda_raw

    # Hyperparameters (logit scale) — same as CRRA model
    HYPER_INIT_STD = 0.1
    for k in range(N_QUANTITIES):
        name = QUANTITY_NAMES[k]
        spec = HYPERPRIOR_SPECS[name]
        mu_star = spec.get('a_mu', 2.0) / (spec.get('a_mu', 2.0) + spec.get('b_mu', 2.0))
        logit_mu_init = np.log(mu_star / (1 - mu_star))
        logit_psi_init = -2.0

        init[:, first_hyper + k * 2 + 0] = rng.normal(logit_mu_init, HYPER_INIT_STD, num_chains)
        init[:, first_hyper + k * 2 + 1] = rng.normal(logit_psi_init, HYPER_INIT_STD, num_chains)

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
    })
    print(f"Registered posterior: {MODEL_ID}")


# Register on import if bamcmc is available
if BAMCMC_AVAILABLE:
    register()
