"""
Prior-only version of the EP+EUT BHM with Quantity-Space Prior (Beta-Beta).

Identical to rdu_ep_eut_bhm_ce.py except the data likelihood is omitted.
This samples from the induced prior on (alpha, r, lambda) implied by the
quantity-space Beta-Beta hierarchy and the Gram determinant correction,
without conditioning on any choice data.

If the Gram determinant is causing sampling problems, they will show up
here without the confound of a difficult likelihood surface.
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

from .common_fns import (
    full_forward_map_ep_eut,
    gram_logdet_ep_eut,
)

MODEL_ID = "rdu_ep_eut_bhm_ce_prior"

NPARAMS = 3
N_HYPER = 8
HYPER_BLOCK_SIZE = 2
N_QUANTITIES = 4

from .rdu_bhm_ce import (
    _log_beta_pdf_normalized,
    _hyperprior_logit_beta,
    _compute_all_hyperpriors,
    _subject_quantity_prior,
    load_hyperpriors_from_file,
    get_hyperprior_array,
    HYPERPRIOR_SPECS,
)

from .rdu_ep_eut_bhm_ce import (
    _get_indices,
    _get_subject_params,
    generated_quantities,
    get_num_gq,
    initial_vector,
    direct_sampler,
)


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """Same block structure as full EP+EUT model."""
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
# Log Posterior (prior only — no data likelihood)
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log prior for EP+EUT quantity-space Beta-Beta hierarchical model.

    Same as rdu_ep_eut_bhm_ce.log_posterior but with the data likelihood removed.
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        dataf = data["float"]
        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        lp_hyper = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        def subject_ll_body(sid, acc):
            alpha_raw, r_val, lambda_raw = _get_subject_params(
                chain_state, sid)
            alpha = jnp.exp(alpha_raw)
            quantities = full_forward_map_ep_eut(
                alpha, r_val, lambda_raw,
                ref_lotteries, ref_pair)
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)
            return acc + lp_pop

        total = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)
        return lp_hyper + total

    def compute_subject_posterior():
        sid = first_param // NPARAMS
        alpha_raw, r_val, lambda_raw = _get_subject_params(
            chain_state, sid)

        dataf = data["float"]
        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        alpha = jnp.exp(alpha_raw)
        quantities = full_forward_map_ep_eut(
            alpha, r_val, lambda_raw,
            ref_lotteries, ref_pair)

        lp_pop = _subject_quantity_prior(
            quantities, chain_state, idx, hyperprior_arr)

        # Gram determinant — autodiff through exp(alpha_raw)
        def _fmap_raw(theta):
            return full_forward_map_ep_eut(
                jnp.exp(theta[0]), theta[1], theta[2],
                ref_lotteries, ref_pair)

        theta = jnp.array([alpha_raw, r_val, lambda_raw])
        J = jax.jacfwd(_fmap_raw)(theta)
        gram = J.T @ J
        sign, logdet = jnp.linalg.slogdet(gram)
        log_gram = jnp.where(sign > 0, 0.5 * logdet, -jnp.inf)

        return lp_pop + log_gram

    return jax.lax.cond(
        is_hyper,
        compute_hyper_posterior,
        compute_subject_posterior
    )


# =============================================================================
# Register with bamcmc
# =============================================================================

if BAMCMC_AVAILABLE:
    register_posterior(MODEL_ID, {
        'log_posterior': log_posterior,
        'batch_type': batch_type,
        'initial_vector': initial_vector,
        'direct_sampler': direct_sampler,
        'generated_quantities': generated_quantities,
        'get_num_gq': get_num_gq,
    })
    print(f"Registered posterior: {MODEL_ID}")
