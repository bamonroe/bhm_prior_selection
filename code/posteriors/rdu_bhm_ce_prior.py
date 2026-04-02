"""
Prior-only version of the RDU BHM with Quantity-Space Prior (Beta-Beta).

Identical to rdu_bhm_ce.py except the data likelihood is omitted from the
subject-level log posterior. This samples from the induced prior on
(r, phi, eta, lambda) implied by the quantity-space Beta-Beta hierarchy
and the Jacobian correction, without conditioning on any choice data.

The output is the hyperprior predictive distribution in parameter space:
what the CE-space specification implies about structural parameters before
seeing any data.
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
    full_forward_map,
    full_jacobian_logabsdet,
)

MODEL_ID = "rdu_bhm_ce_prior"

# Same constants as the full model
NPARAMS = 4
N_HYPER = 8
HYPER_BLOCK_SIZE = 2
N_QUANTITIES = 4
HYPERPRIOR_STRIDE = 6

# Reuse helpers from the full model
from .rdu_bhm_ce import (
    _get_indices,
    _log_beta_pdf_normalized,
    _hyperprior_logit_beta,
    _compute_all_hyperpriors,
    _get_subject_params,
    _subject_quantity_prior,
    generated_quantities,
    get_hyperprior_array,
    load_hyperpriors_from_file,
    HYPERPRIOR_PARAMS,
)


# =============================================================================
# Block Specifications
# =============================================================================

def batch_type(mcmc_config, data):
    """Same block structure as full model."""
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
    """Log prior for RDU quantity-space Beta-Beta hierarchical model.

    Same as rdu_bhm_ce.log_posterior but with the data likelihood removed.
    The subject-level contribution is: population prior + Jacobian correction.
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        dataf = data["float"]
        hyperprior_arr = jnp.asarray(dataf[7])

        lp_hyper = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        def subject_ll_body(sid, acc):
            r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
                chain_state, sid)
            ref_lotteries = jnp.asarray(dataf[8])
            ref_pair = jnp.asarray(dataf[9])
            quantities = full_forward_map(
                r_val, phi_raw, eta_raw, lambda_raw,
                ref_lotteries, ref_pair)
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)
            return acc + lp_pop

        total = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)
        return lp_hyper + total

    def compute_subject_posterior():
        sid = first_param // NPARAMS
        r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
            chain_state, sid)

        dataf = data["float"]
        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        # Forward map
        quantities = full_forward_map(
            r_val, phi_raw, eta_raw, lambda_raw,
            ref_lotteries, ref_pair)

        # Population prior
        lp_pop = _subject_quantity_prior(
            quantities, chain_state, idx, hyperprior_arr)

        # Jacobian correction
        log_det_j = full_jacobian_logabsdet(
            r_val, phi_raw, eta_raw, lambda_raw,
            ref_lotteries, ref_pair)

        # No data likelihood — this is the prior only
        return lp_pop + log_det_j

    return jax.lax.cond(
        is_hyper,
        compute_hyper_posterior,
        compute_subject_posterior
    )


# =============================================================================
# Register with bamcmc
# =============================================================================

if BAMCMC_AVAILABLE:
    register_posterior(
        MODEL_ID,
        log_posterior,
        generated_quantities,
        batch_type,
    )
    print(f"Registered posterior: {MODEL_ID}")
