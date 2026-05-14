"""
Prior-only version of CRRA + Power PWF BHM (no data likelihood).

Same as rdu_power_bhm_ce.py but the log posterior returns
lp_pop + log_gram (no likelihood term). Used for prior predictive
sampling to verify that the quantity-space prior produces sensible
CE distributions before seeing data.
"""

import jax
import jax.numpy as jnp

try:
    from bamcmc import register_posterior
    BAMCMC_AVAILABLE = True
except ImportError:
    BAMCMC_AVAILABLE = False

from .common_fns import full_forward_map_power, gram_logdet_power

from .rdu_bhm_ce import (
    _compute_all_hyperpriors,
    _subject_quantity_prior,
)

from .rdu_power_bhm_ce import (
    MODEL_ID as _PARENT_ID,
    NPARAMS, N_HYPER,
    _get_indices,
    _get_subject_params,
    batch_type,
    generated_quantities,
    get_num_gq,
    initial_vector,
    direct_sampler,
    load_hyperpriors_from_file,
    get_hyperprior_array,
)

MODEL_ID = "rdu_power_bhm_ce_prior"


def log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log posterior for prior-only CRRA+Power model (no likelihood)."""
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
            r_val, delta_raw, lambda_raw = _get_subject_params(
                chain_state, sid)
            quantities = full_forward_map_power(
                r_val, delta_raw, lambda_raw,
                ref_lotteries, ref_pair)
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)
            return acc + lp_pop

        total = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)
        return lp_hyper + total

    def compute_subject_posterior():
        sid = first_param // NPARAMS
        r_val, delta_raw, lambda_raw = _get_subject_params(
            chain_state, sid)

        dataf = data["float"]
        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        quantities = full_forward_map_power(
            r_val, delta_raw, lambda_raw,
            ref_lotteries, ref_pair)

        lp_pop = _subject_quantity_prior(
            quantities, chain_state, idx, hyperprior_arr)

        log_gram = gram_logdet_power(
            r_val, delta_raw, lambda_raw,
            ref_lotteries, ref_pair)

        # No likelihood term
        return lp_pop + log_gram

    return jax.lax.cond(
        is_hyper,
        compute_hyper_posterior,
        compute_subject_posterior
    )


def register():
    if not BAMCMC_AVAILABLE:
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


if BAMCMC_AVAILABLE:
    register()
