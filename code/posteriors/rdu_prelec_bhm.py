"""
RDU Prelec Bayesian Hierarchical Model

Rank-Dependent Utility with Prelec's 2-parameter probability weighting function.
"""

import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as stats

from bamcmc import (
    BlockSpec,
    SamplerType,
    ProposalType,
    create_subject_blocks,
)

from .common_fns import (
    crra_ufn,
    prelec_pwf,
    risk_likelihood,
    direct_dispatch_4params,
)


NPARAMS = 4  # r, alpha_raw, beta_raw, mu_raw per subject


def batch_specs(mcmc_config, data):
    """
    RDU with Prelec PWF - Bayesian Hierarchical Model block specification.

    Parameters per subject (4):
    - r: CRRA utility curvature
    - alpha_raw: log of Prelec curvature (alpha = exp(alpha_raw))
    - beta_raw: log of Prelec elevation (beta = exp(beta_raw))
    - mu_raw: log noise parameter

    Hyperparameters (8 total): mean and logsd for each subject parameter
    """
    n_subjects = data["static"][0]

    proposal_type = ProposalType.CHAIN_MEAN

    # Subject parameters: 4 params per subject
    subject_specs = create_subject_blocks(
        n_subjects=n_subjects,
        params_per_subject=NPARAMS,
        sampler_type=SamplerType.METROPOLIS_HASTINGS,
        proposal_type=proposal_type,
        label_prefix="Subject"
    )

    # Hyperparameters: 8 total
    hyper_specs = [
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_R_Mean",
            metadata={'interpretation': 'Population mean for CRRA r'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_R_LogSD",
            metadata={'interpretation': 'Population log-std for r'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_Alpha_Mean",
            metadata={'interpretation': 'Population mean for log(alpha)'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_Alpha_LogSD",
            metadata={'interpretation': 'Population log-std for log(alpha)'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_Beta_Mean",
            metadata={'interpretation': 'Population mean for log(beta)'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_Beta_LogSD",
            metadata={'interpretation': 'Population log-std for log(beta)'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_Mu_Mean",
            metadata={'interpretation': 'Population mean for log(mu)'}
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=direct_dispatch_4params,
            label="Hyper_Mu_LogSD",
            metadata={'interpretation': 'Population log-std for log(mu)'}
        ),
    ]

    return subject_specs + hyper_specs


def log_posterior(chain_state, param_indices, data):
    """
    Log posterior for RDU with Prelec PWF hierarchical model.

    RDU formula for sorted outcomes x_1 < x_2 < ... < x_n:
        RDU = sum_{i=1}^{n} [w(F(x_i)) - w(F(x_{i-1}))] * u(x_i)
    where F(x_i) is the cumulative probability and w() is the probability weighting function.

    Parameters per subject (4):
        - r: CRRA curvature
        - alpha_raw: log Prelec curvature (alpha = exp(alpha_raw))
        - beta_raw: log Prelec elevation (beta = exp(beta_raw))
        - mu_raw: log noise parameter
    """
    datas = data["static"]
    datai = data["int"]
    dataf = data["float"]

    nids = datas[0]
    first_hyper = nids * NPARAMS

    # Identify Subject ID
    sid = param_indices[0] // NPARAMS

    # Extract Hyperparameters
    r_mean        = chain_state[first_hyper + 0]
    ln_r_sdev     = chain_state[first_hyper + 1]
    alpha_mean    = chain_state[first_hyper + 2]
    ln_alpha_sdev = chain_state[first_hyper + 3]
    beta_mean     = chain_state[first_hyper + 4]
    ln_beta_sdev  = chain_state[first_hyper + 5]
    mu_mean       = chain_state[first_hyper + 6]
    ln_mu_sdev    = chain_state[first_hyper + 7]

    # Extract Subject Parameters
    r_val     = chain_state[param_indices[0]]
    alpha_raw = chain_state[param_indices[1]]
    beta_raw  = chain_state[param_indices[2]]
    mu_raw    = chain_state[param_indices[3]]

    # Local Prior
    lp = 0.0
    lp += stats.norm.logpdf(r_val,     loc=r_mean,     scale=jnp.exp(ln_r_sdev))
    lp += stats.norm.logpdf(alpha_raw, loc=alpha_mean, scale=jnp.exp(ln_alpha_sdev))
    lp += stats.norm.logpdf(beta_raw,  loc=beta_mean,  scale=jnp.exp(ln_beta_sdev))
    lp += stats.norm.logpdf(mu_raw,    loc=mu_mean,    scale=jnp.exp(ln_mu_sdev))

    # Transform to positive values
    alpha = jnp.exp(alpha_raw)
    beta  = jnp.exp(beta_raw)
    mu    = jnp.exp(mu_raw)

    # Lookup tables
    unique_outcomes = dataf[0]
    unique_cumprobs = dataf[1]

    # Precompute utilities for all unique outcomes
    unique_utils = crra_ufn(unique_outcomes, r_val)

    # Precompute weighted cumulative probs w(F) for all unique cumprobs
    unique_weighted = prelec_pwf(unique_cumprobs, alpha, beta)

    # This is the same for all risk models
    log_like = risk_likelihood(unique_weighted, unique_utils, datai, sid, mu)

    total_log_prob = lp + log_like

    return total_log_prob


def generated_quantities(chain_state, data):
    """
    Generated quantities for RDU Prelec model.

    Returns transformed parameters:
    - r for each subject
    - alpha (Prelec curvature) for each subject
    - beta (Prelec elevation) for each subject
    - mu for each subject
    - Population means for each
    """
    nids = data["static"][0]
    first_hyper = nids * NPARAMS

    # Subject-level parameters
    r_subjects = chain_state[0:first_hyper:NPARAMS]
    alpha_raw_subjects = chain_state[1:first_hyper:NPARAMS]
    beta_raw_subjects = chain_state[2:first_hyper:NPARAMS]
    mu_raw_subjects = chain_state[3:first_hyper:NPARAMS]

    # Transform to natural scale
    alpha_subjects = jnp.exp(alpha_raw_subjects)
    beta_subjects = jnp.exp(beta_raw_subjects)
    mu_subjects = jnp.exp(mu_raw_subjects)

    # Population means (transformed)
    r_pop_mean = chain_state[first_hyper + 0]
    alpha_pop_mean = jnp.exp(chain_state[first_hyper + 2])
    beta_pop_mean = jnp.exp(chain_state[first_hyper + 4])
    mu_pop_mean = jnp.exp(chain_state[first_hyper + 6])

    return jnp.concatenate([
        r_subjects,
        alpha_subjects,
        beta_subjects,
        mu_subjects,
        jnp.array([r_pop_mean, alpha_pop_mean, beta_pop_mean, mu_pop_mean])
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    sids = data["static"][0]
    # r, alpha, beta, mu for each subject + 4 population means
    return 4 * sids + 4


def get_initial_vector(mcmc_config, data):
    """Generate initial parameter values for chains."""
    num_chains = mcmc_config["num_chains"]
    K = mcmc_config.get("num_superchains", num_chains)
    num_distinct = K

    sids = data["static"][0]
    total = sids * NPARAMS + 8
    init = np.zeros((num_distinct, total))

    # Hyperparameters initialization (last 8 columns)
    # r: CRRA curvature, typical ~0.5
    init[:, -8] = np.random.normal(0.5, 0.1, num_distinct)   # Hyper r mean
    init[:, -7] = np.random.normal(-1.0, 0.1, num_distinct)  # Hyper r log-sd
    # alpha: Prelec curvature, typical ~0.7 (log ~ -0.36)
    init[:, -6] = np.random.normal(np.log(1), 0.1, num_distinct)  # Hyper alpha mean (log scale)
    init[:, -5] = np.random.normal(-1.0, 0.1, num_distinct)  # Hyper alpha log-sd
    # beta: Prelec elevation, typical ~1.0 (log ~ 0)
    init[:, -4] = np.random.normal(np.log(1), 0.1, num_distinct)   # Hyper beta mean (log scale)
    init[:, -3] = np.random.normal(-1.0, 0.1, num_distinct)  # Hyper beta log-sd
    # mu: noise parameter
    init[:, -2] = np.random.normal(np.log(.15), 0.1, num_distinct)  # Hyper mu mean
    init[:, -1] = np.random.normal(-1.0, 0.1, num_distinct)  # Hyper mu log-sd

    # Subject parameters (indices: 0=r, 1=alpha_raw, 2=beta_raw, 3=mu_raw per subject)
    first_hyper = sids * NPARAMS
    init[:, 0:first_hyper:NPARAMS] = np.random.normal(0.5,         0.3, (num_distinct, sids))   # r
    init[:, 1:first_hyper:NPARAMS] = np.random.normal(np.log(1),   0.2, (num_distinct, sids))  # alpha_raw
    init[:, 2:first_hyper:NPARAMS] = np.random.normal(np.log(1),   0.2, (num_distinct, sids))   # beta_raw
    init[:, 3:first_hyper:NPARAMS] = np.random.normal(np.log(.15), 0.1, (num_distinct, sids))  # mu_raw

    if num_distinct < num_chains:
        full_init = np.zeros((num_chains, total))
        full_init[:num_distinct] = init
        return full_init.flatten()
    return init.flatten()
