"""
Rank-Dependent Utility (RDU) BHM with Quantity-Space Prior (Beta-Beta).

This posterior implements an RDU model where the prior on individual-level
parameters (r, phi_raw, eta_raw, lambda_raw) is specified in quantity space
(3 CEs + 1 choice probability) using a Beta-Beta hierarchical model with
Jacobian correction:

    p(theta) = p_Y(g(theta)) |det J_g(theta)|

where g maps (r, phi_raw, eta_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).

All 4 subject parameters are CENTERED (no NCP). All 4 generated quantities
use the same Beta-Beta hierarchy.

Hyperparameters (8 total, on logit scale):
    [logit_mu_1, logit_psi_1,   # CE1 population mean & dispersion
     logit_mu_2, logit_psi_2,   # CE2
     logit_mu_3, logit_psi_3,   # CE3
     logit_mu_P, logit_psi_P]   # P_choice

Data format: Same as rdu_bhm, plus:
- float[7]: hyperprior_arr (24 values) — Beta-Beta hyperprior for 4 quantities
- float[8]: ref_lotteries (18 values) — 3 lotteries x (3 prizes + 3 probs)
- float[9]: ref_pair (12 values) — 2 lotteries x (3 prizes + 3 probs)
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
    crra_ufn,
    risk_likelihood,
    full_forward_map,
    full_jacobian_logabsdet,
)

MODEL_ID = "rdu_bhm_ce"

# =============================================================================
# Model Constants
# =============================================================================

NPARAMS = 4          # r, phi_raw, eta_raw, lambda_raw per subject
N_HYPER = 8          # (logit_mu, logit_psi) for 4 quantities
HYPER_BLOCK_SIZE = 2
N_QUANTITIES = 4     # CE1, CE2, CE3, P_choice
HYPERPRIOR_STRIDE = 6  # per-quantity: a_mu, b_mu, a_psi, b_psi, lo, hi

QUANTITY_NAMES = ('ce1', 'ce2', 'ce3', 'pchoice')


# =============================================================================
# Index Helpers
# =============================================================================

def _get_indices(nids):
    """Compute index offsets for the state vector."""
    first_hyper = nids * NPARAMS

    # Hyperparameter layout (logit scale):
    # [logit_mu_1, logit_psi_1, logit_mu_2, logit_psi_2,
    #  logit_mu_3, logit_psi_3, logit_mu_P, logit_psi_P]
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
    Beta-Beta BHM block specification.

    Subject blocks: (r, phi_raw, eta_raw, lambda_raw) per subject — MH.
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
# Beta-Beta Hyperprior Functions
# =============================================================================

def _log_beta_pdf(x, a, b):
    """Log PDF of Beta(a, b) evaluated at x, without normalizing constant."""
    return (a - 1) * jnp.log(x) + (b - 1) * jnp.log(1 - x)


def _log_beta_pdf_normalized(x, a, b):
    """Log PDF of Beta(a, b) with full normalizing constant."""
    log_B = jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b) \
            - jax.scipy.special.gammaln(a + b)
    return (a - 1) * jnp.log(x) + (b - 1) * jnp.log(1 - x) - log_B


def _hyperprior_logit_beta(logit_val, a, b):
    """
    Log hyperprior for a logit-transformed Beta parameter.

    If x = sigmoid(logit_val) ~ Beta(a, b), then:
        log p(logit_val) = log Beta(sigmoid(logit_val); a, b)
                         + log sigmoid'(logit_val)
    where sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
    """
    x = jax.nn.sigmoid(logit_val)
    lp = _log_beta_pdf_normalized(x, a, b)
    # Jacobian of logit -> x: d/d(logit) sigmoid(logit) = x * (1-x)
    lp += jnp.log(x) + jnp.log(1 - x)
    return lp


def _compute_all_hyperpriors(chain_state, idx, hyperprior_arr):
    """
    Compute sum of all hyperprior terms for 4 quantities.

    Hyperprior array layout (24 values = 4 quantities x 6):
    For each quantity k: [a_mu_k, b_mu_k, a_psi_k, b_psi_k, lo_k, hi_k]
    """
    lp = 0.0

    for k in range(N_QUANTITIES):
        hs = idx['hyper_starts'][k]
        logit_mu = chain_state[hs]
        logit_psi = chain_state[hs + 1]

        offset = k * HYPERPRIOR_STRIDE
        a_mu = hyperprior_arr[offset + 0]
        b_mu = hyperprior_arr[offset + 1]
        a_psi = hyperprior_arr[offset + 2]
        b_psi = hyperprior_arr[offset + 3]

        lp += _hyperprior_logit_beta(logit_mu, a_mu, b_mu)
        lp += _hyperprior_logit_beta(logit_psi, a_psi, b_psi)

    return lp


# =============================================================================
# Subject Parameter Extraction
# =============================================================================

def _get_subject_params(chain_state, sid):
    """Extract parameters for a subject (all centered)."""
    subject_start = sid * NPARAMS
    r_val = chain_state[subject_start + 0]
    phi_raw = chain_state[subject_start + 1]
    eta_raw = chain_state[subject_start + 2]
    lambda_raw = chain_state[subject_start + 3]
    return r_val, phi_raw, eta_raw, lambda_raw


# =============================================================================
# Population Prior (Beta-Beta on quantities)
# =============================================================================

def _subject_quantity_prior(quantities, chain_state, idx, hyperprior_arr):
    """
    Beta-Beta population prior on 4 generated quantities for one subject.

    For each quantity k:
        z_k = (Y_k - lo_k) / (hi_k - lo_k)  (normalized to [0,1])
        z_k | mu_k, psi_k ~ Beta(mu_k*(1-psi_k)/psi_k, (1-mu_k)*(1-psi_k)/psi_k)
    """
    lp = 0.0

    for k in range(N_QUANTITIES):
        hs = idx['hyper_starts'][k]
        mu_k = jax.nn.sigmoid(chain_state[hs])
        psi_k = jax.nn.sigmoid(chain_state[hs + 1])

        offset = k * HYPERPRIOR_STRIDE
        lo_k = hyperprior_arr[offset + 4]
        hi_k = hyperprior_arr[offset + 5]

        # Normalize quantity to [0, 1]
        z_k = (quantities[k] - lo_k) / (hi_k - lo_k)
        z_k = jnp.clip(z_k, 1e-10, 1 - 1e-10)

        # Beta population distribution parameters
        alpha_pop = mu_k * (1 - psi_k) / psi_k
        beta_pop = (1 - mu_k) * (1 - psi_k) / psi_k

        # Clamp to avoid numerical issues
        alpha_pop = jnp.maximum(alpha_pop, 1e-6)
        beta_pop = jnp.maximum(beta_pop, 1e-6)

        lp += _log_beta_pdf_normalized(z_k, alpha_pop, beta_pop)

        # Jacobian for normalizing Y -> z: dz/dY = 1/(hi-lo)
        lp -= jnp.log(hi_k - lo_k)

    return lp


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log posterior for RDU quantity-space Beta-Beta hierarchical model."""
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)
    first_param = param_indices[0]
    first_hyper = idx['first_hyper']

    is_hyper = first_param >= first_hyper

    def compute_hyper_posterior():
        """Log posterior for hyperparameter blocks.

        Only the hyperprior and population prior terms are needed here.
        The Jacobian |det J_g(theta_i)| and data likelihood p(y_i | theta_i)
        do not depend on hyperparameters and cancel in the MH ratio.
        """
        dataf = data["float"]

        hyperprior_arr = jnp.asarray(dataf[7])
        ref_lotteries = jnp.asarray(dataf[8])
        ref_pair = jnp.asarray(dataf[9])

        lp_hyper = _compute_all_hyperpriors(chain_state, idx, hyperprior_arr)

        def subject_ll_body(sid, acc):
            r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
                chain_state, sid)

            # Forward map: theta -> (CE1, CE2, CE3, P_choice)
            quantities = full_forward_map(
                r_val, phi_raw, eta_raw, lambda_raw,
                ref_lotteries, ref_pair)

            # Beta-Beta population prior on quantities
            lp_pop = _subject_quantity_prior(
                quantities, chain_state, idx, hyperprior_arr)

            # Jacobian and likelihood omitted: they don't depend on
            # hyperparameters and cancel in the MH ratio.

            return acc + lp_pop

        total = jax.lax.fori_loop(0, n_subjects, subject_ll_body, 0.0)

        return lp_hyper + total

    def compute_subject_posterior():
        """Log posterior for a single subject block."""
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

        # Beta-Beta population prior
        lp_pop = _subject_quantity_prior(
            quantities, chain_state, idx, hyperprior_arr)

        # Jacobian correction
        log_det_j = full_jacobian_logabsdet(
            r_val, phi_raw, eta_raw, lambda_raw,
            ref_lotteries, ref_pair)

        # Data likelihood
        phi = jnp.exp(phi_raw)
        eta = jnp.exp(eta_raw)
        lam = jnp.exp(lambda_raw)

        datai_jax = tuple(jnp.asarray(arr) for arr in data["int"])
        dataf_jax = tuple(jnp.asarray(arr) for arr in dataf)
        unique_outcomes = dataf_jax[0]
        unique_utils = crra_ufn(unique_outcomes, r_val)
        log_like = risk_likelihood(
            unique_utils, phi, eta, datai_jax, dataf_jax, sid, lam)

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
    - r, phi, eta, lambda for each subject (4 * n_subjects values)
    - Population mu_k, psi_k (natural scale) for 4 quantities (8 values)
    """
    n_subjects = data["static"][0]
    idx = _get_indices(n_subjects)

    def get_subject_theta(sid):
        r_val, phi_raw, eta_raw, lambda_raw = _get_subject_params(
            chain_state, sid)
        phi = jnp.exp(phi_raw)
        eta = jnp.exp(eta_raw)
        lam = jnp.exp(lambda_raw)
        return jnp.array([r_val, phi, eta, lam])

    subject_params = jax.vmap(get_subject_theta)(jnp.arange(n_subjects))
    r_subjects = subject_params[:, 0]
    phi_subjects = subject_params[:, 1]
    eta_subjects = subject_params[:, 2]
    lambda_subjects = subject_params[:, 3]

    # Population hyperparameters on natural scale
    pop_params = []
    for k in range(N_QUANTITIES):
        hs = idx['hyper_starts'][k]
        mu_k = jax.nn.sigmoid(chain_state[hs])
        psi_k = jax.nn.sigmoid(chain_state[hs + 1])
        pop_params.extend([mu_k, psi_k])

    return jnp.concatenate([
        r_subjects,
        phi_subjects,
        eta_subjects,
        lambda_subjects,
        jnp.array(pop_params),
    ])


def get_num_gq(mcmc_config, data):
    """Get number of generated quantities."""
    n_subjects = data["static"][0]
    return 4 * n_subjects + 2 * N_QUANTITIES


# =============================================================================
# Hyperprior Array (from loaded specs)
# =============================================================================

# Default specs — overwritten by load_hyperpriors_from_file
HYPERPRIOR_SPECS = {
    'ce1':     {'a_mu': 2.0, 'b_mu': 2.0, 'a_psi': 2.0, 'b_psi': 8.0,
                'bound_lo': 10.0, 'bound_hi': 50.0},
    'ce2':     {'a_mu': 2.0, 'b_mu': 2.0, 'a_psi': 2.0, 'b_psi': 8.0,
                'bound_lo': 20.0, 'bound_hi': 40.0},
    'ce3':     {'a_mu': 2.0, 'b_mu': 2.0, 'a_psi': 2.0, 'b_psi': 8.0,
                'bound_lo': 15.0, 'bound_hi': 45.0},
    'pchoice': {'a_mu': 2.0, 'b_mu': 2.0, 'a_psi': 2.0, 'b_psi': 8.0,
                'bound_lo': 0.0, 'bound_hi': 1.0},
}


def load_hyperpriors_from_file(filepath):
    """
    Load hyperprior specifications from a prior_params_ce.npz file.

    Updates the global HYPERPRIOR_SPECS with values from the file.
    """
    global HYPERPRIOR_SPECS

    data = np.load(filepath)

    new_specs = {}
    for name in QUANTITY_NAMES:
        new_specs[name] = {
            'a_mu': float(data[f'{name}_a_mu']),
            'b_mu': float(data[f'{name}_b_mu']),
            'a_psi': float(data[f'{name}_a_psi']),
            'b_psi': float(data[f'{name}_b_psi']),
            'bound_lo': float(data[f'{name}_bound_lo']),
            'bound_hi': float(data[f'{name}_bound_hi']),
        }

    HYPERPRIOR_SPECS.update(new_specs)
    print(f"Loaded CE hyperpriors from {filepath}")
    for name in QUANTITY_NAMES:
        s = new_specs[name]
        print(f"  {name}: a_mu={s['a_mu']:.3f}, b_mu={s['b_mu']:.3f}, "
              f"a_psi={s['a_psi']:.3f}, b_psi={s['b_psi']:.3f}, "
              f"bounds=[{s['bound_lo']:.1f}, {s['bound_hi']:.1f}]")

    return new_specs


def get_hyperprior_array():
    """
    Convert HYPERPRIOR_SPECS to a flat numpy array for passing through data.

    Returns array of shape (24,) with layout:
    For each of 4 quantities: [a_mu, b_mu, a_psi, b_psi, lo, hi]
    """
    arr = np.zeros(N_QUANTITIES * HYPERPRIOR_STRIDE, dtype=np.float64)
    for k, name in enumerate(QUANTITY_NAMES):
        spec = HYPERPRIOR_SPECS[name]
        offset = k * HYPERPRIOR_STRIDE
        arr[offset + 0] = spec['a_mu']
        arr[offset + 1] = spec['b_mu']
        arr[offset + 2] = spec['a_psi']
        arr[offset + 3] = spec['b_psi']
        arr[offset + 4] = spec['bound_lo']
        arr[offset + 5] = spec['bound_hi']
    return arr


# =============================================================================
# Initial Values
# =============================================================================

def initial_vector(mcmc_config, data):
    """
    Generate initial parameter values for all chains.

    Subject params: r near 0.5, phi_raw/eta_raw near 0, lambda_raw near -2.
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
    init[:, 0:first_hyper:NPARAMS] = rng.normal(0.5, INIT_STD, (num_chains, n_subjects))    # r
    init[:, 1:first_hyper:NPARAMS] = rng.normal(0.0, INIT_STD, (num_chains, n_subjects))    # phi_raw
    init[:, 2:first_hyper:NPARAMS] = rng.normal(0.0, INIT_STD, (num_chains, n_subjects))    # eta_raw
    init[:, 3:first_hyper:NPARAMS] = rng.normal(-2.0, 0.5, (num_chains, n_subjects))        # lambda_raw

    # Hyperparameters (logit scale)
    HYPER_INIT_STD = 0.1
    for k in range(N_QUANTITIES):
        name = QUANTITY_NAMES[k]
        spec = HYPERPRIOR_SPECS[name]
        # Initialize logit_mu near logit of normalized target mean
        mu_star = (spec.get('a_mu', 2.0)) / (spec.get('a_mu', 2.0) + spec.get('b_mu', 2.0))
        logit_mu_init = np.log(mu_star / (1 - mu_star))  # logit(E[mu])
        logit_psi_init = -2.0  # psi ~ 0.12

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
