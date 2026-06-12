"""
Central model configuration registry.

Defines all RDU model specifications in one place. Every module that needs
model-specific information (param names, CE computation, plot colors, etc.)
imports from here instead of hardcoding.

To add a new model: add an entry to MODEL_REGISTRY and create the
corresponding posterior and sampling modules.
"""

import numpy as np
from pathlib import Path


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    'crra': {
        'model_id': 'rdu_bhm_ce',
        'name': 'CRRA+Prelec',
        'color': '#1f77b4',
        'linestyle': '-',
        'posterior_module': 'posteriors.rdu_bhm_ce',
        'sampling_module': 'sampling_ce',
        'nparams': 4,
        'n_hyper': 8,
        'param_names': ['r', 'phi', 'eta', 'lambda'],
        'param_labels': [r'$r$', r'$\phi$', r'$\eta$', r'$\lambda$'],
        'utility_type': 'crra',
        'pwf_type': 'prelec',
        'has_prior_only': True,
        'prior_model_id': 'rdu_bhm_ce_prior',
        'prior_posterior_module': 'posteriors.rdu_bhm_ce_prior',
        'prior_sampling_module': 'sampling_ce_prior',
    },
    'ge': {
        'model_id': 'rdu_ge_bhm_ce',
        'name': 'CRRA+GE',
        'color': '#2ca02c',
        'linestyle': '-',
        'posterior_module': 'posteriors.rdu_ge_bhm_ce',
        'sampling_module': 'sampling_ge',
        'nparams': 4,
        'n_hyper': 8,
        'param_names': ['r', 'gamma', 'delta', 'lambda'],
        'param_labels': [r'$r$', r'$\gamma$', r'$\delta$', r'$\lambda$'],
        'utility_type': 'crra',
        'pwf_type': 'ge',
        'has_prior_only': True,
        'prior_model_id': 'rdu_ge_bhm_ce_prior',
        'prior_posterior_module': 'posteriors.rdu_ge_bhm_ce_prior',
        'prior_sampling_module': 'sampling_ge',
    },
    'power': {
        'model_id': 'rdu_power_bhm_ce',
        'name': 'CRRA+Power',
        'color': '#d62728',
        'linestyle': '-',
        'posterior_module': 'posteriors.rdu_power_bhm_ce',
        'sampling_module': 'sampling_power',
        'nparams': 3,
        'n_hyper': 8,
        'param_names': ['r', 'delta', 'lambda'],
        'param_labels': [r'$r$', r'$\delta$', r'$\lambda$'],
        'utility_type': 'crra',
        'pwf_type': 'power',
        'has_prior_only': True,
        'prior_model_id': 'rdu_power_bhm_ce_prior',
        'prior_posterior_module': 'posteriors.rdu_power_bhm_ce_prior',
        'prior_sampling_module': 'sampling_power_prior',
    },
}


# =============================================================================
# Path Helpers
# =============================================================================

def _project_root():
    return Path(__file__).parent.parent


def get_model(key):
    """Return model config dict."""
    return MODEL_REGISTRY[key]


def get_data_dir(key):
    """Return data/output/{model_id}."""
    return _project_root() / 'data' / 'output' / MODEL_REGISTRY[key]['model_id']


def get_preproc_dir(key):
    """Return preprocessed directory for a model."""
    return get_data_dir(key) / 'preprocessed'


def get_history_dir(key):
    """Return history/full directory for a model."""
    return get_data_dir(key) / 'history' / 'full'


def get_plots_dir(key):
    """Return plots directory for a model."""
    return _project_root() / 'plots' / MODEL_REGISTRY[key]['model_id']


# =============================================================================
# CE Computation (unified dispatch)
# =============================================================================

def _compute_pwf_weights(cum_probs, pwf_type, pwf_params):
    """Compute PWF weights for cumulative probabilities.

    Args:
        cum_probs: array of cumulative probabilities (n_outcomes-1,)
        pwf_type: 'prelec', 'ge', or 'power'
        pwf_params: dict with PWF parameter arrays (n_draws,)

    Returns:
        w: array of shape (n_draws, n_outcomes) with w values
    """
    n_draws = len(list(pwf_params.values())[0])
    n_outcomes = len(cum_probs) + 1
    cum_clipped = np.clip(cum_probs, 1e-10, 1 - 1e-10)

    if pwf_type == 'prelec':
        phi = pwf_params['phi'][:, None]
        eta = pwf_params['eta'][:, None]
        neg_ln = -np.log(cum_clipped)[None, :]
        w_inner = np.exp(-eta * np.power(neg_ln, phi))
    elif pwf_type == 'power':
        delta = np.maximum(pwf_params['delta'], 0.01)[:, None]
        w_inner = cum_clipped[None, :] ** delta
    elif pwf_type == 'ge':
        gamma = np.maximum(pwf_params['gamma'], 0.01)[:, None]
        delta = np.maximum(pwf_params['delta'], 0.01)[:, None]
        p_g = cum_clipped[None, :] ** gamma
        q_g = (1 - cum_clipped[None, :]) ** gamma
        w_inner = delta * p_g / (delta * p_g + q_g)
    else:
        raise ValueError(f"Unknown PWF type: {pwf_type}")

    w = np.ones((n_draws, n_outcomes), dtype=np.float64)
    w[:, :-1] = w_inner
    return w


def _compute_crra_utilities_and_inverse(r_arr, sorted_prizes, rdu_values):
    """Compute CRRA utilities and invert RDU values to CEs."""
    x_min, x_max = sorted_prizes[0], sorted_prizes[-1]
    n_draws = len(r_arr)
    s_arr = 1.0 - r_arr
    use_log = np.abs(s_arr) < 1e-10
    s_safe = np.where(use_log, 0.5, s_arr)

    # Utilities
    sp_s = sorted_prizes[None, :] ** s_safe[:, None]
    xmin_s = x_min ** s_safe
    xmax_s = x_max ** s_safe
    denom = (xmax_s - xmin_s)[:, None]
    utilities = (sp_s - xmin_s[:, None]) / denom
    log_utilities = np.log(sorted_prizes / x_min)[None, :] / np.log(x_max / x_min)
    log_utilities = log_utilities * np.ones((n_draws, 1))
    final_utilities = np.where(use_log[:, None], log_utilities, utilities)

    # Inverse (for CEs)
    with np.errstate(invalid='ignore', over='ignore'):
        ce_s = rdu_values * (xmax_s - xmin_s) + xmin_s
        ce_normal = np.power(np.maximum(ce_s, 1e-30), 1.0 / s_safe)
        ce_log = x_min * np.power(x_max / x_min, rdu_values)
    ces = np.where(use_log, ce_log, ce_normal)

    return final_utilities, ces


def compute_ces_for_lottery(model_key, params_dict, lottery):
    """Compute CEs for arrays of parameters under a given model.

    Args:
        model_key: Key into MODEL_REGISTRY ('crra', 'ge', 'power')
        params_dict: Dict of {param_name: 1D array of draws}
        lottery: Dict with 'prizes' and 'probs'

    Returns:
        1D array of CEs
    """
    cfg = MODEL_REGISTRY[model_key]
    prizes = np.array(lottery['prizes'], dtype=np.float64)
    probs = np.array(lottery['probs'], dtype=np.float64)

    sorted_idx = np.argsort(prizes)
    sp = prizes[sorted_idx]
    sq = probs[sorted_idx]
    lo, hi = float(sp[0]), float(sp[-1])
    cum = np.cumsum(sq)

    # PWF weights — extract only the params this model's PWF needs
    pwf_type = cfg['pwf_type']
    if pwf_type == 'prelec':
        pwf_params = {'phi': params_dict['phi'], 'eta': params_dict['eta']}
    elif pwf_type == 'power':
        pwf_params = {'delta': params_dict['delta']}
    elif pwf_type == 'ge':
        pwf_params = {'gamma': params_dict['gamma'], 'delta': params_dict['delta']}
    else:
        raise ValueError(f"Unknown PWF type: {pwf_type}")
    w = _compute_pwf_weights(cum[:-1], pwf_type, pwf_params)

    # Decision weights
    d = np.empty_like(w)
    d[:, 0] = w[:, 0]
    d[:, 1:] = w[:, 1:] - w[:, :-1]

    # Utilities, RDU, and CEs
    r_arr = params_dict['r']
    if cfg['utility_type'] == 'crra':
        utilities, _ = _compute_crra_utilities_and_inverse(r_arr, sp, np.zeros(len(r_arr)))
        rdu = np.sum(d * utilities, axis=1)
        _, ces = _compute_crra_utilities_and_inverse(r_arr, sp, rdu)
    else:
        raise ValueError(f"Unknown utility type: {cfg['utility_type']}")

    ces = np.clip(ces, lo, hi)
    ces[~np.isfinite(ces)] = np.nan
    return ces


# =============================================================================
# Subject Parameter Loading (unified)
# =============================================================================

def load_subject_params(model_key, preprocessed_dir, sid=None):
    """Load subject parameters with correct names for a model.

    Args:
        model_key: Key into MODEL_REGISTRY
        preprocessed_dir: Path to preprocessed/subjects directory
        sid: Subject ID (if None, loads all subjects)

    Returns:
        If sid is not None: dict of {param_name: 1D array} (post burn-in, flattened)
        If sid is None: dict of {param_name: list of 1D arrays}, one per subject
    """
    cfg = MODEL_REGISTRY[model_key]
    preprocessed_dir = Path(preprocessed_dir)

    if sid is not None:
        d = np.load(preprocessed_dir / f'subject_{sid:03d}_params.npz')
        burn = d[cfg['param_names'][0]].shape[0] // 2
        return {k: d[k][burn:].ravel() for k in cfg['param_names']}

    meta = np.load(preprocessed_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])
    all_params = {k: [] for k in cfg['param_names']}
    for s in range(n_subjects):
        d = np.load(preprocessed_dir / f'subject_{s:03d}_params.npz')
        burn = d[cfg['param_names'][0]].shape[0] // 2
        for k in cfg['param_names']:
            all_params[k].append(d[k][burn:].ravel())

    return {k: np.array(v) for k, v in all_params.items()}, n_subjects
