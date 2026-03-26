"""
Stan NUTS sampling for the RDU BHM with quantity-space Beta-Beta prior.

Compiles and runs rdu_bhm_ce.stan via CmdStanPy, then extracts posterior
samples into the preprocessed NPZ format expected by plotting.py and
paper_plots.py.
"""

import numpy as np
from pathlib import Path

from clean_data import load_clean_data
from sampling_ce import _pad_binary_to_3outcome


STAN_CONFIG_DEFAULT = {
    'chains': 4,
    'parallel_chains': 4,
    'iter_warmup': 1000,
    'iter_sampling': 2000,
    'thin': 1,
    'seed': 42,
    'max_treedepth': 12,
    'adapt_delta': 0.9,
}

QUANTITY_NAMES = ('ce1', 'ce2', 'ce3', 'pchoice')
HYPER_KEY_MAP = {0: 'ce1', 1: 'ce2', 2: 'ce3', 3: 'pc'}


def _logit(x):
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return np.log(x / (1 - x))


def prepare_stan_data(clean_data, ref_lotteries, ref_pair, prior_params_path):
    """Build a dict for CmdStanModel.sample(data=...)."""
    n_subjects = int(clean_data['n_subjects'])
    n_choices = int(clean_data['n_choices'])

    prizes_L = np.asarray(clean_data['prizes_L'])
    prizes_R = np.asarray(clean_data['prizes_R'])
    probs_L = np.asarray(clean_data['probs_L'])
    probs_R = np.asarray(clean_data['probs_R'])
    choices = np.asarray(clean_data['choices'], dtype=int)

    ref_lot_prizes = []
    ref_lot_probs = []
    for lot in ref_lotteries:
        prizes, probs = _pad_binary_to_3outcome(lot['prizes'], lot['probs'])
        ref_lot_prizes.append(prizes.tolist())
        ref_lot_probs.append(probs.tolist())

    pair_A_prizes, pair_A_probs = _pad_binary_to_3outcome(
        ref_pair['lottery_A']['prizes'], ref_pair['lottery_A']['probs'])
    pair_B_prizes, pair_B_probs = _pad_binary_to_3outcome(
        ref_pair['lottery_B']['prizes'], ref_pair['lottery_B']['probs'])

    pp = np.load(prior_params_path)

    return {
        'N_subjects': n_subjects,
        'N_choices': n_choices,
        'choices': choices.tolist(),
        'prizes_L': prizes_L.tolist(),
        'prizes_R': prizes_R.tolist(),
        'probs_L': probs_L.tolist(),
        'probs_R': probs_R.tolist(),
        'ref_lot_prizes': ref_lot_prizes,
        'ref_lot_probs': ref_lot_probs,
        'ref_pair_A_prizes': pair_A_prizes.tolist(),
        'ref_pair_A_probs': pair_A_probs.tolist(),
        'ref_pair_B_prizes': pair_B_prizes.tolist(),
        'ref_pair_B_probs': pair_B_probs.tolist(),
        'hp_a_mu': [float(pp[f'{q}_a_mu']) for q in QUANTITY_NAMES],
        'hp_b_mu': [float(pp[f'{q}_b_mu']) for q in QUANTITY_NAMES],
        'hp_a_psi': [float(pp[f'{q}_a_psi']) for q in QUANTITY_NAMES],
        'hp_b_psi': [float(pp[f'{q}_b_psi']) for q in QUANTITY_NAMES],
        'hp_lo': [float(pp[f'{q}_bound_lo']) for q in QUANTITY_NAMES],
        'hp_hi': [float(pp[f'{q}_bound_hi']) for q in QUANTITY_NAMES],
    }


def generate_stan_inits(n_subjects, n_chains=4, seed=42):
    """Generate initial values for each chain."""
    rng = np.random.default_rng(seed)
    inits = []
    for _ in range(n_chains):
        inits.append({
            'mu_pop': rng.beta(2, 2, size=4).tolist(),
            'psi_pop': np.clip(rng.beta(2, 20, size=4), 0.01, 0.5).tolist(),
            'r': rng.normal(0.5, 0.2, size=n_subjects).tolist(),
            'phi_raw': rng.normal(0.0, 0.2, size=n_subjects).tolist(),
            'eta_raw': rng.normal(0.0, 0.2, size=n_subjects).tolist(),
            'lambda_raw': rng.normal(-2.0, 0.3, size=n_subjects).tolist(),
        })
    return inits


def _get_var(fit, var_name, n_draws, n_chains):
    """Extract a Stan variable with shape (n_draws, n_chains, ...)."""
    val = fit.stan_variable(var_name)
    if val.shape[0] == n_draws * n_chains:
        extra = val.shape[1:]
        return val.reshape(n_draws, n_chains, *extra)
    return val


def extract_to_preprocessed(fit, n_subjects, preprocess_dir):
    """Convert CmdStanPy output to the preprocessed NPZ format."""
    preprocess_dir = Path(preprocess_dir)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir = preprocess_dir / 'subjects'
    subjects_dir.mkdir(parents=True, exist_ok=True)

    draws = fit.draws(concat_chains=False)
    n_draws, n_chains, _ = draws.shape

    # --- Hyperparameters (logit scale) ---
    mu_pop = _get_var(fit, 'mu_pop', n_draws, n_chains)
    psi_pop = _get_var(fit, 'psi_pop', n_draws, n_chains)

    hyper_data = {
        'iterations': np.arange(n_draws, dtype=np.int64),
        'batch_boundaries': np.array([0, n_draws], dtype=np.int64),
        'n_subjects': np.int64(n_subjects),
    }
    for k in range(4):
        key = HYPER_KEY_MAP[k]
        hyper_data[f'{key}_logit_mu'] = _logit(mu_pop[:, :, k]).astype(np.float32)
        hyper_data[f'{key}_logit_psi'] = _logit(psi_pop[:, :, k]).astype(np.float32)

    np.savez_compressed(
        preprocess_dir / 'hyperparams_preprocessed.npz', **hyper_data)

    # --- Subject parameters (natural scale) ---
    r_all = _get_var(fit, 'r', n_draws, n_chains)
    phi_all = _get_var(fit, 'phi', n_draws, n_chains)
    eta_all = _get_var(fit, 'eta', n_draws, n_chains)
    lam_all = _get_var(fit, 'lambda', n_draws, n_chains)

    for sid in range(n_subjects):
        np.savez_compressed(
            subjects_dir / f'subject_{sid:03d}_params.npz',
            r=r_all[:, :, sid].astype(np.float32),
            phi=phi_all[:, :, sid].astype(np.float32),
            eta=eta_all[:, :, sid].astype(np.float32),
            **{'lambda': lam_all[:, :, sid].astype(np.float32)},
        )

    np.savez(
        subjects_dir / 'metadata.npz',
        iterations=np.arange(n_draws, dtype=np.int64),
        batch_boundaries=np.array([0, n_draws], dtype=np.int64),
        n_subjects=np.int64(n_subjects),
    )

    print(f"  Saved preprocessed: {n_draws} draws x {n_chains} chains")
    print(f"    Hyperparams: {preprocess_dir / 'hyperparams_preprocessed.npz'}")
    print(f"    Subjects: {n_subjects} files in {subjects_dir}")


def run_sampling_stan(output_dir, ref_lotteries, ref_pair,
                      prior_params_path, config=None):
    """Compile and run the Stan model, then extract preprocessed output."""
    import cmdstanpy

    cfg = {**STAN_CONFIG_DEFAULT, **(config or {})}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile
    stan_file = Path(__file__).parent / 'rdu_bhm_ce.stan'
    print(f"  Compiling {stan_file.name}...")
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    print(f"  Compiled: {model.exe_file}")

    # Data
    print("  Preparing data...")
    clean_data = load_clean_data()
    stan_data = prepare_stan_data(
        clean_data, ref_lotteries, ref_pair, prior_params_path)
    n_subjects = stan_data['N_subjects']

    # Inits
    inits = generate_stan_inits(
        n_subjects, n_chains=cfg['chains'], seed=cfg['seed'])

    # Sample
    stan_output_dir = output_dir / 'stan_output'
    stan_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Sampling: {cfg['chains']} chains, "
          f"{cfg['iter_warmup']} warmup, {cfg['iter_sampling']} draws")
    fit = model.sample(
        data=stan_data,
        chains=cfg['chains'],
        parallel_chains=cfg.get('parallel_chains', cfg['chains']),
        iter_warmup=cfg['iter_warmup'],
        iter_sampling=cfg['iter_sampling'],
        thin=cfg.get('thin', 1),
        seed=cfg['seed'],
        max_treedepth=cfg.get('max_treedepth', 12),
        adapt_delta=cfg.get('adapt_delta', 0.9),
        inits=inits,
        output_dir=str(stan_output_dir),
        save_warmup=False,
    )

    # Diagnostics
    print("\n  Stan diagnostics:")
    print(fit.diagnose())

    # Extract
    print("\n  Extracting posterior samples...")
    extract_to_preprocessed(
        fit, n_subjects, output_dir / 'preprocessed')

    return fit
