"""
Main entry point for the BHM Risk Preferences project.

This is the hub script that controls which analyses are executed via boolean
flags. Implementation details are in the separate pipeline modules.

Usage: Set desired do_* flags to True, then run `python code/main.py`

Pipeline dependency chain:
  clean_data -> gen_paper_data -> generate_prior -> sampling
             -> gen_paper_quantities -> hyper_plots / subject_plots / paper_plots

Each step requires previous steps to have produced their outputs.
"""

# =============================================================================
# JAX Environment Configuration (must be set BEFORE importing jax/bamcmc)
# =============================================================================
import os

# Force unbuffered output (flush=True for all print statements)
os.environ['PYTHONUNBUFFERED'] = '1'
import functools
import builtins
builtins.print = functools.partial(builtins.print, flush=True)

# GPU memory preallocation:
#   'true'  = Preallocate ~90% GPU memory (faster, exclusive GPU use)
#   'false' = Allocate on-demand (allows GPU sharing, slightly slower)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' if True else 'false'

# GPU memory allocator (recommended for large models)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable 64-bit precision in JAX (required for int64 indices)
# Must be set BOTH as environment variable AND via JAX config to ensure it takes effect
os.environ['JAX_ENABLE_X64'] = 'true'

# JAX compilation cache directory (speeds up subsequent runs)
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.path.dirname(__file__) + '/jax_cache'

# Import JAX and set config (backup in case env var isn't picked up)
import jax
jax.config.update("jax_enable_x64", True)

# =============================================================================

import time
import sys
from pathlib import Path

# Track execution time
start_time = time.time()

# Setup paths
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
sys.path.insert(0, str(CODE_DIR))


# =============================================================================
# Path Configuration
# =============================================================================

PLOTS_DIR = PROJECT_ROOT / 'plots'

from helpers import OUTPUTS, check_dependency, get_mcmc_dir, get_plots_dir


# =============================================================================
# CE-Space Prior Configuration
# =============================================================================

# Reference lotteries for CE-space prior (Table 1 in paper)
# probs = [P(low outcome), P(high outcome)]
REFERENCE_LOTTERIES = [
    {'prizes': [10, 50], 'probs': [0.75, 0.25]},  # L1: ($10,$50), p_hi=0.25
    {'prizes': [20, 40], 'probs': [0.5, 0.5]},    # L2: ($20,$40), p_hi=0.50
    {'prizes': [12, 44], 'probs': [0.75, 0.25]},   # L3: ($12,$44), p_hi=0.25
]

# Reference pair for choice probability identification (L1 over L3)
# Same EV ($20), L1 riskier (wider spread). P(choose L1) > 0.5 iff risk-seeking.
REFERENCE_PAIR = {
    'lottery_A': {'prizes': [10, 50], 'probs': [0.75, 0.25]},  # L1
    'lottery_B': {'prizes': [12, 44], 'probs': [0.75, 0.25]},  # L3
}

# CE-space direct prior: Beta beliefs about CEs for reference lotteries
# and choice probability for lambda identification.
# CE beliefs are elicited as (mean_frac, p_risk_seeking):
#   mean_frac: mean CE as fraction of distance from lo to EV
#   p_risk_seeking: P(CE > EV) for a random individual
# These uniquely determine the Beta distribution (and therefore variance).
from prior_selection import build_ce_direct_prior
CE_MEAN_FRAC = 0.85        # mean CE = lo + 0.85 * (EV - lo)
CE_P_RISK_SEEKING = 0.15   # 15% chance a random subject is risk-seeking
DIRECT_PRIOR_CE = build_ce_direct_prior(
    reference_lotteries=REFERENCE_LOTTERIES,
    mean_frac=CE_MEAN_FRAC,
    p_risk_seeking=CE_P_RISK_SEEKING,
    pchoice_mu=0.40,         # mean P(choose riskier lottery)
)

# Researcher choices for CE-space variance decomposition
PRIOR_SELECTION_CE = {
    'alpha': 0.2,
    'kappa': 0.25,
}

# Injectivity analysis configuration (for Appendix C)
INJECTIVITY_CONFIG = {
    'lotteries': [
        {'prizes': (10, 30, 50), 'probs': (0.5, 0.0, 0.5)},   # L1: binary
        {'prizes': (10, 30, 50), 'probs': (0.1, 0.8, 0.1)},   # L2: concentrated middle
        {'prizes': (10, 30, 50), 'probs': (0.3, 0.3, 0.4)},   # L3: mildly right-skewed
    ],
    'domain': {
        'r': (0.01, 0.98),
        'phi': (0.2, 2.0),
        'eta': (0.2, 3.0),
    },
}

# CRRA+Power injectivity analysis configuration (Gram determinant, k=4 > p=3)
INJECTIVITY_POWER_CONFIG = {
    'domain': {
        'r': (0.01, 0.98),
        'delta': (0.1, 3.0),
    },
    'n_grid': 40,
    'x_min': 10.0,
    'x_max': 50.0,
}

# =============================================================================
# Dataset Configuration
# =============================================================================

# Which dataset to use: 'hn2016' (Harrison & Ng) or 'andersen2008' (Andersen et al.)
DATASET = 'hn2016'

# Random seeds for reproducibility
SEEDS = {
    'sampling': 42,
}

# Parallel processing
N_WORKERS = 25  # Number of parallel workers for plotting, etc.

# =============================================================================
# Plot and Sampling Configuration
# =============================================================================

# Diagnostic plots (subject and hyperparameter)
DIAGNOSTIC_PLOT_CONFIG = {
    'subject_trace_figsize': (12, 12),   # Figsize for subject trace plots
    'subject_ce_figsize': (10, 7),       # Figsize for subject CE plots
    'subject_utility_figsize': (10, 7),  # Figsize for subject utility spaghetti
    'n_spaghetti': 100,                  # Posterior draws for spaghetti plots
    'ce_hist_bins': 100,                 # Bins for subject CE histograms
    'chain_subsample': 50,              # Chains to show in subject traces
    'hyper_chain_subsample': 100,        # Chains to show in hyper traces
    'hyper_params_per_page': 5,          # Hyperparams per PDF page
    'hyper_row_height': 2.8,             # Height per row in inches (hyper plots)
}

# Sensitivity analysis grid
SENSITIVITY_CONFIG = {
    'alpha_values': [0.05, 0.1, 0.2, 0.3, 0.5],
    'kappa_values': [0.15, 0.25, 0.50, 1.00],
    'n_mc_draws': 50000,
    'n_subjects': 50,
}

# (alpha, kappa) sensitivity: full model re-estimation at alternative configs
RK_SENSITIVITY_CONFIGS = [
    (0.2, 1.50),   # same heterogeneity, much less confidence
    (0.8, 0.25),   # most variance from mean uncertainty, confident
    (0.8, 1.50),   # most variance from mean uncertainty, uncertain
]

sampling_opts_ce = {
    'model': 'rdu_bhm_ce',
    'output_dir': DATA_DIR / 'output',
    'plots_dir': PLOTS_DIR / 'rdu_bhm_ce',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 10,
    'seed': SEEDS['sampling'],
    'use_double': True,
    'n_temperatures': 0,
    'beta_min': 0.1,
}


# CRRA + Goldstein-Einhorn model config (same quantity-space Beta-Beta prior)
sampling_opts_ge = {
    'model': 'rdu_ge_bhm_ce',
    'output_dir': DATA_DIR / 'output',
    'plots_dir': PLOTS_DIR / 'rdu_ge_bhm_ce',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 10,
    'seed': SEEDS['sampling'],
    'use_double': True,
    'n_temperatures': 0,
    'beta_min': 0.1,
}

# Expo-Power + EUT model config (k=4 > p=3, same quantity-space Beta-Beta prior)
sampling_opts_ep_eut = {
    'model': 'rdu_ep_eut_bhm_ce',
    'output_dir': DATA_DIR / 'output',
    'plots_dir': PLOTS_DIR / 'rdu_ep_eut_bhm_ce',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 10,
    'seed': SEEDS['sampling'],
    'use_double': True,
    'n_temperatures': 0,
    'beta_min': 0.1,
}

# Expo-Power + TK92 model config (same quantity-space Beta-Beta prior)
sampling_opts_ep = {
    'model': 'rdu_ep_bhm_ce',
    'output_dir': DATA_DIR / 'output',
    'plots_dir': PLOTS_DIR / 'rdu_ep_bhm_ce',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 10,
    'seed': SEEDS['sampling'],
    'use_double': True,
    'n_temperatures': 0,
    'beta_min': 0.1,
}

# CRRA + Power model config (k=4 > p=3, same quantity-space Beta-Beta prior)
sampling_opts_power = {
    'model': 'rdu_power_bhm_ce',
    'output_dir': DATA_DIR / 'output',
    'plots_dir': PLOTS_DIR / 'rdu_power_bhm_ce',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 10,
    'seed': SEEDS['sampling'],
    'use_double': True,
    'n_temperatures': 0,
    'beta_min': 0.1,
}

# Stan/NUTS config (quantity-space Beta-Beta prior)
sampling_opts_stan = {
    'model': 'rdu_bhm_ce_stan',
    'output_dir': DATA_DIR / 'output',
    'chains': 8,
    'parallel_chains': 8,
    'iter_warmup': 1000,
    'iter_sampling': 2000,
    'max_treedepth': 12,
    'adapt_delta': 0.9,
    'seed': SEEDS['sampling'],
}

# Prior-only sampling config (no data likelihood)
sampling_opts_prior = {
    'output_dir': DATA_DIR / 'output',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 3,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 10,
    'seed': SEEDS['sampling'],
    'use_double': True,
}

# =============================================================================
# Control Flags
# =============================================================================

# --- Data Pipeline ---
do_clean_data     = False
do_gen_paper_data = False

# --- Prior Selection & Analysis ---
do_sensitivity          = False
do_family_comparison    = False

# --- Per-Model Flags (keyed by model registry key) ---
do_sampling       = {'crra': False, 'ge': False, 'ep_eut': False, 'ep': False, 'power': False}
do_sampling_prior = {'crra': False, 'ge': False, 'ep_eut': False, 'power': False}
do_prior_only_plots = {'crra': False, 'ge': False, 'ep_eut': False, 'power': False}
do_preprocessing  = {'crra': False, 'ge': False, 'ep_eut': False, 'ep': False, 'power': False}
do_diagnostic_plots = {'crra': False, 'ge': False, 'ep_eut': False, 'ep': False, 'power': False}

# --- Stan (separate, not in registry) ---
do_sampling_stan           = False

# --- Prior selection ---
do_generate_prior_ce       = False

# --- Paper Plots ---
do_paper_predictive_overlay      = False
do_paper_prior_predictive_params = False
do_subject_ce_plots              = False

# --- Diagnostics ---
do_mcmc_diagnostics        = False

# --- Marginal likelihoods (corrected RB estimator) ---
do_ml_validation           = False
do_bayes_factors_rb2       = False

# --- Appendix Analyses ---
do_rk_sensitivity          = False
do_injectivity_analysis    = False
do_injectivity_power       = False

# --- Document Compilation ---
do_compile_doc          = False


# =============================================================================
# Run Selected Pipelines
# =============================================================================

def _load_reference_prizes():
    """Load min/mid/max prizes from clean data for CE plots."""
    import numpy as np
    from clean_data import load_clean_data
    clean = load_clean_data()
    all_prizes = np.concatenate([
        np.asarray(clean['prizes_L']).ravel(),
        np.asarray(clean['prizes_R']).ravel(),
    ])
    p_min = float(np.min(all_prizes))
    p_max = float(np.max(all_prizes))
    p_mid = (p_min + p_max) / 2.0
    return np.array([p_min, p_mid, p_max])


if __name__ == '__main__':
    import pipelines

    # -------------------------------------------------------------------------
    # Data Pipeline
    # -------------------------------------------------------------------------
    if do_clean_data:
        pipelines.run_clean_data(DATA_DIR, dataset=DATASET)

    if do_gen_paper_data:
        check_dependency('gen_paper_data', ['clean_data'])
        pipelines.run_gen_paper_data(
            DATA_DIR,
            ce_direct_prior=DIRECT_PRIOR_CE,
            reference_lotteries=REFERENCE_LOTTERIES,
            reference_pair=REFERENCE_PAIR,
            prior_selection_ce=PRIOR_SELECTION_CE,
            ce_mean_frac=CE_MEAN_FRAC,
            ce_p_risk_seeking=CE_P_RISK_SEEKING,
        )

    # -------------------------------------------------------------------------
    # Prior Selection and Sensitivity Analysis
    # -------------------------------------------------------------------------
    if do_sensitivity:
        pipelines.run_sensitivity(
            ce_direct_prior=DIRECT_PRIOR_CE,
            output_dir=DATA_DIR / 'output',
            lottery=REFERENCE_LOTTERIES[0],  # L1 (for caption)
            seed=SEEDS['sampling'],
            config=SENSITIVITY_CONFIG,
        )

    if do_family_comparison:
        from family_comparison import run_family_comparison
        run_family_comparison(
            output_dir=DATA_DIR / 'output',
            plot_dir=PLOTS_DIR / 'paper',
            n_workers=N_WORKERS,
        )

    # -------------------------------------------------------------------------
    # CE-Space Prior Selection and Sampling
    # -------------------------------------------------------------------------
    if do_generate_prior_ce:
        pipelines.run_prior_selection_ce(
            DATA_DIR,
            ce_direct_prior=DIRECT_PRIOR_CE,
            prior_selection_ce=PRIOR_SELECTION_CE,
        )

    # =========================================================================
    # Per-Model Pipeline (sampling, preprocessing, diagnostic plots)
    # =========================================================================
    from model_registry import MODEL_REGISTRY, get_data_dir, get_history_dir, get_preproc_dir
    from model_registry import get_plots_dir as get_model_plots_dir

    sampling_opts_map = {
        'crra': sampling_opts_ce,
        'ge': sampling_opts_ge,
        'ep_eut': sampling_opts_ep_eut,
        'ep': sampling_opts_ep,
        'power': sampling_opts_power,
    }
    prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'

    for model_key, cfg in MODEL_REGISTRY.items():
        # --- Sampling ---
        if do_sampling.get(model_key, False):
            check_dependency(f'sampling_{model_key}', ['clean_data'])
            pipelines.run_sampling_model(
                model_key, sampling_opts_map[model_key],
                ref_lotteries=REFERENCE_LOTTERIES,
                ref_pair=REFERENCE_PAIR,
                prior_params_path=prior_params_ce_path,
            )

        # --- Prior-only sampling ---
        if do_sampling_prior.get(model_key, False) and cfg.get('has_prior_only', False):
            check_dependency(f'sampling_{model_key}_prior', ['clean_data'])
            pipelines.run_sampling_prior(
                model_key, sampling_opts_map[model_key],
                ref_lotteries=REFERENCE_LOTTERIES,
                ref_pair=REFERENCE_PAIR,
                prior_params_path=prior_params_ce_path,
            )

        # --- Preprocessing ---
        if do_preprocessing.get(model_key, False):
            pipelines.run_preprocess_model(model_key)

        # --- Prior-only preprocessing and plots ---
        if do_prior_only_plots.get(model_key, False) and cfg.get('has_prior_only', False):
            prior_model_id = cfg['prior_model_id']
            prior_data_dir = DATA_DIR / 'output' / prior_model_id
            prior_history_dir = prior_data_dir / 'history' / 'full'
            prior_preproc_dir = prior_data_dir / 'preprocessed'
            prior_plots_dir = PLOTS_DIR / prior_model_id

            if prior_history_dir.exists():
                from plotting import (preprocess_hyperparams_ce,
                                      preprocess_subjects,
                                      plot_hyperparams_ce_from_preprocessed,
                                      parallel_plot_subjects)

                print(f"\n  Preprocessing prior-only: {cfg['name']}")
                preprocess_hyperparams_ce(
                    prior_history_dir, prior_preproc_dir, force=True,
                    nparams=cfg['nparams'], n_hyper=cfg['n_hyper'])
                preprocess_subjects(
                    prior_history_dir, prior_preproc_dir / 'subjects', force=True,
                    nparams=cfg['nparams'], n_hyper=cfg['n_hyper'],
                    param_names=cfg['param_names'])

                print(f"  Plotting prior-only hyperparams: {cfg['name']}")
                hyper_preproc = prior_preproc_dir / 'hyperparams_preprocessed.npz'
                if hyper_preproc.exists():
                    plot_hyperparams_ce_from_preprocessed(
                        str(hyper_preproc),
                        str(prior_plots_dir / 'hyperparams'))

                print(f"  Plotting prior-only subjects: {cfg['name']}")
                subjects_dir = prior_preproc_dir / 'subjects'
                if subjects_dir.exists() and (subjects_dir / 'metadata.npz').exists():
                    import numpy as _np
                    meta = _np.load(subjects_dir / 'metadata.npz')
                    pipelines.run_subject_plots(
                        prior_history_dir,
                        prior_plots_dir,
                        n_workers=1,
                        preprocessed_dir=subjects_dir,
                        config=DIAGNOSTIC_PLOT_CONFIG)
            else:
                print(f"  Skipping prior-only plots for {cfg['name']}: no history files")

        # --- Diagnostic plots ---
        if do_diagnostic_plots.get(model_key, False):
            subjects_dir = get_preproc_dir(model_key) / 'subjects'
            model_plots_dir = get_model_plots_dir(model_key)
            history_dir = get_history_dir(model_key)
            if subjects_dir.exists() and (subjects_dir / 'metadata.npz').exists():
                # Only pass prizes for CRRA model (utility/CE pages are CRRA-specific)
                prizes = _load_reference_prizes() if cfg['utility_type'] == 'crra' and cfg['pwf_type'] == 'prelec' else None
                pipelines.run_subject_plots(history_dir, model_plots_dir,
                                            n_workers=N_WORKERS,
                                            preprocessed_dir=subjects_dir,
                                            prizes=prizes,
                                            config=DIAGNOSTIC_PLOT_CONFIG)
            else:
                print(f"Skipping {cfg['name']} subject plots: preprocessed data not found")

    # --- Stan (not in registry) ---
    if do_sampling_stan:
        check_dependency('sampling_stan', ['clean_data'])
        pipelines.run_sampling_stan(
            sampling_opts_stan,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    # -------------------------------------------------------------------------
    # MCMC Diagnostics
    # -------------------------------------------------------------------------
    if do_mcmc_diagnostics:
        from mcmc_diagnostics import run_all_diagnostics
        run_all_diagnostics(DATA_DIR / 'output' / 'mcmc_diagnostics.npz')

    if do_ml_validation:
        from ml_validation import run_ml_validation
        run_ml_validation(
            data_dir=DATA_DIR / 'output',
            output_path=DATA_DIR / 'output' / 'ml_validation.npz',
        )

    if do_bayes_factors_rb2:
        from marginal_likelihood import compute_marginal_likelihoods
        compute_marginal_likelihoods(
            model_keys=['crra', 'ge', 'power'],
            data_dir=DATA_DIR / 'output',
            output_path=DATA_DIR / 'output' / 'bayes_factors_rb2.npz',
            seed=SEEDS['sampling'],
        )

    # -------------------------------------------------------------------------
    # Paper Plots
    # -------------------------------------------------------------------------
    paper_plots_dir = PLOTS_DIR / 'paper'

    if do_paper_predictive_overlay:
        from paper_plots import generate_cross_model_ce_overlay, generate_subject_detail_figures

        # Build hyper paths from registry
        hyper_paths = {}
        subj_dirs = {}
        for mk in MODEL_REGISTRY:
            hp = get_preproc_dir(mk) / 'hyperparams_preprocessed.npz'
            hyper_paths[mk] = str(hp) if hp.exists() else None
            sd = get_preproc_dir(mk) / 'subjects'
            subj_dirs[mk] = str(sd) if sd.exists() else None

        if hyper_paths.get('crra'):
            generate_cross_model_ce_overlay(
                crra_hyper_path=hyper_paths['crra'],
                ge_hyper_path=hyper_paths.get('ge'),
                ep_hyper_path=hyper_paths.get('ep'),
                power_hyper_path=hyper_paths.get('power'),
                prior_params_path=str(DATA_DIR / 'output' / 'prior_params_ce.npz'),
                output_dir=str(paper_plots_dir),
                seed=SEEDS['sampling'],
            )

        if subj_dirs.get('crra'):
            generate_subject_detail_figures(
                crra_subjects_dir=subj_dirs['crra'],
                ge_subjects_dir=subj_dirs.get('ge'),
                ep_subjects_dir=subj_dirs.get('ep'),
                power_subjects_dir=subj_dirs.get('power'),
                output_dir=str(paper_plots_dir),
                seed=SEEDS['sampling'],
            )
        else:
            print("Skipping subject detail figures: not all models preprocessed")

    if do_subject_ce_plots:
        import numpy as np
        from gen_subject_ce_plots import plot_one_subject, MODELS as CE_PLOT_MODELS
        from concurrent.futures import ProcessPoolExecutor
        for model_key, name, model_id, param_names, color in CE_PLOT_MODELS:
            output_dir = PLOTS_DIR / model_id / 'subject_ces'
            output_dir.mkdir(parents=True, exist_ok=True)
            preproc = DATA_DIR / 'output' / model_id / 'preprocessed' / 'subjects'
            if not (preproc / 'metadata.npz').exists():
                print(f"Skipping {name} CE plots: preprocessed data not found")
                continue
            meta = np.load(preproc / 'metadata.npz')
            n_subj = int(meta['n_subjects'])
            print(f"\n{name}: generating {n_subj} subject CE plots...")
            args_list = [(sid, model_key, name, model_id, param_names, color)
                         for sid in range(n_subj)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                done = list(pool.map(plot_one_subject, args_list))
            print(f"  Saved {len(done)} PDFs to {output_dir}")

    if do_paper_prior_predictive_params:
        prior_subjects = preprocess_dir_prior / 'subjects'
        if prior_subjects.exists():
            pipelines.run_prior_predictive_params_plot(
                prior_subjects_dir=str(prior_subjects),
                output_path=str(paper_plots_dir / 'prior_predictive_params.pdf'),
            )
        else:
            print(f"Skipping prior predictive params plot: {prior_subjects} not found")

    # -------------------------------------------------------------------------
    # Appendix Analyses
    # -------------------------------------------------------------------------
    if do_rk_sensitivity:
        pipelines.run_rk_sensitivity(
            sensitivity_configs=RK_SENSITIVITY_CONFIGS,
            direct_prior_ce=DIRECT_PRIOR_CE,
            reference_lotteries=REFERENCE_LOTTERIES,
            reference_pair=REFERENCE_PAIR,
            sampling_opts=sampling_opts_ce,
            seeds=SEEDS,
            data_dir=DATA_DIR,
            plots_dir=PLOTS_DIR,
        )

    if do_injectivity_analysis:
        pipelines.run_injectivity_analysis(
            output_dir=PLOTS_DIR / 'injectivity',
            config=INJECTIVITY_CONFIG,
        )

    if do_injectivity_power:
        pipelines.run_injectivity_power(
            output_dir=PLOTS_DIR / 'injectivity',
            reference_lotteries=REFERENCE_LOTTERIES,
            config=INJECTIVITY_POWER_CONFIG,
        )

    # -------------------------------------------------------------------------
    # Document Compilation
    # -------------------------------------------------------------------------
    if do_compile_doc:
        print("\n" + "=" * 60)
        print("Running: Compile LaTeX Document")
        print("=" * 60)
        import subprocess
        doc_dir = PROJECT_ROOT / 'doc'
        # Full build: pdflatex -> pythontex -> pdflatex -> biber -> pdflatex
        pylib_python = str(PROJECT_ROOT / 'pylib' / 'bin' / 'python')
        pythontex_script = '/usr/share/texmf-dist/scripts/pythontex/pythontex3.py'
        commands = [
            ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
            [pylib_python, pythontex_script, f'--interpreter=python:{pylib_python}', 'main.tex'],
            ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
            ['/usr/bin/vendor_perl/biber', 'main'],
            ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
        ]
        import shutil
        for cmd in commands:
            # Skip commands that aren't installed
            if shutil.which(cmd[0]) is None:
                print(f"  Skipping (not installed): {cmd[0]}")
                continue
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=doc_dir, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='replace')[:500]
                print(f"  Warning: {cmd[0]} returned exit code {result.returncode}")
                if stderr:
                    print(f"  stderr: {stderr}")
        print("Done.\n")

    # -------------------------------------------------------------------------
    # Wrap up
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
