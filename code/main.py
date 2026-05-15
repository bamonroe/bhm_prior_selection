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

# --- MCMC Sampling ---
do_sampling_ce             = False
do_sampling_ep             = False
do_sampling_power          = False
do_sampling_stan           = False
do_sampling_prior          = False
do_sampling_power_prior    = False

# --- Preprocessing (extract from history files) ---
do_generate_prior_ce       = False
do_preprocess_hyper_ce     = False
do_preprocess_subjects_ce  = False
do_preprocess_hyper_ep     = False
do_preprocess_subjects_ep  = False
do_preprocess_hyper_power  = False
do_preprocess_subjects_power = False
do_preprocess_subjects_prior = False
do_preprocess_subjects_power_prior = False

# --- Diagnostic Plots ---
do_hyper_plots_ce          = False
do_subject_plots_ce        = False

# --- Paper Plots ---
do_paper_predictive_overlay      = False
do_paper_prior_predictive_params = False

# --- Appendix Analyses ---
do_rk_sensitivity          = False
do_injectivity_analysis    = False

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

    if do_sampling_ce:
        check_dependency('sampling_ce', ['clean_data'])
        prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'
        pipelines.run_sampling_ce(
            sampling_opts_ce,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    if do_sampling_ep:
        check_dependency('sampling_ep', ['clean_data'])
        prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'
        pipelines.run_sampling_ep(
            sampling_opts_ep,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    if do_sampling_power:
        check_dependency('sampling_power', ['clean_data'])
        prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'
        pipelines.run_sampling_power(
            sampling_opts_power,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    if do_sampling_power_prior:
        check_dependency('sampling_power_prior', ['clean_data'])
        prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'
        pipelines.run_sampling_power_prior(
            sampling_opts_power,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    if do_sampling_stan:
        check_dependency('sampling_stan', ['clean_data'])
        prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'
        pipelines.run_sampling_stan(
            sampling_opts_stan,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    if do_sampling_prior:
        check_dependency('sampling_prior', ['clean_data'])
        prior_params_ce_path = DATA_DIR / 'output' / 'prior_params_ce.npz'
        pipelines.run_sampling_prior(
            sampling_opts_prior,
            ref_lotteries=REFERENCE_LOTTERIES,
            ref_pair=REFERENCE_PAIR,
            prior_params_path=prior_params_ce_path,
        )

    # -------------------------------------------------------------------------
    # Preprocessing (CE-Space Prior)
    # -------------------------------------------------------------------------
    history_dir_ce = get_mcmc_dir('rdu_bhm_ce') / 'history' / 'full'
    preprocess_dir_ce = get_mcmc_dir('rdu_bhm_ce') / 'preprocessed'

    if do_preprocess_hyper_ce:
        if history_dir_ce.exists():
            pipelines.run_preprocess_hyperparams_ce(history_dir_ce, preprocess_dir_ce)
        else:
            print(f"Skipping CE preprocessing: {history_dir_ce} not found")

    if do_preprocess_subjects_ce:
        if history_dir_ce.exists():
            pipelines.run_preprocess_subjects_ce(history_dir_ce, preprocess_dir_ce / 'subjects')
        else:
            print(f"Skipping CE subject preprocessing: {history_dir_ce} not found")

    # -------------------------------------------------------------------------
    # Preprocessing (EP+TK Model)
    # -------------------------------------------------------------------------
    history_dir_ep = get_mcmc_dir('rdu_ep_bhm_ce') / 'history' / 'full'
    preprocess_dir_ep = get_mcmc_dir('rdu_ep_bhm_ce') / 'preprocessed'

    if do_preprocess_hyper_ep:
        if history_dir_ep.exists():
            pipelines.run_preprocess_hyperparams_ce(history_dir_ep, preprocess_dir_ep)
        else:
            print(f"Skipping EP preprocessing: {history_dir_ep} not found")

    if do_preprocess_subjects_ep:
        if history_dir_ep.exists():
            pipelines.run_preprocess_subjects_ce(history_dir_ep, preprocess_dir_ep / 'subjects')
        else:
            print(f"Skipping EP subject preprocessing: {history_dir_ep} not found")

    # -------------------------------------------------------------------------
    # Preprocessing (CRRA+Power Model)
    # -------------------------------------------------------------------------
    history_dir_power = get_mcmc_dir('rdu_power_bhm_ce') / 'history' / 'full'
    preprocess_dir_power = get_mcmc_dir('rdu_power_bhm_ce') / 'preprocessed'

    if do_preprocess_hyper_power:
        if history_dir_power.exists():
            from posteriors.rdu_power_bhm_ce import NPARAMS as NP_POWER, N_HYPER as NH_POWER
            pipelines.run_preprocess_hyperparams_ce(
                history_dir_power, preprocess_dir_power,
                nparams=NP_POWER, n_hyper=NH_POWER)
        else:
            print(f"Skipping Power preprocessing: {history_dir_power} not found")

    if do_preprocess_subjects_power:
        if history_dir_power.exists():
            from posteriors.rdu_power_bhm_ce import NPARAMS as NP_POWER, N_HYPER as NH_POWER
            pipelines.run_preprocess_subjects_ce(
                history_dir_power, preprocess_dir_power / 'subjects',
                nparams=NP_POWER, n_hyper=NH_POWER,
                param_names=['r', 'delta', 'lambda'])
        else:
            print(f"Skipping Power subject preprocessing: {history_dir_power} not found")

    # -------------------------------------------------------------------------
    # Preprocessing (CRRA+Power Prior-Only)
    # -------------------------------------------------------------------------
    history_dir_power_prior = get_mcmc_dir('rdu_power_bhm_ce_prior') / 'history' / 'full'
    preprocess_dir_power_prior = get_mcmc_dir('rdu_power_bhm_ce_prior') / 'preprocessed'

    if do_preprocess_subjects_power_prior:
        if history_dir_power_prior.exists():
            from posteriors.rdu_power_bhm_ce import NPARAMS as NP_POWER, N_HYPER as NH_POWER
            pipelines.run_preprocess_subjects_ce(
                history_dir_power_prior, preprocess_dir_power_prior / 'subjects',
                nparams=NP_POWER, n_hyper=NH_POWER,
                param_names=['r', 'delta', 'lambda'])
        else:
            print(f"Skipping Power prior subject preprocessing: {history_dir_power_prior} not found")

    # -------------------------------------------------------------------------
    # Preprocessing (Prior-Only Model)
    # -------------------------------------------------------------------------
    history_dir_prior = get_mcmc_dir('rdu_bhm_ce_prior') / 'history' / 'full'
    preprocess_dir_prior = get_mcmc_dir('rdu_bhm_ce_prior') / 'preprocessed'

    if do_preprocess_subjects_prior:
        if history_dir_prior.exists():
            pipelines.run_preprocess_subjects_prior(
                history_dir_prior, preprocess_dir_prior / 'subjects')
        else:
            print(f"Skipping prior subject preprocessing: {history_dir_prior} not found")

    # -------------------------------------------------------------------------
    # Plotting (CE-Space Prior)
    # -------------------------------------------------------------------------
    plots_dir_ce = get_plots_dir('rdu_bhm_ce')

    if do_hyper_plots_ce:
        hyper_preprocessed_ce = preprocess_dir_ce / 'hyperparams_preprocessed.npz'
        if hyper_preprocessed_ce.exists():
            pipelines.run_hyperparameter_plots_ce(hyper_preprocessed_ce, plots_dir_ce,
                                                   ce_direct_prior=DIRECT_PRIOR_CE,
                                                   config=DIAGNOSTIC_PLOT_CONFIG)
        else:
            print(f"Skipping CE hyper plots: {hyper_preprocessed_ce} not found")

    if do_subject_plots_ce:
        subjects_preprocessed_ce = preprocess_dir_ce / 'subjects'
        if subjects_preprocessed_ce.exists() and (subjects_preprocessed_ce / 'metadata.npz').exists():
            ref_prizes_ce = _load_reference_prizes()
            pipelines.run_subject_plots(history_dir_ce, plots_dir_ce,
                                        n_workers=N_WORKERS,
                                        preprocessed_dir=subjects_preprocessed_ce,
                                        prizes=ref_prizes_ce,
                                        config=DIAGNOSTIC_PLOT_CONFIG)
        else:
            print(f"Skipping CE subject plots: preprocessed data not found")

    # -------------------------------------------------------------------------
    # Paper Plots
    # -------------------------------------------------------------------------
    paper_plots_dir = PLOTS_DIR / 'paper'

    if do_paper_predictive_overlay:
        from paper_plots import generate_cross_model_ce_overlay
        crra_hyper = preprocess_dir_ce / 'hyperparams_preprocessed.npz'
        ep_hyper = preprocess_dir_ep / 'hyperparams_preprocessed.npz'
        power_hyper = preprocess_dir_power / 'hyperparams_preprocessed.npz'
        if crra_hyper.exists():
            generate_cross_model_ce_overlay(
                crra_hyper_path=str(crra_hyper),
                ep_hyper_path=str(ep_hyper) if ep_hyper.exists() else None,
                power_hyper_path=str(power_hyper) if power_hyper.exists() else None,
                prior_params_path=str(DATA_DIR / 'output' / 'prior_params_ce.npz'),
                output_dir=str(paper_plots_dir),
                seed=SEEDS['sampling'],
            )
        else:
            print(f"Skipping cross-model overlay: CRRA hyper not found")

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
