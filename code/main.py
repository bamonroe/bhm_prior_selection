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
    'rho': 0.2,
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
    'rho_values': [0.05, 0.1, 0.2, 0.3, 0.5],
    'kappa_values': [0.15, 0.25, 0.50, 1.00],
    'n_mc_draws': 50000,
    'n_subjects': 50,
}

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
    'chunk_size': 100,
    'seed': SEEDS['sampling'],
    'n_temperatures': 0,
    'beta_min': 0.1,
}

# Uninformative NIG model config (Gao 2022 diffuse prior)
sampling_opts_uninf = {
    'model': 'rdu_bhm',
    'output_dir': DATA_DIR / 'output',
    'plots_dir': PLOTS_DIR / 'rdu_bhm_uninf',
    'clean': True,
    'reset_runs': 1,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'chunk_size': 100,
    'seed': SEEDS['sampling'],
    'n_temperatures': 0,
    'beta_0_min': 0.1,
}

# Gao (2022) diffuse hyperprior: N(0, 100) means, IG(0.001, 0.001) variances
UNINF_HYPERPRIOR_SPECS = {
    'r':          {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.001, 'beta_0': 0.001},
    'phi_raw':    {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.001, 'beta_0': 0.001},
    'eta_raw':    {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.001, 'beta_0': 0.001},
    'lambda_raw': {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.001, 'beta_0': 0.001},
}

# =============================================================================
# Control Flags
# =============================================================================

# --- Data Pipeline ---
do_clean_data     = True
do_gen_paper_data = True

# --- Prior Selection & Analysis ---
do_sensitivity          = False

# --- MCMC Sampling ---
do_sampling_ce             = True
do_sampling_uninf          = True

# --- Preprocessing (extract from history files) ---
do_generate_prior_ce       = True
do_preprocess_hyper_ce     = True
do_preprocess_subjects_ce  = True
do_preprocess_hyper_uninf  = True
do_preprocess_subjects_uninf = True

# --- Diagnostic Plots ---
do_hyper_plots_ce          = False
do_subject_plots_ce        = False

# --- Paper Comparison Plots ---
do_paper_predictive_overlay = True
do_paper_subject_ce         = False
do_paper_subject_ce_all     = True

# --- Appendix Examples ---
do_injectivity_analysis    = False

# --- Document Compilation ---
do_compile_doc          = True


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
    # Uninformative NIG Model (Gao 2022 diffuse prior)
    # -------------------------------------------------------------------------
    uninf_output_dir = DATA_DIR / 'output' / 'rdu_bhm_uninf'
    history_dir_uninf = uninf_output_dir / 'rdu_bhm' / 'history' / 'full'
    preprocess_dir_uninf = uninf_output_dir / 'preprocessed'

    if do_sampling_uninf:
        # Set diffuse hyperpriors before sampling
        from posteriors.rdu_bhm import HYPERPRIOR_SPECS, PARAM_NAMES
        for pname in PARAM_NAMES:
            HYPERPRIOR_SPECS[pname] = UNINF_HYPERPRIOR_SPECS[pname]
        from sampling import run_sampling as run_sampling_nig
        run_sampling_nig(
            output_dir=uninf_output_dir,
            reset_runs=sampling_opts_uninf.get('reset_runs', 1),
            resume_runs=sampling_opts_uninf.get('resume_runs', 5),
            num_collect=sampling_opts_uninf.get('num_collect', 100),
            config={
                'num_chains_a': sampling_opts_uninf['num_chains'] // 2,
                'num_chains_b': sampling_opts_uninf['num_chains'] // 2,
                'num_collect': sampling_opts_uninf['num_collect'],
                'thin_iteration': sampling_opts_uninf['thin_iteration'],
                'chunk_size': sampling_opts_uninf['chunk_size'],
                'rng_seed': sampling_opts_uninf['seed'],
                'n_temperatures': sampling_opts_uninf.get('n_temperatures', 1),
                'beta_min': sampling_opts_uninf.get('beta_0_min', 0.1),
            },
        )

    if do_preprocess_hyper_uninf:
        if history_dir_uninf.exists():
            pipelines.run_preprocess_hyperparams(history_dir_uninf, preprocess_dir_uninf)
        else:
            print(f"Skipping uninf preprocessing: {history_dir_uninf} not found")

    if do_preprocess_subjects_uninf:
        if history_dir_uninf.exists():
            pipelines.run_preprocess_subjects(history_dir_uninf, preprocess_dir_uninf / 'subjects')
        else:
            print(f"Skipping uninf subject preprocessing: {history_dir_uninf} not found")

    # -------------------------------------------------------------------------
    # Paper Comparison Plots
    # -------------------------------------------------------------------------
    hyper_preprocessed_info = preprocess_dir_ce / 'hyperparams_preprocessed.npz'
    hyper_preprocessed_uninf = preprocess_dir_uninf / 'hyperparams_preprocessed.npz'
    paper_plots_dir = PLOTS_DIR / 'paper'

    if do_paper_predictive_overlay:
        info_subjects = preprocess_dir_ce / 'subjects'
        uninf_subjects = preprocess_dir_uninf / 'subjects'
        if info_subjects.exists() and uninf_subjects.exists():
            from paper_plots import generate_paper_plots
            generate_paper_plots(
                info_subjects_dir=str(info_subjects),
                uninf_subjects_dir=str(uninf_subjects),
                info_hyper_path=str(hyper_preprocessed_info),
                uninf_hyper_path=str(hyper_preprocessed_uninf),
                prior_params_path=str(DATA_DIR / 'output' / 'prior_params_ce.npz'),
                output_dir=str(paper_plots_dir),
                data_output_dir=str(DATA_DIR / 'output'),
                seed=SEEDS['sampling'],
            )
        else:
            missing = []
            if not info_subjects.exists():
                missing.append('informative subjects')
            if not uninf_subjects.exists():
                missing.append('uninformative subjects')
            print(f"Skipping paper plots: missing {', '.join(missing)}")

    if do_paper_subject_ce_all:
        print("\n" + "=" * 60)
        print("Running: Per-Subject CE Comparison Plots (All Subjects)")
        print("=" * 60)
        info_subjects = preprocess_dir_ce / 'subjects'
        uninf_subjects = preprocess_dir_uninf / 'subjects'
        if info_subjects.exists() and uninf_subjects.exists():
            from paper_plots import generate_subject_ce_plots
            generate_subject_ce_plots(
                info_subjects_dir=str(info_subjects),
                uninf_subjects_dir=str(uninf_subjects),
                output_dir=str(PLOTS_DIR / 'paper' / 'subject_ce'),
                lotteries=REFERENCE_LOTTERIES,
                n_sub=5000,
                n_workers=N_WORKERS,
                seed=SEEDS['sampling'],
            )
        else:
            missing = []
            if not info_subjects.exists():
                missing.append('informative subjects')
            if not uninf_subjects.exists():
                missing.append('uninformative subjects')
            print(f"Skipping subject CE plots: missing {', '.join(missing)}")
        print("Done.\n")

    # -------------------------------------------------------------------------
    # Appendix Examples
    # -------------------------------------------------------------------------
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
