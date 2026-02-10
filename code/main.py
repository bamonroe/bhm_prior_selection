"""
Main entry point for the BHM Risk Preferences project.

This is the hub script that controls which analyses are executed via boolean
flags. Implementation details are in the separate pipeline modules.

Usage: Set desired do_* flags to True, then run `python code/main.py`

Pipeline dependency chain:
  clean_data -> gen_paper_data -> generate_prior -> sampling
             -> gen_paper_quantities -> diagnostic_plots / paper_plots

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
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' if False else 'false'

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
# Prior Selection Configuration
# =============================================================================

# Direct Prior for RDU-BHM Model
# This defines the prior distribution for a single random individual.
# The hyperprior derivation ensures the unconditional BHM distribution
# matches this direct prior exactly.
#
# Parameters are specified as (mean, variance) pairs:
# - r: CRRA coefficient (untransformed)
# - phi_raw: log(phi) where phi is Prelec curvature
# - eta_raw: log(eta) where eta is Prelec elevation
# - lambda_raw: log(lambda) where lambda is Fechner noise
DIRECT_PRIOR = {
    'r': {
        'mu': 0.5,       # Mild risk aversion
        'sigma2': 0.25,  # sd=0.5, 95% CI roughly (-0.5, 1.5)
    },
    'phi_raw': {
        'mu': 0.0,       # phi=1 (no curvature distortion)
        'sigma2': 0.15,  # sd=0.3, phi 95% CI roughly (0.55, 1.8)
    },
    'eta_raw': {
        'mu': 0.0,       # eta=1 (no elevation distortion)
        'sigma2': 0.15,  # sd=0.3, eta 95% CI roughly (0.55, 1.8)
    },
    'lambda_raw': {
        'mu': -2.0,      # lambda ~ 0.14 (moderate noise)
        'sigma2': 1.0,   # sd=1.0, lambda 95% CI roughly (0.02, 1.0)
    },
}

# BHM Hyperprior researcher choices:
# - rho: Variance partition ratio (fraction from mean uncertainty vs population spread)
#        0.3 means 30% from hyperprior mean uncertainty, 70% from population variance
# - alpha_0: InverseGamma shape parameter (higher = tighter variance prior)
#            CV[sigma^2] = 1/sqrt(alpha_0-2)
PRIOR_SELECTION = {
    'rho': 0.2,
    'alpha_0': 20.0,
}

# Uninformative (diffuse) hyperprior for comparison
# This directly specifies hyperprior parameters, NOT a direct prior to be transformed.
# - mu_0: prior mean of population mean (centered at neutral values)
# - tau_0_sq: prior variance of population mean (very large = uninformative about location)
# - alpha_0, beta_0: InverseGamma params for population variance
#   Small values (0.1, 0.1) give diffuse prior on sigma^2
UNINFORMATIVE_HYPERPRIOR = {
    'r': {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.1, 'beta_0': 0.1},
    'phi_raw': {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.1, 'beta_0': 0.1},
    'eta_raw': {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.1, 'beta_0': 0.1},
    'lambda_raw': {'mu_0': 0.0, 'tau_0_sq': 100.0, 'alpha_0': 0.1, 'beta_0': 0.1},
}

# Skew-Normal example configuration (for Appendix B)
# Direct prior: SN(ξ*, ω*, α*) - the target unconditional distribution
SKEW_NORMAL_DIRECT_PRIOR = {
    'xi': 0.0,      # location parameter
    'omega': 1.0,   # scale parameter
    'alpha': 2.0,   # shape parameter (positive = right skew)
}

# Researcher choices for Skew-Normal (3 degrees of freedom)
SKEW_NORMAL_RESEARCHER = {
    'rho': 0.2,           # variance partition (same interpretation as Normal case)
    'alpha_omega': 20.0,   # scale concentration (analogous to alpha_0)
    'tau_alpha_sq': 0.25, # shape uncertainty (new degree of freedom)
}

# Random seeds for reproducibility
SEEDS = {
    'sampling': 42,
}

# Parallel processing
N_WORKERS = 25  # Number of parallel workers for plotting, etc.

# =============================================================================
# Sampling Options
# =============================================================================

sampling_opts = {
    'model': 'rdu_bhm',
    'output_dir': DATA_DIR / 'output',  # bamcmc creates {model}/ subdirectory
    'plots_dir': PLOTS_DIR / 'rdu_bhm',
    'clean': True,
    'reset_runs': 0,
    'resume_runs': 5,
    'num_chains': 500,
    'num_collect': 100,
    'thin_iteration': 1,
    'chunk_size': 100,  # Iterations per compiled chunk (1-10000)
    'seed': SEEDS['sampling'],
}

# =============================================================================
# Control Flags
# =============================================================================

# --- Data Pipeline ---
do_clean_data     = False
do_gen_paper_data = False

# --- Prior Selection and Plots ---
do_generate_prior       = False
do_prior_plots          = False
do_prior_predictive_ce  = False

# --- MCMC Sampling (Informative Prior) ---
do_sampling = False

# --- MCMC Sampling (Uninformative Prior) ---
do_sampling_uninformative = False

# --- Preprocessing (Informative) ---
do_preprocess_hyper_info    = False
do_preprocess_subjects_info = False

# --- Preprocessing (Uninformative) ---
do_preprocess_hyper_uninf    = False
do_preprocess_subjects_uninf = False

# --- Plotting (Informative) ---
do_gen_paper_quantities         = False
do_hyper_plots_info             = False
do_subject_plots_info           = False
do_posterior_predictive_ce_info = False
do_paper_plots                  = False

# --- Plotting (Uninformative) ---
do_hyper_plots_uninf             = False
do_subject_plots_uninf           = False
do_posterior_predictive_ce_uninf = False

# --- Appendix Examples ---
do_skew_normal_example  = False

# --- Document Compilation ---
do_compile_doc          = False



# =============================================================================
# Run Selected Pipelines
# =============================================================================

if __name__ == '__main__':
    import pipelines

    # -------------------------------------------------------------------------
    # Data Pipeline
    # -------------------------------------------------------------------------
    if do_clean_data:
        pipelines.run_clean_data(DATA_DIR)

    if do_gen_paper_data:
        check_dependency('gen_paper_data', ['clean_data'])
        pipelines.run_gen_paper_data(
            DATA_DIR,
            direct_prior=DIRECT_PRIOR,
            prior_selection=PRIOR_SELECTION
        )

    # -------------------------------------------------------------------------
    # Prior Selection and Plots
    # -------------------------------------------------------------------------
    if do_generate_prior:
        pipelines.run_prior_selection(
            DATA_DIR,
            rho=PRIOR_SELECTION['rho'],
            alpha_0=PRIOR_SELECTION['alpha_0'],
            direct_prior=DIRECT_PRIOR
        )

    if do_prior_plots:
        pipelines.run_prior_plots(
            direct_prior=DIRECT_PRIOR,
            prior_selection=PRIOR_SELECTION,
            output_dir=PLOTS_DIR / 'priors',
            seed=SEEDS['sampling']
        )

    if do_prior_predictive_ce:
        check_dependency('prior_predictive_ce', ['clean_data'])
        pipelines.run_prior_predictive_ce(
            direct_prior=DIRECT_PRIOR,
            prior_selection=PRIOR_SELECTION,
            data_dir=DATA_DIR,
            output_dir=PLOTS_DIR / 'priors',
            seed=SEEDS['sampling']
        )

    # -------------------------------------------------------------------------
    # MCMC Sampling (Informative Prior)
    # -------------------------------------------------------------------------
    if do_sampling:
        check_dependency('sampling', ['clean_data', 'prior_params'])
        prior_params_path = DATA_DIR / 'output' / 'prior_params.npz'
        pipelines.run_sampling(sampling_opts, prior_params_path=prior_params_path)

    # -------------------------------------------------------------------------
    # MCMC Sampling (Uninformative Prior)
    # -------------------------------------------------------------------------
    if do_sampling_uninformative:
        check_dependency('sampling_uninformative', ['clean_data'])

        # Save uninformative hyperprior directly (no variance decomposition)
        uninformative_prior_path = DATA_DIR / 'output' / 'prior_params_uninformative.npz'
        from prior_selection import save_direct_hyperprior
        save_direct_hyperprior(UNINFORMATIVE_HYPERPRIOR, uninformative_prior_path)

        # Run sampling with uninformative prior
        uninformative_opts = sampling_opts.copy()
        uninformative_opts['output_dir'] = DATA_DIR / 'output_uninformative'
        uninformative_opts['plots_dir'] = PLOTS_DIR / 'rdu_bhm_uninformative'
        uninformative_opts['clean'] = True  # Clean previous uninformative runs
        pipelines.run_sampling(uninformative_opts, prior_params_path=uninformative_prior_path)

    # -------------------------------------------------------------------------
    # Preprocessing (Informative Prior)
    # -------------------------------------------------------------------------
    history_dir_info = get_mcmc_dir('rdu_bhm') / 'history' / 'full'
    preprocess_dir_info = get_mcmc_dir('rdu_bhm') / 'preprocessed'

    if do_preprocess_hyper_info:
        check_dependency('preprocess_hyper', ['clean_data'])
        pipelines.run_preprocess_hyperparams(history_dir_info, preprocess_dir_info)

    if do_preprocess_subjects_info:
        check_dependency('preprocess_subjects', ['clean_data'])
        pipelines.run_preprocess_subjects(history_dir_info, preprocess_dir_info / 'subjects')

    # -------------------------------------------------------------------------
    # Preprocessing (Uninformative Prior)
    # -------------------------------------------------------------------------
    history_dir_uninf = DATA_DIR / 'output_uninformative' / 'rdu_bhm' / 'history' / 'full'
    preprocess_dir_uninf = DATA_DIR / 'output_uninformative' / 'rdu_bhm' / 'preprocessed'

    if do_preprocess_hyper_uninf:
        if history_dir_uninf.exists():
            pipelines.run_preprocess_hyperparams(history_dir_uninf, preprocess_dir_uninf)
        else:
            print(f"Skipping uninformative preprocessing: {history_dir_uninf} not found")

    if do_preprocess_subjects_uninf:
        if history_dir_uninf.exists():
            pipelines.run_preprocess_subjects(history_dir_uninf, preprocess_dir_uninf / 'subjects')
        else:
            print(f"Skipping uninformative subject preprocessing: {history_dir_uninf} not found")

    # -------------------------------------------------------------------------
    # Plotting (Informative Prior)
    # -------------------------------------------------------------------------
    plots_dir_info = get_plots_dir('rdu_bhm')

    if do_gen_paper_quantities:
        check_dependency('gen_paper_quantities', ['clean_data'])
        pipelines.run_gen_paper_quantities(history_dir_info, DATA_DIR)

    if do_hyper_plots_info:
        check_dependency('hyper_plots', ['clean_data'])
        from prior_selection import get_hyperprior_specs_for_plotting
        hyperprior_specs = get_hyperprior_specs_for_plotting(
            DIRECT_PRIOR, rho=PRIOR_SELECTION['rho'], alpha_0=PRIOR_SELECTION['alpha_0']
        )
        # Use preprocessed data if available
        hyper_preprocessed = preprocess_dir_info / 'hyperparams_preprocessed.npz'
        if hyper_preprocessed.exists():
            pipelines.run_hyperparameter_plots(history_dir_info, plots_dir_info,
                                               hyperprior_specs=hyperprior_specs,
                                               preprocessed_path=hyper_preprocessed)
        else:
            pipelines.run_hyperparameter_plots(history_dir_info, plots_dir_info,
                                               hyperprior_specs=hyperprior_specs)

    if do_subject_plots_info:
        check_dependency('subject_plots', ['clean_data'])
        # Use preprocessed data if available
        subjects_preprocessed = preprocess_dir_info / 'subjects'
        if subjects_preprocessed.exists() and (subjects_preprocessed / 'metadata.npz').exists():
            pipelines.run_subject_plots(history_dir_info, plots_dir_info,
                                        n_workers=N_WORKERS,
                                        preprocessed_dir=subjects_preprocessed)
        else:
            pipelines.run_subject_plots(history_dir_info, plots_dir_info, n_workers=N_WORKERS)

    if do_posterior_predictive_ce_info:
        check_dependency('posterior_predictive_ce_info', ['clean_data'])
        prior_ce_path = OUTPUTS['prior_predictive_ce']
        pipelines.run_posterior_predictive_ce(
            history_dir=history_dir_info,
            data_dir=DATA_DIR,
            output_dir=plots_dir_info,
            label='informative',
            prior_ce_path=prior_ce_path,
        )

    if do_paper_plots:
        check_dependency('paper_plots', ['paper_quantities'])
        pipelines.run_paper_plots(OUTPUTS['paper_quantities'], PLOTS_DIR / 'paper')

    # -------------------------------------------------------------------------
    # Plotting (Uninformative Prior)
    # -------------------------------------------------------------------------
    plots_dir_uninf = PLOTS_DIR / 'rdu_bhm_uninformative'

    if do_hyper_plots_uninf:
        if history_dir_uninf.exists():
            # No prior reference lines for uninformative (diffuse prior not useful to display)
            hyper_preprocessed_uninf = preprocess_dir_uninf / 'hyperparams_preprocessed.npz'
            if hyper_preprocessed_uninf.exists():
                pipelines.run_hyperparameter_plots(history_dir_uninf, plots_dir_uninf,
                                                   preprocessed_path=hyper_preprocessed_uninf)
            else:
                pipelines.run_hyperparameter_plots(history_dir_uninf, plots_dir_uninf)
        else:
            print(f"Skipping uninformative hyper plots: {history_dir_uninf} not found")

    if do_subject_plots_uninf:
        if history_dir_uninf.exists():
            # Use preprocessed data if available
            subjects_preprocessed_uninf = preprocess_dir_uninf / 'subjects'
            if subjects_preprocessed_uninf.exists() and (subjects_preprocessed_uninf / 'metadata.npz').exists():
                pipelines.run_subject_plots(history_dir_uninf, plots_dir_uninf,
                                            n_workers=N_WORKERS,
                                            preprocessed_dir=subjects_preprocessed_uninf)
            else:
                pipelines.run_subject_plots(history_dir_uninf, plots_dir_uninf, n_workers=N_WORKERS)
        else:
            print(f"Skipping uninformative subject plots: {history_dir_uninf} not found")

    if do_posterior_predictive_ce_uninf:
        if history_dir_uninf.exists():
            pipelines.run_posterior_predictive_ce(
                history_dir=history_dir_uninf,
                data_dir=DATA_DIR,
                output_dir=plots_dir_uninf,
                label='uninformative',
            )
        else:
            print(f"Skipping uninformative posterior predictive CE: {history_dir_uninf} not found")

    # -------------------------------------------------------------------------
    # Appendix Examples
    # -------------------------------------------------------------------------
    if do_skew_normal_example:
        pipelines.run_skew_normal_example(
            direct_prior=SKEW_NORMAL_DIRECT_PRIOR,
            researcher_choices=SKEW_NORMAL_RESEARCHER,
            output_dir=PLOTS_DIR / 'skew_normal',
            n_mc_samples=50000,
            seed=SEEDS['sampling'],
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
        # Run pdflatex -> biber -> pdflatex -> pdflatex for full compilation
        commands = [
            ['latexmk', '-pdf', 'main.tex'],
        ]
        for cmd in commands:
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=doc_dir, capture_output=True, text=True)
            if result.returncode != 0 and cmd[0] != 'biber':
                print(f"  Warning: {cmd[0]} returned non-zero exit code")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:500]}")
        print("Done.\n")

    # -------------------------------------------------------------------------
    # Wrap up
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
