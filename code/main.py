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
os.environ['JAX_ENABLE_X64'] = 'true'

# JAX compilation cache directory (speeds up subsequent runs)
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.path.dirname(__file__) + '/jax_cache'

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

# BHM Hyperprior researcher choices:
# - rho: Variance partition ratio (fraction from mean uncertainty vs population spread)
#        0.3 means 30% from hyperprior mean uncertainty, 70% from population variance
# - alpha_0: InverseGamma shape parameter (higher = tighter variance prior)
#            CV[sigma^2] = 1/sqrt(alpha_0-2)
PRIOR_SELECTION = {
    'rho': 0.3,
    'alpha_0': 6.0,
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
    'rho': 0.3,           # variance partition (same interpretation as Normal case)
    'alpha_omega': 6.0,   # scale concentration (analogous to alpha_0)
    'tau_alpha_sq': 0.25, # shape uncertainty (new degree of freedom)
}

# Random seeds for reproducibility
SEEDS = {
    'sampling': 42,
}

# Parallel processing
N_WORKERS = 15  # Number of parallel workers for plotting, etc.


# =============================================================================
# Control Flags
# =============================================================================

# --- Data Pipeline ---
do_clean_data     = False
do_gen_paper_data = False

# --- Prior Selection ---
do_generate_prior = False

# --- MCMC Sampling ---
do_sampling = False

# --- Post-Processing ---
do_gen_paper_quantities = True
do_diagnostic_plots     = True
do_paper_plots          = True

# --- Appendix Examples ---
do_skew_normal_example  = False


# =============================================================================
# Sampling Options
# =============================================================================

sampling_opts = {
    'model': 'rdu_bhm',
    'output_dir': DATA_DIR / 'output',  # bamcmc creates {model}/ subdirectory
    'plots_dir': PLOTS_DIR / 'rdu_bhm',
    'clean': True,
    'reset_runs': 0,
    'resume_runs': 1,
    'num_chains': 500,
    'num_collect': 100,
    'thin_iteration': 1,
    'seed': SEEDS['sampling'],
}


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
        pipelines.run_gen_paper_data(DATA_DIR)

    # -------------------------------------------------------------------------
    # Prior Selection
    # -------------------------------------------------------------------------
    if do_generate_prior:
        pipelines.run_prior_selection(
            DATA_DIR,
            rho=PRIOR_SELECTION['rho'],
            alpha_0=PRIOR_SELECTION['alpha_0']
        )

    # -------------------------------------------------------------------------
    # MCMC Sampling
    # -------------------------------------------------------------------------
    if do_sampling:
        check_dependency('sampling', ['clean_data', 'prior_params'])
        pipelines.run_sampling(sampling_opts)

    # -------------------------------------------------------------------------
    # Post-Processing
    # -------------------------------------------------------------------------
    if do_gen_paper_quantities:
        check_dependency('gen_paper_quantities', ['clean_data'])
        history_dir = get_mcmc_dir() / 'history' / 'full'
        pipelines.run_gen_paper_quantities(history_dir, DATA_DIR)

    if do_diagnostic_plots:
        check_dependency('diagnostic_plots', ['clean_data'])
        history_dir = get_mcmc_dir() / 'history' / 'full'
        pipelines.run_diagnostic_plots(history_dir, get_plots_dir(), n_workers=N_WORKERS)

    if do_paper_plots:
        check_dependency('paper_plots', ['paper_quantities'])
        pipelines.run_paper_plots(OUTPUTS['paper_quantities'], PLOTS_DIR / 'paper')

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
    # Wrap up
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
