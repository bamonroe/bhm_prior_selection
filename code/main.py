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
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

# GPU memory allocator (recommended for large models)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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

# Output file paths
OUTPUTS = {
    'clean_data': DATA_DIR / 'input' / 'dbar_clean.npz',
    'paper_data': DATA_DIR / 'output' / 'paper_data.npz',
    'prior_params': DATA_DIR / 'output' / 'prior_params.npz',
    'paper_quantities': DATA_DIR / 'output' / 'paper_quantities.npz',
}


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

# Random seeds for reproducibility
SEEDS = {
    'sampling': 42,
}


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
do_gen_paper_quantities = False
do_diagnostic_plots     = False
do_paper_plots          = False


# =============================================================================
# Sampling Options
# =============================================================================

sampling_opts = {
    'model': 'rdu_bhm',
    'output_dir': DATA_DIR / 'output' / 'rdu_bhm',
    'plots_dir': PLOTS_DIR / 'rdu_bhm',
    'reset_runs': 0,
    'resume_runs': 5,
    'num_chains': 1000,
    'num_collect': 100,
    'thin_iteration': 10,
    'seed': SEEDS['sampling'],
}


# =============================================================================
# Helper Functions
# =============================================================================

def check_dependency(step_name, required_outputs, strict=True):
    """Check that required outputs from previous steps exist."""
    missing = []
    for output_key in required_outputs:
        output_path = OUTPUTS[output_key]
        if not output_path.exists():
            missing.append((output_key, output_path))

    if missing:
        print(f"\nERROR: Missing dependencies for '{step_name}':")
        for key, path in missing:
            print(f"  - {key}: {path}")
        print(f"\nRun the required step(s) first.")
        if strict:
            sys.exit(1)
        return False
    return True


def get_mcmc_dir(model='rdu_bhm'):
    """Get MCMC output directory for a model."""
    return DATA_DIR / 'output' / model


def get_plots_dir(model='rdu_bhm'):
    """Get plots directory for a model."""
    return PLOTS_DIR / model


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
        pipelines.run_diagnostic_plots(history_dir, get_plots_dir())

    if do_paper_plots:
        check_dependency('paper_plots', ['paper_quantities'])
        pipelines.run_paper_plots(OUTPUTS['paper_quantities'], PLOTS_DIR / 'paper')

    # -------------------------------------------------------------------------
    # Wrap up
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
