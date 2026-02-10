"""
Helper functions for main.py pipeline orchestration.
"""

import sys
from pathlib import Path

# Setup paths (same as main.py)
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'

# Output file paths
OUTPUTS = {
    'clean_data': DATA_DIR / 'input' / 'dbar_clean.npz',
    'paper_data': DATA_DIR / 'output' / 'paper_data.npz',
    'prior_params': DATA_DIR / 'output' / 'prior_params.npz',
    'paper_quantities': DATA_DIR / 'output' / 'paper_quantities.npz',
    'prior_predictive_ce': DATA_DIR / 'output' / 'prior_predictive_ce.npz',
    'posterior_predictive_ce_info': DATA_DIR / 'output' / 'posterior_predictive_ce_informative.npz',
    'posterior_predictive_ce_uninf': DATA_DIR / 'output' / 'posterior_predictive_ce_uninformative.npz',
}


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
