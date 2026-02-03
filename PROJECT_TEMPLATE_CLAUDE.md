# CLAUDE.md Template for Research/Analysis Projects

This template captures the organizational patterns from the Bezier risk preferences project. Use it as a starting point for similarly structured computational research projects.

---

# CLAUDE.md - Project Context for AI Assistants

## Project Overview

[Brief description of the research project, its goals, and key collaborators]

**Project:** [Project Name]
**Authors:** [Names and affiliations]

### Project History & Current Focus

[Describe any phases of the project, datasets used, and current work focus]

### Code Status

- **Primary codebase:** `/workspace/code/` (Python)
- **Python Environment:** `/workspace/py_env/` - activate before running code
- **Reference/Legacy:** `/workspace/r_code/` (if applicable)

## Repository Structure

```
/workspace
├── code/                    # Source code (PRIMARY)
│   ├── main.py              # Entry point with do_* flags
│   ├── config.py            # Centralized paths and constants
│   ├── setup.py             # Project initialization and path helpers
│   │
│   ├── data_cleaning/       # Raw data → cleaned data
│   │   ├── __init__.py
│   │   └── clean_*.py       # Dataset-specific cleaning scripts
│   │
│   ├── estimation/          # Data prep and model fitting
│   │   ├── __init__.py
│   │   ├── prepare_data.py  # Clean data → model-ready format
│   │   └── get_samples.py   # MCMC/optimization runner
│   │
│   ├── posteriors/          # Model definitions (if Bayesian)
│   │   ├── __init__.py      # Model registration
│   │   ├── common_fns.py    # Shared likelihood/utility functions
│   │   └── model_*.py       # Individual model specifications
│   │
│   ├── pipelines/           # High-level workflow orchestration
│   │   ├── __init__.py      # Export all pipeline functions
│   │   ├── data.py          # Data cleaning/preparation pipelines
│   │   └── model_*.py       # Model-specific pipelines
│   │
│   ├── analysis/            # Post-estimation analysis
│   │   ├── posterior_summary.py
│   │   ├── classification.py
│   │   └── predictions.py
│   │
│   ├── plotting/            # Visualization
│   │   ├── plot_results.py
│   │   └── plot_diagnostics.py
│   │
│   └── docs/                # Code documentation
│       └── *.md
│
├── data/
│   ├── input/               # Raw experimental/source data (READ-ONLY)
│   │   └── {dataset}/       # Dataset-specific subdirectories
│   └── output/              # Processed data and results
│       ├── {dataset}/       # Cleaned data, prepared arrays
│       └── mcmc/            # MCMC samples by model
│           └── {dataset}/{model}/
│
├── plots/                   # Generated figures (separate from data!)
│   └── {dataset}/{model}/
│
├── doc/                     # Paper/documentation (LaTeX)
├── technical_notes/         # Detailed methodology docs
├── py_env/                  # Python virtual environment
└── TODO.md                  # Session continuity tracking
```

## Key Design Patterns

### 1. Entry Point with Boolean Flags (`main.py`)

```python
# Dataset selection at top
EST_DATA = 'dataset_name'

# Dataset-specific configuration
DATASET_CONFIG = {
    'dataset1': {
        'param1': value1,
        'param2': value2,
    },
    'dataset2': { ... },
}

# Control flags - set True to run
do_clean_data      = False
do_prepare_data    = False
do_run_estimation  = False
do_plot_results    = False
# ... etc

if __name__ == '__main__':
    if do_clean_data:
        pipelines.run_clean_data(EST_DATA, ...)

    if do_prepare_data:
        pipelines.run_prepare_data(EST_DATA, ...)
    # ... etc
```

### 2. Centralized Configuration (`config.py`)

```python
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / 'data'

CONFIG = {
    "paths": {
        "data_root": _DATA_DIR,
        "mcmc_root": _DATA_DIR / 'output' / 'mcmc',
        "plots_root": _PROJECT_ROOT / 'plots',
    },
    # Domain-specific constants...
}

def get_mcmc_dir(model_name: str, dataset: str) -> Path:
    return CONFIG["paths"]["mcmc_root"] / dataset / model_name

def get_plots_dir(model_name: str, dataset: str) -> Path:
    return CONFIG["paths"]["plots_root"] / dataset / model_name
```

### 3. Pipeline Module Pattern (`pipelines/`)

Each pipeline module provides high-level functions that orchestrate lower-level operations:

```python
# pipelines/model_xyz.py
def run_xyz_estimation(dataset, prepared_path, sampling_opts=None):
    """Run estimation for XYZ model."""
    print("=" * 60)
    print(f"Running: XYZ Estimation for {dataset}")
    print("=" * 60)

    from estimation.get_samples import run_sampling
    # ... orchestrate the work

def run_xyz_plotting(dataset, domain_bounds=None):
    """Generate plots for XYZ model."""
    from config import get_mcmc_dir, get_plots_dir
    from plotting.plot_results import plot_xyz
    # ... orchestrate the work
```

Export all pipelines from `__init__.py`:
```python
# pipelines/__init__.py
from .data import run_clean_data, run_prepare_data
from .model_xyz import run_xyz_estimation, run_xyz_plotting
```

### 4. Model Registration Pattern (`posteriors/`)

For Bayesian models with a sampling framework:

```python
# posteriors/__init__.py
from framework import register_model

from . import model_xyz

register_model("model_xyz", {
    'log_posterior': model_xyz.log_posterior,
    'initial_vector': model_xyz.get_initial_vector,
    'generated_quantities': model_xyz.generated_quantities,
    # ... other required functions
})
```

### 5. Setup Module (`setup.py`)

```python
from pathlib import Path
import subprocess
import sys

# Find project root via git
PROJECT_ROOT = Path(subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'],
    capture_output=True, text=True, check=True
).stdout.strip())

CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure code is importable
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

def dp(*args: str) -> Path:
    """Data path helper."""
    return DATA_DIR.joinpath(*args)
```

## TODO.md - Session Continuity

Maintain a `TODO.md` file for tracking work across sessions:

```markdown
# TODO

## In Progress
- [ ] Task description - *status notes, branch name*

## Up Next
- [ ] High priority tasks

## Backlog
- [ ] Lower priority tasks

## Completed (Recent)
- [x] Task - *notes* (YYYY-MM-DD)
```

**Session protocol:**
- Start: Read TODO.md, check for unblocked items
- During: Update status as you work
- End: Mark completions, note follow-ups

## Running the Pipeline

```bash
cd /workspace/code

# Activate environment
source /workspace/py_env/bin/activate
# OR use full path: /workspace/py_env/bin/python

# Run pipeline (edit main.py flags first)
python main.py

# Run specific module directly
python -c "from pipelines import run_xyz_estimation; run_xyz_estimation('dataset')"
```

## Git Workflow

[Adapt based on project needs]

- **main/master**: Stable, tested code only
- **feature branches**: New features, merged when complete
- **Atomic commits**: Small, focused changes
- **Never force push** to shared branches

## Safety Guidelines

- **NEVER delete** files without explicit permission
- **NEVER use** `git reset --hard`, `git clean -fd`
- **Protect** `/workspace/data/input/` - contains irreplaceable raw data
- **Ask first** before destructive operations

## Key Technologies

- **Python**: Primary language
- **[Framework]**: [e.g., JAX, PyTorch, Stan] for computation
- **LaTeX**: Documentation
- **[Other]**: As needed

## Coding Conventions

- Use the project's Python environment
- Prefer editing existing files over creating new ones
- Keep data and plots in separate directory trees
- Use `config.py` for paths, not hardcoded strings
- Follow existing patterns in the codebase

## Plot Styles

### Subject-Level Plots (Per-Subject PDF)

Each subject gets a multi-page PDF with standardized diagnostic and results plots:

**Page 1-2: MCMC Diagnostics**
- **Trace plots** (2-column layout, 3 rows): Show parameter chains over iterations
  - Multiple chains overlaid with transparency (alpha=0.5)
  - Red dashed horizontal line at median value
  - Helps diagnose convergence and mixing

- **Density plots** (2-column layout, 3 rows): Posterior distributions
  - Histograms with 50 bins, density normalized
  - Red vertical line at median with value in legend
  - Steel blue color scheme

**Page 3: Primary Result - Utility/Model Function**
- **Spaghetti plot**: Multiple posterior draws overlaid
  - ~50-150 random samples from posterior in semi-transparent color (alpha=0.3)
  - Median curve in red (lw=2.5) with high z-order
  - Reference line (e.g., risk-neutral, linear) in black dashed
  - Grid with alpha=0.3
  - Domain on x-axis, normalized output on y-axis

**Pages 4-6: Higher-Order Measures** (if applicable)
- Same spaghetti plot style for derived quantities
- Example for risk preferences:
  - Risk Aversion: $-u''/u'$
  - Prudence: $u'''/u'$
  - Temperance: $-u''''/u'$
- Interior domain (5% margin) to avoid boundary derivative issues

**Layout specifications:**
```python
# Traces/Densities: 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(14, 10))  # or (12, 10) for densities

# Spaghetti plots: single axis
fig, ax = plt.subplots(figsize=(10, 7))
```

### Hyperparameter Plots (Model-Level PDF)

For hierarchical models, hyperparameters get their own diagnostic PDF:

**Layout: 5 parameters per page, 2 columns**
- Left column: Trace plots
- Right column: Density plots

**Trace plots (left):**
- Up to 100 chains shown (subset if more)
- Vertical dotted lines at MCMC run boundaries
- Tab20 colormap for chain differentiation
- Line width 0.3, alpha 0.4

**Density plots (right):**
- Separate KDE curves for each MCMC run, overlaid
- Different colors per run (red, blue, green, purple, orange...)
- Shows whether runs converged to same distribution
- Filled density with alpha for visibility

**Parameter naming conventions:**
```python
# Independent structure (diagonal covariance)
HYPERPARAM_INFO = [
    ('a_mean', r'$\mu_a$ (logit scale)'),
    ('ln_a_sdev', r'$\ln\sigma_a$'),
    ('b_mean', r'$\mu_b$ (logit scale)'),
    ('ln_b_sdev', r'$\ln\sigma_b$'),
    # ... etc
]

# Covariance structure (Cholesky parameterization)
HYPERPARAM_INFO_COV = [
    ('mu_a', r'$\mu_a$'),
    ('ln_L_11', r'$\ln L_{11}$'),  # Log of diagonal
    ('L_21', r'$L_{21}$'),          # Off-diagonal as-is
    # ... etc
]
```

**Figure specifications:**
```python
# 5 params per page, trace + density columns
fig, axes = plt.subplots(n_on_page, 2, figsize=(14, 2.8 * n_on_page))
```

### Color Schemes

| Element | Color | Notes |
|---------|-------|-------|
| Posterior samples | `steelblue` | Main data color |
| Median/summary | `red` | High visibility |
| Reference lines | `black`, dashed | Neutral baseline |
| Chain colors | `plt.cm.tab20` | Up to 20 distinct |
| Run colors | `['#e41a1c', '#377eb8', ...]` | ColorBrewer Set1 |

### Output Organization

```
plots/
└── {dataset}/
    └── {model}/
        ├── subjects/           # Per-subject PDFs
        │   ├── subject_000.pdf
        │   ├── subject_001.pdf
        │   └── ...
        ├── {model}_hyperparams.pdf  # Hyperparameter diagnostics
        └── {model}_summary.pdf      # Aggregate results (if applicable)
```

## Contact

- [Name]: [email]
- [Name]: [email]
