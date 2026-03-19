# Code Architecture

This directory contains the core implementation for quantity-space prior specification in Bayesian Hierarchical Models (BHMs) of risk preferences.

## Directory Structure

```
code/
├── __init__.py              # Package initialization
├── main.py                  # Hub: JAX config, flags, configuration
├── pipelines.py             # Pipeline dispatch functions for main.py
├── helpers.py               # Helper functions and path constants
├── clean_data.py            # Data preparation and cleaning
├── prior_selection.py       # Core contribution: Beta-Beta hyperprior derivation
├── sensitivity.py           # Sensitivity analysis over (ρ, κ) grid
├── sampling_ce.py           # MCMC sampling with bamcmc (CE-space model)
├── gen_paper_data.py        # Generate data summary statistics for pythontex
├── plotting.py              # Diagnostic plots (hyperparameters, subjects)
├── injectivity_analysis.py  # CE mapping injectivity analysis (Appendix)
├── skew_normal_hyperprior.py # Skew-Normal 3-parameter example
└── posteriors/
    ├── __init__.py          # Posterior package initialization
    ├── common_fns.py        # Shared utility functions (CRRA, Prelec, likelihood)
    └── rdu_bhm_ce.py        # RDU with Beta-Beta CE-space hierarchy
```

## Module Overview

### main.py - Project Orchestration

Hub script with the following sections:

1. **JAX Environment Config** - GPU memory, compilation cache (must be before JAX imports)
2. **Path Configuration** - Imports OUTPUTS and helpers from helpers.py
3. **CE-Space Prior Configuration** - REFERENCE_LOTTERIES, REFERENCE_PAIR, DIRECT_PRIOR_CE, PRIOR_SELECTION_CE
4. **Control Flags** - Boolean do_* flags grouped by category
5. **Sampling Options** - sampling_opts_ce dict for MCMC configuration
6. **Pipeline Dispatch** - Calls pipelines.run_*() functions

### Control Flags

| Flag | Purpose | Pipeline Function |
|------|---------|-------------------|
| `do_clean_data` | Load raw CSV, filter to RA task | `run_clean_data()` |
| `do_gen_paper_data` | Generate summary statistics | `run_gen_paper_data()` |
| `do_sensitivity` | Sensitivity analysis over (ρ, κ) | `run_sensitivity()` |
| `do_generate_prior_ce` | Derive Beta-Beta hyperpriors | `run_prior_selection_ce()` |
| `do_sampling_ce` | Run MCMC sampling | `run_sampling_ce()` |
| `do_preprocess_hyper_ce` | Preprocess CE hyperparameters | `run_preprocess_hyperparams_ce()` |
| `do_preprocess_subjects_ce` | Preprocess CE subjects | `run_preprocess_subjects_ce()` |
| `do_hyper_plots_ce` | Hyperparameter diagnostic plots | `run_hyperparameter_plots_ce()` |
| `do_subject_plots_ce` | Subject diagnostic plots | `run_subject_plots()` |
| `do_injectivity_analysis` | CE mapping injectivity | `run_injectivity_analysis()` |
| `do_compile_doc` | Compile LaTeX document | (inline) |

**Usage:** Set desired `do_*` flags to `True` in the file, then run:
```bash
python code/main.py
```

### prior_selection.py - Core Methodology

Implements quantity-space prior specification with Beta-Beta hierarchy.

**Key Insight:** Specify beliefs about certainty equivalents and choice probabilities
(cognitively tractable quantities), then derive Beta-Beta hyperpriors via variance
decomposition with (ρ, κ) researcher choices.

**Dataclasses:**
- `PriorSelectionResult`: Individual-level prior specification (4 quantities)
- `DerivedHyperpriorsBeta`: Beta-Beta hyperparameters from variance decomposition

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `variance_from_tail_probability()` | Determine Beta variance from mean + tail probability |
| `build_ce_direct_prior()` | Build direct prior from elicitation inputs |
| `ce_space_prior_selection()` | Prior selection for 3 CEs + 1 choice probability |
| `derive_hyperpriors_beta()` | Variance decomposition to Beta-Beta hyperpriors |
| `verify_hyperpriors_beta()` | Monte Carlo verification of variance targets |
| `print_hyperprior_summary_beta()` | Summary table of derived hyperpriors |
| `run_prior_selection_ce()` | Full pipeline: select → derive → verify → save |

**Variance Decomposition (Beta-Beta):**

For each quantity k on [lo, hi], normalized to z ∈ [0,1]:
- μ_k ~ Beta(a_μ, b_μ) — population mean
- ψ_k ~ Beta(a_ψ, b_ψ) — variance fraction
- z_ik | μ_k, ψ_k ~ Beta(μ(1-ψ)/ψ, (1-μ)(1-ψ)/ψ)

Researcher choices:
- `ρ`: Fraction of variance from uncertainty about population mean
- `κ`: Coefficient of variation of heterogeneity (CV of σ²)

### posteriors/rdu_bhm_ce.py - CE-Space Model

Beta-Beta hierarchy with logit-parameterized hyperparameters.

- 4 params per subject: r, φ, η, λ (via NCP with epsilon ~ N(0,1))
- 8 hyperparameters: logit_μ_k and logit_ψ_k for 4 quantities (CE1, CE2, CE3, P_choice)
- Hyperprior array layout: [a_mu, b_mu, a_psi, b_psi, lo, hi] per quantity
- Chain state stores logit-scale; generated quantities return sigmoid-transformed values

### posteriors/common_fns.py - Shared Functions

- `crra_ufn(x, r)`: CRRA utility (handles r=1 log case)
- `prelec_pwf(p, phi, eta)`: Prelec 2-parameter weighting w(p) = exp(-η(-ln p)^φ)
- `rdu_value_3outcome()`: 3-outcome lottery RDU value
- `risk_likelihood()`: Log-likelihood for subject's choices

### clean_data.py - Data Preparation

Loads and transforms experimental data into a unified 3-outcome format for bamcmc.
Supports two datasets:
- `'hn2016'`: Harrison & Ng (2016), 3-outcome lotteries in USD (default)
- `'andersen2008'`: Andersen et al. (2008), binary lotteries in DKK (padded to 3-outcome)

### sensitivity.py - Sensitivity Analysis

Computes a sensitivity grid over (ρ, κ) values using the Beta-Beta hierarchy.
For each cell, draws from the full hierarchy and reports:
- Between-population SD (robust IQR/1.349 of draw-level mean CEs)
- Within-population SD (median of draw-level CE SDs)
- Within-population IQR (IQR of draw-level CE SDs)

### plotting.py - Diagnostic Plots

Two-stage approach for memory efficiency:
1. Preprocess: Extract per-subject data from full history
2. Plot: Pool of workers generate PDFs from preprocessed data

Supports CE-space model with sigmoid transform to display natural scale.

## Data Flow

```
data/input/hn2016.csv
    |
    v
clean_data.py --> data/input/dbar_clean.npz
    |
    v
gen_paper_data.py --> data/output/paper_data.npz (for pythontex)
    |
    v
prior_selection.py --> data/output/prior_params_ce.npz
    |
    v
sampling_ce.py --> data/output/rdu_bhm_ce/ (MCMC output)
    |
    v
plotting.py --> plots/rdu_bhm_ce/ (diagnostics)
```

## Usage

```bash
# Activate environment
source /workspace/pylib/bin/activate

# Edit main.py to set desired do_* flags to True, then run:
python code/main.py
```
