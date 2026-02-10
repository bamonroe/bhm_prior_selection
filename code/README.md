# Code Architecture

This directory contains the core implementation for principled hyperprior selection in Bayesian Hierarchical Models (BHMs) of risk preferences.

## Directory Structure

```
code/
├── __init__.py              # Package initialization
├── main.py                  # Hub: JAX config, flags, configuration
├── pipelines.py             # Pipeline dispatch functions for main.py
├── helpers.py               # Helper functions and path constants
├── clean_data.py            # Data preparation and cleaning
├── prior_selection.py       # Core contribution: hyperprior derivation
├── sampling.py              # MCMC sampling with bamcmc
├── gen_paper_data.py        # Generate data summary statistics
├── gen_paper_quantities.py  # Extract posterior summaries
├── plotting.py              # Diagnostic and paper plots
├── prior_predictive_ce.py   # Prior and posterior predictive CE distributions
├── skew_normal_hyperprior.py # Skew-Normal 3-parameter example (Appendix B)
└── posteriors/
    ├── __init__.py          # Posterior package initialization
    ├── common_fns.py        # Shared utility functions
    └── rdu_bhm.py           # RDU with Prelec 2-parameter PWF (NCP)
```

## Module Overview

### main.py - Project Orchestration

Hub script with the following sections:

1. **JAX Environment Config** - GPU memory, compilation cache (must be before JAX imports)
2. **Path Configuration** - Imports OUTPUTS and helpers from helpers.py
3. **Prior Selection Config** - DIRECT_PRIOR, PRIOR_SELECTION (rho, alpha_0), UNINFORMATIVE_HYPERPRIOR
4. **Control Flags** - Boolean do_* flags grouped by category
5. **Sampling Options** - sampling_opts dict for MCMC configuration
6. **Pipeline Dispatch** - Calls pipelines.run_*() functions

### helpers.py - Path Constants and Utilities

Provides shared path constants and helper functions:
- `OUTPUTS` - Dict of expected output file paths for dependency checking
- `check_dependency()` - Verify required outputs exist before running a step
- `get_mcmc_dir()` - Get MCMC output directory for a model
- `get_plots_dir()` - Get plots directory for a model

### Control Flags

| Flag | Purpose | Pipeline Function |
|------|---------|-------------------|
| `do_clean_data` | Load raw CSV, filter to RA task | `run_clean_data()` |
| `do_gen_paper_data` | Generate summary statistics | `run_gen_paper_data()` |
| `do_generate_prior` | Derive hyperpriors | `run_prior_selection()` |
| `do_prior_plots` | Prior distribution comparison plots | `run_prior_plots()` |
| `do_prior_predictive_ce` | Prior predictive CE distribution | `run_prior_predictive_ce()` |
| `do_sampling` | Run MCMC (informative prior) | `run_sampling()` |
| `do_sampling_uninformative` | Run MCMC (uninformative prior) | `run_sampling()` |
| `do_preprocess_hyper_info` | Preprocess hyperparams (informative) | `run_preprocess_hyperparams()` |
| `do_preprocess_subjects_info` | Preprocess subjects (informative) | `run_preprocess_subjects()` |
| `do_preprocess_hyper_uninf` | Preprocess hyperparams (uninformative) | `run_preprocess_hyperparams()` |
| `do_preprocess_subjects_uninf` | Preprocess subjects (uninformative) | `run_preprocess_subjects()` |
| `do_gen_paper_quantities` | Extract posterior summaries | `run_gen_paper_quantities()` |
| `do_hyper_plots_info` | Hyperparameter diagnostics (informative) | `run_hyperparameter_plots()` |
| `do_subject_plots_info` | Subject diagnostics (informative) | `run_subject_plots()` |
| `do_posterior_predictive_ce_info` | Posterior predictive CE (informative) | `run_posterior_predictive_ce()` |
| `do_paper_plots` | Publication figures | `run_paper_plots()` |
| `do_hyper_plots_uninf` | Hyperparameter diagnostics (uninformative) | `run_hyperparameter_plots()` |
| `do_subject_plots_uninf` | Subject diagnostics (uninformative) | `run_subject_plots()` |
| `do_posterior_predictive_ce_uninf` | Posterior predictive CE (uninformative) | `run_posterior_predictive_ce()` |
| `do_skew_normal_example` | Skew-Normal hyperprior example | `run_skew_normal_example()` |
| `do_compile_doc` | Compile LaTeX document | (inline) |

Each flag checks dependencies via `check_dependency()` before running.

**Usage:** Set desired `do_*` flags to `True` in the file, then run:
```bash
python code/main.py
```

### pipelines.py - Pipeline Dispatch

Wrapper functions that main.py calls. Each function:
- Prints status header
- Imports and calls the implementation module
- Prints completion message

### clean_data.py - Data Preparation

Transforms raw experimental data (Andersen et al. 2008) into bamcmc-compatible formats.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_raw_data()` | Load CSV, filter to risk aversion task |
| `parse_ra_data()` | Structure lottery data by subject |
| `make_bamcmc_data_indexed()` | Create GPU-optimized indexed format |
| `clean_and_save_data()` | Main pipeline entry point |
| `load_clean_data()` | Load preprocessed data |

**Data Format (indexed):**
- `"static"`: (n_subjects, n_choices, n_unique_outcomes, n_unique_probs)
- `"int"`: (choices, idx_a_high, idx_a_low, idx_b_high, idx_b_low, idx_prob)
- `"float"`: (unique_outcomes, unique_cumprobs)

The indexed format precomputes unique outcomes/probabilities for efficient GPU lookups.

### prior_selection.py - Core Methodology

Implements principled hyperprior selection via variance decomposition.

**Key Insight:** Specify a prior for a single random individual, then derive hyperpriors such that the unconditional BHM distribution matches this prior exactly.

**Dataclasses:**
- `PriorSelectionResult`: Individual-level prior specification
- `DerivedHyperpriors`: BHM hyperparameters derived from prior selection

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `rdu_prelec_prior_selection()` | Define informative individual-level priors (4-param Prelec) |
| `uninformative_prior_selection()` | Baseline N(0, 100^2) priors for comparison |
| `derive_hyperpriors(ps, rho, alpha_0)` | Variance decomposition to get hyperpriors |
| `verify_hyperpriors()` | Validate unconditional variance constraint |
| `get_hyperpriors_for_rdu_bhm()` | Format hyperpriors for rdu_bhm.py |
| `get_hyperprior_specs_for_plotting()` | Format hyperprior specs for prior reference lines |
| `save_direct_hyperprior()` | Save uninformative hyperprior directly (no decomposition) |

**Variance Decomposition:**
```
Var[theta] = E[sigma^2] + tau_0^2
           = (1 - rho) * V + rho * V
           = V  (matches prior selection)
```

Where:
- `rho`: Fraction of variance from uncertainty about population mean (current: 0.2)
- `alpha_0`: Concentration parameter for Inverse-Gamma (current: 20.0)

**Current Values (rho=0.2, alpha_0=20.0):**

| Parameter | mu_0 | tau_0 | E[sigma] | Prior Var |
|-----------|------|-------|----------|-----------|
| r | 0.50 | 0.224 | 0.447 | 0.25 |
| phi_raw | 0.00 | 0.173 | 0.346 | 0.15 |
| eta_raw | 0.00 | 0.173 | 0.346 | 0.15 |
| lambda_raw | -2.00 | 0.447 | 0.894 | 1.00 |

### prior_predictive_ce.py - Certainty Equivalent Analysis

Translates abstract prior/posterior parameter distributions into an economically
interpretable quantity: the certainty equivalent (CE) of a concrete lottery.

**Prior predictive:** For each MC draw, sample hyperpriors → population params → subject
params → compute RDU CE of a 3-outcome equally-likely lottery.

**Posterior predictive:** Load MCMC posterior samples from last history file, extract
generated quantities (r, phi, eta), compute CE for each draw × subject.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `compute_prior_predictive_ce()` | MC sampling from hyperprior → CE |
| `compute_posterior_predictive_ce()` | Posterior samples → CE (JAX vmap) |
| `run_prior_predictive_ce()` | Full prior predictive pipeline |
| `run_posterior_predictive_ce()` | Full posterior predictive pipeline |
| `plot_prior_predictive_ce()` | Histogram of prior CE distribution |
| `plot_posterior_predictive_ce()` | Histogram with optional prior overlay |

### posteriors/common_fns.py - Shared Functions

Reusable functions across RDU model variants.

**Utility Functions:**
- `crra_ufn(x, r)`: CRRA utility (handles r=1 log case)
- `prelec_pwf(p, phi, eta)`: Prelec 2-parameter weighting w(p) = exp(-eta * (-ln p)^phi)

**Likelihood:**
- `rdu_value_indexed()`: Binary lottery RDU value via index lookups
- `risk_likelihood()`: Log-likelihood for subject's choices (JAX-vectorized)

### posteriors/rdu_bhm.py - Primary Model

Rank-Dependent Utility with Prelec 2-parameter probability weighting, using **Non-Centered Parameterization (NCP)**.

**NCP Benefits:**
- Breaks the "funnel" coupling between hyperparameters and subject parameters
- Dramatically improves mixing for hierarchical models
- Hyperparameter updates achieve ~30-50% acceptance rates via theta-preserving transforms

**Parameterization:**
- Subject parameters store epsilon ~ N(0,1) standardized deviations
- Actual parameters computed as: theta = mu + sigma * epsilon
- Transform to positive: phi = exp(phi_raw), eta = exp(eta_raw), lambda = exp(lambda_raw)

**Model Parameters (per subject, stored as epsilon):**
- `r_eps`: CRRA risk aversion epsilon
- `phi_eps`: Prelec curvature epsilon (log scale)
- `eta_eps`: Prelec elevation epsilon (log scale)
- `lambda_eps`: Fechner noise epsilon (log scale)

**Hyperparameters (8 total):**
- Population mean and log-std for each of 4 parameters
- Updated via COUPLED_TRANSFORM sampler for theta-preserving proposals

**Hyperprior Specifications:**
Derived from `prior_selection.py` with rho=0.2, alpha_0=20.0:
- mu ~ N(mu_0, tau_0^2)
- sigma^2 ~ InvGamma(alpha_0, beta_0)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `log_posterior()` | Dispatch to subject or hyperparameter posterior |
| `_transform_eps_to_theta()` | NCP transform: theta = mu + sigma * eps |
| `coupled_transform_dispatch()` | Theta-preserving hyperparameter updates |
| `batch_type()` | Define MCMC block structure (MH + COUPLED_TRANSFORM) |
| `initial_vector()` | Generate epsilon near 0, hyperparams at prior means |
| `generated_quantities()` | Transform epsilon to natural scale parameters |

**Generated Quantities Layout:**
```
[r_0..r_N, phi_0..phi_N, eta_0..eta_N, lambda_0..lambda_N, pop_params(8)]
```
Where pop_params = [r_mean, r_sd, phi_mean_nat, phi_sd, eta_mean_nat, eta_sd, lambda_mean_nat, lambda_sd]

**COUPLED_TRANSFORM Sampler:**
When hyperparameters (mu, sigma) are proposed:
1. Compute new epsilon: eps' = (theta - mu') / sigma'
2. Accept/reject based on priors only (likelihood cancels!)
3. Log acceptance ratio includes:
   - Jacobian: N * log(sigma_old / sigma_new)
   - Epsilon prior ratio: -0.5 * (sum(eps'^2) - sum(eps^2))
   - Hyperprior ratio

### skew_normal_hyperprior.py - Appendix B Example

Demonstrates the prior selection methodology for a 3-parameter Skew-Normal distribution,
showing that the approach extends beyond Normal population distributions.

Uses Monte Carlo integration for law of total cumulance and numerical root-finding
for hyperprior parameters. Generates comparison plots of direct prior vs unconditional distribution.

## Data Flow

```
data/input/dbar.csv (raw Andersen et al. 2008)
    |
    v
load_raw_data() --> filter to RA==1
    |
    v
parse_ra_data() --> structure by subject
    |
    v
make_bamcmc_data_indexed() --> precompute unique values
    |
    v
data/input/dbar_clean.npz
    |
    v
rdu_bhm.log_posterior() --> likelihood computation
```

## Dependencies

- **numpy**: Data structures and array operations
- **jax**: GPU computation, autodiff, JIT compilation
- **jax.scipy.stats**: Log probability functions
- **bamcmc**: MCMC sampler (optional, gracefully handles absence)

## Usage

```bash
# Activate environment
source /workspace/pylib/bin/activate

# Edit main.py to set desired do_* flags to True, then run:
python code/main.py

# Run prior selection demo standalone
python code/prior_selection.py

# Run sampling via runner (alternative)
python runners/run_rdu_bhm.py --mode test --num-chains 800
```

## Design Patterns

1. **Hub-and-Spoke (main.py)**: Central orchestration via boolean flags
2. **JAX Optimization**: JIT compilation and vectorization for GPU efficiency
3. **Indexed Lookups**: Precomputed unique values eliminate redundant computation
4. **Hierarchical Structure**: Subject priors from population, population from hyperpriors
5. **Non-Centered Parameterization**: Store epsilon ~ N(0,1), compute theta = mu + sigma * epsilon
6. **Theta-Preserving Updates**: COUPLED_TRANSFORM sampler for hyperparameters
