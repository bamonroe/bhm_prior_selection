# Code: Quantity-Space Hyperprior Specification for BHMs

This directory contains all estimation, analysis, and plotting code for the
paper. The entire pipeline is controlled through `main.py`, which is the only
file that should be executed directly.

## Requirements

### Software

- **Python 3.12+** (tested with 3.14)
- **JAX** with CUDA support (GPU required for MCMC)
- **bamcmc** — Custom MCMC sampler (included as git submodule at `bamcmc/`)
- **LaTeX** with pythontex (for paper compilation)

### Python Environment

A virtual environment is provided at `pylib/`. Activate it before running:

```bash
source pylib/bin/activate
```

Key packages: `jax`, `jaxlib` (CUDA), `numpy`, `scipy`, `matplotlib`.

### Hardware

- NVIDIA GPU with at least 12 GB VRAM (tested on RTX 4070)
- The sampler runs 2000 parallel chains on GPU. Memory usage scales with
  `chunk_size` (iterations compiled per kernel call). Use `chunk_size=10`
  for 12 GB cards.

## Running the Pipeline

All analyses are controlled by boolean flags in `main.py`. To run a step:

1. Open `code/main.py`
2. Set the desired `do_*` flags to `True`
3. Run: `python code/main.py`
4. Reset flags to `False` when done

The pipeline is designed so that each step assumes its upstream dependencies
have already been run. Outputs are saved to `data/output/` and `plots/`.

**Important**: `main.py` is the only entry point. Do not run other scripts
directly. All code files are called through `main.py` via the `pipelines.py`
dispatch module.

## Pipeline Stages and Flags

### Data Pipeline

| Flag | Description |
|------|-------------|
| `do_clean_data` | Load raw experimental data from `data/input/`, parse into the indexed format used by the sampler, and save to `data/input/dbar_clean.npz`. Must be run once before any estimation. |
| `do_gen_paper_data` | Assemble pre-computed results (data summaries, prior configuration, MCMC diagnostics, cross-model CE medians) into `data/output/paper_data.npz` for pythontex. This file is the single source of truth for all numeric values in the paper. Does not run heavy computation — it loads results from other pipeline steps. |

### Prior Selection and Analysis

| Flag | Description |
|------|-------------|
| `do_generate_prior_ce` | Run the quantity-space prior selection pipeline: specify direct prior beliefs over CEs and choice probabilities, derive Beta-Beta hyperpriors via the (alpha, kappa) variance decomposition, verify via Monte Carlo, and save to `data/output/prior_params_ce.npz`. |
| `do_sensitivity` | Run the Beta-Beta sensitivity analysis over a grid of (alpha, kappa) values. Computes between-population SD, within-population SD, and within-population SD IQR for each grid point. Saves to `data/output/sensitivity_grid.npz` and generates a LaTeX table. |
| `do_family_comparison` | Run the hyperprior family comparison for Appendix A. Computes NIG, Beta-Beta, and Gamma-IG sensitivity heatmaps, KS distances between compound and direct distributions, and CDF difference curves. Saves to `data/output/family_comparison.npz` and generates plots in `plots/paper/`. This step uses multiprocessing and must not be run in the same process as JAX (it is called separately from sampling). |

### MCMC Sampling

Three RDU models are estimated, all sharing the same quantity-space prior:

| Model Key | Name | Parameters | Description |
|-----------|------|-----------|-------------|
| `crra` | CRRA + Prelec | r, phi, eta, lambda (4) | CRRA utility with 2-parameter Prelec PWF |
| `ge` | CRRA + GE | r, gamma, delta, lambda (4) | CRRA utility with 2-parameter Goldstein-Einhorn PWF |
| `power` | CRRA + Power | r, delta, lambda (3) | CRRA utility with 1-parameter Power PWF. The k > p case (4 quantities from 3 parameters, uses Gram determinant) |

Sampling flags are dictionaries keyed by model:

| Flag | Description |
|------|-------------|
| `do_sampling = {'crra': T/F, 'ge': T/F, 'power': T/F}` | Run MCMC estimation for each model. Uses the Metropolis-within-Gibbs sampler with A/B coupled proposal covariance. Each model runs independently with 2000 chains (1000 per A/B group). Configuration is in the `sampling_opts_*` dicts. |
| `do_sampling_prior = {'crra': T/F, 'ge': T/F, 'power': T/F}` | Run prior-only MCMC (no data likelihood). Used to verify the induced parameter-space prior and generate prior predictive plots. |
| `do_sampling_stan` | Run Stan/NUTS sampling for the CRRA model (alternative sampler, not the primary estimation method). |

### Preprocessing

| Flag | Description |
|------|-------------|
| `do_preprocessing = {'crra': T/F, 'ge': T/F, 'power': T/F}` | Extract hyperparameter and subject-level parameter posteriors from raw MCMC history files. For each model, produces `preprocessed/hyperparams_preprocessed.npz` (population-level hyperparameters) and `preprocessed/subjects/subject_XXX_params.npz` (per-subject parameter traces). The preprocessor auto-detects the number of subjects and parameters from the model registry. |

### Diagnostic Plots

| Flag | Description |
|------|-------------|
| `do_diagnostic_plots = {'crra': T/F, 'ge': T/F, 'power': T/F}` | Generate per-subject diagnostic PDFs for each model. Each subject gets a multi-page PDF with trace plots and posterior density plots. Uses `N_WORKERS` parallel workers. Output in `plots/{model_id}/subjects/`. |
| `do_subject_ce_plots` | Generate per-subject CE density plots for all three models. Each subject gets a 3-panel PDF (one per reference lottery) showing the posterior CE distribution. Uses parallel workers. Output in `plots/{model_id}/subject_ces/`. |

### Paper Plots

| Flag | Description |
|------|-------------|
| `do_paper_predictive_overlay` | Generate the cross-model CE overlay figure (3-panel, one per lottery, all three models + hyperprior predictive) and the subject-level detail figures (most agreed and most divergent subjects). These are the main results figures in Section 5. |
| `do_paper_prior_predictive_params` | Generate the prior predictive parameter distribution plot from the prior-only model samples. Used in Appendix D. |

### Diagnostics

| Flag | Description |
|------|-------------|
| `do_mcmc_diagnostics` | Compute per-subject split-R-hat for all parameters, aggregate convergence statistics, worst-20 parameter list, and acceptance rates for each model. Saves to `data/output/mcmc_diagnostics.npz`. Values are loaded by `do_gen_paper_data` for the MCMC appendix table. |

### Marginal Likelihoods

| Flag | Description |
|------|-------------|
| `do_ml_validation` | Validate the marginal likelihood estimator end to end on a small synthetic dataset (5 subjects x 20 choices) against a brute-force ground truth computed with exact hyperprior draws. Runs a short MCMC on the synthetic data the first time. Saves to `data/output/ml_validation.npz` with a `passed` flag. |
| `do_bayes_factors_rb2` | Compute log marginal likelihoods and pairwise Bayes factors for all three models with the Rao-Blackwellized estimator in `marginal_likelihood.py`: per-subject importance sampling with adaptively refined stratified Student-t mixture proposals, two independent hyperparameter-integration routes (Student-t importance sampling and Meng-Wong bridge), and R=8 replicated caches combined on the natural scale. Saves to `data/output/bayes_factors_rb2.npz`. Loaded by `do_gen_paper_data` for the model comparison section. IMPORTANT: the loader excludes parallel-tempering history files by checking the stored `mcmc_config['n_temperatures']`. |

### Appendix Analyses

| Flag | Description |
|------|-------------|
| `do_rk_sensitivity` | Re-estimate the CRRA model at alternative (alpha, kappa) configurations and compare to a diffuse benchmark. Generates CDF difference plots, CE overlays, and subject classification tables for Appendix D. This runs full MCMC estimation for each configuration. |
| `do_injectivity_analysis` | Run the CE mapping injectivity analysis (P-matrix condition verification on a grid) for Appendix B. |
| `do_injectivity_power` | Run the CRRA+Power Gram determinant rank check (k > p case) for Appendix B. |

### Document Compilation

| Flag | Description |
|------|-------------|
| `do_compile_doc` | Compile the paper: `pdflatex -> pythontex -> pdflatex -> biber -> pdflatex`. Requires the pylib Python environment for pythontex. |

## Typical Workflow

A full replication from scratch:

```python
# 1. Data and prior specification
do_clean_data = True
do_generate_prior_ce = True
do_gen_paper_data = True

# 2. MCMC estimation (hours on GPU)
do_sampling = {'crra': True, 'ge': True, 'power': True}

# 3. Postprocessing
do_preprocessing = {'crra': True, 'ge': True, 'power': True}
do_mcmc_diagnostics = True

# 4. Marginal likelihoods (minutes on GPU)
do_ml_validation = True
do_bayes_factors_rb2 = True
do_gen_paper_data = True  # re-run to pick up diagnostics and Bayes factors

# 5. Plots
do_diagnostic_plots = {'crra': True, 'ge': True, 'power': True}
do_paper_predictive_overlay = True
do_subject_ce_plots = True

# 6. Appendix analyses (optional, hours on GPU)
do_family_comparison = True  # run separately, not with sampling
do_rk_sensitivity = True
do_injectivity_analysis = True
do_injectivity_power = True

# 7. Compile paper
do_compile_doc = True
```

Note: `do_family_comparison` uses multiprocessing which conflicts with JAX's
GPU initialization. Run it in a separate invocation from any sampling or
JAX-dependent steps.

## Directory Structure

```
code/
├── main.py                  # Pipeline hub — the only file to run
├── model_registry.py        # Central model configuration (MODEL_REGISTRY)
├── pipelines.py             # Pipeline dispatch functions
├── prior_selection.py       # Quantity-space prior specification
├── clean_data.py            # Raw data loading and preprocessing
├── gen_paper_data.py        # Assemble results for pythontex
├── gen_subject_ce_plots.py  # Per-subject CE density plots (parallel)
├── family_comparison.py     # Appendix A: NIG/BB/Gamma family comparison
├── sensitivity.py           # Beta-Beta (alpha, kappa) sensitivity grid
├── mcmc_diagnostics.py      # R-hat, acceptance rates, convergence stats
├── plotting.py              # Diagnostic trace/density plots (parallel)
├── paper_plots.py           # Paper figures (CE overlays, subject details)
├── helpers.py               # Shared utilities
├── injectivity_analysis.py  # Appendix B: CE mapping injectivity check
├── injectivity_power.py     # Appendix B: Power Gram determinant rank check
├── marginal_likelihood.py   # Corrected RB marginal likelihood estimator
├── ml_validation.py         # Synthetic ground-truth validation of the estimator
├── prior_predictive_ce.py   # Prior predictive CE distributions
├── sampling_ce.py           # MCMC wrapper for CRRA+Prelec model
├── sampling_ge.py           # MCMC wrapper for CRRA+GE model
├── sampling_power.py        # MCMC wrapper for CRRA+Power model
├── sampling_ce_prior.py     # Prior-only MCMC for CRRA+Prelec
├── sampling_power_prior.py  # Prior-only MCMC for CRRA+Power
├── sampling_stan.py         # Stan/NUTS alternative sampler
└── posteriors/              # Model-specific log-posterior definitions
    ├── common_fns.py        # Shared: utility functions, PWFs, forward maps
    ├── rdu_bhm_ce.py        # CRRA+Prelec BHM posterior
    ├── rdu_ge_bhm_ce.py     # CRRA+GE BHM posterior
    ├── rdu_power_bhm_ce.py  # CRRA+Power BHM posterior
    ├── rdu_bhm_ce_prior.py  # CRRA+Prelec prior-only posterior
    ├── rdu_ge_bhm_ce_prior.py     # CRRA+GE prior-only posterior
    ├── rdu_power_bhm_ce_prior.py  # CRRA+Power prior-only posterior
    └── rdu_bhm.py           # Legacy NIG model (used by Appendix D only)
```

## Model Registry

All model-specific configuration is centralized in `model_registry.py`. Each
model is defined by a dict with its posterior module, sampling module, parameter
names, number of parameters, display settings, and utility/PWF types. Pipeline
functions use the registry to dispatch model-specific logic without hardcoding.

To add a new model: create a posterior module in `posteriors/`, a sampling
wrapper, and add an entry to `MODEL_REGISTRY` in `model_registry.py`. No
changes to `main.py`, `pipelines.py`, or plotting code are needed.
