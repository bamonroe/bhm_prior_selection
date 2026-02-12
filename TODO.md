# TODO - BHM Risk Preferences with Informed Prior

## Project Status: H&N 2016 Data Integrated, Paper In Progress

The full pipeline is implemented and working with Harrison & Ng (2016) data:
- Generalized to 3-outcome lotteries (unified format, both datasets supported)
- RDU-BHM model with 4-parameter Prelec PWF and NCP parameterization
- MCMC sampling completed for both informative and uninformative priors
- Prior and posterior predictive CE distributions computed
- Diagnostic plots generated for both prior specifications
- Paper draft updated to reference Harrison & Ng (2016)

**Current branch:** `feat/hn2019-data` (from `master`)

---

## Up Next

### Paper Completion
- [ ] Implement paper figures in `paper_plots()` — currently a stub
- [ ] Review and finalize results section in `doc/main.tex`
- [ ] Write discussion comparing informative vs uninformative posterior estimates
- [ ] Incorporate posterior predictive CE plots into paper (results or appendix)
- [ ] Final LaTeX compilation with pythontex

### Convergence
- [ ] Run longer sampling chains for H&N data (current: 500 iter, not fully converged per-run)
  - Post burn-in nested R-hat is good (max 1.02), but more iterations would improve
- [ ] Consider whether 500+500 chains is adequate or if more are needed

### Testing
- [ ] Write automated tests for core functions (no test suite currently exists)
  - Prior selection: verify variance decomposition identities
  - CE computation: known-value sanity checks
  - Data cleaning: verify choice encoding flip, 3-outcome padding
  - Likelihood: verify 3-outcome RDU matches 2-outcome when slot has prob=0

### Branch Management
- [ ] Squash/tidy commit history on `feat/hn2019-data`
- [ ] Merge `feat/hn2019-data` into `master` once pipeline is fully verified

### Housekeeping
- [ ] Decide whether to commit `technical_notes/coupled_transform_sampler/`
- [ ] Remove stale `data/input/hn2019_insurance.dta` and `data/input/jri/` if not needed

---

## Ideas / Future Work

- Sensitivity analysis on hyperprior choices (vary rho, alpha_0)
- Simulation study validating the methodology
- Extension to other utility models (EUT, CPT)
- Run Andersen et al. (2008) dataset through new 3-outcome pipeline for comparison

---

## Completed

### 1. Data Preparation
- [x] Obtain raw experimental data (Andersen, Harrison, Lau & Rutström 2008)
- [x] Place raw data in `data/input/dbar.csv`
- [x] Document data format and variables (see CLAUDE.md)
- [x] Implement `clean_data.py` — filter to RA task, indexed data structure
- [x] Fix choice encoding bug (choice=1 was right/B, not A)

### 2. Harrison & Ng (2016) Data Integration
- [x] Add `data/input/hn2016.csv` (111 subjects, 80 choices, 3-outcome lotteries, $10/$30/$50)
- [x] Generalize data pipeline to unified 3-outcome format
  - H&N: native 3-outcome, prizes sorted ascending, choice flipped (1-choice)
  - Andersen: binary padded to 3-outcome with zero-prob duplicate slot
- [x] Rewrite `make_bamcmc_data_indexed()` for 3-outcome format
  - static: (n_subjects, n_choices, n_unique_outcomes)
  - int: (choices, idx_L_1..3, idx_R_1..3) — 7 arrays
  - float: (unique_outcomes, prob_L_1..3, prob_R_1..3) — 7 arrays
- [x] Replace `rdu_value_indexed()` with `rdu_value_3outcome()` in common_fns.py
- [x] Update `risk_likelihood()` to accept phi/eta directly, compute PWF per-choice
- [x] Update `rdu_bhm.py` log_posterior to pass phi/eta and dataf to likelihood
- [x] Update all downstream code (sampling.py, gen_paper_data.py, prior_predictive_ce.py, pipelines.py, main.py)
- [x] Add DATASET and PRIZE_SCALE configuration to main.py
- [x] Update paper (`doc/main.tex`) to reference Harrison2016 throughout
- [x] Add Harrison2016 bib entry to `doc/library.bib`
- [x] Update `code/README.md` for 3-outcome format and dual-dataset support

### 3. Code Infrastructure
- [x] Create `code/main.py` hub with do-flags structure (22 flags)
- [x] Create `code/pipelines.py` dispatch layer
- [x] Create `code/helpers.py` with output path registry
- [x] Create `code/posteriors/` directory for model definitions
- [x] Set up pylib Python environment with JAX/numpy/bamcmc
- [x] Fix pipeline ordering: `do_posterior_ce_comparison` moved after uninformative CE step

### 4. Model Definition (Prelec 2-Parameter PWF)
- [x] Define RDU model posterior for bamcmc registry (`code/posteriors/rdu_bhm.py`)
  - 4 parameters per subject: r (CRRA), φ (Prelec curvature), η (Prelec elevation), λ (noise)
  - 8 hyperparameters (mean + logsd for each param)
  - Uses Prelec PWF: w(p) = exp(-η(-ln p)^φ)
  - Uses indexed lookups for efficient likelihood computation
- [x] Create common functions (`code/posteriors/common_fns.py`)
  - CRRA utility, Prelec probability weighting, 3-outcome RDU, risk likelihood
- [x] Non-Centered Parameterization (NCP) for better mixing
- [x] COUPLED_TRANSFORM sampler for hyperparameters

### 5. Prior Selection (Core Contribution)
- [x] Implement prior selection methodology (`code/prior_selection.py`)
- [x] Configure DIRECT_PRIOR in main.py (r, phi_raw, eta_raw, lambda_raw)
- [x] Configure researcher choices: rho=0.2, alpha_0=20.0
- [x] Configure uninformative hyperprior (diffuse, bypasses variance decomposition)
- [x] Prior distribution comparison plots (direct vs unconditional)

### 6. MCMC Sampling (H&N 2016 Data)
- [x] Run informative prior sampling (500 chains x 2 groups, 5 history files, 500 iter)
- [x] Run uninformative prior sampling (500 chains x 2 groups, 5 history files, 500 iter)
- [x] Preprocess hyperparameters and subjects for both prior types
- [x] Generate diagnostic plots (hyperparameters + per-subject trace/density)

### 7. Posterior Analysis
- [x] Extract posterior summaries (`gen_paper_quantities.py` → `paper_quantities.npz`)
  - Nested R-hat max 1.02 after burn-in (excellent convergence)
- [x] Generate paper data summaries (`gen_paper_data.py` → `paper_data.npz`)
- [x] Prior predictive CE distribution
  - Mean CE = $27.5, Median = $27.6 (vs EV = $30 — mild risk aversion)
- [x] Posterior predictive CE distribution (informative)
  - Mean CE = $25.3, Median = $25.2

### 8. Dead Code Cleanup
- [x] Remove `power_pwf()` (unused, replaced by `prelec_pwf`)
- [x] Remove direct sampler factories (~137 lines, superseded by COUPLED_TRANSFORM)
- [x] Remove `rdu_power_prior_selection()` alias
- [x] Remove `uninformative_prior_selection()` (unreachable)
- [x] Remove legacy `diagnostic_plots()` / `run_diagnostic_plots()` (superseded)
- [x] Remove `NUMPY_AVAILABLE` guard and JSON fallback in clean_data.py
- [x] Fix unused variables: `display_mean`, `n_params`, `likelihoods` unpacking
- [x] Remove unused imports: `import os`, `import json`, `import jax.random`
- [x] Fix `paper_plots()` key error (`n_samples` → `n_samples_post_burnin`)

### 9. Paper Draft
- [x] Write `doc/main.tex` with all major sections
  - Introduction, Prior Selection Approach, RDU Model, Results, Conclusion
  - Appendix A: Full covariance derivation
  - Appendix B: Skew-Normal 3-parameter extension
  - Appendix C: MCMC implementation details
- [x] Integrate pythontex for dynamic data from NPZ files
- [x] Skew-Normal example (`skew_normal_hyperprior.py`) for Appendix B

### 10. Technical Notes
- [x] BHM hyperprior derivation (`technical_notes/bhm_hyperprior_derivation/`)
- [x] Coupled transform sampler (`technical_notes/coupled_transform_sampler/`)

---

## Technical Details

### Current Prior Selection Values (rho=0.2, alpha_0=20.0)

| Parameter    | mu_0   | tau_0  | E[sigma] | Prior Var |
|--------------|--------|--------|----------|-----------|
| r            |  0.50  | 0.224  | 0.447    | 0.25      |
| phi_raw      |  0.00  | 0.173  | 0.346    | 0.15      |
| eta_raw      |  0.00  | 0.173  | 0.346    | 0.15      |
| lambda_raw   | -2.00  | 0.447  | 0.894    | 1.00      |

### MCMC Run Summary (H&N 2016)

| Prior         | History Files | Chains | Iters/File | Total Samples |
|---------------|---------------|--------|------------|---------------|
| Informative   | 5             | 1000   | 100        | 500 x 1000    |
| Uninformative | 5             | 1000   | 100        | 500 x 1000    |

### Dataset Summary

| Dataset | Subjects | Choices | Outcomes/Lottery | Prizes | Currency |
|---------|----------|---------|-----------------|--------|----------|
| H&N 2016 | 111 | 80 | 3 | $10, $30, $50 | USD |
| Andersen 2008 | 501 | 40 | 2 (padded to 3) | 50-4500 | DKK |

---

## References

1. **Harrison, G.W. & Ng, J.M. (2016).** "Evaluating the Expected Welfare Gain From Insurance." *Journal of Risk and Insurance*, 83(1), 91-120.

2. **Gao, X.S., Harrison, G.W., & Tchernis, R. (2022).** "Behavioral welfare economics and risk preferences: a Bayesian approach." *Experimental Economics*, 26, 273-303.

3. **Andersen, S., Harrison, G.W., Lau, M.I., & Rutström, E.E. (2008).** "Eliciting Risk and Time Preferences." *Econometrica*, 76(3), 583-618.
