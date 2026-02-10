# TODO - BHM Risk Preferences with Informed Prior

## Project Status: Estimation Complete, Paper In Progress

The full pipeline is implemented and working:
- RDU-BHM model with 4-parameter Prelec PWF and NCP parameterization
- MCMC sampling completed for both informative and uninformative priors
- Prior and posterior predictive CE distributions computed
- Diagnostic plots generated for both prior specifications
- Paper draft with pythontex integration loading posterior summaries

---

## Up Next

### Paper Completion
- [ ] Generate paper figures (`do_paper_plots`) — plots/paper/ is currently empty
- [ ] Review and finalize results section in `doc/main.tex`
- [ ] Write discussion comparing informative vs uninformative posterior estimates
- [ ] Compare posterior estimates to Gao et al. (2022) results
- [ ] Incorporate posterior predictive CE plots into paper (results or appendix)
- [ ] Final LaTeX compilation with pythontex

### Testing
- [ ] Write automated tests for core functions (no test suite currently exists)
  - Prior selection: verify variance decomposition identities
  - CE computation: known-value sanity checks
  - Data cleaning: verify choice encoding flip

### Housekeeping
- [ ] Decide whether to commit `data/output_uninformative/` or add to `.gitignore`
- [ ] Decide whether to commit `technical_notes/coupled_transform_sampler/`
- [ ] Update `code/README.md` — still references "power probability weighting" in places

---

## Ideas / Future Work

- Sensitivity analysis on hyperprior choices (vary rho, alpha_0)
- Simulation study validating the methodology
- Extension to other utility models (EUT, CPT)
- Additional datasets beyond Andersen et al. (2008)

---

## Completed

### 1. Data Preparation
- [x] Obtain raw experimental data (Andersen, Harrison, Lau & Rutström 2008)
- [x] Place raw data in `data/input/dbar.csv`
- [x] Document data format and variables (see CLAUDE.md)
- [x] Implement `clean_data.py` — filter to RA task, indexed data structure
- [x] Fix choice encoding bug (choice=1 was right/B, not A)

### 2. Code Infrastructure
- [x] Create `code/main.py` hub with do-flags structure (21 flags)
- [x] Create `code/pipelines.py` dispatch layer
- [x] Create `code/helpers.py` with output path registry
- [x] Create `code/posteriors/` directory for model definitions
- [x] Set up pylib Python environment with JAX/numpy/bamcmc
- [x] Implement indexed data optimization (13 unique outcomes, 10 unique cumprobs)

### 3. Model Definition (Prelec 2-Parameter PWF)
- [x] Define RDU model posterior for bamcmc registry (`code/posteriors/rdu_bhm.py`)
  - 4 parameters per subject: r (CRRA), φ (Prelec curvature), η (Prelec elevation), λ (noise)
  - 8 hyperparameters (mean + logsd for each param)
  - Uses Prelec PWF: w(p) = exp(-η(-ln p)^φ)
  - Uses indexed lookups for efficient likelihood computation
- [x] Create common functions (`code/posteriors/common_fns.py`)
  - CRRA utility, Prelec probability weighting, risk likelihood
- [x] Non-Centered Parameterization (NCP) for better mixing
- [x] COUPLED_TRANSFORM sampler for hyperparameters
  - Theta-preserving updates: eps' = (theta - mu') / sigma'
  - Likelihood cancels from acceptance ratio

### 4. Prior Selection (Core Contribution)
- [x] Implement prior selection methodology (`code/prior_selection.py`)
  - `rdu_prelec_prior_selection()`: Define individual-level priors for 4-param model
  - `derive_hyperpriors()`: Variance decomposition with rho, alpha_0
  - `verify_hyperpriors()`: Confirm unconditional variance matches target
- [x] Configure DIRECT_PRIOR in main.py (r, phi_raw, eta_raw, lambda_raw)
- [x] Configure researcher choices: rho=0.2, alpha_0=20.0
- [x] Configure uninformative hyperprior (diffuse, bypasses variance decomposition)
- [x] Prior distribution comparison plots (direct vs unconditional)

### 5. MCMC Sampling
- [x] Run informative prior sampling (500 chains x 2 groups, 8 history files)
- [x] Run uninformative prior sampling (500 chains x 2 groups, 5 history files)
- [x] Preprocess hyperparameters and subjects for both prior types
- [x] Generate diagnostic plots (hyperparameters + per-subject trace/density)
  - 1,011 diagnostic plot files across both prior types

### 6. Posterior Analysis
- [x] Extract posterior summaries (`gen_paper_quantities.py` → `paper_quantities.npz`)
- [x] Generate paper data summaries (`gen_paper_data.py` → `paper_data.npz`)
- [x] Prior predictive CE distribution (`prior_predictive_ce.py`)
- [x] Posterior predictive CE distribution (informative + uninformative)
  - Informative: Mean CE = 1691 DKK, Median = 1731 DKK (vs EV = 2275 DKK)
  - Uninformative: Mean CE = 1698 DKK, Median = 1736 DKK
  - Both show risk aversion; similar posteriors indicate data-dominated estimation

### 7. Paper Draft
- [x] Write `doc/main.tex` with all major sections
  - Introduction, Prior Selection Approach, RDU Model, Results, Conclusion
  - Appendix A: Full covariance derivation
  - Appendix B: Skew-Normal 3-parameter extension
  - Appendix C: MCMC implementation details
- [x] Integrate pythontex for dynamic data from NPZ files
- [x] Skew-Normal example (`skew_normal_hyperprior.py`) for Appendix B

### 8. Technical Notes
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

Variance decomposition: `V = tau_0^2 + E[sigma^2]` where:
- tau_0^2 = rho * V (uncertainty about population mean)
- E[sigma^2] = (1-rho) * V (population heterogeneity)

### MCMC Run Summary

| Prior         | History Files | Chains | Iters/File | Total Samples |
|---------------|---------------|--------|------------|---------------|
| Informative   | 7 (+ gap)     | 1000   | 100        | ~700 x 1000   |
| Uninformative | 5             | 1000   | 100        | ~500 x 1000   |

---

## References

1. **Gao, X.S., Harrison, G.W., & Tchernis, R. (2022).** "Behavioral welfare economics and risk preferences: a Bayesian approach." *Experimental Economics*, 26, 273-303.
   https://doi.org/10.1007/s10683-022-09751-0

2. **Andersen, S., Harrison, G.W., Lau, M.I., & Rutström, E.E. (2008).** "Discounting behavior: A reconsideration." *European Economic Review*, 52(8), 1342-1378.

3. CEAR Working Papers: https://cear.gsu.edu/
