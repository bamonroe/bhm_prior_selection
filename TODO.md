# TODO - Quantity-Space Prior Specification for BHMs

## Project Status: JoE Paper — Single-Application (RDU with Real Data)

Paper restructured for Journal of Econometrics:
- 6 sections + 4 appendices
- Single model: Beta-Beta CE-space hierarchy (`rdu_bhm_ce`)
- Full pipeline implemented: prior selection, sampling, preprocessing, plotting
- Bibliography expanded from ~11 to ~30+ references
- Technical note material integrated into main paper body and appendices
- κ parameterization introduced (replaces α₀ as primary interface)
- Previous NIG parameter-space model (`rdu_bhm`) removed from all pipelines

**Current branch:** `master`

---

## Implementation Phases

### Phase 1: Paper Structure — COMPLETE
- [x] Restructure `doc/main.tex` for JoE
- [x] Write Sections 1–6 (Introduction through Conclusion)
- [x] Create Appendices A–D (Full Covariance, Global Injectivity, Jacobian Correction, MCMC)
- [x] Expand `doc/library.bib` from ~11 to ~30+ references

### Phase 2: κ Parameterization + Sensitivity — COMPLETE
- [x] Implement κ → α₀ mapping in `code/prior_selection.py`
- [x] Update `code/main.py` to use κ instead of α₀ in config
- [x] Run sensitivity grid over (ρ, κ) for existing RDU model
- [x] Generate sensitivity tables for paper
- [x] Update paper with sensitivity results

### Phase 3: CE-Space Priors — COMPLETE
- [x] Implement CE Jacobian computation via `jax.jacfwd` (`common_fns.py`)
- [x] Implement CE-space posterior (`code/posteriors/rdu_bhm_ce.py`)
- [x] Implement CE-space prior selection (`code/prior_selection.py`)
- [x] Implement CE-space sampling module (`code/sampling_ce.py`)
- [x] Add CE config and pipeline wiring (`main.py`, `pipelines.py`)
- [x] Derive Beta-of-Betas hierarchical moment-matching formulas
- [x] Implement Beta population distribution in `rdu_bhm_ce.py`
- [x] Implement choice probability prior for λ (block-triangular Jacobian)
- [x] Run full MCMC with CE-space priors on H&N 2016 data
- [x] Generate comparison figures (informative vs diffuse hyperpriors)
- [x] Remove NIG parameter-space model from all pipelines

### Phase 4: Simulation Studies — DEFERRED (future paper)
CPT and discounting stub sections cut from current paper scope.
See "Ideas / Future Work" below.

### Phase 5: Polish
- [ ] Expand bibliography further if needed (~40+ references)
- [ ] Figure captions, notation consistency across sections
- [ ] Code/data availability statement
- [ ] Proofread, compile with pythontex
- [ ] Final verification: all figures, tables, cross-references

---

## Up Next (immediate)

### Paper Polish
- [ ] Proofread all sections for typos and consistency
- [ ] Verify LaTeX compilation with pythontex
- [ ] Check all cross-references resolve correctly

### Testing
- [ ] Write automated tests for core functions
  - Prior selection: verify variance decomposition identities
  - CE computation: known-value sanity checks
  - Data cleaning: verify choice encoding flip
  - Likelihood: verify 3-outcome RDU matches 2-outcome when prob=0

---

## Previously Completed (Economics Letter version)

### Data Preparation
- [x] H&N 2016 data integration (111 subjects, 80 choices, 3-outcome)
- [x] Andersen 2008 data (501 subjects, 40 choices, padded 2→3 outcome)
- [x] Unified 3-outcome format in `clean_data.py`

### Model & Estimation
- [x] RDU-BHM posterior with Prelec 2-param PWF
- [x] Prior selection methodology (`prior_selection.py`)
- [x] MCMC sampling with both informative and diffuse hyperpriors
- [x] Posterior summaries

### Paper Plots
- [x] Hyperparameter posterior comparison (informative vs diffuse vs hyperprior)
- [x] Predictive parameter overlay (r, φ, η, λ)
- [x] Predictive CE overlay (prior/informative/uninformative)
- [x] Subject CE comparison (most/least affected)

### Infrastructure
- [x] `main.py` hub with do-flags
- [x] Config consolidation
- [x] pythontex integration for dynamic data
- [x] Technical notes: BHM derivation, coupled transform, prior predictive extension, Jacobian correction, over-determined CE priors

---

## Ideas / Future Work

- **CPT simulation study** — Full Monte Carlo with 8-10 parameter CPT model; strongest motivating example for the framework (32-40 hyperprior choices). Would add as separate paper section with DGP × sample size × replication grid.
- **Discounting simulation study** — Quasi-hyperbolic (β, δ) model; natural setting for quantity-space priors since generated quantities (present value equivalents) are directly elicited in experiments.
- **Simulation infrastructure** — `code/posteriors/cpt_bhm.py`, `code/posteriors/discounting_bhm.py`, `code/simulation/dgp.py`, `code/simulation/monte_carlo.py`

---

## References

1. **Harrison, G.W. & Ng, J.M. (2016).** "Evaluating the Expected Welfare Gain From Insurance." *Journal of Risk and Insurance*, 83(1), 91-120.

2. **Gao, X.S., Harrison, G.W., & Tchernis, R. (2022).** "Behavioral welfare economics and risk preferences: a Bayesian approach." *Experimental Economics*, 26, 273-303.

3. **Andersen, S., Harrison, G.W., Lau, M.I., & Rutström, E.E. (2008).** "Eliciting Risk and Time Preferences." *Econometrica*, 76(3), 583-618.
