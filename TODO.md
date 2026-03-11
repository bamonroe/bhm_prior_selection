# TODO - Observable-Space Prior Specification for BHMs

## Project Status: JoE Restructure — Single-Application Paper

Paper restructured from Economics Letter to Journal of Econometrics:
- 7 sections + 5 appendices (was 4+3)
- Bibliography expanded from ~11 to ~30+ references
- Technical note material integrated into main paper
- κ parameterization introduced (replaces α₀ as primary interface)
- CPT, discounting stub sections and Appendix F removed (future work)

**Current branch:** `feat/joe-restructure` (Phase 1 complete, not yet merged)

---

## Implementation Phases

### Phase 1: Paper Structure (no new computation) — COMPLETE
- [x] Create feature branch `feat/joe-restructure`
- [x] Restructure `doc/main.tex` into 7 sections + 5 appendices
- [x] Write Section 1: Introduction (broadened for JoE)
- [x] Write Section 2: Literature — Two Camps of Prior Specification
- [x] Write Section 3: Observable-Space Prior Specification
- [x] Write Section 4: Hyperprior Derivation with κ parameterization
- [x] Adapt Section 5: RDU Application from existing material
- [x] Write stub Sections 6-7 (CPT, Discounting) — later removed
- [x] Write Section 8: Discussion, Section 9: Conclusion
- [x] Create Appendices C (Global Injectivity) and D (Jacobian Correction)
- [x] Expand `doc/library.bib` from ~11 to ~30+ references
- [x] Update CLAUDE.md and TODO.md
- [ ] Verify LaTeX compilation
- [ ] Git commit on feature branch

### Phase 2: κ Parameterization + Sensitivity — COMPLETE
- [x] Implement κ → α₀ mapping in `code/prior_selection.py`
- [x] Update `code/main.py` to use κ instead of α₀ in config
- [x] Run sensitivity grid over (ρ, κ) for existing RDU model
- [x] Generate sensitivity tables for paper
- [x] Update paper with sensitivity results

### Phase 3: CE-Space Priors — IN PROGRESS
- [x] Implement CE Jacobian computation via `jax.jacfwd` (`common_fns.py`)
- [x] Implement CE-space posterior (`code/posteriors/rdu_bhm_ce.py`)
- [x] Implement CE-space prior selection (`code/prior_selection.py`)
- [x] Implement CE-space sampling module (`code/sampling_ce.py`)
- [x] Add CE config and pipeline wiring (`main.py`, `pipelines.py`)
- [x] Smoke test MCMC with CE-space priors (10 iter, 40 chains — passes)
- [x] Technical note: over-determined CE priors / Gram determinant
- [x] Technical note: cross-domain priors and data-uninformed parameters
- [x] Technical note: Beta-Beta hierarchy for bounded CEs (prior predictive extension)
- [ ] Derive Beta-of-Betas hierarchical moment-matching formulas (code implementation)
- [ ] Implement Beta population distribution in `rdu_bhm_ce.py` (replacing Normal)
- [ ] Run full MCMC with CE-space priors on H&N 2016 data
- [ ] Generate comparison figures (parameter-space vs CE-space vs diffuse)

### Phase 4: Choice Probability Prior for λ
- [ ] Derive ∂P/∂λ explicitly; verify monotonicity
- [ ] Implement choice-prob mapping for λ identification
- [ ] Implement 4D joint specification with block-triangular Jacobian
- [ ] Test on H&N 2016 data

### Phase 5: Simulation Studies — DEFERRED (future paper)
CPT and discounting stub sections cut from current paper scope.
See "Ideas / Future Work" below.

### Phase 6: Polish
- [ ] Expand bibliography further if needed (~40+ references)
- [ ] Figure captions, notation consistency across sections
- [ ] Code/data availability statement
- [ ] Proofread, compile with pythontex
- [ ] Final verification: all figures, tables, cross-references

---

## Up Next (immediate)

### Phase 3 Completion
- [ ] Implement Beta-Beta hyperprior in code (replacing NIG for CE-space)
- [ ] Update `rdu_bhm_ce.py` to use Beta population distribution
- [ ] Full MCMC run with CE-space priors on H&N 2016 data
- [ ] Compare posteriors: parameter-space vs CE-space priors

### Convergence (ongoing)
- [ ] Run longer sampling chains for H&N data if needed
- [ ] Consider whether 500+500 chains is adequate

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
- [x] RDU-BHM posterior with Prelec 2-param PWF, NCP, COUPLED_TRANSFORM
- [x] Prior selection methodology (`prior_selection.py`)
- [x] MCMC sampling (1000 chains, both informative and uninformative)
- [x] Posterior summaries (R-hat max 1.002)

### Paper Plots
- [x] Prior distribution comparison (direct vs unconditional)
- [x] Hyperparameter posterior comparison (8-panel)
- [x] Predictive CE overlay (prior/informative/uninformative)
- [x] Predictive parameter overlay (r, φ, η, λ)
- [x] Subject CE comparison (most/least affected)

### Infrastructure
- [x] `main.py` hub with 22 do-flags
- [x] Config consolidation (7 config dicts)
- [x] pythontex integration for dynamic data
- [x] Skew-Normal example for Appendix B
- [x] Technical notes: BHM derivation, coupled transform, prior predictive extension, Jacobian correction, over-determined CE priors

---

## New Code Required (Phases 2-5)

| Component | File | Phase | Status |
|-----------|------|-------|--------|
| κ parameterization | `code/prior_selection.py` | 2 | Done |
| Sensitivity grid | `code/sensitivity.py` | 2 | Done |
| CE Jacobian | `code/posteriors/common_fns.py` | 3 | Done |
| CE-space log-posterior | `code/posteriors/rdu_bhm_ce.py` | 3 | Done (Normal) |
| CE-space prior selection | `code/prior_selection.py` | 3 | Done (Normal) |
| CE-space sampling | `code/sampling_ce.py` | 3 | Done |
| Beta hierarchy (code) | `code/posteriors/rdu_bhm_ce.py` | 3 | TODO |
| Choice prob prior | `code/posteriors/rdu_bhm.py` | 4 | |
| CPT posterior | `code/posteriors/cpt_bhm.py` | 5 | |
| Discounting posterior | `code/posteriors/discounting_bhm.py` | 5 | |
| Simulation DGP | `code/simulation/dgp.py` | 5 | |
| Monte Carlo runner | `code/simulation/monte_carlo.py` | 5 | |

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
