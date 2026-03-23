# TODO - Quantity-Space Prior Specification for BHMs

## Project Status: JoE Paper — Single-Application (RDU with Real Data)

Paper restructured for Journal of Econometrics:
- 6 sections + 2 appendices (A: Full Covariance, B: Global Injectivity)
- Single model: Beta-Beta CE-space hierarchy (`rdu_bhm_ce`)
- Comparison model: NIG parameter-space with Gao (2022) diffuse prior (`rdu_bhm`)
- Full pipeline: prior selection, sampling (both models), preprocessing, paper plots
- NIG sensitivity heatmaps (viridis, 3 stacked panels) in Section 3

**Current branch:** `master`

---

## Next Steps

### A. Immediate Cleanup (mechanical) — COMPLETE
- [x] Replace all remaining em-dashes (`---`) with colons/semicolons
- [x] Remove unused NPZ loads (`prior_params.npz`, `sensitivity_grid.npz`) from pythontex
- [x] Fix stale comment line 5 ("Appendix C (Sequential Identification)")
- [x] Verify pythontex CE overlay thresholds match reference lottery bounds

### B. Results Section Rewrite
- [ ] Rewrite CE overlay discussion for 3-panel format (currently describes single CE with $15/$40)
- [ ] Update parameter overlay text (currently mentions hyperprior predictive + diffuse, no informative posterior)
- [ ] Verify subject IDs in text match current `posterior_ce_comparison.npz`

### C. Structural
- [ ] Consider merging Section 4 (Application) and Section 5 (Estimation)
- [ ] Update CLAUDE.md appendix listing (still lists 4, now only 2)

### D. Convergence & Estimation
- [ ] Re-run MCMC with hyper-posterior optimization (Jacobian/likelihood removed from hyper proposals)
- [ ] Address max R-hat (~10.4): longer runs, identify problematic parameters, or discuss in limitations
- [ ] Verify paper_quantities.npz is regenerated after new MCMC run

### E. Final Polish
- [ ] Proofread all sections
- [ ] Code/data availability statement
- [ ] Expand bibliography if needed (~40+ references)
- [ ] Final cross-reference and figure verification in compiled PDF

### F. Testing
- [ ] Write automated tests for core functions
  - Prior selection: verify variance decomposition identities
  - CE computation: known-value sanity checks
  - Data cleaning: verify choice encoding flip
  - Likelihood: verify 3-outcome RDU matches 2-outcome when prob=0

---

## Completed

### Paper Structure & Content
- [x] Restructure for JoE (6 sections + appendices)
- [x] Section 3: quantity-space priors + variance decomposition + NIG example + heatmaps
- [x] Terminology: "generated quantities" → "predictive quantities"
- [x] Drop Appendices C (Jacobian Correction) and D (MCMC Details)
- [x] Drop hyper_comparison figure (incompatible across CE-space vs NIG models)
- [x] Writing rules: no em-dashes, no side-by-side equations, no itemize, no textbf

### Estimation & Plots
- [x] CE-space Beta-Beta model (`rdu_bhm_ce`): full pipeline
- [x] Diffuse NIG model (`rdu_bhm`): sampling + preprocessing
- [x] Paper comparison plots: predictive_params_overlay, predictive_ce_overlay (3-panel),
      subject_ce_most_affected, subject_ce_least_affected
- [x] NIG sensitivity heatmaps for Section 3 (viridis, stacked panels)
- [x] Fix hyper-posterior optimization: remove Jacobian/likelihood from hyper proposals
- [x] Fix CE overlay: draw from Beta-Beta hierarchy (not pooled subject draws)

### Infrastructure
- [x] `paper_plots.py`: correct hyper-posterior predictive computation
- [x] `gen_paper_data.py`: NIG sensitivity + heatmap generation
- [x] Recover `prior_predictive_ce.py`, `sampling.py` from git history
- [x] Wire uninformative model into `main.py` pipeline

---

## Ideas / Future Work

- **CPT simulation study** — Full Monte Carlo with 8-10 parameter CPT model
- **Discounting simulation study** — Quasi-hyperbolic (β, δ) model
- **MCMC appendix** — Rewrite for CE-space model (centered parameterization, no NCP)

---

## References

1. **Harrison, G.W. & Ng, J.M. (2016).** "Evaluating the Expected Welfare Gain From Insurance." *JRI*, 83(1), 91-120.
2. **Gao, X.S., Harrison, G.W., & Tchernis, R. (2022).** "Behavioral welfare economics and risk preferences: a Bayesian approach." *Exp Econ*, 26, 273-303.
3. **Andersen, S., Harrison, G.W., Lau, M.I., & Rutström, E.E. (2008).** "Eliciting Risk and Time Preferences." *Econometrica*, 76(3), 583-618.
