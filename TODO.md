# TODO - Quantity-Space Prior Specification for BHMs

## Project Status: JoE Paper — Single-Application (RDU with Real Data)

Paper restructured for Journal of Econometrics:
- 7 sections + 1 appendix (A: Global Injectivity)
- Single model: Beta-Beta CE-space hierarchy (`rdu_bhm_ce`)
- Comparison model: NIG parameter-space with Gao (2022) diffuse prior (`rdu_bhm`)
- Full pipeline: prior selection, sampling (both models), preprocessing, paper plots
- R-hat converged (max 1.21, median 1.004)
- Reference lotteries: L1=($10,$50, p=0.75/0.25), L2=($20,$40, p=0.50/0.50), L3=($12,$44, p=0.75/0.25)

**Current branch:** `master`

---

## Remaining

### Paper Content
- [ ] Address referee point 3: clarify in Section 3.2 that the Beta-Beta decomposition requires independence of μ and ψ (a parameterization choice, not a consequence of the general framework)
- [ ] Address referee point 4: discuss NCP unavailability as a computational trade-off (centered parameterization is forced by non-invertible CE mapping)
- [ ] Address referee point 5: sensitivity analysis over ρ/κ showing how the fraction of prior-dominated subjects changes
- [ ] Address referee minor 1: trim Section 2 belief elicitation review (tangential)
- [ ] Address referee minor 4: check for remaining "observable quantities" vs "predictive quantities" inconsistency
- [ ] Address referee minor 5: compute prior predictive CEs implied by the Gao et al. diffuse prior
- [ ] Address referee minor 6: expand MCMC details (R-hat distribution, acceptance rates)
- [ ] Address referee minor 12: add posterior summary table for structural parameters
- [ ] Address referee minor 13: add posterior predictive check on choice probabilities
- [ ] Proofread appendix (Global Injectivity) for consistency with current terminology

### Code
- [ ] Write automated tests for core functions
- [ ] Ensure main.py default flags are set to safe defaults (all sampling False)

---

## Completed

### Paper Structure & Content
- [x] Restructure for JoE (7 sections + appendix)
- [x] Section 3: quantity-space priors + variance decomposition + NIG example + heatmaps
- [x] Terminology: "generated quantities" → "predictive quantities"
- [x] Drop Appendices A (Full Covariance), C (Jacobian Correction), D (MCMC Details)
- [x] Drop hyper_comparison and predictive_params_overlay figures
- [x] Writing rules: no em-dashes, no side-by-side equations, no itemize, no textbf
- [x] Rename general hyperparameters from ψ to γ (avoid collision with Beta-Beta ψ_k)
- [x] Add limitation on independence of quantity-space marginals
- [x] Acknowledge asymmetric comparison (CE-space vs NIG)
- [x] Rewrite results for 3-panel CE overlay, 3-lottery subject comparisons
- [x] Add RDU signature paragraph (lottery-specific risk attitudes)
- [x] Add CDF difference figure and discussion
- [x] Strengthen subject 74 discussion with posterior decomposition
- [x] Revise Discussion: prose workflow, 6 limitations, no Bayes factor section
- [x] Revise Conclusion: empirical findings, 4p→4p reframing
- [x] Add code/data availability statement
- [x] Abstract updated with empirical findings
- [x] Fix stale pythontex (paper_quantities.npz → paper_data.npz)
- [x] Move inline calculations from pythontex to gen_paper_data.py
- [x] Simulated referee report written (technical_notes/referee_report/23_03_2026.tex)

### Estimation & Plots
- [x] CE-space Beta-Beta model: full pipeline with hyper-posterior optimization
- [x] Diffuse NIG model: sampling + preprocessing
- [x] Vectorized CE computation (1000x speedup)
- [x] Paper plots: CE overlay (3-panel), subject comparisons (3-lottery), CDF difference
- [x] Per-subject CE diagnostic plots (111 PDFs)
- [x] NIG sensitivity heatmaps (viridis, stacked panels)
- [x] Reference lotteries updated: p=0.25, L3=($12,$44)
- [x] R-hat converged (max 1.21, median 1.004)

---

## Ideas / Future Work

- **CPT simulation study** — Full Monte Carlo with 8-10 parameter CPT model
- **Discounting simulation study** — Quasi-hyperbolic (β, δ) model
- **MCMC appendix** — Rewrite for CE-space model (centered parameterization, no NCP)
- **Symmetric comparison** — Diffuse Beta-Beta model to isolate distributional family effect
- **ρ/κ sensitivity on posterior** — How does fraction of prior-dominated subjects change with ρ/κ
