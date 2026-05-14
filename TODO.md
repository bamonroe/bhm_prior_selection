# TODO - Quantity-Space Prior Specification for BHMs

## Project Status: JoE Paper — Two-Model Application (RDU with Real Data)

Paper restructured for Journal of Econometrics with two-model application:
- 7 sections + 4 appendices
- Model 1: CRRA + Prelec (`rdu_bhm_ce`) — full pipeline working
- Model 2: Expo-Power + TK92 (`rdu_ep_bhm_ce`) — code complete, MCMC in progress
- Comparison model: NIG parameter-space with Gao (2022) diffuse prior (`rdu_bhm`)
- Both models share identical quantity-space prior (same reference lotteries, same α/κ)
- Reference lotteries: L1=($10,$50, p=0.75/0.25), L2=($20,$40, p=0.50/0.50), L3=($12,$44, p=0.75/0.25)

**Current branch:** `feat/expo-power-model` (sub-branch: `ep-core-functions`)

---

## Remaining

### EP Model (In Progress)
- [x] Core functions: ep_ufn, ep_inverse, tk_pwf, forward map, Jacobian (common_fns.py)
- [x] Posterior module: rdu_ep_bhm_ce.py (imports shared Beta-Beta infrastructure)
- [x] Sampling infrastructure: sampling_ep.py, main.py config, pipelines.py
- [x] Smoke test: 10 chains, compilation + basic evaluation verified
- [x] Injectivity numerical verification: det(J) ≠ 0 on prior support
- [ ] Full MCMC estimation (6 runs, ~4.5 hours) — IN PROGRESS
- [ ] EP preprocessing (hyper + subject level)
- [ ] Paper plots: EP CE overlays, EP subject comparisons
- [ ] Cross-model comparison plots
- [ ] gen_paper_data.py: add EP model quantities

### Paper Content
- [x] Restructure Section 4: two RDU specifications side by side
- [x] Update abstract, introduction, discussion, conclusion for two models
- [x] Update CE mapping subsection to be model-agnostic
- [x] Update injectivity limitation to cover both models
- [x] Trim Extensions subsection (EP is no longer hypothetical)
- [ ] Add EP estimation results to Section 5
- [ ] Add cross-model comparison subsection to Section 5
- [ ] Update Appendix B with EP injectivity note
- [ ] Update sensitivity analysis for EP model
- [ ] Proofread full paper for consistency

### Code
- [ ] Write automated tests for core functions
- [ ] Ensure main.py default flags are set to safe defaults (all sampling False)
- [ ] Merge ep-core-functions into feat/expo-power-model
- [ ] Merge feat/expo-power-model into master

---

## Completed

### Paper Structure & Content
- [x] Restructure for JoE (7 sections + appendices)
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
