# TODO - Quantity-Space Prior Specification for BHMs

## Project Status: Submission-ready (June 2026)

Three-model JoE paper: CRRA utility with Prelec, Goldstein-Einhorn, and Power
PWFs, all sharing one quantity-space Beta-Beta prior. Estimation, marginal
likelihoods, paper integration, and the appendix pass are complete; the full
document compiles cleanly (pythontex 0 errors, no undefined references).

Final normalized Bayes factors (bayes_factors_rb2.npz, validated estimator
plus feasible-mass normalization): log BF crra/ge 16.05 +/- 0.35,
crra/power >= 107.4, ge/power >= 91.3 (Power normalizer is a rigorous
convexity bound; see technical_notes/bridge_sampling/rb2_results.md).

---

## Remaining before submission

- [ ] Human proofread of the full PDF (104 pp) — especially Section 5.4
      (model comparison) and Appendix E, which were rewritten on June 12
- [ ] Decide target length: current build is long; consider moving some
      appendix figures to an online supplement per JoE guidelines
- [ ] Cover letter, highlights, and submission metadata
- [x] Merge `feat/ml-repair` into `power-model`, then `power-model` into
      `master` (both merged; master 692c0a1 and later)
- [ ] Uncommitted `fixed_beta` patch in the bamcmc submodule
      (src/bamcmc/mcmc/config.py) — leftover from the deleted TI machinery;
      decide whether to commit upstream or discard
- [ ] Optional: archive or delete remaining legacy data
      (`rdu_bhm_uninf` 15G is still used to regenerate the Appendix D diffuse
      benchmark; `rdu_bhm_ce_stan` 133M is the sampler cross-check)

## Completed (June 2026, feat/ml-repair)

- [x] Marginal likelihoods: corrected RB estimator (marginal_likelihood.py),
      synthetic ground-truth validation PASSED (ml_validation.py),
      two psi-integration routes agree on all three models
- [x] Found and fixed: invalid IS in old RB bridge, Power psi-indexing bug,
      parallel-tempering files contaminating history/full (loader now filters
      on stored mcmc_config n_temperatures)
- [x] Big cleanup: deleted superseded ML machinery (full-space bridge, both
      TI variants), EP+TK and EP-EUT models, ~170 GB of tempered samples and
      EP outputs; registry now contains exactly crra/ge/power
- [x] Paper integration: model comparison subsection and Appendix E rewritten
      around the actual estimator, all numbers via pythontex from
      paper_data.npz (populated from bayes_factors_rb2.npz, ml_validation.npz,
      and history-file timing diagnostics)
- [x] Appendix pass: restored silently-missing sensitivity table rows
      (stale 'rho' NPZ key), renamed rho-named plot dirs to alpha naming,
      removed dead pythontex loads and the \pending macro
- [x] Project docs updated: CLAUDE.md, code/README.md, TODO.md

## Earlier milestones

- [x] Three-model restructure of paper and pipeline (power-model branch)
- [x] CE-space Beta-Beta pipeline for all three models, R-hat diagnostics
- [x] Paper plots: cross-model CE overlay, subject comparisons, CDF difference
- [x] (alpha, kappa) sensitivity analysis with diffuse NIG benchmark (App D)
- [x] Injectivity: analytic for Models 1 and 3, numerical for Model 2 (App B)

## Ideas / Future Work

- CPT simulation study (8-10 parameter model)
- Discounting simulation study (quasi-hyperbolic beta, delta)
- Expo-Power models (removed June 2026; recover from snapshot commit 0ce199f)
- Symmetric comparison: diffuse Beta-Beta to isolate distributional family
- Float64 re-run of the uninformative NIG benchmark (claude_review note)
