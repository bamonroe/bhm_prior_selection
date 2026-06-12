# Corrected RB Marginal Likelihoods — Final Results (June 12, 2026)

Handoff summary for the docs session. All numbers below come from
`data/output/bayes_factors_rb2.npz`, produced by
`code/marginal_likelihood.py` (`do_bayes_factors_rb2` in main.py) on the
`feat/ml-repair` branch. These supersede ALL previous marginal
likelihood outputs (see "Superseded results" below).

## Headline numbers

Log marginal likelihoods (IS route = primary; bridge route = cross-check):

| Model            | log ML (IS)        | log ML (bridge)    | route gap |
|------------------|--------------------|--------------------|-----------|
| CRRA + Prelec    | -5254.71 +/- 0.03  | -5254.74 +/- 0.02  | 0.030     |
| CRRA + GE        | -5271.32 +/- 0.02  | -5271.43 +/- 0.02  | 0.110     |
| CRRA + Power     | -5426.41 +/- 0.02  | -5426.42 +/- 0.02  | 0.009     |

Log Bayes factors (IS route):

| Comparison       | log BF             |
|------------------|--------------------|
| CRRA vs GE       | 16.60 +/- 0.03     |
| CRRA vs Power    | 171.70 +/- 0.03    |
| GE vs Power      | 155.09 +/- 0.03    |

All three comparisons are decisive on the Kass-Raftery scale (2 ln BF >
10 is "very strong"; here 2 log BF = 33, 343, 310). Ordering: Prelec >
GE >> Power. The data strongly favor two-parameter probability weighting,
and between the two-parameter PWFs favor Prelec over Goldstein-Einhorn.

The Power model marginal likelihood is now RELIABLE. The corrected
estimator needs only the (well-converged) beta=1 posterior — no
tempering — so the Gram-determinant/tempering pathology documented in
May never enters. Power actually has the best per-subject diagnostics
of the three models.

## NPZ keys (for gen_paper_data.py / pythontex)

Per model `{m}` in {crra, ge, power}:
- `{m}_log_ml_is`, `{m}_log_ml_is_se` — primary estimate and SE
- `{m}_log_ml_bridge`, `{m}_log_ml_bridge_se` — cross-check route
- `{m}_rep_is`, `{m}_rep_bridge` — per-replicate values (8 each)
- `{m}_ess_subject_min`, `{m}_ess_subject_med`, `{m}_psi_ess_mean`,
  `{m}_nonfinite_frac`, `{m}_routes_agree`

Pairwise: `log_bf_rb2_{m1}_vs_{m2}`, `log_bf_rb2_{m1}_vs_{m2}_se`.
Metadata: `K` (16384; GE rerun at `ge_K` = 32768), `R` (8), `seed`.

Validation artifact: `data/output/ml_validation.npz` (`passed` = True;
ground truth -72.384 +/- 0.027 from two independent brute-force
proposals agreeing to 0.009; estimator IS/bridge within 0.035/0.028).

## Method (one paragraph for the paper)

Rao-Blackwellized estimator: per-subject integrated likelihoods
p(y_i|psi) = E_q[f_i(theta) p_pop(theta|psi)/q_i(theta)] with f_i =
likelihood x Jacobian, using per-subject proposals (PC1-stratified
Student-t mixtures fitted to each subject's marginal posterior, refined
once by importance-resampling a probe of the integrand at eight
posterior psi draws). The 8-D hyperparameter integral
int pi(psi) prod_i p(y_i|psi) dpsi is computed two independent ways —
importance sampling from a Student-t fitted to the psi posterior, and
Meng-Wong bridge sampling — which must agree within SE. Because a
single theta cache biases the log estimate downward by roughly
sum_i 1/(2 ESS_i) and hides cache-level noise, the full procedure runs
on R=8 independent caches (K=16384 fresh draws per subject each) and
combines on the natural scale; SEs come from the replicate spread. The
estimator was validated end-to-end on a 5-subject synthetic dataset
against a brute-force ground truth (exact hyperprior outer draws), to
within 0.035 nats.

Sanity identity: E_post[log L] - log p(y) = KL(posterior || prior)
~ 556 nats for CRRA over 452 parameters (~1.2 nats/parameter) —
a plausible KL gap, unlike the old TI numbers which implied ~82.

## Known residual

The GE route gap (0.110 vs tolerance 0.100) is systematic, stable in K
(identical at K=16384 and 32768), and harmless at the BF scale. Likely
cause: the bridge route requires posterior draws to be exactly from
p_tilde, while the IS route is consistent regardless (posterior draws
only shape its proposal); a small residual over-dispersion in the GE
sample biases the bridge down by ~0.1. The IS route is therefore the
primary estimate for all models. Block-wise hyperparameter means show
both CRRA and GE stationary from block 2 of 6; the kept second half is
flat.

## Superseded results — do not cite

- `bayes_factors.npz` (May 25): full 452-D bridge sampling. The
  technical note documents the CRRA estimate was still drifting ~11
  log-units/iter at 2000 iterations. All levels unreliable.
- `bayes_factors_rb.npz` (May 25): the per-subject integral used
  posterior theta draws WITHOUT dividing by their proposal density —
  not a valid importance-sampling estimate. The Power entry was further
  corrupted by a psi-indexing bug (`_get_indices` from the 4-parameter
  CRRA module applied to the 3-parameter Power state; fixed in
  bayes_factors_rb.py).
- `bayes_factors_ti.npz` (Jun 9): two independent fatal problems.
  (i) The intermediate-beta rungs never equilibrated (E_beta[log L]
  flat from beta=0.02 to 1, impossible for a ~450-parameter model since
  dE/dbeta = Var_beta[log L]); (ii) the beta=1 rung loaded
  `rdu_bhm_ce/history/full`, which by June 9 contained the May 27
  parallel-tempering files (see below).

## Critical data gotcha (now handled in code)

`data/output/rdu_bhm_ce/history/full/history_007..010.npz` (May 27) are
PARALLEL-TEMPERING runs (n_temperatures=20, ladder down to beta=0.01).
Only ~1/20 of their saved chain states are beta=1 posterior draws.
Treating that directory as posterior samples mixes 20 tempered
distributions and displaces the apparent psi posterior by ~275 nats of
log p_tilde. `marginal_likelihood.load_posterior_states` now filters
history files on the `mcmc_config['n_temperatures']` metadata stored in
each NPZ. The paper's preprocessed CRRA summaries
(`preprocessed/hyperparams_preprocessed.npz`, Apr 30) predate these
files and are unaffected, as is `mcmc_diagnostics.npz` (May 20).
Recommendation: move TI/PT history files out of `history/full/` (e.g.
into the existing `ti/` and `pt/` subdirectories) to prevent future
accidents.

## Recommendation on thermodynamic integration

Skip the repaired-TI rerun. The result is already triangulated three
ways: two independent psi-integration routes agreeing within ~0.1 nats
on all models, an end-to-end synthetic ground-truth validation of the
exact pipeline, and the KL sanity identity. Repairing TI (denser ladder
near beta=0, long per-rung equilibration, sequential warm starts) would
cost GPU-days to add a third method family to a settled number. The
existing TI/PT machinery and runs can be kept for the technical record
or deleted.
