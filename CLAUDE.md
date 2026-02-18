# Workspace Instructions for Claude

Instructions for Claude Code sessions in this workspace.

## Project Status

**Current State:** Full pipeline working. Paper draft nearly complete.

The full estimation pipeline is implemented with Harrison & Ng (2016) data:
- RDU-BHM model with Prelec 2-parameter PWF: w(p) = exp(-η(-ln p)^φ)
- 4 parameters per subject: r (CRRA), φ (Prelec curvature), η (Prelec elevation), λ (noise)
- MCMC sampling complete for both informative and uninformative priors (1000 chains, R-hat max 1.002)
- Prior and posterior predictive CE distributions computed
- All paper plots generated: prior comparison, hyperparameter posteriors, predictive overlays, subject CE
- Paper draft with introduction, methodology, application, results, and discussion
- Appendices: covariance derivation, Skew-Normal extension, MCMC details
- Hardcoded plot/computation config consolidated into main.py config dicts

**Current branch:** `master` (all feature branches merged)

See `TODO.md` for detailed task tracking.

## Project Goal

Working title: "Principled Hyperprior Selection for Bayesian Hierarchical Models of Risk Preferences"

The goal is to write an Economics Letter demonstrating a principled approach to
selecting hyperpriors for Bayesian Hierarchical Models (BHMs).

**Core Insight:** It is far easier to specify a prior for a single random
individual than to directly specify population-level hyperpriors. Given an
individual-level prior, we derive hyperpriors such that the unconditional BHM
distribution matches this prior exactly.

**Key Result:** For a model with p parameters, this reduces 4p hyperprior
choices to just 2 interpretable parameters:
- **ρ (rho):** Variance partition ratio - fraction of variance from uncertainty
  about population mean vs. population heterogeneity
- **α₀ (alpha_0):** Concentration parameter - confidence in prior selection variance

We apply this to Rank-Dependent Utility (RDU) estimation, replicating:

> Gao, Harrison & Tchernis (2022). "Behavioral welfare economics and risk
> preferences: a Bayesian approach." *Experimental Economics*.
> https://doi.org/10.1007/s10683-022-09751-0

A copy is in the `lit/` directory.


## Key Code Files

**For detailed code architecture, see `code/README.md`.**

### Core Implementation

- **`code/prior_selection.py`** - The paper's core contribution
  - `rdu_prelec_prior_selection()`: Define individual-level priors for RDU params
  - `derive_hyperpriors(ps, rho=0.2, alpha_0=20.0)`: Variance decomposition
  - `verify_hyperpriors()`: Confirm unconditional variance matches target

- **`code/posteriors/rdu_bhm.py`** - BHM log posterior for bamcmc
  - 4 params per subject: r (CRRA), φ (Prelec curvature), η (Prelec elevation), λ (noise)
  - 8 hyperparameters: population mean + log-sdev for each
  - Uses Prelec PWF: w(p) = exp(-η(-ln p)^φ)
  - Non-Centered Parameterization (NCP) for better mixing

- **`code/posteriors/common_fns.py`** - Shared utility functions
  - CRRA utility with normalization, Prelec probability weighting, risk likelihood

- **`code/prior_predictive_ce.py`** - CE distribution analysis
  - Prior predictive: MC draws from hyperprior → subject params → CE
  - Posterior predictive: loads MCMC history → extracts GQ params → CE
  - Both informative and uninformative, with prior/posterior overlay plots

- **`code/clean_data.py`** - Data preparation
  - Filters to RA task, creates indexed data structure for GPU efficiency
  - Flips choice encoding: raw 1=right/B → model 1=A

- **`code/skew_normal_hyperprior.py`** - Appendix B example
  - Demonstrates methodology for 3-parameter Skew-Normal distribution
  - Monte Carlo integration for law of total cumulance
  - Numerical root-finding for hyperprior parameters
  - Comparison plots: direct prior vs unconditional distribution

### Running Sampling

```bash
source /workspace/pylib/bin/activate
python runners/run_rdu_bhm.py --mode test --num-chains 800
```

Options: `--mode test|full`, `--num-chains N`, `--show-priors`

### Model Parameters

The RDU-BHM model uses 4 parameters per subject:
- **r**: CRRA risk aversion coefficient
- **φ (phi)**: Prelec curvature (log scale internally)
- **η (eta)**: Prelec elevation (log scale internally)
- **λ (lambda)**: Fechner noise (log scale internally)

Prelec probability weighting: `w(p) = exp(-η(-ln p)^φ)`
- When φ = η = 1: linear PWF (Expected Utility)
- φ < 1: inverse-S shape (overweight small p, underweight large p)

### Variance Decomposition

Variance decomposition: `V = τ₀² + E[σ²]` where:
- τ₀² = ρ × V (uncertainty about population mean)
- E[σ²] = (1-ρ) × V (population heterogeneity)

Default researcher choices: ρ = 0.2, α₀ = 20.0

## Data

Two datasets are supported via the `DATASET` config in main.py (currently `'hn2016'`).

### Primary Dataset: `data/input/hn2016.csv` (Harrison & Ng 2016)

**Harrison, G.W. & Ng, J.M. (2016).** "Evaluating the Expected Welfare Gain
From Insurance." *Journal of Risk and Insurance*, 83(1), 91-120.

- 111 subjects, 80 choices each
- 3-outcome lotteries with prizes $10, $30, $50 (USD)
- Binary choice between left and right lotteries

### Secondary Dataset: `data/input/dbar.csv` (Andersen et al. 2008)

**Andersen, S., Harrison, G.W., Lau, M.I., & Rutström, E.E. (2008).**
"Discounting behavior: A reconsideration."
*European Economic Review*, 52(8), 1342-1378.

- 501 subjects, 40 risk aversion choices each
- 2-outcome lotteries (padded to 3-outcome format), prizes 50-4500 DKK
- Joint elicitation of risk and time preferences

Both datasets are cleaned by `clean_data.py` into a unified 3-outcome format.


## Git

Maintaining a healthy git repo is of very high importance for this project.

There is a master branch which is assumed to hold always working code.

From the master branch new features will be added by first checking out a
feature branch.

If while developing a feature, there are new ideas about how to complete the
feature, create a "sub feature" branch that branches off this feature branch.

Branches should never be deeper than a feature and sub-feature branch.

Commits to the sub-feature branches should be made extremely atomically with
the idea that they may be discarded or rolled back.

The git history unique to the sub-feature branch should be squashed before being
merged into the feature branch.

The feature branch should also have atomic commits, but if there are many steps
to be undertaken, this is when a sub-feature should be branched.

The feature branch should have its git commit history reduced but not fully
squashed before being merged into master.

Before merging the feature branch, ensure documentation about the feature is
written in an appropriate area. This documentation is meant to be for humans,
and also for future Claude sessions to get up to speed.

The final step before merging is to make sure that existing tests pass, or are
rewritten when there are fundamental changes, and that appropriate tests are
written to make sure that the new feature works as intended.

The feature branch is only merged into master when its goal is working
correctly - even if in a limited nature.

*Open sub-feature branches frequently!*

**ALWAYS PROMPT WHEN MERGING ANY BRANCH!**

## Project structure

The code will make use of the `bamcmc` sampler added as a submodule to this git
repo. This sampler is written in python and makes extensive use of JAX.

```bash
source /workspace/pylib/bin/activate
```
Always use this environment for Python code, tests, and scripts.

The directory structure:

```
pylib/                  # python venv which contains all necessary packages
bamcmc/                 # the git submodule of the bamcmc sampler
runners/                # CLI scripts for sampling, diagnostics, data download
data/input/             # Raw data from the paper
data/output/$model/     # MCMC output (checkpoints/, history/full/)
code/                   # Location of the python code used to estimate the data
plots/{$model, paper}/  # generated plots used in the paper or for diagnostics
technical_notes/$note/  # LaTeX technical documentation
lit/                    # Published academic papers that are relevent
```

In the above `$model` stands in for the name of a model being used. We should
anticipate that multiple models will be developed and tested.

In the above `$note` is meant to indicate that each note is in a subdirectory

## Code

In `code/` there will be a structure that follows a hub and spoke model where a
`main.py` file will be the hub. This file will have a series of `do` flags that
toggle which aspect of the project is being run. Each do flag will assume that
the previous step has been completed, and throw an error if it hasn't. The
`main.py` file should have extremely little code other than the flags to call
other python files.

```python
# main.py control flags:

# --- Data Pipeline ---
do_clean_data = False       # Shape raw data for bamcmc
do_gen_paper_data = False   # Generate data summaries for paper (paper_data.npz)

# --- Prior Selection & Analysis ---
do_generate_prior = False       # Run prior selection methodology
do_prior_predictive_ce = False  # Prior predictive CE distribution

# --- MCMC Sampling ---
do_sampling = False               # Run informative prior sampling
do_sampling_uninformative = False # Run uninformative prior sampling

# --- Preprocessing (extract from history files) ---
do_preprocess_hyper_info = False     # Preprocess hyperparameters (informative)
do_preprocess_subjects_info = False  # Preprocess subjects (informative)
do_preprocess_hyper_uninf = False    # Preprocess hyperparameters (uninformative)
do_preprocess_subjects_uninf = False # Preprocess subjects (uninformative)

# --- Posterior Analysis (compute summaries from preprocessed data) ---
do_gen_paper_quantities = False          # Extract posterior summaries (paper_quantities.npz)
do_posterior_predictive_ce_info = False   # Posterior predictive CE (informative)
do_posterior_predictive_ce_uninf = False  # Posterior predictive CE (uninformative)

# --- Diagnostic Plots ---
do_hyper_plots_info = False              # Hyperparameter diagnostic plots (informative)
do_subject_plots_info = False            # Subject diagnostic plots (informative)
do_hyper_plots_uninf = False             # Hyperparameter diagnostic plots (uninformative)
do_subject_plots_uninf = False           # Subject diagnostic plots (uninformative)

# --- Paper Plots ---
do_paper_prior_comparison = False    # Prior distribution comparison
do_paper_hyper_comparison = False    # Hyperparameter prior vs posterior (8-panel)
do_paper_predictive_overlay = False  # Predictive CE and parameter overlay
do_paper_subject_ce = False          # Subject most/least affected CE

# --- Appendix Examples ---
do_skew_normal_example = False   # Run Skew-Normal hyperprior example

# --- Document ---
do_compile_doc = False           # Compile LaTeX document
```

## Technical Notes

Technical notes are detailed notes written in their own subdirectories, written
in latex.

When writing a note, there should be extensive detail about the topic. I have a
high degree of technical proficiency in math, statistics, economics, code, and
finance. I expect very detailed summaries of any issues where a technical note
is asked for, not just a high level overview. These should be written in the
style of a detailed report.


## MCMC Configuration Guidelines

When configuring the bamcmc sampler, keep in mind:

### Chain Count

The number of chains should be at least equal to the size of the lower triangle
of the covariance matrix of the largest block. For a block of size `n`, this is:

```
min_chains = n * (n + 1) / 2
```

For example, with a maximum block size of 15 parameters:
- Lower triangle size: 15 × 16 / 2 = 120
- Recommended chains: 200-500 (per A/B group)

This is acceptable because we run on GPU, not CPU — hundreds of chains run in
parallel with minimal additional cost.

### Block Size

Keep blocks small (≤15 parameters) for good acceptance rates. Large blocks
(50+ parameters) will have very low acceptance rates even with adaptive
proposals like MCOV_MODE.

### Burn-in Strategy

Prefer keeping the full history and dropping the first half during inference,
rather than using the `burn_iter` parameter. This preserves the complete trace
for diagnostics and allows flexible burn-in decisions post-hoc.

Use small `num_collect` values (e.g., 100) with multiple runs rather than one
large collection. This provides checkpoints for crash recovery and allows
monitoring convergence between runs.

## TODO Tracking

The file `TODO.md` in the project root tracks project progress and next steps.

At the start of a session, review TODO.md to understand current project state.

When work is completed:
- Move completed items to the "Completed" section with [x] checkmark
- Add new tasks discovered during implementation
- Update "In Progress" section if work spans multiple sessions

When planning new work:
- Check TODO.md for prioritized next steps
- Add new ideas to "Ideas / Future Work" section

Keep TODO.md current so future sessions can quickly understand project status.


## Paper Document

The paper is in `doc/main.tex`. Key sections:

1. **Introduction** - Motivation for principled hyperprior selection
2. **The Prior Selection Approach** - Core methodology
   - Law of total cumulance (iterated expectations, total variance)
   - Variance decomposition with ρ and α₀
   - Derived hyperpriors formulas
3. **Application: RDU Estimation** - Model, priors, data, estimation
4. **Results** - Posterior summaries with informative vs uninformative comparison
5. **Discussion** - Bayes factor critique, defense of prior specification
6. **Appendix A** - Full covariance derivation
7. **Appendix B** - Skew-Normal example (3-parameter extension)
   - Law of total cumulance for skewness
   - 3 degrees of freedom: ρ, α_ω, τ_α²
   - Numerical solution approach
8. **Appendix C** - MCMC implementation details

To compile (with pythontex for dynamic data):
```bash
cd /workspace/doc
pdflatex main.tex
pythontex main.tex   # Processes \py{} calls
pdflatex main.tex
biber main           # Processes citations
pdflatex main.tex
```

The paper uses pythontex to load data from NPZ files:
- `data/output/paper_data.npz` - Data summary (n_subjects, n_choices, etc.)
- `data/output/paper_quantities.npz` - Posterior summaries (means, CIs, R-hat)

In LaTeX, access via `\py{D['n_subjects']}` or `\py{fmt(Q.get('r_mean_mean', 0))}`.


## Informative vs Uninformative Priors

The project compares two prior specifications:

**Informative Prior** (`DIRECT_PRIOR` in main.py):
- Specifies a direct prior for a single individual (mu, sigma2 per param)
- Gets transformed via variance decomposition to derive hyperpriors
- Uses `rho` and `alpha_0` researcher choices

**Uninformative Prior** (`UNINFORMATIVE_HYPERPRIOR` in main.py):
- Directly specifies hyperprior parameters (mu_0, tau_0_sq, alpha_0, beta_0)
- Bypasses variance decomposition entirely
- Uses diffuse values: tau_0_sq=100, alpha_0=beta_0=0.001 (diffuse priors)

This distinction is important: the uninformative case should NOT go through
`derive_hyperpriors()` because that would impose structure on the variance.

