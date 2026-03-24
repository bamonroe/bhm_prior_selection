# Workspace Instructions for Claude

Instructions for Claude Code sessions in this workspace.

## Project Status

**Current State:** Paper restructured for Journal of Econometrics. Single model: Beta-Beta CE-space hierarchy.

The paper has been restructured from an Economics Letter into a full JoE paper:
- Title: "Quantity-Space Prior Specification for Bayesian Hierarchical Models"
- 6 sections + 4 appendices
- Two contributions: (i) quantity-space priors with Jacobian correction; (ii) variance decomposition with κ parameterization
- Single application: RDU with real data (working)
- Bibliography expanded from ~11 to ~30+ references
- Technical notes integrated into main paper body and appendices

**Active model: `rdu_bhm_ce`** (Beta-Beta, CE-space): Full pipeline implemented —
prior selection, sampling, preprocessing, and diagnostic plotting. Uses
logit-parameterized hyperparams internally; diagnostic plots display on
natural scale (dollars for CEs, [0,1] for ψ).

The previous NIG parameter-space model (`rdu_bhm`) has been removed from all
pipelines. Its posterior file (`posteriors/rdu_bhm.py`) remains on disk for
reference but is not imported or used.

**Current branch:** `master`

See `TODO.md` for detailed task tracking.

## Project Goal

Working title: "Quantity-Space Prior Specification for Bayesian Hierarchical Models: Theory and Applications to Risk Preferences"

Target: Journal of Econometrics, full-length paper (~30-35 pages with appendices).

**Two Contributions:**

1. **Quantity-space prior specification** — Specify priors on generated
   quantities (certainty equivalents, choice probabilities) and transform to
   parameter space via Jacobian correction: p(θ) = p_Y(g(θ)) |det J_g(θ)|

2. **Distribution-free variance decomposition** — Reduce 4p hyperprior choices
   to 2 interpretable parameters:
   - **ρ (rho):** Variance partition ratio — fraction of variance from
     uncertainty about population mean vs. population heterogeneity
   - **κ (kappa):** Coefficient of variation of heterogeneity — replaces α₀
     with a distribution-free measure: κ = SD[σ²]/E[σ²]

**Application:**
- RDU estimation with Harrison & Ng (2016) real data (working)
- CPT and discounting simulation studies cut from current scope (future work)

The paper bridges two prior specification literatures: "robust/uninformative"
(Gelman 2006, Simpson 2017) and "prior predictive calibration" (Kadane 1980,
Gabry 2019), providing analytical machinery the latter has lacked.

We replicate/extend:
> Gao, Harrison & Tchernis (2022). "Behavioral welfare economics and risk
> preferences: a Bayesian approach." *Experimental Economics*.
> https://doi.org/10.1007/s10683-022-09751-0

A copy is in the `lit/` directory.


## Key Code Files

**For detailed code architecture, see `code/README.md`.**

### Core Implementation

- **`code/prior_selection.py`** - The paper's core contribution
  - `build_ce_direct_prior()`: Build direct prior from elicitation inputs
  - `ce_space_prior_selection()`: Prior selection for 3 CEs + 1 choice probability
  - `derive_hyperpriors_beta()`: Beta-Beta variance decomposition with (ρ, κ)
  - `verify_hyperpriors_beta()`: Monte Carlo verification of variance targets
  - `run_prior_selection_ce()`: Full pipeline: select → derive → verify → save

- **`code/posteriors/rdu_bhm_ce.py`** - Beta-Beta CE-space BHM log posterior
  - 4 params per subject: r (CRRA), φ (Prelec curvature), η (Prelec elevation), λ (noise)
  - 8 hyperparameters: logit_μ_k and logit_ψ_k for 4 quantities (CE1, CE2, CE3, P_choice)
  - NPARAMS=4, N_HYPER=8, N_QUANTITIES=4, HYPERPRIOR_STRIDE=6
  - Chain state stores logit-scale; generated quantities return sigmoid-transformed values
  - Hyperprior array layout: [a_mu, b_mu, a_psi, b_psi, lo, hi] per quantity

- **`code/posteriors/common_fns.py`** - Shared utility functions
  - CRRA utility with normalization, Prelec probability weighting, risk likelihood

- **`code/sampling_ce.py`** - CE-space model sampling setup
  - Configures bamcmc for `rdu_bhm_ce` model
  - Passes reference lottery data and hyperprior params to posterior

- **`code/sensitivity.py`** - Sensitivity analysis over (ρ, κ) grid
  - Beta-Beta hierarchy sampling for between/within population SD
  - Three-panel LaTeX table generation

- **`code/clean_data.py`** - Data preparation
  - Filters to RA task, creates indexed data structure for GPU efficiency
  - Flips choice encoding: raw 1=right/B → model 1=A

- **`code/gen_paper_data.py`** - Generate `paper_data.npz` for pythontex
  - Data summaries (n_subjects, n_choices, prize/probability ranges)
  - CE-space config: direct prior bounds, reference lotteries, reference pair, prior selection
  - NPZ keys: `ce_mu_1`, `ce_lo_1`, `ref_lot_1_prize_1`, `ref_pair_A_prob_2`, `ce_ps_rho`, etc.

- **`code/plotting.py`** - Diagnostic plotting for CE-space model
  - CE hyperparameter plots apply sigmoid transform to display natural scale
  - Two-stage preprocessing + parallel subject plot generation

### Model Parameters

The RDU-BHM model uses 4 parameters per subject:
- **r**: CRRA risk aversion coefficient
- **φ (phi)**: Prelec curvature (log scale internally)
- **η (eta)**: Prelec elevation (log scale internally)
- **λ (lambda)**: Fechner noise (log scale internally)

Prelec probability weighting: `w(p) = exp(-η(-ln p)^φ)`
- When φ = η = 1: linear PWF (Expected Utility)
- φ < 1: inverse-S shape (overweight small p, underweight large p)

### Variance Decomposition (Beta-Beta)

For each quantity k on [lo, hi], normalized to z ∈ [0,1]:
- μ_k ~ Beta(a_μ, b_μ) — population mean
- ψ_k ~ Beta(a_ψ, b_ψ) — variance fraction
- z_ik | μ_k, ψ_k ~ Beta(μ(1-ψ)/ψ, (1-μ)(1-ψ)/ψ)

Researcher choices:
- **ρ (rho):** Fraction of variance from uncertainty about population mean (default: 0.2)
- **κ (kappa):** Coefficient of variation of heterogeneity (default: 0.25)
  - Distribution-free: defined without reference to any specific family
  - Interpretable as "confidence in heterogeneity specification"
  - κ = 0: certain about heterogeneity level; κ = 1: high uncertainty

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
data/output/            # paper_data.npz, prior_params_ce.npz
data/output/$model/     # MCMC output (checkpoints/, history/full/, preprocessed/)
code/                   # Location of the python code used to estimate the data
code/posteriors/        # Model-specific log posteriors (rdu_bhm_ce.py)
doc/                    # Paper LaTeX (main.tex + appendix_*.tex)
plots/{$model, paper}/  # generated plots used in the paper or for diagnostics
technical_notes/$note/  # LaTeX technical documentation
lit/                    # Published academic papers that are relevant
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
do_clean_data     = False
do_gen_paper_data = False

# --- Prior Selection & Analysis ---
do_sensitivity          = False

# --- MCMC Sampling ---
do_sampling_ce             = False

# --- Preprocessing (extract from history files) ---
do_generate_prior_ce       = False
do_preprocess_hyper_ce     = False
do_preprocess_subjects_ce  = False

# --- Diagnostic Plots ---
do_hyper_plots_ce          = False
do_subject_plots_ce        = False

# --- Appendix Examples ---
do_injectivity_analysis    = False

# --- Document Compilation ---
do_compile_doc          = False
```

## Technical Notes

Technical notes are detailed notes written in their own subdirectories, written
in latex.

When writing a note, there should be extensive detail about the topic. I have a
high degree of technical proficiency in math, statistics, economics, code, and
finance. I expect very detailed summaries of any issues where a technical note
is asked for, not just a high level overview. These should be written in the
style of a detailed report.

### Next Steps Notes

When asked to write a "next steps" note (or similar assessment of paper/project
status), create a new timestamped file in `technical_notes/next_steps/` using
the format `DD_MM_YYYY.tex` (e.g., `28_02_2026.tex`). Each note is a snapshot
in time — never overwrite previous notes. Previous notes in the directory
provide context on what has already been assessed and addressed.


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


## Writing Tone for the Paper

The paper builds on the work of others and the tone must reflect that. The
cited authors (Bland, Manderson, Segert, Gelman, Simpson, etc.) are the most
likely reviewers. Do not characterize their methods negatively — avoid words
like "informal," "black-box," "ad hoc," "naive," or "limited." Instead,
describe what each contribution does and then state what the present paper adds
as a complement or extension, not a correction. The framing is: "X did Y; I
extend this by adding Z" — never "X is insufficient; I fix it."

When contrasting approaches, use neutral language: "requires numerical
optimization" rather than "requires black-box numerical optimization";
"relies on visual calibration" rather than "relies on informal visual
inspection." The goal is to position the contribution clearly without
diminishing prior work.

Do not use bullet-point lists (`\begin{itemize}`) anywhere in the paper.
Use prose to present information, even when listing properties or conditions.

Do not use `\textbf{}` for emphasis in the paper body. Use `\textit{}` or
`\emph{}` sparingly, or let the structure speak for itself.

Do not use em-dashes (`---`) in the paper. Use colons, semicolons, or
parenthetical commas instead.

Do not place multiple equations side-by-side in a single `align` row. Each
equation should be on its own line.


## Paper Document

The paper is in `doc/main.tex`. Restructured for Journal of Econometrics:

### Main Sections
1. **Introduction** - Two contributions, bridge between two literatures
2. **Literature: Two Camps of Prior Specification** - Camp 1 (uninformative), Camp 2 (knowledge-based), BHMs in economics
3. **General Hyperprior Specification from Predictive Quantities** - Framework, variance decomposition, NIG example with heatmaps, κ parameterization
4. **Application: RDU with Real Data** - Model, CE mapping, Beta-Beta hierarchy, comparison prior (Gao 2022)
5. **Estimation** - Data, MCMC, CE overlay (3 panels), subject comparisons
6. **Discussion** - Practical guidance, limitations, Bayes factor comparison
7. **Conclusion**

### Appendices (1 total, loaded via `\input{}`)
- **A**: Global Injectivity — `appendix_global_injectivity.tex`

### Compiling

The paper uses pythontex to load data from NPZ files. Compilation is handled
by `do_compile_doc` in `main.py`, which runs the full build sequence using the
pylib Python environment:

```
pdflatex -> pythontex (with --interpreter=python:pylib/bin/python) -> pdflatex -> biber -> pdflatex
```

Manual compilation:
```bash
cd /workspace/doc
pdflatex -interaction=nonstopmode main.tex
pythontex --interpreter=python:/workspace/pylib/bin/python main.tex
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
```

The paper uses pythontex to load data from `data/output/paper_data.npz`.

In LaTeX, access via `\py{D['n_subjects']}`.

The `paper_data.npz` is the single source of truth for all numeric values in the paper.
Table 1 (lottery design) and CE-space prior descriptions are pythontex-driven, loading
reference lottery probabilities, CE bounds, and prior selection choices (ρ, κ) from this file.


## Prior Specification

**CE-Space Prior** (`DIRECT_PRIOR_CE` + `rdu_bhm_ce`):
- Specifies beliefs about certainty equivalents and choice probabilities
- Beta-Beta hierarchy: (mu, sigma2, lo, hi) per quantity on bounded domain
- 3 reference lotteries (`REFERENCE_LOTTERIES`) define CE quantities
- 1 reference pair (`REFERENCE_PAIR`) identifies noise parameter λ
- Uses `rho` and `kappa` for variance decomposition (`PRIOR_SELECTION_CE`)
- Hyperparams are logit-scale internally; diagnostic plots show natural scale

