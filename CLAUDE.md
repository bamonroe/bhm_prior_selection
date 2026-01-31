# Workspace Instructions for Claude

Instructions for Claude Code sessions in this workspace.

## Project Goal

Working title: "A BHM of Risk Preferences with an informed prior"

The goal of this project is to write an Economics Letter discussing a
principled way to select hyper priors for a Bayesian Hierarchical Model.

The insight is that it is often much easier to charachterize the preferences of
a single economic agent than a population of agents. However, we can decompose
the variance of the prior distribution for a single agent into a hyper-prior
distribution over the population of prefernces. The predictive prior for a
single agent should be the same whether we charachterize that agent directly or
through a BHM.

I've been working with CLAUDE to write up this theory on another project, but
for this letter we will use a much simpler application. We will redo the
estimation procedure of:

Behavioral welfare economics and risk preferences: a Bayesian approach
Xiaoxue Sherry Gao, Glenn W. Harrison, Rusty Tchernis
Experimental Economics
https://doi.org/10.1007/s10683-022-09751-0

A copy of this paper is located in the lit/ directory

Thus an RDU model of risk preferences, with 4 parameters per subject. We will
largely only be concerned with the 3 parameters of the model.


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
# main.py:

# Take raw data and drop unused variables and shape it into a
# format that can be loaded by the bamcmc MCMC process
do_clean_data = True

# Generate npz files to load in the pythontex of the doc/main.tex file - this
# set of npz files describes the data
do_gen_paper_data = True

# Use the process we will describe in the paper to
# select a prior distribution
do_generate_prior = True

# use the bamcmc sampler to sample from the posterior distribution
# and save per-model output to the output directory
do_sampling = True

# Generate npz files with relevant information
# about the sampling process and results. Should have 1 npz file with everything
# referenced in the paper and nothing else, and another npz file that is used in
# subsequent steps
do_gen_paper_quantities = True

# Load paper quantities and sampling data to generate detailed diagnostic plots
do_diagnostic_plots = True

# Load paper quantities and sampling data to generate plots used in the paper
do_paper_plots = True

## actually do the things

if do_clean_data:
  # do cleaning stuff

if do_generate_prior:
  # do cleaning stuff

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

