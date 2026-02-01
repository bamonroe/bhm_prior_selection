# TODO - BHM Risk Preferences with Informed Prior

## Project Status: Test Sampling Successful

The RDU-BHM model with informed hyperpriors is working. Test sampling shows excellent
convergence (R-hat max 1.037, 0% > 1.1). Ready for production sampling.

---

## Data

### Current Dataset: `data/input/dbar.csv`

Data from **Andersen, Harrison, Lau & Rutström (2008)** "Discounting behavior: A reconsideration."

- 40,080 decision rows, 36 columns
- Joint elicitation of risk and time preferences (Denmark field experiment)
- Risk aversion task: binary lottery choices (`RA=1`)
- Discounting task: intertemporal choices (`RA=0`)

See CLAUDE.md for full variable documentation.

---

## In Progress

(none currently)

---

## Up Next

### 5. Production Sampling
- [ ] Run full sampling with many chains and iterations
- [ ] Generate convergence diagnostics
- [ ] Extract posterior summaries

### 6. Paper Writing
- [ ] Fill in `doc/main.tex` sections
- [ ] Generate paper figures in `plots/paper/`
- [ ] Write results comparing our approach to original Gao et al.

---

## Completed

### 1. Data Preparation
- [x] Obtain raw experimental data
- [x] Place raw data in `data/input/dbar.csv`
- [x] Document data format and variables (see CLAUDE.md)

### 2. Code Infrastructure
- [x] Create `code/main.py` hub with do-flags structure
- [x] Implement `do_clean_data` - shape raw data for bamcmc
- [x] Create `code/posteriors/` directory for model definitions
- [x] Set up pylib Python environment with JAX/numpy/bamcmc
- [x] Implement indexed data optimization (13 unique outcomes, 10 unique cumprobs)

### 3. Model Definition
- [x] Define RDU model posterior for bamcmc registry (`code/posteriors/rdu_bhm.py`)
  - 3 parameters per subject: r (CRRA), gamma (prob weighting), mu (noise)
  - 6 hyperparameters (mean + logsd for each param)
  - Uses indexed lookups for efficient likelihood computation
- [x] Create common functions (`code/posteriors/common_fns.py`)
  - CRRA utility function, power/Prelec probability weighting
  - risk_likelihood with JAX-compatible dynamic_slice
- [x] Test log posterior returns finite values

### 4. Prior Selection (Core Contribution)
- [x] Implement prior selection methodology (`code/prior_selection.py`)
  - `rdu_power_prior_selection()`: Define reasonable individual-level priors
  - `derive_hyperpriors()`: Variance decomposition with rho=0.3, alpha_0=6
  - `verify_hyperpriors()`: Confirm unconditional variance matches prior selection
- [x] Update `rdu_bhm.py` with derived hyperpriors:
  - r: mu_0=0.5, E[sigma]=0.42
  - gamma_raw: mu_0=0.0, E[sigma]=0.25
  - mu_raw: mu_0=-2.0, E[sigma]=0.84
- [x] Document the prior selection process (see prior_selection.py docstrings)

### 5. Test Sampling
- [x] Create runner script (`runners/run_rdu_bhm.py`)
- [x] Test sampling with bamcmc
  - 32 chains, 500 burn-in, 200 collection (700 total iterations)
  - **R-hat max: 1.037** (excellent convergence)
  - **R-hat median: 1.015**
  - **0% of parameters > 1.1**
  - MH acceptance rates: 26.5% (ideal)
  - Time per iteration: ~0.05s on CPU

---

## Technical Details

### Prior Selection Values (rho=0.3, alpha_0=6)

| Parameter  | mu_0  | tau_0  | E[sigma] | Prior Selection Var |
|------------|-------|--------|----------|---------------------|
| r          | 0.50  | 0.274  | 0.42     | 0.25                |
| gamma_raw  | 0.00  | 0.164  | 0.25     | 0.09                |
| mu_raw     | -2.00 | 0.548  | 0.84     | 1.00                |

Variance decomposition: `V = tau_0^2 + E[sigma^2]` where:
- tau_0^2 = rho * V (uncertainty about population mean)
- E[sigma^2] = (1-rho) * V (population heterogeneity)

---

## Ideas / Future Work

- Sensitivity analysis on hyperprior choices (vary rho, alpha_0)
- Comparison with flat/uninformative priors
- Extension to other utility models (EUT, CPT, Prelec weighting)
- Simulation study validating the methodology
- Compare posterior estimates to Gao et al. (2022) results

---

## References

1. **Gao, X.S., Harrison, G.W., & Tchernis, R. (2022).** "Behavioral welfare economics and risk preferences: a Bayesian approach." *Experimental Economics*, 26, 273-303.
   https://doi.org/10.1007/s10683-022-09751-0

2. **Andersen, S., Harrison, G.W., Lau, M.I., & Rutström, E.E. (2008).** "Discounting behavior: A reconsideration." *European Economic Review*, 52(8), 1342-1378.

3. CEAR Working Papers: https://cear.gsu.edu/
