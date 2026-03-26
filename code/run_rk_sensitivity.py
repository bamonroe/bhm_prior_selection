"""
Run (rho, kappa) sensitivity analysis.

Estimates the CE-space Beta-Beta model at 3 additional (rho, kappa)
configurations beyond the baseline (0.2, 0.25), then generates the
paper comparison plots for each. The baseline is not re-run.

Configurations:
  (0.2, 0.25)  — baseline (already run, not repeated here)
  (0.2, 1.50)  — same heterogeneity, much less confidence
  (0.8, 0.25)  — most variance from mean uncertainty, confident
  (0.8, 1.50)  — most variance from mean uncertainty, uncertain

Each configuration gets its own output directory and plot directory.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Project paths
CODE_DIR = Path(__file__).parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
PLOTS_DIR = PROJECT_DIR / 'plots'

sys.path.insert(0, str(CODE_DIR))

from main import (DIRECT_PRIOR_CE, REFERENCE_LOTTERIES, REFERENCE_PAIR,
                  UNINF_HYPERPRIOR_SPECS, DIAGNOSTIC_PLOT_CONFIG, SEEDS,
                  sampling_opts_ce)
from prior_selection import (ce_space_prior_selection, derive_hyperpriors_beta,
                              verify_hyperpriors_beta, print_hyperprior_summary_beta)
from clean_data import load_clean_data
import pipelines

# The 3 new configurations (baseline 0.2/0.25 is already done)
SENSITIVITY_CONFIGS = [
    (0.2, 1.50),
    (0.8, 0.25),
    (0.8, 1.50),
]


def config_label(rho, kappa):
    return f"rho{rho:.1f}_kappa{kappa:.2f}".replace('.', 'p')


def run_one_config(rho, kappa):
    """Run full pipeline for a single (rho, kappa) configuration."""
    label = config_label(rho, kappa)
    model_name = f'rdu_bhm_ce_{label}'

    print("\n" + "=" * 70)
    print(f"  Configuration: rho={rho}, kappa={kappa} ({label})")
    print("=" * 70)

    # --- Step 1: Generate prior params ---
    print("\n  Step 1: Prior selection...")
    ps = ce_space_prior_selection(DIRECT_PRIOR_CE)
    quantity_keys = ['ce1', 'ce2', 'ce3', 'pchoice']
    bounds_lo = np.array([DIRECT_PRIOR_CE[k]['lo'] for k in quantity_keys])
    bounds_hi = np.array([DIRECT_PRIOR_CE[k]['hi'] for k in quantity_keys])

    hp = derive_hyperpriors_beta(ps, bounds_lo, bounds_hi, rho=rho, kappa=kappa)
    print_hyperprior_summary_beta(hp)
    verify_hyperpriors_beta(hp)

    # Save prior params
    prior_params_path = DATA_DIR / 'output' / f'prior_params_ce_{label}.npz'
    save_dict = {'distribution': 'beta', 'rho': rho, 'kappa': kappa}
    for k, qname in enumerate(quantity_keys):
        save_dict[f'{qname}_a_mu'] = hp.a_mu[k]
        save_dict[f'{qname}_b_mu'] = hp.b_mu[k]
        save_dict[f'{qname}_a_psi'] = hp.a_psi[k]
        save_dict[f'{qname}_b_psi'] = hp.b_psi[k]
        save_dict[f'{qname}_bound_lo'] = bounds_lo[k]
        save_dict[f'{qname}_bound_hi'] = bounds_hi[k]
    np.savez(prior_params_path, **save_dict)
    print(f"  Saved: {prior_params_path}")

    # --- Step 2: Sampling ---
    # bamcmc hardcodes MODEL_ID='rdu_bhm_ce' as the subdirectory name,
    # so we set output_dir to a config-specific parent to avoid overwriting.
    print("\n  Step 2: MCMC sampling...")
    config_output_dir = DATA_DIR / 'output' / label
    opts = {**sampling_opts_ce, 'output_dir': config_output_dir}
    pipelines.run_sampling_ce(
        opts,
        ref_lotteries=REFERENCE_LOTTERIES,
        ref_pair=REFERENCE_PAIR,
        prior_params_path=prior_params_path,
    )

    # --- Step 3: Preprocessing ---
    print("\n  Step 3: Preprocessing...")
    model_dir = config_output_dir / 'rdu_bhm_ce'
    history_dir = model_dir / 'history' / 'full'
    preprocess_dir = model_dir / 'preprocessed'

    pipelines.run_preprocess_hyperparams_ce(
        history_dir=str(history_dir),
        output_dir=str(preprocess_dir),
        force=True,
    )
    pipelines.run_preprocess_subjects_ce(
        history_dir=str(history_dir),
        output_dir=str(preprocess_dir / 'subjects'),
        force=True,
    )

    # --- Step 4: Paper plots ---
    print("\n  Step 4: Paper plots...")
    paper_plots_dir = PLOTS_DIR / 'paper' / label
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    info_subjects = preprocess_dir / 'subjects'
    uninf_subjects = (DATA_DIR / 'output' / 'rdu_bhm_uninf'
                      / 'preprocessed' / 'subjects')
    hyper_info = str(preprocess_dir / 'hyperparams_preprocessed.npz')
    hyper_uninf = str(DATA_DIR / 'output' / 'rdu_bhm_uninf'
                      / 'preprocessed' / 'hyperparams_preprocessed.npz')

    from paper_plots import generate_paper_plots
    generate_paper_plots(
        info_subjects_dir=str(info_subjects),
        uninf_subjects_dir=str(uninf_subjects),
        info_hyper_path=hyper_info,
        uninf_hyper_path=hyper_uninf,
        prior_params_path=str(prior_params_path),
        output_dir=str(paper_plots_dir),
        data_output_dir=str(paper_plots_dir),
        seed=SEEDS['sampling'],
    )

    # --- Step 5: Extract subject classification ---
    print("\n  Step 5: Extracting subject classification...")
    comp = np.load(paper_plots_dir / 'posterior_ce_comparison.npz')
    median_diffs = comp['median_diffs']

    # Same thresholds as paper_plots.py _plot_cdf_difference_all
    # well-identified < $2, moderate $2-$10, prior-dominated >= $10
    n_well = int(np.sum(median_diffs < 2))
    n_moderate = int(np.sum((median_diffs >= 2) & (median_diffs < 10)))
    n_dominated = int(np.sum(median_diffs >= 10))

    print(f"  Results: {n_well} well-identified, {n_moderate} moderate, "
          f"{n_dominated} prior-dominated")

    return {
        'rho': rho, 'kappa': kappa, 'label': label,
        'n_well': n_well, 'n_moderate': n_moderate,
        'n_dominated': n_dominated,
    }


def main():
    t0 = time.time()
    results = []

    for rho, kappa in SENSITIVITY_CONFIGS:
        result = run_one_config(rho, kappa)
        results.append(result)

    # Print summary table
    print("\n" + "=" * 70)
    print("  (rho, kappa) Sensitivity Summary")
    print("=" * 70)
    print(f"  {'rho':>5} {'kappa':>6} | {'Well-ID':>8} {'Moderate':>9} {'Prior-Dom':>10}")
    print("  " + "-" * 48)

    # Include baseline from existing results
    baseline_comp = DATA_DIR / 'output' / 'posterior_ce_comparison.npz'
    if baseline_comp.exists():
        bc = np.load(baseline_comp)
        md = bc['median_diffs']
        n_w = int(np.sum(md < 2))
        n_m = int(np.sum((md >= 2) & (md < 10)))
        n_d = int(np.sum(md >= 10))
        print(f"  {0.2:5.1f} {0.25:6.2f} | {n_w:8d} {n_m:9d} {n_d:10d}  (baseline)")

    for r in results:
        print(f"  {r['rho']:5.1f} {r['kappa']:6.2f} | "
              f"{r['n_well']:8d} {r['n_moderate']:9d} {r['n_dominated']:10d}")

    # Save summary
    summary_path = DATA_DIR / 'output' / 'rk_sensitivity_summary.npz'
    np.savez(summary_path,
             rho=np.array([r['rho'] for r in results]),
             kappa=np.array([r['kappa'] for r in results]),
             n_well=np.array([r['n_well'] for r in results]),
             n_moderate=np.array([r['n_moderate'] for r in results]),
             n_dominated=np.array([r['n_dominated'] for r in results]))
    print(f"\n  Summary saved to: {summary_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
