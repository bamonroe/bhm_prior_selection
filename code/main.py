"""
Main entry point for the BHM Risk Preferences project.

This file serves as a hub for all project steps. Each step is controlled
by a do_* flag. Steps assume previous steps have been completed.

Usage:
    python code/main.py

    # Or run specific steps:
    python code/main.py --sampling --num-runs 5
"""

import sys
import argparse
from pathlib import Path

# Add code directory to path
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# =============================================================================
# Control Flags (defaults)
# =============================================================================

# Take raw data and drop unused variables and shape it into a
# format that can be loaded by the bamcmc MCMC process
do_clean_data = True

# Generate npz files to load in the pythontex of the doc/main.tex file - this
# set of npz files describes the data
do_gen_paper_data = False

# Use the process we will describe in the paper to
# select a prior distribution
do_generate_prior = False

# Use the bamcmc sampler to sample from the posterior distribution
# and save per-model output to the output directory
do_sampling = False

# Generate npz files with relevant information
# about the sampling process and results
do_gen_paper_quantities = False

# Load paper quantities and sampling data to generate detailed diagnostic plots
do_diagnostic_plots = False

# Load paper quantities and sampling data to generate plots used in the paper
do_paper_plots = False

# =============================================================================
# Configuration
# =============================================================================

# Sampling configuration
SAMPLING_CONFIG = {
    'output_dir': PROJECT_ROOT / 'data' / 'output' / 'rdu_bhm',
    'num_chains_a': 100,
    'num_chains_b': 100,
    'num_collect': 1000,
    'thin_iteration': 10,
    'num_runs': 5,
    'rng_seed': 42,
}


# =============================================================================
# Step Implementations
# =============================================================================

def step_clean_data():
    """Step 1: Clean and prepare data for MCMC."""
    from clean_data import clean_and_save_data
    clean_and_save_data()


def step_gen_paper_data():
    """Step 2: Generate summary statistics for the paper."""
    import numpy as np
    from clean_data import load_raw_data, parse_ra_data

    print("Loading data...")
    ra_rows = load_raw_data()
    data = parse_ra_data(ra_rows)

    # Summary statistics
    n_subjects = len(data['subject_ids'])
    n_choices_per_subject = len(data['choices'][0])
    total_choices = n_subjects * n_choices_per_subject

    # Prize statistics
    all_prizes = []
    for prizes in [data['prize_a_high'], data['prize_a_low'],
                   data['prize_b_high'], data['prize_b_low']]:
        all_prizes.extend([p for subj in prizes for p in subj])
    all_prizes = np.array(all_prizes)

    # Probability statistics
    all_probs = np.array([p for subj in data['prob_high'] for p in subj])

    print(f"\nData Summary:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices_per_subject}")
    print(f"  Total choices: {total_choices}")
    print(f"  Prize range: [{all_prizes.min():.0f}, {all_prizes.max():.0f}] DKK")
    print(f"  Probability range: [{all_probs.min():.2f}, {all_probs.max():.2f}]")

    # Save for paper
    output_path = PROJECT_ROOT / 'data' / 'output' / 'paper_data.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        n_subjects=n_subjects,
        n_choices_per_subject=n_choices_per_subject,
        total_choices=total_choices,
        prize_min=all_prizes.min(),
        prize_max=all_prizes.max(),
        prob_min=all_probs.min(),
        prob_max=all_probs.max(),
    )
    print(f"\nSaved to: {output_path}")


def step_generate_prior():
    """Step 3: Run prior selection and derive hyperpriors."""
    from prior_selection import (
        rdu_power_prior_selection,
        derive_hyperpriors,
        print_hyperprior_summary,
        verify_hyperpriors,
        get_hyperpriors_for_rdu_bhm,
    )
    import numpy as np

    print("Running prior selection for RDU model...")
    ps = rdu_power_prior_selection()
    print(ps.description)

    # Derive hyperpriors with default rho=0.3, alpha_0=6.0
    hp = derive_hyperpriors(ps, rho=0.3, alpha_0=6.0)
    print_hyperprior_summary(hp)

    # Verify constraints
    print()
    verify_hyperpriors(hp)

    # Get formatted hyperpriors
    params = get_hyperpriors_for_rdu_bhm(rho=0.3, alpha_0=6.0)

    # Save for paper
    output_path = PROJECT_ROOT / 'data' / 'output' / 'prior_params.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **params)
    print(f"\nSaved hyperpriors to: {output_path}")


def step_sampling(num_runs=None, num_collect=None):
    """Step 4: Run MCMC sampling with bamcmc."""
    from bamcmc import rmcmc
    from clean_data import load_raw_data, parse_ra_data, make_bamcmc_data_indexed
    from posteriors.rdu_bhm import MODEL_ID, NPARAMS, N_HYPER

    # Load data
    print("Loading data...")
    ra_rows = load_raw_data()
    data = parse_ra_data(ra_rows)
    indexed_data = make_bamcmc_data_indexed(data)

    n_subjects = indexed_data['static'][0]
    print(f"  Subjects: {n_subjects}")
    print(f"  Total parameters: {n_subjects * NPARAMS + N_HYPER}")

    # MCMC configuration
    output_dir = SAMPLING_CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    mcmc_config = {
        'posterior_id': MODEL_ID,
        'num_chains_a': SAMPLING_CONFIG['num_chains_a'],
        'num_chains_b': SAMPLING_CONFIG['num_chains_b'],
        'burn_iter': 0,  # Keep full history, drop burn-in post-hoc
        'num_collect': num_collect or SAMPLING_CONFIG['num_collect'],
        'thin_iteration': SAMPLING_CONFIG['thin_iteration'],
        'rng_seed': SAMPLING_CONFIG['rng_seed'],
    }

    actual_runs = num_runs or SAMPLING_CONFIG['num_runs']

    print(f"\nSampling Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Chains: {mcmc_config['num_chains_a']} + {mcmc_config['num_chains_b']}")
    print(f"  Collection iterations: {mcmc_config['num_collect']}")
    print(f"  Thinning: {mcmc_config['thin_iteration']}")
    print(f"  Number of runs: {actual_runs}")
    print(f"  Output: {output_dir}")

    # Run schedule: first run resets, subsequent runs resume
    run_schedule = [("reset", 1), ("resume", actual_runs - 1)] if actual_runs > 1 else [("reset", 1)]

    print(f"\nStarting MCMC with schedule: {run_schedule}")
    summary = rmcmc(
        mcmc_config,
        indexed_data,
        output_dir=str(output_dir),
        run_schedule=run_schedule,
        calculate_rhat=True,
        burn_in_fresh=True,
    )

    print(f"\nSampling complete!")
    print(f"  Total runs: {summary['total_runs_completed']}")
    print(f"  Final iteration: {summary['final_iteration']}")
    print(f"  History files: {len(summary['history_files'])}")

    return summary


def step_gen_paper_quantities():
    """Step 5: Extract posterior summaries for the paper."""
    import numpy as np
    from bamcmc import combine_batch_histories, apply_burnin, compute_rhat_from_history
    from posteriors.rdu_bhm import NPARAMS, N_HYPER

    output_dir = SAMPLING_CONFIG['output_dir']
    history_dir = output_dir / 'history' / 'full'

    # Find all history files
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"Found {len(history_files)} history files")

    # Combine histories
    print("Combining histories...")
    history_paths = [str(f) for f in history_files]
    combined = combine_batch_histories(history_paths)

    history = combined['history']
    iterations = combined['iterations']

    print(f"Combined history shape: {history.shape}")
    print(f"Iteration range: [{iterations.min()}, {iterations.max()}]")

    # Apply burn-in (drop first half)
    burn_in_iter = iterations.max() // 2
    print(f"Applying burn-in at iteration {burn_in_iter}...")
    history_post, iter_post, _ = apply_burnin(history, iterations, min_iteration=burn_in_iter)

    print(f"Post burn-in shape: {history_post.shape}")

    # Compute R-hat
    K = 20  # Superchains
    M = history_post.shape[1] // K
    print(f"Computing R-hat with K={K}, M={M}...")
    rhat = compute_rhat_from_history(history_post, K, M)

    # Get number of subjects from parameter count
    n_params = history_post.shape[2]
    # history has params + generated quantities
    # params = n_subjects * NPARAMS + N_HYPER
    # gq = n_subjects * 3 + 6
    # total = params + gq = n_subjects * (NPARAMS + 3) + N_HYPER + 6
    n_subjects = (n_params - N_HYPER - 6) // (NPARAMS + 3)
    first_hyper = n_subjects * NPARAMS

    # Extract posterior means
    flat_samples = history_post.reshape(-1, n_params)
    posterior_means = flat_samples.mean(axis=0)
    posterior_stds = flat_samples.std(axis=0)

    # Hyperparameter posteriors (raw state values)
    hyper_means = posterior_means[first_hyper:first_hyper + N_HYPER]
    hyper_stds = posterior_stds[first_hyper:first_hyper + N_HYPER]
    hyper_rhat = rhat[first_hyper:first_hyper + N_HYPER]

    print(f"\nHyperparameter Posterior Summary:")
    print(f"  r_mean:      {hyper_means[0]:.4f} ± {hyper_stds[0]:.4f} (R-hat: {hyper_rhat[0]:.3f})")
    print(f"  r_logsd:     {hyper_means[1]:.4f} ± {hyper_stds[1]:.4f} (R-hat: {hyper_rhat[1]:.3f})")
    print(f"  gamma_mean:  {hyper_means[2]:.4f} ± {hyper_stds[2]:.4f} (R-hat: {hyper_rhat[2]:.3f})")
    print(f"  gamma_logsd: {hyper_means[3]:.4f} ± {hyper_stds[3]:.4f} (R-hat: {hyper_rhat[3]:.3f})")
    print(f"  mu_mean:     {hyper_means[4]:.4f} ± {hyper_stds[4]:.4f} (R-hat: {hyper_rhat[4]:.3f})")
    print(f"  mu_logsd:    {hyper_means[5]:.4f} ± {hyper_stds[5]:.4f} (R-hat: {hyper_rhat[5]:.3f})")

    # R-hat summary
    print(f"\nR-hat Summary:")
    print(f"  Max: {rhat.max():.4f}")
    print(f"  % > 1.1: {100 * np.mean(rhat > 1.1):.1f}%")

    # Save for paper
    output_path = PROJECT_ROOT / 'data' / 'output' / 'paper_quantities.npz'
    np.savez(
        output_path,
        hyper_means=hyper_means,
        hyper_stds=hyper_stds,
        hyper_rhat=hyper_rhat,
        rhat_max=rhat.max(),
        rhat_pct_above_1_1=np.mean(rhat > 1.1),
        n_subjects=n_subjects,
        n_samples=flat_samples.shape[0],
    )
    print(f"\nSaved to: {output_path}")


def step_diagnostic_plots():
    """Step 6: Generate diagnostic plots."""
    import numpy as np
    import matplotlib.pyplot as plt
    from bamcmc import combine_batch_histories
    from posteriors.rdu_bhm import NPARAMS, N_HYPER

    output_dir = SAMPLING_CONFIG['output_dir']
    history_dir = output_dir / 'history' / 'full'
    plots_dir = PROJECT_ROOT / 'plots' / 'rdu_bhm'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load combined history
    history_files = sorted(history_dir.glob('history_*.npz'))
    if not history_files:
        raise FileNotFoundError(f"No history files found in {history_dir}")

    print(f"Loading {len(history_files)} history files...")
    history_paths = [str(f) for f in history_files]
    combined = combine_batch_histories(history_paths)

    history = combined['history']
    iterations = combined['iterations']

    n_params = history.shape[2]
    n_subjects = (n_params - N_HYPER - 6) // (NPARAMS + 3)
    first_hyper = n_subjects * NPARAMS

    # Plot 1: Hyperparameter traces
    print("Generating hyperparameter trace plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    hyper_names = ['r_mean', 'r_logsd', 'gamma_mean', 'gamma_logsd', 'mu_mean', 'mu_logsd']

    for i, (ax, name) in enumerate(zip(axes.flat, hyper_names)):
        param_idx = first_hyper + i
        # Plot a subset of chains
        for chain_idx in range(0, min(20, history.shape[1]), 2):
            ax.plot(iterations, history[:, chain_idx, param_idx], alpha=0.5, lw=0.5)
        ax.set_title(name)
        ax.set_xlabel('Iteration')

    plt.tight_layout()
    trace_path = plots_dir / 'hyperparameter_traces.png'
    plt.savefig(trace_path, dpi=150)
    plt.close()
    print(f"  Saved: {trace_path}")

    # Plot 2: Hyperparameter posteriors
    print("Generating hyperparameter posterior plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Use second half as post burn-in
    burn_idx = len(iterations) // 2
    post_samples = history[burn_idx:].reshape(-1, n_params)

    for i, (ax, name) in enumerate(zip(axes.flat, hyper_names)):
        param_idx = first_hyper + i
        samples = post_samples[:, param_idx]
        ax.hist(samples, bins=50, density=True, alpha=0.7)
        ax.axvline(samples.mean(), color='r', linestyle='--', label=f'Mean: {samples.mean():.3f}')
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    post_path = plots_dir / 'hyperparameter_posteriors.png'
    plt.savefig(post_path, dpi=150)
    plt.close()
    print(f"  Saved: {post_path}")

    print(f"\nDiagnostic plots saved to: {plots_dir}")


def step_paper_plots():
    """Step 7: Generate publication-ready plots."""
    import numpy as np
    import matplotlib.pyplot as plt

    plots_dir = PROJECT_ROOT / 'plots' / 'paper'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load paper quantities
    quant_path = PROJECT_ROOT / 'data' / 'output' / 'paper_quantities.npz'
    if not quant_path.exists():
        raise FileNotFoundError(f"Run do_gen_paper_quantities first: {quant_path}")

    data = np.load(quant_path)

    print("Paper quantities loaded:")
    print(f"  n_subjects: {data['n_subjects']}")
    print(f"  n_samples: {data['n_samples']}")
    print(f"  R-hat max: {data['rhat_max']:.4f}")

    # TODO: Add specific paper plots as needed
    print("\nPaper plots: Not yet implemented")
    print("Add specific visualizations based on paper requirements")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BHM Risk Preferences - Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python code/main.py                    # Run with default flags
    python code/main.py --all              # Run all steps
    python code/main.py --sampling         # Run only sampling
    python code/main.py --sampling --num-runs 10
        """
    )

    # Step flags
    parser.add_argument('--clean-data', action='store_true', help='Run data cleaning')
    parser.add_argument('--gen-paper-data', action='store_true', help='Generate paper data summaries')
    parser.add_argument('--generate-prior', action='store_true', help='Run prior selection')
    parser.add_argument('--sampling', action='store_true', help='Run MCMC sampling')
    parser.add_argument('--gen-paper-quantities', action='store_true', help='Extract paper quantities')
    parser.add_argument('--diagnostic-plots', action='store_true', help='Generate diagnostic plots')
    parser.add_argument('--paper-plots', action='store_true', help='Generate paper plots')
    parser.add_argument('--all', action='store_true', help='Run all steps')

    # Sampling options
    parser.add_argument('--num-runs', type=int, help='Number of MCMC runs')
    parser.add_argument('--num-collect', type=int, help='Collection iterations per run')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override flags from command line
    if args.all:
        do_clean_data = True
        do_gen_paper_data = True
        do_generate_prior = True
        do_sampling = True
        do_gen_paper_quantities = True
        do_diagnostic_plots = True
        do_paper_plots = True
    else:
        # Check if any specific flags were set
        any_flag_set = any([
            args.clean_data, args.gen_paper_data, args.generate_prior,
            args.sampling, args.gen_paper_quantities,
            args.diagnostic_plots, args.paper_plots
        ])

        if any_flag_set:
            # Use only explicitly set flags
            do_clean_data = args.clean_data
            do_gen_paper_data = args.gen_paper_data
            do_generate_prior = args.generate_prior
            do_sampling = args.sampling
            do_gen_paper_quantities = args.gen_paper_quantities
            do_diagnostic_plots = args.diagnostic_plots
            do_paper_plots = args.paper_plots

    # Execute steps
    if do_clean_data:
        print("=" * 60)
        print("Step 1: Cleaning data")
        print("=" * 60)
        step_clean_data()
        print("Done.\n")

    if do_gen_paper_data:
        print("=" * 60)
        print("Step 2: Generating paper data summaries")
        print("=" * 60)
        step_gen_paper_data()
        print("Done.\n")

    if do_generate_prior:
        print("=" * 60)
        print("Step 3: Generating informed prior")
        print("=" * 60)
        step_generate_prior()
        print("Done.\n")

    if do_sampling:
        print("=" * 60)
        print("Step 4: Running MCMC sampling")
        print("=" * 60)
        step_sampling(num_runs=args.num_runs, num_collect=args.num_collect)
        print("Done.\n")

    if do_gen_paper_quantities:
        print("=" * 60)
        print("Step 5: Generating paper quantities")
        print("=" * 60)
        step_gen_paper_quantities()
        print("Done.\n")

    if do_diagnostic_plots:
        print("=" * 60)
        print("Step 6: Generating diagnostic plots")
        print("=" * 60)
        step_diagnostic_plots()
        print("Done.\n")

    if do_paper_plots:
        print("=" * 60)
        print("Step 7: Generating paper plots")
        print("=" * 60)
        step_paper_plots()
        print("Done.\n")

    print("All requested steps completed.")
