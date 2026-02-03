"""
Main entry point for the BHM Risk Preferences project.

This file serves as a hub for all project steps. Each step is controlled
by a do_* flag. Steps assume previous steps have been completed.

Usage:
    python code/main.py
    python code/main.py --all
    python code/main.py --sampling --num-runs 10
"""

import sys
import argparse
from pathlib import Path

# Add code directory to path
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Paths
    'output_dir': PROJECT_ROOT / 'data' / 'output' / 'rdu_bhm',
    'plots_dir': PROJECT_ROOT / 'plots' / 'rdu_bhm',
    'paper_plots_dir': PROJECT_ROOT / 'plots' / 'paper',

    # Prior selection
    'rho': 0.3,
    'alpha_0': 6.0,

    # Sampling
    'num_chains_a': 100,
    'num_chains_b': 100,
    'num_collect': 1000,
    'thin_iteration': 10,
    'num_runs': 5,
    'rng_seed': 42,
}


# =============================================================================
# Control Flags (defaults)
# =============================================================================

do_clean_data = True
do_gen_paper_data = False
do_generate_prior = False
do_sampling = False
do_gen_paper_quantities = False
do_diagnostic_plots = False
do_paper_plots = False


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

    # Sampling options (override CONFIG)
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
        any_flag_set = any([
            args.clean_data, args.gen_paper_data, args.generate_prior,
            args.sampling, args.gen_paper_quantities,
            args.diagnostic_plots, args.paper_plots
        ])
        if any_flag_set:
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
        from clean_data import clean_and_save_data
        clean_and_save_data()
        print("Done.\n")

    if do_gen_paper_data:
        print("=" * 60)
        print("Step 2: Generating paper data summaries")
        print("=" * 60)
        from gen_paper_data import gen_paper_data
        gen_paper_data()
        print("Done.\n")

    if do_generate_prior:
        print("=" * 60)
        print("Step 3: Generating informed prior")
        print("=" * 60)
        from prior_selection import run_prior_selection
        run_prior_selection(rho=CONFIG['rho'], alpha_0=CONFIG['alpha_0'])
        print("Done.\n")

    if do_sampling:
        print("=" * 60)
        print("Step 4: Running MCMC sampling")
        print("=" * 60)
        from sampling import run_sampling
        run_sampling(
            output_dir=CONFIG['output_dir'],
            num_runs=args.num_runs or CONFIG['num_runs'],
            num_collect=args.num_collect or CONFIG['num_collect'],
            config=CONFIG,
        )
        print("Done.\n")

    if do_gen_paper_quantities:
        print("=" * 60)
        print("Step 5: Generating paper quantities")
        print("=" * 60)
        from gen_paper_quantities import gen_paper_quantities
        history_dir = CONFIG['output_dir'] / 'history' / 'full'
        gen_paper_quantities(history_dir)
        print("Done.\n")

    if do_diagnostic_plots:
        print("=" * 60)
        print("Step 6: Generating diagnostic plots")
        print("=" * 60)
        from plotting import diagnostic_plots
        history_dir = CONFIG['output_dir'] / 'history' / 'full'
        diagnostic_plots(history_dir, CONFIG['plots_dir'])
        print("Done.\n")

    if do_paper_plots:
        print("=" * 60)
        print("Step 7: Generating paper plots")
        print("=" * 60)
        from plotting import paper_plots
        quantities_path = PROJECT_ROOT / 'data' / 'output' / 'paper_quantities.npz'
        paper_plots(quantities_path, CONFIG['paper_plots_dir'])
        print("Done.\n")

    print("All requested steps completed.")
