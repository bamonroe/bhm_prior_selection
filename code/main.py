"""
Main entry point for the BHM Risk Preferences project.

This file serves as a hub for all project steps. Each step is controlled
by a do_* flag. Steps assume previous steps have been completed.

Usage:
    python code/main.py
"""

import sys
from pathlib import Path

# Add code directory to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

# =============================================================================
# Control Flags
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
# Execute Steps
# =============================================================================

if __name__ == "__main__":

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
        # TODO: Implement
        raise NotImplementedError("do_gen_paper_data not yet implemented")

    if do_generate_prior:
        print("=" * 60)
        print("Step 3: Generating informed prior")
        print("=" * 60)
        # TODO: Implement
        raise NotImplementedError("do_generate_prior not yet implemented")

    if do_sampling:
        print("=" * 60)
        print("Step 4: Running MCMC sampling")
        print("=" * 60)
        # TODO: Implement
        raise NotImplementedError("do_sampling not yet implemented")

    if do_gen_paper_quantities:
        print("=" * 60)
        print("Step 5: Generating paper quantities")
        print("=" * 60)
        # TODO: Implement
        raise NotImplementedError("do_gen_paper_quantities not yet implemented")

    if do_diagnostic_plots:
        print("=" * 60)
        print("Step 6: Generating diagnostic plots")
        print("=" * 60)
        # TODO: Implement
        raise NotImplementedError("do_diagnostic_plots not yet implemented")

    if do_paper_plots:
        print("=" * 60)
        print("Step 7: Generating paper plots")
        print("=" * 60)
        # TODO: Implement
        raise NotImplementedError("do_paper_plots not yet implemented")

    print("All requested steps completed.")
