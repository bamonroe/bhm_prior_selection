"""
Generate summary statistics for the paper.

This module creates NPZ files with data summaries that can be loaded
in the paper's LaTeX (via pythontex or similar).
"""

import numpy as np
from pathlib import Path

from clean_data import load_clean_data


def gen_paper_data(output_path=None, direct_prior=None, prior_selection=None):
    """
    Generate summary statistics about the experimental data.

    Saves to NPZ file that can be loaded in LaTeX via pythontex.

    Args:
        output_path: Path to save NPZ file (default: data/output/paper_data.npz)
        direct_prior: Dict with direct prior specification for each parameter
        prior_selection: Dict with researcher choices (rho, kappa)
    """
    print("Loading data...")
    clean = load_clean_data()

    # Summary statistics
    n_subjects = int(clean['n_subjects'])
    n_choices_per_subject = int(clean['n_choices'])
    total_choices = n_subjects * n_choices_per_subject

    # Prize statistics (from unified 3-outcome format)
    prizes_L = np.asarray(clean['prizes_L'])
    prizes_R = np.asarray(clean['prizes_R'])
    all_prizes = np.concatenate([prizes_L.ravel(), prizes_R.ravel()])

    # Probability statistics
    probs_L = np.asarray(clean['probs_L'])
    probs_R = np.asarray(clean['probs_R'])
    # Non-zero probabilities for summary
    all_probs = np.concatenate([probs_L.ravel(), probs_R.ravel()])
    nonzero_probs = all_probs[all_probs > 0]

    print(f"\nData Summary:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices_per_subject}")
    print(f"  Total choices: {total_choices}")
    print(f"  Prize range: [{all_prizes.min():.0f}, {all_prizes.max():.0f}]")
    print(f"  Probability range (nonzero): [{nonzero_probs.min():.2f}, {nonzero_probs.max():.2f}]")

    # Default output path
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'paper_data.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build save dict
    save_dict = {
        'n_subjects': n_subjects,
        'n_choices_per_subject': n_choices_per_subject,
        'total_choices': total_choices,
        'prize_min': all_prizes.min(),
        'prize_max': all_prizes.max(),
        'prob_min': nonzero_probs.min(),
        'prob_max': nonzero_probs.max(),
    }

    # Add direct prior if provided
    if direct_prior is not None:
        print("\nDirect Prior:")
        for param, spec in direct_prior.items():
            print(f"  {param}: mu={spec['mu']}, sigma2={spec['sigma2']}")
            save_dict[f'dprior_{param}_mu'] = spec['mu']
            save_dict[f'dprior_{param}_sigma2'] = spec['sigma2']

    # Add prior selection choices if provided
    if prior_selection is not None:
        print("\nPrior Selection Choices:")
        for key, val in prior_selection.items():
            print(f"  {key}: {val}")
            save_dict[f'ps_{key}'] = val

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    gen_paper_data()
