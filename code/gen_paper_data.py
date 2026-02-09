"""
Generate summary statistics for the paper.

This module creates NPZ files with data summaries that can be loaded
in the paper's LaTeX (via pythontex or similar).
"""

import numpy as np
from pathlib import Path

from clean_data import load_raw_data, parse_ra_data


def gen_paper_data(output_path=None, direct_prior=None, prior_selection=None):
    """
    Generate summary statistics about the experimental data.

    Saves to NPZ file that can be loaded in LaTeX via pythontex.

    Args:
        output_path: Path to save NPZ file (default: data/output/paper_data.npz)
        direct_prior: Dict with direct prior specification for each parameter
        prior_selection: Dict with researcher choices (rho, alpha_0)
    """
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
        'prob_min': all_probs.min(),
        'prob_max': all_probs.max(),
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
