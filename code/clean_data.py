"""
Data cleaning module for the BHM Risk Preferences project.

Loads raw experimental data and shapes it for use with bamcmc.
Supports two datasets:
  - 'hn2016': Harrison & Ng (2016), 3-outcome lotteries in USD
  - 'andersen2008': Andersen et al. (2008), binary lotteries in DKK

Both are output in a unified 3-outcome format:
  prizes_L/R: (n_subjects, n_choices, 3) sorted ascending
  probs_L/R:  (n_subjects, n_choices, 3) summing to 1
  choices:    (n_subjects, n_choices)     1=Left(A), 0=Right(B)

For the Andersen binary lotteries, the 3rd slot is padded with a
zero-probability duplicate of the lowest outcome.
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_ANDERSEN = PROJECT_ROOT / "data" / "input" / "dbar.csv"
RAW_DATA_HN = PROJECT_ROOT / "data" / "input" / "hn2016.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "input" / "dbar_clean.npz"


# =============================================================================
# Harrison & Ng (2016) Data
# =============================================================================

def load_raw_data_hn():
    """Load Harrison & Ng (2016) risk data from CSV.

    Filters to risk task rows (risk==1).

    Returns list of row dicts (one per lottery choice).
    """
    import pandas as pd
    df = pd.read_csv(RAW_DATA_HN)
    # Filter to risk task
    df = df[df['risk'] == 1].copy()
    return df.to_dict('records')


def parse_hn_data(rows):
    """
    Parse Harrison & Ng (2016) data into unified 3-outcome format.

    The raw data has 4 outcome slots per side (L/R), but slot 1 is always
    zero (unused). Slots 2-4 carry the actual lottery with probs summing to 1.

    Choice encoding: raw choice=1 means "chose Right". We flip to model
    convention: 1=Left(A), 0=Right(B).

    Returns dict with unified 3-outcome arrays.
    """
    # Group by subject
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r['IDnum']].append(r)

    subject_ids = sorted(by_subject.keys(), key=lambda x: int(x))
    n_subjects = len(subject_ids)

    n_choices_list = [len(by_subject[s]) for s in subject_ids]
    assert len(set(n_choices_list)) == 1, f"Unequal choices per subject: {set(n_choices_list)}"
    n_choices = n_choices_list[0]

    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices}")

    # Initialize arrays: (n_subjects, n_choices, 3)
    prizes_L = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    prizes_R = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    probs_L = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    probs_R = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    choices = [[0]*n_choices for _ in range(n_subjects)]

    for i, sid in enumerate(subject_ids):
        subject_rows = by_subject[sid]
        for j, r in enumerate(subject_rows):
            # Extract prizes and probs from slots 2-4 (slot 1 is always zero)
            # Sort each side by prize value ascending
            for side, prizes_arr, probs_arr in [
                ('L', prizes_L, probs_L),
                ('R', prizes_R, probs_R),
            ]:
                raw_prizes = [
                    float(r[f'prize2{side}']),
                    float(r[f'prize3{side}']),
                    float(r[f'prize4{side}']),
                ]
                raw_probs = [
                    float(r[f'prob2{side}']),
                    float(r[f'prob3{side}']),
                    float(r[f'prob4{side}']),
                ]
                # Sort by prize ascending
                paired = sorted(zip(raw_prizes, raw_probs), key=lambda x: x[0])
                for k in range(3):
                    prizes_arr[i][j][k] = paired[k][0]
                    probs_arr[i][j][k] = paired[k][1]

            # Flip choice: raw 1=Right -> model 1=Left
            choices[i][j] = 1 - int(r['choice'])

    return {
        'subject_ids': [int(s) for s in subject_ids],
        'n_subjects': n_subjects,
        'n_choices': n_choices,
        'prizes_L': prizes_L,
        'prizes_R': prizes_R,
        'probs_L': probs_L,
        'probs_R': probs_R,
        'choices': choices,
    }


# =============================================================================
# Andersen et al. (2008) Data
# =============================================================================

def load_raw_data():
    """Load raw Andersen CSV and filter to risk aversion task."""
    with open(RAW_DATA_ANDERSEN, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    ra_rows = [r for r in rows if r['RA'] == '1']
    return ra_rows


def parse_ra_data(ra_rows):
    """
    Parse Andersen (2008) risk aversion data into unified 3-outcome format.

    Binary lottery A=(x_high, p; x_low, 1-p) becomes 3-outcome:
        prizes = [x_low, x_low, x_high]    (duplicate lowest)
        probs  = [0,     1-p,    p]

    Choice encoding: raw choicea=1 means "chose A" (Left). But the raw data
    actually has choicea as a right-side indicator (1=chose right/B), so we
    flip: model_choice = 1 - choicea => 1=A(Left), 0=B(Right).

    Returns dict with unified 3-outcome arrays.
    """
    by_subject = defaultdict(list)
    for r in ra_rows:
        by_subject[r['userid']].append(r)

    subject_ids = sorted(by_subject.keys(), key=lambda x: int(x))
    n_subjects = len(subject_ids)

    n_choices_list = [len(by_subject[s]) for s in subject_ids]
    assert len(set(n_choices_list)) == 1, f"Unequal choices per subject: {set(n_choices_list)}"
    n_choices = n_choices_list[0]

    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices}")

    # Initialize arrays: (n_subjects, n_choices, 3)
    prizes_L = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    prizes_R = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    probs_L = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    probs_R = [[[0.0]*3 for _ in range(n_choices)] for _ in range(n_subjects)]
    choices = [[0]*n_choices for _ in range(n_subjects)]

    for i, sid in enumerate(subject_ids):
        subject_rows = sorted(by_subject[sid], key=lambda r: int(r['decision']))

        for j, r in enumerate(subject_rows):
            pa_high = float(r['vprizea1'])  # A high prize
            pa_low = float(r['vprizea2'])   # A low prize
            pb_high = float(r['vprizeb1'])  # B high prize
            pb_low = float(r['vprizeb2'])   # B low prize
            p = float(r['rowproba'])        # prob of high prize

            # Pad binary -> 3-outcome: [x_low, x_low, x_high] with probs [0, 1-p, p]
            # Left = A
            prizes_L[i][j] = [pa_low, pa_low, pa_high]
            probs_L[i][j] = [0.0, 1.0 - p, p]
            # Right = B
            prizes_R[i][j] = [pb_low, pb_low, pb_high]
            probs_R[i][j] = [0.0, 1.0 - p, p]

            # choicea in raw data: 1 = chose right/B. Flip for model convention.
            choices[i][j] = 1 - int(r['choicea'])

    return {
        'subject_ids': [int(s) for s in subject_ids],
        'n_subjects': n_subjects,
        'n_choices': n_choices,
        'prizes_L': prizes_L,
        'prizes_R': prizes_R,
        'probs_L': probs_L,
        'probs_R': probs_R,
        'choices': choices,
    }


# =============================================================================
# Unified Data Summary
# =============================================================================

def _flatten(nested_list):
    """Flatten a 2D nested list."""
    return [x for row in nested_list for x in row]


def _flatten3d(nested_list):
    """Flatten a 3D nested list."""
    return [x for subj in nested_list for row in subj for x in row]


def print_data_summary(data):
    """Print summary statistics of cleaned data."""
    print(f"\n  Data summary:")

    all_prizes_L = _flatten3d(data['prizes_L'])
    all_prizes_R = _flatten3d(data['prizes_R'])
    all_probs_L = _flatten3d(data['probs_L'])
    all_probs_R = _flatten3d(data['probs_R'])
    ch = _flatten(data['choices'])

    print(f"    Prize L range: [{min(all_prizes_L):.0f}, {max(all_prizes_L):.0f}]")
    print(f"    Prize R range: [{min(all_prizes_R):.0f}, {max(all_prizes_R):.0f}]")
    print(f"    Choice Left rate: {sum(ch) / len(ch):.3f}")

    # Unique prize values
    all_prizes = sorted(set(all_prizes_L + all_prizes_R))
    print(f"    Unique prizes: {len(all_prizes)}")
    print(f"      {all_prizes}")

    # Unique probabilities (non-zero)
    nonzero_probs = sorted(set(p for p in all_probs_L + all_probs_R if p > 0))
    print(f"    Unique non-zero probs: {len(nonzero_probs)}")
    if len(nonzero_probs) <= 20:
        print(f"      {nonzero_probs}")


# =============================================================================
# Clean and Save
# =============================================================================

def clean_and_save_data(dataset='hn2016'):
    """
    Main function to clean data and save.

    Args:
        dataset: 'hn2016' for Harrison & Ng, 'andersen2008' for Andersen et al.
    """
    if dataset == 'hn2016':
        print(f"  Loading raw data from: {RAW_DATA_HN}")
        rows = load_raw_data_hn()
        print(f"  Loaded {len(rows)} rows")
        print(f"\n  Parsing H&N (2016) data...")
        data = parse_hn_data(rows)
    elif dataset == 'andersen2008':
        print(f"  Loading raw data from: {RAW_DATA_ANDERSEN}")
        ra_rows = load_raw_data()
        print(f"  Loaded {len(ra_rows)} risk aversion task rows")
        print(f"\n  Parsing Andersen (2008) data...")
        data = parse_ra_data(ra_rows)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print_data_summary(data)

    print(f"\n  Saving cleaned data to: {CLEAN_DATA_PATH}")
    np.savez(
        CLEAN_DATA_PATH,
        subject_ids=np.array(data['subject_ids'], dtype=np.int32),
        n_subjects=data['n_subjects'],
        n_choices=data['n_choices'],
        prizes_L=np.array(data['prizes_L'], dtype=np.float64),
        prizes_R=np.array(data['prizes_R'], dtype=np.float64),
        probs_L=np.array(data['probs_L'], dtype=np.float64),
        probs_R=np.array(data['probs_R'], dtype=np.float64),
        choices=np.array(data['choices'], dtype=np.int32),
    )


def load_clean_data():
    """Load cleaned data from npz file."""
    if CLEAN_DATA_PATH.exists():
        return dict(np.load(CLEAN_DATA_PATH))
    else:
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEAN_DATA_PATH}. "
            "Run clean_and_save_data() first."
        )


# =============================================================================
# bamcmc Indexed Format
# =============================================================================

def make_bamcmc_data_indexed(clean_data):
    """
    Convert cleaned data to bamcmc format with indexed lookups (optimized).

    For 3-outcome lotteries, we index into a unique outcomes array for each
    of the 6 outcome slots (3 per side). Probabilities are passed as float
    arrays per choice (not indexed, since they vary per choice).

    Prize amounts are stored in original currency units (no normalization).
    The contextual utility model in the likelihood normalizes utility
    differences by the outcome range, making the scale irrelevant.

    Args:
        clean_data: Dict from load_clean_data()

    Returns:
        data dict for bamcmc with:
        - static: (n_subjects, n_choices, n_unique_outcomes)
        - int: (choices, idx_L_1, idx_L_2, idx_L_3, idx_R_1, idx_R_2, idx_R_3)
        - float: (unique_outcomes, prob_L_1, prob_L_2, prob_L_3,
                   prob_R_1, prob_R_2, prob_R_3, hyperprior_arr_placeholder)
    """
    n_subjects = int(clean_data['n_subjects'])
    n_choices = int(clean_data['n_choices'])

    prizes_L = np.asarray(clean_data['prizes_L'])  # (n_subjects, n_choices, 3)
    prizes_R = np.asarray(clean_data['prizes_R'])
    probs_L = np.asarray(clean_data['probs_L'])
    probs_R = np.asarray(clean_data['probs_R'])
    choices = np.asarray(clean_data['choices'])

    # Find unique outcomes (in original currency units)
    all_prizes = np.concatenate([prizes_L.ravel(), prizes_R.ravel()])
    unique_outcomes = np.unique(all_prizes)

    print(f"  Indexed data:")
    print(f"    Unique outcomes: {len(unique_outcomes)} (from {len(all_prizes)} total)")

    # Create lookup dictionary
    outcome_to_idx = {v: i for i, v in enumerate(unique_outcomes)}

    # Create index arrays: (n_subjects, n_choices) for each of 6 outcome slots
    idx_L_1 = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_L_2 = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_L_3 = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_R_1 = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_R_2 = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_R_3 = np.zeros((n_subjects, n_choices), dtype=np.int32)

    for i in range(n_subjects):
        for j in range(n_choices):
            idx_L_1[i, j] = outcome_to_idx[prizes_L[i, j, 0]]
            idx_L_2[i, j] = outcome_to_idx[prizes_L[i, j, 1]]
            idx_L_3[i, j] = outcome_to_idx[prizes_L[i, j, 2]]
            idx_R_1[i, j] = outcome_to_idx[prizes_R[i, j, 0]]
            idx_R_2[i, j] = outcome_to_idx[prizes_R[i, j, 1]]
            idx_R_3[i, j] = outcome_to_idx[prizes_R[i, j, 2]]

    # Probability arrays: (n_subjects, n_choices) for each of 6 prob slots
    prob_L_1 = probs_L[:, :, 0].astype(np.float64)
    prob_L_2 = probs_L[:, :, 1].astype(np.float64)
    prob_L_3 = probs_L[:, :, 2].astype(np.float64)
    prob_R_1 = probs_R[:, :, 0].astype(np.float64)
    prob_R_2 = probs_R[:, :, 1].astype(np.float64)
    prob_R_3 = probs_R[:, :, 2].astype(np.float64)

    return {
        "static": (
            n_subjects,
            n_choices,
            len(unique_outcomes),
        ),
        "int": (
            choices.astype(np.int32),
            idx_L_1,
            idx_L_2,
            idx_L_3,
            idx_R_1,
            idx_R_2,
            idx_R_3,
        ),
        "float": (
            unique_outcomes.astype(np.float64),
            prob_L_1,
            prob_L_2,
            prob_L_3,
            prob_R_1,
            prob_R_2,
            prob_R_3,
        ),
    }


if __name__ == "__main__":
    clean_and_save_data()
