"""
Data cleaning module for the BHM Risk Preferences project.

Loads raw experimental data and shapes it for use with bamcmc.

Input: data/input/dbar.csv (Andersen et al. 2008)
Output: data/input/dbar_clean.npz (when numpy available)
        data/input/dbar_clean.json (fallback)
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

# Try to import numpy; if not available, use JSON fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "input" / "dbar.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "input" / "dbar_clean.npz"
CLEAN_DATA_JSON_PATH = PROJECT_ROOT / "data" / "input" / "dbar_clean.json"


def load_raw_data():
    """Load raw CSV and filter to risk aversion task."""
    with open(RAW_DATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to RA task (lottery choices for risk aversion)
    ra_rows = [r for r in rows if r['RA'] == '1']

    return ra_rows


def parse_ra_data(ra_rows):
    """
    Parse risk aversion task data into structured arrays.

    Each lottery choice has:
    - Lottery A: prizes (vprizea1, vprizea2) with prob (rowproba, 1-rowproba)
    - Lottery B: prizes (vprizeb1, vprizeb2) with prob (rowproba, 1-rowproba)
    - Choice: choicea = 1 if chose lottery A, 0 if chose lottery B

    Returns dict with:
    - subject_ids: unique subject identifiers
    - n_subjects: number of subjects
    - n_choices: number of choices per subject (should be 40)
    - prize_a_high: (n_subjects, n_choices) high prize for lottery A
    - prize_a_low: (n_subjects, n_choices) low prize for lottery A
    - prize_b_high: (n_subjects, n_choices) high prize for lottery B
    - prize_b_low: (n_subjects, n_choices) low prize for lottery B
    - prob_high: (n_subjects, n_choices) probability of high prize
    - choices: (n_subjects, n_choices) 1 = chose A, 0 = chose B
    """
    # Group by subject
    by_subject = defaultdict(list)
    for r in ra_rows:
        by_subject[r['userid']].append(r)

    # Sort subjects by ID
    subject_ids = sorted(by_subject.keys(), key=lambda x: int(x))
    n_subjects = len(subject_ids)

    # Check all subjects have same number of choices
    n_choices_list = [len(by_subject[s]) for s in subject_ids]
    assert len(set(n_choices_list)) == 1, f"Unequal choices per subject: {set(n_choices_list)}"
    n_choices = n_choices_list[0]

    print(f"  Subjects: {n_subjects}")
    print(f"  Choices per subject: {n_choices}")

    # Initialize as nested lists (works without numpy)
    prize_a_high = [[0.0] * n_choices for _ in range(n_subjects)]
    prize_a_low = [[0.0] * n_choices for _ in range(n_subjects)]
    prize_b_high = [[0.0] * n_choices for _ in range(n_subjects)]
    prize_b_low = [[0.0] * n_choices for _ in range(n_subjects)]
    prob_high = [[0.0] * n_choices for _ in range(n_subjects)]
    choices = [[0] * n_choices for _ in range(n_subjects)]

    # Fill arrays
    for i, sid in enumerate(subject_ids):
        # Sort by decision number within subject
        subject_rows = sorted(by_subject[sid], key=lambda r: int(r['decision']))

        for j, r in enumerate(subject_rows):
            prize_a_high[i][j] = float(r['vprizea1'])
            prize_a_low[i][j] = float(r['vprizea2'])
            prize_b_high[i][j] = float(r['vprizeb1'])
            prize_b_low[i][j] = float(r['vprizeb2'])
            prob_high[i][j] = float(r['rowproba'])
            # choicea: 1 = chose lottery A, 0 = chose lottery B
            choices[i][j] = int(r['choicea'])

    return {
        'subject_ids': [int(s) for s in subject_ids],
        'n_subjects': n_subjects,
        'n_choices': n_choices,
        'prize_a_high': prize_a_high,
        'prize_a_low': prize_a_low,
        'prize_b_high': prize_b_high,
        'prize_b_low': prize_b_low,
        'prob_high': prob_high,
        'choices': choices,
    }


def _flatten(nested_list):
    """Flatten a 2D nested list."""
    return [x for row in nested_list for x in row]


def print_data_summary(data):
    """Print summary statistics of cleaned data."""
    print(f"\n  Data summary:")

    pa_high = _flatten(data['prize_a_high'])
    pa_low = _flatten(data['prize_a_low'])
    pb_high = _flatten(data['prize_b_high'])
    pb_low = _flatten(data['prize_b_low'])
    p_high = _flatten(data['prob_high'])
    ch = _flatten(data['choices'])

    print(f"    Prize A high: [{min(pa_high):.0f}, {max(pa_high):.0f}]")
    print(f"    Prize A low:  [{min(pa_low):.0f}, {max(pa_low):.0f}]")
    print(f"    Prize B high: [{min(pb_high):.0f}, {max(pb_high):.0f}]")
    print(f"    Prize B low:  [{min(pb_low):.0f}, {max(pb_low):.0f}]")
    print(f"    Prob high:    [{min(p_high):.2f}, {max(p_high):.2f}]")
    print(f"    Choice A rate: {sum(ch) / len(ch):.3f}")

    # Unique probability values
    unique_probs = sorted(set(p_high))
    print(f"    Unique probabilities: {len(unique_probs)}")
    print(f"      {unique_probs}")


def clean_and_save_data():
    """Main function to clean data and save."""
    print(f"  Loading raw data from: {RAW_DATA_PATH}")
    ra_rows = load_raw_data()
    print(f"  Loaded {len(ra_rows)} risk aversion task rows")

    print(f"\n  Parsing data...")
    data = parse_ra_data(ra_rows)

    print_data_summary(data)

    if NUMPY_AVAILABLE:
        # Save to npz (preferred format for bamcmc)
        print(f"\n  Saving cleaned data to: {CLEAN_DATA_PATH}")
        np.savez(
            CLEAN_DATA_PATH,
            subject_ids=np.array(data['subject_ids'], dtype=np.int32),
            n_subjects=data['n_subjects'],
            n_choices=data['n_choices'],
            prize_a_high=np.array(data['prize_a_high'], dtype=np.float64),
            prize_a_low=np.array(data['prize_a_low'], dtype=np.float64),
            prize_b_high=np.array(data['prize_b_high'], dtype=np.float64),
            prize_b_low=np.array(data['prize_b_low'], dtype=np.float64),
            prob_high=np.array(data['prob_high'], dtype=np.float64),
            choices=np.array(data['choices'], dtype=np.int32),
        )
    else:
        # Save to JSON (fallback when numpy not available)
        print(f"\n  NumPy not available. Saving to JSON: {CLEAN_DATA_JSON_PATH}")
        with open(CLEAN_DATA_JSON_PATH, 'w') as f:
            json.dump(data, f)
        print("  Note: Run with numpy to generate .npz file for bamcmc.")


def load_clean_data():
    """Load cleaned data from npz or json file."""
    if NUMPY_AVAILABLE and CLEAN_DATA_PATH.exists():
        return dict(np.load(CLEAN_DATA_PATH))
    elif CLEAN_DATA_JSON_PATH.exists():
        with open(CLEAN_DATA_JSON_PATH, 'r') as f:
            data = json.load(f)
        if NUMPY_AVAILABLE:
            # Convert lists to numpy arrays
            data['subject_ids'] = np.array(data['subject_ids'], dtype=np.int32)
            data['prize_a_high'] = np.array(data['prize_a_high'], dtype=np.float64)
            data['prize_a_low'] = np.array(data['prize_a_low'], dtype=np.float64)
            data['prize_b_high'] = np.array(data['prize_b_high'], dtype=np.float64)
            data['prize_b_low'] = np.array(data['prize_b_low'], dtype=np.float64)
            data['prob_high'] = np.array(data['prob_high'], dtype=np.float64)
            data['choices'] = np.array(data['choices'], dtype=np.int32)
        return data
    else:
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEAN_DATA_PATH} or {CLEAN_DATA_JSON_PATH}. "
            "Run clean_and_save_data() first."
        )


def make_bamcmc_data(clean_data):
    """
    Convert cleaned data to bamcmc format (simple version).

    bamcmc expects:
    data = {
        "static": tuple of Python scalars (hashable),
        "int": tuple of integer arrays,
        "float": tuple of float arrays,
    }
    """
    return {
        "static": (
            int(clean_data['n_subjects']),
            int(clean_data['n_choices']),
        ),
        "int": (
            clean_data['choices'],  # (n_subjects, n_choices)
        ),
        "float": (
            clean_data['prize_a_high'],   # (n_subjects, n_choices)
            clean_data['prize_a_low'],    # (n_subjects, n_choices)
            clean_data['prize_b_high'],   # (n_subjects, n_choices)
            clean_data['prize_b_low'],    # (n_subjects, n_choices)
            clean_data['prob_high'],      # (n_subjects, n_choices)
        ),
    }


def make_bamcmc_data_indexed(clean_data, prize_scale=1000.0):
    """
    Convert cleaned data to bamcmc format with indexed lookups (optimized).

    This version precomputes unique outcomes and probabilities, then creates
    index arrays for efficient likelihood computation. Instead of computing
    utilities for all (n_subjects * n_choices * 4) prizes, we compute only
    for the unique values and use index lookups.

    Args:
        clean_data: Dict from load_clean_data()
        prize_scale: Normalization factor for prizes (default 1000.0)

    Returns:
        data dict for bamcmc with:
        - static: (n_subjects, n_choices, n_unique_outcomes, n_unique_probs)
        - int: (choices, idx_a_high, idx_a_low, idx_b_high, idx_b_low, idx_prob)
        - float: (unique_outcomes, unique_probs)
    """
    if not NUMPY_AVAILABLE:
        raise RuntimeError("NumPy required for indexed data format")

    n_subjects = int(clean_data['n_subjects'])
    n_choices = int(clean_data['n_choices'])

    # Get all prize arrays
    prize_a_high = np.asarray(clean_data['prize_a_high'])
    prize_a_low = np.asarray(clean_data['prize_a_low'])
    prize_b_high = np.asarray(clean_data['prize_b_high'])
    prize_b_low = np.asarray(clean_data['prize_b_low'])
    prob_high = np.asarray(clean_data['prob_high'])
    choices = np.asarray(clean_data['choices'])

    # Find unique outcomes (normalized by prize_scale)
    all_prizes = np.concatenate([
        prize_a_high.ravel(),
        prize_a_low.ravel(),
        prize_b_high.ravel(),
        prize_b_low.ravel(),
    ])
    unique_outcomes_raw = np.unique(all_prizes)
    unique_outcomes = unique_outcomes_raw / prize_scale

    # Find unique cumulative probabilities
    # For binary lotteries, prob_high IS the cumulative probability for RDU:
    # - P(get x_high or better) = prob_high
    # - P(get x_low or better) = 1 (always, since x_low is worst outcome)
    unique_cumprobs = np.unique(prob_high.ravel())

    print(f"  Indexed data:")
    print(f"    Unique outcomes: {len(unique_outcomes)} (from {len(all_prizes)} total)")
    print(f"    Unique cumulative probs: {len(unique_cumprobs)}")

    # Create lookup dictionaries
    outcome_to_idx = {v: i for i, v in enumerate(unique_outcomes_raw)}
    prob_to_idx = {v: i for i, v in enumerate(unique_cumprobs)}

    # Create index arrays
    idx_a_high = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_a_low = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_b_high = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_b_low = np.zeros((n_subjects, n_choices), dtype=np.int32)
    idx_prob = np.zeros((n_subjects, n_choices), dtype=np.int32)

    for i in range(n_subjects):
        for j in range(n_choices):
            idx_a_high[i, j] = outcome_to_idx[prize_a_high[i, j]]
            idx_a_low[i, j] = outcome_to_idx[prize_a_low[i, j]]
            idx_b_high[i, j] = outcome_to_idx[prize_b_high[i, j]]
            idx_b_low[i, j] = outcome_to_idx[prize_b_low[i, j]]
            idx_prob[i, j] = prob_to_idx[prob_high[i, j]]

    return {
        "static": (
            n_subjects,
            n_choices,
            len(unique_outcomes),
            len(unique_cumprobs),
        ),
        "int": (
            choices.astype(np.int32),
            idx_a_high,
            idx_a_low,
            idx_b_high,
            idx_b_low,
            idx_prob,
        ),
        "float": (
            unique_outcomes.astype(np.float64),
            unique_cumprobs.astype(np.float64),
        ),
    }


if __name__ == "__main__":
    clean_and_save_data()
