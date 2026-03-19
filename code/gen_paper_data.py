"""
Generate summary statistics for the paper.

This module creates NPZ files with data summaries that can be loaded
in the paper's LaTeX (via pythontex or similar).
"""

import numpy as np
from pathlib import Path

from clean_data import load_clean_data


def gen_paper_data(output_path=None,
                   ce_direct_prior=None, reference_lotteries=None,
                   reference_pair=None, prior_selection_ce=None,
                   ce_mean_frac=None, ce_p_risk_seeking=None):
    """
    Generate summary statistics about the experimental data.

    Saves to NPZ file that can be loaded in LaTeX via pythontex.

    Args:
        output_path: Path to save NPZ file (default: data/output/paper_data.npz)
        ce_direct_prior: Dict with CE-space direct prior (ce1, ce2, ce3, pchoice)
        reference_lotteries: List of 3 reference lottery dicts (prizes, probs)
        reference_pair: Dict with lottery_A and lottery_B for choice probability
        prior_selection_ce: Dict with CE-space researcher choices (rho, kappa)
        ce_mean_frac: Mean CE as fraction of distance from lo to EV
        ce_p_risk_seeking: P(risk-seeking) for each quantity
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

    # Add CE-space direct prior if provided
    # Key mapping: ce1->1, ce2->2, ce3->3 (for pythontex compatibility)
    if ce_direct_prior is not None:
        _ce_key_map = {'ce1': '1', 'ce2': '2', 'ce3': '3'}
        print("\nCE-Space Direct Prior:")
        for qty, spec in ce_direct_prior.items():
            print(f"  {qty}: mu={spec['mu']}, sigma2={spec['sigma2']}, "
                  f"lo={spec['lo']}, hi={spec['hi']}")
            if qty in _ce_key_map:
                suffix = _ce_key_map[qty]
                save_dict[f'ce_mu_{suffix}'] = spec['mu']
                save_dict[f'ce_sigma2_{suffix}'] = spec['sigma2']
                save_dict[f'ce_lo_{suffix}'] = spec['lo']
                save_dict[f'ce_hi_{suffix}'] = spec['hi']
            elif qty == 'pchoice':
                save_dict['pchoice_mu'] = spec['mu']
                save_dict['pchoice_sigma2'] = spec['sigma2']
                save_dict['pchoice_lo'] = spec['lo']
                save_dict['pchoice_hi'] = spec['hi']

    # Add reference lotteries if provided
    if reference_lotteries is not None:
        print("\nReference Lotteries:")
        for i, lot in enumerate(reference_lotteries):
            print(f"  L{i+1}: prizes={lot['prizes']}, probs={lot['probs']}")
            for j, prize in enumerate(lot['prizes']):
                save_dict[f'ref_lot_{i+1}_prize_{j+1}'] = prize
            for j, prob in enumerate(lot['probs']):
                save_dict[f'ref_lot_{i+1}_prob_{j+1}'] = prob
            save_dict[f'ref_lot_{i+1}_n_outcomes'] = len(lot['prizes'])

    # Add reference pair if provided
    if reference_pair is not None:
        print("\nReference Pair:")
        for side, key in [('A', 'lottery_A'), ('B', 'lottery_B')]:
            lot = reference_pair[key]
            print(f"  {side}: prizes={lot['prizes']}, probs={lot['probs']}")
            for j, prize in enumerate(lot['prizes']):
                save_dict[f'ref_pair_{side}_prize_{j+1}'] = prize
            for j, prob in enumerate(lot['probs']):
                save_dict[f'ref_pair_{side}_prob_{j+1}'] = prob

    # Add CE-space prior selection choices if provided
    if prior_selection_ce is not None:
        print("\nCE-Space Prior Selection:")
        for key, val in prior_selection_ce.items():
            print(f"  {key}: {val}")
            save_dict[f'ce_ps_{key}'] = val

    # Add CE elicitation inputs and derived quantities
    if ce_mean_frac is not None and ce_p_risk_seeking is not None:
        save_dict['ce_mean_frac'] = ce_mean_frac
        save_dict['ce_p_risk_seeking'] = ce_p_risk_seeking
        print(f"\nCE Elicitation Inputs:")
        print(f"  mean_frac: {ce_mean_frac}")
        print(f"  p_risk_seeking: {ce_p_risk_seeking}")
        # Save per-lottery derived quantities (EV, mean CE, implied var/sd)
        if ce_direct_prior is not None and reference_lotteries is not None:
            for k, lot in enumerate(reference_lotteries):
                suffix = str(k + 1)
                spec = ce_direct_prior[f'ce{k+1}']
                ev = spec.get('ev', sum(p * x for p, x in zip(lot['probs'], lot['prizes'])))
                sd = np.sqrt(spec['sigma2'])
                save_dict[f'ce_ev_{suffix}'] = ev
                save_dict[f'ce_mean_{suffix}'] = spec['mu']
                save_dict[f'ce_var_{suffix}'] = spec['sigma2']
                save_dict[f'ce_sd_{suffix}'] = sd
                print(f"  L{suffix}: EV={ev:.1f}, mean={spec['mu']:.2f}, "
                      f"var={spec['sigma2']:.2f}, sd={sd:.2f}")
        # Save pchoice derived quantities
        if ce_direct_prior is not None and 'pchoice' in ce_direct_prior:
            pc = ce_direct_prior['pchoice']
            pc_sd = np.sqrt(pc['sigma2'])
            save_dict['pchoice_var'] = pc['sigma2']
            save_dict['pchoice_sd'] = pc_sd
            print(f"  P: mu={pc['mu']:.2f}, var={pc['sigma2']:.4f}, sd={pc_sd:.4f}")

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    gen_paper_data()
