"""
Generate summary statistics for the paper.

This module creates NPZ files with data summaries that can be loaded
in the paper's LaTeX (via pythontex or similar).
"""

import numpy as np
from pathlib import Path
from scipy.special import gamma as gammafn
from scipy.stats import invgamma

from clean_data import load_clean_data


def _plot_nig_sensitivity(rho_vals, kappa_vals, between_sd, within_sd,
                          within_iqr, output_path):
    """Generate three-panel heatmap of NIG sensitivity analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{lmodern}',
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman'],
        'mathtext.fontset': 'cm',
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (between_sd, 'nig_sensitivity_between.pdf', [0.15, 0.40, 0.65, 0.90]),
        (within_sd, 'nig_sensitivity_within.pdf', [0.15, 0.40, 0.65, 0.90]),
        (within_iqr, 'nig_sensitivity_iqr.pdf', [0.05, 0.15, 0.25, 0.35]),
    ]

    for data, filename, cbar_ticks in panels:
        fig, ax = plt.subplots(figsize=(2.8, 1.9))
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[kappa_vals[0], kappa_vals[-1],
                    rho_vals[0], rho_vals[-1]],
        )
        ax.set_xlabel(r'$\kappa$', fontsize=10)
        ax.set_ylabel(r'$\rho$', fontsize=10)
        ax.set_xticks([0.2, 0.5, 0.8, 1.1, 1.4, 1.7])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks)
        cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fpath = output_path.parent / filename
        fig.savefig(fpath, bbox_inches='tight', backend='pdf')
        plt.close(fig)
        print(f"  Saved: {fpath}")


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
            ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))
            print(f"  L{i+1}: prizes={lot['prizes']}, probs={lot['probs']}, EV={ev:.0f}")
            for j, prize in enumerate(lot['prizes']):
                save_dict[f'ref_lot_{i+1}_prize_{j+1}'] = prize
            for j, prob in enumerate(lot['probs']):
                save_dict[f'ref_lot_{i+1}_prob_{j+1}'] = prob
            save_dict[f'ref_lot_{i+1}_n_outcomes'] = len(lot['prizes'])
            save_dict[f'ref_lot_{i+1}_ev'] = ev

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

    # Pre-compute display values for pythontex (avoid inline math in LaTeX)
    if prior_selection_ce is not None:
        rho = prior_selection_ce['rho']
        kappa = prior_selection_ce['kappa']
        save_dict['rho_pct'] = rho * 100
        save_dict['one_minus_rho_pct'] = (1 - rho) * 100
        save_dict['kappa_pct'] = kappa * 100

    # Add CE elicitation inputs and derived quantities
    if ce_mean_frac is not None and ce_p_risk_seeking is not None:
        save_dict['ce_mean_frac'] = ce_mean_frac
        save_dict['ce_p_risk_seeking'] = ce_p_risk_seeking
        save_dict['ce_mean_frac_pct'] = ce_mean_frac * 100
        save_dict['ce_p_risk_seeking_pct'] = ce_p_risk_seeking * 100
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

    # NIG sensitivity analysis for Section 3 (analytical, no external data needed)
    nig_rho_vals = np.linspace(0.01, 0.99, 50)
    nig_kappa_vals = np.linspace(0.10, 1.70, 50)
    nig_v_star = 1.0
    n_rho, n_kappa = len(nig_rho_vals), len(nig_kappa_vals)

    nig_between_sd = np.zeros((n_rho, n_kappa))
    nig_within_sd = np.zeros((n_rho, n_kappa))
    nig_within_iqr = np.zeros((n_rho, n_kappa))

    for i, rho in enumerate(nig_rho_vals):
        for j, kappa in enumerate(nig_kappa_vals):
            tau0 = np.sqrt(rho * nig_v_star)
            nig_between_sd[i, j] = tau0

            alpha0 = 2.0 + 1.0 / kappa**2
            beta0 = (1.0 - rho) * nig_v_star * (alpha0 - 1.0)

            e_sigma = np.sqrt(beta0) * gammafn(alpha0 - 0.5) / gammafn(alpha0)
            nig_within_sd[i, j] = e_sigma

            q25 = invgamma.ppf(0.25, alpha0, scale=beta0)
            q75 = invgamma.ppf(0.75, alpha0, scale=beta0)
            nig_within_iqr[i, j] = np.sqrt(q75) - np.sqrt(q25)

    save_dict['nig_rho_values'] = nig_rho_vals
    save_dict['nig_kappa_values'] = nig_kappa_vals
    save_dict['nig_between_sd'] = nig_between_sd
    save_dict['nig_within_sd'] = nig_within_sd
    save_dict['nig_within_iqr'] = nig_within_iqr

    # Generate heatmap figure
    _plot_nig_sensitivity(
        nig_rho_vals, nig_kappa_vals,
        nig_between_sd, nig_within_sd, nig_within_iqr,
        output_path.parent.parent.parent / 'plots' / 'paper' / 'nig_sensitivity.pdf',
    )

    print("\nNIG Sensitivity (Section 3):")
    print(f"  rho: [{nig_rho_vals[0]:.2f}, ..., {nig_rho_vals[-1]:.2f}] ({n_rho} values)")
    print(f"  kappa: [{nig_kappa_vals[0]:.2f}, ..., {nig_kappa_vals[-1]:.2f}] ({n_kappa} values)")
    print(f"  v* = {nig_v_star}")

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    gen_paper_data()
