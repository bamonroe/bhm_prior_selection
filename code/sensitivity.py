"""
Sensitivity Analysis: Hyperprior parameters and prior predictive CEs over (ρ, κ) grid.

For each (ρ, κ) combination:
1. Derive hyperprior parameters via variance decomposition
2. Compute prior predictive CE distribution (MC sampling)
3. Report summary statistics: mean, median, SD, IQR, P(CE < min_prize), P(CE > max_prize)

Output: NPZ file with all grid results + formatted LaTeX table.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_sensitivity_grid(
    direct_prior: Dict,
    rho_values: List[float],
    kappa_values: List[float],
    prizes: np.ndarray,
    n_mc_draws: int = 5000,
    n_subjects: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Compute hyperprior parameters and prior predictive CE statistics over (ρ, κ) grid.

    Args:
        direct_prior: Direct prior specification (same format as main.py DIRECT_PRIOR)
        rho_values: List of ρ values to evaluate
        kappa_values: List of κ values to evaluate
        prizes: Reference lottery prizes [min, mid, max]
        n_mc_draws: MC draws per grid point for CE computation
        n_subjects: Subjects per MC draw
        seed: Base random seed

    Returns:
        Dict with keys:
            'rho_values', 'kappa_values': 1D arrays
            'alpha_0_values': 1D array (corresponding to kappa_values)
            'ce_mean': (n_rho, n_kappa) array
            'ce_median': (n_rho, n_kappa) array
            'ce_sd': (n_rho, n_kappa) array
            'ce_iqr': (n_rho, n_kappa) array (IQR = Q75 - Q25)
            'ce_q05': (n_rho, n_kappa) array (5th percentile)
            'ce_q95': (n_rho, n_kappa) array (95th percentile)
            'tau_0_r': (n_rho, n_kappa) array (SD of population mean prior for r)
            'expected_sigma_r': (n_rho, n_kappa) array (E[σ] for r)
    """
    from prior_selection import (
        rdu_prelec_prior_selection, derive_hyperpriors, kappa_to_alpha0
    )
    from prior_predictive_ce import compute_prior_predictive_ce

    ps = rdu_prelec_prior_selection(direct_prior)

    n_rho = len(rho_values)
    n_kappa = len(kappa_values)

    # Pre-compute alpha_0 values
    alpha_0_values = np.array([kappa_to_alpha0(k) for k in kappa_values])

    # Output arrays
    ce_mean = np.zeros((n_rho, n_kappa))
    ce_median = np.zeros((n_rho, n_kappa))
    ce_sd = np.zeros((n_rho, n_kappa))
    ce_iqr = np.zeros((n_rho, n_kappa))
    ce_q05 = np.zeros((n_rho, n_kappa))
    ce_q95 = np.zeros((n_rho, n_kappa))
    tau_0_r = np.zeros((n_rho, n_kappa))
    expected_sigma_r = np.zeros((n_rho, n_kappa))

    total = n_rho * n_kappa
    count = 0

    for i, rho in enumerate(rho_values):
        for j, kappa in enumerate(kappa_values):
            count += 1
            print(f"  [{count}/{total}] ρ={rho:.2f}, κ={kappa:.3f} (α₀={alpha_0_values[j]:.1f})")

            # Derive hyperpriors
            hp = derive_hyperpriors(ps, rho=rho, kappa=kappa)

            hyperprior_params = {
                'mu_0': hp.mu_0,
                'tau_0_sq': hp.tau_0_sq,
                'alpha_0': hp.alpha_0,
                'beta_0': hp.beta_0,
            }

            # Store hyperprior summaries (for r parameter, index 0)
            tau_0_r[i, j] = np.sqrt(hp.tau_0_sq[0])
            expected_sigma_r[i, j] = np.sqrt(hp.beta_0[0] / (hp.alpha_0 - 1))

            # Compute prior predictive CE distribution
            grid_seed = seed + i * n_kappa + j
            ces = compute_prior_predictive_ce(
                hyperprior_params, prizes, n_subjects,
                n_mc_draws=n_mc_draws, seed=grid_seed,
            )
            ces_flat = np.array(ces).ravel()

            # Clip extreme CEs for robust statistics
            valid = np.isfinite(ces_flat)
            ces_valid = ces_flat[valid]

            if len(ces_valid) > 0:
                ce_mean[i, j] = np.mean(ces_valid)
                ce_median[i, j] = np.median(ces_valid)
                ce_sd[i, j] = np.std(ces_valid)
                q05, q25, q75, q95 = np.percentile(ces_valid, [5, 25, 75, 95])
                ce_iqr[i, j] = q75 - q25
                ce_q05[i, j] = q05
                ce_q95[i, j] = q95
            else:
                ce_mean[i, j] = np.nan
                ce_median[i, j] = np.nan
                ce_sd[i, j] = np.nan
                ce_iqr[i, j] = np.nan
                ce_q05[i, j] = np.nan
                ce_q95[i, j] = np.nan

    return {
        'rho_values': np.array(rho_values),
        'kappa_values': np.array(kappa_values),
        'alpha_0_values': alpha_0_values,
        'ce_mean': ce_mean,
        'ce_median': ce_median,
        'ce_sd': ce_sd,
        'ce_iqr': ce_iqr,
        'ce_q05': ce_q05,
        'ce_q95': ce_q95,
        'tau_0_r': tau_0_r,
        'expected_sigma_r': expected_sigma_r,
    }


def format_latex_table(results: Dict, prizes: np.ndarray) -> str:
    """
    Format sensitivity results as a LaTeX table.

    Produces a table with ρ in rows, κ in columns, showing
    prior predictive CE median [90% CI].

    Args:
        results: Output from compute_sensitivity_grid()
        prizes: Reference lottery prizes for context

    Returns:
        LaTeX table string
    """
    rho_vals = results['rho_values']
    kappa_vals = results['kappa_values']
    alpha_0_vals = results['alpha_0_values']

    n_kappa = len(kappa_vals)

    # Header
    col_spec = 'r' + 'c' * n_kappa
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Prior predictive CE distribution: median [90\% CI] across $(\rho, \kappa)$ grid.}')
    lines.append(r'\label{tab:sensitivity}')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Column headers: kappa values with alpha_0 equivalent
    header = r'$\rho$'
    for k, a0 in zip(kappa_vals, alpha_0_vals):
        header += f' & $\\kappa = {k:.2f}$'
    header += r' \\'
    lines.append(header)

    # Sub-header with alpha_0 equivalents
    subheader = ''
    for a0 in alpha_0_vals:
        subheader += f' & $(\\alpha_0 = {a0:.0f})$'
    subheader += r' \\'
    lines.append(subheader)
    lines.append(r'\midrule')

    # Data rows
    for i, rho in enumerate(rho_vals):
        row = f'{rho:.2f}'
        for j in range(n_kappa):
            med = results['ce_median'][i, j]
            q05 = results['ce_q05'][i, j]
            q95 = results['ce_q95'][i, j]
            row += f' & ${med:.1f}\\;[{q05:.1f},\\,{q95:.1f}]$'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    ev = np.mean(prizes)
    lines.append(f'\\\\[2pt]\\footnotesize{{Reference lottery: prizes '
                 f'\\${prizes[0]:.0f}/\\${prizes[1]:.0f}/\\${prizes[2]:.0f} '
                 f'equally likely (EV = \\${ev:.0f}).}}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def run_sensitivity(
    direct_prior: Dict,
    data_dir: Path,
    output_dir: Path,
    rho_values: Optional[List[float]] = None,
    kappa_values: Optional[List[float]] = None,
    n_mc_draws: int = 5000,
    n_subjects: int = 50,
    seed: int = 42,
):
    """
    Run sensitivity analysis and save results.

    Args:
        direct_prior: Direct prior specification
        data_dir: Path to data directory (for loading prizes)
        output_dir: Path to save results
        rho_values: ρ grid values (default: [0.05, 0.1, 0.2, 0.3, 0.5])
        kappa_values: κ grid values (default: [0.15, 0.24, 0.5, 1.0])
        n_mc_draws: MC draws per grid point
        n_subjects: Subjects per MC draw
        seed: Random seed
    """
    from clean_data import load_clean_data

    if rho_values is None:
        rho_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    if kappa_values is None:
        kappa_values = [0.15, 0.24, 0.50, 1.00]

    # Load prizes from clean data
    clean = load_clean_data()
    all_prizes = np.concatenate([
        np.asarray(clean['prizes_L']).ravel(),
        np.asarray(clean['prizes_R']).ravel(),
    ])
    p_min = float(np.min(all_prizes))
    p_max = float(np.max(all_prizes))
    p_mid = (p_min + p_max) / 2.0
    prizes = np.array([p_min, p_mid, p_max])

    print(f"Sensitivity analysis over {len(rho_values)} × {len(kappa_values)} grid")
    print(f"  ρ ∈ {rho_values}")
    print(f"  κ ∈ {kappa_values}")
    print(f"  Reference lottery prizes: ${prizes[0]:.0f}, ${prizes[1]:.0f}, ${prizes[2]:.0f}")
    print(f"  MC draws: {n_mc_draws}, subjects/draw: {n_subjects}")
    print()

    results = compute_sensitivity_grid(
        direct_prior=direct_prior,
        rho_values=rho_values,
        kappa_values=kappa_values,
        prizes=prizes,
        n_mc_draws=n_mc_draws,
        n_subjects=n_subjects,
        seed=seed,
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / 'sensitivity_grid.npz'
    np.savez(npz_path, **results)
    print(f"\nSaved results to: {npz_path}")

    # Generate and save LaTeX table
    latex_table = format_latex_table(results, prizes)
    tex_path = output_dir / 'sensitivity_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {tex_path}")

    # Print hyperprior parameter summary (for r parameter)
    print("\n" + "=" * 70)
    print("Derived Hyperpriors for r (CRRA): τ₀ / E[σ]")
    print("=" * 70)
    header = f"{'ρ':>6}"
    for k in kappa_values:
        header += f"  {'κ=' + f'{k:.2f}':>20}"
    print(header)
    print("-" * 70)
    for i, rho in enumerate(rho_values):
        row = f"{rho:>6.2f}"
        for j in range(len(kappa_values)):
            t0 = results['tau_0_r'][i, j]
            es = results['expected_sigma_r'][i, j]
            row += f"  {t0:>8.3f} / {es:.3f}  "
        print(row)
    print("=" * 70)

    # Print CE summary
    print("\n" + "=" * 70)
    print("Prior Predictive CE: Median [90% CI]")
    print("=" * 70)
    header = f"{'ρ':>6}"
    for k in kappa_values:
        header += f"  {'κ=' + f'{k:.2f}':>20}"
    print(header)
    print("-" * 70)
    for i, rho in enumerate(rho_values):
        row = f"{rho:>6.2f}"
        for j in range(len(kappa_values)):
            med = results['ce_median'][i, j]
            q05 = results['ce_q05'][i, j]
            q95 = results['ce_q95'][i, j]
            row += f"  {med:>5.1f} [{q05:>5.1f},{q95:>5.1f}]"
        print(row)
    print("=" * 70)

    return results
