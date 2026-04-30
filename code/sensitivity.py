"""
Sensitivity Analysis: Beta-Beta hierarchical variance allocation over (ρ, κ) grid.

For each (ρ, κ) combination:
1. Derive Beta-Beta hyperpriors via variance decomposition
2. Sample prior predictive CEs from the hierarchy (pure Beta draws)
3. Report between-population SD, within-population SD, and within-population SD IQR

The marginal CE distribution is invariant to (ρ, κ) by construction. The
interesting sensitivity is how the hierarchical structure allocates variance:
  - Between-population SD: robust scale of population-mean CEs across MC draws
    (measures uncertainty about the population center — governed by ρ)
  - Within-population SD: median of within-draw CE SDs across MC draws
    (measures typical individual heterogeneity — governed by 1−ρ)
  - Within-population SD IQR: interquartile range of draw-level SDs
    (measures variability in heterogeneity — governed by κ)

Output: NPZ file with all grid results + formatted LaTeX table.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def _sample_beta_beta_ce(
    hp,
    quantity_idx: int,
    n_mc_draws: int,
    n_subjects: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample prior predictive CEs from the Beta-Beta hierarchy.

    For a single quantity k:
      μ_k ~ Beta(a_μ[k], b_μ[k])
      ψ_k ~ Beta(a_ψ[k], b_ψ[k])
      z_{ik} | μ_k, ψ_k ~ Beta(μ_k(1-ψ_k)/ψ_k, (1-μ_k)(1-ψ_k)/ψ_k)
      Y_{ik} = lo_k + z_{ik} × (hi_k - lo_k)

    Args:
        hp: DerivedHyperpriorsBeta object
        quantity_idx: Index into the quantity arrays (0=CE1, 1=CE2, 2=CE3, 3=P)
        n_mc_draws: Number of population draws
        n_subjects: Number of subjects per population
        rng: numpy random Generator

    Returns:
        Array of shape (n_mc_draws, n_subjects) with CEs on natural scale
    """
    k = quantity_idx
    a_mu = hp.a_mu[k]
    b_mu = hp.b_mu[k]
    a_psi = hp.a_psi[k]
    b_psi = hp.b_psi[k]
    lo = hp.bounds_lo[k]
    hi = hp.bounds_hi[k]

    # Draw population-level parameters: shape (n_mc_draws,)
    mu = rng.beta(a_mu, b_mu, size=n_mc_draws)
    psi = rng.beta(a_psi, b_psi, size=n_mc_draws)

    # Conditional Beta parameters: shape (n_mc_draws,)
    # z_i | mu, psi ~ Beta(mu*(1-psi)/psi, (1-mu)*(1-psi)/psi)
    ratio = (1 - psi) / psi  # (n_mc_draws,)
    a_z = mu * ratio         # (n_mc_draws,)
    b_z = (1 - mu) * ratio   # (n_mc_draws,)

    # Draw subjects: shape (n_mc_draws, n_subjects)
    z = rng.beta(
        a_z[:, None] * np.ones(n_subjects),
        b_z[:, None] * np.ones(n_subjects),
    )

    # Scale to natural units
    return lo + z * (hi - lo)


def compute_sensitivity_grid(
    ce_direct_prior: Dict,
    alpha_values: List[float],
    kappa_values: List[float],
    quantity_idx: int = 0,
    n_mc_draws: int = 50000,
    n_subjects: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Compute hierarchical variance structure over (ρ, κ) grid using Beta-Beta hierarchy.

    For each grid point, MC sampling produces CEs of shape (n_mc_draws, n_subjects).
    The key outputs are:

    - between_sd: Robust SD of population-mean CEs (IQR/1.349 of draw-level
      means). Measures uncertainty about the population center (governed by ρ).
    - within_sd: Median of draw-level CE SDs. Measures typical individual
      heterogeneity (governed by 1−ρ).
    - within_iqr: IQR of draw-level CE SDs. Measures variability in the
      heterogeneity level across populations (governed by κ).

    Args:
        ce_direct_prior: CE-space direct prior (same format as main.py DIRECT_PRIOR_CE)
        alpha_values: List of ρ values to evaluate
        kappa_values: List of κ values to evaluate
        quantity_idx: Which quantity to analyze (0=CE1, 1=CE2, 2=CE3, 3=pchoice)
        n_mc_draws: MC draws per grid point
        n_subjects: Subjects per MC draw
        seed: Base random seed

    Returns:
        Dict with keys:
            'alpha_values', 'kappa_values': 1D arrays
            'between_sd': (n_alpha, n_kappa) — robust SD of population-mean CEs
            'within_sd': (n_alpha, n_kappa) — median within-population CE SD
            'within_iqr': (n_alpha, n_kappa) — IQR of within-population CE SDs
    """
    from prior_selection import ce_space_prior_selection, derive_hyperpriors_beta

    ps = ce_space_prior_selection(ce_direct_prior)

    # Extract bounds
    quantity_keys = ['ce1', 'ce2', 'ce3', 'pchoice']
    bounds_lo = np.array([ce_direct_prior[k]['lo'] for k in quantity_keys])
    bounds_hi = np.array([ce_direct_prior[k]['hi'] for k in quantity_keys])

    n_alpha = len(alpha_values)
    n_kappa = len(kappa_values)

    # Output arrays
    between_sd = np.zeros((n_alpha, n_kappa))
    within_sd = np.zeros((n_alpha, n_kappa))
    within_iqr = np.zeros((n_alpha, n_kappa))

    total = n_alpha * n_kappa
    count = 0

    for i, alpha in enumerate(alpha_values):
        for j, kappa in enumerate(kappa_values):
            count += 1
            print(f"  [{count}/{total}] ρ={alpha:.2f}, κ={kappa:.2f}")

            # Derive Beta-Beta hyperpriors
            hp = derive_hyperpriors_beta(ps, bounds_lo, bounds_hi,
                                         alpha=alpha, kappa=kappa)

            # Sample prior predictive CEs: shape (n_mc_draws, n_subjects)
            grid_seed = seed + i * n_kappa + j
            rng = np.random.default_rng(grid_seed)
            ces_2d = _sample_beta_beta_ce(
                hp, quantity_idx, n_mc_draws, n_subjects, rng)

            # Between-population SD: robust scale of draw-level means
            draw_means = np.mean(ces_2d, axis=1)  # (n_mc_draws,)
            q75_m, q25_m = np.percentile(draw_means, [75, 25])
            between_sd[i, j] = (q75_m - q25_m) / 1.349

            # Within-population: draw-level SDs
            draw_sds = np.std(ces_2d, axis=1)  # (n_mc_draws,)
            within_sd[i, j] = np.median(draw_sds)
            q75_s, q25_s = np.percentile(draw_sds, [75, 25])
            within_iqr[i, j] = q75_s - q25_s

    return {
        'alpha_values': np.array(alpha_values),
        'kappa_values': np.array(kappa_values),
        'between_sd': between_sd,
        'within_sd': within_sd,
        'within_iqr': within_iqr,
    }


def format_latex_table(results: Dict, prizes: np.ndarray,
                       probs: Optional[np.ndarray] = None) -> str:
    """
    Format sensitivity results as a LaTeX table with three panels.

    Panel A: Between-population SD (uncertainty about population center).
    Panel B: Within-population SD (typical individual heterogeneity).
    Panel C: Within-population SD IQR (variability of heterogeneity).

    Args:
        results: Output from compute_sensitivity_grid()
        prizes: Reference lottery prizes for context
        probs: Probabilities for each prize (for caption)

    Returns:
        LaTeX table string
    """
    alpha_vals = results['alpha_values']
    kappa_vals = results['kappa_values']
    n_kappa = len(kappa_vals)
    ncol = n_kappa + 1

    col_spec = 'r' + 'c' * n_kappa
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')

    if probs is None:
        probs = np.ones(len(prizes)) / len(prizes)
    ev = np.dot(prizes, probs)
    prize_str = '/'.join(f'\\${p:.0f}' for p in prizes)
    prob_str = '/'.join(f'{p:.1f}' for p in probs)
    lines.append(f'\\caption{{Hierarchical variance allocation across '
                 f'$(\\alpha, \\kappa)$ grid. '
                 f'Reference lottery: ${prize_str}$ with probabilities '
                 f'$({prob_str})$, $\\text{{EV}} = \\${ev:.0f}$.}}')
    lines.append(r'\label{tab:sensitivity}')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    header = r'$\alpha$'
    for k in kappa_vals:
        header += f' & $\\kappa = {k:.2f}$'
    header += r' \\'
    lines.append(header)

    for key, label in [
        ('between_sd', 'Between-population SD'),
        ('within_sd', 'Within-population SD'),
        ('within_iqr', 'Within-population SD IQR'),
    ]:
        lines.append(r'\midrule')
        lines.append(f'\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{label}}}}} \\\\')
        for i, alpha in enumerate(alpha_vals):
            row = f'{alpha:.2f}'
            for j in range(n_kappa):
                row += f' & ${results[key][i, j]:.2f}$'
            row += r' \\'
            lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def run_sensitivity(
    ce_direct_prior: Dict,
    output_dir: Path,
    lottery: Optional[Dict] = None,
    alpha_values: Optional[List[float]] = None,
    kappa_values: Optional[List[float]] = None,
    quantity_idx: int = 0,
    n_mc_draws: int = 50000,
    n_subjects: int = 50,
    seed: int = 42,
):
    """
    Run sensitivity analysis using Beta-Beta hierarchy and save results.

    Args:
        ce_direct_prior: CE-space direct prior specification
        output_dir: Path to save results
        lottery: Dict with 'prizes' and 'probs' keys (for table caption)
        alpha_values: ρ grid values (default: [0.05, 0.1, 0.2, 0.3, 0.5])
        kappa_values: κ grid values (default: [0.15, 0.25, 0.5, 1.0])
        quantity_idx: Which quantity to analyze (0=CE1)
        n_mc_draws: MC draws per grid point
        n_subjects: Subjects per MC draw
        seed: Random seed
    """
    if alpha_values is None:
        alpha_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    if kappa_values is None:
        kappa_values = [0.15, 0.25, 0.50, 1.00]

    # Extract lottery info for caption
    if lottery is not None:
        prizes = np.array(sorted(lottery['prizes']))
        probs = np.array([lottery['probs'][lottery['prizes'].index(p)]
                          if isinstance(lottery['prizes'], list)
                          else lottery['probs'][list(lottery['prizes']).index(p)]
                          for p in sorted(lottery['prizes'])])
    else:
        prizes = np.array([
            ce_direct_prior['ce1']['lo'],
            ce_direct_prior['ce1']['hi'],
        ])
        probs = None

    ev = np.dot(prizes, probs) if probs is not None else np.mean(prizes)
    prob_str = ', '.join(f'{p:.1f}' for p in probs) if probs is not None else 'equal'
    quantity_names = ['CE₁', 'CE₂', 'CE₃', 'P(choose L₁)']
    print(f"Sensitivity analysis over {len(alpha_values)} × {len(kappa_values)} grid")
    print(f"  Quantity: {quantity_names[quantity_idx]}")
    print(f"  ρ ∈ {alpha_values}")
    print(f"  κ ∈ {kappa_values}")
    print(f"  Reference lottery: prizes={prizes}, probs=({prob_str}), EV=${ev:.0f}")
    print(f"  MC draws: {n_mc_draws}, subjects/draw: {n_subjects}")
    print()

    results = compute_sensitivity_grid(
        ce_direct_prior=ce_direct_prior,
        alpha_values=alpha_values,
        kappa_values=kappa_values,
        quantity_idx=quantity_idx,
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
    latex_table = format_latex_table(results, prizes, probs=probs)
    tex_path = output_dir / 'sensitivity_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {tex_path}")

    # Print between-population SD
    print("\n" + "=" * 70)
    print("Between-Population SD (uncertainty about population center)")
    print("=" * 70)
    header = f"{'ρ':>6}"
    for k in kappa_values:
        header += f"  {'κ=' + f'{k:.2f}':>12}"
    print(header)
    print("-" * 70)
    for i, alpha in enumerate(alpha_values):
        row = f"{alpha:>6.2f}"
        for j in range(len(kappa_values)):
            row += f"  {results['between_sd'][i, j]:>10.2f}"
        print(row)
    print("=" * 70)

    # Print within-population SD
    print("\n" + "=" * 70)
    print("Within-Population SD (median individual heterogeneity)")
    print("=" * 70)
    header = f"{'ρ':>6}"
    for k in kappa_values:
        header += f"  {'κ=' + f'{k:.2f}':>12}"
    print(header)
    print("-" * 70)
    for i, alpha in enumerate(alpha_values):
        row = f"{alpha:>6.2f}"
        for j in range(len(kappa_values)):
            row += f"  {results['within_sd'][i, j]:>10.2f}"
        print(row)
    print("=" * 70)

    # Print within-population SD IQR
    print("\n" + "=" * 70)
    print("Within-Population SD IQR (variability of heterogeneity)")
    print("=" * 70)
    header = f"{'ρ':>6}"
    for k in kappa_values:
        header += f"  {'κ=' + f'{k:.2f}':>12}"
    print(header)
    print("-" * 70)
    for i, alpha in enumerate(alpha_values):
        row = f"{alpha:>6.2f}"
        for j in range(len(kappa_values)):
            row += f"  {results['within_iqr'][i, j]:>10.2f}"
        print(row)
    print("=" * 70)

    return results
