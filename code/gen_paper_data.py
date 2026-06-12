"""
Assemble pre-computed results into paper_data.npz for pythontex.

This module collects data summaries, prior configuration, MCMC stats,
and pre-computed analysis results into a single NPZ file that the paper
loads via pythontex. It does NOT run any heavy computation itself.

Heavy computations are handled by separate modules:
- family_comparison.py: NIG/BB/Gamma sensitivity, KS distances, CDF diffs
- sensitivity.py: (alpha, kappa) sensitivity analysis
- paper_plots.py: cross-model CE overlays and subject comparisons
"""

import numpy as np
from pathlib import Path

from clean_data import load_clean_data


def gen_paper_data(output_path=None,
                   ce_direct_prior=None, reference_lotteries=None,
                   reference_pair=None, prior_selection_ce=None,
                   ce_mean_frac=None, ce_p_risk_seeking=None):
    """
    Assemble summary statistics for the paper.

    Saves to NPZ file that can be loaded in LaTeX via pythontex.

    Args:
        output_path: Path to save NPZ file (default: data/output/paper_data.npz)
        ce_direct_prior: Dict with CE-space direct prior (ce1, ce2, ce3, pchoice)
        reference_lotteries: List of 3 reference lottery dicts (prizes, probs)
        reference_pair: Dict with lottery_A and lottery_B for choice probability
        prior_selection_ce: Dict with CE-space researcher choices (alpha, kappa)
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

    # Pre-compute display values for pythontex
    if prior_selection_ce is not None:
        alpha = prior_selection_ce['alpha']
        kappa = prior_selection_ce['kappa']
        save_dict['alpha_pct'] = alpha * 100
        save_dict['one_minus_alpha_pct'] = (1 - alpha) * 100
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
        if ce_direct_prior is not None and 'pchoice' in ce_direct_prior:
            pc = ce_direct_prior['pchoice']
            pc_sd = np.sqrt(pc['sigma2'])
            save_dict['pchoice_var'] = pc['sigma2']
            save_dict['pchoice_sd'] = pc_sd
            print(f"  P: mu={pc['mu']:.2f}, var={pc['sigma2']:.4f}, sd={pc_sd:.4f}")

    # MCMC summary stats (from preprocessed hyperparameters if available)
    hyper_ce_path = output_path.parent / 'rdu_bhm_ce' / 'preprocessed' / 'hyperparams_preprocessed.npz'
    if hyper_ce_path.exists():
        hd = np.load(hyper_ce_path)
        n_samples, n_chains = hd['ce1_logit_mu'].shape
        burn = n_samples // 2
        save_dict['n_chains'] = n_chains
        save_dict['n_samples_post_burnin'] = n_samples - burn
        save_dict['n_total_params'] = int(n_subjects) * 4 + 8
        print(f"\nMCMC Summary (from preprocessed):")
        print(f"  Chains: {n_chains}, samples post burn-in: {n_samples - burn}")
        print(f"  Total posterior samples: {(n_samples - burn) * n_chains:,}")

    # Load pre-computed family comparison results (from family_comparison.py)
    family_comp_path = output_path.parent / 'family_comparison.npz'
    if family_comp_path.exists():
        fc = dict(np.load(family_comp_path))
        for key, val in fc.items():
            save_dict[key] = val
        print(f"\nLoaded family comparison from: {family_comp_path}")
        print(f"  Keys: {len(fc)}")
    else:
        print(f"\nSkipping family comparison (run do_family_comparison first)")

    # Per-model average iteration time (from history file diagnostics)
    from model_registry import MODEL_REGISTRY
    for mk, cfg in MODEL_REGISTRY.items():
        hist_dir = output_path.parent / cfg['model_id'] / 'history' / 'full'
        times = []
        for f in sorted(hist_dir.glob('*.npz')):
            d = np.load(f, allow_pickle=True)
            try:
                times.append(float(d['diagnostics'].item()['avg_iter_time']))
            except (KeyError, AttributeError):
                pass
        if times:
            save_dict[f'{mk}_time_per_iter'] = float(np.mean(times))

    # Load marginal likelihoods / Bayes factors (from marginal_likelihood.py)
    ml_path = output_path.parent / 'bayes_factors_rb2.npz'
    if ml_path.exists():
        ml = dict(np.load(ml_path))
        gaps = []
        for m in ['crra', 'ge', 'power']:
            save_dict[f'ml_{m}_is'] = float(ml[f'{m}_log_ml_is'])
            save_dict[f'ml_{m}_is_se'] = float(ml[f'{m}_log_ml_is_se'])
            save_dict[f'ml_{m}_bridge'] = float(ml[f'{m}_log_ml_bridge'])
            save_dict[f'ml_{m}_bridge_se'] = float(ml[f'{m}_log_ml_bridge_se'])
            save_dict[f'ml_{m}_ess_med'] = float(ml[f'{m}_ess_subject_med'])
            gaps.append(abs(save_dict[f'ml_{m}_is']
                            - save_dict[f'ml_{m}_bridge']))
        save_dict['ml_route_gap_max'] = max(gaps)
        # Feasible-mass normalization (truncated-prior construction)
        if 'crra_log_Z' in ml:
            for m in ['crra', 'ge', 'power']:
                save_dict[f'ml_{m}_logZ'] = float(ml[f'{m}_log_Z'])
                save_dict[f'ml_{m}_logZ_se'] = float(ml[f'{m}_log_Z_se'])
                save_dict[f'ml_{m}_is_corr'] = \
                    float(ml[f'{m}_log_ml_is_corrected'])
                save_dict[f'ml_{m}_bridge_corr'] = \
                    float(ml[f'{m}_log_ml_bridge_corrected'])
            for pair in ['crra_vs_ge', 'crra_vs_power', 'ge_vs_power']:
                short = pair.replace('_vs', '')
                save_dict[f'bf_{short}_corr'] = \
                    float(ml[f'log_bf_rb2_{pair}_corrected'])
                save_dict[f'bf_{short}_corr_se'] = \
                    float(ml[f'log_bf_rb2_{pair}_corrected_se'])
        # Prior feasible mass at the posterior-mean psi (measurement run)
        pm_path = output_path.parent / 'prior_mass_measurement.npz'
        if pm_path.exists():
            pm = dict(np.load(pm_path))
            for m in ['crra', 'ge', 'power']:
                if f'{m}_log_C_bar' in pm:
                    save_dict[f'ml_{m}_logC_bar'] = \
                        float(pm[f'{m}_log_C_bar'])
        for pair in ['crra_vs_ge', 'crra_vs_power', 'ge_vs_power']:
            short = pair.replace('_vs', '')
            save_dict[f'bf_{short}'] = float(ml[f'log_bf_rb2_{pair}'])
            save_dict[f'bf_{short}_se'] = float(ml[f'log_bf_rb2_{pair}_se'])
        save_dict['ml_K'] = int(ml['K'])
        save_dict['ml_K_ge'] = int(ml.get('ge_K', ml['K']))
        save_dict['ml_R'] = int(ml['R'])
        print(f"\nLoaded marginal likelihoods from: {ml_path}")
        print(f"  log BF crra/ge = {save_dict['bf_crra_ge']:.2f}, "
              f"crra/power = {save_dict['bf_crra_power']:.2f}, "
              f"ge/power = {save_dict['bf_ge_power']:.2f}")
    else:
        print(f"\nSkipping marginal likelihoods (run do_bayes_factors_rb2 first)")

    # Marginal likelihood validation summary (from ml_validation.py)
    val_path = output_path.parent / 'ml_validation.npz'
    if val_path.exists():
        val = dict(np.load(val_path))
        save_dict['mlval_gt'] = float(val['gt_wide_log_ml'])
        save_dict['mlval_gt_se'] = float(val['gt_wide_se'])
        save_dict['mlval_abs_err_is'] = abs(float(val['rb_is_log_ml'])
                                            - float(val['gt_wide_log_ml']))
        save_dict['mlval_abs_err_bridge'] = abs(float(val['rb_bridge_log_ml'])
                                                - float(val['gt_wide_log_ml']))
        save_dict['mlval_n_subjects'] = int(val['n_subjects'])
        save_dict['mlval_n_choices'] = int(val['n_choices'])
        print(f"Loaded ML validation: |err| IS = "
              f"{save_dict['mlval_abs_err_is']:.4f}, bridge = "
              f"{save_dict['mlval_abs_err_bridge']:.4f}")

    # Load pre-computed MCMC diagnostics (from mcmc_diagnostics.py)
    diag_path = output_path.parent / 'mcmc_diagnostics.npz'
    if diag_path.exists():
        diag = dict(np.load(diag_path, allow_pickle=True))
        # Only load scalar values (skip arrays for paper_data.npz size)
        for key, val in diag.items():
            if val.ndim == 0 or (isinstance(val, np.ndarray) and val.size == 1):
                save_dict[key] = val
        print(f"\nLoaded MCMC diagnostics from: {diag_path}")
        for model in ['crra', 'ge', 'power']:
            med = diag.get(f'{model}_rhat_median', np.nan)
            mx = diag.get(f'{model}_rhat_max', np.nan)
            acc = diag.get(f'{model}_accept_mean', np.nan)
            # Hyperparameters occupy the last 8 slots of the state vector
            arr = diag.get(f'{model}_all_rhats')
            if arr is not None and np.asarray(arr).size >= 8:
                save_dict[f'{model}_rhat_hyper_max'] = \
                    float(np.asarray(arr)[-8:].max())
            print(f"  {model}: R-hat median={float(med):.4f}, max={float(mx):.4f}, accept={float(acc):.1%}")
    else:
        print(f"\nSkipping MCMC diagnostics (run do_mcmc_diagnostics first)")

    # Cross-model comparison (from preprocessed posteriors for all 3 models)
    ge_hyper_path = output_path.parent / 'rdu_ge_bhm_ce' / 'preprocessed' / 'hyperparams_preprocessed.npz'
    power_hyper_path = output_path.parent / 'rdu_power_bhm_ce' / 'preprocessed' / 'hyperparams_preprocessed.npz'

    model_hypers = {
        'ge': ge_hyper_path,
        'power': power_hyper_path,
    }
    available_models = {k: v for k, v in model_hypers.items() if v.exists()}

    if available_models and reference_lotteries is not None:
        from paper_plots import _draw_ce_from_beta_hierarchy

        print("\nCross-Model Comparison:")
        prior_params_file = str(output_path.parent / 'prior_params_ce.npz')
        crra_hyper_file = str(hyper_ce_path)

        n_ce_draws = 50000
        for k, lot in enumerate(reference_lotteries):
            ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))
            suffix = str(k + 1)

            crra_ce = _draw_ce_from_beta_hierarchy(
                crra_hyper_file, prior_params_file, k,
                n_draws=n_ce_draws, seed=42+k, use_posterior=True)
            save_dict[f'xmodel_crra_median_{suffix}'] = np.nanmedian(crra_ce)
            msg = f"  L{suffix}: CRRA=${np.nanmedian(crra_ce):.1f}"

            for model_name, hyper_path in available_models.items():
                model_ce = _draw_ce_from_beta_hierarchy(
                    str(hyper_path), prior_params_file, k,
                    n_draws=n_ce_draws, seed=42 + 80 + k,
                    use_posterior=True)
                save_dict[f'xmodel_{model_name}_median_{suffix}'] = np.nanmedian(model_ce)
                msg += f", {model_name.upper()}=${np.nanmedian(model_ce):.1f}"

            print(msg)
    else:
        print("\nSkipping cross-model comparison (model posteriors not found)")

    np.savez(output_path, **save_dict)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    gen_paper_data()
