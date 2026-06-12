"""
Pipeline functions for main.py dispatch.

This module provides wrapper functions that main.py calls to execute each
pipeline step. Each function handles printing status messages and calling
the underlying implementation modules.
"""

from pathlib import Path


def run_clean_data(data_dir, dataset='hn2016'):
    """Run data cleaning pipeline."""
    print("\n" + "=" * 60)
    print(f"Running: Data Cleaning (dataset={dataset})")
    print("=" * 60)
    from clean_data import clean_and_save_data
    clean_and_save_data(dataset=dataset)
    print("Done.\n")


def run_gen_paper_data(data_dir,
                       ce_direct_prior=None, reference_lotteries=None,
                       reference_pair=None, prior_selection_ce=None,
                       ce_mean_frac=None, ce_p_risk_seeking=None):
    """Generate paper data summaries."""
    print("\n" + "=" * 60)
    print("Running: Generate Paper Data Summaries")
    print("=" * 60)
    from gen_paper_data import gen_paper_data
    gen_paper_data(ce_direct_prior=ce_direct_prior,
                   reference_lotteries=reference_lotteries,
                   reference_pair=reference_pair,
                   prior_selection_ce=prior_selection_ce,
                   ce_mean_frac=ce_mean_frac,
                   ce_p_risk_seeking=ce_p_risk_seeking)
    print("Done.\n")


def run_prior_selection_ce(data_dir, ce_direct_prior, prior_selection_ce):
    """Run quantity-space prior selection (Beta-Beta) and save hyperpriors.

    Args:
        data_dir: Data directory root
        ce_direct_prior: Dict with per-quantity mu, sigma2, lo, hi
        prior_selection_ce: Dict with alpha and kappa
    """
    print("\n" + "=" * 60)
    print("Running: Quantity-Space Prior Selection (Beta-Beta)")
    print("=" * 60)
    from prior_selection import run_prior_selection_ce as _run
    output_path = Path(data_dir) / 'output' / 'prior_params_ce.npz'
    _run(
        ce_direct_prior=ce_direct_prior,
        alpha=prior_selection_ce['alpha'],
        kappa=prior_selection_ce['kappa'],
        output_path=output_path,
    )
    print("Done.\n")


def run_sampling_model(model_key, sampling_opts, ref_lotteries, ref_pair,
                       prior_params_path=None):
    """Run MCMC sampling for any registered model.

    Uses the model registry to resolve the posterior and sampling modules
    dynamically. Handles hyperprior loading, kernel cache clearing, and
    file cleanup.

    Args:
        model_key: Key into MODEL_REGISTRY ('crra', 'ep', 'power')
        sampling_opts: Sampling configuration dict
        ref_lotteries: List of 3 reference lottery dicts
        ref_pair: Dict with 'lottery_A' and 'lottery_B'
        prior_params_path: Path to prior_params_ce.npz
    """
    import importlib
    from model_registry import MODEL_REGISTRY

    cfg = MODEL_REGISTRY[model_key]
    print("\n" + "=" * 60)
    print(f"Running: MCMC Sampling ({cfg['name']})")
    print("=" * 60)

    from bamcmc.mcmc.compile import get_compiled_kernel_cache
    cache = get_compiled_kernel_cache()
    if cache:
        print(f"Clearing {len(cache)} cached kernel(s)...")
        cache.clear()

    if prior_params_path is not None:
        mod = importlib.import_module(cfg['posterior_module'])
        mod.load_hyperpriors_from_file(prior_params_path)

    if sampling_opts.get('clean', False):
        from bamcmc import clean_model_files
        output_dir = str(sampling_opts['output_dir'])
        model_name = sampling_opts['model']
        print(f"Cleaning existing files for {model_name}...")
        result = clean_model_files(output_dir, model_name, mode='all')
        print(f"  Deleted {len(result['deleted_checkpoints'])} checkpoints")
        print(f"  Deleted {len(result['deleted_histories'])} histories")

    samp_mod = importlib.import_module(cfg['sampling_module'])
    _run_sampling_dispatch(sampling_opts, ref_lotteries, ref_pair,
                           samp_mod.run_sampling)
    print("Done.\n")


def run_sampling_prior(model_key, sampling_opts, ref_lotteries, ref_pair,
                       prior_params_path=None, n_subjects=1):
    """Run prior-only MCMC sampling for any registered model with a prior-only variant.

    Uses a single subject by default (n_subjects=1) since prior-only sampling
    does not use the data likelihood, and one subject is sufficient to
    characterize the induced prior while being much faster.
    """
    import importlib
    from model_registry import MODEL_REGISTRY

    cfg = MODEL_REGISTRY[model_key]
    if not cfg.get('has_prior_only', False):
        print(f"Skipping prior-only sampling: {cfg['name']} has no prior-only variant")
        return

    print("\n" + "=" * 60)
    print(f"Running: Prior-Only Sampling ({cfg['name']}, {n_subjects} subject(s))")
    print("=" * 60)

    from bamcmc.mcmc.compile import get_compiled_kernel_cache
    cache = get_compiled_kernel_cache()
    if cache:
        cache.clear()

    if prior_params_path is not None:
        mod = importlib.import_module(cfg['prior_posterior_module'])
        mod.load_hyperpriors_from_file(prior_params_path)

    if sampling_opts.get('clean', False):
        from bamcmc import clean_model_files
        output_dir = str(sampling_opts['output_dir'])
        model_name = cfg['prior_model_id']
        print(f"Cleaning existing files for {model_name}...")
        result = clean_model_files(output_dir, model_name, mode='all')
        print(f"  Deleted {len(result['deleted_checkpoints'])} checkpoints")
        print(f"  Deleted {len(result['deleted_histories'])} histories")

    # Build data with reduced subject count for prior-only sampling
    from clean_data import load_clean_data, make_bamcmc_data_indexed
    from sampling_ce import pack_ref_lotteries, pack_ref_pair

    clean_data = load_clean_data()
    indexed_data = make_bamcmc_data_indexed(clean_data)

    # Reduce to n_subjects
    indexed_data['static'] = (n_subjects,) + indexed_data['static'][1:]
    indexed_data['int'] = tuple(arr[:n_subjects] for arr in indexed_data['int'])
    indexed_data['float'] = tuple(
        arr[:n_subjects] if arr.ndim > 0 and arr.shape[0] > n_subjects else arr
        for arr in indexed_data['float']
    )

    # Append hyperprior, ref lottery, ref pair
    from posteriors.rdu_bhm_ce import get_hyperprior_array
    hyperprior_arr = get_hyperprior_array()
    indexed_data['float'] = indexed_data['float'] + (hyperprior_arr,)
    ref_lot_arr = pack_ref_lotteries(ref_lotteries)
    indexed_data['float'] = indexed_data['float'] + (ref_lot_arr,)
    ref_pair_arr = pack_ref_pair(ref_pair)
    indexed_data['float'] = indexed_data['float'] + (ref_pair_arr,)

    from pathlib import Path
    from bamcmc import rmcmc

    output_dir = Path(sampling_opts['output_dir'])
    num_chains = sampling_opts.get('num_chains', 200)

    mcmc_config = {
        'posterior_id': cfg['prior_model_id'],
        'num_chains_a': num_chains,
        'num_chains_b': num_chains,
        'burn_iter': 0,
        'num_collect': sampling_opts.get('num_collect', 200),
        'thin_iteration': sampling_opts.get('thin_iteration', 10),
        'chunk_size': sampling_opts.get('chunk_size', 10),
        'rng_seed': sampling_opts.get('seed', 42),
        'use_double': sampling_opts.get('use_double', True),
        'n_temperatures': 1,
        'beta_min': 0.1,
        'per_temp_proposals': True,
        'use_deo': True,
        'run_schedule': [('reset', 1), ('resume', 4)],
    }

    print(f"  Subjects: {n_subjects}")
    print(f"  Chains: {num_chains}+{num_chains}")
    print(f"  Runs: 5 (1 reset + 4 resume)")

    summary = rmcmc(mcmc_config, indexed_data, output_dir=str(output_dir),
                    calculate_rhat=True, burn_in_fresh=True)
    print(f"  Completed {summary['total_runs_completed']} runs")
    print("Done.\n")


def _run_sampling_dispatch(sampling_opts, ref_lotteries, ref_pair, run_fn):
    """Shared sampling dispatch for any quantity-space model."""
    num_chains = sampling_opts['num_chains']
    run_fn(
        output_dir=sampling_opts['output_dir'],
        ref_lotteries=ref_lotteries,
        ref_pair=ref_pair,
        reset_runs=sampling_opts.get('reset_runs', 1),
        resume_runs=sampling_opts.get('resume_runs', 0),
        num_collect=sampling_opts['num_collect'],
        config={
            'num_chains_a': num_chains,
            'num_chains_b': num_chains,
            'thin_iteration': sampling_opts['thin_iteration'],
            'chunk_size': sampling_opts.get('chunk_size', 10),
            'rng_seed': sampling_opts['seed'],
            'use_double': sampling_opts.get('use_double', True),
            'n_temperatures': max(sampling_opts.get('n_temperatures', 1), 1),
            'beta_min': sampling_opts.get('beta_min', 0.1),
            'per_temp_proposals': sampling_opts.get('per_temp_proposals', True),
            'use_deo': sampling_opts.get('use_deo', True),
        },
    )


def run_sampling_stan(sampling_opts, ref_lotteries, ref_pair,
                      prior_params_path=None):
    """Run MCMC sampling via Stan/NUTS with quantity-space Beta-Beta prior."""
    print("\n" + "=" * 60)
    print("Running: Stan NUTS Sampling (Quantity-Space Beta-Beta Prior)")
    print("=" * 60)

    from sampling_stan import run_sampling_stan as _run
    output_dir = Path(sampling_opts['output_dir']) / sampling_opts['model']
    _run(
        output_dir=output_dir,
        ref_lotteries=ref_lotteries,
        ref_pair=ref_pair,
        prior_params_path=prior_params_path,
        config=sampling_opts,
    )
    print("Done.\n")


def run_sampling_prior(sampling_opts, ref_lotteries, ref_pair,
                       prior_params_path=None):
    """Run prior-only MCMC sampling (no data likelihood)."""
    print("\n" + "=" * 60)
    print("Running: Prior-Only Sampling (Quantity-Space Beta-Beta)")
    print("=" * 60)

    from bamcmc.mcmc.compile import get_compiled_kernel_cache
    cache = get_compiled_kernel_cache()
    if cache:
        cache.clear()

    if prior_params_path is not None:
        from posteriors.rdu_bhm_ce_prior import load_hyperpriors_from_file
        load_hyperpriors_from_file(prior_params_path)

    if sampling_opts.get('clean', False):
        from bamcmc import clean_model_files
        output_dir = str(sampling_opts['output_dir'])
        model_name = 'rdu_bhm_ce_prior'
        result = clean_model_files(output_dir, model_name, mode='all')
        print(f"  Deleted {len(result['deleted_checkpoints'])} checkpoints")
        print(f"  Deleted {len(result['deleted_histories'])} histories")

    from sampling_ce_prior import run_sampling_prior as _run
    _run(
        output_dir=sampling_opts['output_dir'],
        ref_lotteries=ref_lotteries,
        ref_pair=ref_pair,
        reset_runs=sampling_opts.get('reset_runs', 1),
        resume_runs=sampling_opts.get('resume_runs', 3),
        num_collect=sampling_opts['num_collect'],
        config={
            'num_chains_a': sampling_opts['num_chains'],
            'num_chains_b': sampling_opts['num_chains'],
            'thin_iteration': sampling_opts['thin_iteration'],
            'chunk_size': sampling_opts.get('chunk_size', 100),
            'rng_seed': sampling_opts['seed'],
            'use_double': sampling_opts.get('use_double', True),
        },
    )
    print("Done.\n")


def run_sensitivity(ce_direct_prior, output_dir, lottery=None,
                    seed=42, config=None):
    """Run sensitivity analysis over (ρ, κ) grid using Beta-Beta hierarchy."""
    print("\n" + "=" * 60)
    print("Running: Sensitivity Analysis (Beta-Beta)")
    print("=" * 60)
    from sensitivity import run_sensitivity as _run_sensitivity
    _run_sensitivity(
        ce_direct_prior=ce_direct_prior,
        output_dir=Path(output_dir),
        lottery=lottery,
        alpha_values=config.get('alpha_values') if config else None,
        kappa_values=config.get('kappa_values') if config else None,
        n_mc_draws=config.get('n_mc_draws', 50000) if config else 50000,
        n_subjects=config.get('n_subjects', 50) if config else 50,
        seed=seed,
    )
    print("Done.\n")


def run_preprocess_model(model_key, force=False):
    """Preprocess both hyperparameters and subjects for a registered model.

    Args:
        model_key: Key into MODEL_REGISTRY
        force: If True, overwrite existing preprocessed data
    """
    from model_registry import MODEL_REGISTRY, get_data_dir
    from plotting import preprocess_hyperparams_ce, preprocess_subjects

    cfg = MODEL_REGISTRY[model_key]
    data_dir = get_data_dir(model_key)
    history_dir = data_dir / 'history' / 'full'
    preproc_dir = data_dir / 'preprocessed'

    if not history_dir.exists():
        print(f"Skipping {cfg['name']}: {history_dir} not found")
        return

    print("\n" + "=" * 60)
    print(f"Running: Preprocess {cfg['name']}")
    print("=" * 60)

    # Hyperparameters
    preprocess_hyperparams_ce(history_dir, preproc_dir, force=force,
                              nparams=cfg['nparams'], n_hyper=cfg['n_hyper'])

    # Subject-level parameters
    preprocess_subjects(history_dir, preproc_dir / 'subjects', force=force,
                        nparams=cfg['nparams'], n_hyper=cfg['n_hyper'],
                        param_names=cfg['param_names'])
    print("Done.\n")


# Keep old names for backward compatibility during transition
def run_preprocess_hyperparams_ce(history_dir, output_dir, force=False,
                                  nparams=None, n_hyper=None):
    """Preprocess hyperparameter data. Prefer run_preprocess_model() for new code."""
    print("\n" + "=" * 60)
    print("Running: Preprocess CE Hyperparameters")
    print("=" * 60)
    from plotting import preprocess_hyperparams_ce
    result = preprocess_hyperparams_ce(history_dir, output_dir, force=force,
                                       nparams=nparams, n_hyper=n_hyper)
    print("Done.\n")
    return result


def run_preprocess_subjects_ce(history_dir, output_dir, force=False,
                               nparams=None, n_hyper=None, param_names=None):
    """Preprocess subject data. Prefer run_preprocess_model() for new code."""
    print("\n" + "=" * 60)
    print("Running: Preprocess CE Subjects")
    print("=" * 60)
    from plotting import preprocess_subjects
    if nparams is None or n_hyper is None:
        from posteriors.rdu_bhm_ce import NPARAMS as NP, N_HYPER as NH
        nparams = nparams or NP
        n_hyper = n_hyper or NH
    result = preprocess_subjects(history_dir, output_dir, force=force,
                                  nparams=nparams, n_hyper=n_hyper,
                                  param_names=param_names)
    print("Done.\n")
    return result


def run_hyperparameter_plots_ce(preprocessed_path, plots_dir,
                                 ce_direct_prior=None, config=None):
    """Generate CE-model hyperparameter diagnostic plots.

    Args:
        preprocessed_path: Path to preprocessed hyperparameter file
        plots_dir: Directory to save plots
        ce_direct_prior: Optional dict with per-quantity {lo, hi} bounds
        config: Optional dict with keys hyper_chain_subsample, hyper_params_per_page, hyper_row_height
    """
    print("\n" + "=" * 60)
    print("Running: CE Hyperparameter Plots")
    print("=" * 60)
    from plotting import plot_hyperparams_ce_from_preprocessed
    plot_hyperparams_ce_from_preprocessed(preprocessed_path, plots_dir,
                                           ce_direct_prior=ce_direct_prior,
                                           config=config)
    print("Done.\n")


def run_subject_plots(history_dir, plots_dir, n_workers=1, preprocessed_dir=None,
                      prizes=None, config=None):
    """Generate per-subject diagnostic plots.

    Args:
        history_dir: Directory containing history_*.npz files (used if preprocessed_dir is None)
        plots_dir: Directory to save plots
        n_workers: Number of parallel workers for plotting
        preprocessed_dir: Directory with preprocessed subject files. If provided,
                         skips preprocessing step.
        prizes: Array of 3 reference prizes [min, mid, max] for CE page (None to skip)
        config: Optional dict with subject plotting config (forwarded to plot fns)
    """
    print("\n" + "=" * 60)
    print("Running: Subject Plots")
    print("=" * 60)
    from plotting import preprocess_subjects, parallel_plot_subjects
    from pathlib import Path
    import numpy as np

    plots_dir = Path(plots_dir)
    subjects_dir = plots_dir / 'subjects'
    subjects_dir.mkdir(parents=True, exist_ok=True)

    if preprocessed_dir is not None:
        # Load metadata and plot from preprocessed data
        preprocessed_dir = Path(preprocessed_dir)
        meta_path = preprocessed_dir / 'metadata.npz'
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        meta = np.load(meta_path)
        iterations = meta['iterations']
        batch_boundaries = list(meta['batch_boundaries'])

        print(f"  Using preprocessed data from {preprocessed_dir}")
        parallel_plot_subjects(preprocessed_dir, subjects_dir, iterations,
                              batch_boundaries, n_workers=n_workers,
                              prizes=prizes, config=config)
    else:
        # Preprocess then plot
        history_dir = Path(history_dir)
        params_dir = history_dir.parent / 'subject_params'
        print("  Stage 1: Preprocessing subject data...")
        params_dir, iterations, batch_boundaries = preprocess_subjects(
            history_dir, params_dir, force=False
        )
        print("  Stage 2: Parallel plotting...")
        parallel_plot_subjects(
            params_dir, subjects_dir, iterations, batch_boundaries,
            n_workers=n_workers, prizes=prizes, config=config
        )
    print("Done.\n")


def run_injectivity_analysis(output_dir, config=None):
    """Run CE mapping injectivity analysis for appendix (CRRA+Prelec)."""
    print("\n" + "=" * 60)
    print("Running: CE Mapping Injectivity Analysis (CRRA+Prelec)")
    print("=" * 60)
    from injectivity_analysis import run_injectivity_analysis as _run
    _run(output_dir, config=config)
    print("Done.\n")


def run_injectivity_power(output_dir, reference_lotteries, config=None):
    """Run CRRA+Power injectivity analysis (Gram determinant, k=4 > p=3)."""
    print("\n" + "=" * 60)
    print("Running: CRRA+Power Injectivity Analysis")
    print("=" * 60)
    from injectivity_power import run_injectivity_power as _run
    config = config or {}
    _run(
        output_dir=output_dir,
        reference_lotteries=reference_lotteries,
        domain=config.get('domain'),
        n_grid=config.get('n_grid', 40),
        x_min=config.get('x_min', 10.0),
        x_max=config.get('x_max', 50.0),
    )
    print("Done.\n")


def run_injectivity_ep_eut(output_dir, reference_lotteries, config=None):
    """Run EP+EUT injectivity analysis (Gram determinant, k=4 > p=3).

    Parameters
    ----------
    output_dir : Path
        Output directory for report and NPZ files.
    reference_lotteries : list of dict
        Current lottery design from main.py.
    config : dict, optional
        Keys: 'domain', 'n_grid', 'x_min', 'x_max'.
    """
    print("\n" + "=" * 60)
    print("Running: EP+EUT Injectivity Analysis")
    print("=" * 60)
    from injectivity_ep_eut import run_injectivity_ep_eut as _run
    config = config or {}
    _run(
        output_dir=output_dir,
        reference_lotteries=reference_lotteries,
        domain=config.get('domain'),
        n_grid=config.get('n_grid', 40),
        x_min=config.get('x_min', 10.0),
        x_max=config.get('x_max', 50.0),
    )
    print("Done.\n")


# =============================================================================
# Legacy functions kept for Appendix D (α,κ) sensitivity analysis
# =============================================================================


def run_prior_predictive_params_plot(prior_subjects_dir, output_path):
    """Generate prior predictive parameter-space plot.

    Args:
        prior_subjects_dir: Path to preprocessed prior-only subjects directory
        output_path: Path to save the output PDF
    """
    print("\n" + "=" * 60)
    print("Running: Prior Predictive Parameters Plot")
    print("=" * 60)
    from paper_plots import generate_prior_predictive_params_plot
    generate_prior_predictive_params_plot(prior_subjects_dir, output_path)
    print("Done.\n")


# =============================================================================
# (alpha, kappa) Sensitivity Analysis
# =============================================================================

def _rk_config_label(alpha, kappa):
    """Generate directory label for a (alpha, kappa) configuration."""
    return f"alpha{alpha:.1f}_kappa{kappa:.2f}".replace('.', 'p')


def run_rk_sensitivity_config(alpha, kappa, direct_prior_ce, reference_lotteries,
                               reference_pair, sampling_opts, seeds,
                               data_dir, plots_dir):
    """Run full pipeline for a single (alpha, kappa) sensitivity configuration.

    Executes: prior selection -> sampling -> preprocessing -> paper plots.

    Returns:
        dict with keys: alpha, kappa, label, n_well, n_moderate, n_dominated
    """
    import numpy as np
    label = _rk_config_label(alpha, kappa)

    print("\n" + "=" * 70)
    print(f"  Configuration: alpha={alpha}, kappa={kappa} ({label})")
    print("=" * 70)

    # --- Prior selection ---
    print("\n  Prior selection...")
    from prior_selection import (ce_space_prior_selection, derive_hyperpriors_beta,
                                  verify_hyperpriors_beta, print_hyperprior_summary_beta)
    ps = ce_space_prior_selection(direct_prior_ce)
    quantity_keys = ['ce1', 'ce2', 'ce3', 'pchoice']
    bounds_lo = np.array([direct_prior_ce[k]['lo'] for k in quantity_keys])
    bounds_hi = np.array([direct_prior_ce[k]['hi'] for k in quantity_keys])

    hp = derive_hyperpriors_beta(ps, bounds_lo, bounds_hi, alpha=alpha, kappa=kappa)
    print_hyperprior_summary_beta(hp)
    verify_hyperpriors_beta(hp)

    # Save prior params
    prior_params_path = data_dir / 'output' / f'prior_params_ce_{label}.npz'
    save_dict = {'distribution': 'beta', 'alpha': alpha, 'kappa': kappa}
    for k, qname in enumerate(quantity_keys):
        save_dict[f'{qname}_a_mu'] = hp.a_mu[k]
        save_dict[f'{qname}_b_mu'] = hp.b_mu[k]
        save_dict[f'{qname}_a_psi'] = hp.a_psi[k]
        save_dict[f'{qname}_b_psi'] = hp.b_psi[k]
        save_dict[f'{qname}_bound_lo'] = bounds_lo[k]
        save_dict[f'{qname}_bound_hi'] = bounds_hi[k]
    np.savez(prior_params_path, **save_dict)
    print(f"  Saved: {prior_params_path}")

    # --- Sampling ---
    print("\n  MCMC sampling...")
    config_output_dir = data_dir / 'output' / label
    opts = {**sampling_opts, 'output_dir': config_output_dir}
    run_sampling_ce(
        opts,
        ref_lotteries=reference_lotteries,
        ref_pair=reference_pair,
        prior_params_path=prior_params_path,
    )

    # --- Preprocessing ---
    print("\n  Preprocessing...")
    model_dir = config_output_dir / 'rdu_bhm_ce'
    history_dir = model_dir / 'history' / 'full'
    preprocess_dir = model_dir / 'preprocessed'

    run_preprocess_hyperparams_ce(
        history_dir=str(history_dir),
        output_dir=str(preprocess_dir),
        force=True,
    )
    run_preprocess_subjects_ce(
        history_dir=str(history_dir),
        output_dir=str(preprocess_dir / 'subjects'),
        force=True,
    )

    # --- Paper plots ---
    print("\n  Paper plots...")
    paper_plots_dir = plots_dir / 'paper' / label
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    info_subjects = preprocess_dir / 'subjects'
    uninf_subjects = (data_dir / 'output' / 'rdu_bhm_uninf'
                      / 'preprocessed' / 'subjects')
    hyper_info = str(preprocess_dir / 'hyperparams_preprocessed.npz')
    hyper_uninf = str(data_dir / 'output' / 'rdu_bhm_uninf'
                      / 'preprocessed' / 'hyperparams_preprocessed.npz')

    from paper_plots import generate_paper_plots
    generate_paper_plots(
        info_subjects_dir=str(info_subjects),
        uninf_subjects_dir=str(uninf_subjects),
        info_hyper_path=hyper_info,
        uninf_hyper_path=hyper_uninf,
        prior_params_path=str(prior_params_path),
        output_dir=str(paper_plots_dir),
        data_output_dir=str(paper_plots_dir),
        seed=seeds['sampling'],
    )

    # --- Extract classification ---
    print("\n  Extracting subject classification...")
    comp = np.load(paper_plots_dir / 'posterior_ce_comparison.npz')
    median_diffs = comp['median_diffs']
    n_well = int(np.sum(median_diffs < 2))
    n_moderate = int(np.sum((median_diffs >= 2) & (median_diffs < 10)))
    n_dominated = int(np.sum(median_diffs >= 10))

    print(f"  Results: {n_well} well-identified, {n_moderate} moderate, "
          f"{n_dominated} prior-dominated")

    return {
        'alpha': alpha, 'kappa': kappa, 'label': label,
        'n_well': n_well, 'n_moderate': n_moderate,
        'n_dominated': n_dominated,
    }


def run_rk_sensitivity(sensitivity_configs, direct_prior_ce,
                        reference_lotteries, reference_pair,
                        sampling_opts, seeds, data_dir, plots_dir):
    """Run (alpha, kappa) sensitivity analysis across multiple configurations.

    For each configuration, runs the full pipeline (prior selection, sampling,
    preprocessing, paper plots) and then saves an aggregate summary NPZ.

    Args:
        sensitivity_configs: List of (alpha, kappa) tuples
        direct_prior_ce: CE-space direct prior dict
        reference_lotteries: List of reference lottery dicts
        reference_pair: Reference pair dict
        sampling_opts: Sampling configuration dict
        seeds: Dict with 'sampling' key
        data_dir: Project data directory
        plots_dir: Project plots directory
    """
    import numpy as np

    print("\n" + "=" * 70)
    print("Running: (alpha, kappa) Sensitivity Analysis")
    print("=" * 70)

    results = []
    for alpha, kappa in sensitivity_configs:
        result = run_rk_sensitivity_config(
            alpha, kappa, direct_prior_ce, reference_lotteries, reference_pair,
            sampling_opts, seeds, data_dir, plots_dir,
        )
        results.append(result)

    # Print summary table (include baseline)
    print("\n" + "=" * 70)
    print("  (alpha, kappa) Sensitivity Summary")
    print("=" * 70)
    print(f"  {'alpha':>5} {'kappa':>6} | {'Well-ID':>8} {'Moderate':>9} {'Prior-Dom':>10}")
    print("  " + "-" * 48)

    baseline_comp = data_dir / 'output' / 'posterior_ce_comparison.npz'
    if baseline_comp.exists():
        bc = np.load(baseline_comp)
        md = bc['median_diffs']
        n_w = int(np.sum(md < 2))
        n_m = int(np.sum((md >= 2) & (md < 10)))
        n_d = int(np.sum(md >= 10))
        print(f"  {0.2:5.1f} {0.25:6.2f} | {n_w:8d} {n_m:9d} {n_d:10d}  (baseline)")
    else:
        n_w, n_m, n_d = 0, 0, 0

    for r in results:
        print(f"  {r['alpha']:5.1f} {r['kappa']:6.2f} | "
              f"{r['n_well']:8d} {r['n_moderate']:9d} {r['n_dominated']:10d}")

    # Save summary NPZ
    summary_path = data_dir / 'output' / 'rk_sensitivity_summary.npz'
    n_subj = len(np.load(baseline_comp)['median_diffs']) if baseline_comp.exists() else 111

    all_n_dom = [n_d] + [r['n_dominated'] for r in results]
    all_n_mod = [n_m] + [r['n_moderate'] for r in results]
    all_pct_dom = [round(100 * d / n_subj) for d in all_n_dom]

    pct_dom = {
        'lo_lo': all_pct_dom[0], 'lo_hi': all_pct_dom[1],
        'hi_lo': all_pct_dom[2], 'hi_hi': all_pct_dom[3],
    }

    np.savez(summary_path,
             alpha=np.array([r['alpha'] for r in results]),
             kappa=np.array([r['kappa'] for r in results]),
             n_well=np.array([r['n_well'] for r in results]),
             n_moderate=np.array([r['n_moderate'] for r in results]),
             n_dominated=np.array([r['n_dominated'] for r in results]),
             pct_dom_lo_lo=pct_dom['lo_lo'],
             pct_dom_lo_hi=pct_dom['lo_hi'],
             pct_dom_hi_lo=pct_dom['hi_lo'],
             pct_dom_hi_hi=pct_dom['hi_hi'],
             pct_dom_min=min(all_pct_dom),
             pct_dom_max=max(all_pct_dom),
             mod_min=min(all_n_mod),
             mod_max=max(all_n_mod))
    print(f"\n  Summary saved to: {summary_path}")
    print("Done.\n")
