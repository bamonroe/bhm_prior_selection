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
        prior_selection_ce: Dict with rho and kappa
    """
    print("\n" + "=" * 60)
    print("Running: Quantity-Space Prior Selection (Beta-Beta)")
    print("=" * 60)
    from prior_selection import run_prior_selection_ce as _run
    output_path = Path(data_dir) / 'output' / 'prior_params_ce.npz'
    _run(
        ce_direct_prior=ce_direct_prior,
        rho=prior_selection_ce['rho'],
        kappa=prior_selection_ce['kappa'],
        output_path=output_path,
    )
    print("Done.\n")


def run_sampling_ce(sampling_opts, ref_lotteries, ref_pair,
                    prior_params_path=None):
    """Run MCMC sampling with quantity-space Beta-Beta BHM.

    Args:
        sampling_opts: Sampling configuration dict
        ref_lotteries: List of 3 reference lottery dicts
        ref_pair: Dict with 'lottery_A' and 'lottery_B' for choice probability
        prior_params_path: Path to prior_params_ce.npz to load hyperpriors from
    """
    print("\n" + "=" * 60)
    print("Running: MCMC Sampling (Quantity-Space Beta-Beta Prior)")
    print("=" * 60)

    # Clear kernel cache
    from bamcmc.mcmc.compile import get_compiled_kernel_cache
    cache = get_compiled_kernel_cache()
    if cache:
        print(f"Clearing {len(cache)} cached kernel(s)...")
        cache.clear()

    # Load hyperpriors from file if specified
    if prior_params_path is not None:
        from posteriors.rdu_bhm_ce import load_hyperpriors_from_file
        load_hyperpriors_from_file(prior_params_path)

    # Clean existing files if requested
    if sampling_opts.get('clean', False):
        from bamcmc import clean_model_files
        output_dir = str(sampling_opts['output_dir'])
        model_name = sampling_opts['model']
        print(f"Cleaning existing files for {model_name}...")
        result = clean_model_files(output_dir, model_name, mode='all')
        print(f"  Deleted {len(result['deleted_checkpoints'])} checkpoints")
        print(f"  Deleted {len(result['deleted_histories'])} histories")

    from sampling_ce import run_sampling as _run_sampling
    num_chains = sampling_opts['num_chains']
    _run_sampling(
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
            'chunk_size': sampling_opts.get('chunk_size', 100),
            'rng_seed': sampling_opts['seed'],
            'n_temperatures': max(sampling_opts.get('n_temperatures', 1), 1),
            'beta_min': sampling_opts.get('beta_min', 0.1),
            'per_temp_proposals': sampling_opts.get('per_temp_proposals', True),
            'use_deo': sampling_opts.get('use_deo', True),
        },
    )
    print("Done.\n")


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
        rho_values=config.get('rho_values') if config else None,
        kappa_values=config.get('kappa_values') if config else None,
        n_mc_draws=config.get('n_mc_draws', 50000) if config else 50000,
        n_subjects=config.get('n_subjects', 50) if config else 50,
        seed=seed,
    )
    print("Done.\n")


def run_preprocess_hyperparams_ce(history_dir, output_dir, force=False):
    """Preprocess CE-model hyperparameter data from history files.

    Args:
        history_dir: Directory containing history_*.npz files
        output_dir: Directory to save preprocessed hyperparameter file
        force: If True, overwrite existing preprocessed data

    Returns:
        Path to the preprocessed hyperparameter file
    """
    print("\n" + "=" * 60)
    print("Running: Preprocess CE Hyperparameters")
    print("=" * 60)
    from plotting import preprocess_hyperparams_ce
    result = preprocess_hyperparams_ce(history_dir, output_dir, force=force)
    print("Done.\n")
    return result


def run_preprocess_subjects_ce(history_dir, output_dir, force=False):
    """Preprocess CE-model subject data from history files.

    Args:
        history_dir: Directory containing history_*.npz files
        output_dir: Directory to save per-subject parameter files
        force: If True, overwrite existing preprocessed data

    Returns:
        tuple: (output_dir, iterations, batch_boundaries)
    """
    print("\n" + "=" * 60)
    print("Running: Preprocess CE Subjects")
    print("=" * 60)
    from plotting import preprocess_subjects
    from posteriors.rdu_bhm_ce import NPARAMS as NP, N_HYPER as NH
    result = preprocess_subjects(history_dir, output_dir, force=force,
                                  nparams=NP, n_hyper=NH)
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
    """Run CE mapping injectivity analysis for appendix."""
    print("\n" + "=" * 60)
    print("Running: CE Mapping Injectivity Analysis")
    print("=" * 60)
    from injectivity_analysis import run_injectivity_analysis as _run
    _run(output_dir, config=config)
    print("Done.\n")


# =============================================================================
# NIG (PARAMETER-SPACE) MODEL PIPELINE FUNCTIONS
#
# Recovered from commit 4ac7847 to support the "diffuse" / uninformative
# comparison analysis.
# =============================================================================


def run_prior_selection(data_dir, rho=0.3, alpha_0=None, kappa=None, direct_prior=None):
    """Run NIG prior selection and save hyperpriors."""
    print("\n" + "=" * 60)
    print("Running: Prior Selection")
    print("=" * 60)
    from prior_selection import run_prior_selection as _run_prior_selection
    output_path = Path(data_dir) / 'output' / 'prior_params.npz'
    _run_prior_selection(rho=rho, alpha_0=alpha_0, kappa=kappa,
                         output_path=output_path, direct_prior=direct_prior)
    print("Done.\n")


def run_sampling(sampling_opts, prior_params_path=None):
    """Run MCMC sampling with NIG model.

    Args:
        sampling_opts: Sampling configuration dict
        prior_params_path: Path to prior_params.npz to load hyperpriors from.
                          If None, uses default hyperpriors.
    """
    print("\n" + "=" * 60)
    print("Running: MCMC Sampling")
    print("=" * 60)

    # Clear bamcmc's in-memory kernel cache
    from bamcmc.mcmc.compile import get_compiled_kernel_cache
    cache = get_compiled_kernel_cache()
    if cache:
        print(f"Clearing {len(cache)} cached kernel(s)...")
        cache.clear()

    # Load hyperpriors from file if specified
    if prior_params_path is not None:
        from posteriors.rdu_bhm import load_hyperpriors_from_file
        load_hyperpriors_from_file(prior_params_path)

    # Clean existing files if requested
    if sampling_opts.get('clean', False):
        from bamcmc import clean_model_files
        output_dir = str(sampling_opts['output_dir'])
        model_name = sampling_opts['model']
        print(f"Cleaning existing files for {model_name}...")
        result = clean_model_files(output_dir, model_name, mode='all')
        print(f"  Deleted {len(result['deleted_checkpoints'])} checkpoints")
        print(f"  Deleted {len(result['deleted_histories'])} histories")

    from sampling import run_sampling as _run_sampling
    num_chains = sampling_opts['num_chains']
    _run_sampling(
        output_dir=sampling_opts['output_dir'],
        reset_runs=sampling_opts.get('reset_runs', 1),
        resume_runs=sampling_opts.get('resume_runs', 0),
        num_collect=sampling_opts['num_collect'],
        config={
            'num_chains_a': num_chains,
            'num_chains_b': num_chains,
            'thin_iteration': sampling_opts['thin_iteration'],
            'chunk_size': sampling_opts.get('chunk_size', 100),
            'rng_seed': sampling_opts['seed'],
            'n_temperatures': sampling_opts.get('n_temperatures', 1),
            'beta_min': sampling_opts.get('beta_min', 0.1),
        },
    )
    print("Done.\n")


def run_gen_paper_quantities(history_dir, data_dir, config=None):
    """Extract posterior summaries for paper."""
    print("\n" + "=" * 60)
    print("Running: Generate Paper Quantities")
    print("=" * 60)
    if config is None:
        config = {}
    from gen_paper_quantities import gen_paper_quantities
    output_path = Path(data_dir) / 'output' / 'paper_quantities.npz'
    gen_paper_quantities(
        history_dir, output_path,
        burnin_fraction=config.get('burnin_fraction', 0.5),
        rhat_K=config.get('rhat_K', 20),
    )
    print("Done.\n")


def run_preprocess_hyperparams(history_dir, output_dir, force=False):
    """Preprocess NIG hyperparameter data from history files.

    Args:
        history_dir: Directory containing history_*.npz files
        output_dir: Directory to save preprocessed hyperparameter file
        force: If True, overwrite existing preprocessed data

    Returns:
        Path to the preprocessed hyperparameter file
    """
    print("\n" + "=" * 60)
    print("Running: Preprocess Hyperparameters")
    print("=" * 60)
    from plotting import preprocess_hyperparams
    result = preprocess_hyperparams(history_dir, output_dir, force=force)
    print("Done.\n")
    return result


def run_preprocess_subjects(history_dir, output_dir, force=False):
    """Preprocess NIG subject data from history files.

    Args:
        history_dir: Directory containing history_*.npz files
        output_dir: Directory to save per-subject parameter files
        force: If True, overwrite existing preprocessed data

    Returns:
        tuple: (output_dir, iterations, batch_boundaries)
    """
    print("\n" + "=" * 60)
    print("Running: Preprocess Subjects")
    print("=" * 60)
    from plotting import preprocess_subjects
    result = preprocess_subjects(history_dir, output_dir, force=force)
    print("Done.\n")
    return result


def run_hyperparameter_plots(history_dir, plots_dir, hyperprior_specs=None,
                             preprocessed_path=None, config=None):
    """Generate NIG hyperparameter diagnostic plots.

    Args:
        history_dir: Directory containing history_*.npz files (used if preprocessed_path is None)
        plots_dir: Directory to save plots
        hyperprior_specs: Optional dict with hyperprior specs for prior reference lines
        preprocessed_path: Path to preprocessed hyperparameter file. If provided,
                          loads from this file instead of history_dir.
        config: Optional dict with keys hyper_chain_subsample, hyper_params_per_page, hyper_row_height
    """
    print("\n" + "=" * 60)
    print("Running: Hyperparameter Plots")
    print("=" * 60)

    if preprocessed_path is not None:
        from plotting import plot_hyperparams_from_preprocessed
        plot_hyperparams_from_preprocessed(preprocessed_path, plots_dir,
                                           hyperprior_specs=hyperprior_specs,
                                           config=config)
    else:
        from plotting import plot_hyperparams_only
        plot_hyperparams_only(history_dir, plots_dir, hyperprior_specs=hyperprior_specs,
                              config=config)
    print("Done.\n")


def run_paper_plots(quantities_path, plots_dir,
                    info_hyper_path=None, uninf_hyper_path=None,
                    hyperprior_specs_info=None, hyperprior_specs_uninf=None,
                    hyper_comparison_config=None):
    """Generate paper plots."""
    print("\n" + "=" * 60)
    print("Running: Paper Plots")
    print("=" * 60)
    from plotting import paper_plots
    paper_plots(quantities_path, plots_dir,
                info_hyper_path=info_hyper_path,
                uninf_hyper_path=uninf_hyper_path,
                hyperprior_specs_info=hyperprior_specs_info,
                hyperprior_specs_uninf=hyperprior_specs_uninf,
                hyper_comparison_config=hyper_comparison_config)
    print("Done.\n")


def run_skew_normal_example(direct_prior, researcher_choices, output_dir,
                             n_mc_samples=50000, seed=42):
    """Run Skew-Normal hyperprior selection example for appendix."""
    print("\n" + "=" * 60)
    print("Running: Skew-Normal Hyperprior Example")
    print("=" * 60)
    from skew_normal_hyperprior import run_skew_normal_example as _run
    _run(direct_prior, researcher_choices, output_dir,
         n_mc_samples=n_mc_samples, seed=seed)
    print("Done.\n")


def run_prior_predictive_ce(direct_prior, prior_selection, data_dir,
                             output_dir, n_mc_draws=10000, seed=42,
                             config=None):
    """Run prior predictive CE analysis."""
    print("\n" + "=" * 60)
    print("Running: Prior Predictive CE Distribution")
    print("=" * 60)
    if config is None:
        config = {}
    from prior_predictive_ce import run_prior_predictive_ce as _run
    _run(direct_prior, prior_selection, data_dir, output_dir,
         n_mc_draws=config.get('n_mc_draws', n_mc_draws), seed=seed,
         config=config)
    print("Done.\n")


def run_posterior_predictive_ce(history_dir, data_dir, output_dir, label,
                                 prior_ce_path=None):
    """Run posterior predictive CE analysis."""
    print("\n" + "=" * 60)
    print(f"Running: Posterior Predictive CE Distribution ({label})")
    print("=" * 60)
    from prior_predictive_ce import run_posterior_predictive_ce as _run
    _run(history_dir, data_dir, output_dir, label,
         prior_ce_path=prior_ce_path)
    print("Done.\n")


def run_posterior_ce_comparison(data_dir, output_dir, prior_ce_path=None,
                                figsize=(6.3, 3.5), style=None):
    """Compare per-subject posterior predictive CE across prior specifications."""
    print("\n" + "=" * 60)
    print("Running: Posterior CE Comparison (Informative vs Uninformative)")
    print("=" * 60)
    from prior_predictive_ce import run_posterior_ce_comparison as _run
    _run(data_dir, output_dir, prior_ce_path=prior_ce_path,
         figsize=figsize, style=style)
    print("Done.\n")


def run_predictive_ce_overlay(hyper_preprocessed_info, hyper_preprocessed_uninf,
                               data_dir, direct_prior, prior_selection,
                               output_dir, seed=123, config=None):
    """Run population-level predictive CE and parameter overlay plots."""
    print("\n" + "=" * 60)
    print("Running: Predictive CE & Parameter Overlay (Prior + Both Posteriors)")
    print("=" * 60)
    from prior_predictive_ce import run_predictive_ce_overlay as _run
    _run(hyper_preprocessed_info, hyper_preprocessed_uninf, data_dir,
         direct_prior, prior_selection, output_dir,
         seed=seed, config=config or {})
    print("Done.\n")


def run_prior_plots(direct_prior, prior_selection, output_dir, seed=42,
                    config=None):
    """Generate prior distribution plots."""
    print("\n" + "=" * 60)
    print("Running: Prior Distribution Plots")
    print("=" * 60)
    if config is None:
        config = {}
    figsize = config.get('figsize', (10, 8))

    from plotting import plot_direct_prior, plot_unconditional_prior, plot_prior_comparison

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Plotting direct prior...")
    plot_direct_prior(direct_prior, output_dir / 'direct_prior.pdf', seed=seed,
                      figsize=figsize)

    print("  Plotting unconditional prior (BHM)...")
    plot_unconditional_prior(direct_prior, prior_selection,
                              output_dir / 'unconditional_prior.pdf', seed=seed,
                              figsize=figsize)

    print("  Plotting prior comparison...")
    plot_prior_comparison(direct_prior, prior_selection,
                           output_dir / 'prior_comparison.pdf', seed=seed,
                           figsize=figsize)

    print("Done.\n")
