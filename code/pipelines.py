"""
Pipeline functions for main.py dispatch.

This module provides wrapper functions that main.py calls to execute each
pipeline step. Each function handles printing status messages and calling
the underlying implementation modules.
"""

from pathlib import Path


def run_clean_data(data_dir):
    """Run data cleaning pipeline."""
    print("\n" + "=" * 60)
    print("Running: Data Cleaning")
    print("=" * 60)
    from clean_data import clean_and_save_data
    clean_and_save_data()
    print("Done.\n")


def run_gen_paper_data(data_dir, direct_prior=None, prior_selection=None):
    """Generate paper data summaries including direct prior specification."""
    print("\n" + "=" * 60)
    print("Running: Generate Paper Data Summaries")
    print("=" * 60)
    from gen_paper_data import gen_paper_data
    gen_paper_data(direct_prior=direct_prior, prior_selection=prior_selection)
    print("Done.\n")


def run_prior_selection(data_dir, rho=0.3, alpha_0=6.0, direct_prior=None):
    """Run prior selection and save hyperpriors."""
    print("\n" + "=" * 60)
    print("Running: Prior Selection")
    print("=" * 60)
    from prior_selection import run_prior_selection as _run_prior_selection
    output_path = Path(data_dir) / 'output' / 'prior_params.npz'
    _run_prior_selection(rho=rho, alpha_0=alpha_0, output_path=output_path,
                         direct_prior=direct_prior)
    print("Done.\n")


def run_sampling(sampling_opts, prior_params_path=None):
    """Run MCMC sampling.

    Args:
        sampling_opts: Sampling configuration dict
        prior_params_path: Path to prior_params.npz to load hyperpriors from.
                          If None, uses default hyperpriors.
    """
    print("\n" + "=" * 60)
    print("Running: MCMC Sampling")
    print("=" * 60)

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
        num_runs=sampling_opts['resume_runs'],
        num_collect=sampling_opts['num_collect'],
        config={
            'num_chains_a': num_chains,
            'num_chains_b': num_chains,
            'thin_iteration': sampling_opts['thin_iteration'],
            'chunk_size': sampling_opts.get('chunk_size', 100),
            'rng_seed': sampling_opts['seed'],
        },
    )
    print("Done.\n")


def run_gen_paper_quantities(history_dir, data_dir):
    """Extract posterior summaries for paper."""
    print("\n" + "=" * 60)
    print("Running: Generate Paper Quantities")
    print("=" * 60)
    from gen_paper_quantities import gen_paper_quantities
    output_path = Path(data_dir) / 'output' / 'paper_quantities.npz'
    gen_paper_quantities(history_dir, output_path)
    print("Done.\n")


def run_diagnostic_plots(history_dir, plots_dir, n_workers=1):
    """Generate diagnostic plots."""
    print("\n" + "=" * 60)
    print("Running: Diagnostic Plots")
    print("=" * 60)
    from plotting import diagnostic_plots
    diagnostic_plots(history_dir, plots_dir, n_workers=n_workers)
    print("Done.\n")


def run_paper_plots(quantities_path, plots_dir):
    """Generate paper plots."""
    print("\n" + "=" * 60)
    print("Running: Paper Plots")
    print("=" * 60)
    from plotting import paper_plots
    paper_plots(quantities_path, plots_dir)
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


def run_prior_plots(direct_prior, prior_selection, output_dir, seed=42):
    """Generate prior distribution plots."""
    print("\n" + "=" * 60)
    print("Running: Prior Distribution Plots")
    print("=" * 60)
    from plotting import plot_direct_prior, plot_unconditional_prior, plot_prior_comparison

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Plotting direct prior...")
    plot_direct_prior(direct_prior, output_dir / 'direct_prior.pdf', seed=seed)

    print("  Plotting unconditional prior (BHM)...")
    plot_unconditional_prior(direct_prior, prior_selection,
                              output_dir / 'unconditional_prior.pdf', seed=seed)

    print("  Plotting prior comparison...")
    plot_prior_comparison(direct_prior, prior_selection,
                           output_dir / 'prior_comparison.pdf', seed=seed)

    print("Done.\n")
