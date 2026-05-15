"""
MCMC diagnostics: per-subject R-hat, aggregate convergence statistics,
and acceptance rates for all estimated models.

Output: data/output/mcmc_diagnostics.npz with per-model stats that
feed into the paper's MCMC appendix via pythontex.
"""

import numpy as np
from pathlib import Path


def split_rhat(x):
    """Compute split-R-hat for a single parameter trace.

    Args:
        x: shape (n_iters, n_chains) — full trace including burn-in.
            The first half is discarded, then the second half is split
            again to form 2 * n_chains chains of length n_iters/4.

    Returns:
        R-hat (scalar)
    """
    n_iters, n_chains = x.shape
    half = n_iters // 2
    x_post = x[half:]
    n_post = x_post.shape[0]
    split = n_post // 2

    # Stack splits: 2*n_chains chains of length split
    chains = np.concatenate([x_post[:split], x_post[split:2*split]], axis=1)
    n, m = chains.shape

    chain_means = chains.mean(axis=0)
    grand_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(np.var(chains, axis=0, ddof=1))

    if W <= 0:
        return np.inf
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def compute_subject_rhats(preprocessed_dir, param_names):
    """Compute split-R-hat for every subject and parameter.

    Args:
        preprocessed_dir: Path to preprocessed subjects directory
        param_names: List of parameter names in subject files

    Returns:
        rhats: array of shape (n_subjects, n_params)
        subject_ids: array of subject indices
    """
    preprocessed_dir = Path(preprocessed_dir)
    meta = np.load(preprocessed_dir / 'metadata.npz')
    n_subjects = int(meta['n_subjects'])

    rhats = np.zeros((n_subjects, len(param_names)))

    for sid in range(n_subjects):
        d = np.load(preprocessed_dir / f'subject_{sid:03d}_params.npz')
        for p_idx, pname in enumerate(param_names):
            rhats[sid, p_idx] = split_rhat(d[pname])

    return rhats, np.arange(n_subjects)


def compute_hyper_rhats(hyper_preprocessed_path):
    """Compute split-R-hat for all hyperparameters.

    Args:
        hyper_preprocessed_path: Path to hyperparams_preprocessed.npz

    Returns:
        rhats: array of shape (n_hyper,)
        names: list of hyperparameter names
    """
    hp = np.load(hyper_preprocessed_path)

    hyper_names = []
    hyper_traces = []
    for qname in ['ce1', 'ce2', 'ce3', 'pc']:
        for suffix in ['logit_mu', 'logit_psi']:
            key = f'{qname}_{suffix}'
            if key in hp:
                hyper_names.append(key)
                hyper_traces.append(hp[key])

    rhats = np.array([split_rhat(trace) for trace in hyper_traces])
    return rhats, hyper_names


def compute_model_diagnostics(model_name, preprocessed_dir, hyper_path,
                              param_names):
    """Compute full diagnostics for one model.

    Returns dict with:
        - Per-subject R-hat array
        - Aggregate stats (median, max, percentiles)
        - Worst 20 parameters with labels
        - All R-hat values as a flat vector
    """
    print(f"\n  Computing diagnostics for {model_name}...")

    # Subject-level R-hats
    subjects_dir = Path(preprocessed_dir) / 'subjects'
    subject_rhats, subject_ids = compute_subject_rhats(subjects_dir, param_names)
    n_subjects, n_params = subject_rhats.shape
    print(f"    Subjects: {n_subjects}, params per subject: {n_params}")

    # Hyper-level R-hats
    hyper_rhats, hyper_names = compute_hyper_rhats(hyper_path)
    print(f"    Hyperparameters: {len(hyper_names)}")

    # Combine all R-hats into one vector with labels
    all_rhats = []
    all_labels = []

    for sid in range(n_subjects):
        for p_idx, pname in enumerate(param_names):
            all_rhats.append(subject_rhats[sid, p_idx])
            all_labels.append(f"Subject {sid} {pname}")

    for h_idx, hname in enumerate(hyper_names):
        all_rhats.append(hyper_rhats[h_idx])
        all_labels.append(f"Hyper {hname}")

    all_rhats = np.array(all_rhats)
    all_labels = np.array(all_labels)

    # Aggregate stats
    stats = {
        'rhat_median': np.median(all_rhats),
        'rhat_max': np.max(all_rhats),
        'rhat_min': np.min(all_rhats),
        'rhat_p5': np.percentile(all_rhats, 5),
        'rhat_p25': np.percentile(all_rhats, 25),
        'rhat_p75': np.percentile(all_rhats, 75),
        'rhat_p95': np.percentile(all_rhats, 95),
        'rhat_pct_above_1p1': 100 * np.mean(all_rhats > 1.1),
        'rhat_pct_above_1p2': 100 * np.mean(all_rhats > 1.2),
        'n_params_total': len(all_rhats),
    }

    print(f"    R-hat: median={stats['rhat_median']:.4f}, "
          f"max={stats['rhat_max']:.4f}, "
          f">{1.1}: {stats['rhat_pct_above_1p1']:.1f}%, "
          f">{1.2}: {stats['rhat_pct_above_1p2']:.1f}%")

    # Worst 20
    worst_idx = np.argsort(all_rhats)[-20:][::-1]
    worst_names = all_labels[worst_idx]
    worst_values = all_rhats[worst_idx]

    print(f"    Worst 20:")
    for name, val in zip(worst_names[:10], worst_values[:10]):
        print(f"      {name}: {val:.4f}")
    if len(worst_names) > 10:
        print(f"      ... ({len(worst_names) - 10} more)")

    stats['worst20_names'] = worst_names
    stats['worst20_values'] = worst_values
    stats['all_rhats'] = all_rhats

    # Per-subject max R-hat (useful for identifying problem subjects)
    subject_max_rhat = subject_rhats.max(axis=1)
    stats['subject_max_rhat'] = subject_max_rhat
    stats['n_subjects_above_1p2'] = int(np.sum(subject_max_rhat > 1.2))
    stats['n_subjects_above_2p0'] = int(np.sum(subject_max_rhat > 2.0))

    print(f"    Subjects with max R-hat > 1.2: {stats['n_subjects_above_1p2']}")
    print(f"    Subjects with max R-hat > 2.0: {stats['n_subjects_above_2p0']}")

    return stats


def extract_acceptance_rates(model_dir):
    """Extract MH acceptance rates from checkpoint acceptance counts.

    Reports both cumulative rates (across all runs) and last-run rates
    (from the final run only, reflecting current mixing quality).

    Returns dict with mean, median, min, max acceptance rates.
    """
    model_dir = Path(model_dir)
    checkpoint_dir = model_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        return {k: np.nan for k in ['accept_mean', 'accept_median',
                'accept_min', 'accept_max', 'accept_last_run_mean']}

    cp_files = sorted(checkpoint_dir.glob('checkpoint_*.npz'))
    if len(cp_files) < 1:
        return {k: np.nan for k in ['accept_mean', 'accept_median',
                'accept_min', 'accept_max', 'accept_last_run_mean']}

    cp_last = np.load(cp_files[-1], allow_pickle=True)
    if 'acceptance_counts' not in cp_last:
        return {k: np.nan for k in ['accept_mean', 'accept_median',
                'accept_min', 'accept_max', 'accept_last_run_mean']}

    counts_last = cp_last['acceptance_counts']
    iter_last = int(cp_last['iteration'])
    n_chains = int(cp_last['num_chains_a']) + int(cp_last['num_chains_b'])

    # Cumulative rates
    total_proposals = iter_last * n_chains
    cum_rates = counts_last / total_proposals if total_proposals > 0 else counts_last * 0

    result = {
        'accept_mean': float(np.mean(cum_rates)),
        'accept_median': float(np.median(cum_rates)),
        'accept_min': float(np.min(cum_rates)),
        'accept_max': float(np.max(cum_rates)),
    }

    # Last-run rates (difference between last two checkpoints)
    if len(cp_files) >= 2:
        cp_prev = np.load(cp_files[-2], allow_pickle=True)
        counts_prev = cp_prev['acceptance_counts']
        iter_prev = int(cp_prev['iteration'])
        delta_counts = counts_last - counts_prev
        delta_iters = iter_last - iter_prev
        delta_proposals = delta_iters * n_chains
        if delta_proposals > 0:
            last_rates = delta_counts / delta_proposals
            result['accept_last_run_mean'] = float(np.mean(last_rates))
            result['accept_last_run_median'] = float(np.median(last_rates))
        else:
            result['accept_last_run_mean'] = np.nan
            result['accept_last_run_median'] = np.nan
    else:
        result['accept_last_run_mean'] = result['accept_mean']
        result['accept_last_run_median'] = result['accept_median']

    return result


def run_all_diagnostics(output_path=None):
    """Run diagnostics for all three models and save to NPZ.

    Args:
        output_path: Path for output NPZ (default: data/output/mcmc_diagnostics.npz)
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'output' / 'mcmc_diagnostics.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = Path(__file__).parent.parent / 'data' / 'output'

    models = [
        ('crra', 'CRRA+Prelec',
         base / 'rdu_bhm_ce',
         ['r', 'phi', 'eta', 'lambda']),
        ('ep', 'EP+TK',
         base / 'rdu_ep_bhm_ce',
         ['alpha', 'r', 'gamma', 'lambda']),
        ('power', 'CRRA+Power',
         base / 'rdu_power_bhm_ce',
         ['r', 'delta', 'lambda']),
    ]

    print("=" * 60)
    print("MCMC Diagnostics")
    print("=" * 60)

    save_dict = {}

    for key, name, model_dir, param_names in models:
        preproc_dir = model_dir / 'preprocessed'
        hyper_path = preproc_dir / 'hyperparams_preprocessed.npz'

        if not (preproc_dir / 'subjects').exists():
            print(f"\n  Skipping {name}: preprocessed subjects not found")
            continue

        stats = compute_model_diagnostics(
            name, preproc_dir, hyper_path, param_names)

        # Save scalar stats
        for stat_name in ['rhat_median', 'rhat_max', 'rhat_min',
                          'rhat_p5', 'rhat_p25', 'rhat_p75', 'rhat_p95',
                          'rhat_pct_above_1p1', 'rhat_pct_above_1p2',
                          'n_params_total', 'n_subjects_above_1p2',
                          'n_subjects_above_2p0']:
            save_dict[f'{key}_{stat_name}'] = stats[stat_name]

        # Save worst 20
        save_dict[f'{key}_worst20_names'] = stats['worst20_names']
        save_dict[f'{key}_worst20_values'] = stats['worst20_values']

        # Save full R-hat vector for distribution analysis
        save_dict[f'{key}_all_rhats'] = stats['all_rhats']

        # Save per-subject max R-hat
        save_dict[f'{key}_subject_max_rhat'] = stats['subject_max_rhat']

        # Acceptance rates
        if model_dir.exists():
            accept = extract_acceptance_rates(model_dir)
            for ak, av in accept.items():
                save_dict[f'{key}_{ak}'] = av
            last_run = accept.get('accept_last_run_mean', np.nan)
            print(f"    Acceptance: cumulative={accept['accept_mean']:.1%}, "
                  f"last run={last_run:.1%}")

    np.savez(output_path, **save_dict)
    print(f"\nSaved diagnostics to: {output_path}")


if __name__ == '__main__':
    run_all_diagnostics()
