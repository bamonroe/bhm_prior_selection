"""Generate per-subject CE density plots for all models.

Each subject gets a 3-panel PDF (one per reference lottery) showing
the posterior CE density under a given model. Plots are saved to
plots/{model_id}/subject_ces/subject_XXX_ces.pdf.

Usage: python code/gen_subject_ce_plots.py
"""

import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

REFERENCE_LOTTERIES = [
    {'prizes': [10, 50], 'probs': [0.75, 0.25], 'label': r'$L_1$'},
    {'prizes': [20, 40], 'probs': [0.5, 0.5], 'label': r'$L_2$'},
    {'prizes': [12, 44], 'probs': [0.75, 0.25], 'label': r'$L_3$'},
]

MODELS = [
    ('crra', 'CRRA+Prelec', 'rdu_bhm_ce', ['r', 'phi', 'eta', 'lambda'], '#1f77b4'),
    ('ge', 'CRRA+GE', 'rdu_ge_bhm_ce', ['r', 'gamma', 'delta', 'lambda'], '#2ca02c'),
    ('power', 'CRRA+Power', 'rdu_power_bhm_ce', ['r', 'delta', 'lambda'], '#d62728'),
]


def plot_one_subject(args):
    sid, model_key, name, model_id, param_names, color = args
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from model_registry import compute_ces_for_lottery

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{lmodern}',
        'font.family': 'serif', 'font.serif': ['Latin Modern Roman'],
        'mathtext.fontset': 'cm',
    })

    preproc = f'data/output/{model_id}/preprocessed/subjects'
    d = np.load(f'{preproc}/subject_{sid:03d}_params.npz')
    burn = d[param_names[0]].shape[0] // 2
    params = {k: d[k][burn:].ravel() for k in param_names}

    rng = np.random.default_rng(42 + sid)
    n_avail = len(params[param_names[0]])
    n_sub = min(3000, n_avail)
    idx = rng.choice(n_avail, n_sub, replace=False)
    sub_p = {k: v[idx] for k, v in params.items()}

    fig, axes = plt.subplots(3, 1, figsize=(5, 7))
    for lot_idx, lot in enumerate(REFERENCE_LOTTERIES):
        ax = axes[lot_idx]
        lo, hi = min(lot['prizes']), max(lot['prizes'])
        ev = sum(p * x for p, x in zip(lot['probs'], lot['prizes']))

        ces = compute_ces_for_lottery(model_key, sub_p, lot)
        ce_clip = ces[(ces >= lo) & (ces <= hi) & np.isfinite(ces)]

        if len(ce_clip) > 100:
            x_grid = np.linspace(lo, hi, 200)
            kde = gaussian_kde(ce_clip)
            ax.plot(x_grid, kde(x_grid), color=color, lw=1.5)

        ax.axvline(ev, color='black', ls=':', lw=0.8, alpha=0.5)
        ax.set_title(lot['label'], fontsize=9)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel('CE (\\$)', fontsize=9)
    fig.suptitle(f'Subject {sid} — {name}', fontsize=10)
    fig.tight_layout()
    out = f'plots/{model_id}/subject_ces/subject_{sid:03d}_ces.pdf'
    fig.savefig(out, bbox_inches='tight', backend='pdf')
    plt.close(fig)
    return sid


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    for model_key, name, model_id, param_names, color in MODELS:
        output_dir = Path(f'plots/{model_id}/subject_ces')
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f'{name}: generating CE density plots...', flush=True)

        meta = np.load(f'data/output/{model_id}/preprocessed/subjects/metadata.npz')
        n_subjects = int(meta['n_subjects'])

        args_list = [(sid, model_key, name, model_id, param_names, color)
                     for sid in range(n_subjects)]

        with ProcessPoolExecutor(max_workers=20) as pool:
            done = list(pool.map(plot_one_subject, args_list))

        print(f'  Saved {len(done)} PDFs to {output_dir}', flush=True)
