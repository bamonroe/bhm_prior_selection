"""Run just the paper's CRRA+Prelec posterior sampling stage in a test container."""
import os
from pathlib import Path
import sys

# Require the work GPU, rather than reporting CPU fallback as a passing test.
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax

sys.path.insert(0, '/source/code')
from main import REFERENCE_LOTTERIES, REFERENCE_PAIR, sampling_opts_ce
import clean_data
from pipelines import run_sampling_model

# main.py defaults target production runs; override memory and cache settings
# before the first JAX device operation without changing the paper's flags.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['BAMCMC_BENCHMARKDIR'] = '/tmp/posterior-benchmarks'
jax.config.update('jax_compilation_cache_dir', '/tmp/jax-cache')
print(f'Smoke test only; JAX devices: {jax.devices()}')
clean_data.CLEAN_DATA_PATH = Path('/dataset/dbar_clean.npz')
run_sampling_model(
    'crra',
    {**sampling_opts_ce, 'output_dir': Path('/output'), 'clean': False,
     'reset_runs': 1, 'resume_runs': 0, 'num_chains': 8, 'num_collect': 12,
     'thin_iteration': 1, 'chunk_size': 4},
    ref_lotteries=REFERENCE_LOTTERIES,
    ref_pair=REFERENCE_PAIR,
    prior_params_path=Path('/dataset/prior_params_ce.npz'),
)
