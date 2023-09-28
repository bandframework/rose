'''
Functions for computing metrics.
'''

import pickle

import numpy as np

from .reduced_basis_emulator import ReducedBasisEmulator
from .free_solutions import phase_shift_interp

def wave_function_metric(
    rbe: ReducedBasisEmulator,
    filename: str
):
    with open(filename, 'rb') as f:
        benchmark_data = pickle.load(f)
    
    thetas = np.array([bd.theta for bd in benchmark_data])
    wave_functions = np.array([bd.phi for bd in benchmark_data])
    emulated_wave_functions = np.array([rbe.emulate_wave_function(theta) for theta in thetas])
    abs_residuals = np.abs(emulated_wave_functions - wave_functions)
    norm = np.sqrt(np.sum(abs_residuals**2, axis=1))
    return np.quantile(norm, [0.5, 0.95])


def phase_shift_metric(
    rbe: ReducedBasisEmulator,
    filename: str
):
    with open(filename, 'rb') as f:
        benchmark_data = pickle.load(f)
    
    thetas = np.array([bd.theta for bd in benchmark_data])

    wave_functions = np.array([bd.phi for bd in benchmark_data])
    phase_shifts = np.array([phase_shift_interp(u, rbe.s_mesh, rbe.l, rbe.interaction.eta, rbe.s_0) for u in wave_functions])

    emulated_phase_shifts = np.array([rbe.emulate_phase_shift(theta) for theta in thetas])

    rel_diff = np.abs((emulated_phase_shifts - phase_shifts) / emulated_phase_shifts)
    med, upper = np.quantile(rel_diff, [0.5, 0.95])
    return [med, upper, np.max(np.abs(rel_diff))]


def run_metrics(
    rbe: ReducedBasisEmulator,
    filename: str,
    verbose: bool = False
):
    wave_function_results = wave_function_metric(rbe, filename)
    phase_shift_results = phase_shift_metric(rbe, filename)

    if verbose:
        print('Wave function residuals (root of sum of squares):\n50% and 95% quantiles')
        print(f'{wave_function_results[0]:.4e}  {wave_function_results[1]:.4e}')
        print('Phase shift residuals (relative difference):\n50% and 95% quantiles')
        print(f'{phase_shift_results[0]:.4e}  {phase_shift_results[1]:.4e}')
        print('Maximum phase shift residuals')
        print(f'{phase_shift_results[2]:.4e}')

    return wave_function_results, phase_shift_results