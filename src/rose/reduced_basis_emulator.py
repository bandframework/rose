'''
Defines a ReducedBasisEmulator.
'''
import pickle

import numpy as np

from .interaction import Interaction
from .schroedinger import *
from .basis import Basis
from .constants import HBARC
from .free_solutions import phi_free, phase_shift
import numpy.typing as npt

class ReducedBasisEmulator:
    '''
    A ReducedBasisEmulator (RBE) uses the specified interaction and theta_train
    to generate solutions to the Schrödinger equation at a specific energy
    (energy) and partial wave (l).

    Using the Galerkin projection method, a linear combination of those
    solutions (or a PCA basis of them) is found at some "arbitrary" point in
    parameter space, theta.
    '''
    def __init__(self,
        interaction: Interaction, # desired local interaction
        theta_train: npt.ArrayLike, # training points in parameter space
        energy: float, # center-of-mass energy (MeV)
        l: int, # angular momentum
        s_mesh: npt.ArrayLike = None, # s = kr; solutions are u(s)
        s_0: float = None, # phase shift is "extracted" at s_0
        **kwargs # passed to SchroedingerEquation.solve_se
    ):
        self.energy = energy
        k = np.sqrt(2*interaction.mu*energy/HBARC)
        self.l = l
        self.se = SchroedingerEquation(interaction)

        if s_mesh is None:
            self.s_mesh = np.linspace(k*DEFAULT_R_MIN, k*DEFAULT_R_MAX, DEFAULT_NUM_PTS)
        else:
            self.s_mesh = np.copy(s_mesh)

        if s_0 is None:
            self.s_0 =  k * DEFAULT_R_0

        self.phi_0 = phi_free(k*self.s_mesh, self.l)
        self.basis = Basis(
            np.array([
                self.se.phi(self.energy, theta, self.s_mesh, self.l, **kwargs) - self.phi_0 for theta in theta_train
            ]).T,
            self.s_mesh
        )
    

    def emulate(self,
        theta: npt.ArrayLike,
        n_basis: int = 4,
        ni: int = 2 # How many points should be ignored at the beginning
                    # and end of the vectors (due to finite-difference
                    # inaccuracies)?
    ):
        utilde = self.se.interaction.tilde(self.s_mesh, theta, self.energy)
        phi_basis = self.basis.vectors(use_svd=True, n_basis=n_basis)
        d2 = self.basis.d2_svd[:, :n_basis]

        A_right = -d2[ni:-ni] + utilde[ni:-ni, np.newaxis] * phi_basis[ni:-ni] - phi_basis[ni:-ni]
        A = phi_basis[ni:-ni].T @ A_right

        d2_phi_0 = self.basis.d2_operator @ self.phi_0
        b = -phi_basis[ni:-ni].T @ (-d2_phi_0[ni:-ni] + utilde[ni:-ni]*self.phi_0[ni:-ni] - self.phi_0[ni:-ni])

        # A += np.vstack([phi_basis[0, :] for _ in range(n_basis)])
        # b = self.s_mesh[0]*np.ones(n_basis)
        # return np.sum(x * phi_basis, axis=1)

        x = np.linalg.solve(A, b)
        return self.phi_0 + np.sum(x * phi_basis, axis=1)


    def emulate_no_svd(self,
        theta: npt.ArrayLike
    ):
        n = self.basis.phi_train.shape[1]
        utilde = self.se.interaction.tilde(self.s_mesh, theta, self.energy)[:, np.newaxis]
        phi_basis = self.basis.vectors(use_svd=False)
        d2 = np.copy(self.basis.d2_train)

        A_right = -d2 + utilde * phi_basis - phi_basis
        A = phi_basis.T @ A_right
        A += np.vstack([phi_basis[0, :] for _ in range(n)])
        b = self.s_mesh[0]*np.ones(n)
        x = np.linalg.solve(A, b)
        u = np.sum(x * phi_basis, axis=1)
        return u / np.max(np.abs(u)) # normalized to 1
    

    def wave_function_metric(self,
        filename: str,
        n_basis: int = 4
    ):
        with open(filename, 'rb') as f:
            benchmark_data = pickle.load(f)
        
        thetas = np.array([bd.theta for bd in benchmark_data])
        wave_functions = np.array([bd.phi for bd in benchmark_data])
        emulated_wave_functions = np.array([self.emulate(theta) for theta in thetas])
        abs_residuals = np.abs(emulated_wave_functions - wave_functions)
        norm = np.sqrt(np.sum(abs_residuals**2, axis=1))
        return np.quantile(norm, [0.5, 0.95])


    def phase_shift_metric(self,
        filename: str,
        n_basis: int = 4
    ):
        with open(filename, 'rb') as f:
            benchmark_data = pickle.load(f)
        
        thetas = np.array([bd.theta for bd in benchmark_data])

        wave_functions = np.array([bd.phi for bd in benchmark_data])
        phase_shifts = np.array([phase_shift(u, self.s_mesh, self.l, self.s_0) for u in wave_functions])

        emulated_wave_functions = np.array([self.emulate(theta) for theta in thetas])
        emulated_phase_shifts = np.array([phase_shift(u, self.s_mesh, self.l, self.s_0) for u in emulated_wave_functions])

        rel_diff = np.abs((emulated_phase_shifts - phase_shifts) / emulated_phase_shifts)
        return np.quantile(rel_diff, [0.5, 0.95])


    def run_metrics(self,
        filename: str,
        n_basis: int = 4,
        verbose: bool = False
    ):
        wave_function_results = self.wave_function_metric(filename, n_basis)
        phase_shift_results = self.phase_shift_metric(filename, n_basis)

        if verbose:
            print('Wave function residuals (root of sum of squares):\n50% and 95% quantiles')
            print(f'{wave_function_results[0]:.4e}  {wave_function_results[1]:.4e}')
            print('Phase shift residuals (relative difference):\n50% and 95% quantiles')
            print(f'{phase_shift_results[0]:.4e}  {phase_shift_results[1]:.4e}')

        return wave_function_results, phase_shift_results