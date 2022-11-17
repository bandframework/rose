'''
Defines a ReducedBasisEmulator.
'''
import pickle

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags

from .interaction import Interaction
from .schroedinger import *
from .basis import StandardBasis, RelativeBasis
from .constants import HBARC
from .free_solutions import phi_free, phase_shift, phase_shift_interp

def finite_difference_second_derivative(
    s_mesh: npt.ArrayLike
):
    '''
    Computes a finite difference matrix that represents the second derivative
    (w.r.t. s or rho) operator in coordinate space.
    '''
    ds = s_mesh[1] - s_mesh[0]
    assert np.all(np.abs(s_mesh[1:] - s_mesh[:-1] - ds) < 1e-14), '''
Spacing must be consistent throughout the entire mesh.
    '''
    ns = s_mesh.size
    D2 = diags([-30, 16, 16, -1, -1], [0, 1, -1, 2, -2], shape=(ns, ns)).toarray() / (12*ds**2)
    D2[0, 0] = -2/ds**2
    D2[0, 1] = 1/ds**2
    D2[0, 2] = 0
    return D2


# How many points should be ignored at the beginning
# and end of the vectors (due to finite-difference
# inaccuracies)?
ni = 2

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
        basis_type: str = 'relative', # How is hat{phi} expanded?
        n_basis: int = 4, # How many basis vectors?
        use_svd: bool = True, # Use principal components as basis vectors?
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
            s_0 =  k * DEFAULT_R_0

        # Index of the point in the s mesh that is closest to s_0.
        self.i_0 = np.argmin(np.abs(self.s_mesh - s_0))
        # We want to choose a point at which the solution has already been
        # calculated so we can avoid interpolation.
        self.s_0 = self.s_mesh[self.i_0]
        
        self.i_0 = np.argmin(np.abs(self.s_mesh - self.s_0))

        if basis_type == 'standard':
            self.basis = StandardBasis(
                self.se.interaction,
                theta_train,
                self.s_mesh,
                n_basis,
                self.energy,
                self.l,
                use_svd
            )
        elif basis_type == 'relative':
            self.basis = RelativeBasis(
                self.se.interaction,
                theta_train,
                self.s_mesh,
                n_basis,
                self.energy,
                self.l,
                use_svd
            )

        # \tilde{U}_{bare} takes advantage of the linear dependence of \tilde{U}
        # on the parameters. The first column is multiplied by args[0]. The
        # second by args[1]. And so on. The "total" potential is the sum across
        # columns.
        self.utilde_bare = np.array([
            self.se.interaction.tilde(self.s_mesh, row, self.energy) for row in np.eye(theta_train.shape[1])
        ]).T

        d2_operator = finite_difference_second_derivative(self.s_mesh)
        phi_basis = self.basis.vectors
        self.d2 = -d2_operator @ phi_basis
        self.A_1 = phi_basis[ni:-ni].T @ self.d2[ni:-ni]
        self.A_2 = np.array([
            phi_basis[ni:-ni].T @ (row[:, np.newaxis] * phi_basis[ni:-ni]) for row in self.utilde_bare[ni:-ni, :].T
        ])
        self.A_3 = phi_basis[ni:-ni].T @ -phi_basis[ni:-ni]

        d2_phi_0 = d2_operator @ self.basis.phi_0
        self.b_1 = phi_basis[ni:-ni].T @ d2_phi_0[ni:-ni]
        self.b_2 = np.array([
            phi_basis[ni:-ni].T @ (-row * self.basis.phi_0[ni:-ni]) for row in self.utilde_bare[ni:-ni].T
        ])
        self.b_3 = phi_basis[ni:-ni].T @ self.basis.phi_0[ni:-ni]
    

    def emulate(self,
        theta: npt.ArrayLike
    ):
        A_utilde = np.sum([
            xi * Ai for (xi, Ai) in zip(theta, self.A_2)
        ], axis=0)
        A = self.A_1 + A_utilde + self.A_3 # I should go ahead and store A_1 + A_3

        b_utilde = np.sum([
            xi * Ai for (xi, Ai) in zip(theta, self.b_2)
        ], axis=0)
        b = self.b_1 + b_utilde + self.b_3 # I should store b_1 + b_3.

        return np.linalg.solve(A, b)


    def emulate_wave_function(self,
        theta: npt.ArrayLike
    ):
        x = self.emulate(theta)
        return self.basis.phi_hat(x)
    

    def wave_function_metric(self,
        filename: str
    ):
        with open(filename, 'rb') as f:
            benchmark_data = pickle.load(f)
        
        thetas = np.array([bd.theta for bd in benchmark_data])
        wave_functions = np.array([bd.phi for bd in benchmark_data])
        emulated_wave_functions = np.array([self.emulate_wave_function(theta) for theta in thetas])
        abs_residuals = np.abs(emulated_wave_functions - wave_functions)
        norm = np.sqrt(np.sum(abs_residuals**2, axis=1))
        return np.quantile(norm, [0.5, 0.95])


    def phase_shift_metric(self,
        filename: str
    ):
        with open(filename, 'rb') as f:
            benchmark_data = pickle.load(f)
        
        thetas = np.array([bd.theta for bd in benchmark_data])

        wave_functions = np.array([bd.phi for bd in benchmark_data])
        phase_shifts = np.array([phase_shift_interp(u, self.s_mesh, self.l, self.s_0) for u in wave_functions])

        emulated_wave_functions = np.array([self.emulate_wave_function(theta) for theta in thetas])
        emulated_phase_shifts = np.array([phase_shift_interp(u, self.s_mesh, self.l, self.s_0) for u in emulated_wave_functions])

        rel_diff = np.abs((emulated_phase_shifts - phase_shifts) / emulated_phase_shifts)
        med, upper = np.quantile(rel_diff, [0.5, 0.95])
        return [med, upper, np.max(np.abs(rel_diff))]


    def run_metrics(self,
        filename: str,
        verbose: bool = False
    ):
        wave_function_results = self.wave_function_metric(filename)
        phase_shift_results = self.phase_shift_metric(filename)

        if verbose:
            print('Wave function residuals (root of sum of squares):\n50% and 95% quantiles')
            print(f'{wave_function_results[0]:.4e}  {wave_function_results[1]:.4e}')
            print('Phase shift residuals (relative difference):\n50% and 95% quantiles')
            print(f'{phase_shift_results[0]:.4e}  {phase_shift_results[1]:.4e}')
            print('Maximum phase shift residuals')
            print(f'{phase_shift_results[2]:.4e}')

        return wave_function_results, phase_shift_results