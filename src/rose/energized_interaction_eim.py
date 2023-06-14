'''
Defines a class for "affinizing" Interactions using the Empirical Interpolation
Method (EIM).
'''

from typing import Callable
import numpy as np
from scipy.stats import qmc

from .interaction import Interaction, InteractionSpace, couplings
from .interaction_eim import InteractionEIM, max_vol
from .constants import HBARC, DEFAULT_RHO_MESH
from .spin_orbit import SpinOrbitTerm

class EnergizedInteractionEIM(Interaction):
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.array], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        ell: int,
        training_info: np.array,
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        is_complex: bool = False,
        spin_orbit_term: SpinOrbitTerm = None,
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        # super().__init__(coordinate_space_potential, n_theta, mu, None,
        #     training_info, Z_1=Z_1, Z_2=Z_2, is_complex=is_complex, n_basis=n_basis,
        #     explicit_training=explicit_training, n_train=n_train,
        #     rho_mesh=rho_mesh, match_points=match_points)
        super().__init__(coordinate_space_potential, n_theta, mu, None, ell,
            Z_1=Z_1, Z_2=Z_2, is_complex=is_complex, spin_orbit_term=spin_orbit_term)

        # Generate a basis used to approximate the potential.
        # Did the user specify the training points?
        if explicit_training:
            snapshots = np.array([self.tilde(rho_mesh, theta) for theta in training_info]).T
        else:
            # Generate training points using the user-provided bounds.
            sampler = qmc.LatinHypercube(d=len(training_info))
            sample = sampler.random(n_train)
            train = qmc.scale(sample, training_info[:, 0], training_info[:, 1])
            snapshots = np.array([self.tilde(rho_mesh, theta) for theta in train]).T
        
        U, S, _ = np.linalg.svd(snapshots, full_matrices=False)
        self.singular_values = np.copy(S)
        
        if match_points is None:
            if n_basis is None:
                n_basis = n_theta
            self.snapshots = np.copy(U[:, :n_basis])
            # random r points between 0 and 2π fm
            i_max = self.snapshots.shape[0] // 4
            di = i_max // (n_basis - 1)
            i_init = np.arange(0, i_max + 1, di)
            self.match_indices = max_vol(self.snapshots, i_init)
                # np.random.randint(0, self.snapshots.shape[0], size=self.snapshots.shape[1]))
            self.match_points = rho_mesh[self.match_indices]
            self.r_i = np.copy(self.match_points)
        else:
            n_basis = match_points.size
            self.snapshots = np.copy(U[:, :n_basis])
            self.match_points = np.copy(match_points)
            self.match_indices = np.array([np.argmin(np.abs(rho_mesh - ri)) for ri in self.match_points])
            self.r_i = rho_mesh[self.match_indices]

        self.Ainv = np.linalg.inv(self.snapshots[self.match_indices])


    def tilde(self,
        s: float,
        alpha: np.array
    ):
        '''
        theta[0] is the energy
        '''
        energy = alpha[0]
        k = np.sqrt(2*self.mu*energy/HBARC)
        return  1.0/energy * self.v_r(s/k, alpha[1:])


    def coefficients(self,
        alpha: np.array # interaction parameters
    ):
        '''
        alpha[0] is the energy
        '''
        k = np.sqrt(2*self.mu*alpha[0]/HBARC)
        u_true = self.tilde(self.r_i, alpha)
        return 1/k, self.Ainv @ u_true


    def eta(self,
        alpha: np.array
    ):
        return self.k_c / np.sqrt(2*self.mu*alpha[0]/HBARC)
    

    def tilde_emu(self,
        s: float,
        alpha: np.array
    ):
        '''
        tilde{U}(s, alpha, E)
        Does not include the Coulomb term.
        s = pr/hbar
        alpha are the parameters we are varying
        E = E_{c.m.}, [E] = MeV = [v_r]

        '''
        _, x = self.coefficients(alpha)
        emu = np.sum(x * self.snapshots, axis=1)
        return emu
    

    def basis_functions(self, rho_mesh: np.array):
        return np.copy(self.snapshots)
    

    def momentum(self, alpha: np.array):
        return np.sqrt(2*self.mu*alpha[0]/HBARC)


class EnergizedInteractionEIMSpace(InteractionSpace):
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.array], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        l_max: int,
        training_info: np.array,
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        is_complex: bool = False,
        spin_orbit_potential: Callable[[float, np.array, float], float] = None, #V_{SO}(r, theta, l•s)
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        self.interactions = []
        if spin_orbit_potential is None:
            for l in range(l_max+1):
                self.interactions.append(
                    [EnergizedInteractionEIM(coordinate_space_potential, n_theta, mu,
                        l, training_info, Z_1=Z_1, Z_2=Z_2,
                        is_complex=is_complex, n_basis=n_basis, explicit_training=explicit_training,
                        n_train=n_train, rho_mesh=rho_mesh, match_points=match_points)]
                )
        else:
            for l in range(l_max+1):
                self.interactions.append(
                    [EnergizedInteractionEIM(coordinate_space_potential, n_theta, mu,
                        l, training_info, Z_1=Z_1, Z_2=Z_2, is_complex=is_complex,
                        spin_orbit_term=SpinOrbitTerm(spin_orbit_potential, lds),
                        n_basis=n_basis, explicit_training=explicit_training,
                        n_train=n_train, rho_mesh=rho_mesh, match_points=match_points)
                        for lds in couplings(l)]
                )