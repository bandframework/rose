'''Interactions that leverage the Empirical Interpolation Method (EIM) to allow
the emulation of parameters in which the coordinate-space potential is not
affine.
'''

from typing import Callable
import numpy as np
from scipy.stats import qmc

from .interaction import Interaction, InteractionSpace, couplings
from .constants import HBARC, DEFAULT_RHO_MESH
from .spin_orbit import SpinOrbitTerm

def max_vol(basis, indxGuess):
    r'''basis looks like a long matrix, the columns are the "pillars" V_i(x):
        [   V_1(x)
            V_2(x)
            .
            .
            .
        ]
        indxGuess is a first guess of where we should "measure", or ask the questions

    '''
    nbases = basis.shape[1]
    interpBasis = np.copy(basis)

    for ij in range(len(indxGuess)):
        interpBasis[[ij,indxGuess[ij]],:] = interpBasis[[indxGuess[ij],ij],:]
    indexing = np.array(range(len(interpBasis)))

    for ij in range(len(indxGuess)):
        indexing[[ ij,indxGuess[ij] ]] = indexing[[ indxGuess[ij],ij ]]
    
    for iIn in range(1, 100):
        B = np.dot(interpBasis, np.linalg.inv(interpBasis[:nbases]))
        b = np.max(B)
        if b > 1:
            
            p1, p2 = np.where(B == b)[0][0], np.where(B == b)[1][0]
            interpBasis[[p1,p2],:] = interpBasis[[p2,p1],:]
            indexing[[p1,p2]] = indexing[[p2,p1]]
        else:
            break
        #this thing returns the indexes of where we should measure
    return np.sort(indexing[:nbases])


class InteractionEIM(Interaction):
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.array], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        ell: int,
        training_info: np.array,
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        R_C: float = 0.0, # Coulomb "cutoff"
        is_complex: bool = False,
        spin_orbit_term: SpinOrbitTerm = None,
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        r'''
        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r,
                theta) where theta are the interaction parameters
            n_theta (int): number of interaction parameters
            mu (float): reduced mass (MeV); converted to 1/fm
            energy (float): center-of-mass scattering energy
            ell (int): angular momentum
            training_info (ndarray): Either (1) parameters bounds or (2)
                explicit training points

                If (1):
                    This is a 2-column matrix. The first column are the lower
                    bounds. The second are the upper bounds. Each row maps to a
                    single parameter.

                If (2):
                    This is an MxN matrix. N is the number of parameters. M is
                    the number of samples.
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex (e.g. optical
                potentials)?
            spin_orbit_term (SpinOrbitTerm): spin-orbit part of the interaction
            n_basis (int): number of basis states, or "pillars" in $\hat{U}$ approximation
            explicit_training (bool): Is training_info (1) or (2)? (1) is
                default
            n_train (int): How many snapshots to generate? Ignored if
                explicit_training is True.
            rho_mesh (ndarray): coordinate-space points at which the interaction
                is generated (used for training)
            match_points (ndarray): $\rho$ points where agreement with the true
                potential is enforced

        Attributes:
            singular_values (ndarray): `S` in `U, S, Vt = numpy.linalg.svd(...)`
            snapshots (ndarray): pillars, columns of `U`
            match_indices (ndarray): indices of points in $\rho$ mesh that are
                matched to the true potential
            match_points (ndarray): points in $\rho$ mesh that are matched to
                the true potential
            r_i (ndarray): copy of `match_points` (???)
            Ainv (ndarray): inverse of A matrix (Ax = b)
        '''

        super().__init__(coordinate_space_potential, n_theta, mu, energy, ell,
            Z_1=Z_1, Z_2=Z_2, R_C=R_C, is_complex=is_complex, spin_orbit_term=spin_orbit_term)

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
            i_max = self.snapshots.shape[0] // 2
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


    def coefficients(self,
        alpha: np.array
    ):
        r'''Computes the EIM expansion coefficients.

        Parameters:
            alpha (ndarray): interaction parameters
        
        Returns:
            coefficients (ndarray): EIM expansion coefficients

        '''
        u_true = self.tilde(self.r_i, alpha)
        return 1/self.k, self.Ainv @ u_true


    def tilde_emu(self,
        alpha: np.array
    ):
        r'''Emulated interaction = $\hat{U}(s, \alpha, E)$

        Parameters:
            alpha (ndarray): interaction parameters
        
        Returns:
            u_hat (ndarray): emulated interaction

        '''
        _, x = self.coefficients(alpha)
        emu = np.sum(x * self.snapshots, axis=1)
        return emu
    

    def basis_functions(self, s_mesh: np.array):
        r'''$u_j$ in $\tilde{U} \approx \hat{U} \equiv \sum_j \beta_j(\alpha) u_j$

        Parameters:
            s_mesh (ndarray): $s$ mesh points
        
        Returns:
            u_j (ndarray): "pillars" (MxN matrix; M = number of mesh points; N = number of pillars)

        '''
        return np.copy(self.snapshots)
    

class InteractionEIMSpace(InteractionSpace):
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.array], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        l_max: int,
        training_info: np.array,
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        R_C: float = 0.0, # Coulomb "cutoff"
        is_complex: bool = False,
        spin_orbit_potential: Callable[[float, np.array, float], float] = None, #V_{SO}(r, theta, l•s)
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        r'''Generates a list of $\ell$-specific, EIMed interactions.
        
        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r, theta)
            n_theta (int): number of parameters
            mu (float): reduced mass
            energy (float): center-of-mass, scattering energy
            l_max (int): maximum angular momentum
            training_info (ndarray): See `InteractionEIM` documentation.
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex?
            spin_orbit_potential (Callable[[float, np.array, float], float]):
                used to create a `SpinOrbitTerm`
            n_basis (int): number of pillars --- basis states in $\hat{U}$ expansion
            explicit_training (bool): See `InteractionEIM` documentation.
            n_train (int): number of training samples
            rho_mesh (ndarray): discrete $\rho$ points
            match_points (ndarray): $\rho$ points where agreement with the true
                potential is enforced
        
        Returns:
            instance (InteractionEIMSpace): instance of InteractionEIMSpace
        
        Attributes:
            interactions (list): list of `InteractionEIM`s
        '''
        self.interactions = []
        if spin_orbit_potential is None:
            for l in range(l_max+1):
                self.interactions.append(
                    [InteractionEIM(coordinate_space_potential, n_theta, mu,
                        energy, l, training_info, Z_1=Z_1, Z_2=Z_2, R_C=R_C,
                        is_complex=is_complex, n_basis=n_basis, explicit_training=explicit_training,
                        n_train=n_train, rho_mesh=rho_mesh, match_points=match_points)]
                )
        else:
            for l in range(l_max+1):
                self.interactions.append(
                    [InteractionEIM(coordinate_space_potential, n_theta, mu,
                        energy, l, training_info, Z_1=Z_1, Z_2=Z_2, R_C=R_C,
                        is_complex=is_complex,
                        spin_orbit_term=SpinOrbitTerm(spin_orbit_potential,
                        lds), n_basis=n_basis,
                        explicit_training=explicit_training, n_train=n_train,
                        rho_mesh=rho_mesh, match_points=match_points) for lds in
                        couplings(l)]
                )