import pickle
import numpy as np
from scipy.special import eval_legendre, gamma
from tqdm import tqdm 

from .interaction import InteractionSpace
from .reduced_basis_emulator import ReducedBasisEmulator
from .constants import DEFAULT_RHO_MESH, DEFAULT_ANGLE_MESH
from .schroedinger import SchroedingerEquation
from .basis import RelativeBasis, CustomBasis
from .utility import eval_assoc_legendre

class ScatteringAmplitudeEmulator:

    @classmethod
    def load(obj, filename):
        r'''Loads a previously trained emulator.

        Parameters:
            filename (string): name of file
        
        Returns:
            emulator (ScatteringAmplitudeEmulator): previously trainined `ScatteringAmplitudeEmulator`
        
        '''
        with open(filename, 'rb') as f:
            sae = pickle.load(f)
        return sae


    @classmethod
    def from_train(cls,
        interaction_space: InteractionSpace,
        theta_train: np.array,
        l_max: int,
        angles: np.array = DEFAULT_ANGLE_MESH,
        n_basis: int = 4,
        use_svd: bool = True,
        s_mesh: np.array = DEFAULT_RHO_MESH,
        s_0: float = 6*np.pi,
        hf_tols: list = None
    ):
        r'''Trains a reduced-basis emulator based on the provided interaction and training space.
        
        Parameters:
            interaction_space (InteractionSpace): local interaction up to (and including $\ell_\max$)
            theta_train (ndarray): training points in parameter space; shape = (n_points, n_parameters)
            l_max (int): maximum angular momentum to include in the sum approximating the cross section
            angles (ndarray): Differential cross sections are functions of the
                angles. These are the specific values at which the user wants to
                emulate the cross section.
            n_basis (int): number of basis vectors for $\hat{\phi}$ expansion
            use_svd (bool): Use principal components of training wave functions?
            s_mesh (ndarray): $s$ (or $\rho$) grid on which wave functions are evaluated
            s_0 (float): $s$ point where the phase shift is extracted
            hf_tols (list): 2-element list passed to `SchroedingerEquation`;
                [relative tolerance, absolute tolerance]
        
        Returns:
            sae (ScatteringAmplitudeEmulator): scattering amplitude emulator
        
        '''
        bases = []
        for interaction_list in tqdm(interaction_space.interactions):
            basis_list = [RelativeBasis(
                SchroedingerEquation(interaction, hifi_tolerances=hf_tols),
                theta_train, s_mesh, n_basis, interaction.ell, use_svd
            ) for interaction in interaction_list]
            bases.append(basis_list)

        return cls(interaction_space, bases, l_max, angles=angles, s_0=s_0)


    def __init__(self,
        interaction_space: InteractionSpace,
        bases: list,
        l_max: int,
        angles: np.array = DEFAULT_ANGLE_MESH,
        s_0: float = 6*np.pi,
        verbose: bool = True
    ):
        r'''Trains a reduced-basis emulator that computes differential and total cross sections (from emulated phase shifts).

        Parameters:
            interaction_space (InteractionSpace): local interaction up to (and including $\ell_\max$)
            bases (list[Basis]): list of `Basis` objects
            l_max (int): maximum angular momentum to include in the sum approximating the cross section
            angles (ndarray): Differential cross sections are functions of the
                angles. These are the specific values at which the user wants to
                emulate the cross section.
            s_0 (float): $s$ point where the phase shift is extracted
            verbose (bool): Do you want the class to print out warnings?

        Attributes:
            l_max (int): maximum angular momentum
            angles (ndarray): angle values at which the differential cross section is desired
            rbes (list): list of `ReducedBasisEmulators`; one for each partial wave (and total $j$ with spin-orbit)
            ls (ndarray): angular momenta; shape = (`l_max`+1, 1)
            P_l_costheta (ndarray): Legendre polynomials evaluated at `angles`
            P_1_l_costheta (ndarray): **associated** Legendre polynomials evalated at `angles`
            k_c (float): Coulomb momentum, $k\eta$
            eta (float): Sommerfeld parameter
            sigma_l (float): Coulomb phase shift
            f_c (ndarray): scattering amplitude
            rutherford (ndarray): Rutherford scattering

        '''
        self.l_max = l_max
        self.angles = angles.copy()
        bases_types = [isinstance(bases[l], CustomBasis) for l in range(self.l_max+1)]
        if np.any(bases_types) and verbose:
            print('''
NOTE: When supplying a CustomBasis, the ROSE high-fidelity solver is \n
instantiated for the sake of future evaluations.  Any requests to the solver \n
will NOT be communicated to the user's own high-fidelity solver.
''')
        self.rbes = []
        for (interaction_list, basis_list) in zip(interaction_space.interactions, bases):
            self.rbes.append([ReducedBasisEmulator(interaction, basis, s_0=s_0) for
                (interaction, basis) in zip(interaction_list, basis_list)])


        # Let's precompute the things we can.
        self.ls = np.arange(self.l_max+1)[:, np.newaxis]
        self.P_l_costheta = eval_legendre(self.ls, np.cos(self.angles))
        self.P_1_l_costheta = np.array([eval_assoc_legendre(l, np.cos(self.angles)) for l in self.ls])
        # Coulomb scattering amplitude
        # (This is dangerous because it's not fixed when we emulate across
        # energies, BUT we don't do that with Coulomb (yet). When we do emulate
        # across energies, f_c is zero anyway.)
        k = self.rbes[0][0].interaction.momentum(None)
        self.k_c = self.rbes[0][0].interaction.k_c
        self.eta = self.k_c / k
        self.sigma_l = np.angle(gamma(1 + self.ls + 1j*self.eta))
        sin2 = np.sin(self.angles/2)**2
        self.f_c = -self.eta / (2*k*sin2) * np.exp(-1j*self.eta*np.log(sin2) + 2j*self.sigma_l[0])
        self.rutherford = self.eta**2 / (4*k**2*np.sin(self.angles/2)**4)


    def emulate_phase_shifts(self,
        theta: np.array
    ):
        r'''Gives the phase shifts for each partial wave.  Order is [l=0, l=1,
            ..., l=l_max-1].
        
        Parameters:
            theta (ndarray): parameter-space vector
        
        Returns:
            phase_shift (list): emulated phase shifts

        '''
        return [[rbe.emulate_phase_shift(theta) for rbe in rbe_list] for rbe_list in self.rbes]

    def exact_phase_shifts(self,
        theta: np.array
    ):
        r'''Gives the phase shifts for each partial wave. Order is [l=0, l=1,
            ..., l=l_max-1].
        
        Parameters:
            theta (ndarray): parameter-space vector
        
        Returns:
            phase_shift (list): high-fidelity phase shifts

        '''
        return [[rbe.exact_phase_shift(theta) for rbe in rbe_list] for rbe_list in self.rbes]


    def dsdo(self, theta : np.array, deltas : np.array):
        r'''Gives the differential cross section (dsigma/dOmega = dsdo).

        Parameters:
            theta (ndarray): parameter-space vector
            deltas (ndarray): phase shifts
        
        Returns:
            dsdo (ndarray): differential cross section (fm^2)

        '''
        k = self.rbes[0][0].interaction.momentum(theta)

        # Coulomb-distorted, nuclear scattering amplitude
        deltas = self.emulate_phase_shifts(theta)
        deltas_plus = np.array([d[0] for d in deltas])
        deltas_minus = np.array([d[1] for d in deltas[1:]])

        S_l_plus = np.exp(2j*deltas_plus)[:, np.newaxis]
        if self.rbes[0][0].interaction.include_spin_orbit:
            # If there is spin-orbit, the l=0 term for B has to be zero.
            S_l_minus = np.hstack((S_l_plus[0], np.exp(2j*deltas_minus)))[:, np.newaxis]
        else:
            # This ensures that A reduces to the non-spin-orbit formula, and B = 0.
            S_l_minus = S_l_plus.copy()
        
        A = self.f_c + 1/(2j*k) * np.sum(
            np.exp(2j*self.sigma_l) * ((self.ls+1)*(S_l_plus - 1) + \
                self.ls*(S_l_minus - 1)) * self.P_l_costheta,
            axis=0
        )
        B = 1/(2j*k) * np.sum(
            np.exp(2j*self.sigma_l) * (S_l_plus - S_l_minus) * self.P_l_costheta,
            axis=0
        )

        dsdo = (np.conj(A)*A + np.conj(B)*B).real
        if self.k_c > 0:
            return dsdo / self.rutherford
        else:
            return dsdo


    def emulate_dsdo(self,
        theta: np.array
    ):
        r'''Emulates the differential cross section (dsigma/dOmega = dsdo).

        Parameters:
            theta (ndarray): parameter-space vector

        Returns:
            dsdo (ndarray): emulated differential cross section

        '''
        deltas = self.emulate_phase_shifts(theta)
        return self.dsdo(theta, deltas)

    def exact_dsdo(self,
        theta: np.array
    ):
        r'''Calculates the high-fidelity differential cross section (dsigma/dOmega = dsdo).

        Parameters:
            theta (ndarray): parameter-space vector

        Returns:
            dsdo (ndarray): high-fidelity differential cross section

        '''
        deltas = self.exact_phase_shifts(theta)
        return self.dsdo(theta, deltas)


    def emulate_wave_functions(self,
        theta: np.array
    ):
        r'''Gives the wave functions for each partial wave.  Returns a list of
            arrays.  Order is [l=0, l=1, ..., l=l_max-1].
        
        Parameters:
            theta (ndarray): parameter-space vector

        Returns:
            wave_functions (list): emulated wave functions

        
        '''
        return [[x.emulate_wave_function(theta) for x in rbe] for rbe in self.rbes]


    def S_matrix_elements(self,
        deltas: list
    ):
        deltas_plus = np.array([d[0] for d in deltas])
        deltas_minus = np.array([d[1] for d in deltas[1:]])

        S_l_plus = np.exp(2j*deltas_plus)[:, np.newaxis]
        if self.rbes[0][0].interaction.include_spin_orbit:
            # If there is spin-orbit, the l=0 term for B has to be zero.
            S_l_minus = np.hstack((S_l_plus[0], np.exp(2j*deltas_minus)))[:, np.newaxis]
        else:
            # This ensures that A reduces to the non-spin-orbit formula, and B = 0.
            S_l_minus = S_l_plus.copy()
        
        return S_l_plus, S_l_minus


    def emulate_total_cross_section(self,
        theta: np.array
    ):
        r'''Gives the "total" (angle-integrated) cross section.  See Eq. (63)
            in Carlson's notes.
        
        Parameters:
            theta (ndarray): parameter-space vector
        
        Returns:
            cross_section (ndarray): emulated total cross section

        '''
        # What do we do here when Coulomb and/or spin-orbit is present?
        if self.k_c > 0:
            raise Exception('The total cross section is infinite in the presence of Coulomb.')
        
        k = self.rbes[0][0].interaction.momentum(theta)
        S_l_plus, S_l_minus = self.S_matrix_elements(self.emulate_phase_shifts(theta))

        sum = np.sum(np.pi/k**2 * (2*self.ls + 2) * (1 - S_l_plus.real))
        sum += np.sum(np.pi/k**2 * (2*self.ls - 2) * (1 - S_l_minus.real))

        return sum


    def exact_total_cross_section(self,
        theta: np.array
    ):
        r'''Gives the "total" (angle-integrated) cross section.  See Eq. (63)
            in Carlson's notes.
        
        Parameters:
            theta (ndarray): parameter-space vector
        
        Returns:
            cross_section (ndarray): emulated total cross section

        '''
        # What do we do here when Coulomb and/or spin-orbit is present?
        if self.k_c > 0:
            raise Exception('The total cross section is infinite in the presence of Coulomb.')
        
        k = self.rbes[0][0].interaction.momentum(theta)
        S_l_plus, S_l_minus = self.S_matrix_elements(self.exact_phase_shifts(theta))

        sum = np.sum(np.pi/k**2 * (2*self.ls + 2) * (1 - S_l_plus.real))
        sum += np.sum(np.pi/k**2 * (2*self.ls - 2) * (1 - S_l_minus.real))

        return sum


    def save(self, filename):
        r'''Saves the emulator to the desired file.
        
        Parameters:
            filename (string): name of file
        
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)