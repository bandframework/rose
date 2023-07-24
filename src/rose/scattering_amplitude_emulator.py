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
        '''
        :param interaction:
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
        self.P_1_l_costheta = np.array([[eval_assoc_legendre(l, a) for a in np.cos(self.angles)] 
                                        for l in self.ls])
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
        '''
        Gives the phase shifts for each partial wave.
        Order is [l=0, l=1, ..., l=l_max-1].
        '''
        return [[rbe.emulate_phase_shift(theta) for rbe in rbe_list] for rbe_list in self.rbes]

    def exact_phase_shifts(self,
        theta: np.array
    ):
        '''
        Gives the phase shifts for each partial wave.
        Order is [l=0, l=1, ..., l=l_max-1].
        '''
        return [[rbe.exact_phase_shift(theta) for rbe in rbe_list] for rbe_list in self.rbes]


    def dsdo(self, theta : np.array, deltas : np.array)
        '''
        Gives the differential cross section (dsigma/dOmega = dsdo).
        '''
        k = self.rbes[0][0].interaction.momentum(theta)

        # Coulomb-distorted, nuclear scattering amplitude
        S_l_plus = np.array([np.exp(2j*d[0]) for d in deltas])[:, np.newaxis]
        if self.rbes[0][0].interaction.include_spin_orbit:
            S_l_minus = np.array([S_l_plus[0]] + [np.exp(2j*d[1]) for d in deltas[1:]])[:, np.newaxis]
        else:
            S_l_minus = S_l_plus.copy()
        A = self.f_c + 1/(2j*k) * np.sum(
            np.exp(2j*self.sigma_l) * ((self.ls+1)*(S_l_plus - 1) + \
                self.ls*(S_l_minus - 1)) * self.P_l_costheta,
            axis=0
        )
        B = 1/(2j*k) * np.sum(
            np.exp(2j*self.sigma_l) * (S_l_plus - S_l_minus) * self.P_1_l_costheta,
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
        '''
        Gives the differential cross section (dsigma/dOmega = dsdo).
        '''
        # Coulomb-distorted, nuclear scattering amplitude
        deltas = self.emulate_phase_shifts(theta)
        return self.dsdo(theta, deltas)

    def exact_dsdo(self,
        theta: np.array
    ):
        '''
        Gives the differential cross section (dsigma/dOmega = dsdo).
        '''
        # Coulomb-distorted, nuclear scattering amplitude
        deltas = self.exact_phase_shifts(theta)
        return self.dsdo(theta, deltas)


    def emulate_wave_functions(self,
        theta: np.array
    ):
        '''
        Gives the wave functions for each partial wave.
        Returns a list of arrays.
        Order is [l=0, l=1, ..., l=l_max-1].
        '''
        return [[x.emulate_wave_function(theta) for x in rbe] for rbe in self.rbes]


    def emulate_total_cross_section(self,
        theta: np.array,
        rel_ruth: bool = True
    ):
        '''
        Gives the "total" (angle-integrated) cross section.
        See Eq. (3.1.50) in Thompson and Nunes.
        :param rel_ruth: Report the total cross section relative to Rutherford?
        '''
        # What do we do here when Coulomb and/or spin-orbit is present?
        return None


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)