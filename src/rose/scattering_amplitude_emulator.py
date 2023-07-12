import pickle
import numpy as np
from scipy.special import eval_legendre, gamma
from tqdm import tqdm 

from .interaction import Interaction, InteractionSpace
from .reduced_basis_emulator import ReducedBasisEmulator
from .constants import DEFAULT_RHO_MESH, DEFAULT_ANGLE_MESH, HBARC
from .schroedinger import SchroedingerEquation
from .basis import RelativeBasis, CustomBasis

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


    def predict(self, alpha):
        k = self.rbes[0][0].interaction.momentum(alpha)
        phase_shifts = np.array([
            rbe.emulate_phase_shift(alpha) for rbe in self.rbes
        ])
        tl = np.exp(1j*phase_shifts) * np.sin(phase_shifts)
        f = np.array([
            1/k * (2*l+1) * eval_legendre(l, np.cos(self.angles)) * t for (l, t) in enumerate(tl)
        ])
        return np.sum(f, axis=0)
    

    def exact(self, alpha: np.array):
        k = self.rbes[0].interaction.momentum(alpha)
        phase_shifts = np.array([
            rbe.exact_phase_shift(alpha) for rbe in self.rbes
        ])
        tl = np.exp(1j*phase_shifts) * np.sin(phase_shifts)
        f = np.array([
            1/k * (2*l+1) * eval_legendre(l, np.cos(self.angles)) * t for (l, t) in enumerate(tl)
        ])
        return np.sum(f, axis=0)
    

    def emulate_dsdo(self,
        theta: np.array
    ):
        '''
        Gives the differential cross section (dsigma/dOmega = dsdo).
        '''
        k = self.rbes[0].interaction.momentum(theta)
        eta = self.rbes[0].interaction.k_c / k
        ls = np.arange(self.l_max+1)[:, np.newaxis]
        sigma_l = np.angle(gamma(1 + ls + 1j*eta))
        S_n = np.exp(2j*np.array(self.emulate_phase_shifts(theta)))[:, np.newaxis]
        f_n = np.sum(
            1/(2j*k) * (2*ls + 1) * \
                eval_legendre(ls, np.cos(self.angles)) * np.exp(2j*sigma_l) * (S_n - 1),
            axis=0
        )

        sin2 = np.sin(self.angles/2)**2
        f_c = -eta / (2*k*sin2) * np.exp(-1j*eta*np.log(sin2) + 2j*sigma_l[0])

        f = f_n + f_c
        return (np.conj(f) * f).real


    def emulate_wave_functions(self,
        theta: np.array
    ):
        '''
        Gives the wave functions for each partial wave.
        Returns a list of arrays.
        Order is [l=0, l=1, ..., l=l_max-1].
        '''
        return [[x.emulate_wave_function(theta) for x in rbe] for rbe in self.rbes]


    def emulate_phase_shifts(self,
        theta: np.array
    ):
        '''
        Gives the phase shifts for each partial wave.
        Order is [l=0, l=1, ..., l=l_max-1].
        '''
        return [[rbe.emulate_phase_shift(theta) for rbe in rbe_list] for rbe_list in self.rbes]


    def emulate_total_cross_section(self,
        theta: np.array,
        rel_ruth: bool = True
    ):
        '''
        Gives the "total" (angle-integrated) cross section.
        See Eq. (3.1.50) in Thompson and Nunes.
        :param rel_ruth: Report the total cross section relative to Rutherford?
        '''
        dsdo = self.emulate_dsdo(theta)

        if rel_ruth:
            k = self.rbes[0].interaction.momentum(theta)
            eta = self.rbes[0].interaction.k_c / k
            return dsdo / (eta**2 / (4*k**2*np.sin(self.angles/2)**4))
        else:
            return dsdo


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)