import pickle
import numpy as np
from scipy.special import eval_legendre
from tqdm import tqdm 

from .interaction import Interaction
from .reduced_basis_emulator import ReducedBasisEmulator
from .constants import DEFAULT_RHO_MESH, DEFAULT_ANGLE_MESH, HBARC
from .schroedinger import SchroedingerEquation
from .basis import RelativeBasis

class ScatteringAmplitudeEmulator:

    @classmethod
    def load(obj, filename):
        with open(filename, 'rb') as f:
            sae = pickle.load(f)
        return sae


    @classmethod
    def from_train(cls,
        interaction: Interaction,
        theta_train: np.array,
        l_max: int,
        angles: np.array = DEFAULT_ANGLE_MESH,
        n_basis: int = 4,
        use_svd: bool = True,
        s_mesh: np.array = DEFAULT_RHO_MESH,
        s_0: float = 6*np.pi,
        hf_tols: list = None
    ):
        solver = SchroedingerEquation(interaction, hifi_tolerances=hf_tols)
        
        bases = []
        for l in tqdm(range(l_max+1)):
            bases.append(RelativeBasis(
                solver,
                theta_train,
                s_mesh,
                n_basis,
                l,
                use_svd
            ))
        return cls(interaction, bases, l_max, angles=angles, s_0=s_0, hf_tols=hf_tols)


    def __init__(self,
        interaction: Interaction,
        bases: list,
        l_max: int,
        angles: np.array = DEFAULT_ANGLE_MESH,
        s_0: float = 6*np.pi,
        hf_tols: list = None
    ):
        '''
        :param interaction:
        '''
        self.l_max = l_max
        self.angles = angles.copy()
        self.rbes = []
        for l in range(self.l_max + 1):
            self.rbes.append(
                ReducedBasisEmulator(
                    interaction, bases[l], l, s_0=s_0
                )
            )
        self.k = np.sqrt(2*interaction.mu*interaction.energy/HBARC)

    def predict(self, alpha):
        phase_shifts = np.array([
            rbe.emulate_phase_shift(alpha) for rbe in self.rbes
        ])
        tl = np.exp(1j*phase_shifts) * np.sin(phase_shifts)
        f = np.array([
            1/self.k * (2*l+1) * eval_legendre(l, np.cos(self.angles)) * t for (l, t) in enumerate(tl)
        ])
        return np.sum(f, axis=0)
    

    def exact(self, alpha: np.array):
        phase_shifts = np.array([
            rbe.exact_phase_shift(alpha) for rbe in self.rbes
        ])
        tl = np.exp(1j*phase_shifts) * np.sin(phase_shifts)
        f = np.array([
            1/self.k * (2*l+1) * eval_legendre(l, np.cos(self.angles)) * t for (l, t) in enumerate(tl)
        ])
        return np.sum(f, axis=0)
    

    def emulate_dsdo(self,
        theta: np.array
    ):
        Sls = np.array([rbe.S_matrix_element(theta) for rbe in self.rbes])
        f = np.array([
            -1j/(2*self.k) * (2*l + 1) * eval_legendre(l, np.cos(self.angles)) * (Sl - 1) for (l, Sl) in enumerate(Sls)
        ])
        f = np.sum(f, axis=0)
        return np.conj(f) * f


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)