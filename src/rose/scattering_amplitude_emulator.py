import pickle
import numpy as np
from scipy.special import eval_legendre
from tqdm import tqdm 

from .interaction import Interaction
from .reduced_basis_emulator import ReducedBasisEmulator
from .constants import DEFAULT_RHO_MESH, DEFAULT_ANGLE_MESH, HBARC

class ScatteringAmplitudeEmulator:

    @classmethod
    def load(obj, filename):
        with open(filename, 'rb') as f:
            sae = pickle.load(f)
        return sae


    def __init__(self,
        interaction: Interaction,
        theta_train: np.array,
        energy: float,
        l_max: int,
        angles: np.array = DEFAULT_ANGLE_MESH,
        n_basis: int = 4,
        use_svd: bool = True,
        s_mesh: np.array = DEFAULT_RHO_MESH,
        s_0: float = 6*np.pi,
        hf_tols: list = None
    ):
        '''
        :param interaction:
        '''
        self.l_max = l_max
        self.angles = angles.copy()
        self.rbes = []
        for l in tqdm(range(self.l_max + 1)):
            self.rbes.append(
                ReducedBasisEmulator(
                    interaction, theta_train, energy, l,
                    n_basis=n_basis,
                    use_svd=use_svd,
                    s_mesh=s_mesh,
                    s_0=s_0,
                    hf_tols=hf_tols
                )
            )
        self.k = np.sqrt(2*interaction.mu*energy/HBARC)

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