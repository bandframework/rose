import numpy as np
from scipy.special import eval_legendre

from .interaction import Interaction
from .reduced_basis_emulator import ReducedBasisEmulator
from .constants import DEFAULT_RHO_MESH, DEFAULT_ANGLE_MESH, HBARC

class ScatteringAmplitudeEmulator:
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
        **kwargs
    ):
        '''
        :param interaction: 
        '''
        self.l_max = l_max
        self.angles = angles.copy()
        self.rbes = [
            ReducedBasisEmulator(interaction, theta_train, energy, l,
                n_basis=n_basis, use_svd=use_svd, s_mesh=s_mesh, s_0=s_0,
                **kwargs) for l in range(self.l_max+1)
        ]
        self.k = np.sqrt(2*interaction.mu*energy/HBARC)

    def predict(self, alpha):
        phase_shifts = np.array([
            rbe.emulate_phase_shift(alpha) for rbe in self.rbes
        ])
        tl = np.exp(1j*phase_shifts) * np.sin(phase_shifts)
        dsdo = np.array([
            1/self.k * (2*l+1) * eval_legendre(l, np.cos(self.angles)) * t for (l, t) in enumerate(tl)
        ])
        return np.sum(dsdo, axis=0)