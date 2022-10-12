'''
Defines a benchmark class that computes:
    * phase shifts
    * partial wave amplitudes
    * scattering amplitude
    * wave functions: u(s)
    * mesh: s
'''
import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre

import rose


ANGLES = np.linspace(0, np.pi, 100)
COSTHETA = np.cos(ANGLES)

class BenchmarkData():
    '''
    Computes a set of quantities that we can compare to later.
    '''
    def __init__(self,
        schrodeq: rose.SchroedingerEquation,
        energy: float, # MeV, c.m.
        theta: npt.ArrayLike, # interaction parameters; not related to cos(theta)
        n: int = 10000, # number of point in r to calculate u(r)
        l_max: int = 3, # maximum angular momentum
        r_0: float = 20.0, # fm, delta <- u(r_0)
        costheta: npt.ArrayLike = COSTHETA # angles at which P_l(cos(theta)) are calculated
    ):
        self.theta = np.copy(theta)

        k = np.sqrt(2*rose.MN_Potential.mu*energy/rose.constants.HBARC)
        r_mesh = np.linspace(rose.schroedinger.DEFAULT_R_MIN,
                             rose.schroedinger.DEFAULT_R_MAX, n)
        s_mesh = k*r_mesh
            
        self.l_max = l_max
        self.costheta = np.copy(costheta)
        l_values = np.arange(self.l_max+1)
        self.deltas_l = np.array(
            [schrodeq.delta(energy, theta, s_mesh, l, k*r_0) for l in l_values]
        )
        self.fs_l = np.array(
            [1/(k/np.tan(delta_l) - 1j*k) for delta_l in self.deltas_l]
        )
        Pl = np.array([eval_legendre(l, self.costheta) for l in l_values]).T
        self.scattering_amplitude = np.sum((2*l_values + 1.0) * self.fs_l * Pl, axis=1)

        sol = schrodeq.solve_se(energy, theta, s_mesh, 0)
        self.s = np.copy(sol[:, 0])
        c = np.max(np.abs(sol[:, 1])) # normalize to 1
        self.u = np.copy(sol[:, 1]/c)