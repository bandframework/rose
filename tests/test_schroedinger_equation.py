import unittest
import numpy as np

import rose

# Values used in Pablo's notebook.
ENERGY = 50 # MeV
V0R = 200
V0S = -91.85
_, u_pg = np.loadtxt('u_mn_test.txt', unpack=True)
u_pg /= np.max(u_pg) # This is a very hacky way to handle normalization.

class TestSchrEq(unittest.TestCase):
    # def test_schrodinger_equation(self):
    #     se = roqse.SchroedingerEquation(roqse.MN_Potential)
    #     solution = se.solve_se(ENERGY, np.array([V0R, V0S]), r_min=0.01)
    #     s = solution[:, 0]
    #     u = solution[:, 1] / np.max(solution[:, 1])
    #     self.assertTrue(
    #         np.linalg.norm(u - u_pg) < 0.1
    #     )
    

    def test_energy(self):
        mu = 1.0
        energy = 50.0
        ell = 0
        rho = rose.constants.DEFAULT_RHO_MESH
        potential = lambda r, theta: theta[0] * np.exp(-(r/theta[1])**2) 

        interaction = rose.Interaction(potential, 2, mu, energy, 1, 1)
        energized_interaction = rose.interaction.EnergizedInteraction(potential, 3, mu, 1, 1)

        se1 = rose.SchroedingerEquation(interaction)
        se2 = rose.SchroedingerEquation(energized_interaction)

        theta1 = np.array([-10.0, 1.0])
        theta2 = np.hstack((energy, theta1))

        phi1 = se1.phi(theta1, rho, ell)
        phi2 = se2.phi(theta2, rho, ell)
        norm_diff = np.linalg.norm(phi1 - phi2)

        self.assertTrue(
            norm_diff < 1e-16,
            msg=f'norm(difference) = {norm_diff:.2e}'
        )


if __name__ == '__main__':
    unittest.main()