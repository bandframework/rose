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
        '''
        Tests the correct handling of energy-dependent interactions. Does not
        use emulation.
        Compares the HF solutions from two interactions -- one that is
        energy-dependent and one that is not.
        '''
        mu = 1.0
        energy = 50.0
        ell = 0
        z = 0
        rho = rose.constants.DEFAULT_RHO_MESH
        potential = lambda r, theta: theta[0] * rose.interaction_eim.wood_saxon(r, theta[1], theta[2]) 

        interaction = rose.Interaction(potential, 3, mu, energy, z, z)
        se1 = rose.SchroedingerEquation(interaction)

        theta1 = np.array([-10.0, 3.0, 1.0])
        theta2 = np.hstack((energy, theta1))

        train = np.array([
            [49.0, 51.0],
            [-15.0, -5.0],
            [2.0, 4.0],
            [0.8, 1.2]
        ])
        energized_interaction = rose.EnergizedInteractionEIM(
            potential, 3, mu, ell, train, Z_1=z, Z_2=z, n_train=20
        )
        se2 = rose.SchroedingerEquation(energized_interaction)

        phi1 = se1.phi(theta1, rho, ell)
        phi2 = se2.phi(theta2, rho, ell)
        norm_diff = np.linalg.norm(phi1 - phi2)

        self.assertTrue(
            norm_diff < 1e-16,
            msg=f'norm(difference) = {norm_diff:.2e}'
        )


if __name__ == '__main__':
    unittest.main()