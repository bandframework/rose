import unittest
import numpy as np

import roqse

# Values used in Pablo's notebook.
ENERGY = 50 # MeV
V0R = 200
V0S = -91.85
_, u_pg = np.loadtxt('/Users/danielodell/u_mn_test.txt', unpack=True)
u_pg /= np.max(u_pg) # This is a very hacky way to handle normalization.

class TestSchrEq(unittest.TestCase):
    def test_schrodinger_equation(self):
        se = roqse.SchroedingerEquation(roqse.MN_Potential)
        solution = se.solve_se(ENERGY, np.array([V0R, V0S]), r_min=0.01)
        s = solution[:, 0]
        u = solution[:, 1] / np.max(solution[:, 1])

        self.assertTrue(
            np.linalg.norm(u - u_pg) < 0.1
        )


if __name__ == '__main__':
    unittest.main()