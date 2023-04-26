import unittest
import numpy as np
import rose

def relative_difference(a, b):
    return np.abs((a-b)/b)


class TestMNPotential(unittest.TestCase):
    
    def test_mn_potential(self):
        self.assertTrue(
            relative_difference(
                rose.MN_Potential.v_r(1.0, np.array([1.0, 1.0])),
                0.854184893887586) < 1e-12,
            msg=f'V_MN = {rose.MN_Potential.v_r(1.0, np.array([1.0, 1.0]))}')


if __name__ == '__main__':
    unittest.main()