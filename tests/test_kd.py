import unittest
import numpy as np
import rose

from pathlib import Path

test_dir = Path(__file__).parent


def relative_difference(a, b):
    return np.abs((a - b) / b)


class TestKDProton(unittest.TestCase):
    def test_kd_params_default(self):
        potential = rose.koning_delaroche.KDGlobal(rose.Projectile.proton)
        Ca48 = (48, 20)
        E_lab = 25
        mu, E_com, k, eta = rose.utility.kinematics(
            target=Ca48, projectile=(1, 1), E_lab=E_lab
        )
        # use 24.9999 because CHEX does not a kinematic correction
        E_com = 24.999999999999996
        R_C, params = potential.get_params(*Ca48, mu, E_com, k)

        expected_params = (
            50.963183201704517,
            4.33328708,
            0.6706624,
            2.2986048514462647,
            4.33328708,
            0.6706624,
            8.1377218298207339,
            4.6692632103376672,
            0.54368400387465954,
            5.3212867840316207,
            3.6610296856505364,
            0.58999997377395630,
            -0.12462904135634313,
            3.6610296856505364,
            0.58999997377395630,
        )
        expected_RC = 4.620096149993784

        self.assertAlmostEqual(R_C, expected_RC)
        [
            self.assertAlmostEqual(params[i], expected_params[i], places=4)
            for i in range(len(params))
        ]


if __name__ == "__main__":
    unittest.main()
