import unittest
import numpy as np
import rose

from pathlib import Path

test_dir = Path(__file__).parent


def relative_difference(a, b):
    return np.abs((a - b) / b)


class TestKDNeutron(unittest.TestCase):
    def test_kd_params_default(self):
        potential = rose.koning_delaroche.KDGlobal(rose.Projectile.neutron)
        Sc48 = (48, 21)
        E_lab = 17.677562500000001
        mu, E_com, k, eta = rose.utility.kinematics(
            target=Sc48, projectile=(1, 1), E_lab=E_lab
        )
        R_C, params = potential.get_params(*Sc48, mu, E_lab, k)

        expected_params = (
            45.319795995543529,
            4.3332872219656791,
            0.67066239984706044,
            1.4756965267922542,
            4.3332872219656791,
            0.67066239984706044,
            6.5289602674606506,
            4.6692632103376672,
            0.53665121016092598,
            5.4300256835300234,
            3.6610296110056351,
            0.58999997377395630,
            -9.0139921732509662e-02,
            3.6610296110056351,
            0.58999997377395630,
        )
        [
            self.assertAlmostEqual(params[i], expected_params[i], places=4)
            for i in range(len(params))
        ]

        r = 0.5
        self.assertAlmostEqual(
            rose.koning_delaroche.KD_simple(r, params),
            -45.171005786503798 - 1j * 1.4818793948938964,
            places=4,
        )
        lds = 1
        self.assertAlmostEqual(
            rose.koning_delaroche.KD_simple_so(r, params, lds),
            -0.17166788460689075 + 1j * 2.84973416044525998e-003,
            places=3,
        )


class TestKDProton(unittest.TestCase):
    def test_kd_params_default(self):
        potential = rose.koning_delaroche.KDGlobal(rose.Projectile.proton)
        Ca48 = (48, 20)
        E_lab = 25
        mu, E_com, k, eta = rose.utility.kinematics(
            target=Ca48, projectile=(1, 1), E_lab=E_lab
        )
        R_C, params = potential.get_params(*Ca48, mu, E_lab, k)

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

        Zz = Ca48[1]

        r = 0.5
        self.assertAlmostEqual(
            rose.koning_delaroche.KD_simple(r, params),
            -50.795865090152176 - 1j * 2.3062550528188286,
            places=4,
        )
        lds = 1
        self.assertAlmostEqual(
            rose.koning_delaroche.KD_simple_so(r, params, lds),
            -0.16823013082309032 + 1j * 3.94009208348830662e-003,
            places=3,
        )
        self.assertAlmostEqual(
            Zz
            * rose.constants.ALPHA
            * rose.constants.HBARC
            * rose.utility.regular_inverse_r(r, R_C),
            9.3137422616617105,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
