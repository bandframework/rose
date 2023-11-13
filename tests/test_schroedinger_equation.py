import unittest
import numpy as np

from pathlib import Path
test_dir = Path(__file__).parent

import rose

# Values used in Pablo's notebook.
ENERGY = 50  # MeV
V0R = 200
V0S = -91.85
_, u_pg = np.loadtxt( test_dir / "u_mn_test.txt", unpack=True)
u_pg /= np.max(u_pg)  # This is a very hacky way to handle normalization.


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
        """
        Tests the correct handling of energy-dependent interactions. Does not
        use emulation.
        Compares the HF solutions from two interactions -- one that is
        energy-dependent and one that is not.
        """
        mu = 1.0
        energy = 50.0
        ell = 0
        z = 0
        rho = rose.constants.DEFAULT_RHO_MESH
        potential = lambda r, theta: theta[0] * rose.koning_delaroche.woods_saxon(
            r, theta[1], theta[2]
        )

        interaction = rose.Interaction(potential, 3, mu, energy, z, z)
        se1 = rose.SchroedingerEquation(interaction)

        theta1 = np.array([-10.0, 3.0, 1.0])
        theta2 = np.hstack((energy, theta1))

        train = np.array([[49.0, 51.0], [-15.0, -5.0], [2.0, 4.0], [0.8, 1.2]])
        energized_interaction = rose.EnergizedInteractionEIM(
            potential, 3, mu, ell, train, Z_1=z, Z_2=z, n_train=20
        )
        se2 = rose.SchroedingerEquation(energized_interaction)

        phi1 = se1.phi(theta1, rho, ell)
        phi2 = se2.phi(theta2, rho, ell)
        norm_diff = np.linalg.norm(phi1 - phi2)

        self.assertTrue(norm_diff < 1e-16, msg=f"norm(difference) = {norm_diff:.2e}")

    def test_emulation(self):
        AMU = 931.5
        A = 40
        mu = A / (A + 1) * AMU
        energy = 13.659

        alphastar = np.array(
            [
                51.9,
                4.00134372,
                0.75,
                0.4,
                4.5143365,
                0.51,
                8.3,
                4.5143365,
                0.51,
                6.2,
                3.45415141,
                0.75,
                1.0,
                3.45415141,
                0.75,
            ]
        )
        ntheta = alphastar.size

        train = np.load(test_dir / "train.npy")

        ell = 3
        interaction = rose.InteractionEIM(
            rose.koning_delaroche.KD_simple,
            ntheta,
            mu,
            energy,
            ell,
            train,
            is_complex=True,
            spin_orbit_term=rose.SpinOrbitTerm(rose.koning_delaroche.KD_simple_so, ell),
            n_basis=ntheta + 16,
            explicit_training=True,
        )

        y = interaction.tilde(interaction.s_mesh, alphastar)
        yp = interaction.tilde_emu(alphastar)

        norm_diff = np.linalg.norm(y - yp)
        self.assertTrue(norm_diff < 1e-4, msg=f"norm(difference) = {norm_diff:.2e}")


if __name__ == "__main__":
    unittest.main()
