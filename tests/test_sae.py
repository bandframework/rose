import unittest
import numpy as np
from numba import njit

from pathlib import Path

test_dir = Path(__file__).parent

import rose

# Values used in Pablo's notebook.
ENERGY = 50  # MeV
V0R = 200
V0S = -91.85


class TestSAE(unittest.TestCase):
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

        interaction = rose.InteractionEIMSpace(
            coordinate_space_potential=rose.koning_delaroche.KD_simple,
            spin_orbit_term=rose.koning_delaroche.KD_simple_so,
            n_theta=ntheta,
            mu=mu,
            energy=energy,
            is_complex=True,
            training_info=train,
            n_basis=15,
            explicit_training=True,
        )

        sae = rose.ScatteringAmplitudeEmulator.from_train(
            interaction_space=interaction,
            alpha_train=train[::10],
            n_basis=16,
        )

        xs_emu = sae.emulate_dsdo(alphastar)
        xs = sae.exact_dsdo(alphastar)

        np.testing.assert_allclose(xs_emu, xs, rtol=0.05)

        # test saving and loading
        sae.save("./test_sae.pkl")
        sae2 = rose.ScatteringAmplitudeEmulator.load("./test_sae.pkl")
        xs_emu2 = sae2.emulate_dsdo(alphastar)
        np.testing.assert_equal(xs_emu2, xs_emu)


if __name__ == "__main__":
    unittest.main()
