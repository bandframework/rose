r"""
Some helpful utilities for training an emulator
"""

import numpy as np
from scipy.stats import qmc


def sample_params_LHC(
    N: int, central_vals: np.array, scale: float = 0.5, seed: int = None
):
    r"""
    Sampling parameters from a finite box in parameter space around some central values using the Latin hypercube method
    Parameters:
        N : number of samples
        central_vals : central values of each parameter
        scale : fraction of central vals, such that (1 +/- scale)*abs(central_vals) defines the bounds
                of the box
        seed : RNG seed. If None, uses entropy from the system
    Returns:
        (ndarray) : N samples
    """
    bounds = np.array(
        [
            central_vals - np.fabs(central_vals * scale),
            central_vals + np.fabs(central_vals * scale),
        ]
    )
    return qmc.scale(
        qmc.LatinHypercube(d=parameters.size, seed=seed).random(N),
        bounds[:, 0],
        bounds[:, 1],
    )
