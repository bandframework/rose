r"""
Some helpful utilities for training an emulator
"""
import numpy as np
from scipy.stats import qmc

from .basis import Basis
from .interaction_eim import InteractionEIM, InteractionEIMSpace
from .scattering_amplitude_emulator import ScatteringAmplitudeEmulator


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
    ).T
    return qmc.scale(
        qmc.LatinHypercube(d=central_vals.size, seed=seed).random(N),
        bounds[:, 0],
        bounds[:, 1],
    )


def CAT_trainer_EIM(
    sae_config: tuple,
    base_interaction: InteractionEIMSpace,
    bases : list  = None,
    theta_train: np.array = None,
    **SAE_kwargs,
):
    r"""
    build an EIM emulator to specification of sae_config, using base_interaction
    Parameters:
        sae_config :  (size of reduced basis, number of EIM terms)
        base_interaction :  interaction on which to train
        bases (optional) : if a full set of bases for each interaction has been solved
            for already, re-use basis.vectors rather than re-calculating them
        theta_train (optional) : is bases is not provided, simply pass in the
            training samples and re-train the emulator
        SAE_kwargs : passed to ScatteringAmplitudeEmulator
    """

    (n_basis, n_EIM) = sae_config

    interactions = InteractionEIMSpace(
        base_interaction.coordinate_space_potential,
        base_interaction.n_theta,
        base_interaction.mu,
        base_interaction.energy,
        base_interaction.training_info,
        l_max=base_interaction.l_max,
        Z_1=base_interaction.Z_1,
        Z_2=base_interaction.Z_2,
        R_C=base_interaction.R_C,
        is_complex=base_interaction.is_complex,
        spin_orbit_potential=base_interaction.spin_orbit_potential,
        explicit_training=base_interaction.explicit_training,
        n_train=base_interaction.n_train,
        rho_mesh=base_interaction.rho_mesh,
        n_basis=n_EIM,
    )

    if theta_train is None:
        assert bases is not None
        new_bases = []
        for interaction_list, basis_list in zip(interactions.interactions, bases):
            basis_list = []
            for interaction, basis in zip(interaction_list, basis_list):
                # add back free solution to get HF solutions
                solutions = (basis.pillars.T + basis.phi_0).T
                basis_list.append(
                    CustomBasis(
                        solutions[:, :n_basis],
                        basis.phi_0,
                        basis.rho_mesh,
                        n_basis,
                        interaction.ell,
                        use_svd=False,
                    )
                )
            new_bases.append(basis_list)
        emulator = ScatteringAmplitudeEmulator(interaction, new_bases,  **SAE_kwargs)
    else:
        emulator = ScatteringAmplitudeEmulator.from_train(
            interaction,
            training_samples,
            n_basis=n_basis,
            **SAE_kwargs,
        )

    return interactions, emulator
