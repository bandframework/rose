# ROSE - The **R**educed-**O**rder **S**cattering **E**mulator

ROSE makes it easy to build and train a scattering emulator. ROSE enables aspiring graduate students and long-suffering postdocs to emulate nuclear scattering observables with optical potentials, trading negligible amounts of accuracy for orders-of-magnitude gains in speed.

For any bug reports or feature requests, please make use of the Github issues tab on the repository. We also welcome all pull requests for software, documentation, and user-contributed tutorials! 

## Installation

To install, run the following 

`pip install nuclear-rose`

## Usage

The primary class is `ReducedBasisEmulator`. To create an instance, minimally, an instance of the `Interaction` class, a set of training points, the energy, and angular momentum need to be specified. 

For a full set of examples walking through emulation and calibration, check the [tutorials](docs/tutorials/) directory.

You can also check out the [documentation page](https://reduced-order-scattering-emulator.readthedocs.io/en/latest/)

You could even check the other [BAND softwares](https://bandframework.github.io/software/) and start combining them to write all your Bayesian papers!

