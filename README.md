[![Python package](https://github.com/bandframework/rose/actions/workflows/python-package.yml/badge.svg)](https://github.com/bandframework/rose/actions/workflows/python-package.yml)

# ROSE - The **R**educed-**O**rder **S**cattering **E**mulator

ROSE makes it easy to build and train a scattering emulator. ROSE enables aspiring graduate students and long-suffering postdocs to emulate nuclear scattering observables with optical potentials, trading negligible amounts of accuracy for orders-of-magnitude gains in speed.

For any bug reports or feature requests, please make use of the Github issues tab on the repository. We also welcome all pull requests for software, documentation, and user-contributed tutorials! 

## Installation

ROSE is hosted at [pypi.org/project/nuclear-rose/](https://pypi.org/project/nuclear-rose/). To install as a user, run the following 

`pip install nuclear-rose`.

To install as a developer, clone the repository and run

`pip install -e .`

from within the project root directory.

## Usage

To emulate an interaction, you will make an `Interaction` class, or something similar. Then you will typically make a `ScatteringAmplitudeEmulator`, which will train an emulator to emulate elastic cross sections. 

For a full set of examples walking through emulation and calibration, check the [tutorials](docs/tutorials/) directory.

## Documentation

You can also check out the [documentation page](https://reduced-order-scattering-emulator.readthedocs.io/en/latest/).

## More software

You could even check the other [BAND softwares](https://bandframework.github.io/software/) and start combining them to write all your Bayesian papers!

## Citation

ROSE, and the theory behind it, were introduced in [this publication](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.109.044612). If you use ROSE, please cite it like so:

```tex
@article{PhysRevC.109.044612,
  title = {ROSE: A reduced-order scattering emulator for optical models},
  author = {Odell, D. and Giuliani, P. and Beyer, K. and Catacora-Rios, M. and Chan, M. Y.-H. and Bonilla, E. and Furnstahl, R. J. and Godbey, K. and Nunes, F. M.},
  journal = {Phys. Rev. C},
  volume = {109},
  issue = {4},
  pages = {044612},
  numpages = {17},
  year = {2024},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevC.109.044612},
  url = {https://link.aps.org/doi/10.1103/PhysRevC.109.044612}
}
```

Additionally, as ROSE is part of the BAND software framework, please consider citing the [BAND Manifesto](https://iopscience.iop.org/article/10.1088/1361-6471/abf1df).
