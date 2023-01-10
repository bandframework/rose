Overview
========

In principle, the user should be able to train, build, and store a reduced-basis
emulator (RBE) using the `ReducedBasisEmulator` and `Interaction` classes.
That's it.

`ReducedBasisEmulator`
----------------------

The `ReducedBasisEmulator` class is the primary object. To create an instance,
one needs minimally an `Interaction`, a set of training points, an energy, and
an angular momentum.

`Interaction`
-------------

The `Interaction` class represents the local, 2-body potential that
characterizes the physics of the problem.

$$
V(r,\alpha)
$$

`InteractionEIM`
````````````````

The `InteractionEIM` class is a subclass of `Interaction`. It allows one to
treat all of the interaction parameters as affine (whether they are or not).