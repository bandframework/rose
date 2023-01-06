# ROSE

**R**educed-**O**rder **S**cattering **E**mulator

ROSE makes it easy to build and train a scattering emulator.

The primary class is `ReducedBasisEmulator`. To create an instance, minimally, an instance of the `Interaction` class, a set of training points, the energy, and angular momentum need to be specified. For example,

```
import rose

energy = 50 # MeV
ell = 0 # S waves

# The we are varying two parameters of the Minnesota potential, so the training
# space is an array of 2-component arrays
training_points = np.array([
    [119.51219512195122, -14.634146341463415],
    [139.02439024390245, -4.878048780487805],
    [158.53658536585365, -48.78048780487805],
    [178.0487804878049, -117.07317073170732],
    [197.5609756097561, -131.70731707317074],
    [217.0731707317073, -126.82926829268293],
    [236.58536585365854, -82.92682926829268],
    [256.0975609756098, -175.609756097561],
    [275.609756097561, -19.51219512195122],
    [295.1219512195122, -170.73170731707316]
])

# The Minnesota potential has already been hard-coded in ROSE as
# rose.MN_Potential.
rbe = rose.ReducedBasisEmulator(
    rose.MN_Potential,
    training_points,
    energy,
    ell
)

# Now, to get a the wave function or phase shift at a new point in parameter
# space, we simply call...
theta = np.array([200,-91.85])
phi = rbe.emulate_wave_function(theta)
# or...
delta = rbe.emulate_phase_shift(theta)
```
