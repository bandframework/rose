# Interactions

In general, `rose` supports local, complex interactions. While we expect that
most users will take advantage of the hard-coded Koning-Delaroche potential, we
have always written `rose` with the expectation that many users will want to
define their own interactions. The classes below make that process more
convenient.

The most basic interaction is affectionately and creatively referred to as:
`Interaction`. It supports fixed-energy interactions whose parameter dependece
is affine. The corresponding class, `InteractionSpace`, generates a list of
$\ell$-specific `Interactions`.

For fixed-energy interactions for which the parameter dependence is non-affine,
we have `InteractionEIM` and `InteractionEIMSpace`. The classes leverage the
Empirical Interpolation Method (EIM) to render that which was non-affine affine.

For non-affine, **energy-dependent** interactions, we have
`EnergizedInteractionEIM` and `EnergizedInteractionEIMSpace`. `rose` works with
the energy-scaled Schrödinger equation, so one might think that the scaled
interaction is linear in $1/E$. However, because we also work in the
dimensionless space, $s\equiv kr$, the dependence is more complex. We again rely
on EIM to capture these complexities. (You don't need to know all of that. We
just wanted you to impress you.)

## Affine, Fixed-Energy Interactions

::: rose.interaction

## Non-Affine, Fixed-Energy Interactions

::: rose.interaction_eim

## Non-Affine, Energy-Emulated Interactions

::: rose.energized_interaction_eim

## Spin-Orbit

::: rose.spin_orbit

## Koning-Delaroche

::: rose.koning_delaroche
