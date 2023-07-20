'''Minnesota potentials for testing.
'''

import numpy as np
from .interaction import Interaction

NUCLEON_MASS = 939.565 # neutron mass (MeV)
MU_NN = NUCLEON_MASS / 2 # reduced mass of the NN system (MeV)

def mn_potential(r, args):
    '''
    Minnesota potential
    '''
    v_0r, v_0s = args
    return v_0r * np.exp(-1.487*r**2) + v_0s*np.exp(-0.465*r**2)

# Stored instances of the Minnesota interaction for testing.
# Fixed at E_{c.m.} = 50 MeV.
MN_Potential = Interaction(
    mn_potential,
    2,
    MU_NN,
    50,
    0
)

def complex_mn_potential(r, args):
    vr, vi = args
    return mn_potential(r, [vr, 1j*vi])


Complex_MN_Potential = Interaction(
    complex_mn_potential,
    2,
    MU_NN,
    50,
    0,
    is_complex = True
)
