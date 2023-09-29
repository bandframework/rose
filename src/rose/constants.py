import numpy as np

HBARC = 197.326
DEFAULT_RHO_MESH = np.linspace(1e-6, 8*np.pi, 2000)
DEFAULT_ANGLE_MESH = np.linspace(0.01, np.pi, 10)
ALPHA = 0.0072973525693 # 1.0/137.056
MASS_PION = np.sqrt(1/2) # 2.0 per Thomspon&Nuñes below Eq. (4.3.10) 1/fm
AMU = 931.494102  # MeV/c^2, Particle Data Group
MASS_N = 1.008665 * AMU  # MeV/c^2 PDG
MASS_P = 1.007276 * AMU  # MeV/c^2 PDG
