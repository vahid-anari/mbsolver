"""Physical constants in SI units, used by :mod:`mbsolver.sample`."""

import math

PI: float = math.pi
"""Circle constant."""

H_BAR: float = 6.62607015e-34 / (2.0 * PI)
"""Reduced Planck constant in joule-seconds."""

C: float = 2.99792458e8
"""Speed of light in meters per second."""

MU_0: float = 4.0 * PI * 1e-7
"""Vacuum permeability in henries per meter."""

EPSILON_0: float = 1.0 / (MU_0 * C**2)
"""Vacuum permittivity in farads per meter."""

DEBYE: float = 1e-21 / C
"""One debye in coulomb-meters."""
