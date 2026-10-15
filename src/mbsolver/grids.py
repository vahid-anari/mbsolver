"""Simulation grids."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline="always", nogil=True, cache=True)
def grids_1d(t_max: float, nt: int, nz: int):
    """Return ``(t, dt, z, dz)``: ``nt`` times from 0 to ``t_max``, ``nz`` positions from 0 to 1."""
    t = np.linspace(0.0, t_max, nt)
    dt = t_max / (nt - 1)
    z = np.linspace(0.0, 1.0, nz)
    dz = 1.0 / (nz - 1)
    return t, dt, z, dz
