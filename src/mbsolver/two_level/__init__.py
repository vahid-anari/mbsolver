"""Two-level Maxwell-Bloch model.

Dimensionless equations for population inversion ``w``, coherence ``R`` and
field ``A`` along a 1-D medium ``0 <= z <= 1``::

    dw/dt = A R - (w - w0) / T1 + Λ(t)
    dR/dt = -A w - (R - R0) / T2
    dA/dz = -η R

with the field ``A(t, z=0)`` given at the entrance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from mbsolver.grids import grids_1d
from mbsolver.profiles import Pulse, Steps, evaluate
from mbsolver.two_level._kernel_1d import solve_kernel

__all__ = ["Result", "solve_1d"]


@dataclass(frozen=True)
class Result:
    """Output of :func:`solve_1d`.

    Attributes:
        time: Time grid, shape ``(nt,)``.
        z: Positions of the stored planes, shape ``(n_planes,)``, from 0 to 1.
        intensity: ``A²`` on the stored planes, shape ``(n_planes, nt)``.
        w: Population inversion on the stored planes, shape ``(n_planes, nt)``.
        pump: Total pump ``Λ(t)``.
        field_in: Boundary field ``A(t, z=0)``.
        t1, t2: Relaxation times on the time grid.
    """

    time: np.ndarray
    z: np.ndarray
    intensity: np.ndarray
    w: np.ndarray
    pump: np.ndarray
    field_in: np.ndarray
    t1: np.ndarray
    t2: np.ndarray


Profile = Union[None, float, Pulse, Steps]


def solve_1d(
        *,
        t_max: float,
        nt: int,
        nz: int,
        eta: float,
        t1: Union[float, Steps],
        t2: Union[float, Steps],
        theta0: Optional[float] = None,
        w0: Optional[float] = None,
        R0: Optional[float] = None,
        pump: Union[Profile, Sequence[Profile]] = None,
        field_in: Profile = None,
        n_planes: int = 2,
) -> Result:
    """Solve the two-level Maxwell-Bloch equations in one dimension.

    Args:
        t_max: Duration; the time grid runs from 0 to ``t_max``.
        nt: Number of time points (>= 2).
        nz: Number of z points (>= 2).
        eta: Coupling coefficient ``η``.
        t1: Population relaxation time, a number or :class:`Steps` (> 0; ``inf`` = none).
        t2: Coherence relaxation time, a number or :class:`Steps` (> 0; ``inf`` = none).
        theta0: Initial Bloch angle: ``w0 = cos θ0``, ``R0 = sin θ0``.
        w0, R0: Initial population and coherence, instead of ``theta0``.
            The medium also relaxes towards these values.
        pump: Pump ``Λ(t)``: a :class:`Pulse`, a number, or a list of them (summed).
        field_in: Field ``A(t)`` entering at ``z = 0``: a :class:`Pulse`, a number or ``None``.
        n_planes: Number of evenly spaced z planes to return, from 0 to 1 (<= ``nz``).

    Returns:
        A :class:`Result`.

    Example:
        >>> res = solve_1d(t_max=200, nt=4000, nz=200, eta=5.0, theta0=0.01,
        ...                t1=1.0, t2=0.5, pump=Pulse(a=0.1, x0=50, w=10))
        >>> exit_intensity = res.intensity[-1]
    """
    nt, nz, n_planes = int(nt), int(nz), int(n_planes)
    if nt < 2 or nz < 2:
        raise ValueError("nt and nz must be at least 2")
    if not 1 <= n_planes <= nz:
        raise ValueError(f"n_planes must be between 1 and nz ({nz}), got {n_planes}")
    if not t_max > 0:
        raise ValueError("t_max must be positive")

    if theta0 is not None:
        if w0 is not None or R0 is not None:
            raise ValueError("give either theta0, or w0 and R0, not both")
        w_init, R_init = math.cos(theta0), math.sin(theta0)
    elif w0 is not None and R0 is not None:
        w_init, R_init = float(w0), float(R0)
    else:
        raise ValueError("give the initial state as theta0, or as w0 and R0")

    t, dt, z, dz = grids_1d(float(t_max), nt, nz)

    pumps = pump if isinstance(pump, (list, tuple)) else [pump]
    pump_t = np.zeros_like(t)
    for p in pumps:
        pump_t = pump_t + evaluate(p, t)
    field_t = evaluate(field_in, t)
    t1_t = evaluate(t1, t)
    t2_t = evaluate(t2, t)
    for name, arr in (("t1", t1_t), ("t2", t2_t)):
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be positive")

    w_z = np.full_like(z, w_init)
    R_z = np.full_like(z, R_init)
    w, intensity = solve_kernel(t, dt, z, dz, n_planes, w_z, R_z, field_t, pump_t,
                                t1_t, t2_t, float(eta))

    planes = z[np.rint(np.linspace(0.0, 1.0, n_planes) * (nz - 1)).astype(np.int64)]
    return Result(time=t, z=planes, intensity=intensity, w=w, pump=pump_t,
                  field_in=field_t, t1=t1_t, t2=t2_t)
