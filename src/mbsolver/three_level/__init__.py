"""Λ-type three-level Maxwell-Bloch model.

Levels 1 and 2 lie below level 3; the transitions 3→1 and 3→2 carry the
fields ``E31`` and ``E32``. Dimensionless equations along a 1-D medium
``0 <= z <= 1`` (all quantities real)::

    dn31/dt = -(E31 R31 + ½ E32 R32) - (n31 - n31⁰) / T1_31 + Λ31(t)
    dn32/dt = -(½ E31 R31 + E32 R32) - (n32 - n32⁰) / T1_32 + Λ32(t)
    dR31/dt = E31 n31 - ½ E32 R21 - (R31 - R31⁰) / T2_31
    dR32/dt = E32 n32 - ½ E31 R21 - (R32 - R32⁰) / T2_32
    dR21/dt = ½ (E31 R32 + E32 R31) - (R21 - R21⁰) / T2_21
    dE31/dz = η R31,   dE32/dz = R32,   η = ω d²

with ``d = |d31/d32|``, ``ω = k31/k32`` and the fields at ``z = 0`` given.
The initial state comes from ``r1 = n3/n1``, ``r2 = n3/n2`` and the
coherence fractions ``c_ij``: ``R_ij⁰ = c_ij · 2√(n_i n_j)``.

The equations were derived and checked with Toktam Rashidi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from mbsolver.grids import grids_1d
from mbsolver.profiles import Pulse, Steps, evaluate
from mbsolver.three_level._kernel_1d import solve_kernel
from mbsolver.three_level.sample import populations, scaling

__all__ = ["Result", "solve_1d", "populations", "scaling"]


@dataclass(frozen=True)
class Result:
    """Output of :func:`solve_1d`.

    Arrays on the stored planes have shape ``(n_planes, nt)``.

    Attributes:
        time: Time grid, shape ``(nt,)``.
        z: Positions of the stored planes, shape ``(n_planes,)``, from 0 to 1.
        e31, e32: Fields.
        intensity31, intensity32: ``(E31/d)²`` and ``E32²``, both in units of the 3→2 scale.
        n31, n32: Population differences ``n3 - n1`` and ``n3 - n2``.
        r31, r32, r21: Coherences.
        pump31, pump32: Pumps ``Λ31(t)``, ``Λ32(t)``.
        field31_in, field32_in: Fields at ``z = 0``.
        t1_31, t1_32, t2_31, t2_32, t2_21: Relaxation times on the time grid.
    """

    time: np.ndarray
    z: np.ndarray
    e31: np.ndarray
    e32: np.ndarray
    intensity31: np.ndarray
    intensity32: np.ndarray
    n31: np.ndarray
    n32: np.ndarray
    r31: np.ndarray
    r32: np.ndarray
    r21: np.ndarray
    pump31: np.ndarray
    pump32: np.ndarray
    field31_in: np.ndarray
    field32_in: np.ndarray
    t1_31: np.ndarray
    t1_32: np.ndarray
    t2_31: np.ndarray
    t2_32: np.ndarray
    t2_21: np.ndarray


Profile = Union[None, float, Pulse, Steps]


def _sum(profile, t):
    """A profile or a list of profiles (summed) on ``t``."""
    items = profile if isinstance(profile, (list, tuple)) else [profile]
    total = np.zeros_like(t)
    for p in items:
        total = total + evaluate(p, t)
    return total


def solve_1d(
        *,
        t_max: float,
        nt: int,
        nz: int,
        d: float,
        omega: float,
        r1: float,
        r2: float,
        c31: float,
        c32: float,
        t1_31: Union[float, Steps],
        t1_32: Union[float, Steps],
        t2_31: Union[float, Steps],
        t2_32: Union[float, Steps],
        t2_21: Union[float, Steps] = math.inf,
        c21: float = 0.0,
        pump31: Union[Profile, Sequence[Profile]] = None,
        pump32: Union[Profile, Sequence[Profile]] = None,
        field31_in: Profile = None,
        field32_in: Profile = None,
        n_planes: int = 2,
) -> Result:
    """Solve the Λ-type three-level Maxwell-Bloch equations in one dimension.

    Args:
        t_max: Duration; the time grid runs from 0 to ``t_max``.
        nt: Number of time points (>= 2).
        nz: Number of z points (>= 2).
        d: Dipole ratio ``|d31/d32|`` (> 0).
        omega: Wave-number ratio ``k31/k32`` (> 0); the 3→1 coupling is ``η = ω d²``.
        r1, r2: Population ratios ``n3/n1`` and ``n3/n2`` (>= 0).
        c31, c32, c21: Initial coherences as fractions of their largest values.
        t1_31, t1_32: Population relaxation times, numbers or :class:`Steps` (> 0; ``inf`` = none).
        t2_31, t2_32, t2_21: Coherence relaxation times, the same way.
        pump31, pump32: Pumps of ``n31`` and ``n32``: a :class:`Pulse`, a number,
            or a list of them (summed).
        field31_in, field32_in: Fields entering at ``z = 0``: a :class:`Pulse`, a number or ``None``.
        n_planes: Number of evenly spaced z planes to return, from 0 to 1 (<= ``nz``).

    Returns:
        A :class:`Result`.

    Example:
        >>> res = solve_1d(t_max=2500, nt=4000, nz=400, d=0.875, omega=35.95,
        ...                r1=0.94, r2=2.33e19, c31=1e-10, c32=1.0,
        ...                t1_31=31.3, t1_32=313, t2_31=2.96, t2_32=29.6,
        ...                pump32=Pulse(a=5.63e-3, x0=795.5, w=154.8))
        >>> exit_intensity = res.intensity32[-1]
    """
    nt, nz, n_planes = int(nt), int(nz), int(n_planes)
    if nt < 2 or nz < 2:
        raise ValueError("nt and nz must be at least 2")
    if not 1 <= n_planes <= nz:
        raise ValueError(f"n_planes must be between 1 and nz ({nz}), got {n_planes}")
    if not t_max > 0:
        raise ValueError("t_max must be positive")
    if not (d > 0 and omega > 0):
        raise ValueError("d and omega must be positive")

    pops = populations(float(r1), float(r2))
    init = np.array([pops["n31"], pops["n32"], c31 * pops["r31"], c32 * pops["r32"],
                     c21 * pops["r21"]], dtype=np.float64)

    t, dt, z, dz = grids_1d(float(t_max), nt, nz)
    pump31_t, pump32_t = _sum(pump31, t), _sum(pump32, t)
    field31_t, field32_t = evaluate(field31_in, t), evaluate(field32_in, t)
    times = {}
    for name, value in (("t1_31", t1_31), ("t1_32", t1_32), ("t2_31", t2_31),
                        ("t2_32", t2_32), ("t2_21", t2_21)):
        times[name] = evaluate(value, t)
        if np.any(times[name] <= 0):
            raise ValueError(f"{name} must be positive")

    eta = float(omega) * float(d) * float(d)
    v, E = solve_kernel(t, dt, z, dz, n_planes, init, field31_t, field32_t,
                        pump31_t, pump32_t, times["t1_31"], times["t1_32"],
                        times["t2_31"], times["t2_32"], times["t2_21"], eta)

    e31, e32 = E[:, 0], E[:, 1]
    planes = z[np.rint(np.linspace(0.0, 1.0, n_planes) * (nz - 1)).astype(np.int64)]
    return Result(
        time=t, z=planes, e31=e31, e32=e32,
        intensity31=e31 ** 2 / float(d) ** 2, intensity32=e32 ** 2,
        n31=v[:, 0], n32=v[:, 1], r31=v[:, 2], r32=v[:, 3], r21=v[:, 4],
        pump31=pump31_t, pump32=pump32_t, field31_in=field31_t, field32_in=field32_t,
        **times,
    )
