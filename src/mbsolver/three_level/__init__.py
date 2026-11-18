"""Three-level Maxwell-Bloch model: Λ, V and ladder configurations.

Levels are ordered by energy, ``E1 < E2 < E3``. Two of the three transitions
are dipole allowed and carry fields; the third is forbidden:

=========  ===================  =========
config     allowed              forbidden
=========  ===================  =========
"lambda"   3→1, 3→2             2↔1
"v"        2→1, 3→1             3↔2
"ladder"   2→1, 3→2             3↔1
=========  ===================  =========

Dimensionless equations along a 1-D medium ``0 <= z <= 1`` (all quantities
real; fields are Rabi frequencies, ``W_ij = ½ E_ij R_ij``, ``n_ij = n_i - n_j``)::

    dn1/dt =  W21 + W31,   dn2/dt = W32 - W21,   dn3/dt = -W31 - W32
    dR21/dt = E21 n21 + ½ (E31 R32 + E32 R31) - (R21 - R21⁰) / T2_21
    dR31/dt = E31 n31 + ½ (E21 R32 - E32 R21) - (R31 - R31⁰) / T2_31
    dR32/dt = E32 n32 - ½ (E21 R31 + E31 R21) - (R32 - R32⁰) / T2_32
    dE_ij/dz = η_ij R_ij

The population differences of the two allowed transitions relax with their
``T1`` and are pumped; the field of the forbidden transition is zero. The
derivation is in ``docs/three_level.pdf``.

The equations were derived and checked with Toktam Rashidi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from mbsolver.grids import grids_1d
from mbsolver.profiles import Pulse, Steps, evaluate
from mbsolver.three_level._kernel_1d import LADDER, LAMBDA, V, solve_kernel
from mbsolver.three_level.scales import scaling

__all__ = ["CONFIGS", "Result", "solve_1d", "scaling"]

#: Allowed transitions of each configuration, as ``(a, b)``.
CONFIGS = {"lambda": ("31", "32"), "v": ("21", "31"), "ladder": ("21", "32")}
_CODES = {"lambda": LAMBDA, "v": V, "ladder": LADDER}
_PAIRS = ("21", "31", "32")


@dataclass(frozen=True)
class Result:
    """Output of :func:`solve_1d`.

    Arrays on the stored planes have shape ``(n_planes, nt)``. Inputs that
    belong only to the allowed transitions (``pump``, ``field_in``, ``t1``)
    are ``None`` for the forbidden one.

    Attributes:
        config: ``"lambda"``, ``"v"`` or ``"ladder"``.
        time: Time grid, shape ``(nt,)``.
        z: Positions of the stored planes, shape ``(n_planes,)``, from 0 to 1.
        e21, e31, e32: Fields (Rabi frequencies); zeros for the forbidden transition.
        n21, n31, n32: Population differences ``n2 - n1``, ``n3 - n1``, ``n3 - n2``.
        r21, r31, r32: Coherences.
        pump21, pump31, pump32: Pumps on the time grid.
        field21_in, field31_in, field32_in: Fields at ``z = 0``.
        t1_21, t1_31, t1_32: Population relaxation times on the time grid.
        t2_21, t2_31, t2_32: Coherence relaxation times on the time grid.
    """

    config: str
    time: np.ndarray
    z: np.ndarray
    e21: np.ndarray
    e31: np.ndarray
    e32: np.ndarray
    n21: np.ndarray
    n31: np.ndarray
    n32: np.ndarray
    r21: np.ndarray
    r31: np.ndarray
    r32: np.ndarray
    pump21: Optional[np.ndarray]
    pump31: Optional[np.ndarray]
    pump32: Optional[np.ndarray]
    field21_in: Optional[np.ndarray]
    field31_in: Optional[np.ndarray]
    field32_in: Optional[np.ndarray]
    t1_21: Optional[np.ndarray]
    t1_31: Optional[np.ndarray]
    t1_32: Optional[np.ndarray]
    t2_21: np.ndarray
    t2_31: np.ndarray
    t2_32: np.ndarray


Profile = Union[None, float, Pulse, Steps]
Time = Union[None, float, Steps]


def _sum(profile, t):
    """A profile or a list of profiles (summed) on ``t``."""
    items = profile if isinstance(profile, (list, tuple)) else [profile]
    total = np.zeros_like(t)
    for p in items:
        total = total + evaluate(p, t)
    return total


def _times(name, value, t):
    """A relaxation time on ``t``, checked to be positive."""
    out = evaluate(value, t)
    if np.any(out <= 0):
        raise ValueError(f"{name} must be positive")
    return out


def solve_1d(
        *,
        config: str,
        t_max: float,
        nt: int,
        nz: int,
        n_init: Tuple[float, float, float],
        eta21: Optional[float] = None,
        eta31: Optional[float] = None,
        eta32: Optional[float] = None,
        c21: float = 0.0,
        c31: float = 0.0,
        c32: float = 0.0,
        t1_21: Time = None,
        t1_31: Time = None,
        t1_32: Time = None,
        t2_21: Time = None,
        t2_31: Time = None,
        t2_32: Time = None,
        pump21: Union[Profile, Sequence[Profile]] = None,
        pump31: Union[Profile, Sequence[Profile]] = None,
        pump32: Union[Profile, Sequence[Profile]] = None,
        field21_in: Profile = None,
        field31_in: Profile = None,
        field32_in: Profile = None,
        n_planes: int = 2,
) -> Result:
    """Solve the three-level Maxwell-Bloch equations in one dimension.

    Parameters named ``…ij`` belong to the transition ``i→j``. ``eta``,
    ``t1``, ``pump`` and ``field_in`` are given only for the two allowed
    transitions of ``config`` (``eta`` and ``t1`` are required there);
    ``t2`` is required for the two allowed coherences and defaults to
    ``inf`` for the forbidden one.

    Args:
        config: ``"lambda"`` (allowed 3→1, 3→2), ``"v"`` (2→1, 3→1) or ``"ladder"`` (2→1, 3→2).
        t_max: Duration; the time grid runs from 0 to ``t_max``.
        nt: Number of time points (>= 2).
        nz: Number of z points (>= 2).
        n_init: Initial populations ``(n1, n2, n3)``, not negative, summing to 1.
        eta21, eta31, eta32: Couplings ``η_ij = T0 / T_R`` of the allowed transitions (>= 0).
        c21, c31, c32: Initial coherences as fractions of their largest values ``2√(n_i n_j)``.
        t1_21, t1_31, t1_32: Relaxation times of the population differences, numbers or
            :class:`Steps` (> 0; ``inf`` = none).
        t2_21, t2_31, t2_32: Relaxation times of the coherences, the same way.
        pump21, pump31, pump32: Pumps of the population differences: a :class:`Pulse`,
            a number, or a list of them (summed).
        field21_in, field31_in, field32_in: Fields entering at ``z = 0``: a :class:`Pulse`,
            a number or ``None``.
        n_planes: Number of evenly spaced z planes to return, from 0 to 1 (<= ``nz``).

    Returns:
        A :class:`Result`.

    Example:
        >>> res = solve_1d(config="ladder", t_max=200, nt=2001, nz=101,
        ...                n_init=(0.05, 0.05, 0.9), eta21=10.0, eta32=10.0,
        ...                c21=1e-3, c32=1e-3, t1_21=math.inf, t1_32=math.inf,
        ...                t2_21=math.inf, t2_32=math.inf)
        >>> lower_step_exit = res.e21[-1]
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {sorted(CONFIGS)}, got {config!r}")
    nt, nz, n_planes = int(nt), int(nz), int(n_planes)
    if nt < 2 or nz < 2:
        raise ValueError("nt and nz must be at least 2")
    if not 1 <= n_planes <= nz:
        raise ValueError(f"n_planes must be between 1 and nz ({nz}), got {n_planes}")
    if not t_max > 0:
        raise ValueError("t_max must be positive")

    allowed = CONFIGS[config]
    forbidden = next(p for p in _PAIRS if p not in allowed)
    given = {
        "eta": {"21": eta21, "31": eta31, "32": eta32},
        "t1_": {"21": t1_21, "31": t1_31, "32": t1_32},
        "pump": {"21": pump21, "31": pump31, "32": pump32},
        "field_in": {"21": field21_in, "31": field31_in, "32": field32_in},
    }
    for prefix, values in given.items():
        if values[forbidden] is not None:
            name = f"field{forbidden}_in" if prefix == "field_in" else f"{prefix}{forbidden}"
            raise ValueError(f"{name} is not allowed for config={config!r} "
                             f"({forbidden[0]}↔{forbidden[1]} is forbidden)")
    t2s = {"21": t2_21, "31": t2_31, "32": t2_32}
    for p in allowed:
        for prefix, values in (("eta", given["eta"]), ("t1_", given["t1_"]), ("t2_", t2s)):
            if values[p] is None:
                raise ValueError(f"{prefix}{p} is required for config={config!r}")
        if not given["eta"][p] >= 0:
            raise ValueError(f"eta{p} must not be negative")
    if t2s[forbidden] is None:
        t2s[forbidden] = math.inf

    n = np.asarray(n_init, dtype=np.float64)
    if n.shape != (3,) or np.any(n < 0) or abs(n.sum() - 1.0) > 1e-12:
        raise ValueError("n_init must be three non-negative populations (n1, n2, n3) summing to 1")
    n1, n2, n3 = float(n[0]), float(n[1]), float(n[2])
    diff = {"21": n2 - n1, "31": n3 - n1, "32": n3 - n2}
    pops = {"1": n1, "2": n2, "3": n3}
    c = {"21": c21, "31": c31, "32": c32}
    r0 = {p: c[p] * 2.0 * math.sqrt(pops[p[0]] * pops[p[1]]) for p in _PAIRS}
    a, b = allowed
    init = np.array([diff[a], diff[b], r0["21"], r0["31"], r0["32"]], dtype=np.float64)

    t, dt, z, dz = grids_1d(float(t_max), nt, nz)
    pumps = {p: (_sum(given["pump"][p], t) if p in allowed else None) for p in _PAIRS}
    fields = {p: (evaluate(given["field_in"][p], t) if p in allowed else None) for p in _PAIRS}
    t1 = {p: (_times(f"t1_{p}", given["t1_"][p], t) if p in allowed else None) for p in _PAIRS}
    t2 = {p: _times(f"t2_{p}", t2s[p], t) for p in _PAIRS}

    zeros = np.zeros_like(t)
    field_in = np.array([fields[p] if p in allowed else zeros for p in _PAIRS])
    t2_arr = np.array([t2[p] for p in _PAIRS])
    eta = np.array([float(given["eta"][p]) if p in allowed else 0.0 for p in _PAIRS])

    v, E = solve_kernel(_CODES[config], t, dt, z, dz, n_planes, init, field_in,
                        pumps[a], pumps[b], t1[a], t1[b], t2_arr, eta)

    na, nb = v[:, 0], v[:, 1]
    if config == "lambda":
        n31, n32 = na, nb
        n21 = n31 - n32
    elif config == "v":
        n21, n31 = na, nb
        n32 = n31 - n21
    else:
        n21, n32 = na, nb
        n31 = n21 + n32

    planes = z[np.rint(np.linspace(0.0, 1.0, n_planes) * (nz - 1)).astype(np.int64)]
    return Result(
        config=config, time=t, z=planes,
        e21=E[:, 0], e31=E[:, 1], e32=E[:, 2],
        n21=n21, n31=n31, n32=n32,
        r21=v[:, 2], r31=v[:, 3], r32=v[:, 4],
        pump21=pumps["21"], pump31=pumps["31"], pump32=pumps["32"],
        field21_in=fields["21"], field31_in=fields["31"], field32_in=fields["32"],
        t1_21=t1["21"], t1_31=t1["31"], t1_32=t1["32"],
        t2_21=t2["21"], t2_31=t2["31"], t2_32=t2["32"],
    )
