"""Initial populations and scales of the three-level model.

:func:`populations` gives the level populations and population differences
from the ratios ``r1 = n3/n1`` and ``r2 = n3/n2``; :func:`scaling` turns
the physical values of the two transitions into the solver's ``d`` and
``omega`` and the time, field, intensity and pump scales.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from mbsolver.constants import C, EPSILON_0, H_BAR, PI

__all__ = ["populations", "scaling"]


def populations(r1: float, r2: float) -> Dict[str, float]:
    """Level populations from ``r1 = n3/n1`` and ``r2 = n3/n2`` (``n1 + n2 + n3 = 1``).

    Returns:
        Mapping with ``n1``, ``n2``, ``n3``, the differences ``n31 = n3 - n1``
        and ``n32 = n3 - n2``, and the largest coherences
        ``r31 = 2√(n3 n1)``, ``r32 = 2√(n3 n2)``, ``r21 = 2√(n2 n1)``
        (the solver's initial coherences are ``c_ij`` times these).
    """
    if r1 < 0 or r2 < 0:
        raise ValueError("r1 and r2 must not be negative")
    c = r1 + r2 + r1 * r2
    if c == 0.0:
        return {"n1": 0.5, "n2": 0.5, "n3": 0.0, "n31": -0.5, "n32": -0.5,
                "r31": 0.0, "r32": 0.0, "r21": 1.0}
    return {
        "n1": r2 / c, "n2": r1 / c, "n3": r1 * r2 / c,
        "n31": r2 * (r1 - 1) / c, "n32": r1 * (r2 - 1) / c,
        "r31": 2.0 * r2 * math.sqrt(r1) / c,
        "r32": 2.0 * r1 * math.sqrt(r2) / c,
        "r21": 2.0 * math.sqrt(r1 * r2) / c,
    }


def scaling(nu31: float, nu32: float, d31: float, d32: float, l: float, n0: float,
            gamma: Optional[float] = None) -> Dict[str, float]:
    """Compute the solver parameters and scales from the physical values (SI units).

    The time unit is ``T0 = 2 ε0 ħ / (n0 L k32 |d32|²)``. This equals
    ``8π / (3 Γ n0 L λ32²)`` with ``Γ`` the Einstein A of the 3→2 line
    computed from ``d32``; ``gamma`` replaces that ``Γ`` by a given rate.

    Args:
        nu31, nu32: Frequencies of the 3→1 and 3→2 transitions in Hz.
        d31, d32: Dipole moments in C·m (``mbsolver.constants.DEBYE`` converts from debye).
        l: Length of the medium L in m.
        n0: Density scale n₀ in m⁻³.
        gamma: Decay rate Γ of the 3→2 line in s⁻¹; by default the Einstein A from ``d32``.

    Returns:
        Mapping:

        - ``d`` = |d31/d32| and ``omega`` = k31/k32 (the solver's ``d`` and ``omega``),
          ``eta`` = ω d² (the coupling of the 3→1 field),
        - ``k31``, ``k32``, ``lambda31``, ``lambda32``: wave numbers and wavelengths,
        - ``gamma``: the decay rate used, ``t0``: time unit T₀ in s,
        - ``e0`` = ħ / (T₀ d32): field scale in V/m,
        - ``i0`` = ½ ε₀ c E₀²: intensity scale in W/m², ``i0_jy`` = 10²⁶ ``i0``,
        - ``lambda0`` = n₀ / T₀: pump scale in m⁻³ s⁻¹.
    """
    k31 = 2 * PI * nu31 / C
    k32 = 2 * PI * nu32 / C
    if gamma is None:
        gamma = k32**3 * d32**2 / (3 * PI * EPSILON_0 * H_BAR)
        t0 = 2.0 * EPSILON_0 * H_BAR / (n0 * l * k32 * d32**2)
    else:
        t0 = 2.0 * k32**2 / (3.0 * PI * gamma * n0 * l)
    d = abs(d31 / d32)
    omega = k31 / k32
    e0 = H_BAR / (t0 * d32)
    i0 = 0.5 * EPSILON_0 * C * e0**2
    return {
        "d": d, "omega": omega, "eta": omega * d**2,
        "k31": k31, "k32": k32, "lambda31": C / nu31, "lambda32": C / nu32,
        "gamma": gamma, "t0": t0, "e0": e0, "i0": i0, "i0_jy": i0 * 1e26,
        "lambda0": n0 / t0,
    }
