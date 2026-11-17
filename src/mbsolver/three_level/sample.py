"""Physical scales of the three-level model.

:func:`scaling` turns the physical values of the transitions into the solver's
couplings ``eta_ij`` and the field, intensity and pump scales.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from mbsolver.constants import C, EPSILON_0, H_BAR, PI

__all__ = ["scaling"]


def scaling(*, t0: float, l: float, n0: float,
            nu21: Optional[float] = None, d21: Optional[float] = None, gamma21: Optional[float] = None,
            nu31: Optional[float] = None, d31: Optional[float] = None, gamma31: Optional[float] = None,
            nu32: Optional[float] = None, d32: Optional[float] = None, gamma32: Optional[float] = None,
            ) -> Dict[str, float]:
    """Compute the solver couplings and the scales from physical values (SI units).

    Give, for each transition ``i→j`` of interest, its frequency ``nu_ij`` and
    either its dipole moment ``d_ij`` or its Einstein coefficient ``gamma_ij``
    (they are related by ``Γ = k³ |d|² / (3π ε0 ħ)``). The cooperation time
    of a transition is ``T_R = 2 ε0 ħ / (n0 L k |d|²) = 8π / (3 Γ n0 L λ²)``
    and its coupling ``η = T0 / T_R``.

    Args:
        t0: Time unit T₀ in s (free; ``t0 = t_r_ij`` makes ``eta_ij = 1``).
        l: Length of the medium L in m.
        n0: Number density of the molecules in m⁻³.
        nu21, nu31, nu32: Transition frequencies in Hz.
        d21, d31, d32: Dipole moments in C·m (``mbsolver.constants.DEBYE`` converts from debye).
        gamma21, gamma31, gamma32: Einstein coefficients in s⁻¹ (instead of the dipole moments).

    Returns:
        Mapping with, for each given transition ``ij``: ``eta_ij``, ``t_r_ij``
        (s), ``gamma_ij`` (s⁻¹), ``d_ij`` (C·m), ``k_ij`` (m⁻¹), ``lambda_ij``
        (m), ``e0_ij`` = ħ / (T₀ d) (V/m) and ``i0_ij`` = ½ ε₀ c E₀² (W/m²);
        and ``lambda0`` = n₀ / T₀, the pump scale in m⁻³ s⁻¹.

    Example:
        >>> s = scaling(t0=1e4, l=1e12, n0=1e-6, nu21=1e9, gamma21=1e-9, nu32=2e9, gamma32=5e-9)
        >>> eta21, eta32 = s["eta21"], s["eta32"]
    """
    values = {"21": (nu21, d21, gamma21), "31": (nu31, d31, gamma31), "32": (nu32, d32, gamma32)}
    if not (t0 > 0 and l > 0 and n0 > 0):
        raise ValueError("t0, l and n0 must be positive")
    out: Dict[str, float] = {}
    for p, (nu, d, gamma) in values.items():
        if nu is None:
            if d is not None or gamma is not None:
                raise ValueError(f"d{p} or gamma{p} given without nu{p}")
            continue
        if (d is None) == (gamma is None):
            raise ValueError(f"give exactly one of d{p} and gamma{p}")
        if not nu > 0:
            raise ValueError(f"nu{p} must be positive")
        k = 2 * PI * nu / C
        if d is not None:
            d = abs(d)
            gamma = k**3 * d**2 / (3 * PI * EPSILON_0 * H_BAR)
        else:
            gamma = float(gamma)
            d = math.sqrt(3 * PI * EPSILON_0 * H_BAR * gamma / k**3)
        if not (d > 0 and gamma > 0):
            raise ValueError(f"d{p} and gamma{p} must be positive")
        t_r = 2.0 * k**2 / (3.0 * PI * gamma * n0 * l)
        e0 = H_BAR / (t0 * d)
        out.update({
            f"eta{p}": t0 / t_r, f"t_r{p}": t_r, f"gamma{p}": gamma, f"d{p}": d,
            f"k{p}": k, f"lambda{p}": C / nu, f"e0_{p}": e0, f"i0_{p}": 0.5 * EPSILON_0 * C * e0**2,
        })
    if not out:
        raise ValueError("give at least one transition (nu_ij with d_ij or gamma_ij)")
    out["lambda0"] = n0 / t0
    return out
