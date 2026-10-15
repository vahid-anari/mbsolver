"""From the physical sample to the solver's dimensionless parameters.

The solvers work in dimensionless units: time in units of ``T0``, the medium
from ``z = 0`` to ``z = 1``. :func:`scaling` turns the physical values of a
maser sample into the coupling ``eta`` and initial Bloch angle ``theta0`` the
solvers take, together with the other derived scales::

    import math
    import mbsolver
    from mbsolver.two_level import solve_1d

    s = mbsolver.scaling(nu=6.67e9, gamma=1.56e-9, t0=86400, l=7.5e12, n0=5.3e-7)
    res = solve_1d(t_max=2000, nt=20001, nz=101, eta=s["eta"], theta0=s["theta0"],
                   t1=math.inf, t2=math.inf)
"""

from __future__ import annotations

import math
from typing import Dict

from mbsolver.constants import C, EPSILON_0, H_BAR, MU_0, PI

__all__ = ["scaling"]


def scaling(nu: float, gamma: float, t0: float, l: float, n0: float) -> Dict[str, float]:
    """Compute the solver parameters and derived scales from the sample values.

    Args:
        nu: Transition frequency ν in Hz.
        gamma: Spontaneous decay rate Γ in s⁻¹.
        t0: Time scale T₀ in s (the solver's unit of time).
        l: Length of the medium L in m.
        n0: Inverted population density n₀ in m⁻³.

    Returns:
        Mapping, all in SI units:

        - ``eta``: coupling coefficient η = T₀ / T_R (the solvers' ``eta``),
        - ``theta0``: initial Bloch angle θ₀ = 2 / √N (the solvers' ``theta0``),
        - ``t_r``: superradiance time scale T_R,
        - ``N``: number of molecules in a column of one wavelength squared,
        - ``omega``, ``lambda``, ``k``: angular frequency, wavelength, wave number,
        - ``lambda0``: pump scale Λ₀ = 1 / T₀,
        - electric dipole: ``d`` (dipole moment), ``e0`` (field scale),
          ``p0`` (polarization scale), ``i0_electric`` (intensity scale),
        - magnetic dipole: ``mu``, ``b0``, ``m0``, ``i0_magnetic``.
    """
    omega0 = 2 * PI * nu
    k = omega0 / C
    _lambda = C / nu
    t_r = 2.0 * k**2 / (3.0 * PI * gamma * n0 * l)
    N = n0 * _lambda * l**2
    eta = t0 / t_r
    theta0 = 2.0 / math.sqrt(N)
    lambda0 = 1.0 / t0

    d = math.sqrt(3 * PI * EPSILON_0 * H_BAR * gamma / k**3)
    p0 = n0 * d
    e0 = H_BAR / (t0 * d)
    i0_electric = 0.5 * EPSILON_0 * C * e0**2

    mu = math.sqrt(3 * PI * H_BAR * gamma / (MU_0 * k**3))
    m0 = n0 * mu
    b0 = H_BAR / (t0 * mu)
    i0_magnetic = C / (2.0 * MU_0) * b0**2

    return {
        "omega": omega0, "lambda": _lambda, "k": k, "t_r": t_r, "eta": eta,
        "N": N, "theta0": theta0, "lambda0": lambda0,
        "d": d, "e0": e0, "p0": p0, "i0_electric": i0_electric,
        "mu": mu, "b0": b0, "m0": m0, "i0_magnetic": i0_magnetic,
    }
