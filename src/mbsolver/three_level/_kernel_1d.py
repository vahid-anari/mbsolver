"""Three-level Maxwell-Bloch equations in one dimension (numba kernel).

RK4 in time for the two population differences of the allowed transitions and
the three coherences, explicit step in z for the fields. All quantities are
real. With ``W_ij = ½ E_ij R_ij``::

    dR21/dτ = E21 n21 + ½ (E31 R32 + E32 R31) - (R21 - R21⁰) / τ21⁽²⁾
    dR31/dτ = E31 n31 + ½ (E21 R32 - E32 R21) - (R31 - R31⁰) / τ31⁽²⁾
    dR32/dτ = E32 n32 - ½ (E21 R31 + E31 R21) - (R32 - R32⁰) / τ32⁽²⁾
    dE_ij/dξ = η_ij R_ij

and for the two differences ``na``, ``nb`` of the allowed transitions::

    Λ      (na, nb) = (n31, n32):  -(2 W31 + W32),  -(W31 + 2 W32)
    V      (na, nb) = (n21, n31):  -(2 W21 + W31),  -(W21 + 2 W31)
    ladder (na, nb) = (n21, n32):  -(2 W21 - W32),  -(2 W32 - W21)

each plus ``-(n - n⁰) / τ⁽¹⁾ + Λ``. The field and coupling of the forbidden
transition are zero. In the Λ case the arithmetic is that of the earlier
Λ-only kernel, so its results are unchanged.
"""

from __future__ import annotations

import numpy as np
from numba import njit

LAMBDA, V, LADDER = 0, 1, 2


@njit(inline="always", nogil=True, cache=True)
def _derivs(config, na, nb, R21, R31, R32, kna, knb, kR21, kR31, kR32,
            na_0, nb_0, R21_0, R31_0, R32_0, E21, E31, E32,
            pump_a, pump_b, t1_a, t1_b, t2_21, t2_31, t2_32):
    """Right-hand sides for one RK stage."""
    xa = na + kna
    xb = nb + knb
    x21 = R21 + kR21
    x31 = R31 + kR31
    x32 = R32 + kR32
    W21 = 0.5 * E21 * x21
    W31 = 0.5 * E31 * x31
    W32 = 0.5 * E32 * x32
    if config == LAMBDA:
        n31, n32 = xa, xb
        n21 = xa - xb
        coh_a = -(2.0 * W31 + W32)
        coh_b = -(W31 + 2.0 * W32)
    elif config == V:
        n21, n31 = xa, xb
        n32 = xb - xa
        coh_a = -(2.0 * W21 + W31)
        coh_b = -(W21 + 2.0 * W31)
    else:
        n21, n32 = xa, xb
        n31 = xa + xb
        coh_a = -(2.0 * W21 - W32)
        coh_b = -(2.0 * W32 - W21)
    return (
        coh_a - (xa - na_0) / t1_a + pump_a,
        coh_b - (xb - nb_0) / t1_b + pump_b,
        E21 * n21 + 0.5 * (E31 * x32 + E32 * x31) - (x21 - R21_0) / t2_21,
        E31 * n31 + 0.5 * (E21 * x32 - E32 * x21) - (x31 - R31_0) / t2_31,
        E32 * n32 - 0.5 * (E21 * x31 + E31 * x21) - (x32 - R32_0) / t2_32,
    )


@njit(inline="always", nogil=True, cache=True)
def _rk4_sum(config, na, nb, R21, R31, R32, na_0, nb_0, R21_0, R31_0, R32_0, E21, E31, E32,
             pump_a, pump_b, t1_a, t1_b, t2_21, t2_31, t2_32, dt):
    """``k1 + 2 k2 + 2 k3 + k4`` for the five variables."""
    h = 0.5 * dt
    a1, b1, c1, d1, e1 = _derivs(config, na, nb, R21, R31, R32, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 na_0, nb_0, R21_0, R31_0, R32_0, E21, E31, E32,
                                 pump_a, pump_b, t1_a, t1_b, t2_21, t2_31, t2_32)
    a2, b2, c2, d2, e2 = _derivs(config, na, nb, R21, R31, R32,
                                 h * a1, h * b1, h * c1, h * d1, h * e1,
                                 na_0, nb_0, R21_0, R31_0, R32_0, E21, E31, E32,
                                 pump_a, pump_b, t1_a, t1_b, t2_21, t2_31, t2_32)
    a3, b3, c3, d3, e3 = _derivs(config, na, nb, R21, R31, R32,
                                 h * a2, h * b2, h * c2, h * d2, h * e2,
                                 na_0, nb_0, R21_0, R31_0, R32_0, E21, E31, E32,
                                 pump_a, pump_b, t1_a, t1_b, t2_21, t2_31, t2_32)
    a4, b4, c4, d4, e4 = _derivs(config, na, nb, R21, R31, R32,
                                 dt * a3, dt * b3, dt * c3, dt * d3, dt * e3,
                                 na_0, nb_0, R21_0, R31_0, R32_0, E21, E31, E32,
                                 pump_a, pump_b, t1_a, t1_b, t2_21, t2_31, t2_32)
    return (a1 + 2.0 * a2 + 2.0 * a3 + a4,
            b1 + 2.0 * b2 + 2.0 * b3 + b4,
            c1 + 2.0 * c2 + 2.0 * c3 + c4,
            d1 + 2.0 * d2 + 2.0 * d3 + d4,
            e1 + 2.0 * e2 + 2.0 * e3 + e4)


@njit(nogil=True, cache=True)
def _slice(config, v, E, init, pump_a, pump_b, t1_a, t1_b, t2, dt):
    """Advance one z slice in time; ``v`` is ``(5, nt)`` with column 0 set, ``E`` is ``(3, nt)``."""
    nt = v.shape[1]
    dt6 = dt / 6.0
    na_0, nb_0, R21_0, R31_0, R32_0 = init[0], init[1], init[2], init[3], init[4]
    for j in range(1, nt):
        i = j - 1
        s0, s1, s2, s3, s4 = _rk4_sum(
            config, v[0, i], v[1, i], v[2, i], v[3, i], v[4, i],
            na_0, nb_0, R21_0, R31_0, R32_0, E[0, i], E[1, i], E[2, i],
            pump_a[i], pump_b[i], t1_a[i], t1_b[i], t2[0, i], t2[1, i], t2[2, i], dt)
        v[0, j] = v[0, i] + dt6 * s0
        v[1, j] = v[1, i] + dt6 * s1
        v[2, j] = v[2, i] + dt6 * s2
        v[3, j] = v[3, i] + dt6 * s3
        v[4, j] = v[4, i] + dt6 * s4


@njit(nogil=True, cache=True)
def solve_kernel(config, t, dt, z, dz, n_z_planes, init, field_in, pump_a, pump_b,
                 t1_a, t1_b, t2, eta):
    """Advance the system over the grid.

    Args:
        config: ``LAMBDA``, ``V`` or ``LADDER``.
        t, dt: Time grid and step.
        z, dz: Spatial grid and step.
        n_z_planes: Number of evenly spaced z planes to store (<= len(z)).
        init: Initial (and equilibrium) ``(na, nb, R21, R31, R32)``, the same at every z.
        field_in: Fields ``E21, E31, E32`` at z = 0, shape ``(3, len(t))`` (zero for the forbidden one).
        pump_a, pump_b: Pumps of ``na`` and ``nb`` on the time grid.
        t1_a, t1_b: Relaxation times of ``na`` and ``nb`` on the time grid (> 0, may be inf).
        t2: Dephasing times of ``R21, R31, R32``, shape ``(3, len(t))`` (> 0, may be inf).
        eta: Couplings ``η21, η31, η32`` (zero for the forbidden transition).

    Returns:
        ``(v, E)``: ``v`` of shape ``(n_z_planes, 5, len(t))`` holding
        ``na, nb, R21, R31, R32``; ``E`` of shape ``(n_z_planes, 3, len(t))``
        holding ``E21, E31, E32``.
    """
    nt = len(t)
    nz = len(z)
    z_planes = np.linspace(0.0, 1.0, n_z_planes)
    z_plane_indices = np.rint(z_planes * (nz - 1)).astype(np.int64)

    v_old = np.empty((5, nt), dtype=np.float64)
    v_new = np.empty((5, nt), dtype=np.float64)
    E_old = field_in.copy()
    E_new = np.empty((3, nt), dtype=np.float64)

    v_result = np.empty((n_z_planes, 5, nt), dtype=np.float64)
    E_result = np.empty((n_z_planes, 3, nt), dtype=np.float64)

    v_old[:, 0] = init
    _slice(config, v_old, E_old, init, pump_a, pump_b, t1_a, t1_b, t2, dt)
    v_result[0] = v_old
    E_result[0] = E_old
    i = 1

    for z_idx in range(1, nz):
        for k in range(3):
            for j in range(nt):
                E_new[k, j] = E_old[k, j] + eta[k] * v_old[2 + k, j] * dz
        v_new[:, 0] = init
        _slice(config, v_new, E_new, init, pump_a, pump_b, t1_a, t1_b, t2, dt)

        if i < n_z_planes and z_plane_indices[i] == z_idx:
            v_result[i] = v_new
            E_result[i] = E_new
            i += 1

        v_old, v_new = v_new, v_old
        E_old, E_new = E_new, E_old

    return v_result, E_result
