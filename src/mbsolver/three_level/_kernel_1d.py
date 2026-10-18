"""Λ-type three-level Maxwell-Bloch equations in one dimension (numba kernel).

Moved from the three-level solver of 2025 (``maxwell_bloch_three_level_v2.py``)
with the same arithmetic: RK4 in time for populations and coherences,
explicit step in z for the fields. All quantities are real.

    dn31/dτ = -(E1 R31 + ½ E2 R32) - (n31 - n31⁰) / τ31⁽¹⁾ + Λ31
    dn32/dτ = -(½ E1 R31 + E2 R32) - (n32 - n32⁰) / τ32⁽¹⁾ + Λ32
    dR31/dτ = E1 n31 - ½ E2 R21 - (R31 - R31⁰) / τ31⁽²⁾
    dR32/dτ = E2 n32 - ½ E1 R21 - (R32 - R32⁰) / τ32⁽²⁾
    dR21/dτ = ½ (E1 R32 + E2 R31) - (R21 - R21⁰) / τ21⁽²⁾
    dE1/dξ = η R31,   dE2/dξ = R32
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline="always", nogil=True, cache=True)
def _derivs(n31, n32, R31, R32, R21, kn31, kn32, kR31, kR32, kR21,
            n31_0, n32_0, R31_0, R32_0, R21_0, E31, E32,
            pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21):
    """Right-hand sides for one RK stage."""
    return (
        -(E31 * (R31 + kR31) + 0.5 * E32 * (R32 + kR32)) - (n31 + kn31 - n31_0) / t1_31 + pump31,
        -(0.5 * E31 * (R31 + kR31) + E32 * (R32 + kR32)) - (n32 + kn32 - n32_0) / t1_32 + pump32,
        E31 * (n31 + kn31) - 0.5 * E32 * (R21 + kR21) - (R31 + kR31 - R31_0) / t2_31,
        E32 * (n32 + kn32) - 0.5 * E31 * (R21 + kR21) - (R32 + kR32 - R32_0) / t2_32,
        0.5 * (E31 * (R32 + kR32) + E32 * (R31 + kR31)) - (R21 + kR21 - R21_0) / t2_21,
    )


@njit(inline="always", nogil=True, cache=True)
def _rk4_sum(n31, n32, R31, R32, R21, n31_0, n32_0, R31_0, R32_0, R21_0, E31, E32,
             pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21, dt):
    """``k1 + 2 k2 + 2 k3 + k4`` for the five variables."""
    h = 0.5 * dt
    a1, b1, c1, d1, e1 = _derivs(n31, n32, R31, R32, R21, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 n31_0, n32_0, R31_0, R32_0, R21_0, E31, E32,
                                 pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21)
    a2, b2, c2, d2, e2 = _derivs(n31, n32, R31, R32, R21,
                                 h * a1, h * b1, h * c1, h * d1, h * e1,
                                 n31_0, n32_0, R31_0, R32_0, R21_0, E31, E32,
                                 pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21)
    a3, b3, c3, d3, e3 = _derivs(n31, n32, R31, R32, R21,
                                 h * a2, h * b2, h * c2, h * d2, h * e2,
                                 n31_0, n32_0, R31_0, R32_0, R21_0, E31, E32,
                                 pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21)
    a4, b4, c4, d4, e4 = _derivs(n31, n32, R31, R32, R21,
                                 dt * a3, dt * b3, dt * c3, dt * d3, dt * e3,
                                 n31_0, n32_0, R31_0, R32_0, R21_0, E31, E32,
                                 pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21)
    return (a1 + 2.0 * a2 + 2.0 * a3 + a4,
            b1 + 2.0 * b2 + 2.0 * b3 + b4,
            c1 + 2.0 * c2 + 2.0 * c3 + c4,
            d1 + 2.0 * d2 + 2.0 * d3 + d4,
            e1 + 2.0 * e2 + 2.0 * e3 + e4)


@njit(nogil=True, cache=True)
def _slice(v, E31, E32, init, pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21, dt):
    """Advance one z slice in time; ``v`` is ``(5, nt)`` with column 0 set."""
    nt = v.shape[1]
    dt6 = dt / 6.0
    n31_0, n32_0, R31_0, R32_0, R21_0 = init[0], init[1], init[2], init[3], init[4]
    for j in range(1, nt):
        i = j - 1
        s0, s1, s2, s3, s4 = _rk4_sum(
            v[0, i], v[1, i], v[2, i], v[3, i], v[4, i],
            n31_0, n32_0, R31_0, R32_0, R21_0, E31[i], E32[i],
            pump31[i], pump32[i], t1_31[i], t1_32[i], t2_31[i], t2_32[i], t2_21[i], dt)
        v[0, j] = v[0, i] + dt6 * s0
        v[1, j] = v[1, i] + dt6 * s1
        v[2, j] = v[2, i] + dt6 * s2
        v[3, j] = v[3, i] + dt6 * s3
        v[4, j] = v[4, i] + dt6 * s4


@njit(nogil=True, cache=True)
def solve_kernel(t, dt, z, dz, n_z_planes, init, field31_in, field32_in, pump31, pump32,
                 t1_31, t1_32, t2_31, t2_32, t2_21, eta):
    """Advance the system over the grid.

    Args:
        t, dt: Time grid and step.
        z, dz: Spatial grid and step.
        n_z_planes: Number of evenly spaced z planes to store (<= len(z)).
        init: Initial (and equilibrium) ``(n31, n32, R31, R32, R21)``, the same at every z.
        field31_in, field32_in: Fields at z = 0 on the time grid.
        pump31, pump32: Pumps on the time grid.
        t1_31, t1_32, t2_31, t2_32, t2_21: Relaxation times on the time grid (> 0, may be inf).
        eta: Coupling coefficient of the 3→1 field (that of 3→2 is 1).

    Returns:
        ``(v, E)``: ``v`` of shape ``(n_z_planes, 5, len(t))`` holding
        ``n31, n32, R31, R32, R21``; ``E`` of shape ``(n_z_planes, 2, len(t))``
        holding ``E31, E32``.
    """
    nt = len(t)
    nz = len(z)
    z_planes = np.linspace(0.0, 1.0, n_z_planes)
    z_plane_indices = np.rint(z_planes * (nz - 1)).astype(np.int64)

    v_old = np.empty((5, nt), dtype=np.float64)
    v_new = np.empty((5, nt), dtype=np.float64)
    E31_old = field31_in.copy()
    E32_old = field32_in.copy()
    E31_new = np.empty(nt, dtype=np.float64)
    E32_new = np.empty(nt, dtype=np.float64)

    v_result = np.empty((n_z_planes, 5, nt), dtype=np.float64)
    E_result = np.empty((n_z_planes, 2, nt), dtype=np.float64)

    v_old[:, 0] = init
    _slice(v_old, E31_old, E32_old, init, pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21, dt)
    v_result[0] = v_old
    E_result[0, 0] = E31_old
    E_result[0, 1] = E32_old
    i = 1

    for z_idx in range(1, nz):
        for j in range(nt):
            E31_new[j] = E31_old[j] + eta * v_old[2, j] * dz
            E32_new[j] = E32_old[j] + 1.0 * v_old[3, j] * dz
        v_new[:, 0] = init
        _slice(v_new, E31_new, E32_new, init, pump31, pump32, t1_31, t1_32, t2_31, t2_32, t2_21, dt)

        if i < n_z_planes and z_plane_indices[i] == z_idx:
            v_result[i] = v_new
            E_result[i, 0] = E31_new
            E_result[i, 1] = E32_new
            i += 1

        v_old, v_new = v_new, v_old
        E31_old, E31_new = E31_new, E31_old
        E32_old, E32_new = E32_new, E32_old

    return v_result, E_result
