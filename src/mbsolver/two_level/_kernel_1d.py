"""Two-level Maxwell-Bloch equations in one dimension (numba kernel).

Moved unchanged from MB Solver 8.13 (``solver/maxwell_bloch.py``):
RK4 in time for the population and coherence, explicit step in z for the field.

    dw/dt = A R - (w - w0) / T1 + Λ
    dR/dt = -A w - (R - R0) / T2
    dA/dz = -η R
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline="always", nogil=True, cache=True)
def _dw_dt(w, R, A, kw, kR, w0, inv_t1, lambda_n):
    """Population derivative for one RK stage."""
    return A * (R + kR) - (w + kw - w0) * inv_t1 + lambda_n


@njit(inline="always", nogil=True, cache=True)
def _dR_dt(w, R, A, kw, kR, R0, inv_t2):
    """Coherence derivative for one RK stage."""
    return -A * (w + kw) - (R + kR - R0) * inv_t2


@njit(inline="always", nogil=True, cache=True)
def _dA_dz(eta, R, dz):
    """Field increment across one spatial step."""
    return -eta * R * dz


@njit(inline="always", nogil=True, cache=True)
def _rk4_dn_dR(w, R, A, inv_t1, inv_t2, w0, R0, lambda_n, dt, half_dt, dt6):
    """One RK4 step for population and coherence; returns ``(dw, dR)``."""
    kw_1 = _dw_dt(w, R, A, 0, 0, w0, inv_t1, lambda_n)
    kR_1 = _dR_dt(w, R, A, 0, 0, R0, inv_t2)

    kw_2 = _dw_dt(w, R, A, half_dt * kw_1, half_dt * kR_1, w0, inv_t1, lambda_n)
    kR_2 = _dR_dt(w, R, A, half_dt * kw_1, half_dt * kR_1, R0, inv_t2)

    kw_3 = _dw_dt(w, R, A, half_dt * kw_2, half_dt * kR_2, w0, inv_t1, lambda_n)
    kR_3 = _dR_dt(w, R, A, half_dt * kw_2, half_dt * kR_2, R0, inv_t2)

    kw_4 = _dw_dt(w, R, A, dt * kw_3, dt * kR_3, w0, inv_t1, lambda_n)
    kR_4 = _dR_dt(w, R, A, dt * kw_3, dt * kR_3, R0, inv_t2)

    dw = dt6 * (kw_1 + 2.0 * kw_2 + 2.0 * kw_3 + kw_4)
    dR = dt6 * (kR_1 + 2.0 * kR_2 + 2.0 * kR_3 + kR_4)

    return dw, dR


@njit(nogil=True, cache=True)
def solve_kernel(t, dt, z, dz, n_z_planes, w0, R0, A0, lambda_n, t1, t2, eta):
    """Advance the system over the grid.

    Args:
        t, dt: Time grid and step.
        z, dz: Spatial grid and step.
        n_z_planes: Number of evenly spaced z planes to store (<= len(z)).
        w0, R0: Initial (and equilibrium) population and coherence along z.
        A0: Boundary field at z = 0 on the time grid.
        lambda_n: Pump on the time grid.
        t1, t2: Relaxation times on the time grid (> 0, may be inf).
        eta: Coupling coefficient.

    Returns:
        ``(w, I)``, each of shape ``(n_z_planes, len(t))``.
    """
    nt = len(t)
    nz = len(z)
    z_planes = np.linspace(0.0, 1.0, n_z_planes)
    z_plane_indices = np.rint(z_planes * (nz - 1)).astype(np.int64)

    inv_t1 = 1.0 / t1
    inv_t2 = 1.0 / t2
    half_dt = 0.5 * dt
    dt6 = dt / 6.0

    w_old = np.empty(nt, dtype=np.float64)
    R_old = np.empty(nt, dtype=np.float64)
    A_old = np.empty(nt, dtype=np.float64)

    w_new = np.empty_like(w_old)
    R_new = np.empty_like(R_old)
    A_new = np.empty_like(A_old)

    w_result = np.empty((n_z_planes, nt), dtype=np.float64)
    I_result = np.empty((n_z_planes, nt), dtype=np.float64)

    w_old[0], R_old[0] = w0[0], R0[0]
    A_old[0] = A0[0]
    w0_0 = w0[0]
    R0_0 = R0[0]

    for t_idx in range(1, nt):
        dw, dR = _rk4_dn_dR(
            w_old[t_idx - 1], R_old[t_idx - 1], A_old[t_idx - 1],
            inv_t1[t_idx - 1], inv_t2[t_idx - 1], w0_0, R0_0,
            lambda_n[t_idx - 1], dt, half_dt, dt6,
        )
        w_old[t_idx] = w_old[t_idx - 1] + dw
        R_old[t_idx] = R_old[t_idx - 1] + dR
        A_old[t_idx] = A0[t_idx]

    w_result[0, :] = w_old
    I_result[0, :] = A_old * A_old
    i = 1

    for z_idx in range(1, nz):
        w_new[0], R_new[0] = w0[z_idx], R0[z_idx]
        A_new[0] = A_old[0] + _dA_dz(eta, R_old[0], dz)
        w0_i = w0[z_idx]
        R0_i = R0[z_idx]

        for t_idx in range(1, nt):
            dw, dR = _rk4_dn_dR(
                w_new[t_idx - 1], R_new[t_idx - 1], A_new[t_idx - 1],
                inv_t1[t_idx - 1], inv_t2[t_idx - 1], w0_i, R0_i,
                lambda_n[t_idx - 1], dt, half_dt, dt6,
            )
            w_new[t_idx] = w_new[t_idx - 1] + dw
            R_new[t_idx] = R_new[t_idx - 1] + dR
            A_new[t_idx] = A_old[t_idx] + _dA_dz(eta, R_old[t_idx], dz)

        if i < n_z_planes and z_plane_indices[i] == z_idx:
            w_result[i, :] = w_new
            I_result[i, :] = A_new * A_new
            i += 1

        w_old, w_new = w_new, w_old
        R_old, R_new = R_new, R_old
        A_old, A_new = A_new, A_old

    return w_result, I_result
