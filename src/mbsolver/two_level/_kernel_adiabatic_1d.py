"""Two-level Maxwell-Bloch equations, adiabatic (rate-equation) limit, one dimension.

Temporary test solver. With dR/dt = 0 the coherence follows the field and the population::

    R = R0 - T2 A w

so only two equations remain::

    dw/dt = A R0 - T2 A² w - (w - w0) / T1 + Λ
    dA/dz = -η R = -η R0 + η T2 A w

Same grids and z step as ``_kernel_1d`` (explicit step in z for the field using ``R`` of the
previous plane; ``A``, ``T1``, ``T2`` and ``Λ`` held at the start of each time step). In time,
``w`` is advanced by the exact solution of its (then linear) equation instead of RK4, because
``T2 A²`` makes the equation stiff when the field is strong.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline="always", nogil=True, cache=True)
def _step_w(w, A, inv_t1, t2, w0, R0, lambda_n, dt):
    """Exact step of the linear equation dw/dt = a - b w (coefficients frozen over the step).

    ``b = T2 A² + 1/T1`` can be large for a strong field (stiff), so an explicit RK4 step
    would blow up; this step is stable for any ``dt``.
    """
    a = A * R0 + w0 * inv_t1 + lambda_n
    b = t2 * A * A + inv_t1
    bdt = b * dt
    if bdt < 1e-12:
        return w + (a - b * w) * dt
    return w + (a - b * w) * (-np.expm1(-bdt)) / b


@njit(nogil=True, cache=True)
def _march_plane(w, R, A, w0, R0, inv_t1, t2, lambda_n, dt):
    """Fill ``w`` and ``R`` on one z plane for the field ``A`` already known there."""
    nt = len(w)
    w[0] = w0
    R[0] = R0 - t2[0] * A[0] * w[0]
    for t_idx in range(1, nt):
        w[t_idx] = _step_w(w[t_idx - 1], A[t_idx - 1], inv_t1[t_idx - 1], t2[t_idx - 1], w0, R0,
                           lambda_n[t_idx - 1], dt)
        R[t_idx] = R0 - t2[t_idx] * A[t_idx] * w[t_idx]


@njit(nogil=True, cache=True)
def solve_kernel(t, dt, z, dz, n_z_planes, w0, R0, A0, lambda_n, t1, t2, eta):
    """Advance the system over the grid (same arguments as ``_kernel_1d.solve_kernel``).

    ``t2`` must be finite.

    Returns:
        ``(w, I, R)``, each of shape ``(n_z_planes, len(t))``.
    """
    nt = len(t)
    nz = len(z)
    z_planes = np.linspace(0.0, 1.0, n_z_planes)
    z_plane_indices = np.rint(z_planes * (nz - 1)).astype(np.int64)

    inv_t1 = 1.0 / t1

    w_old = np.empty(nt, dtype=np.float64)
    R_old = np.empty(nt, dtype=np.float64)
    A_old = A0.copy()

    w_new = np.empty_like(w_old)
    R_new = np.empty_like(R_old)
    A_new = np.empty_like(A_old)

    w_result = np.empty((n_z_planes, nt), dtype=np.float64)
    I_result = np.empty((n_z_planes, nt), dtype=np.float64)
    R_result = np.empty((n_z_planes, nt), dtype=np.float64)

    _march_plane(w_old, R_old, A_old, w0[0], R0[0], inv_t1, t2, lambda_n, dt)
    w_result[0, :] = w_old
    I_result[0, :] = A_old * A_old
    R_result[0, :] = R_old
    i = 1

    for z_idx in range(1, nz):
        for t_idx in range(nt):
            A_new[t_idx] = A_old[t_idx] - eta * R_old[t_idx] * dz
        _march_plane(w_new, R_new, A_new, w0[z_idx], R0[z_idx], inv_t1, t2, lambda_n, dt)

        if i < n_z_planes and z_plane_indices[i] == z_idx:
            w_result[i, :] = w_new
            I_result[i, :] = A_new * A_new
            R_result[i, :] = R_new
            i += 1

        w_old, w_new = w_new, w_old
        R_old, R_new = R_new, R_old
        A_old, A_new = A_new, A_old

    return w_result, I_result, R_result
