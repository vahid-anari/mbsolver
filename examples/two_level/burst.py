"""Superradiant bursts of a two-level medium.

A fully inverted medium with a small initial tipping angle θ0 emits a delayed
burst followed by ringing. A stronger coupling η gives an earlier and shorter
burst; its peak intensity grows as η², so the intensity is shown divided by
η². No relaxation, pump or incident field. Everything is dimensionless (time
in units of T0).

Run from this folder (needs matplotlib):  python burst.py
"""

import math

import matplotlib.pyplot as plt

from mbsolver.two_level import solve_1d

fig, (ax_i, ax_w) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
for eta in (10.0, 20.0, 40.0):
    res = solve_1d(t_max=15, nt=6001, nz=201, eta=eta, theta0=1e-4,
                   t1=math.inf, t2=math.inf)
    ax_i.plot(res.time, res.intensity[-1] / eta**2, label=rf"$\eta$ = {eta:g}")
    ax_w.plot(res.time, res.w[-1])

ax_i.set_ylabel(r"exit intensity $|A'|^2 / \eta^2$")
ax_i.legend()
ax_w.set_ylabel(r"exit inversion $w$")
ax_w.set_xlabel(r"time $\tau'$ (units of $T_0$)")
fig.suptitle(r"Two-level superradiance, $\theta_0 = 10^{-4}$")
fig.tight_layout()
plt.show()
