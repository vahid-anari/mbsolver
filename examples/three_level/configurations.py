"""The three configurations of the three-level model side by side.

Each column starts with both allowed transitions inverted and slightly seeded,
with the same couplings and no relaxation, pump or incident field. In Λ and V
the two transitions compete for the shared level; in the ladder, emission on
the upper step fills the middle level and the lower step bursts later (a
cascade). Everything is dimensionless (time in units of T0).

Run from this folder (needs matplotlib):  python configurations.py
"""

import math

import matplotlib.pyplot as plt

from mbsolver.three_level import CONFIGS, solve_1d

CASES = {
    "lambda": ("Λ", (0.1, 0.2, 0.7)),
    "v": ("V", (0.1, 0.4, 0.5)),
    "ladder": ("ladder", (0.05, 0.05, 0.9)),
}
INF = math.inf

fig, axes = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
for col, (config, (title, n_init)) in enumerate(CASES.items()):
    a, b = CONFIGS[config]
    res = solve_1d(
        config=config, t_max=30, nt=3001, nz=101, n_init=n_init,
        **{f"eta{a}": 10.0, f"eta{b}": 10.0, f"c{a}": 1e-3, f"c{b}": 1e-3,
           f"t1_{a}": INF, f"t1_{b}": INF, f"t2_{a}": INF, f"t2_{b}": INF},
    )
    for p in (a, b):
        label = f"{p[0]}→{p[1]}"
        axes[0, col].plot(res.time, getattr(res, f"e{p}")[-1] ** 2, label=label)
        axes[1, col].plot(res.time, getattr(res, f"n{p}")[-1], label=rf"$n_{{{p}}}$")
    axes[0, col].set_title(f"{title}, n_init = {n_init}")
    axes[0, col].legend()
    axes[1, col].legend()
    axes[1, col].set_xlabel(r"time $\tau$ (units of $T_0$)")

axes[0, 0].set_ylabel(r"exit intensity $|\tilde E_{ij}|^2$")
axes[1, 0].set_ylabel(r"exit population difference $n_{ij}$")
fig.suptitle(r"Three-level configurations, $\eta_{ij} = 10$, $c_{ij} = 10^{-3}$")
fig.tight_layout()
plt.show()
