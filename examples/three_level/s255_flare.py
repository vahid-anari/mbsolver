"""S255IR-NIRS3: the 6.7 GHz methanol flare with a 3→1 leakage pathway.

The three-level case of the paper: r2 = 2.33e19 and c32 = 1 for the observed
3→2 line, a weakly coherent 3→1 pathway (c31 = 1e-10) with shorter
relaxation times, and a sech² pump. Runs the model for three values of
r1 and plots both intensities in Jy, with the observed 3→2 flux at
v = 6.932 km/s (``s255_v6.932.txt``), observed with the Toruń 32 m
telescope by Szymczak et al. 2018, A&A 617, A80
(https://doi.org/10.1051/0004-6361/201833443); please cite it when you
use these data.

Run from this folder (needs matplotlib):  python s255_flare.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mbsolver import Pulse
from mbsolver.constants import DEBYE
from mbsolver.three_level import populations, scaling, solve_1d

HERE = Path(__file__).parent
DAY = 86400.0

# Sample: CH3OH 3→1 (239.7 GHz) and 3→2 (6.67 GHz)
theta0 = 4.14e-10                                      # Bloch angle of the two-level fit
r1 = 0.94
r2 = 1.0 / np.tan(theta0 / 2) ** 2                    # (1 + cos θ0) / (1 − cos θ0) = 2.33e19
n0_2level = 1.17e-6                                    # m⁻³, molecules in levels 2 and 3
n0 = n0_2level * (r1 + r2 + r1 * r2) / (r1 * (1 + r2))  # m⁻³, all three levels
s = scaling(nu31=239_746.219e6, nu32=6_668.564e6, d31=0.59425 * DEBYE,
            d32=0.67936 * DEBYE, l=2.1e13, n0=n0, gamma=1.56e-9)
t0 = s["t0"]

# Relaxation times and pump, physical values / scales
t1_32, t2_32 = 1.64e7 / t0, 1.55e6 / t0
pump32 = Pulse(a=2.6e-13 / s["lambda0"], x0=795.5, w=8.1e6 / t0)

# Time axis of the plots: model start at MJD 56800; observed at 50000 + (JD − 2450000)
T_START, DATA_OFFSET = 56800.0, 50000.0
I_NORM = 0.21                                          # flux factor of the paper's figures

data = np.loadtxt(HERE / "s255_v6.932.txt")

fig, (ax31, ax32) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for value in (0.79, 0.89, 0.94):
    pops = populations(value, r2)
    res = solve_1d(t_max=2500, nt=4000, nz=400, d=s["d"], omega=s["omega"],
                   r1=value, r2=r2, c31=1e-10, c32=1.0,
                   t1_31=0.1 * t1_32, t1_32=t1_32, t2_31=0.1 * t2_32, t2_32=t2_32,
                   pump32=pump32)
    t = T_START + res.time * t0 / DAY
    label = rf"$r_1 = {value}$  ($n_{{31}}^0 = {pops['n31']:.3f}$)"
    ax31.plot(t, res.intensity31[-1] * s["i0_jy"] * I_NORM, label=label)
    ax32.plot(t, res.intensity32[-1] * s["i0_jy"] * I_NORM, label=label)

ax32.plot(DATA_OFFSET + data[:, 0], data[:, 1], "o", color="black", ms=3, label="observed")
ax31.set_ylabel(r"$I_{31}$ (Jy)")
ax32.set_ylabel(r"$I_{32}$ (Jy)")
ax32.set_xlabel("t (MJD)")
ax32.set_xlim(57100, 58150)
ax31.set_ylim(0, 8e-14)
ax31.legend()
ax32.legend()
fig.suptitle(rf"S255IR-NIRS3, $T_0$ = {t0:.3g} s, $\eta$ = {s['eta']:.2f}")
fig.tight_layout()
plt.show()
