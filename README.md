# mbsolver

[![tests](https://github.com/vahid-anari/mbsolver/actions/workflows/tests.yml/badge.svg)](https://github.com/vahid-anari/mbsolver/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/mbsolver)](https://pypi.org/project/mbsolver/)
[![Python](https://img.shields.io/pypi/pyversions/mbsolver)](https://pypi.org/project/mbsolver/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23407483-blue)](https://doi.org/10.5281/zenodo.23407483)
![License](https://img.shields.io/badge/license-MIT-green)

Fast Maxwell-Bloch solvers in Python (numpy + numba) for astrophysical masers,
superradiance and coherent pulse propagation. Use them from scripts,
notebooks or cluster jobs; the
[Maxwell Bloch Studio](https://github.com/vahid-anari/maxwell-bloch-studio)
desktop app runs on the same code.

## Installation

```
pip install mbsolver
```

Requires Python 3.10+ (numpy and numba are installed with it).

## Two-level model in one dimension

Dimensionless equations for the population inversion `w`, the coherence `R`
and the field `A` along a medium `0 ≤ z ≤ 1`:

```
dw/dt = A R − (w − w0) / T1 + Λ(t)
dR/dt = −A w − (R − R0) / T2
dA/dz = −η R
```

The field `A(t)` entering at `z = 0` is given; population and coherence are
advanced with fourth-order Runge-Kutta in time, the field step by step in `z`.
The derivation from the two-level Hamiltonian is in
[`docs/two_level.pdf`](docs/two_level.pdf).

```python
import matplotlib.pyplot as plt
from mbsolver import Pulse, Steps
from mbsolver.two_level import solve_1d

res = solve_1d(
    t_max=300, nt=3001, nz=121, eta=40.0,
    theta0=0.003,                                  # or w0=..., R0=...
    pump=[Pulse(a=[2e-3, 1e-3], x0=[60, 180], w=[15, 25])],
    field_in=Pulse(a=1e-3, x0=30, w=5),            # A(t) at z = 0
    t1=Steps(y=[80, 40, 60], x0=[100, 200], w=[10, 0]),
    t2=5.0,
    n_planes=7,                                    # stored z planes, 0 ... 1
)

plt.plot(res.time, res.intensity[-1])              # intensity leaving the medium
plt.show()
```

### Arguments

| Argument | Meaning |
|----------|---------|
| `t_max`, `nt` | time grid from 0 to `t_max` with `nt` points |
| `nz` | number of z points |
| `eta` | coupling coefficient η |
| `theta0`, or `w0` and `R0` | initial state (`w0 = cos θ0`, `R0 = sin θ0`); the medium also relaxes towards it |
| `t1`, `t2` | relaxation times: a number or `Steps` (> 0, `math.inf` for none) |
| `pump` | Λ(t): a `Pulse`, a number, or a list of them (summed) |
| `field_in` | A(t) at z = 0: a `Pulse`, a number, or `None` |
| `n_planes` | number of evenly spaced z planes returned (default 2: entrance and exit) |

### Result

`res.time`, `res.z` (positions of the stored planes), `res.intensity` and
`res.w` (shape `(n_planes, nt)`), `res.pump`, `res.field_in`, `res.t1`, `res.t2`.

### Adiabatic (rate-equation) limit

When `T2` is short compared with the times over which `A` and `w` change, the
coherence follows its steady state, `R = R0 − T2 A w`, and two equations remain:

```
dw/dt = A R0 − T2 A² w − (w − w0) / T1 + Λ(t)
dA/dz = −η R0 + η T2 A w
```

`solve_1d_adiabatic` takes the same arguments as `solve_1d` (with a finite
`t2`) and returns the same result plus `res.R`, so the two can be compared
directly. The limit differs from the full system during the first few `T2`
and in bursts faster than `T2`; see the section on it in
[`docs/two_level.pdf`](docs/two_level.pdf).

## Three-level (Λ) model in one dimension

Levels 1 and 2 lie below level 3; the transitions 3→1 and 3→2 carry the fields
`E31` and `E32`. With the population differences `n31 = n3 − n1`,
`n32 = n3 − n2` and the coherences `R31`, `R32`, `R21` (all real):

```
dn31/dt = −(E31 R31 + ½ E32 R32) − (n31 − n31⁰) / T1_31 + Λ31(t)
dn32/dt = −(½ E31 R31 + E32 R32) − (n32 − n32⁰) / T1_32 + Λ32(t)
dR31/dt = E31 n31 − ½ E32 R21 − (R31 − R31⁰) / T2_31
dR32/dt = E32 n32 − ½ E31 R21 − (R32 − R32⁰) / T2_32
dR21/dt = ½ (E31 R32 + E32 R31) − (R21 − R21⁰) / T2_21
dE31/dz = η R31,   dE32/dz = R32,   η = ω d²
```

with `d = |d31/d32|` and `ω = k31/k32`. Time is in units of the 3→2 time
scale `T0 = 2ε0ħ / (n0 L k32 |d32|²)`. The initial state comes from the
population ratios `r1 = n3/n1`, `r2 = n3/n2` and the coherence fractions
`c_ij`: `R_ij⁰ = c_ij · 2√(n_i n_j)`.

```python
from mbsolver import Pulse
from mbsolver.three_level import solve_1d

res = solve_1d(
    t_max=2500, nt=4000, nz=400, d=0.875, omega=35.95,
    r1=0.94, r2=2.33e19, c31=1e-10, c32=1.0,       # c21 = 0 by default
    t1_31=31.3, t1_32=313, t2_31=2.96, t2_32=29.6, # t2_21 = inf by default
    pump32=Pulse(a=5.63e-3, x0=795.5, w=154.8),    # pump31, field31_in, field32_in too
)
res.intensity32[-1]                                # 3→2 intensity leaving the medium
```

`t1_*` and `t2_*` take numbers or `Steps`, the pumps and entrance fields
`Pulse`s, numbers or lists, as in the two-level model. The result holds
`time`, `z`, the fields `e31`, `e32`, the intensities `intensity31 = (E31/d)²`
and `intensity32 = E32²` (both in units of the 3→2 scale), `n31`, `n32`,
`r31`, `r32`, `r21` (shape `(n_planes, nt)`), and the pumps, entrance fields
and relaxation times on the time grid.

`populations(r1, r2)` gives `n1`, `n2`, `n3`, `n31⁰`, `n32⁰` and the largest
coherences. `scaling(nu31, nu32, d31, d32, l, n0)` (SI units;
`mbsolver.constants.DEBYE` converts dipoles from debye) gives `d`, `omega`,
`eta`, `t0`, the field and intensity scales `e0`, `i0` (and `i0_jy` = 10²⁶ `i0`)
and the pump scale `lambda0 = n0/t0`. By default the decay rate of the 3→2
line is the Einstein A computed from `d32`; pass `gamma=` to use a measured one.

The three-level equations were derived and checked with Toktam Rashidi. The
derivation, from the Hamiltonian to the dimensionless equations, with the
initial conditions, physical scales and the S255IR-NIRS3 values, is in
[`docs/three_level.pdf`](docs/three_level.pdf);
[`examples/three_level/s255_flare.py`](examples/three_level/s255_flare.py)
runs the S255IR-NIRS3 flare and plots it with the observed flux (needs
matplotlib). The observations are from the Toruń 32 m telescope,
[Szymczak et al. 2018, A&A 617, A80](https://doi.org/10.1051/0004-6361/201833443).

## From physical values: `scaling` (two-level)

The solvers take dimensionless parameters. `mbsolver.scaling` computes them for the two-level model from the
physical sample values (SI units): transition frequency `nu` (Hz), spontaneous
decay rate `gamma` (s⁻¹), time unit `t0` (s), length `l` (m) and inverted
population density `n0` (m⁻³).

```python
import math
import mbsolver
from mbsolver.two_level import solve_1d

s = mbsolver.scaling(nu=6.67e9, gamma=1.56e-9, t0=86400, l=7.5e12, n0=5.3e-7)
res = solve_1d(t_max=2000, nt=20001, nz=101, eta=s["eta"], theta0=s["theta0"],
               t1=math.inf, t2=math.inf)
```

Besides `eta` (η = T₀/T_R) and `theta0` (θ₀ = 2/√N), the result holds `t_r`,
`N`, `omega`, `lambda`, `k`, `lambda0` (Λ₀ = 1/T₀) and the field, polarization
and intensity scales for an electric dipole (`d`, `e0`, `p0`, `i0_electric`)
and a magnetic dipole (`mu`, `b0`, `m0`, `i0_magnetic`). Times in the solver
are in units of `t0`.

## Profiles

`Pulse(a, x0, w)` is a sum of sech² pulses, `a_i sech²((t − x0_i) / w_i)`.
Give `wl` and `wr` instead of `w` for a different width on each side.

`Steps(y, x0, w)` goes from level `y[0]` to `y[1]` around `x0[0]`, then to
`y[2]` around `x0[1]`, and so on, each step smoothed as `(1 + tanh)/2` with
width `w` (0 for a sharp step). `x0` and `w` have one value fewer than `y`.

A zero width gives a single point (pulse) or a sharp step; an infinite width
a constant. `profile.evaluate(t)` returns the values on any grid.

## Development

```
pip install -e ".[dev]"
pytest
```

The tests include a reference run that must match the solver of the Maxwell
Bloch app (version 8.13) exactly, and physics checks (Bloch rotation at the
entrance, relaxation balance, uncoupled propagation). The three-level model
matches the 2025 three-level solver exactly for the S255IR-NIRS3 case. `scaling`
is checked against the values of Maxwell Bloch Studio.

The model documents `docs/two_level.pdf` and `docs/three_level.pdf` are built
from their LaTeX sources with `sh tools/build_docs.sh` (needs LaTeX with
`latexmk`); commit the rebuilt PDFs with the sources.

What changed in each version is in [`CHANGELOG.md`](CHANGELOG.md). Pushing a
tag `vX.Y.Z` publishes a GitHub Release with that version's section of the
changelog, and Zenodo gives it a DOI; update `version` and `date-released` in
`CITATION.cff` before tagging.

## Earlier name

Up to 0.1.0 this package was called `maxbloch`; 0.2.0 is the same code under
the new name. `maxbloch` stays on PyPI unchanged (0.1.1 only adds a
deprecation warning) but gets no updates; change `maxbloch` to `mbsolver` in
your imports.

## Citing

If you use mbsolver in published work, please cite it:

> Anari, V., & Rashidi, T. *mbsolver: Fast Maxwell-Bloch solvers for masers, superradiance and
> coherent pulse propagation*. Zenodo. https://doi.org/10.5281/zenodo.23407483

This DOI always points to the latest version. Each release also has its own
DOI (for 0.3.1: [10.5281/zenodo.23407484](https://doi.org/10.5281/zenodo.23407484));
cite the version you used, which you find under *Versions* on the Zenodo page.
GitHub's **Cite this repository** button gives the same in BibTeX and APA.

## License

MIT License. See [LICENSE](LICENSE).

## Authors

Vahid Anari ·
[GitHub](https://github.com/vahid-anari) ·
[LinkedIn](https://www.linkedin.com/in/vahid-anari/)

Toktam Rashidi ·
[GitHub](https://github.com/ToktamS) ·
[LinkedIn](https://www.linkedin.com/in/toktam-r-71951996/)

Related projects: [sciwidgets](https://github.com/vahid-anari/sciwidgets) ·
[Maxwell Bloch Studio](https://github.com/vahid-anari/maxwell-bloch-studio) ·
[linecat](https://github.com/vahid-anari/linecat) ·
[Daftarche](https://github.com/vahid-anari/daftarche)
