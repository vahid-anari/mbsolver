# mbsolver

[![tests](https://github.com/vahid-anari/mbsolver/actions/workflows/tests.yml/badge.svg)](https://github.com/vahid-anari/mbsolver/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/mbsolver)](https://pypi.org/project/mbsolver/)
[![Python](https://img.shields.io/pypi/pyversions/mbsolver)](https://pypi.org/project/mbsolver/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23407483.svg)](https://doi.org/10.5281/zenodo.23407483)

Fast Maxwell-Bloch solvers in Python (numpy + numba) for astrophysical masers,
superradiance and coherent pulse propagation. Use them from scripts,
notebooks or cluster jobs; the
[Maxwell Bloch Studio](https://github.com/vahid-anari/maxwell-bloch-studio)
desktop app runs on the same code.

```
pip install mbsolver
```

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

## From physical values: `scaling`

The solvers take dimensionless parameters. `scaling` computes them from the
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

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests include a reference run that must match the solver of the Maxwell
Bloch app (version 8.13) exactly, and physics checks (Bloch rotation at the
entrance, relaxation balance, uncoupled propagation). `scaling` is checked
against the values of Maxwell Bloch Studio.

## Earlier name

Up to 0.1.0 this package was called `maxbloch`; 0.2.0 is the same code under
the new name. `maxbloch` stays on PyPI unchanged (0.1.1 only adds a
deprecation warning) but gets no updates; change `maxbloch` to `mbsolver` in
your imports.

## Citing

If you use mbsolver in published work, please cite it:

> Anari, V. *mbsolver: Fast Maxwell-Bloch solvers for masers, superradiance and
> coherent pulse propagation*. Zenodo. https://doi.org/10.5281/zenodo.23407483

This DOI always points to the latest version. Each release also has its own
DOI (for 0.3.1: [10.5281/zenodo.23407484](https://doi.org/10.5281/zenodo.23407484));
cite the version you used, which you find under *Versions* on the Zenodo page.
GitHub's **Cite this repository** button gives the same in BibTeX and APA.

## License

MIT
