# mbsolver

[![tests](https://github.com/vahid-anari/mbsolver/actions/workflows/tests.yml/badge.svg)](https://github.com/vahid-anari/mbsolver/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/mbsolver)](https://pypi.org/project/mbsolver/)
[![Python](https://img.shields.io/pypi/pyversions/mbsolver)](https://pypi.org/project/mbsolver/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23407483-blue)](https://doi.org/10.5281/zenodo.23407483)
![License](https://img.shields.io/badge/license-MIT-green)

Fast Maxwell-Bloch solvers in Python (numpy + numba) for astrophysical masers,
superradiance and coherent pulse propagation. Use them from scripts,
notebooks or cluster jobs; the Maxwell Bloch Studio desktop app runs on
the same code.

## Installation

```
pip install mbsolver
```

Requires Python 3.10+ (numpy and numba are installed with it). On Intel Macs numba
stays at 0.62, the last version built for them; with it, Python 3.10–3.13 are supported there.

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

`res.time`, `res.z` (positions of the stored planes), `res.intensity`,
`res.w` and `res.R` (shape `(n_planes, nt)`), `res.pump`, `res.field_in`, `res.t1`, `res.t2`.

### Adiabatic (rate-equation) limit

When `T2` is short compared with the times over which `A` and `w` change, the
coherence follows its steady state, `R = R0 − T2 A w`, and two equations remain:

```
dw/dt = A R0 − T2 A² w − (w − w0) / T1 + Λ(t)
dA/dz = −η R0 + η T2 A w
```

`solve_1d_adiabatic` takes the same arguments as `solve_1d` (with a finite
`t2`) and returns the same result (`res.R` included, as from `solve_1d`), so the two can be compared
directly. The limit differs from the full system during the first few `T2`
and in bursts faster than `T2`; see the section on it in
[`docs/two_level.pdf`](docs/two_level.pdf).

## Three-level model in one dimension

Levels are ordered by energy, `E1 < E2 < E3`. Two of the three transitions
are dipole allowed and carry fields; the third is forbidden:

| `config`   | allowed     | forbidden |
|------------|-------------|-----------|
| `"lambda"` | 3→1, 3→2    | 2↔1       |
| `"v"`      | 2→1, 3→1    | 3↔2       |
| `"ladder"` | 2→1, 3→2    | 3↔1       |

With `n_ij = n_i − n_j`, `W_ij = ½ E_ij R_ij` and fields as Rabi frequencies,
the dimensionless equations along `0 ≤ z ≤ 1` are

```
dn1/dt =  W21 + W31,   dn2/dt = W32 − W21,   dn3/dt = −W31 − W32
dR21/dt = E21 n21 + ½ (E31 R32 + E32 R31) − (R21 − R21⁰) / T2_21
dR31/dt = E31 n31 + ½ (E21 R32 − E32 R21) − (R31 − R31⁰) / T2_31
dR32/dt = E32 n32 − ½ (E21 R31 + E31 R21) − (R32 − R32⁰) / T2_32
dE_ij/dz = η_ij R_ij
```

The field of the forbidden transition is zero. The population differences of
the two allowed transitions relax with their `T1` and are pumped. In Λ and V
the two transitions compete for the shared level; in the ladder, emission on
the upper step raises the inversion of the lower one (a cascade).

```python
import math
from mbsolver.three_level import solve_1d

res = solve_1d(
    config="ladder", t_max=200, nt=2001, nz=101,
    n_init=(0.05, 0.05, 0.9),                 # (n1, n2, n3), summing to 1
    eta21=10.0, eta32=10.0,                   # couplings of the allowed transitions
    c21=1e-3, c32=1e-3,                       # seeds: R_ij⁰ = c_ij · 2√(n_i n_j)
    t1_21=math.inf, t1_32=math.inf,
    t2_21=math.inf, t2_32=math.inf,           # t2 of the forbidden pair: inf by default
)
res.e32[-1], res.e21[-1]                      # upper step bursts first, then the lower one
```

`eta_ij`, `t1_ij`, `pump_ij` and `field_ij_in` are given only for the allowed
transitions; `t2_ij` and `c_ij` exist for all three coherences. Relaxation
times take numbers or `Steps`, pumps and entrance fields `Pulse`s, numbers or
lists, as in the two-level model. The result holds `time`, `z`, the fields
`e21`, `e31`, `e32`, the differences `n21`, `n31`, `n32` and the coherences
`r21`, `r31`, `r32` (shape `(n_planes, nt)`), and the inputs on the time grid.

Time is in units of a free time unit `T0`; each transition has
`η_ij = T0 / T_R` with the cooperation time `T_R = 2ε0ħ / (n0 L k_ij |d_ij|²)`.
`mbsolver.three_level.scaling(t0=, l=, n0=, nu31=, d31=, ...)` computes, for
each transition given by its frequency and its dipole moment (or Einstein
coefficient `gamma_ij`), `eta_ij`, `t_r_ij` and the field and intensity scales
`e0_ij`, `i0_ij`; `mbsolver.constants.DEBYE` converts dipoles from debye.

The three-level equations were derived and checked with Toktam Rashidi. The
derivation, from the Hamiltonian to the dimensionless equations, with the
three configurations, initial conditions and physical scales, is in
[`docs/three_level.pdf`](docs/three_level.pdf).

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
entrance, relaxation balance, uncoupled propagation, small-signal gain). The three-level
Λ case must match a stored run of the earlier Λ-only solver; all three configurations
are checked for purity conservation, Bloch rotation and the reduction to the two-level
model. `scaling`
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
[LinkedIn](https://www.linkedin.com/in/vahid-anari/) ·
[ORCID](https://orcid.org/0000-0002-2464-3989)

Toktam Rashidi ·
[GitHub](https://github.com/toktamrashidi) ·
[LinkedIn](https://www.linkedin.com/in/toktam-r-71951996/) ·
[ORCID](https://orcid.org/0009-0007-2978-7985)

Related projects: [sciwidgets](https://github.com/vahid-anari/sciwidgets) ·
[linecat](https://github.com/vahid-anari/linecat)
