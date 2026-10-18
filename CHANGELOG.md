# Changelog

Each version is a Git tag (`vX.Y.Z`) and is on [PyPI](https://pypi.org/project/mbsolver/).
Pushing a tag publishes a GitHub Release with that version's notes, and Zenodo
then gives the release its own DOI.

## 0.4.0 — 2026-10-17

- `mbsolver.three_level`: the Λ-type three-level model in one dimension (`solve_1d`), with `populations` (initial state from r1 = n3/n1, r2 = n3/n2) and `scaling` (d, ω, η, T₀, E₀, I₀, Λ₀ from the physical values of the two transitions). A reference test matches the 2025 three-level solver exactly.
- `mbsolver.constants.DEBYE`.
- `docs/three_level/` (derivation, equations, initial conditions, CH₃OH data) and `examples/three_level/` (S255IR-NIRS3 flare with the observed flux).
- Toktam Rashidi added as an author (three-level equations).

## 0.3.1 — 2026-10-15

- Citable: `CITATION.cff` and a Zenodo DOI (10.5281/zenodo.23407483 for all versions); Citing section in the README.
- Tests run on GitHub on every push.

## 0.3.0 — 2026-10-15

- `mbsolver.scaling`: the solver's dimensionless parameters (η, θ₀, …) from physical sample values in SI units.

## 0.2.0 — 2026-10-14

- Renamed from `maxbloch` to `mbsolver`; same code. `maxbloch` 0.1.1 on PyPI only adds a deprecation warning.

## 0.1.0 — 2026-10-14

- First version (as `maxbloch`): the one-dimensional two-level solver of the Maxwell Bloch app 8.13, with `Pulse` and `Steps` profiles; a reference test matches the app's results.
