"""Tests for mbsolver.three_level."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mbsolver import Pulse
from mbsolver.constants import DEBYE
from mbsolver.three_level import populations, scaling, solve_1d

DATA = Path(__file__).parent / "data"


def test_matches_three_level_solver_2025():
    """Same results as the 2025 three-level solver (v7) for the S255IR-NIRS3 case."""
    p = json.loads((DATA / "reference_three_level_params.json").read_text())
    ref = np.load(DATA / "reference_three_level.npz")
    a, x0, w = p.pop("pump32")
    res = solve_1d(**p, pump32=Pulse(a=a, x0=x0, w=w))
    for name in ("time", "pump31", "pump32"):
        np.testing.assert_allclose(getattr(res, name), ref[name], rtol=1e-12, atol=0, err_msg=name)
    for name in ("e31", "e32", "intensity31", "intensity32", "n31", "n32", "r31", "r32", "r21"):
        np.testing.assert_allclose(getattr(res, name)[-1], ref[name], rtol=1e-12, atol=0,
                                   err_msg=name)


BASE = dict(t_max=50, nt=501, nz=11, d=0.9, omega=2.0, r1=0.9, r2=5.0, c31=1e-3, c32=1e-2,
            t1_31=10, t1_32=20, t2_31=5, t2_32=8)


def test_no_coupling_leaves_fields_unchanged():
    """With η = 0 for 3→1 and no medium for 3→2 the entrance fields pass unchanged."""
    res = solve_1d(**{**BASE, "r1": 1.0, "r2": 1.0, "c31": 0.0, "c32": 0.0},
                   field31_in=Pulse(a=0.05, x0=20, w=5), n_planes=4)
    for plane in res.e31:
        np.testing.assert_allclose(plane, res.field31_in, rtol=0, atol=1e-15)


def test_relaxation_and_pump_balance():
    """Without fields, n relaxes to n⁰ + T1 · Λ for a constant pump."""
    pops = populations(BASE["r1"], BASE["r2"])
    res = solve_1d(**{**BASE, "c31": 0.0, "c32": 0.0, "t_max": 400, "nt": 4001, "nz": 2},
                   pump31=1e-3, pump32=2e-3)
    assert res.n31[-1, -1] == pytest.approx(pops["n31"] + 10 * 1e-3, rel=1e-6)
    assert res.n32[-1, -1] == pytest.approx(pops["n32"] + 20 * 2e-3, rel=1e-6)


def test_entrance_follows_the_bloch_equations():
    """At z = 0 a weak 3→2 field rotates R32 like a two-level system (n32 nearly fixed)."""
    res = solve_1d(**{**BASE, "c31": 0.0, "c32": 0.0, "t1_31": math.inf, "t1_32": math.inf,
                      "t2_31": math.inf, "t2_32": math.inf, "nt": 4001, "nz": 2},
                   field32_in=Pulse(a=1e-4, x0=20, w=5))
    dt = res.time[1] - res.time[0]
    area = np.concatenate(([0.0], np.cumsum(res.field32_in[:-1]) * dt))
    n32 = populations(BASE["r1"], BASE["r2"])["n32"]
    np.testing.assert_allclose(res.r32[0], n32 * area, rtol=1e-3, atol=1e-12)


def test_result_shapes_and_planes():
    res = solve_1d(**{**BASE, "nz": 21}, n_planes=5)
    for name in ("e31", "e32", "intensity31", "intensity32", "n31", "n32", "r31", "r32", "r21"):
        assert getattr(res, name).shape == (5, 501), name
    np.testing.assert_allclose(res.z, [0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(res.intensity31, res.e31 ** 2 / BASE["d"] ** 2)


@pytest.mark.parametrize("kwargs, message", [
    (dict(nt=1), "nt and nz"),
    (dict(n_planes=50), "n_planes"),
    (dict(d=0.0), "d and omega"),
    (dict(t2_21=0.0), "t2_21 must be positive"),
    (dict(r1=-1.0), "must not be negative"),
    (dict(pump32="x"), "expected None"),
])
def test_invalid_input(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        solve_1d(**{**BASE, **kwargs})


def test_populations():
    p = populations(0.94, 2.33e19)
    assert p["n1"] + p["n2"] + p["n3"] == pytest.approx(1.0, rel=1e-15)
    assert p["n31"] == pytest.approx(p["n3"] - p["n1"], abs=1e-15)
    assert p["n32"] == pytest.approx(p["n3"] - p["n2"], abs=1e-15)
    assert p["r31"] == pytest.approx(2 * math.sqrt(p["n3"] * p["n1"]), rel=1e-12)
    # values in the paper's S255IR-NIRS3 case
    assert p["n31"] == pytest.approx(-0.0309, abs=1e-4)
    assert p["n32"] == pytest.approx(0.4845, abs=1e-4)


# S255IR-NIRS3 values of the 2025 plotting script
S255 = dict(nu31=239_746.219e6, nu32=6_668.564e6, d31=0.59425 * DEBYE, d32=0.67936 * DEBYE,
            l=2.1e13, n0=1.17e-6 * (1 / 0.94 + 1))


def test_scaling_paper_values():
    s = scaling(**S255)
    assert s["d"] == pytest.approx(0.59425 / 0.67936, rel=1e-15)
    assert s["omega"] == pytest.approx(239_746.219 / 6_668.564, rel=1e-15)
    assert s["eta"] == pytest.approx(27.51, abs=0.01)
    assert s["t0"] == pytest.approx(5.13e4, rel=1e-3)
    assert scaling(**S255, gamma=1.56e-9)["t0"] == pytest.approx(5.24e4, rel=1e-3)


def test_scaling_gamma_from_d32_gives_the_same_t0():
    """With Γ = Einstein A from d32, both forms of T0 agree."""
    s = scaling(**S255)
    assert scaling(**S255, gamma=s["gamma"])["t0"] == pytest.approx(s["t0"], rel=1e-13)
    assert s["lambda0"] == pytest.approx(S255["n0"] / s["t0"], rel=1e-15)
    assert s["i0_jy"] == pytest.approx(1e26 * s["i0"], rel=1e-15)
