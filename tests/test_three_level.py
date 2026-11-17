"""Tests for mbsolver.three_level."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mbsolver import Pulse
from mbsolver.constants import C, DEBYE, EPSILON_0, H_BAR, PI
from mbsolver.three_level import CONFIGS, scaling, solve_1d
from mbsolver.two_level import solve_1d as solve_two_level

DATA = Path(__file__).parent / "data"
INF = math.inf


def test_lambda_matches_reference_run():
    """The Λ case reproduces a stored run of the earlier Λ-only solver."""
    p = json.loads((DATA / "reference_three_level_params.json").read_text())
    ref = np.load(DATA / "reference_three_level.npz")
    r1, r2 = p["r1"], p["r2"]
    c = r1 + r2 + r1 * r2
    a, x0, w = p["pump32"]
    res = solve_1d(config="lambda", t_max=p["t_max"], nt=p["nt"], nz=p["nz"],
                   n_init=(r2 / c, r1 / c, r1 * r2 / c),
                   eta31=p["omega"] * p["d"] * p["d"], eta32=1.0,
                   c21=p["c21"], c31=p["c31"], c32=p["c32"],
                   t1_31=p["t1_31"], t1_32=p["t1_32"], t2_31=p["t2_31"], t2_32=p["t2_32"],
                   pump32=Pulse(a=a, x0=x0, w=w))
    np.testing.assert_allclose(res.time, ref["time"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(res.pump32, ref["pump32"], rtol=1e-12, atol=0)
    for name in ("e31", "e32", "n31", "n32", "r31", "r32", "r21"):
        np.testing.assert_allclose(getattr(res, name)[-1], ref[name], rtol=1e-12, atol=0,
                                   err_msg=name)
    assert np.all(res.e21 == 0.0)


def _base(config, **kw):
    """Keyword arguments for ``config`` with the allowed parameters filled in."""
    a, b = CONFIGS[config]
    args = dict(config=config, t_max=50, nt=501, nz=11, n_init=(0.2, 0.3, 0.5),
                **{f"eta{a}": 2.0, f"eta{b}": 1.5, f"c{a}": 1e-3, f"c{b}": 2e-3,
                   f"t1_{a}": 10.0, f"t1_{b}": 20.0, f"t2_{a}": 5.0, f"t2_{b}": 8.0})
    args.update(kw)
    return args


ALL = pytest.mark.parametrize("config", sorted(CONFIGS))


@ALL
def test_forbidden_field_stays_zero(config):
    a, b = CONFIGS[config]
    forbidden = ({"21", "31", "32"} - {a, b}).pop()
    res = solve_1d(**_base(config, nz=21), n_planes=5)
    assert np.all(getattr(res, f"e{forbidden}") == 0.0)
    for name in ("pump", "field_in_", "t1_"):
        attr = f"field{forbidden}_in" if name == "field_in_" else f"{name}{forbidden}"
        assert getattr(res, attr) is None, attr
    np.testing.assert_allclose(getattr(res, f"t2_{forbidden}"), INF)


@ALL
def test_differences_are_consistent(config):
    res = solve_1d(**_base(config), n_planes=3)
    np.testing.assert_allclose(res.n31, res.n32 + res.n21, rtol=0, atol=1e-15)
    assert res.n21[0, 0] == pytest.approx(0.1)
    assert res.n31[0, 0] == pytest.approx(0.3)
    assert res.n32[0, 0] == pytest.approx(0.2)


@ALL
def test_no_coupling_leaves_fields_unchanged(config):
    """With η = 0 the entrance fields pass the medium unchanged."""
    a, b = CONFIGS[config]
    res = solve_1d(**_base(config, **{f"eta{a}": 0.0, f"eta{b}": 0.0}), n_planes=4,
                   **{f"field{a}_in": Pulse(a=0.05, x0=20, w=5), f"field{b}_in": 0.01})
    for name in (a, b):
        for plane in getattr(res, f"e{name}"):
            np.testing.assert_allclose(plane, getattr(res, f"field{name}_in"), rtol=0, atol=1e-15)


@ALL
def test_relaxation_and_pump_balance(config):
    """Without fields, each relaxed difference tends to n⁰ + T1 · Λ for a constant pump."""
    a, b = CONFIGS[config]
    res = solve_1d(**_base(config, **{f"c{a}": 0.0, f"c{b}": 0.0}, t_max=400, nt=4001, nz=2),
                   **{f"pump{a}": 1e-3, f"pump{b}": 2e-3})
    n0 = {"21": 0.1, "31": 0.3, "32": 0.2}
    assert getattr(res, f"n{a}")[-1, -1] == pytest.approx(n0[a] + 10 * 1e-3, rel=1e-6)
    assert getattr(res, f"n{b}")[-1, -1] == pytest.approx(n0[b] + 20 * 2e-3, rel=1e-6)


@ALL
def test_entrance_follows_the_bloch_equations(config):
    """At z = 0 a weak field rotates its coherence like a two-level system."""
    a, b = CONFIGS[config]
    n0 = {"21": 0.1, "31": 0.3, "32": 0.2}
    for p in (a, b):
        res = solve_1d(**_base(config, **{f"c{a}": 0.0, f"c{b}": 0.0, f"t1_{a}": INF, f"t1_{b}": INF,
                                          f"t2_{a}": INF, f"t2_{b}": INF}, nt=4001, nz=2),
                       **{f"field{p}_in": Pulse(a=1e-4, x0=20, w=5)})
        field = getattr(res, f"field{p}_in")
        dt = res.time[1] - res.time[0]
        area = np.concatenate(([0.0], np.cumsum(field[:-1]) * dt))
        np.testing.assert_allclose(getattr(res, f"r{p}")[0], n0[p] * area, rtol=1e-3, atol=1e-12)


@ALL
def test_purity_is_conserved_at_the_entrance(config):
    """Without relaxation, strong fields on both transitions keep Tr ρ² constant at z = 0.

    This checks the cross terms that couple the three coherences.
    """
    a, b = CONFIGS[config]
    res = solve_1d(**_base(config, **{f"t1_{a}": INF, f"t1_{b}": INF, f"t2_{a}": INF, f"t2_{b}": INF},
                         c21=0.3, c31=0.2, c32=0.4, t_max=20, nt=20001, nz=2),
                   **{f"field{a}_in": Pulse(a=0.8, x0=8, w=3), f"field{b}_in": Pulse(a=0.6, x0=11, w=2)})
    n21, n31 = res.n21[0], res.n31[0]
    n1 = (1.0 - n21 - n31) / 3.0
    n2, n3 = n1 + n21, n1 + n31
    purity = n1**2 + n2**2 + n3**2 + 0.5 * (res.r21[0]**2 + res.r31[0]**2 + res.r32[0]**2)
    assert np.ptp(purity) < 1e-9
    assert np.ptp(n1 + n2 + n3) < 1e-12
    assert np.max(np.abs(res.r21[0] - res.r21[0, 0])) > 1e-2  # the coupling acted


@ALL
@pytest.mark.parametrize("which", [0, 1])
def test_reduces_to_two_level(config, which):
    """With one allowed transition idle, the other follows the two-level model (A' = -E)."""
    a, b = CONFIGS[config]
    p, idle = (a, b) if which == 0 else (b, a)
    n0 = {"21": 0.1, "31": 0.3, "32": 0.2}
    pops = {"1": 0.2, "2": 0.3, "3": 0.5}
    c_p = 1e-3
    r0 = c_p * 2 * math.sqrt(pops[p[0]] * pops[p[1]])
    pump = Pulse(a=2e-3, x0=20, w=6)
    res = solve_1d(**_base(config, **{f"c{idle}": 0.0, f"c{p}": c_p, f"eta{p}": 6.0}, t_max=60,
                         nt=1201, nz=41), n_planes=3, **{f"pump{p}": pump, f"field{p}_in": 1e-4})
    two = solve_two_level(t_max=60, nt=1201, nz=41, eta=6.0, w0=n0[p], R0=r0,
                          t1=10.0 if p == a else 20.0, t2=5.0 if p == a else 8.0,
                          pump=pump, field_in=-1e-4, n_planes=3)
    np.testing.assert_allclose(getattr(res, f"n{p}"), two.w, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(getattr(res, f"r{p}"), two.R, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(getattr(res, f"e{p}") ** 2, two.intensity, rtol=1e-9, atol=1e-16)
    assert np.all(getattr(res, f"e{idle}") == 0.0)
    assert np.all(getattr(res, f"r{idle}") == 0.0)
    forbidden = ({"21", "31", "32"} - {a, b}).pop()
    assert np.all(getattr(res, f"r{forbidden}") == 0.0)


def test_ladder_cascade_feeds_the_lower_step():
    """In the ladder, emission on 3→2 fills level 2 and raises the inversion n21 of 2→1."""
    res = solve_1d(config="ladder", t_max=60, nt=3001, nz=101, n_init=(0.0, 0.05, 0.95),
                   eta21=8.0, eta32=8.0, c21=0.0, c32=1e-3,
                   t1_21=INF, t1_32=INF, t2_21=INF, t2_32=INF)
    assert res.n21[-1].max() > 0.3
    assert res.n32[-1].min() < 0.5


def test_lambda_competition_lowers_both():
    """In Λ, emission on 3→2 empties level 3 and lowers n31 too."""
    res = solve_1d(config="lambda", t_max=60, nt=3001, nz=101, n_init=(0.3, 0.05, 0.65),
                   eta31=1e-9, eta32=8.0, c31=0.0, c32=1e-3,
                   t1_31=INF, t1_32=INF, t2_31=INF, t2_32=INF)
    assert res.n32[-1, 0] == pytest.approx(0.6)
    assert res.n32[-1].min() < 0.2
    assert res.n31[-1].min() < 0.35 - 0.15


def test_result_shapes_and_planes():
    res = solve_1d(**_base("v", nz=21), n_planes=5)
    for name in ("e21", "e31", "e32", "n21", "n31", "n32", "r21", "r31", "r32"):
        assert getattr(res, name).shape == (5, 501), name
    np.testing.assert_allclose(res.z, [0, 0.25, 0.5, 0.75, 1.0])
    assert res.config == "v"


@pytest.mark.parametrize("kwargs, message", [
    (dict(config="delta"), "config must be one of"),
    (dict(nt=1), "nt and nz"),
    (dict(n_planes=50), "n_planes"),
    (dict(eta21=1.0), "eta21 is not allowed"),
    (dict(t1_21=1.0), "t1_21 is not allowed"),
    (dict(pump21=1e-3), "pump21 is not allowed"),
    (dict(field21_in=1e-3), "field21_in is not allowed"),
    (dict(eta31=None), "eta31 is required"),
    (dict(t2_32=None), "t2_32 is required"),
    (dict(eta32=-1.0), "eta32 must not be negative"),
    (dict(t2_21=0.0), "t2_21 must be positive"),
    (dict(n_init=(0.5, 0.5, 0.5)), "n_init"),
    (dict(n_init=(-0.1, 0.6, 0.5)), "n_init"),
    (dict(pump32="x"), "expected None"),
])
def test_invalid_input(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        solve_1d(**{**_base("lambda"), **kwargs})


def test_scaling_dipole_and_gamma_agree():
    """A transition given by its dipole or by its Einstein coefficient gives the same scales."""
    nu, d = 3.0e9, 0.5 * DEBYE
    s = scaling(t0=1e4, l=1e12, n0=1e-6, nu31=nu, d31=d)
    g = scaling(t0=1e4, l=1e12, n0=1e-6, nu31=nu, gamma31=s["gamma31"])
    for key in ("eta31", "t_r31", "d31", "e0_31", "i0_31"):
        assert g[key] == pytest.approx(s[key], rel=1e-13), key
    k = 2 * PI * nu / C
    assert s["t_r31"] == pytest.approx(2 * EPSILON_0 * H_BAR / (1e-6 * 1e12 * k * d**2), rel=1e-13)
    assert s["eta31"] == pytest.approx(1e4 / s["t_r31"], rel=1e-15)
    assert s["i0_31"] == pytest.approx(0.5 * EPSILON_0 * C * s["e0_31"] ** 2, rel=1e-15)
    assert s["lambda0"] == pytest.approx(1e-6 / 1e4, rel=1e-15)


def test_scaling_ratio_is_fixed_by_the_transitions():
    s = scaling(t0=1.0, l=1e10, n0=1e-3, nu21=1e9, d21=1 * DEBYE, nu32=4e9, d32=0.5 * DEBYE)
    assert s["eta32"] / s["eta21"] == pytest.approx(0.25 * 4.0, rel=1e-13)
    t = scaling(t0=s["t_r21"], l=1e10, n0=1e-3, nu21=1e9, d21=1 * DEBYE)
    assert t["eta21"] == pytest.approx(1.0, rel=1e-15)
    assert "eta31" not in s


@pytest.mark.parametrize("kwargs, message", [
    (dict(nu21=1e9), "exactly one"),
    (dict(nu21=1e9, d21=DEBYE, gamma21=1e-9), "exactly one"),
    (dict(d21=DEBYE), "without nu21"),
    (dict(), "at least one"),
    (dict(nu21=1e9, d21=DEBYE, t0=0.0), "must be positive"),
])
def test_scaling_invalid_input(kwargs, message):
    args = dict(t0=1.0, l=1.0, n0=1.0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        scaling(**args)
