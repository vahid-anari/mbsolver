"""Tests for maxbloch.two_level.solve_1d."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from maxbloch import Pulse, Steps
from maxbloch.two_level import solve_1d

DATA = Path(__file__).parent / "data"


def _pulse(p):
    """MB Solver widget value -> Pulse."""
    if p["symmetric"]:
        return Pulse(a=p["a"], x0=p["x0"], w=p["w"])
    return Pulse(a=p["a"], x0=p["x0"], wl=p["wl"], wr=p["wr"])


def _steps(s):
    """MB Solver widget value -> number or Steps (index 0 of x0/w unused)."""
    if not isinstance(s, dict):
        return s
    if s["symmetric"]:
        return Steps(y=s["y"], x0=s["x0"][1:], w=s["w"][1:])
    return Steps(y=s["y"], x0=s["x0"][1:], wl=s["wl"][1:], wr=s["wr"][1:])


def test_matches_mb_solver_8_13():
    """Same results as MB Solver 8.13's own solver for a reference case."""
    p = json.loads((DATA / "reference_params.json").read_text())
    ref = np.load(DATA / "reference_two_level.npz")
    res = solve_1d(
        t_max=p["solve.grid.t_max"], nt=p["solve.grid.nt"], nz=p["solve.grid.nz"],
        eta=p["solve.sample"]["eta"], theta0=p["solve.sample"]["theta0"],
        pump=[_pulse(p[f"solve.pump.pulse{i}"]) for i in (1, 2, 3)],
        field_in=_pulse(p["solve.bcs"]),
        t1=_steps(p["solve.t1"]), t2=_steps(p["solve.t2"]),
        n_planes=min(p["slice.z"]["arr_length"], p["solve.grid.nz"]),
    )
    for ours, theirs in (("time", "time"), ("intensity", "intensity"), ("w", "w"),
                         ("pump", "lambda_n"), ("field_in", "A0"), ("t1", "t1"), ("t2", "t2")):
        np.testing.assert_allclose(getattr(res, ours), ref[theirs], rtol=1e-12, atol=1e-14,
                                   err_msg=ours)

def test_entrance_is_a_bloch_rotation():
    """At z = 0 without relaxation or pump, the field rotates the Bloch angle by minus its area."""
    theta0, a, x0, width = 0.2, 0.05, 40.0, 8.0
    res = solve_1d(t_max=100, nt=4001, nz=3, eta=1.0, theta0=theta0, t1=math.inf, t2=math.inf,
                   field_in=Pulse(a=a, x0=x0, w=width))
    dt = res.time[1] - res.time[0]
    area = np.concatenate(([0.0], np.cumsum(res.field_in[:-1]) * dt))
    np.testing.assert_allclose(res.w[0], np.cos(theta0 - area), atol=1e-9)


def test_no_coupling_leaves_field_unchanged():
    """With eta = 0 the field crosses the medium unchanged."""
    res = solve_1d(t_max=50, nt=501, nz=11, eta=0.0, theta0=0.1, t1=10, t2=5,
                   field_in=Pulse(a=0.1, x0=20, w=5), n_planes=4)
    for plane in res.intensity:
        np.testing.assert_allclose(plane, res.field_in ** 2, rtol=0, atol=1e-15)


def test_relaxation_and_pump_balance():
    """Without a field, w relaxes to w0 + T1 * pump for a constant pump."""
    res = solve_1d(t_max=200, nt=2001, nz=2, eta=1.0, w0=0.0, R0=0.0, t1=10.0, t2=5.0,
                   pump=0.01)
    assert res.w[-1, -1] == pytest.approx(0.1, rel=1e-6)


def test_result_shapes_and_planes():
    res = solve_1d(t_max=10, nt=101, nz=21, eta=1.0, theta0=0.0, t1=1, t2=1, n_planes=5)
    assert res.intensity.shape == res.w.shape == (5, 101)
    np.testing.assert_allclose(res.z, [0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize("kwargs, message", [
    (dict(nt=1), "nt and nz"),
    (dict(n_planes=50), "n_planes"),
    (dict(t1=0.0), "t1 must be positive"),
    (dict(theta0=0.1, w0=1.0, R0=0.0), "not both"),
    (dict(theta0=None), "initial state"),
    (dict(pump="x"), "expected None"),
])
def test_invalid_input(kwargs, message):
    args = dict(t_max=10, nt=11, nz=11, eta=1.0, theta0=0.0, t1=1.0, t2=1.0)
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        solve_1d(**args)
