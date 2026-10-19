"""Tests for mbsolver.two_level.solve_1d_adiabatic (rate-equation limit)."""

import math

import numpy as np
import pytest

from mbsolver import Pulse
from mbsolver.two_level import solve_1d, solve_1d_adiabatic

CASE = dict(t_max=300, nt=6001, nz=201, eta=40.0, theta0=0.003, t1=50.0,
            pump=Pulse(a=2e-3, x0=100, w=20), n_planes=2)


def test_short_t2_matches_full_solver():
    """With T2 much shorter than every other time scale, R follows its steady state.

    Only the start-up transient (a few T2, while the full solver builds R) differs.
    """
    full = solve_1d(**CASE, t2=0.05)
    adiabatic = solve_1d_adiabatic(**CASE, t2=0.05)
    later = full.time > 1.0
    np.testing.assert_allclose(adiabatic.w[:, later], full.w[:, later], atol=2e-3)
    np.testing.assert_allclose(adiabatic.intensity[-1, later], full.intensity[-1, later], rtol=2e-2)


@pytest.mark.parametrize("t2", [0.5, 5.0])
def test_late_times_match_full_solver(t2):
    """Once the transients are over, both solvers reach the same state."""
    full = solve_1d(**CASE, t2=t2)
    adiabatic = solve_1d_adiabatic(**CASE, t2=t2)
    np.testing.assert_allclose(adiabatic.intensity[-1, -1], full.intensity[-1, -1], rtol=1e-3)
    np.testing.assert_allclose(adiabatic.w[-1, -1], full.w[-1, -1], rtol=1e-3, atol=1e-6)


def test_coherence_is_the_steady_state():
    """R returned is R0 - T2 A w on every stored plane."""
    res = solve_1d_adiabatic(t_max=50, nt=501, nz=11, eta=2.0, w0=0.9, R0=0.01, t1=10.0, t2=0.3,
                             field_in=Pulse(a=0.05, x0=20, w=5), n_planes=11)
    a = np.sqrt(res.intensity[0])
    np.testing.assert_allclose(res.R[0], 0.01 - 0.3 * res.field_in * res.w[0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(a, np.abs(res.field_in), rtol=1e-12)
    assert res.R.shape == res.w.shape == (11, 501)


def test_stable_for_a_strong_field():
    """T2 A² dt far beyond the explicit limit still gives finite, bounded populations."""
    res = solve_1d_adiabatic(t_max=20, nt=201, nz=5, eta=1.0, w0=1.0, R0=0.0, t1=math.inf, t2=1.0,
                             field_in=50.0)
    assert np.all(np.isfinite(res.w)) and np.all(np.abs(res.w) <= 1.0 + 1e-12)


def test_full_solver_returns_no_coherence():
    assert solve_1d(**CASE, t2=0.5).R is None


def test_infinite_t2_is_refused():
    with pytest.raises(ValueError, match="t2 must be finite"):
        solve_1d_adiabatic(**CASE, t2=math.inf)
