"""Small-signal gain of the two-level model, for solve_1d and solve_1d_adiabatic.

A weak steady field enters a medium with fixed inversion w0 and R0 = 0. Once the coherence
follows the field, R = -T2 A w0 and dA/dz = eta T2 w0 A, so the field grows as exp(eta T2 w0 z).
The explicit step in z turns this into a factor (1 + eta T2 w0 dz) per step.
"""

import math

import pytest

from mbsolver.two_level import solve_1d, solve_1d_adiabatic

SOLVERS = pytest.mark.parametrize("solve", [solve_1d, solve_1d_adiabatic], ids=["full", "adiabatic"])
A_IN = 1e-6


def _run(solve, *, eta=4.0, t2=0.5, w0=1.0, nz=101):
    """Weak constant input field; returns the result after the coherence has settled."""
    return solve(t_max=20, nt=1001, nz=nz, eta=eta, w0=w0, R0=0.0, field_in=A_IN, t1=50.0, t2=t2)


def _gain(res):
    """Exit-to-entrance intensity ratio at the last time."""
    return res.intensity[-1, -1] / res.intensity[0, -1]


def _discrete_gain(g, nz):
    """Intensity gain of the explicit z step for gain coefficient g = eta T2 w0."""
    return (1.0 + g / (nz - 1)) ** (2 * (nz - 1))


@SOLVERS
def test_matches_discrete_gain(solve):
    """The exit intensity is the entrance intensity times (1 + eta T2 w0 dz)^(2 (nz - 1))."""
    assert _gain(_run(solve)) == pytest.approx(_discrete_gain(2.0, 101), rel=1e-8)


@SOLVERS
def test_gain_depends_on_eta_t2_w0(solve):
    """Only the product eta T2 w0 counts; an absorbing medium (w0 < 0) attenuates."""
    assert _gain(_run(solve, eta=8.0, t2=0.25)) == pytest.approx(_discrete_gain(2.0, 101), rel=1e-8)
    assert _gain(_run(solve, w0=-1.0)) == pytest.approx(_discrete_gain(-2.0, 101), rel=1e-8)


@SOLVERS
def test_converges_to_exponential(solve):
    """The gain tends to exp(2 eta T2 w0) at first order in dz."""
    exact = math.exp(4.0)
    err_101 = abs(_gain(_run(solve, nz=101)) / exact - 1)
    err_401 = abs(_gain(_run(solve, nz=401)) / exact - 1)
    assert err_401 < 0.015
    assert err_101 / err_401 == pytest.approx(4.0, rel=0.05)


@SOLVERS
def test_coherence_follows_field(solve):
    """At the exit the coherence is R = -T2 A w0."""
    res = _run(solve)
    a_out = math.sqrt(res.intensity[-1, -1])
    assert res.R[-1, -1] == pytest.approx(-0.5 * a_out, rel=1e-6)
