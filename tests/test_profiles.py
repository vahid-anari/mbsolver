"""Tests for Pulse and Steps."""

import math

import numpy as np
import pytest

from mbsolver import Pulse, Steps

X = np.linspace(-10, 10, 201)


def test_pulse_symmetric():
    y = Pulse(a=2.0, x0=1.0, w=3.0).evaluate(X)
    np.testing.assert_allclose(y, 2.0 / np.cosh((X - 1.0) / 3.0) ** 2)


def test_pulse_sum_and_asymmetric():
    p = Pulse(a=[1.0, 0.5], x0=[-2.0, 4.0], wl=[1.0, 2.0], wr=[3.0, 0.5])
    left = X < -2.0
    y = p.evaluate(X)
    expected_first = np.where(left, 1 / np.cosh((X + 2) / 1.0) ** 2, 1 / np.cosh((X + 2) / 3.0) ** 2)
    expected_second = 0.5 * np.where(X < 4.0, 1 / np.cosh((X - 4) / 2.0) ** 2,
                                     1 / np.cosh((X - 4) / 0.5) ** 2)
    np.testing.assert_allclose(y, expected_first + expected_second)


def test_pulse_limits():
    assert np.all(Pulse(a=3.0, x0=0.0, w=math.inf).evaluate(X) == 3.0)
    y = Pulse(a=3.0, x0=0.0, w=0.0).evaluate(X)
    assert y[100] == 3.0 and np.count_nonzero(y) == 1


def test_steps():
    s = Steps(y=[1.0, 3.0, 2.0], x0=[-5.0, 5.0], w=[0.0, 2.0])
    y = s.evaluate(X)
    assert y[0] == pytest.approx(1.0, abs=1e-6)
    assert y[50] == pytest.approx(2.0, abs=1e-3)  # sharp step at x = -5 is halfway
    assert y[80] == pytest.approx(3.0, abs=1e-3)
    assert y[150] == pytest.approx(2.5)         # smooth step at x = 5 is halfway
    assert y[-1] == pytest.approx(2.0 + 0.5 * (1 - math.tanh(2.5)))


def test_single_level_is_constant():
    assert np.all(Steps(y=4.0).evaluate(X) == 4.0)


@pytest.mark.parametrize("make, message", [
    (lambda: Pulse(a=[1, 2], x0=[0], w=[1, 1]), "x0 must have 2"),
    (lambda: Pulse(a=1, x0=0), "either w"),
    (lambda: Pulse(a=1, x0=0, w=1, wl=1, wr=1), "not both"),
    (lambda: Pulse(a=1, x0=0, w=-1), "negative"),
    (lambda: Steps(y=[1, 2], x0=[0, 1], w=[1, 1]), "x0 must have 1"),
    (lambda: Steps(y=[]), "at least one level"),
])
def test_invalid(make, message):
    with pytest.raises(ValueError, match=message):
        make()
