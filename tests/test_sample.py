"""Tests for mbsolver.scaling (physical sample -> solver parameters)."""

import math

import pytest

import mbsolver
from mbsolver.constants import C, PI

SAMPLE = dict(nu=6668.519e6, gamma=1.56e-9, t0=86400.0, l=7.5e12, n0=5.3e-7)

# Values from Maxwell Bloch Studio 10.0.3 (mbstudio.scaling) for SAMPLE.
REFERENCE = {
    "N": 1.340262006320294e+18,
    "b0": 1.8155752119506778e-18,
    "d": 2.2424742818774253e-30,
    "e0": 5.442957554745647e-10,
    "eta": 0.12925265980137085,
    "i0_electric": 3.9319621867605606e-22,
    "i0_magnetic": 3.9319621867605615e-22,
    "k": 139.76182349940206,
    "lambda": 0.04495637757049204,
    "lambda0": 1.1574074074074073e-05,
    "m0": 3.5630674479188363e-28,
    "mu": 6.722768769658182e-22,
    "omega": 41899540601.44791,
    "p0": 1.1885113693950354e-36,
    "t_r": 668458.1975548919,
    "theta0": 1.7275679662364094e-09,
}


def test_matches_studio():
    s = mbsolver.scaling(**SAMPLE)
    assert sorted(s) == sorted(REFERENCE)
    for key, value in REFERENCE.items():
        assert s[key] == pytest.approx(value, rel=1e-13), key


def test_relations():
    s = mbsolver.scaling(**SAMPLE)
    assert s["eta"] == pytest.approx(SAMPLE["t0"] / s["t_r"], rel=1e-15)
    assert s["theta0"] == pytest.approx(2.0 / math.sqrt(s["N"]), rel=1e-15)
    assert s["lambda"] == pytest.approx(C / SAMPLE["nu"], rel=1e-15)
    assert s["k"] == pytest.approx(2 * PI / s["lambda"], rel=1e-14)
    # electric and magnetic dipole give the same intensity scale
    assert s["i0_electric"] == pytest.approx(s["i0_magnetic"], rel=1e-12)


def test_eta_scales_with_length_and_density():
    s1 = mbsolver.scaling(**SAMPLE)
    s2 = mbsolver.scaling(**{**SAMPLE, "l": 2 * SAMPLE["l"], "n0": 3 * SAMPLE["n0"]})
    assert s2["eta"] == pytest.approx(6 * s1["eta"], rel=1e-14)


def test_feeds_the_solver():
    from mbsolver.two_level import solve_1d

    s = mbsolver.scaling(**SAMPLE)
    res = solve_1d(t_max=10, nt=101, nz=11, eta=s["eta"], theta0=s["theta0"], t1=math.inf, t2=math.inf)
    assert res.intensity.shape == (2, 101)
