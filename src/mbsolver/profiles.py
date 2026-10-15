"""Time profiles: sums of sech² pulses and smoothed steps.

``Pulse`` describes pump pulses and boundary fields, ``Steps`` describes
quantities that change between levels, such as relaxation times. Both are
evaluated on a grid with numba.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from numba import njit

ArrayLike = Union[float, Sequence[float], np.ndarray]


def _array(name: str, values: Optional[ArrayLike]) -> Optional[np.ndarray]:
    """Return ``values`` as a 1-D float array (``None`` stays ``None``)."""
    if values is None:
        return None
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a number or a 1-D sequence")
    return arr


def _widths(n: int, w, wl, wr) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Check the width arguments and return ``(symmetric, w, wl, wr)`` arrays of length ``n``."""
    w, wl, wr = _array("w", w), _array("wl", wl), _array("wr", wr)
    if w is not None:
        if wl is not None or wr is not None:
            raise ValueError("give either w, or wl and wr, not both")
        symmetric, wl, wr = True, w, w
    elif wl is not None and wr is not None:
        symmetric, w = False, wl
    else:
        raise ValueError("give either w, or both wl and wr")
    for name, arr in (("w", w), ("wl", wl), ("wr", wr)):
        if arr.shape[0] != n:
            raise ValueError(f"{name} must have {n} values, got {arr.shape[0]}")
        if np.any(arr < 0):
            raise ValueError(f"{name} must not be negative")
    return symmetric, w, wl, wr


@dataclass(frozen=True, init=False)
class Pulse:
    """A sum of sech² pulses.

    ``y(x) = sum_i a_i * sech²((x - x0_i) / w_i)``; with ``wl`` and ``wr``
    instead of ``w``, each pulse has its own width on each side of ``x0_i``.
    A zero width gives ``a_i`` only at ``x0_i``; an infinite width a constant.

    Args:
        a: Amplitudes.
        x0: Centres.
        w: Widths (symmetric pulses).
        wl, wr: Left and right widths (asymmetric pulses).

    Example:
        ``Pulse(a=[0.1, 0.05], x0=[50, 120], w=[10, 20])``
    """

    a: np.ndarray
    x0: np.ndarray
    w: np.ndarray
    wl: np.ndarray
    wr: np.ndarray
    symmetric: bool

    def __init__(self, a: ArrayLike, x0: ArrayLike, w: Optional[ArrayLike] = None,
                 wl: Optional[ArrayLike] = None, wr: Optional[ArrayLike] = None) -> None:
        a, x0 = _array("a", a), _array("x0", x0)
        n = a.shape[0]
        if x0.shape[0] != n:
            raise ValueError(f"x0 must have {n} values (one per amplitude), got {x0.shape[0]}")
        symmetric, w, wl, wr = _widths(n, w, wl, wr)
        for name, value in (("a", a), ("x0", x0), ("w", w), ("wl", wl), ("wr", wr),
                            ("symmetric", symmetric)):
            object.__setattr__(self, name, value)

    def evaluate(self, x: ArrayLike) -> np.ndarray:
        """Return the profile on the coordinates ``x``."""
        x = np.asarray(x, dtype=np.float64)
        return pulse_func(self.symmetric, x, self.a, self.x0, self.w, self.wl, self.wr)


@dataclass(frozen=True, init=False)
class Steps:
    """Smoothed steps between levels.

    Starts at ``y[0]`` and changes to ``y[i]`` around ``x0[i-1]``:
    ``y(x) = y_0 + sum_i (y_i - y_{i-1}) * (1 + tanh((x - x0_i) / w_i)) / 2``.
    A zero width gives a sharp step; one level gives a constant.

    Args:
        y: Levels (``N`` values).
        x0: Step positions (``N - 1`` values).
        w: Step widths (``N - 1`` values, symmetric steps).
        wl, wr: Left and right widths (asymmetric steps).

    Example:
        ``Steps(y=[0.5, 0.2], x0=[100], w=[10])``: 0.5 before t = 100, 0.2 after.
    """

    y: np.ndarray
    x0: np.ndarray
    w: np.ndarray
    wl: np.ndarray
    wr: np.ndarray
    symmetric: bool

    def __init__(self, y: ArrayLike, x0: ArrayLike = (), w: Optional[ArrayLike] = None,
                 wl: Optional[ArrayLike] = None, wr: Optional[ArrayLike] = None) -> None:
        y, x0 = _array("y", y), _array("x0", x0)
        n = y.shape[0] - 1
        if n < 0:
            raise ValueError("y must contain at least one level")
        if x0.shape[0] != n:
            raise ValueError(f"x0 must have {n} values (one per step), got {x0.shape[0]}")
        if n == 0 and w is None and wl is None and wr is None:
            w = ()
        symmetric, w, wl, wr = _widths(n, w, wl, wr)
        for name, value in (("y", y), ("x0", x0), ("w", w), ("wl", wl), ("wr", wr),
                            ("symmetric", symmetric)):
            object.__setattr__(self, name, value)

    def evaluate(self, x: ArrayLike) -> np.ndarray:
        """Return the profile on the coordinates ``x``."""
        x = np.asarray(x, dtype=np.float64)
        pad = np.zeros(1)
        return step_func(self.symmetric, x, self.y,
                         np.concatenate((pad, self.x0)), np.concatenate((pad, self.w)),
                         np.concatenate((pad, self.wl)), np.concatenate((pad, self.wr)))


def evaluate(profile: Union[None, float, Pulse, Steps], x: np.ndarray) -> np.ndarray:
    """Evaluate a profile argument of a solver on the grid ``x``.

    ``None`` gives zeros, a number a constant, a ``Pulse`` or ``Steps`` its values.
    """
    if profile is None:
        return np.zeros(np.shape(x), dtype=np.float64)
    if isinstance(profile, (Pulse, Steps)):
        return profile.evaluate(x)
    if isinstance(profile, (int, float, np.integer, np.floating)) and not isinstance(profile, bool):
        return np.full(np.shape(x), float(profile), dtype=np.float64)
    raise TypeError(f"expected None, a number, Pulse or Steps, got {type(profile).__name__}")


# -----------------------------------------------------------------------------
# numba kernels (arguments as in MB Solver's widgets)
# -----------------------------------------------------------------------------
@njit(nogil=True, cache=True)
def pulse_func(symmetric, x, a, x0, w, wl, wr):
    """Sum of sech² components ``a_i sech²((x - x0_i) / w_i)`` on ``x``.

    All of ``a, x0, w, wl, wr`` have length N; ``w`` is used when
    ``symmetric``, otherwise ``wl`` left and ``wr`` right of ``x0_i``.
    """
    n = a.shape[0]
    m = x.shape[0]
    if x0.shape[0] != n or w.shape[0] != n or wl.shape[0] != n or wr.shape[0] != n:
        raise ValueError("x0, w, wl and wr must have the same length as a")

    y = np.zeros(m, dtype=np.float64)

    for i in range(n):
        ai = a[i]
        if ai == 0.0:
            continue

        x0i = x0[i]
        wi = w[i]
        wli = wl[i]
        wri = wr[i]

        for j in range(m):
            dx = x[j] - x0i
            width = wi if symmetric else wli if dx < 0.0 else wri

            if width == 0.0:
                if dx == 0.0:
                    y[j] += ai
            elif np.isinf(width):
                y[j] += ai
            else:
                t = np.tanh(dx / width)
                y[j] += ai * (1.0 - t * t)

    return y


@njit(nogil=True, cache=True)
def step_func(symmetric, x, levels, x0, w, wl, wr):
    """Smoothed steps between ``levels`` on ``x``.

    ``levels`` has length N; ``x0, w, wl, wr`` also have length N with
    index 0 unused (the layout of MB Solver's step widgets).
    """
    n = levels.shape[0]
    m = x.shape[0]
    if n == 0:
        raise ValueError("levels must contain at least y^0")
    if x0.shape[0] != n or w.shape[0] != n or wl.shape[0] != n or wr.shape[0] != n:
        raise ValueError("x0, w, wl and wr must have the same length as levels")

    y = np.empty(m, dtype=np.float64)
    y[:] = levels[0]

    for i in range(1, n):
        di = levels[i] - levels[i - 1]
        if di == 0.0:
            continue

        x0i = x0[i]
        wi = w[i]
        wli = wl[i]
        wri = wr[i]

        for j in range(m):
            dx = x[j] - x0i
            width = wi if symmetric else wli if dx < 0.0 else wri

            if width == 0.0:
                if dx > 0.0:
                    y[j] += di
                elif dx == 0.0:
                    y[j] += 0.5 * di
            elif np.isinf(width):
                y[j] += 0.5 * di
            else:
                y[j] += 0.5 * di * (1.0 + np.tanh(dx / width))

    return y
