"""Maxwell-Bloch solvers.

Profiles (pump pulses, boundary fields, relaxation-time schedules) are built
with :class:`Pulse` and :class:`Steps`; the solvers live in model subpackages,
e.g. ``maxbloch.two_level.solve_1d``.
"""

from importlib.metadata import PackageNotFoundError, version

import warnings

from maxbloch.profiles import Pulse, Steps

warnings.warn("maxbloch has been renamed to mbsolver (pip install mbsolver; import mbsolver); "
              "maxbloch gets no further updates", DeprecationWarning, stacklevel=2)

try:
    __version__ = version("maxbloch")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = ["Pulse", "Steps", "__version__"]
