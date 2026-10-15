"""Maxwell-Bloch solvers.

Profiles (pump pulses, boundary fields, relaxation-time schedules) are built
with :class:`Pulse` and :class:`Steps`; the solvers live in model subpackages,
e.g. ``mbsolver.two_level.solve_1d``. :func:`scaling` turns the physical
sample values into the solvers' ``eta`` and ``theta0``.
"""

from importlib.metadata import PackageNotFoundError, version

from mbsolver.profiles import Pulse, Steps
from mbsolver.sample import scaling

try:
    __version__ = version("mbsolver")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = ["Pulse", "Steps", "scaling", "__version__"]
