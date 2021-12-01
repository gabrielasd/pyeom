# This file is part of EOMEE.
#
# EOMEE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# EOMEE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

r"""Equations-of-Motion and Extended RPA package."""


__all__ = [
    "__version__",
    "EOMState",
    "EOMIP",
    "EOMIPDoubleCommutator",
    "EOMIPAntiCommutator",
    "EOMEA",
    "EOMEADoubleCommutator",
    "EOMEAAntiCommutator",
    "EOMDIP",
    "EOMExc",
    "EOMDEA",
    "EOMDIPCIS",
    "EOMDEA_2",
    "EOMExc0",
]


from .base import EOMState
from .ionization import EOMIP
from .ionization import EOMIPDoubleCommutator
from .ionization import EOMIPAntiCommutator
from .electronaff import EOMEA
from .electronaff import EOMEADoubleCommutator
from .electronaff import EOMEAAntiCommutator
from .doubleionization import EOMDIP, EOMDIPCIS
from .excitation import EOMExc, EOMExc0
from .doubleelectronaff import EOMDEA, EOMDEA_2


__version__: str = "0.0.1"
r"""EOMEE version string."""
