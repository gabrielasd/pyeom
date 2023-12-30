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
    "EOMIPc",
    "EOMIPa",
    "EOMEA",
    "EOMEAc",
    "EOMEAa",
    "EOMDIP",
    "EOMDIP0",
    "EOMExc",
    "EOMExc0",
    "EOMDEA",        
    "EOMDEA0",
    # "EOMExcSA",
]


from .base import EOMState
from .ionization import EOMIP
from .ionization import EOMIPc
from .ionization import EOMIPa
from .eaffinity import EOMEA
from .eaffinity import EOMEAc
from .eaffinity import EOMEAa
from .doubleionization import EOMDIP, EOMDIP0
from .excitation import EOMExc, EOMExc0
from .doubleaffinity import EOMDEA, EOMDEA0
from .spinadapted.particlehole import EOMEE1, EOMEE3


__version__: str = "0.0.1"
r"""EOMEE version string."""
