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
    "IP",
    "IPc",
    "IPa",
    "IPcm",
    "IPam",
    "EA",
    "EAc",
    "EAa",
    "EOMDIP",
    "EOMDIP0",
    "EE",
    "EEm",
    "EOMExc0SA",
    "EOMEE1",
    "EOMEE3",
    "EOMDEA",        
    "EOMDEA0",
]


from .base import EOMState
from .eomip import IP, IPc, IPa, IPcm, IPam
from .eomea import EA, EAc, EAa
from .doubleionization import EOMDIP, EOMDIP0
from .excitation import EE, EEm
from .spinadapted.particlehole import EOMExc0SA
from .doubleaffinity import EOMDEA, EOMDEA0
from .spinadapted.particlehole import EOMEE1, EOMEE3


__version__: str = "0.0.1"
r"""EOMEE version string."""
