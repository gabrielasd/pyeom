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
    "EE",
    "EEm",
    "EES",
    "EET",
    "EESm",
    "EETm",
    "DIP",
    "DIPm",
    "DIPS",
    "DIPT",
    "DEA",        
    "DEAm",
]


from .base import EOMState
from .eomip import IP, IPc, IPa, IPcm, IPam
from .eomea import EA, EAc, EAa
from .excitation import EE, EEm
from .eomdip import DIP, DIPm
from .eomdea import DEA, DEAm
from .spinadapted.excitation import EES, EESm, EET, EETm
from .spinadapted.eomdip import DIPS, DIPT


__version__: str = "0.0.1"
r"""EOMEE version string."""
