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
    "parse_inputfile",
    "check_inputs",
    "ElectronIntegrals",
    "WfnRDMs",
    "dump",
]


from .base import EOMState
from .ionization import EOMIP
from .ionization import EOMIPDoubleCommutator
from .ionization import EOMIPAntiCommutator
from .electronaff import EOMEA
from .electronaff import EOMEADoubleCommutator
from .electronaff import EOMEAAntiCommutator
from .doubleionization import EOMDIP
from .excitation import EOMExc
from .doubleelectronaff import EOMDEA
from .load import parse_inputfile, check_inputs
from .integrals import ElectronIntegrals
from .density import WfnRDMs
from .output import dump
