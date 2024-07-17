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

r"""Equations-of-motion testing package."""

from os import path


__all__ = [
    "find_datafile",
]


DIRPATH = path.join(path.dirname(__file__), "data/")


def find_datafile(file_name):
    r""" Return the full path of a test data file. """
    datapath = path.join(path.abspath(DIRPATH), file_name)
    return path.abspath(datapath)
