..
    : This file is part of EOMEE.
    :
    : EOMEE is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : EOMEE is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

|Python 3|

EOMEE Documentation
###################

EOMEE is a pure Python library implementing Rowe's Equations-of-Motions (EOM) and the Extended
Random Phase Approximation (ERPA) *ab-initio* quantum chemistry methods.

To use EOMEE, you need only provide the following as input, in the form of NumPy arrays:

* one- and two- particle molecular integrals
* one- and two- particle reduced density matrices

EOMEE is distributed under the GNU General Public License version 3 (GPLv3).

See http://www.gnu.org/licenses/ for more information.

Table of Contents
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   installation
   eom
   erpa
   api

.. |Python 3| image:: http://img.shields.io/badge/python-3-blue.svg
   :target: http://docs.python.org/3/
