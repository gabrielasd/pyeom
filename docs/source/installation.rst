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

Installation
############

EOMEE might be available to install from PyPI or Conda later on, but for now
you should install it manually via ``pip``. EOMEE is a pure Python library,
so this process is simple.

Dependencies
============

The following programs/libraries are required to run EOMEE:

-  Python_ (≥3.6)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)
-  Pytest_ (optional: to run tests)
-  Pip_ (optional: to install EOMEE)

The following programs/libraries are required to build the EOMEE documentation:

-  Sphinx_ (≥ 3.5)
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

Installing package
==================

Make sure compatible versions of NumPy_ and SciPy_ are installed.

Run the following in your shell to download EOMEE via ``git``:

.. code:: shell

    git clone https://github.com/gabrielasd/eomee.git && cd eomee

Then, run the following to install EOMEE via ``pip``:

.. code:: shell

    pip install -e .

Building Documentation
======================

Make sure a compatible version of Sphinx_ is installed.

Run the following in your shell to install the `Read the Docs Sphinx Theme`__ via ``pip``:

__ Sphinx-RTD-Theme_

.. code:: shell

    pip install sphinx-rtd-theme --user

Then, after installing EOMEE, run the following to build the HTML documentation:

.. code:: shell

    cd docs && make html

.. _Python: http://docs.python.org/3/
.. _NumPy: http://numpy.org/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _Pytest: http://docs.pytest.org/en/latest/
.. _Pip: http://pip.pypa.io/en/stable/
.. _Sphinx: http://sphinx-doc.org/
.. _Sphinx-RTD-Theme: http://sphinx-rtd-theme.readthedocs.io/
