..
    : This file is part of PyEOM.
    :
    : PyEOM is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : PyEOM is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with PyEOM. If not, see <http://www.gnu.org/licenses/>.

|Python ≥3.6|

PyEOM
#####

PyEOM_ is a pure Python library implementing Rowe's Equations-of-Motions (EOM) and the Extended
Random Phase Approximation (ERPA) *ab-initio* quantum chemistry methods.

To use PyEOM, you need only provide the following as input, in the form of NumPy arrays:

* one- and two- particle molecular integrals
* one- and two- particle reduced density matrices

PyEOM is distributed under the GNU General Public License version 3 (GPLv3).

See http://www.gnu.org/licenses/ for more information.

Dependencies
============

The following programs/libraries are required to run PyEOM:

-  Python_ (≥3.6)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)
-  Pytest_ (optional: to run tests)
-  Pip_ (optional: to install PyEOM)

The following programs/libraries are required to build the PyEOM documentation:

-  Sphinx_ (≥ 3.5)
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

Installation
============

Make sure compatible versions of NumPy_ and SciPy_ are installed.

Run the following in your shell to download PyEOM via ``git``:

.. code:: shell

    git clone https://github.com/gabrielasd/pyeom.git && cd pyeom

Then, run the following to install PyEOM via ``pip``:

.. code:: shell

    pip install -e .

Building Documentation
======================

Make sure a compatible version of Sphinx_ is installed.

Run the following in your shell to install the `Read the Docs Sphinx Theme`__ via ``pip``:

__ Sphinx-RTD-Theme_

.. code:: shell

    pip install sphinx-rtd-theme

Then, after installing PyEOM, run the following to build the HTML documentation:

.. code:: shell

    cd docs && make html

Notebooks
=========

To illustrate how to use PyEOM, we provide a set of Jupyter notebooks under the `docs/notebooks` folder.
You can access the notebooks folder here_.

.. _PyEOM: http:github.com/gabrielasd/pyeom/
.. _Python: http://docs.python.org/3/
.. _NumPy: http://numpy.org/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _Pytest: http://docs.pytest.org/en/latest/
.. _Pip: http://pip.pypa.io/en/stable/
.. _Sphinx: http://sphinx-doc.org/
.. _Sphinx-RTD-Theme: http://sphinx-rtd-theme.readthedocs.io/
.. _here : http://github.com/gabrielasd/pyeom/tree/master/docs/notebooks

.. |Python ≥3.6| image:: http://img.shields.io/badge/python-≥3.6-blue.svg
   :target: http://docs.python.org/3/
