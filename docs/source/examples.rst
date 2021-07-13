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

Examples
########

EOMEE computes excited states energies though the implementation of the equation-of-motion
formalism. 

Some example scripts showcasing the supported features can be found inside the project's *examples* folder.
These assume the package has been installed following the instructions in :ref:`Installation
<installation>`.

To run particle-hole RPA for Hellium with EOMEE do:

    .. code-block:: python

        from eomee import EOMExc

        from eomee.tools import (
            find_datafiles,
            spinize,
            hartreefock_rdms,
        )

        import numpy as np

        # System He
        nalpha = 1
        nbeta = 1
        # Load one- and two-electron integrals (in molecular orbitals basis)
        one_mo = np.load(find_datafiles("he_ccpvdz_oneint.npy"))
        two_mo = np.load(find_datafiles("he_ccpvdz_twoint.npy"))

        # Make spin-resolved one- and two-particle density matrices for Hartree-Fock slater determinant
        nbasis = one_mo.shape[0] # Number of molecular orbitals in the basis set
        one_dm, two_dm = hartreefock_rdms(nbasis, nalpha, nbeta)

        # Transform electron integrlas from spatial to spin-resolved representation
        one_mo = spinize(one_mo)
        two_mo = spinize(two_mo)

        # Solve particle-hole EOM
        pheom = EOMExc(one_mo, two_mo, one_dm, two_dm)
        ev, cv = pheom.solve_dense(orthog="asymmetric")

        print("Number of eigenvalues: ", pheom.neigs)
        print("Left-hand-side matrix: ", pheom.lhs, "\n")
        print("Right-hand-side matrix: ", pheom.rhs, "\n")

        print("Transition energies: ", ev)

One can get the electron integrals required as input in above's example thorugh a mean-field calculation using an external package (e.g. PySCF). 