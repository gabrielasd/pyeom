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

r"""Test eomee.ionization."""


from eomes import EOMExc

from eomes.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
    pickpositiveeig
)

import numpy as np

from scipy.linalg import eig, svd


def test_eomexc_neigs():
    """

    """
    nspino = 4
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMExc(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == 4 ** 2

# FIXME: Use TDHF reference value
# def test_eomexc_heh_sto3g():
#     """Test ExcitationEOM for HeH+ (STO-3G)
#     against Gaussian's CIS computation.

#     E_S1: 24.7959 eV

#     """
#     nbasis = 2
#     one_mo = np.load(find_datafiles("heh+_sto3g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("heh+_sto3g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

#     eom = EOMExc(one_mo, two_mo, one_dm, two_dm)
#     aval, avec = eom.solve_dense()
#     aval = np.sort(aval)
#     # Lowest excited singlet state fom Gaussian's CIS
#     # E_S1 = 24.7959 eV = 0.91123209 Hartree
#     e = 0.91123209
#     assert abs(aval[-1] - e) < 1e-6


def test_excitationeom_erpa_heh_sto3g():
    """Test Excitation ERPA for HeH+ (STO-3G)"""
    nbasis = 2
    one_mo = np.load(find_datafiles("heh+_sto3g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("heh+_sto3g_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    n = one_mo.shape[0]
    aa = one_mo[:1, :1]
    bb = one_mo[n // 2 : (n // 2 + 1), n // 2 : (n // 2 + 1)]
    aaaa = two_mo[:1, :1, :1, :1]
    abab = two_mo[:1, n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1)]
    baba = two_mo[n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1), :1]
    bbbb = two_mo[
        n // 2 : (n // 2 + 1),
        n // 2 : (n // 2 + 1),
        n // 2 : (n // 2 + 1),
        n // 2 : (n // 2 + 1),
    ]
    one_mo_0 = np.zeros_like(one_mo)
    two_mo_0 = np.zeros_like(two_mo)
    one_mo_0[:1, :1] = aa
    one_mo_0[n // 2 : (n // 2 + 1), n // 2 : (n // 2 + 1)] = bb
    two_mo_0[:1, :1, :1, :1] = aaaa
    two_mo_0[:1, n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1)] = abab
    two_mo_0[n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1), :1] = baba
    two_mo_0[
        n // 2 : (n // 2 + 1),
        n // 2 : (n // 2 + 1),
        n // 2 : (n // 2 + 1),
        n // 2 : (n // 2 + 1),
    ] = bbbb

    ecorr = EOMExc.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
    print(ecorr)
