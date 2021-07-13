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

r"""Test eomee.doubleionization."""

import numpy as np

from scipy.linalg import eig, svd

import pytest

from eomee import EOMDIP
from eomee.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)


def test_eomdip_neigs():
    """Check number of eigenvalues.

    """
    nspino = 4
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMDIP(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == nspino ** 2

def test_eomdip_one_body_term():
    r"""
    Check the one-body terms of the double ionization potential equation of motion are correct.

    """
    nbasis = 2
    # Load integrals files and transform from molecular orbital
    # to spin orbital basis (internal representation in eomee code)
    # For this test the two-electron integrals are ignored and the
    # Hartree-Fock density matrices are used.
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    # Expected value
    w, _ = eig(one_mo)
    dip = -2 * np.real(w[0])
    # EOM solution
    eom = EOMDIP(one_mo, two_mo, one_dm, two_dm)
    aval, _ = eom.solve_dense()
    aval = np.sort(aval)

    assert abs(aval[-1] - dip) < 1e-8


def test_eomdip_two_body_terms():
    r"""
    Check the two-body teerms of the double ionization potential
    equation of motion are correct.

    """
    nbasis = 2
    # Load integrals files and transform from molecular orbital
    # to spin orbital basis (internal representation in eomee code)
    # For this test the two-electron integrals are ignored and the
    # Hartree-Fock density matrices are used.
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = EOMDIP(one_mo, two_mo, one_dm, two_dm)
    aval, _ = eom.solve_dense()
    aval = np.sort(aval)

    # Recomputing the left- and right-hand-sides for the double electron removal EOM
    # operator using the two-electron reduced Hamiltonian. The contribution
    # of the two-electron integrals is ignored.
    nbasis = one_mo.shape[0]
    I = np.eye(nbasis, dtype=one_mo.dtype)

    # left-hand-side
    # A_klnm
    w = np.einsum("nk,ml->klnm", one_mo, I) + np.einsum("ml,nk->klnm", one_mo, I)
    w -= np.einsum("mk,nl->klnm", one_mo, I) + np.einsum("nl,mk->klnm", one_mo, I)
    a = np.einsum("mr,kr->mk", I, one_dm)
    w += np.einsum("nl,mk->klnm", one_mo, a)
    a = np.einsum("mr,kr->mk", one_mo, one_dm)
    w += np.einsum("mk,nl->klnm", a, I)
    a = np.einsum("nr,kr->nk", I, one_dm)
    w -= np.einsum("ml,nk->klnm", one_mo, a)
    a = np.einsum("nr,kr->nk", one_mo, one_dm)
    w -= np.einsum("nk,ml->klnm", a, I)

    a = np.einsum("mr,lr->ml", I, one_dm)
    w -= np.einsum("nk,ml->klnm", one_mo, a)
    a = np.einsum("mr,lr->ml", one_mo, one_dm)
    w -= np.einsum("ml,nk->klnm", a, I)
    a = np.einsum("nr,lr->nl", I, one_dm)
    w += np.einsum("mk,nl->klnm", one_mo, a)
    a = np.einsum("nr,lr->nl", one_mo, one_dm)
    w += np.einsum("nl,mk->klnm", a, I)

    a = np.einsum("qk,qm->km", I, one_dm)
    w += 2 * np.einsum("nl,km->klnm", one_mo, a)
    a = np.einsum("qk,qm->km", one_mo, one_dm)
    w += 2 * np.einsum("km,nl->klnm", a, I)
    a = np.einsum("ql,qm->lm", I, one_dm)
    w -= 2 * np.einsum("nk,lm->klnm", one_mo, a)
    a = np.einsum("ql,qm->lm", one_mo, one_dm)
    w -= 2 * np.einsum("lm,nk->klnm", a, I)
    w *= 4 / 4
    # B_klnm
    a = np.einsum("mr,qk->mqrk", one_mo, I)
    b = np.einsum("mqrk,qr->mk", a, one_dm)
    w += 8 / 4 * np.einsum("mk,ln->klnm", b, I)
    a = np.einsum("qk,mr->qmkr", one_mo, I)
    b = np.einsum("qmkr,qr->mk", a, one_dm)
    w += 8 / 4 * np.einsum("mk,ln->klnm", b, I)
    a = np.einsum("qr,mk->qmrk", one_mo, I)
    b = np.einsum("qmrk,qr->mk", a, one_dm)
    w -= 8 / 4 * np.einsum("mk,ln->klnm", b, I)
    a = np.einsum("mk,qr->mqkr", one_mo, I)
    b = np.einsum("qmrk,qr->mk", a, one_dm)
    w -= 8 / 4 * np.einsum("mk,ln->klnm", b, I)

    a = np.einsum("mr,ql->mqrl", one_mo, I)
    b = np.einsum("mqrl,qr->ml", a, one_dm)
    w -= 8 / 4 * np.einsum("ml,kn->klnm", b, I)
    a = np.einsum("ql,mr->qmlr", one_mo, I)
    b = np.einsum("qmrl,qr->ml", a, one_dm)
    w -= 8 / 4 * np.einsum("ml,kn->klnm", b, I)
    a = np.einsum("qr,ml->qmrl", one_mo, I)
    b = np.einsum("qmrl,qr->ml", a, one_dm)
    w += 8 / 4 * np.einsum("ml,kn->klnm", b, I)
    a = np.einsum("ml,qr->mqlr", one_mo, I)
    b = np.einsum("mqrl,qr->ml", a, one_dm)
    w += 8 / 4 * np.einsum("ml,kn->klnm", b, I)
    # C_klnm
    a = np.einsum("qk,nr->qnkr", one_mo, I)
    w += 8 / 4 * np.einsum("qnkr,qlrm->klnm", a, two_dm)
    a = np.einsum("nr,qk->nqrk", one_mo, I)
    w += 8 / 4 * np.einsum("nqrk,qlrm->klnm", a, two_dm)
    a = np.einsum("nk,qr->nqkr", one_mo, I)
    w -= 8 / 4 * np.einsum("nqkr,qlrm->klnm", a, two_dm)
    a = np.einsum("qr,nk->qnrk", one_mo, I)
    w -= 8 / 4 * np.einsum("qnrk,qlrm->klnm", a, two_dm)

    a = np.einsum("ql,nr->qnlr", one_mo, I)
    w -= 8 / 4 * np.einsum("qnlr,qkrm->klnm", a, two_dm)
    a = np.einsum("nr,ql->nqrl", one_mo, I)
    w -= 8 / 4 * np.einsum("nqrl,qkrm->klnm", a, two_dm)
    a = np.einsum("nl,qr->nqlr", one_mo, I)
    w += 8 / 4 * np.einsum("nqlr,qkrm->klnm", a, two_dm)
    a = np.einsum("qr,nl->qnrl", one_mo, I)
    w += 8 / 4 * np.einsum("qnlk,qkrm->klnm", a, two_dm)
    # D_klnm
    a = np.einsum("qr,ns->qnrs", one_mo, I)
    b = np.einsum("qnrs,qlrs->nl", a, two_dm)
    w += 4 / 4 * np.einsum("nl,km->klnm", b, I)
    a = np.einsum("ns,qr->nqsr", one_mo, I)
    b = np.einsum("nqsr,qlrs->nl", a, two_dm)
    w += 4 / 4 * np.einsum("nl,km->klnm", b, I)
    a = np.einsum("nr,qs->nqrs", one_mo, I)
    b = np.einsum("nqrs,qlrs->nl", a, two_dm)
    w -= 4 / 4 * np.einsum("nl,km->klnm", b, I)
    a = np.einsum("qs,nr->qnsr", one_mo, I)
    b = np.einsum("qnsr,qlrs->nl", a, two_dm)
    w -= 4 / 4 * np.einsum("nl,km->klnm", b, I)

    a = np.einsum("qr,ns->qnrs", one_mo, I)
    b = np.einsum("qnrs,qkrs->nk", a, two_dm)
    w -= 4 / 4 * np.einsum("nk,lm->klnm", b, I)
    a = np.einsum("ns,qr->nqsr", one_mo, I)
    b = np.einsum("nqsr,qkrs->nk", a, two_dm)
    w -= 4 / 4 * np.einsum("nk,lm->klnm", b, I)
    a = np.einsum("nr,qs->nqrs", one_mo, I)
    b = np.einsum("nqrs,qkrs->nk", a, two_dm)
    w += 4 / 4 * np.einsum("nk,lm->klnm", b, I)
    a = np.einsum("qs,nr->qnsr", one_mo, I)
    b = np.einsum("qnsr,qkrs->nk", a, two_dm)
    w += 4 / 4 * np.einsum("nk,lm->klnm", b, I)

    lhs = w.reshape(nbasis ** 2, nbasis ** 2)
    # right-hand-side
    rhs = two_dm
    rhs = rhs.reshape(nbasis ** 2, nbasis ** 2)

    # Diagonalization
    U, s, V = svd(rhs)
    with np.errstate(divide="ignore"):
        s = s ** (-1)
    s[s >= 1 / (1.0e-10)] = 0.0
    S_inv = np.diag(s)
    rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
    A = np.dot(rhs_inv, lhs)
    w, _ = eig(A)
    aval2 = np.real(w)
    aval2 = np.sort(aval2)

    w, _ = eig(one_mo)
    dip = -2 * np.real(w[0])

    assert abs(aval2[-1] / 2 - dip) < 1e-8
    assert abs(aval2[-1] / 2 - aval[-1]) < 1e-8


@pytest.mark.parametrize(
    "filename, nparts, ehomo, nbasis, idx",
    [
    ("h2_hf_sto6g", (1, 1), -0.58205888, 2, -1),
    ("heh+_sto3g", (1, 1), -1.52378328, 2, -1),
    ("he_ccpvdz", (1, 1), -0.91414765, 5, -1),
    ],
)
def test_eomdip(filename, nparts, ehomo, nbasis, idx):
    """Test EOMDIP against Hartree-Fock canonical orbitals energy difference.

    case 1: H-H bond 0.742 A

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    assert np.allclose(nbasis, one_mo.shape[0])
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Evaluate hole-hole EOM
    eom = EOMDIP(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval, _ = eom.solve_dense()
    result = np.sort(aval)
    # Expected value
    dip = -2 * (ehomo) + antisymmetrize(spinize(two_mo))[0, nbasis, 0, nbasis]

    assert np.allclose(result[idx], dip)


@pytest.mark.skip(reason="need to update v integrals format")
def test_doubleionization_erpa_HeHcation_sto3g():
    r"""
    Test DoubleElectronRemovalEOM ERPA for HeH^{+} (STO-3G).

    """
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

    ecorr = EOMDIP.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)


@pytest.mark.skip(reason="need to update v integrals format")
def test_doubleionization_erpa_Ne_321g():
    r"""
    Test DoubleElectronRemovalEOM ERPA for Ne 321g.

    """
    nbasis = 9
    one_mo = np.load(find_datafiles("ne_321g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("ne_321g_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 5, 5)

    n = one_mo.shape[0]
    aa = one_mo[:5, :5]
    bb = one_mo[n // 2 : (n // 2 + 5), n // 2 : (n // 2 + 5)]
    aaaa = two_mo[:5, :5, :5, :5]
    abab = two_mo[:5, n // 2 : (n // 2 + 5), :5, n // 2 : (n // 2 + 5)]
    baba = two_mo[n // 2 : (n // 2 + 5), :5, n // 2 : (n // 2 + 5), :5]
    bbbb = two_mo[
        n // 2 : (n // 2 + 5),
        n // 2 : (n // 2 + 5),
        n // 2 : (n // 2 + 5),
        n // 2 : (n // 2 + 5),
    ]
    one_mo_0 = np.zeros_like(one_mo)
    two_mo_0 = np.zeros_like(two_mo)
    one_mo_0[:5, :5] = aa
    one_mo_0[n // 2 : (n // 2 + 5), n // 2 : (n // 2 + 5)] = bb
    two_mo_0[:5, :5, :5, :5] = aaaa
    two_mo_0[:5, n // 2 : (n // 2 + 5), :5, n // 2 : (n // 2 + 5)] = abab
    two_mo_0[n // 2 : (n // 2 + 5), :5, n // 2 : (n // 2 + 5), :5] = baba
    two_mo_0[
        n // 2 : (n // 2 + 5),
        n // 2 : (n // 2 + 5),
        n // 2 : (n // 2 + 5),
        n // 2 : (n // 2 + 5),
    ] = bbbb

    ecorr = EOMDIP.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
