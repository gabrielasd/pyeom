"""Test eomee.doubleionization."""


import eomee

import numpy as np
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

from scipy.linalg import eig, svd


def test_doubleionization_one_body_term_H2():
    """Check the one-body terms of the double ionization potential
    equation of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)
    # the two-electron integrals are ignored
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = eomee.DoubleElectronRemovalEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = np.sort(aval)

    w, v = eig(one_mo)
    dip = -2 * np.real(w[0])
    assert abs(aval[-1] - dip) < 1e-8


def test_doubleionization_two_body_term_H2():
    """Check the two-body teerms of the double ionization potential
    equation of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)
    # the two-electron integrals are ignored
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = eomee.DoubleElectronRemovalEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
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
    s = s ** (-1)
    s[s >= 1 / (1.0e-10)] = 0.0
    S_inv = np.diag(s)
    rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
    A = np.dot(rhs_inv, lhs)
    w, v = eig(A)
    aval2 = np.real(w)
    aval2 = np.sort(aval2)

    w, v = eig(one_mo)
    dip = -2 * np.real(w[0])
    assert abs(aval2[-1] / 2 - dip) < 1e-8
    assert abs(aval2[-1] / 2 - aval[-1]) < 1e-8


def test_doubleionization_H2_sto6g():
    """Test DoubleElectronRemovalEOM for H2 (STO-6G)
    against Hartree-Fock canonical orbitals energy
    difference.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("h2_hf_sto6g_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = eomee.DoubleElectronRemovalEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = np.sort(aval)

    dip = -2 * (-0.58205888) + two_mo[0, 2, 0, 2]
    assert abs(aval[-1] - dip) < 1e-7


def test_doubleionization_He_ccpvdz():
    """Test DoubleElectronRemovalEOM for He (cc-pVDZ)
    against Hartree-Fock canonical orbitals energy
    difference.

    """
    nbasis = 5
    one_mo = np.load(find_datafiles("he_ccpvdz_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("he_ccpvdz_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = eomee.DoubleElectronRemovalEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = np.sort(aval)

    dip = -2 * (-0.91414765) + two_mo[0, 5, 0, 5]
    assert abs(aval[-1] - dip) < 1e-6


def test_doubleionization_HeHcation_sto3g():
    """Test DoubleElectronRemovalEOM for HeH^{+} (STO-3G)
    against Hartree-Fock canonical orbitals energy
    difference.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("heh+_sto3g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("heh+_sto3g_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = eomee.DoubleElectronRemovalEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = np.sort(aval)

    dip = -2 * (-1.52378328) + two_mo[0, 2, 0, 2]
    assert abs(aval[-1] - dip) < 1e-6


def test_doubleionization_erpa_HeHcation_sto3g():
    """Test DoubleElectronRemovalEOM ERPA for HeH^{+} (STO-3G)"""
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

    ecorr = eomee.DoubleElectronRemovalEOM.erpa(
        one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
    )


def test_doubleionization_erpa_Ne_321g():
    """Test DoubleElectronRemovalEOM ERPA for Ne 321g"""
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

    ecorr = eomee.DoubleElectronRemovalEOM.erpa(
        one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
    )