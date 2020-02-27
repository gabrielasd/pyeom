"""Test eomee.doubleelectronaff."""


import eomee
from eomee.tools import find_datafiles

import numpy as np
import numpy.testing as npt

from scipy.linalg import eig, svd


def check_inputs_symm(oneint, twoint, onedm, twodm):
    """Check symmetry of electron integrals and Density Matrices."""
    # Electron integrals and DMs symmetric permutations
    assert np.allclose(oneint, oneint.T)
    assert np.allclose(onedm, onedm.T)
    assert np.allclose(twoint, np.einsum("pqrs->rspq", twoint))
    assert np.allclose(twoint, np.einsum("pqrs->qpsr", twoint))
    assert np.allclose(twodm, np.einsum("pqrs->rspq", twodm))
    assert np.allclose(twodm, np.einsum("pqrs->qpsr", twodm))
    # Two-electron integrals  and 2DM antisymmetric permutations
    assert np.allclose(twoint, -np.einsum("pqrs->pqsr", twoint))
    assert np.allclose(twoint, -np.einsum("pqrs->qprs", twoint))
    assert np.allclose(twodm, -np.einsum("pqrs->pqsr", twodm))
    assert np.allclose(twodm, -np.einsum("pqrs->qprs", twodm))


def test_doubleelectronaff_one_body_term_H2():
    """
    Check the one-body terms are correct for the double
    electron affinity equation of motion .

    """
    one_mo = np.load(find_datafiles("h2_sto6g_oneint_genzd.npy"))
    # the two-electron integrals are ignored
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm = np.load(find_datafiles("1dm_h2_sto6g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_h2_sto6g_genzd_anti.npy"))

    eomea = eomee.DoubleElectronAttachmentEOM(one_mo, two_mo, one_dm, two_dm)
    avalea, avecea = eomea.solve_dense()
    avalea = np.sort(avalea)

    # Hartree-Fock eigenvalues ignoring two-electron terms
    w, v = eig(one_mo)
    w = np.sort(w)
    # Warning: There might be a sign difference
    # DEA = 2 * epsilon_a
    dea = 2 * np.real(w[-1])

    assert abs(avalea[0] - dea) < 1e-8


def test_righthandside_2particle_4spin():
    """
    Check pp-EOM right-hand side for
    a 2 particles, 4 spin-orbitals system.

    """
    npart = 2
    nspatial = 2
    nspin = 2 * nspatial
    nhole = nspin - npart
    I = np.eye(nspin)
    temp = np.diag([1.0, 0.0])
    one_mo = np.eye(nspin)
    two_mo = np.zeros((nspin,) * 4, dtype=one_mo.dtype)

    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    eomea = eomee.DoubleElectronAttachmentEOM(one_mo, two_mo, one_dm, two_dm)

    temp = np.diag([0.0, 1.0])
    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm_conj = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm_conj -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    two_dm_conj = two_dm_conj.reshape(nspin ** 2, nspin ** 2)

    npt.assert_allclose(eomea.rhs, two_dm_conj)
    trrhs = np.trace(eomea.rhs)
    trdm = np.trace(two_dm_conj)
    assert trrhs == trdm == nhole * (nhole - 1)


def test_righthandside_4particle_6spin():
    """
    Check pp-EOM right-hand side for
    a 4 particles, 6 spin-orbitals system.

    """
    npart = 4
    nspatial = 3
    nspin = 2 * nspatial
    nhole = nspin - npart
    I = np.eye(nspin)
    temp = np.diag([1.0, 1.0, 0.0])
    one_mo = np.eye(nspin)
    two_mo = np.zeros((nspin,) * 4, dtype=one_mo.dtype)

    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    eomea = eomee.DoubleElectronAttachmentEOM(one_mo, two_mo, one_dm, two_dm)

    temp = np.diag([0.0, 0.0, 1.0])
    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm_conj = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm_conj -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    two_dm_conj = two_dm_conj.reshape(nspin ** 2, nspin ** 2)

    npt.assert_allclose(eomea.rhs, two_dm_conj)
    trrhs = np.trace(eomea.rhs)
    trdm = np.trace(two_dm_conj)
    assert trrhs == trdm == nhole * (nhole - 1)


def test_doubleelectronaff_beIV_sto6g():
    """
    Test DoubleElectronAttachmentEOM on Be+4 (STO-6G).
    Model system for double electron attachment on top of
    the vacuum state.

    """
    npart = 0
    nspatial = 5
    nspin = 2 * nspatial
    nhole = nspin - npart
    one_mo = np.load(find_datafiles("beII_sto6g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("beII_sto6g_twoint_genzd_anti.npy"))
    one_dm = np.zeros((one_mo.shape[0],) * 2, dtype=one_mo.dtype)
    two_dm = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)
    assert nspin == one_mo.shape[0]

    eomea = eomee.DoubleElectronAttachmentEOM(one_mo, two_mo, one_dm, two_dm)
    avalea, avecea = eomea.solve_dense()
    avalea = np.sort(avalea)

    # Double-electron attachment EOM on vacuum satate
    # LHS = < | k l H m+ n+ | >
    I = np.eye(nspin)
    lhs = np.einsum("lm,kn->klnm", one_mo, I)
    lhs += np.einsum("kn,lm->klnm", one_mo, I)
    lhs *= 2
    lhs += two_mo
    lhs = lhs.reshape(nspin ** 2, nspin ** 2)
    # RHS = < | k l m+ n+ | >
    rhs = np.einsum('kn,lm->klnm', I, I)
    rhs -= np.einsum('km,ln->klnm', I, I)
    rhs = rhs.reshape(nspin ** 2, nspin ** 2)

    # Diagonalization
    U, s, V = svd(rhs)
    s = s ** (-1)
    s[s >= 1 / (1.0e-10)] = 0.0
    S_inv = np.diag(s)
    rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
    A = np.dot(rhs_inv, lhs)
    w, v = eig(A)
    avalvacuum = np.real(w)
    avalvacuum = np.sort(avalvacuum)

    # RHF/sto-6g Energy
    # Erhf = -13.593656013473
    # FullCI Energy
    Efullci = -1.359391958290e01

    assert np.trace(eomea.rhs) == nhole * (nhole - 1)
    assert np.trace(rhs) == nhole * (nhole - 1)
    assert abs(avalvacuum[0] - avalea[0]) < 1e-8
    assert abs(avalvacuum[0] - Efullci) < 1e-8


def test_doubleelectronaff_beII_sto6g():
    """
    Test DoubleElectronAttachmentEOM on Be+2 (STO-6G).

    """
    one_mo = np.load(find_datafiles("beII_sto6g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("beII_sto6g_twoint_genzd_anti.npy"))
    # DMs for a single Slater determinant
    nspatial = 5
    nspin = 2 * nspatial
    npart = 2
    nhole = nspin - npart
    occs = np.array([1., 0., 0., 0., 0.])
    temp = np.diag(occs)
    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm = np.einsum('pr,qs->pqrs', one_dm, one_dm)
    two_dm -= np.einsum('ps,qr->pqrs', one_dm, one_dm)
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)
    assert np.trace(one_dm) == 2
    assert np.einsum('klkl', two_dm) == 2

    eomea = eomee.DoubleElectronAttachmentEOM(one_mo, two_mo, one_dm, two_dm)
    avalea, avecea = eomea.solve_dense()
    avalea = np.sort(avalea)

    # Be(+2) RHF/sto-6g energy and
    # Be FrozenCore CCD energy
    Erhf = -13.593656013473
    CCD = -14.5557601
    approxbeEccd = Erhf + avalea[0]

    # Tr(RHS) = 59.9999
    assert np.trace(eomea.rhs) == nhole * (nhole - 1)
    assert abs(approxbeEccd - CCD) < 1e-3
