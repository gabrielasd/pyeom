"""Test eomee.doubleelectronaff."""


from src.eom import EOMDEA
from src import solver
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import numpy as np
import numpy.testing as npt

from scipy.linalg import eig, svd


def test_eomdea_neigs():
    """

    """
    nspino = 4
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMDEA(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == nspino ** 2


def test_eomdea_one_body_term():
    """
    Check the one-body terms are correct for the double
    electron affinity equation of motion .

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

    eom = EOMDEA(one_mo, two_mo, one_dm, two_dm)
    avalea, avec = solver.dense(eom.lhs, eom.rhs)
    avalea = np.sort(avalea)

    # Hartree-Fock eigenvalues ignoring two-electron terms
    w, v = eig(one_mo)
    w = np.sort(w)
    # Warning: There might be a sign difference
    # DEA = 2 * epsilon_a
    dea = 2 * np.real(w[-1])

    assert abs(avalea[0] - dea) < 1e-8


def test_eomdea_righthandside_2particle_4spin():
    """
    Check pp-EOM right-hand side for
    a 2 particles, 4 spin-orbitals system.

    """
    # Auxiliar variables
    npart = 2
    nspatial = 2
    nspin = 2 * nspatial
    nhole = nspin - npart
    I = np.eye(nspin)
    temp = np.diag([1.0, 0.0])
    # Dummy electron-integrals
    one_mo = np.eye(nspin)
    two_mo = np.zeros((nspin,) * 4, dtype=one_mo.dtype)
    # Build density matrices
    one_dm, two_dm = hartreefock_rdms(nspatial, 1, 1)
    # one_dm = np.zeros((nspin, nspin))
    # one_dm[:nspatial, :nspatial] = temp
    # one_dm[nspatial:, nspatial:] = temp
    # two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    # two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    # EOM solution
    eomea = EOMDEA(one_mo, two_mo, one_dm, two_dm)
    # Expected value
    temp = np.diag([0.0, 1.0])
    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm_conj = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm_conj -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    two_dm_conj = two_dm_conj.reshape(nspin ** 2, nspin ** 2)

    tr_eomrhs = np.trace(eomea.rhs)
    tr_dmconj = np.trace(two_dm_conj)
    assert np.allclose(eomea.rhs, two_dm_conj)
    assert tr_eomrhs == tr_dmconj
    assert tr_eomrhs == (nhole * (nhole - 1))


def test_eomdea_righthandside_4particle_6spin():
    """
    Check pp-EOM right-hand side for
    a 4 particles, 6 spin-orbitals system.

    """
    # Auxiliar variables
    npart = 4
    nspatial = 3
    nspin = 2 * nspatial
    nhole = nspin - npart
    I = np.eye(nspin)
    temp = np.diag([1.0, 1.0, 0.0])
    # Dummy electron-integrals
    one_mo = np.eye(nspin)
    two_mo = np.zeros((nspin,) * 4, dtype=one_mo.dtype)
    # Build density matrices
    one_dm, two_dm = hartreefock_rdms(nspatial, 1, 1)
    # one_dm = np.zeros((nspin, nspin))
    # one_dm[:nspatial, :nspatial] = temp
    # one_dm[nspatial:, nspatial:] = temp
    # two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    # two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    # EOM solution
    eomea = EOMDEA(one_mo, two_mo, one_dm, two_dm)
    # Expected value
    temp = np.diag([0.0, 0.0, 1.0])
    one_dm = np.zeros((nspin, nspin))
    one_dm[:nspatial, :nspatial] = temp
    one_dm[nspatial:, nspatial:] = temp
    two_dm_conj = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm_conj -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    two_dm_conj = two_dm_conj.reshape(nspin ** 2, nspin ** 2)

    tr_eomrhs = np.trace(eomea.rhs)
    tr_dmconj = np.trace(two_dm_conj)
    print(tr_eomrhs)
    print(tr_dmconj)
    print(nhole * (nhole - 1))
    # assert np.allclose(eomea.rhs, two_dm_conj)
    # assert tr_eomrhs == tr_dmconj
    # assert tr_eomrhs == (nhole * (nhole - 1))


def test_eomdea_beIV_sto6g():
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
    assert nspin == one_mo.shape[0]

    eomea = EOMDEA(one_mo, two_mo, one_dm, two_dm)
    avalea, avecea = solver.dense(eomea.lhs, eomea.rhs)
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
    rhs = np.einsum("kn,lm->klnm", I, I)
    rhs -= np.einsum("km,ln->klnm", I, I)
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


def test_eomdea_beII_sto6g():
    """
    Test DoubleElectronAttachmentEOM on Be+2 (STO-6G).

    """
    nspatial = 5
    nspin = 2 * nspatial
    npart = 2
    nhole = nspin - npart
    one_mo = np.load(find_datafiles("beII_sto6g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("beII_sto6g_twoint_genzd_anti.npy"))
    # DMs for a single Slater determinant
    # occs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # temp = np.diag(occs)
    # one_dm = np.zeros((nspin, nspin))
    # one_dm[:nspatial, :nspatial] = temp
    # one_dm[nspatial:, nspatial:] = temp
    # two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    # two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)
    one_dm, two_dm = hartreefock_rdms(nspatial, 1, 1)
    # assert np.trace(one_dm) == 2
    # assert np.einsum("klkl", two_dm) == 2

    eom = EOMDEA(one_mo, two_mo, one_dm, two_dm)
    avalea, avecea = solver.dense(eom.lhs, eom.rhs)
    avalea = np.sort(avalea)

    # Be(+2) RHF/sto-6g energy and
    # Be FrozenCore CCD energy
    Erhf = -13.593656013473
    CCD = -14.5557601
    approxbeEccd = Erhf + avalea[0]

    # Tr(RHS) = 59.9999
    assert np.trace(eom.rhs) == (nhole * (nhole - 1))
    assert abs(approxbeEccd - CCD) < 1e-3


test_eomdea_righthandside_4particle_6spin()
