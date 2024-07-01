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

r"""Test eomee.eomdea."""


from eomee import DEAm

from eomee.tools import (
    find_datafiles,
    spinize,
    antisymmetrize,
    hartreefock_rdms,
)

import numpy as np

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

    eom = DEAm(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == nspino ** 2


def test_eomdea_one_body_term():
    """
    Check the one-body terms are correct for the double electron affinity equation of motion.

    """
    nbasis = 2
    # Load integrals files and transform from molecular orbital
    # to spin orbital basis (internal representation in eomee code)
    # For this test the two-electron integrals are ignored and the
    # Hartree-Fock density matrices are used.
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    # Because I am neglecting the e-e repulsion integrals is likely that "effective" virtual MOs
    # of the fictitious H2 system have negative energies causing the DEA transition to appear
    # in the negative side of the RPA eigenvalue problem. The parameter `pick_posw=False` will identify
    # the DEA transition based on the eigenvector norm.
    eom = DEAm(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    avalea, _ = eom.solve_dense(pick_posw=False)
    # avalea = np.sort(avalea)

    # Hartree-Fock eigenvalues ignoring two-electron terms
    w, v = eig(one_mo)
    w = np.sort(w)
    # Warning: There might be a sign difference
    # DEA = 2 * epsilon_a
    dea = 2 * np.real(w[-1])

    assert abs(avalea[0] - dea) < 1e-8


def test_eomdea_righthandside_2particle_4spin():
    """
    Check pp-EOM right-hand side for a 2 particles, 4 spin-orbitals system.

    """
    # Auxiliar variables
    npart = 2
    nspatial = 2
    nspin = 2 * nspatial
    nhole = nspin - npart
    # I = np.eye(nspin)
    temp = np.diag([1.0, 0.0])
    # Dummy electron-integrals
    one_mo = np.eye(nspin)
    two_mo = np.zeros((nspin,) * 4, dtype=one_mo.dtype)
    # Build density matrices
    one_dm, two_dm = hartreefock_rdms(nspatial, 1, 1)

    # EOM solution
    eomea = DEAm(one_mo, two_mo, one_dm, two_dm)
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
    Check pp-EOM right-hand side for a 4 particles, 6 spin-orbitals system.

    """
    # Auxiliar variables
    npart = 4
    nspatial = 3
    nspin = 2 * nspatial
    nhole = nspin - npart
    # I = np.eye(nspin)
    temp = np.diag([1.0, 1.0, 0.0])
    # Dummy electron-integrals
    one_mo = np.eye(nspin)
    two_mo = np.zeros((nspin,) * 4, dtype=one_mo.dtype)
    # Build density matrices
    one_dm, two_dm = hartreefock_rdms(nspatial, 1, 1)

    # EOM solution
    eomea = DEAm(one_mo, two_mo, one_dm, two_dm)
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
    Test DEAm on Be+4 (STO-6G). Model system for double electron attachment
    on top of the vacuum state.

    """
    # The energy for the process of adding two electron to the vacuum state (Be^+4):
    # Be^+4 + 2e --> Be^+2   DEA < 0
    # using the MOs from neutral Be SCF calculation will give a negative DEA 
    # (i.e. the eigenvalue appears on the negative side of the RPA spectrum)
    one_mo = spinize(np.load(find_datafiles("beII_sto6g_oneint.npy")))
    two_mo = spinize(np.load(find_datafiles("beII_sto6g_twoint.npy")))
    one_dm = np.zeros((one_mo.shape[0],) * 2, dtype=one_mo.dtype)
    two_dm = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    nspin = one_mo.shape[0]
    npart = 0
    nhole = nspin - npart

    eomea = DEAm(one_mo, two_mo, one_dm, two_dm)
    avalea, _ = eomea.solve_dense(pick_posw=False)
    # avalea = np.sort(avalea)

    # Double-electron attachment EOM on vacuum satate
    # LHS = < | k l H m+ n+ | >
    I = np.eye(nspin)
    lhs = np.einsum("lm,kn->klnm", one_mo, I)
    lhs += np.einsum("kn,lm->klnm", one_mo, I)
    lhs *= 2
    lhs += antisymmetrize(two_mo)
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
    w, _ = eig(A)
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
    Test DEAm on Be+2 (STO-6G).

    """
    one_mo = np.load(find_datafiles("beII_sto6g_oneint.npy"))
    two_mo = np.load(find_datafiles("beII_sto6g_twoint.npy"))
    nspatial = one_mo.shape[0]
    nspin = 2 * nspatial
    npart = 2
    nhole = nspin - npart
    one_dm, two_dm = hartreefock_rdms(nspatial, 1, 1)

    eom = DEAm(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    avalea, _ = eom.solve_dense(pick_posw=False)
    # avalea = np.sort(avalea)

    # Be(+2) RHF/sto-6g energy and
    # Be FrozenCore CCD energy
    Erhf = -13.593656013473
    CCD = -14.5557601
    approxbeEccd = Erhf + avalea[0]

    # Tr(RHS) = 59.9999
    assert np.trace(eom.rhs) == (nhole * (nhole - 1))
    assert abs(approxbeEccd - CCD) < 1e-3
