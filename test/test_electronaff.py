"""Test eomes.electronaff."""


import numpy as np
from scipy.linalg import eig, svd
import pytest
from eomes.eom import EOMEA
from eomes import solver
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)


def test_eomea_neigs():
    """

    """
    nspino = 4
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == 4


def test_eomea_one_body_term():
    """Check that the one-body teerms of the electron affinities
    equations of motion are correct.

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
    w, v = eig(one_mo)
    ea = np.real(w)
    # EOM solution
    eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = solver.dense(eom.lhs, eom.rhs)
    assert abs(sorted(aval1)[0] - ea[1]) < 1e-8


# def test_eomea_h2_sto6g():
#     """Test Electron Affinities EOMs for H2 (STO-6G)
#     against Hartree-Fock canonical orbital energy.

#     HF MO_a: 0.66587228

#     """
#     nbasis = 2
#     one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("h2_hf_sto6g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

#     eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
#     aval1, avec = solver.dense(eom.lhs, eom.rhs)
#     aval1 = sorted(aval1)

#     # Reference value from
#     # HORTON RHF
#     # horton_emo = [-0.58205888, 0.66587228]
#     ea = 0.66587228
#     assert abs(sorted(aval1)[-1] - ea) < 1e-7


# def test_eomea_heh_sto3g():
#     """Test Electron Affinities EOMs for HeH+ (STO-3G)
#     against Hartree-Fock canonical orbital energy.

#     HF MO_a: -0.26764028

#     """
#     nbasis = 2
#     one_mo = np.load(find_datafiles("heh+_sto3g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("heh+_sto3g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

#     eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
#     aval1, avec = solver.dense(eom.lhs, eom.rhs)
#     aval1 = sorted(aval1)

#     # Reference value from
#     # HORTON RHF
#     # horton_emo = [-1.52378328, -0.26764028]
#     ea = -0.26764028
#     assert abs(sorted(aval1)[0] - ea) < 1e-6


# def test_eomea_he_ccpvdz():
#     """Test Electron Affinities EOMs for He (cc-pVDZ)
#     against Hartree-Fock canonical orbital energy.

#     HF MO_a: 1.39744193

#     """
#     nbasis = 5
#     one_mo = np.load(find_datafiles("he_ccpvdz_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("he_ccpvdz_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

#     eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
#     aval1, avec = solver.dense(eom.lhs, eom.rhs)
#     aval1 = sorted(aval1)

#     # Reference value from
#     # HORTON RHF
#     # horton_emo = [-0.91414765, 1.39744193, 2.52437241, 2.52437241, 2.52437241]
#     ea = 1.39744193
#     assert abs(aval1[2] - ea) < 1e-6


# def test_eomea_ne_321g():
#     """Test Electron Affinities EOMs for Ne (3-21G)
#     against Hartree-Fock canonical orbital energy.

#     HF MO_a: 2.68726251

#     """
#     nbasis = 9
#     one_mo = np.load(find_datafiles("ne_321g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("ne_321g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 5, 5)

#     eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
#     aval1, avec = solver.dense(eom.lhs, eom.rhs)
#     aval1 = sorted(aval1)

#     # Reference value from
#     # HORTON RHF
#     # horton_emo = [
#     #     -32.56471038,
#     #     -1.8651519,
#     #     -0.79034293,
#     #     -0.79034293,
#     #     -0.79034293,
#     #     2.68726251,
#     #     2.68726251,
#     #     2.68726251,
#     #     4.08280903,
#     # ]
#     ea = 2.68726251
#     assert abs(sorted(aval1)[-3] - ea) < 1e-5


# def test_eomea_be_sto3g():
#     """Test Electron Affinities EOMs for Be (STO-3G)
#     against Hartree-Fock canonical orbital energy.

#     HF MO_a: 0.22108596

#     """
#     nbasis = 5
#     one_mo = np.load(find_datafiles("be_sto3g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("be_sto3g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 2, 2)

#     eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
#     aval1, avec = solver.dense(eom.lhs, eom.rhs)
#     aval1 = sorted(aval1)

#     # Reference value from
#     # HORTON RHF
#     # horton_mos = np.asarray(
#     #     [-4.48399211, -0.25403769, 0.22108596, 0.22108596, 0.22108596]
#     # )
#     ea = 0.22108596
#     assert abs(sorted(aval1)[-1] - ea) < 1e-8


# def test_eomea_b_sto3g():
#     """Test Electron Affinities EOMs for B (STO-3G)
#     against Hartree-Fock canonical orbital energies.

#     HF MO_a: 0.29136562, 0.32299525, 0.38625451

#     """
#     nbasis = 5
#     one_mo = np.load(find_datafiles("b_sto3g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("b_sto3g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 3, 2)

#     eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
#     aval1, avec = solver.dense(eom.lhs, eom.rhs)
#     aval1 = sorted(aval1)

#     # HORTON UHF alpha HOMO
#     # horton_emo_a = [-7.26583392, -0.428277, -0.20051823, 0.29136562, 0.29136562]
#     # horton_emo_b = [-7.24421665, -0.31570904, 0.32299525, 0.32299525, 0.38625451]
#     ea1 = 0.29136562
#     ea2 = 0.32299525
#     ea3 = 0.38625451
#     assert abs(aval1[5] - ea1) < 1e-8
#     assert abs(aval1[7] - ea2) < 1e-8
#     assert abs(aval1[9] - ea3) < 1e-8


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, expected, tol",
    [
        ("h2_hf_sto6g", 2, (1, 1), -1, 0.66587228, 1e-7),
        ("heh+_sto3g", 2, (1, 1), 0, -0.26764028, 1e-6),
        ("he_ccpvdz", 5, (1, 1), 2, 1.39744193, 1e-6),
        ("ne_321g", 9, (5, 5), -3, 2.68726251, 1e-5),
        ("be_sto3g", 5, (2, 2), -1, 0.22108596, 1e-8),
        ("b_sto3g", 5, (3, 2), 5, 0.29136562, 1e-8),
    ],
)
def test_ionizationeomstate_h2_sto6g(filename, nbasis, nocc, evidx, expected, tol):
    """Test IonizationEOMState for H2 (STO-6G)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.58205888
    Experiment: 15.42593 eV

    """
    na, nb = nocc
    one_mo = np.load(find_datafiles("{0}_oneint.npy".format(filename)))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("{0}_twoint.npy".format(filename)))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    eom = EOMEA(one_mo, two_mo, one_dm, two_dm)
    aval, avec = solver.dense(eom.lhs, eom.rhs)
    aval = sorted(aval)
    # Reference value from HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    assert abs(aval[evidx] - expected) < tol


def test_compute_tdm():
    nbasis = 5
    one_mo = np.load(find_datafiles("be_sto3g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("be_sto3g_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 2, 2)
    eom = EOMEA(one_mo, two_mo, one_dm, two_dm)

    # Solve the EOM and compute the TDMs
    aval, avec = solver.dense(eom.lhs, eom.rhs)
    tdms = eom.compute_tdm(avec)

    # Tentative verification of the TDM for some ionized state.
    non0_idxs = np.flatnonzero(aval)
    idx = non0_idxs[0]
    tdm_psi_idx = tdms[idx]

    dm_oo = np.einsum("i,k->ik", tdm_psi_idx, tdm_psi_idx.conj())
    assert np.allclose(dm_oo, dm_oo.T)
    _, s, _ = svd(dm_oo)
    assert np.allclose(sum(s[:]), 1.0)
