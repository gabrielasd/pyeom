"""Test eomee.ionization."""


import numpy as np
from scipy.linalg import eig, svd
import pytest
from src.eom import EOMIP
from src import solver
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)


def test_eomip_neigs():
    """

    """
    nspino = 4
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMIP(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == 4


def test_eomip_one_body_term():
    """Check the one-body teerms of the ionization potential
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
    ip = -np.real(w)
    ip = np.sort(ip)
    # EOM solution
    eom = EOMIP(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = solver.dense(eom.lhs, eom.rhs)
    aval1 = np.sort(aval1)
    assert abs(aval1[-1] - ip[-1]) < 1e-8


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, expected, tol, eom_type",
    [
        ("heh+_sto3g", 2, (1, 1), 0, 1.52378328, 1e-6, EOMIP),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, EOMIP),
        ("ne_321g", 9, (5, 5), 4, 0.79034293, 1e-5, EOMIP),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, EOMIP),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, EOMIP),
    ],
)
def test_ionizationeomstate_h2_sto6g(
    filename, nbasis, nocc, evidx, expected, tol, eom_type
):
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

    eom = eom_type(one_mo, two_mo, one_dm, two_dm)
    aval, avec = solver.dense(eom.lhs, eom.rhs)
    # Reference value from HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    assert abs(aval[evidx] - expected) < tol
