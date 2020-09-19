"""Test eomee.ionization."""


# import eomee
from eomee import (
    IonizationEOMState,
    IonizationDoubleCommutator,
    IonizationAntiCommutator,
)
from eomee.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import numpy as np
from scipy.linalg import eig, svd
import pytest


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


def test_ionizationeom_one_body_term_h2():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)
    # the two-electron integrals are ignored
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = np.sort(aval1)

    eom = IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = np.sort(aval2)

    eom = IonizationAntiCommutator(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = np.sort(aval3)

    w, v = eig(one_mo)
    ip = -np.real(w)
    ip = np.sort(ip)
    assert abs(aval1[-1] - ip[-1]) < 1e-8
    assert abs(aval2[-1] - ip[-1]) < 1e-8
    assert abs(aval3[-1] - ip[-1]) < 1e-8


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, expected, tol, eom_type",
    [
        ("heh+_sto3g", 2, (1, 1), 0, 1.52378328, 1e-6, IonizationEOMState),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, IonizationEOMState),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, IonizationDoubleCommutator),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, IonizationAntiCommutator),
        ("ne_321g", 9, (5, 5), 4, 0.79034293, 1e-5, IonizationEOMState),
        ("ne_321g", 9, (5, 5), 4, 0.79034293, 1e-5, IonizationDoubleCommutator),
        ("ne_321g", 9, (5, 5), 3, 0.79034293, 1e-5, IonizationAntiCommutator),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, IonizationEOMState),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, IonizationDoubleCommutator),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, IonizationAntiCommutator),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, IonizationEOMState),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, IonizationDoubleCommutator),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, IonizationAntiCommutator),
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
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eom_type(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    print(aval)
    assert abs(aval[evidx] - expected) < tol
