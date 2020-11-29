"""Test src.load."""

import numpy as np
from src import solver

import pytest


def test_solver_dense_orthogonalization():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8


def test_solver_dense_tolerance():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8


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
