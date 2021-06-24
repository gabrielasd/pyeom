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


# import eomee
from eomes import (
    EOMIP,
    EOMIPDoubleCommutator,
    EOMIPAntiCommutator,
)

from eomes.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import numpy as np

from scipy.linalg import eig

import pytest


def incorrect_inputs():
    listparam = np.load(find_datafiles("be_sto3g_oneint_spino.npy"))
    listparam = listparam.tolist()
    matrix = np.load(find_datafiles("be_sto3g_oneint_spino.npy"))
    tensor = np.load(find_datafiles("be_sto3g_twoint_spino.npy"))
    hdm2shape = np.zeros((2, 2, 2, 2))

    cases = [
        (listparam, tensor, matrix, tensor),
        (matrix, matrix, matrix, tensor),
        (matrix, tensor, tensor, tensor),
        (matrix, tensor, matrix, listparam),
        (matrix, tensor, matrix, hdm2shape),
    ]

    for case in cases:
        yield case


@pytest.mark.parametrize(
    "one_mo, two_mo, one_dm, two_dm", incorrect_inputs(),
)
def test_load_invalid_integrals(one_mo, two_mo, one_dm, two_dm):
    """Check that bad inputs are
    detected. The cases considered are:
    Case 1: Incorrect file extension (only .npy is allowed)
    Case 2: Incorrect integral data type (only NumPy array is allowed)
    Case 3: Incorrect integrals dimensions (oneint must be 2D, twoint 4D)
    Case 4: Incorrect integrals shape (oneint must be a square matrix,
        twoint a tensor with 4 equivalent dimensions)
    Case 4: Basis set mismatch between integrals (oneint and twoint must
        have the same number of spinorbitals)

    """

    with pytest.raises(ValueError):
        EOMIP(one_mo, two_mo, one_dm, two_dm)


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
    assert eom.n == nspino
    assert eom.neigs == nspino


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
    aval1, _ = eom.solve_dense()
    aval1 = np.sort(aval1)
    assert abs(aval1[-1] - ip[-1]) < 1e-8

    eom = EOMIPDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval2, _ = eom.solve_dense()
    aval2 = np.sort(aval2)
    assert abs(aval2[-1] - ip[-1]) < 1e-8

    eom = EOMIPAntiCommutator(one_mo, two_mo, one_dm, two_dm)
    aval3, _ = eom.solve_dense()
    aval3 = np.sort(aval3)
    assert abs(aval3[-1] - ip[-1]) < 1e-8


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, expected, tol, eom_type",
    [
        ("heh+_sto3g", 2, (1, 1), 0, 1.52378328, 1e-6, EOMIP),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, EOMIP),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, EOMIPDoubleCommutator),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, EOMIPAntiCommutator),
        ("ne_321g", 9, (5, 5), 4, 0.79034293, 1e-5, EOMIP),
        ("ne_321g", 9, (5, 5), 4, 0.79034293, 1e-5, EOMIPDoubleCommutator),
        ("ne_321g", 9, (5, 5), 3, 0.79034293, 1e-5, EOMIPAntiCommutator),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, EOMIP),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, EOMIPDoubleCommutator),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, EOMIPAntiCommutator),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, EOMIP),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, EOMIPDoubleCommutator),
        ("b_sto3g", 5, (3, 2), 4, 0.20051823, 1e-8, EOMIPAntiCommutator),
    ],
)
def test_ionizationeomstate_h2_sto6g(filename, nbasis, nocc, evidx, expected, tol, eom_type):
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
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    print(aval)
    assert abs(aval[evidx] - expected) < tol
