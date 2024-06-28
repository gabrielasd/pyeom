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

r"""Test eomee.electronaff."""


from eomee import EA, EAc, EAa

from eomee.tools import (
    find_datafiles,
    spinize,
    hartreefock_rdms,
)

import numpy as np

from scipy.linalg import eig

import pytest


def test_eomea_neigs():
    """Check number of eigenvalues.

    """
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EA(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == 4


@pytest.mark.xfail("True", reason="Solver selects only positive side of spectrum")
def test_eomea_one_body_term():
    """Check that the one-body teerms of the electron affinities
    equations of motion are correct.

    """
    # Load integrals files and transform from molecular orbital
    # to spin orbital basis (internal representation in eomee code)
    # For this test the two-electron integrals are neglected and the
    # Hartree-Fock density matrices are used.
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    # Expected value
    w, _ = eig(one_mo)
    ea = np.real(w)
    # EOM solution
    eom = EA(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval1, _ = eom.solve_dense()
    # assert abs(sorted(aval1)[0] - ea[1]) < 1e-8
    assert abs(aval1[0] - ea[1]) < 1e-8

    eom = EAa(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval2, _ = eom.solve_dense()
    # assert abs(sorted(aval2)[-1] - ea[1]) < 1e-8
    assert abs(aval2[-1] - ea[1]) < 1e-8

    eom = EAc(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval3, _ = eom.solve_dense()
    # assert abs(sorted(aval3)[0] - ea[1]) < 1e-8
    assert abs(aval3[0] - ea[1]) < 1e-8


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, hf_vmo, tol",
    [
        ("h2_hf_sto6g", 2, (1, 1), -1, 0.66587228, 1e-7),
        # this one fails because of solver choosing only positive side of spectrum
        # ("heh+_sto3g", 2, (1, 1), 0, -0.26764028, 1e-6), 
        ("he_ccpvdz", 5, (1, 1), 0, 1.39744193, 1e-6),
        ("ne_321g", 9, (5, 5), 0, 2.68726251, 1e-5),
        ("be_sto3g", 5, (2, 2), 0, 0.22108596, 1e-8),
        ("b_sto3g", 5, (3, 2), 0, 0.29136562, 1e-8),
    ],
)
def test_eomea(filename, nbasis, nocc, evidx, hf_vmo, tol):
    """Test EA class against Hartree-Fock canonical orbital energy.

    """
    na, nb = nocc
    one_mo = np.load(find_datafiles("{0}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{0}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    eom = EA(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval, _ = eom.solve_dense()
    aval = sorted(aval)

    assert abs(aval[evidx] - hf_vmo) < tol


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, hf_vmo, tol",
    [
        ("h2_hf_sto6g", 2, (1, 1), 0, 0.66587228, 1e-7),
        ("he_ccpvdz", 5, (1, 1), 0, 1.39744193, 1e-6),
        ("ne_321g", 9, (5, 5), 0, 2.68726251, 1e-5),
        ("be_sto3g", 5, (2, 2), 0, 0.22108596, 1e-8),
    ],
)
def test_eadoublecommutator(filename, nbasis, nocc, evidx, hf_vmo, tol):
    """Test EOMEADoubleCommutator against Hartree-Fock canonical orbital energy.

    """
    na, nb = nocc
    one_mo = np.load(find_datafiles("{0}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{0}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    eom = EAc(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval, _ = eom.solve_dense()
    # aval = sorted(aval)

    assert abs(aval[evidx] - hf_vmo) < tol


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, hf_vmo, tol",
    [
        ("h2_hf_sto6g", 2, (1, 1), 0, 0.66587228, 1e-7),
        ("he_ccpvdz", 5, (1, 1), 0, 1.39744193, 1e-6),
        ("ne_321g", 9, (5, 5), 0, 2.68726251, 1e-5),
        ("be_sto3g", 5, (2, 2), 0, 0.22108596, 1e-8),
    ],
)
def test_eaanticommutator(filename, nbasis, nocc, evidx, hf_vmo, tol):
    """Test EOMEAAntiCommutator against Hartree-Fock canonical orbital energy.

    """
    na, nb = nocc
    one_mo = np.load(find_datafiles("{0}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{0}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    eom = EAa(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval, _ = eom.solve_dense()
    aval = sorted(aval)

    assert abs(aval[evidx] - hf_vmo) < tol


def test_electronaff_b_sto3g():
    """Test Electron Affinities EOMs for B (STO-3G)
    against Hartree-Fock canonical orbital energies.

    HF MO_a: 0.29136562, 0.32299525, 0.38625451

    """
    one_mo = np.load(find_datafiles("b_sto3g_oneint.npy"))
    two_mo = np.load(find_datafiles("b_sto3g_twoint.npy"))
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, 3, 2)

    eom = EAc(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval2, _ = eom.solve_dense()
    # aval2 = sorted(aval2)

    eom = EAa(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    aval3, _ = eom.solve_dense()
    # aval3 = sorted(aval3)

    # HORTON UHF alpha HOMO
    # horton_emo_a = [-7.26583392, -0.428277, -0.20051823, 0.29136562, 0.29136562]
    # horton_emo_b = [-7.24421665, -0.31570904, 0.32299525, 0.32299525, 0.38625451]
    ea1 = 0.29136562
    ea2 = 0.32299525
    ea3 = 0.38625451
    assert abs(aval2[0] - ea1) < 1e-8
    assert abs(aval2[2] - ea2) < 1e-8
    assert abs(aval2[4] - ea3) < 1e-8
    assert abs(aval3[0] - ea1) < 1e-8
    assert abs(aval3[2] - ea2) < 1e-8
    assert abs(aval3[4] - ea3) < 1e-8
