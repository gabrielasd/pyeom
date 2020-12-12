"""Test src.load."""

import numpy as np
from src.eom import EOMIP
from src.solver import dense
from test.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import pytest


@pytest.mark.parametrize(
    "filename, nbasis, nocc, evidx, expected, tol, orthog",
    [
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, "symmetric"),
        ("he_ccpvdz", 5, (1, 1), 0, 0.91414765, 1e-6, "asymmetric"),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, "symmetric"),
        ("be_sto3g", 5, (2, 2), 1, 0.25403769, 1e-8, "asymmetric"),
    ],
)
def test_dense_orthogonalization(filename, nbasis, nocc, evidx, expected, tol, orthog):
    """Verify orthogonalization methos of solver module.

    """
    na, nb = nocc
    one_mo = np.load(find_datafiles("{0}_oneint.npy".format(filename)))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("{0}_twoint.npy".format(filename)))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    eom = EOMIP(one_mo, two_mo, one_dm, two_dm)
    aval, _ = dense(eom.lhs, eom.rhs, orthog=orthog)
    assert abs(aval[evidx] - expected) < tol


def test_inputs():
    one_mo = np.load(find_datafiles("{0}_oneint.npy".format("be_sto3g")))
    with pytest.raises(ValueError):
        dense(one_mo, one_mo, orthog=True)
    with pytest.raises(TypeError):
        dense(one_mo, one_mo, tol=1)

