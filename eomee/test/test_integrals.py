"""Test eomes.integrals."""

import numpy as np
from eomes import ElectronIntegrals
from eomes.tools import find_datafiles

import pytest


@pytest.mark.parametrize(
    "filename, astype", [("be_sto3g", "asfiles"), ("be_sto3g", "asarrays")],
)
def test_load_integrals(filename, astype):
    """Check the one- and two-electron integrals are
    loaded properly.

    """
    nbasis = 5
    if astype == "asfiles":
        one_so = find_datafiles("{}_oneint_spino.npy".format(filename))
        two_so = find_datafiles("{}_twoint_spino.npy".format(filename))
    else:
        one_so = np.load(find_datafiles("{}_oneint_spino.npy".format(filename)))
        two_so = np.load(find_datafiles("{}_twoint_spino.npy".format(filename)))

    expected_h = np.load(find_datafiles("{}_oneint_spino.npy".format(filename)))
    expected_v = np.load(find_datafiles("{}_twoint_spino.npy".format(filename)))

    ham = ElectronIntegrals(one_so, two_so)

    assert np.allclose(expected_h, ham.h)
    assert np.allclose(expected_v, ham.v)
    assert np.allclose(nbasis * 2, ham.nspino)


@pytest.mark.parametrize(
    "astype, error",
    [
        ("fileext", ValueError),
        ("listtype", TypeError),
        ("h4dim", ValueError),
        ("vshape", ValueError),
        ("hvshape", ValueError),
    ],
)
def test_check_invalid_integrals(astype, error):
    """Check that bad integral inputs are
    detected. The cases considered are:
    Case 1: Incorrect file extension (only .npy is allowed)
    Case 2: Incorrect integral data type (only NumPy array is allowed)
    Case 3: Incorrect integrals dimensions (oneint must be 2D, twoint 4D)
    Case 4: Incorrect integrals shape (oneint must be a square matrix,
        twoint a tensor with 4 equivalent dimensions)
    Case 4: Basis set mismatch between integrals (oneint and twoint must
        have the same number of spinorbitals)

    """
    if astype == "fileext":
        oneint = find_datafiles("test_input1.in")
        twoint = find_datafiles("be_sto3g_twoint_spino.npy")
    elif astype == "listtype":
        oneint = np.load(find_datafiles("be_sto3g_oneint_spino.npy"))
        oneint = oneint.tolist()
        twoint = np.load(find_datafiles("be_sto3g_twoint_spino.npy"))
    elif astype == "h4dim":
        oneint = np.load(find_datafiles("be_sto3g_twoint_spino.npy"))
        twoint = np.load(find_datafiles("be_sto3g_twoint_spino.npy"))
    elif astype == "vshape":
        oneint = np.load(find_datafiles("be_sto3g_oneint_spino.npy"))
        twoint = np.zeros((2, 2, 3, 3))
    elif astype == "hvshape":
        oneint = np.load(find_datafiles("be_sto3g_oneint_spino.npy"))
        twoint = np.zeros((2, 2, 2, 2))

    with pytest.raises(error):
        ElectronIntegrals(oneint, twoint)


def example_integrals():
    hnosymm = np.arange(4).reshape(2, 2)
    vnosymm = np.arange(16).reshape(2, 2, 2, 2)
    test_onint = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    test_twoint = np.load(find_datafiles("h2_hf_sto6g_twoint.npy"))

    cases = [(hnosymm, test_twoint), (test_onint, vnosymm), (test_onint, test_twoint)]
    for case in cases:
        yield case


@pytest.mark.parametrize(
    "oneint, twoint", example_integrals(),
)
def test_verify_symmetry(oneint, twoint):
    with pytest.raises(ValueError):
        ElectronIntegrals(oneint, twoint)
