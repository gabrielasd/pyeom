"""Test eomes.density."""

import tempfile
import numpy as np
from eomes.density import WfnRDMs
from test.tools import find_datafiles

import pytest


@pytest.mark.parametrize(
    "filename, nelec", [("be_sto3g", 4), ("be_sto3g", (2, 2)),],
)
def test_assign_rdms(filename, nelec):
    """Check the one- and two-electron reduced density
    matrices are loaded properly.

    """
    # Setup test parameters
    nbasis = 5
    nparts = 4
    onedm = find_datafiles("{}_onedm_spino.npy".format(filename))
    twodm = find_datafiles("{}_twodm_spino.npy".format(filename))
    # Reference values
    expecteddm1 = np.load(find_datafiles("{}_onedm_spino.npy".format(filename)))
    expecteddm2 = np.load(find_datafiles("{}_twodm_spino.npy".format(filename)))

    wfn = WfnRDMs(nelec, onedm, twodm)

    assert np.allclose(expecteddm1, wfn.dm1)
    assert np.allclose(expecteddm2, wfn.dm2)
    assert np.allclose(nbasis * 2, wfn.nspino)
    assert np.allclose(nparts, wfn.nparts)


@pytest.mark.parametrize(
    "astype, error",
    [
        ("fileext", ValueError),
        ("dm1dim", ValueError),
        ("dm2shape", ValueError),
        ("dmsshape", ValueError),
    ],
)
def test_check_invalid_rdms(astype, error):
    """Check that bad DM inputs are
    detected. The cases considered are:
    Case 1: Incorrect file extension (only .npy is allowed)
    Case 2: Incorrect DM dimensions (onedm must be 2D, twodm 4D)
    Case 4: Incorrect DM shape (onedm must be a square matrix,
        twodm a tensor with 4 equivalent dimensions)
    Case 5: Basis set mismatch between DMs (onedm and onedm must
        have the same number of spinorbitals)

    """
    nparts = 4

    if astype == "fileext":
        onedm = find_datafiles("test_input1.in")
        twodm = find_datafiles("be_sto3g_onedm_spino.npy")
    elif astype == "dm1dim":
        onedm = find_datafiles("be_sto3g_twodm_spino.npy")
        twodm = find_datafiles("be_sto3g_twodm_spino.npy")
    elif astype == "dm2shape":
        onedm = find_datafiles("be_sto3g_onedm_spino.npy")
        ftmp = tempfile.NamedTemporaryFile(delete=True)
        fname = ftmp.name + ".npy"
        dm2shape = np.zeros((2, 2, 3, 3))
        np.save(fname, dm2shape, allow_pickle=False)
        twodm = fname
    elif astype == "dmsshape":
        onedm = find_datafiles("be_sto3g_onedm_spino.npy")
        ftmp = tempfile.NamedTemporaryFile(delete=True)
        fname = ftmp.name + ".npy"
        dm2shape = np.zeros((2, 2, 2, 2))
        np.save(fname, dm2shape, allow_pickle=False)
        twodm = fname

    with pytest.raises(error):
        WfnRDMs(nparts, onedm, twodm)


def example_rdms():
    dm1nosymm = np.arange(4).reshape(2, 2)
    dm2nosymm = np.arange(16).reshape(2, 2, 2, 2)
    test_dm1 = dm1nosymm + dm1nosymm.T
    test_dm2 = np.load(find_datafiles("h2_hf_sto6g_twoint.npy"))

    cases = [(dm1nosymm, test_dm2), (test_dm1, dm2nosymm), (test_dm1, test_dm2)]
    for case in cases:
        yield case


@pytest.mark.parametrize(
    "onedm, twodm", example_rdms(),
)
def test_verify_symmetry(onedm, twodm):
    nparts = 2
    # setup temporal files
    # DM1
    ftmp1 = tempfile.NamedTemporaryFile(delete=True)
    fname1 = ftmp1.name + ".npy"
    np.save(fname1, onedm, allow_pickle=False)
    onedmfile = fname1
    # DM2
    ftmp2 = tempfile.NamedTemporaryFile(delete=True)
    fname2 = ftmp2.name + ".npy"
    np.save(fname2, twodm, allow_pickle=False)
    twodmfile = fname2

    with pytest.raises(ValueError):
        WfnRDMs(nparts, onedmfile, twodmfile)


@pytest.mark.parametrize(
    "nelec, filename, case", [(2, "be_sto3g", "dm1norm"), (4, "be_sto3g", "dm2norm")],
)
def test_verify_normalization(nelec, filename, case):
    onedm = find_datafiles("{}_onedm_spino.npy".format(filename))
    twodm = find_datafiles("{}_twodm_spino.npy".format(filename))
    if case == "dm2norm":
        # check normalization of 2-RDM
        twodm = 2 * np.load(twodm)
        ftmp2 = tempfile.NamedTemporaryFile(delete=True)
        fname2 = ftmp2.name + ".npy"
        np.save(fname2, twodm, allow_pickle=False)
        twodm = fname2

    with pytest.raises(ValueError):
        WfnRDMs(nelec, onedm, twodm)
