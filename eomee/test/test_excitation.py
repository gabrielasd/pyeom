
"""Test eomee.ionization."""


import eomee
from eomee.tools import find_datafiles

import numpy as np
from scipy.linalg import eig, svd


def test_ionization_heh_sto3g_symmetrized():
    """
    HeH+ sto3g
    
    """
    one_mo = np.load(find_datafiles('test/heh+_sto3g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('test/heh+_sto3g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_heh+_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_heh+_sto3g_genzd_anti.npy'))

    # One-electron integral symmetric permutations
    assert np.allclose(one_mo, one_mo.conj().T)
    # Two-electron integrals symmetric permutations
    assert np.allclose(two_mo, two_mo.conj().transpose((2,3,0,1)))
    assert np.allclose(two_mo, two_mo.transpose((3,2,1,0)))
    assert np.allclose(two_mo, two_mo.transpose((1,0,3,2)))
    # 1DM symmetric permutations
    assert np.allclose(one_dm, one_dm.conj().T)
    # 2DM symmetric permutations
    assert np.allclose(two_dm, two_dm.transpose((1,0,3,2)))
    assert np.allclose(two_dm, two_dm.transpose((3,2,1,0)))
    assert np.allclose(two_dm, two_dm.transpose((2,3,0,1)))
    # 2DM antisymmetric permutations
    assert np.allclose(two_dm, -two_dm.transpose((0,1,3,2)))
    assert np.allclose(two_dm, -two_dm.transpose((1,0,2,3)))
    assert np.allclose(two_dm, -two_dm.transpose((2,3,1,0)))
    assert np.allclose(two_dm, -two_dm.transpose((3,2,0,1)))

    eom = eomee.ExcitationEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # print(eom.lhs)
    # print(eom.neigs)
    print(aval)
    # Reference value
    e = 0.911
    assert abs(aval[0] - e) < 1e-6


test_ionization_heh_sto3g_symmetrized()