
"""Test eomee.electronaff."""


import eomee
from eomee.tools import find_datafiles

import numpy as np
from scipy.linalg import eig, svd


def test_electronaff_h2_sto6g_symmetrized():
    """
    H2 sto6g
    
    """
    one_mo = np.load(find_datafiles('test/h2_sto6g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('test/h2_sto6g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_h2_sto6g_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_h2_sto6g_genzd_anti.npy'))

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

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = sorted(aval)
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    ea = 0.66587228
    print(ea)
    assert abs(aval[2] - ea) < 1e-8


def test_electronaff_heh_sto3g_symmetrized():
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

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = sorted(aval)
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = [-1.52378328, -0.26764028]
    ea = -0.26764028
    assert abs(aval[0] - ea) < 1e-6


def test_electronaff_he_ccpvdz_symmetrized():
    """
    He ccpvdz antisymmetrized
    
    """
    one_mo = np.load(find_datafiles('test/he_ccpvdz_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('test/he_ccpvdz_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_he_ccpvdz_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_he_ccpvdz_genzd_anti.npy'))

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
    assert np.allclose(two_dm, -two_dm.transpose((0,1,3,2)))
    assert np.allclose(two_dm, -two_dm.transpose((1,0,2,3)))
    assert np.allclose(two_dm, -two_dm.transpose((2,3,1,0)))
    assert np.allclose(two_dm, -two_dm.transpose((3,2,0,1)))

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = sorted(aval)
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = [-0.91414765, 1.39744193, 2.52437241, 2.52437241, 2.52437241]
    ie = 1.39744193
    assert abs(aval[2] - ie) < 1e-6


def test_electronaff_ne_321g_symmetrized():
    """
    Ne 3-21g
    
    """
    one_mo = np.load(find_datafiles('test/ne_321g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('test/ne_321g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_ne_321g_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_ne_321g_genzd_anti.npy'))

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
    assert np.allclose(two_dm, -two_dm.transpose((0,1,3,2)))
    assert np.allclose(two_dm, -two_dm.transpose((1,0,2,3)))
    assert np.allclose(two_dm, -two_dm.transpose((2,3,1,0)))
    assert np.allclose(two_dm, -two_dm.transpose((3,2,0,1)))

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = sorted(aval)
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = [-32.56471038, -1.8651519, -0.79034293, -0.79034293, -0.79034293, 2.68726251, 2.68726251, 2.68726251, 4.08280903]
    ea = 2.68726251
    print(ea)
    print(aval[10])
    assert abs(aval[10] - ea) < 1e-5


def test_electronaff_be_sto3g_symmetrized():
    """
    Be sto-3g antisymmetrized

    """
    one_mo = np.load(find_datafiles('test/be_sto3g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('test/be_sto3g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_be_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_be_sto3g_genzd_anti.npy'))

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
    assert np.allclose(two_dm, -two_dm.transpose((0,1,3,2)))
    assert np.allclose(two_dm, -two_dm.transpose((1,0,2,3)))
    assert np.allclose(two_dm, -two_dm.transpose((2,3,1,0)))
    assert np.allclose(two_dm, -two_dm.transpose((3,2,0,1)))

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = sorted(aval)
    print(aval)
    # Reference value from
    # HORTON RHF
    horton_mos = np.asarray([-4.48399211, -0.25403769, 0.22108596, 0.22108596, 0.22108596])
    ea = 0.22108596
    print(aval[4])
    assert abs(aval[4] - ea) < 1e-8


def test_electronaff_b_sto3g_symmetrized():
    """
    B sto-3g
    
    """
    one_mo = np.load(find_datafiles('test/1mo_b_sto3g_genzd.npy'))
    two_mo = np.load(find_datafiles('test/2mo_b_sto3g_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_b_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_b_sto3g_genzd_anti.npy'))

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
    assert np.allclose(two_dm, -two_dm.transpose((0,1,3,2)))
    assert np.allclose(two_dm, -two_dm.transpose((1,0,2,3)))
    assert np.allclose(two_dm, -two_dm.transpose((2,3,1,0)))
    assert np.allclose(two_dm, -two_dm.transpose((3,2,0,1)))

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = sorted(aval)
    print(aval)
    # HORTON UHF alpha HOMO
    # horton_emo_a = [-7.26583392, -0.428277, -0.20051823, 0.29136562, 0.29136562]
    # horton_emo_b = [-7.24421665, -0.31570904, 0.32299525, 0.32299525, 0.38625451]
    ea1 = 0.29136562
    ea2 = 0.32299525
    ea3 = 0.38625451
    assert abs(aval[5] - ea1) < 1e-8
    assert abs(aval[7] - ea2) < 1e-8
    assert abs(aval[9] - ea3) < 1e-8
