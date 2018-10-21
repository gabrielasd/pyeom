
"""Test eomee.ionization."""


import eomee
from eomee.tools import find_datafiles

import numpy as np
from scipy.linalg import eig, svd


def test_ionization_h2_sto6g_symmetrized():
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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    ie = 0.58205888
    print(ie)
    assert abs(aval[0] - ie) < 1e-8
    assert abs(aval[2] - ie) < 1e-8
    # Reference value from
    # https://webbook.nist.gov/chemistry/
    # IE = 15.42593 eV
    ie = 0.566892206
    assert abs(aval[0] - ie) < 0.1
    assert abs(aval[2] - ie) < 0.1


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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = [-1.52378328, -0.26764028]
    ie = 1.52378328
    assert abs(aval[0] - ie) < 1e-6


def test_ionization_he_ccpvdz_symmetrized():
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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # Reference value from
    # HORTON RHF
    ie = 0.91414765
    assert abs(aval[0] - ie) < 1e-6
    # Reference value from
    # https://webbook.nist.gov/chemistry/
    # IE = 24.58738880 eV
    ie = 0.90356944896
    assert abs(aval[0] - ie) < 0.1


def test_ionization_ne_sto6g_symmetrized():
    """
    Ne sto-6g antisymmetrized
    
    """
    one_mo = np.load(find_datafiles('test/ne_sto6g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('test/ne_sto6g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('test/1dm_ne_sto6g_genzd.npy'))
    two_dm = np.load(find_datafiles('test/2dm_ne_sto6g_genzd_anti.npy'))

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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # Reference value from
    # HORTON RHF
    ie = 0.5607611
    print(ie)
    assert abs(aval[-1] - ie) < 1e-8
    # Reference value from
    # https://webbook.nist.gov/chemistry/
    # IE =  21.564540 eV
    ie = 0.79248185659
    # THIS ASSERT FAILS
    # assert abs(aval[-1] - ie) < 0.1


def test_ionization_ne_321g_symmetrized():
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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # Reference value from
    # HORTON RHF
    # horton_emo = np.asarray([-32.56471038, -1.8651519, -0.79034293, -0.79034293, -0.79034293, 2.68726251, 2.68726251, 2.68726251, 4.08280903])
    ie = 0.79034293
    print(ie)
    print(aval[4])
    assert abs(aval[4] - ie) < 1e-5
    # assert np.allclose(-aval, horton_mos)
    # Reference value from
    # https://webbook.nist.gov/chemistry/
    # IE =  21.564540 eV
    ie = 0.79248185659
    assert abs(aval[4] - ie) < 0.01


def test_ionization_be_sto3g_symmetrized():
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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # Reference value from
    # HORTON RHF
    horton_mos = np.asarray([-4.48399211, -0.25403769, 0.22108596, 0.22108596, 0.22108596])
    ie = 0.25403769
    print(aval[1])
    assert abs(aval[1] - ie) < 1e-8
    assert np.allclose(-aval[:2], horton_mos[:2])
    # Reference value from
    # https://webbook.nist.gov/chemistry/
    # IE =  9.322699 eV
    ie = 0.3426027085
    assert abs(aval[1] - ie) < 0.1


def test_ionization_b_sto3g_symmetrized():
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

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    print(aval)
    # HORTON UHF alpha HOMO
    ie = 0.20051823
    assert abs(aval[-3] - ie) < 1e-8
    # Reference value from
    # https://webbook.nist.gov/chemistry/
    # IE =  9.322699 eV
    ie = 0.3426027085
    # This assert fails
    # assert abs(aval[-3] - ie) < 0.1
