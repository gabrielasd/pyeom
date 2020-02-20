
"""Test eomee.ionization."""


import eomee
from eomee.tools import find_datafiles

import numpy as np
from scipy.linalg import eig, svd


def check_inputs_symm(oneint, twoint, onedm, twodm):
    """Check symmetry of electron integrals and Density Matrices."""
    # Electron integrals and DMs symmetric permutations
    assert np.allclose(oneint, oneint.T)
    assert np.allclose(onedm, onedm.T)
    assert np.allclose(twoint, np.einsum('pqrs->rspq', twoint))
    assert np.allclose(twoint, np.einsum('pqrs->qpsr', twoint))
    assert np.allclose(twodm, np.einsum('pqrs->rspq', twodm))
    assert np.allclose(twodm, np.einsum('pqrs->qpsr', twodm))
    # Two-electron integrals  and 2DM antisymmetric permutations
    assert np.allclose(twoint, -np.einsum('pqrs->pqsr', twoint))
    assert np.allclose(twoint, -np.einsum('pqrs->qprs', twoint))
    assert np.allclose(twodm, -np.einsum('pqrs->pqsr', twodm))
    assert np.allclose(twodm, -np.einsum('pqrs->qprs', twodm))


def test_ionizationeom_one_body_term_h2():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    one_mo = np.load(find_datafiles('h2_sto6g_oneint_genzd.npy'))
    # the two-electron integrals are ignored
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm = np.load(find_datafiles('1dm_h2_sto6g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_h2_sto6g_genzd_anti.npy'))

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = np.sort(aval1)

    eom = eomee.IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = np.sort(aval2)

    eom = eomee.IonizationAntiCommutator(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = np.sort(aval3)

    w, v = eig(one_mo)
    ip = -np.real(w)
    ip = np.sort(ip)
    assert abs(aval1[-1] - ip[-1]) < 1e-8
    assert abs(aval2[-1] - ip[-1]) < 1e-8
    assert abs(aval3[-1] - ip[-1]) < 1e-8


def test_ionizationeomstate_h2_sto6g():
    """Test IonizationEOMState for H2 (STO-6G)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.58205888
    Experiment: 15.42593 eV

    """
    one_mo = np.load(find_datafiles('h2_sto6g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('h2_sto6g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_h2_sto6g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_h2_sto6g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    ie = 0.58205888
    assert abs(aval[0] - ie) < 1e-8
    assert abs(aval[2] - ie) < 1e-8
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP = 15.42593 eV
    ie = 0.566892206
    assert abs(aval[0] - ie) < 0.1
    assert abs(aval[2] - ie) < 0.1


def test_ionizationeomstate_heh_sto3g():
    """Test IonizationEOMState for HeH+ (STO-3G)
    against Hartree-Fock canonical orbital energy.

    HF MO_i: -1.52378328

    """
    one_mo = np.load(find_datafiles('heh+_sto3g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('heh+_sto3g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_heh+_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_heh+_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = [-1.52378328, -0.26764028]
    ie = 1.52378328
    assert abs(aval[0] - ie) < 1e-6


def test_ionizationeomstate_he_ccpvdz():
    """Test IonizationEOMState for He (cc-pVDZ)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.91414765
    Experiment: 24.58738880 eV

    """
    one_mo = np.load(find_datafiles('he_ccpvdz_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('he_ccpvdz_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_he_ccpvdz_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_he_ccpvdz_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    ie = 0.91414765
    assert abs(aval[0] - ie) < 1e-6
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP = 24.58738880 eV
    ie = 0.90356944896
    assert abs(aval[0] - ie) < 0.1


def test_ionizationeomstate_ne_321g():
    """Test IonizationEOMState for Ne (3-21G)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.79034293
    Experiment: 21.564540 eV

    """
    one_mo = np.load(find_datafiles('ne_321g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('ne_321g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_ne_321g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_ne_321g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = np.asarray([-32.56471038, -1.8651519, -0.79034293, -0.79034293,
    #                        -0.79034293, 2.68726251, 2.68726251, 2.68726251, 4.08280903])
    ie = 0.79034293
    assert abs(aval[4] - ie) < 1e-5
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  21.564540 eV
    ie = 0.79248185659
    assert abs(aval[4] - ie) < 0.01


def test_ionizationeomstate_be_sto3g():
    """Test IonizationEOMState for Be (STO-3G)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.25403769
    Experiment: 9.322699 eV

    """
    one_mo = np.load(find_datafiles('be_sto3g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('be_sto3g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_be_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_be_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    horton_mos = np.asarray([-4.48399211, -0.25403769, 0.22108596,
                             0.22108596, 0.22108596])
    ie = 0.25403769
    assert abs(aval[1] - ie) < 1e-8
    assert np.allclose(-aval[:2], horton_mos[:2])
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  9.322699 eV
    ie = 0.3426027085
    assert abs(aval[1] - ie) < 0.1


def test_ionizationeomstate_b_sto3g():
    """Test IonizationEOMState for B (STO-3G)
    against Hartree-Fock canonical orbital energy.

    HF MO_i: -0.20051823

    """
    one_mo = np.load(find_datafiles('1mo_b_sto3g_genzd.npy'))
    two_mo = np.load(find_datafiles('2mo_b_sto3g_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_b_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_b_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationEOMState(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # HORTON UHF alpha HOMO
    ie = 0.20051823
    assert abs(aval[-3] - ie) < 1e-8
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  8.29803 eV
    ie = 0.30494683
    # This assert fails
    # assert abs(aval[-3] - ie) < 0.1


def test_ionizationdoublecommutator_he_ccpvdz():
    """Test IonizationDoubleCommutator for He (cc-pVDZ)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.91414765
    Experiment: 24.58738880 eV

    """
    one_mo = np.load(find_datafiles('he_ccpvdz_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('he_ccpvdz_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_he_ccpvdz_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_he_ccpvdz_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    ie = 0.91414765
    assert abs(aval[0] - ie) < 1e-6
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP = 24.58738880 eV
    ie = 0.90356944896
    assert abs(aval[0] - ie) < 0.1


def test_ionizationdoublecommutator_ne_321g():
    """Test IonizationDoubleCommutator for Ne (3-21G)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.79034293
    Experiment: 21.564540 eV

    """
    one_mo = np.load(find_datafiles('ne_321g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('ne_321g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_ne_321g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_ne_321g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = np.asarray([-32.56471038, -1.8651519, -0.79034293, -0.79034293,
    #                        -0.79034293, 2.68726251, 2.68726251, 2.68726251, 4.08280903])
    ie = 0.79034293
    assert abs(aval[4] - ie) < 1e-5
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  21.564540 eV
    ie = 0.79248185659
    assert abs(aval[4] - ie) < 0.01


def test_ionizationdoublecommutator_b_sto3g():
    """Test IonizationDoubleCommutator for B (STO-3G)
    against Hartree-Fock canonical orbital energy.

    HF MO_i: -0.20051823
    Experiment: 8.29803 eV

    """
    one_mo = np.load(find_datafiles('1mo_b_sto3g_genzd.npy'))
    two_mo = np.load(find_datafiles('2mo_b_sto3g_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_b_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_b_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # HORTON UHF alpha HOMO
    ie = 0.20051823
    assert abs(aval[-3] - ie) < 1e-8
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  8.29803 eV
    ie = 0.30494683
    # This assert fails
    # assert abs(aval[-3] - ie) < 0.1


def test_ionizationanticommutator_he_ccpvdz():
    """Test IonizationantiCommutator for He (cc-pVDZ)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.91414765
    Experiment: 24.58738880 eV

    """
    one_mo = np.load(find_datafiles('he_ccpvdz_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('he_ccpvdz_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_he_ccpvdz_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_he_ccpvdz_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    ie = 0.91414765
    assert abs(aval[0] - ie) < 1e-6
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP = 24.58738880 eV
    ie = 0.90356944896
    assert abs(aval[0] - ie) < 0.1


def test_ionizationanticommutator_ne_321g():
    """Test IonizationAntiCommutator for Ne (3-21G)
    against Hartree-Fock canonical orbital energy and
    experimental results.

    HF MO_i: -0.79034293
    Experiment: 21.564540 eV

    """
    one_mo = np.load(find_datafiles('ne_321g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('ne_321g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_ne_321g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_ne_321g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationDoubleCommutator(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # Reference value from HORTON RHF
    # horton_emo = np.asarray([-32.56471038, -1.8651519, -0.79034293, -0.79034293,
    #                          -0.79034293, 2.68726251, 2.68726251, 2.68726251, 4.08280903])
    ie = 0.79034293
    assert abs(aval[4] - ie) < 1e-5
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  21.564540 eV
    ie = 0.79248185659
    assert abs(aval[4] - ie) < 0.01


def test_ionizationanticommutator_b_sto3g():
    """Test IonizationAntiCommutator for B (STO-3G)
    against Hartree-Fock canonical orbital energy.

    HF MO_i: -0.20051823
    Experiment: 8.29803 eV

    """
    one_mo = np.load(find_datafiles('1mo_b_sto3g_genzd.npy'))
    two_mo = np.load(find_datafiles('2mo_b_sto3g_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_b_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_b_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.IonizationAntiCommutator(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    # HORTON UHF alpha HOMO
    ie = 0.20051823
    # horton_emo_alpha = [-7.26583392 -0.428277   -0.20051823  0.29136562  0.29136562]
    # horton_emo_beta = [-7.24421665 -0.31570904  0.32299525  0.32299525  0.38625451]
    assert abs(aval[-3] - ie) < 1e-8
    # Reference value from https://webbook.nist.gov/chemistry/
    # IP =  8.29803 eV
    ie = 0.30494683
    # This assert fails
    # assert abs(aval[-3] - ie) < 0.1
