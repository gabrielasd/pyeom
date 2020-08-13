"""Test eomee.electronaff."""


import eomee
from eomee.tools import find_datafiles

import numpy as np
from scipy.linalg import eig, svd


def check_inputs_symm(oneint, twoint, onedm, twodm):
    """Check symmetry of electron integrals and Density Matrices."""
    # Electron integrals and DMs symmetric permutations
    assert np.allclose(oneint, oneint.T)
    assert np.allclose(onedm, onedm.T)
    assert np.allclose(twoint, np.einsum("pqrs->rspq", twoint))
    assert np.allclose(twoint, np.einsum("pqrs->qpsr", twoint))
    assert np.allclose(twodm, np.einsum("pqrs->rspq", twodm))
    assert np.allclose(twodm, np.einsum("pqrs->qpsr", twodm))
    # Two-electron integrals  and 2DM antisymmetric permutations
    assert np.allclose(twoint, -np.einsum("pqrs->pqsr", twoint))
    assert np.allclose(twoint, -np.einsum("pqrs->qprs", twoint))
    assert np.allclose(twodm, -np.einsum("pqrs->pqsr", twodm))
    assert np.allclose(twodm, -np.einsum("pqrs->qprs", twodm))


def test_electronffinities_one_body_term_H2():
    """Check that the one-body teerms of the electron affinities
    equations of motion are correct.

    """
    one_mo = np.load(find_datafiles("h2_sto6g_oneint_genzd.npy"))
    # the two-electron integrals are ignored
    two_mo = np.zeros((one_mo.shape[0],) * 4, dtype=one_mo.dtype)
    one_dm = np.load(find_datafiles("1dm_h2_sto6g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_h2_sto6g_genzd_anti.npy"))

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()

    w, v = eig(one_mo)
    ea = np.real(w)
    assert abs(sorted(aval1)[0] - ea[1]) < 1e-8
    assert abs(sorted(aval2)[-1] - ea[1]) < 1e-8
    assert abs(sorted(aval3)[0] - ea[1]) < 1e-8


def test_electronaff_h2_sto6g():
    """Test Electron Affinities EOMs for H2 (STO-6G)
    against Hartree-Fock canonical orbital energy.

    HF MO_a: 0.66587228

    """
    one_mo = np.load(find_datafiles("h2_sto6g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("h2_sto6g_twoint_genzd_anti.npy"))
    one_dm = np.load(find_datafiles("1dm_h2_sto6g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_h2_sto6g_genzd_anti.npy"))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = sorted(aval1)

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = sorted(aval2)

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = sorted(aval3)

    # Reference value from
    # HORTON RHF
    # horton_emo = [-0.58205888, 0.66587228]
    ea = 0.66587228
    assert abs(sorted(aval1)[-1] - ea) < 1e-8
    assert abs(sorted(aval2)[-1] - ea) < 1e-8
    assert abs(sorted(aval3)[-1] - ea) < 1e-8


def test_electronaff_heh_sto3g():
    """Test Electron Affinities EOMs for HeH+ (STO-3G)
    against Hartree-Fock canonical orbital energy.

    HF MO_a: -0.26764028

    """
    one_mo = np.load(find_datafiles("heh+_sto3g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("heh+_sto3g_twoint_genzd_anti.npy"))
    one_dm = np.load(find_datafiles("1dm_heh+_sto3g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_heh+_sto3g_genzd_anti.npy"))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = sorted(aval1)

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = sorted(aval2)

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = sorted(aval3)

    # Reference value from
    # HORTON RHF
    # horton_emo = [-1.52378328, -0.26764028]
    ea = -0.26764028
    assert abs(sorted(aval1)[0] - ea) < 1e-6
    assert abs(sorted(aval2)[-1] - ea) < 1e-6
    assert abs(sorted(aval3)[-1] - ea) < 1e-6


def test_electronaff_he_ccpvdz():
    """Test Electron Affinities EOMs for He (cc-pVDZ)
    against Hartree-Fock canonical orbital energy.

    HF MO_a: 1.39744193

    """
    one_mo = np.load(find_datafiles("he_ccpvdz_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("he_ccpvdz_twoint_genzd_anti.npy"))
    one_dm = np.load(find_datafiles("1dm_he_ccpvdz_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_he_ccpvdz_genzd_anti.npy"))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = sorted(aval1)

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = sorted(aval2)

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = sorted(aval3)

    # Reference value from
    # HORTON RHF
    # horton_emo = [-0.91414765, 1.39744193, 2.52437241, 2.52437241, 2.52437241]
    ea = 1.39744193
    assert abs(aval1[2] - ea) < 1e-6
    assert abs(aval2[2] - ea) < 1e-6
    assert abs(aval3[2] - ea) < 1e-6


def test_electronaff_ne_321g():
    """Test Electron Affinities EOMs for Ne (3-21G)
    against Hartree-Fock canonical orbital energy.

    HF MO_a: 2.68726251

    """
    one_mo = np.load(find_datafiles("ne_321g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("ne_321g_twoint_genzd_anti.npy"))
    one_dm = np.load(find_datafiles("1dm_ne_321g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_ne_321g_genzd_anti.npy"))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = sorted(aval1)

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = sorted(aval2)

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = sorted(aval3)

    # Reference value from
    # HORTON RHF
    # horton_emo = [-32.56471038, -1.8651519, -0.79034293, -0.79034293, -0.79034293, 2.68726251, 2.68726251, 2.68726251, 4.08280903]
    ea = 2.68726251
    assert abs(sorted(aval1)[-3] - ea) < 1e-5
    assert abs(sorted(aval2)[-3] - ea) < 1e-5
    assert abs(sorted(aval3)[-3] - ea) < 1e-5


def test_electronaff_be_sto3g():
    """Test Electron Affinities EOMs for Be (STO-3G)
    against Hartree-Fock canonical orbital energy.

    HF MO_a: 0.22108596

    """
    one_mo = np.load(find_datafiles("be_sto3g_oneint_genzd.npy"))
    two_mo = np.load(find_datafiles("be_sto3g_twoint_genzd_anti.npy"))
    one_dm = np.load(find_datafiles("1dm_be_sto3g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_be_sto3g_genzd_anti.npy"))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = sorted(aval1)

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = sorted(aval2)

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = sorted(aval3)

    # Reference value from
    # HORTON RHF
    horton_mos = np.asarray(
        [-4.48399211, -0.25403769, 0.22108596, 0.22108596, 0.22108596]
    )
    ea = 0.22108596
    assert abs(sorted(aval1)[-1] - ea) < 1e-8
    assert abs(sorted(aval2)[-1] - ea) < 1e-8
    assert abs(sorted(aval3)[-1] - ea) < 1e-8


def test_electronaff_b_sto3g():
    """Test Electron Affinities EOMs for B (STO-3G)
    against Hartree-Fock canonical orbital energies.

    HF MO_a: 0.29136562, 0.32299525, 0.38625451

    """
    one_mo = np.load(find_datafiles("1mo_b_sto3g_genzd.npy"))
    two_mo = np.load(find_datafiles("2mo_b_sto3g_genzd_anti.npy"))
    one_dm = np.load(find_datafiles("1dm_b_sto3g_genzd.npy"))
    two_dm = np.load(find_datafiles("2dm_b_sto3g_genzd_anti.npy"))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ElectronAffinitiesEOM1(one_mo, two_mo, one_dm, two_dm)
    aval1, avec = eom.solve_dense()
    aval1 = sorted(aval1)

    eom = eomee.ElectronAffinitiesEOM2(one_mo, two_mo, one_dm, two_dm)
    aval2, avec = eom.solve_dense()
    aval2 = sorted(aval2)

    eom = eomee.ElectronAffinitiesEOM3(one_mo, two_mo, one_dm, two_dm)
    aval3, avec = eom.solve_dense()
    aval3 = sorted(aval3)

    # HORTON UHF alpha HOMO
    # horton_emo_a = [-7.26583392, -0.428277, -0.20051823, 0.29136562, 0.29136562]
    # horton_emo_b = [-7.24421665, -0.31570904, 0.32299525, 0.32299525, 0.38625451]
    ea1 = 0.29136562
    ea2 = 0.32299525
    ea3 = 0.38625451
    assert abs(aval1[5] - ea1) < 1e-8
    assert abs(aval1[7] - ea2) < 1e-8
    assert abs(aval1[9] - ea3) < 1e-8
    assert abs(aval2[5] - ea1) < 1e-8
    assert abs(aval2[7] - ea2) < 1e-8
    assert abs(aval2[9] - ea3) < 1e-8
    assert abs(aval3[-4] - ea1) < 1e-8
    assert abs(aval3[-2] - ea2) < 1e-8
    assert abs(aval3[-1] - ea3) < 1e-8
