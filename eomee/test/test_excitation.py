
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


def test_excitationeom_heh_sto3g():
    """Test ExcitationEOM for HeH+ (STO-3G)
    against Gaussian's CIS computation.

    E_S1: 24.7959 eV

    """
    one_mo = np.load(find_datafiles('heh+_sto3g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('heh+_sto3g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_heh+_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_heh+_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    eom = eomee.ExcitationEOM(one_mo, two_mo, one_dm, two_dm)
    aval, avec = eom.solve_dense()
    aval = np.sort(aval)
    # Lowest excited singlet state fom Gaussian's CIS
    # E_S1 = 24.7959 eV = 0.91123209 Hartree
    e = 0.91123209
    assert abs(aval[-1] - e) < 1e-6

def test_excitationeom_erpa_heh_sto3g():
    """Test Excitation ERPA for HeH+ (STO-3G)

    """
    one_mo = np.load(find_datafiles('heh+_sto3g_oneint_genzd.npy'))
    two_mo = np.load(find_datafiles('heh+_sto3g_twoint_genzd_anti.npy'))
    one_dm = np.load(find_datafiles('1dm_heh+_sto3g_genzd.npy'))
    two_dm = np.load(find_datafiles('2dm_heh+_sto3g_genzd_anti.npy'))
    check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    n = one_mo.shape[0]
    aa = one_mo[:1, :1]
    bb = one_mo[n//2:(n//2 + 1), n//2:(n//2 + 1)]
    aaaa = two_mo[:1, :1, :1, :1]
    abab = two_mo[:1, n//2:(n//2 + 1), :1, n//2:(n//2 + 1)]
    baba = two_mo[n//2:(n//2 + 1), :1, n//2:(n//2 + 1), :1]
    bbbb = two_mo[n//2:(n//2 + 1), n//2:(n//2 + 1), n//2:(n//2 + 1), n//2:(n//2 + 1)]
    one_mo_0 = np.zeros_like(one_mo)
    two_mo_0 = np.zeros_like(two_mo)
    one_mo_0[:1, :1] = aa
    one_mo_0[n//2:(n//2 + 1), n//2:(n//2 + 1)] = bb
    two_mo_0[:1, :1, :1, :1] = aaaa
    two_mo_0[:1, n//2:(n//2 + 1), :1, n//2:(n//2 + 1)] = abab
    two_mo_0[n//2:(n//2 + 1), :1, n//2:(n//2 + 1), :1] = baba
    two_mo_0[n//2:(n//2 + 1), n//2:(n//2 + 1), n//2:(n//2 + 1), n//2:(n//2 + 1)] = bbbb

    ecorr = eomee.ExcitationEOM.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
    print(ecorr)
