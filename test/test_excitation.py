"""Test eomee.ionization."""


import numpy as np
from scipy.linalg import eig, svd
import pytest
from src.eom import EOMExc
from src import solver
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)


def test_eomexc_neigs():
    """

    """
    nspino = 4
    one_mo = np.arange(16, dtype=float).reshape(4,4)
    two_mo = np.arange(16*16, dtype=float).reshape(4,4,4,4)
    one_dm = np.zeros((4,4), dtype=float)
    one_dm[0,0], one_dm[2,2] = 1., 1.
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMExc(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == 4**2

def test_eomexc_heh_sto3g():
    """Test ExcitationEOM for HeH+ (STO-3G)
    against Gaussian's CIS computation.

    E_S1: 24.7959 eV

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("heh+_sto3g_oneint.npy"))
    one_mo = spinize(one_mo)
    two_mo = np.load(find_datafiles("heh+_sto3g_twoint.npy"))
    two_mo = symmetrize(spinize(two_mo))
    two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

    eom = EOMExc(one_mo, two_mo, one_dm, two_dm)
    aval, avec = solver.dense(eom.lhs, eom.rhs)
    aval = np.sort(aval)
    # Lowest excited singlet state fom Gaussian's CIS
    # E_S1 = 24.7959 eV = 0.91123209 Hartree
    e = 0.91123209
    assert abs(aval[-1] - e) < 1e-6
