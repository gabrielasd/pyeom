"""Title: $title"""
import os

import numpy as np

import pyci
from iodata import load_one

from eomee.doubleionization import EOMDIP, EOMDIP2
from eomee.tools import spinize, hartreefock_rdms
from eomee.solver import nonsymmetric as solver_d
from eomee.solver import eig_pinv as solver_d3


def _get_lhs_spinblocks(_erpa):
    nbasis = _erpa._n // 2
    lhs = _erpa.lhs.reshape(_erpa._n, _erpa._n, _erpa._n, _erpa._n)
    A_abab = lhs[:nbasis, nbasis:, :nbasis, nbasis:]
    A_baab = lhs[nbasis:, :nbasis, :nbasis, nbasis:]
    A_abba = lhs[:nbasis, nbasis:, nbasis:, :nbasis]    
    A_baba = lhs[nbasis:, :nbasis, nbasis:, :nbasis]
    return (A_abab, A_baab, A_abba, A_baba)

def _get_rhs_spinblocks(_erpa):
    nbasis = _erpa._n // 2
    rhs = _erpa.rhs.reshape(_erpa._n, _erpa._n, _erpa._n, _erpa._n)
    M_abab = rhs[:nbasis, nbasis:, :nbasis, nbasis:]
    M_baab = rhs[nbasis:, :nbasis, :nbasis, nbasis:]
    M_abba = rhs[:nbasis, nbasis:, nbasis:, :nbasis]    
    M_baba = rhs[nbasis:, :nbasis, nbasis:, :nbasis]
    return (M_abab, M_baab, M_abba, M_baba)


def spin_adapt_00(_erpa):
    A_abab, A_baab, A_abba, A_baba = _get_lhs_spinblocks(_erpa)
    M_abab, M_baab, M_abba, M_baba = _get_rhs_spinblocks(_erpa)
    nbasis = A_abab.shape[0]
    A_sa = A_abab - A_baab - A_abba + A_baba
    M_sa = M_abab - M_baab - M_abba + M_baba
    A_sa = 0.5*A_sa.reshape(nbasis**2, nbasis**2)
    M_sa = 0.5*M_sa.reshape(nbasis**2, nbasis**2)
    return A_sa, M_sa


def spin_adapt_30(_erpa):
    A_abab, A_baab, A_abba, A_baba = _get_lhs_spinblocks(_erpa)
    M_abab, M_baab, M_abba, M_baba = _get_rhs_spinblocks(_erpa)
    A_sa = A_abab + A_baab + A_abba + A_baba
    M_sa = M_abab + M_baab + M_abba + M_baba
    nbasis = A_abab.shape[0]
    A_sa = 0.5*A_sa.reshape(nbasis**2, nbasis**2)
    M_sa = 0.5*M_sa.reshape(nbasis**2, nbasis**2)
    return A_sa, M_sa


def run_acph(NAME, operator, solver, eigtol, _mult):
    _eigh = {'nonsymm': solver_d, 'qtrunc': solver_d3}
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    
    print('Load Molden file')
    if not os.path.isfile(f"{NAME}.molden"):
        raise ValueError(f'f"{NAME}.molden" not found')
    mol = load_one(f"{NAME}.molden")
    nelec = int(mol.nelec)
    na = nelec // 2
    nb = nelec // 2
    rdm1, rdm2 = hartreefock_rdms(ham.nbasis, na, nb)

    # Evaluate ERPA (hole-hole)
    print('Run ERPA (hole-hole)')
    if operator != 'hh':
        raise ValueError('Invalid operator.')
    h = spinize(one_mo) 
    v = spinize(two_mo)
    erpa = EOMDIP(h, v, rdm1, rdm2)
    if _mult == 1:
        lhs0, rhs0 = spin_adapt_00(erpa)
        w, cv = _eigh[solver](lhs0, rhs0, tol=eigtol)
        w, cv = w.real, cv.real
    elif _mult == 3:
        lhs3, rhs3 = spin_adapt_30(erpa)
        w, cv = _eigh[solver](lhs3, rhs3, tol=eigtol)
        w, cv = w.real, cv.real
    else:        
        raise ValueError('Invalid multiplicity.')
    
    np.savez(f"{NAME}.erpa{operator}{_mult}{solver}.npz", energies=w, coeffs=cv)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7
mult = 3


run_acph(NAME, 'hh', 'qtrunc', eigtol, mult) # 'nonsymm'
