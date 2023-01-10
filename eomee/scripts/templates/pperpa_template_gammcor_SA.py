"""Title: $title"""
import os

import numpy as np


from eomee.doubleelectronaff import EOMDEA, EOMDEA_2
from eomee.tools import spinize, make_gvbpp_hamiltonian
from eomee.solver import nonsymmetric as solver_d
from eomee.solver import eig_pinv as solver_d3


def from_spins(blocks):
    r"""
    Return a two- or four- index array in the spin representation from blocks.

    A two-index array is recontrcuted from blocks (a, b).
    A four-index array is recontrcuted from blocks (aa, ab, ba, bb).

    """
    if len(blocks) == 2:
        for b in blocks:
            if b.ndim != 2:
                raise ValueError("Input must have ndim == 2")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k))
        y[:n, :n] = blocks[0]
        y[n:, n:] = blocks[1]
    elif len(blocks) == 4:
        for b in blocks:
            if b.ndim != 4:
                raise ValueError("Input must have ndim == 4")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k, k, k))
        y[:n, :n, :n, :n] = blocks[0]
        y[:n, n:, :n, n:] = blocks[1]
        y[n:, :n, n:, :n] = blocks[2]
        y[n:, n:, n:, n:] = blocks[3]
        y[:n, n:, n:, :n] = -blocks[1].transpose((0,1,3,2))
        y[n:, :n, :n, n:] = -blocks[1].transpose((1,0,2,3))
    else:
        raise ValueError("Invalid input")
    return y


def _get_lhs_spinblocks(_n, _erpa_l):
    nbasis = _n//2 #_erpa._n // 2
    # lhs = _erpa.lhs.reshape(_erpa._n, _erpa._n, _erpa._n, _erpa._n)
    lhs = _erpa_l.reshape(_n, _n, _n, _n)
    A_abab = lhs[:nbasis, nbasis:, :nbasis, nbasis:]
    A_baab = lhs[nbasis:, :nbasis, :nbasis, nbasis:]
    A_abba = lhs[:nbasis, nbasis:, nbasis:, :nbasis]    
    A_baba = lhs[nbasis:, :nbasis, nbasis:, :nbasis]
    return (A_abab, A_baab, A_abba, A_baba)

def _get_rhs_spinblocks(_n, _erpa_r):
    nbasis = _n//2 #_erpa._n // 2
    # rhs = _erpa.rhs.reshape(_erpa._n, _erpa._n, _erpa._n, _erpa._n)
    rhs = _erpa_r.reshape(_n, _n, _n, _n)
    M_abab = rhs[:nbasis, nbasis:, :nbasis, nbasis:]
    M_baab = rhs[nbasis:, :nbasis, :nbasis, nbasis:]
    M_abba = rhs[:nbasis, nbasis:, nbasis:, :nbasis]    
    M_baba = rhs[nbasis:, :nbasis, nbasis:, :nbasis]
    return (M_abab, M_baab, M_abba, M_baba)


def spin_adapt_00(n, _erpa_l, _erpa_r):
    A_abab, A_baab, A_abba, A_baba = _get_lhs_spinblocks(n, _erpa_l)
    M_abab, M_baab, M_abba, M_baba = _get_rhs_spinblocks(n, _erpa_r)
    nbasis = A_abab.shape[0]
    A_sa = A_abab - A_baab - A_abba + A_baba
    M_sa = M_abab - M_baab - M_abba + M_baba
    A_sa = 0.5*A_sa.reshape(nbasis**2, nbasis**2)
    M_sa = 0.5*M_sa.reshape(nbasis**2, nbasis**2)
    return A_sa, M_sa


def spin_adapt_30(n, _erpa_l, _erpa_r):
    A_abab, A_baab, A_abba, A_baba = _get_lhs_spinblocks(n, _erpa_l)
    M_abab, M_baab, M_abba, M_baba = _get_rhs_spinblocks(n, _erpa_r)
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
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    two_mo = data["twomo"]
    nucnuc = data["nuc"]
    nbasis = one_mo.shape[0]
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.gvb.npz"):
        raise ValueError(f"{NAME}.gvb.npz not found")
    data = np.load(f"{NAME}.gvb.npz")
    dm1aa, dm1ab = data['rdm1']
    rdm1 = from_spins([dm1aa, dm1ab])
    dm2aaaa, dm2abab = data['rdm2']
    dm2baba = dm2abab.transpose((1,0,3,2))
    rdm2 = from_spins([dm2aaaa, dm2abab, dm2baba, dm2aaaa])
    
    nparts = int(np.ceil(np.trace(rdm1)))
    npairs = nparts // 2
    ngems = npairs + 1     # considering fictitious geminals

    print('Load Geminals data')
    if not os.path.isfile("gvb_geminals.dat"):
        raise ValueError(f"gvb_geminals.dat not found")
    index_m = np.loadtxt("gvb_geminals.dat", dtype=int)
    assert index_m.shape[0] == nbasis
    assert index_m[nbasis-1, 1] == ngems
    index_m -= 1
    gem_mtrix = np.zeros((nbasis, ngems))
    for n,g in index_m:
        gem_mtrix[n, g] = 1.0

    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_mtrix, dm1aa)
    h = spinize(one_mo_0) 
    h += spinize(two_mo_0_inter)
    v = spinize(two_mo_0)

    # Evaluate ERPA (particle-particle)
    print('Run ERPA (particle-particle)')
    if operator != 'pp':
        raise ValueError('Invalid operator.')
    erpa = EOMDEA_2(h, v, rdm1, rdm2)
    print('finish building GEVP')
    print('solving...')
    if _mult == 1:
        lhs0, rhs0 = spin_adapt_00(erpa._n, erpa.lhs, erpa.rhs)
        w, cv = _eigh[solver](lhs0, rhs0, tol=eigtol)
        w, cv = w.real, cv.real
    elif _mult == 3:
        lhs3, rhs3 = spin_adapt_30(erpa._n, erpa.lhs, erpa.rhs)
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
mult = 1


run_acph(NAME, 'pp', 'qtrunc', eigtol, mult) # 'nonsymm'
