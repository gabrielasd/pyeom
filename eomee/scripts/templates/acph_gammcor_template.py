"""Title: $title"""
import os

import numpy as np

from iodata import load_one

import pyci

from eomee.excitation import EOMExc, WrappNonlinear,  _pherpa_linearterms
from eomee.tools import spinize


def fill_ham_inter(two_mo0, two_mo, set_i, set_j, dm1): 
    # for p in set_i:
    #     for q in set_j:
    #         # g_pqrs_aaaa
    #         two_mo0[p, p] += dm1[q,q]*two_mo[p,q,p,q]
    #         two_mo0[p, p] -= dm1[q,q]*two_mo[p,q,q,p]
    #         # g_pqrs_abab
    #         two_mo0[p, p] += dm1[q,q]*two_mo[p,q,p,q]
    for p in set_i:
        for q in set_i:
            for r in set_j:
                # g_pqrs_aaaa
                two_mo0 [p, q]+= dm1[r,r]*two_mo[p,r,q,r]
                two_mo0[p, q] -= dm1[r,r]*two_mo[p,r,r,q]
                # g_pqrs_abab
                two_mo0[p, q] += dm1[r,r]*two_mo[p,r,q,r]
    return two_mo0


def fill_ham_intra(one_mo0, two_mo0, one_mo, two_mo, set_i):
    # # abab
    # for p in set_i:
    #     one_mo0[p, p] += one_mo[p, p]
    #     two_mo0[p, p, p, p] += two_mo[p, p, p, p]
    #     for q in set_i:
    #         if p != q:
    #             two_mo0[p, p, q, q] += two_mo[p, p, q, q]
    for p in set_i:
        for q in set_i:
            one_mo0[p, q] = one_mo[p, q]
            for r in set_i:
                for s in set_i:
                    two_mo0[p, q, r, s] = two_mo[p, q, r, s]
    return one_mo0, two_mo0


def make_gvbpp_ham_zero(one_mo, two_mo, gem_matrix, dm1a):
    k = one_mo.shape[0]
    assert k == gem_matrix.shape[0]
    n_gems = gem_matrix.shape[1]    

    one_mo0 = np.zeros_like(one_mo)
    two_mo0 = np.zeros_like(two_mo)
    two_mo_inter = np.zeros_like(one_mo)
    for i in range(n_gems):
        gem_i = np.nonzero(gem_matrix.T[i])[0]
        one_mo_0, two_mo_0 = fill_ham_intra(one_mo0, two_mo0, one_mo, two_mo, gem_i)
        for j in range(n_gems):
            if j != i:
                gem_j = np.nonzero(gem_matrix.T[j])[0]
                two_mo_inter = fill_ham_inter(two_mo_inter, two_mo, gem_i, gem_j, dm1a)    

    return one_mo_0, two_mo_0, two_mo_inter


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


def run_acph(NAME, operator, solver, eigtol, alpha, epsilon=None):
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
    # FIXME: this is a hack to get the number of elgeminals right for H2 STO-6G
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

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_ham_zero(one_mo, two_mo, gem_mtrix, dm1aa)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)
    min, max, step = alpha
    alphas = np.arange(min, max, step)
    function = WrappNonlinear(EOMExc, h0, v0, dh, dv, rdm1, rdm2)

    if epsilon is None:
        dE_da = [function(alpha, mode=solver, tol=eigtol) for alpha in alphas]
    else:
        if not solver == 'nonsymm2':
            raise ValueError('Invalid eigensolver, must be `nonsymm2`.')
        dE_da = [function(alpha, mode=solver, tol=eigtol, eps=epsilon) for alpha in alphas]
    nonlinear = np.trapz(dE_da, dx=step)
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc
    # ecorr = EOMExc.erpa(h0, v0, h1, v1, rdm1, rdm2, nint=5, mode=solver, tol=eigtol)
    # etot = energy + ecorr + nucnuc

    # Save EOM results
    np.savez(f"{NAME}.ac{operator}{solver}.npz", energy=etot, ecorr=ecorr, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7
eps = 1.0e-2
alpha = [0.0, 1.0, 0.1]


run_acph(NAME, 'ph', 'nonsymm2', eigtol, alpha, epsilon=eps) #'safe', 'nonsymm
