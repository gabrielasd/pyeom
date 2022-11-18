"""Title: $title"""
import os

import numpy as np

from scipy.integrate import quadrature as gauss

import pyci

# from eomee.excitation import EOMExc
from eomee.spinadapted.particlehole import EOMExcSA as EOMExc
from eomee.tools import spinize, make_gvbpp_hamiltonian, pickpositiveeig, TDM
# from eomee.solver import nonsymmetric as solver_d
# from eomee.solver import nonsymmetric2 as solver_d2
# from eomee.solver import eig_pinv as solver_d3


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


def run_acph(NAME, operator, solver, eigtol, summall):
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

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_mtrix, dm1aa)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    energy_ref = np.einsum('ij,ji', h1, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v1, rdm2)

    data = EOMExc.erpa_ecorr(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, summall=summall)
    ecorr = data['ecorr']
    linear = data['linear']
    etot = energy_ref + ecorr + nucnuc

    # Save EOM results
    np.savez(f"{NAME}.ac{operator}{solver}.npz", energy=etot, ecorr=ecorr, ctnt=linear, integ=None, intega=None, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-5
fulloperator = False

run_acph(NAME, 'ph', 'qtrunc', eigtol, fulloperator) # 'nonsymm' 'qtrunc'
