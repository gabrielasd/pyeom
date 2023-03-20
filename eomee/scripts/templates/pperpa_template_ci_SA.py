"""Title: $title"""
import os

import numpy as np

import pyci
from iodata import load_one

from eomee.spinadapted.particleparticle import DEASA, DEA2SA
from eomee.tools import spinize, make_doci_ham_spinized


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


def run_acph(NAME, operator, solver, eigtol, _mult):
    methods = {'pp':DEASA, 'tdpp':DEA2SA}
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    
    # print('Load Molden file')
    # if not os.path.isfile(f"{NAME}.molden"):
    #     raise ValueError(f'f"{NAME}.molden" not found')
    # mol = load_one(f"{NAME}.molden")
    # nelec = int(mol.nelec)
    # na = nelec // 2
    # nb = nelec // 2
    # rdm1, rdm2 = hartreefock_rdms(ham.nbasis, na, nb)
    print('Load CI')
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    # rdm1, rdm2 = pyci.spinize_rdms(data['rdm1'], data['rdm2'])
    rdm1 = from_spins(data['rdm1'])
    dm2aa, dm2ab, dm2bb = data['rdm2'] # transform 2-RDMs to our notation <|p*q*sr|>=\Gamma_pqrs
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    dm2bb = np.einsum("ijkl->ikjl", dm2bb)
    rdm2 = from_spins([dm2aa, dm2ab, dm2ab, dm2bb])

    print('Run RPA (particle-particle)')
    if operator not in ['pp', 'tdpp']:
        raise ValueError('Invalid operator.')
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    method = methods[operator]
    erpa = method(h1, v1, rdm1, rdm2)
    print('finish building GEVP')
    print('solving...')
    w, cv = erpa.solve_dense(tol=eigtol, mode=solver, mult=_mult)
    
    np.savez(f"{NAME}.{operator}{_mult}{solver}.npz", energies=w, coeffs=cv)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7
mult = 1


run_acph(NAME, 'tdpp', 'qtrunc', eigtol, mult) # 'nonsymm'
