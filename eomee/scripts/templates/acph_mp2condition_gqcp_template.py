"""Title: $title"""
import os

import numpy as np


# from eomee.excitation import EOMExc
from eomee.spinadapted.particlehole import EOMExcSA as EOMExc
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


def run_acph(NAME, operator, solver, eigtol, summall):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    two_mo = np.einsum('ijkl->ikjl', data["twomo"])
    nucnuc = data["nuc"]
    nbasis = one_mo.shape[0]
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    rdm1 = from_spins(data['rdm1'])
    dm2aa, dm2ab, dm2ba, dm2bb = data['rdm2'] # transform 2-RDMs to our notation <|p*q*sr|>=\Gamma_pqrs
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    dm2ba = np.einsum("ijkl->ikjl", dm2ba)
    dm2bb = np.einsum("ijkl->ikjl", dm2bb)
    rdm2 = from_spins([dm2aa, dm2ab, dm2ba, dm2bb])

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    ### Use explicit spinized representation of DOCI Hamiltonian
    h0, v0 = make_doci_ham_spinized(one_mo, two_mo)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    energy_ref = np.einsum('ij,ji', h1, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v1, rdm2)

    data = EOMExc.erpa_ecorr(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, summall=summall, mult=1)
    int_vtdtd_s = data['ecorr'] - data['linear']
    data = EOMExc.erpa_ecorr(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, summall=summall, mult=3)
    int_vtdtd_t = data['ecorr'] - data['linear']

    cnst = data['linear']
    int_vtdtd = int_vtdtd_s + int_vtdtd_t   # 1/2 factor already included
    ecorr = cnst + int_vtdtd
    etot = energy_ref + ecorr + nucnuc

    # Save EOM results
    if summall:
        mode='f' # full operator
    else:
        mode='t' # truncated operator
    np.savez(f"{NAME}.ac{mode}{operator}{solver}.npz", energy=etot, ecorr=ecorr, ctnt=cnst, integ=int_vtdtd, intega=int_vtdtd_s, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-5
fulloperator = False

run_acph(NAME, 'ph', 'qtrunc', eigtol, fulloperator) # 'nonsymm' 'qtrunc'
