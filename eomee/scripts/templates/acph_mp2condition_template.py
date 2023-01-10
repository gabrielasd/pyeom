"""Title: $title"""
import os

import numpy as np

import pyci

from eomee.spinadapted.particlehole import EOMExcSA as EOMExc
from eomee.tools import spinize, make_doci_ham_spinized


def run_acph(NAME, operator, solver, eigtol, summall):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    nucnuc = ham.ecore
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    rdm1, rdm2 = pyci.spinize_rdms(data['rdm1'], data['rdm2'])

    # Evaluate AC-ERPA
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    h0, v0 = make_doci_ham_spinized(one_mo, two_mo)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    energy_ref = np.einsum('ij,ji', h1, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v1, rdm2)

    data = EOMExc.erpa_ecorr(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, summall=summall, mult=1)
    int_vtdtd_s = data['ecorr'] - data['linear']
    data = EOMExc.erpa_ecorr(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, summall=summall, mult=3)
    int_vtdtd_t = data['ecorr'] - data['linear']

    cnst = data['linear']
    int_vtdtd = int_vtdtd_s + int_vtdtd_t   # 1/2 faactor already included
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

run_acph(NAME, 'ph', 'qtrunc', eigtol, fulloperator) # 'nonsymm' 
