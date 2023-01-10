"""Title: $title"""
import os

import numpy as np

import pyci

from iodata import load_one

from eomee.spinadapted.holehole import DIPSA
from eomee.tools import spinize, hartreefock_rdms, make_spinized_fock_hamiltonian


def run_achh(NAME, operator, solver, eigtol):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    nucnuc = ham.ecore
    nbasis = ham.nbasis

    if not os.path.isfile(f"{NAME}.molden"):
        raise ValueError(f'f"{NAME}.molden" not found')
    mol = load_one(f"{NAME}.molden")
    nelec = int(mol.nelec)
    na = nelec // 2
    nb = nelec // 2
    rdm1, rdm2 = hartreefock_rdms(nbasis, na, nb)

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (hole-hole)')
    if operator != 'hh':
        raise ValueError('Invalid operator.')
    h0, v0 = make_spinized_fock_hamiltonian(one_mo, two_mo, rdm1)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    data = DIPSA.erpa(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, mult=1)
    int_vtdtd_s = data['ecorr'] - data['linear']
    data = DIPSA.erpa(h0, v0, h1, v1, rdm1, rdm2, solver=solver, eigtol=eigtol, mult=3)
    int_vtdtd_t = data['ecorr'] - data['linear']

    cnst = data['linear']
    int_vtdtd = int_vtdtd_s + int_vtdtd_t   # 1/2 faactor already included
    ecorr = cnst + int_vtdtd
    etot = energy + ecorr + nucnuc
    # Save ERPA results
    np.savez(f"{NAME}.ac{operator}{solver}.npz", energy=etot, ecorr=ecorr, ctnt=cnst, integ=int_vtdtd, intega=int_vtdtd_s, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7


run_achh(NAME, 'hh', 'qtrunc', eigtol)
