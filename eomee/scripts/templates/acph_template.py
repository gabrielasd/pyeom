"""Title: $title"""
import os

import numpy as np

import pyci

from eomee.excitation import EOMExc, WrappNonlinear, _pherpa_linearterms
from eomee.tools import spinize


def make_doci_hamiltonian(one_mo, two_mo):
    nbasis = one_mo.shape[0]
    one_mo_sen0 = np.zeros_like(one_mo)
    two_mo_sen0 = np.zeros_like(two_mo)
    for p in range(nbasis):
        one_mo_sen0[p, p] = one_mo[p, p]
        for q in range(nbasis):
            two_mo_sen0[p, p, q, q] = two_mo[p, p, q, q]
            two_mo_sen0[p, q, p, q] = two_mo[p, q, p, q]
            two_mo_sen0[p, q, q, p] = two_mo[p, q, q, p]
    return one_mo_sen0, two_mo_sen0


def run_acph(NAME, operator, solver, eigtol, alpha, epsilon=None):
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

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    one_mo_0, two_mo_0 = make_doci_hamiltonian(one_mo, two_mo)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo) 
    h0 = spinize(one_mo_0) 
    v0 = spinize(two_mo_0)
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)
    min, max, step = alpha
    alphas = np.arange(min, max, step)
    function = WrappNonlinear(EOMExc, h0, v0, dh, dv, rdm1, rdm2)

    # dE_da = [function(alpha, mode=solver, tol=eigtol) for alpha in alphas]
    # nonlinear = np.trapz(dE_da, dx=step)
    # ecorr = linear + 0.5 * nonlinear
    # etot = energy + ecorr + nucnuc
    if epsilon is None:
        dE_da = [function(alpha, mode=solver, tol=eigtol) for alpha in alphas]
    else:
        if not solver == 'nonsymm2':
            raise ValueError('Invalid eigensolver, must be `nonsymm2`.')
        dE_da = [function(alpha, mode=solver, tol=eigtol, eps=epsilon) for alpha in alphas]
    nonlinear = np.trapz(dE_da, dx=step)
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc
    # Save EOM results
    # ecorr = EOMExc.erpa(h0, v0, h1, v1, rdm1, rdm2, nint=5, mode=solver, tol=eigtol)
    # etot = energy + ecorr + nucnuc

    np.savez(f"{NAME}.ac{operator}{solver}.npz", energy=etot, ecorr=ecorr, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7
eps = 1.0e-7
alpha = [0.0, 1.0, 0.1]


run_acph(NAME, 'ph', 'nonsymm', eigtol, alpha, epsilon=None)
