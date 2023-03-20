"""Title: $title"""
import os

import numpy as np

import pyci
from iodata import load_one

from eomee.ionization import EOMIP, EOMIPDoubleCommutator, EOMIPAntiCommutator
from eomee.tools import spinize, make_doci_ham_spinized


def run_acph(NAME, operator, solver, eigtol):
    methods = {'ip': EOMIP, 'ipc': EOMIPDoubleCommutator, 'ipa': EOMIPAntiCommutator}
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if os.path.isfile(f'{NAME}.FCIDUMP'):
        ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    elif os.path.isfile(f'{NAME}.fcidump'):
        ham = pyci.hamiltonian(f"{NAME}.fcidump")
    else:
        raise ValueError(f'{NAME}.FCIDUMP not found')
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    rdm1, rdm2 = pyci.spinize_rdms(data['rdm1'], data['rdm2'])

    print('Run Ionization Potential')
    if operator not in ['ip', 'ipc', 'ipa']:
        raise ValueError('Invalid operator.')
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    method = methods[operator]
    eom = method(h1, v1, rdm1, rdm2)
    w, cv = eom.solve_dense(tol=eigtol, mode=solver)
    
    np.savez(f"{NAME}.{operator}{solver}.npz", energies=w, coeffs=cv)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7


run_acph(NAME, 'ipa', 'nonsymm', eigtol) # 'nonsymm' 'qtrunc'
