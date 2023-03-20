"""Title: $title"""
import os

import numpy as np

import pyci
from iodata import load_one

# from eomee.doubleelectronaff import EOMDEA, EOMDEA_2
from eomee.spinadapted.particleparticle import DEASA, DEA2SA
from eomee.tools import spinize, hartreefock_rdms


def run_acph(NAME, operator, solver, eigtol, _mult):
    methods = {'pp':DEASA, 'tdpp':DEA2SA}
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

    print('Run ERPA (particle-particle)')
    if operator not in ['pp', 'tdpp']:
        raise ValueError('Invalid operator.')
    h = spinize(one_mo) 
    v = spinize(two_mo)
    method = methods[operator]
    erpa = method(h, v, rdm1, rdm2)
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
