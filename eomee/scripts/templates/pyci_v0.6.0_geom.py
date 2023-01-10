"""Title: $title"""
from sys import argv

import numpy as np

from pyscf import gto, scf

from pyscf.tools import molden, fcidump

import pyci


method = {'fci': pyci.fullci_wfn, 'doci': pyci.doci_wfn}


def run_scf(atcoords, charge, mult, basis, fname, unit='B'):
    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = atcoords
    mol.charge = charge
    mol.spin = mult - 1
    mol.basis = gto.basis.parse(basis)
    mol.unit = unit
    mol.build()
    # print(mol.atom_coords())

    # Run PySCF ROHF
    hf = scf.RHF(mol)
    hf.conv_tol = 1.0e-12
    hf.run()
    # Run internal stability analysis for the SCF wave function (default: internal)
    mo1 = hf.stability()[0]
    dm1 = hf.make_rdm1(mo1, hf.mo_occ)
    hf = hf.run(dm1)
    hf.stability()

    # Write MOLDEN and FCIDUMP
    molden.from_scf(hf, f"{fname}.molden", ignore_h=False)
    fcidump.from_scf(hf, f"{fname}.FCIDUMP", tol=0.0)

    nelec = mol.nelec
    # mo_energy = hf.mo_energy
    # mo_occ = hf.mo_occ

    return (nelec, hf.e_tot, hf.energy_nuc())


def run_pyci(n_up, n_dn, fname, n_procs, nsol=1, wfn_type=pyci.fullci_wfn):
    """run pyci v0.6.0"""
    pyci.set_num_threads(int(n_procs))
    ham = pyci.hamiltonian(f"{fname}.FCIDUMP")
    wfn = wfn_type(ham.nbasis, n_up, n_dn)
    wfn.add_all_dets()

    # Solve
    op = pyci.sparse_op(ham, wfn)
    ev, cv = op.solve(n=nsol, tol=1.0e-9)
    return ev, cv, wfn


NAME = '$output'

# ELEM = NAME.split("_", 1)[0]
GEOM = '$geometry'
CHARGE = $charge
MULT = $spinmult
nprocs = $nprocs
WFN = '$lot'.lower()

BASIS = """$basis_set1"""


# Run PySCF
n_elec = run_scf(GEOM, CHARGE, MULT, BASIS, NAME, unit='A')[0]

# n_elec = int(ELEM) - CHARGE
# n_a = (n_elec + MULT - 1) // 2
# n_b = n_elec - n_a
n_a, n_b = n_elec

# Run HCI
eigenvals, eigenvecs, pyci_wfn = run_pyci(n_a, n_b, NAME, nprocs, wfn_type=method[WFN], nsol=1)
energy = eigenvals[0]
print("HCI energy", energy)

# Compute HCI RDMs
rdm1, rdm2 = pyci.compute_rdms(pyci_wfn, eigenvecs[0])
# Save HCI data
np.savez(f"{NAME}.ci.npz", energy=eigenvals, coeff=eigenvecs, dets=pyci_wfn.to_occ_array(), rdm1=rdm1, rdm2=rdm2)
