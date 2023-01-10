"""Title: $title"""
from sys import argv

import numpy as np

from pyscf import gto, scf

from pyscf.tools import molden, fcidump


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
    norb = hf.mo_coeff.shape[0]
    print(f"norb = {norb}, nelec = {nelec}")
    # mo_energy = hf.mo_energy
    # mo_occ = hf.mo_occ
    nuc_rep = hf.energy_nuc()

    print('Saving RHF results')
    np.savez(f"{fname}.scf.npz", energy=hf.e_tot, nuc=nuc_rep, coeff=None)


NAME = '$output'
GEOM = """$geometry"""
CHARGE = $charge
MULT = $spinmult

BASIS = """$basis_set1"""


# Run PySCF
run_scf(GEOM, CHARGE, MULT, BASIS, NAME, unit='A')
