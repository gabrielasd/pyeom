"""Title: $title"""
import numpy as np
from functools import reduce

from pyscf import gto, scf, ao2mo
from pyscf import doci
from pyscf.tools import molden, fcidump

import pyci


def run_casscf(atcoords, charge, mult, basis, fname, unit='B'):
    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = atcoords
    mol.charge = charge
    mol.spin = mult - 1
    mol.basis = gto.basis.parse(basis)
    mol.unit = unit
    mol.build()

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
    # fcidump.from_scf(hf, f"{fname}.FCIDUMP", tol=0.0)
    print('Saving RHF results')
    np.savez(f"{fname}.scf.npz", energy=hf.e_tot, nuc=hf.energy_nuc(), coeff=None)

    # Run PySCF DOCI-CASSCF
    nelec = mol.nelectron
    norb = hf.mo_coeff.shape[0]
    mc = doci.CASSCF(hf, norb, nelec)
    # mc.conv_tol = 1e-12
    # mc.conv_tol_grad = 1e-6
    # mc.max_cycle_macro = 25
    mc.verbose = 4
    mc.kernel()
    print('Saving CASSCF results')
    c = mc.mo_coeff
    h1e = reduce(np.dot, (c.T, mc.get_hcore(), c))
    eri = ao2mo.kernel(hf.mol, c)
    energy_core = mc.get_h1eff(c)[1]
    # Overwrite FCIDUMP from SCF
    fcidump.from_integrals(f"{fname}.FCIDUMP", h1e, eri, c.shape[1], nelec, ms=0)
    np.savez(f"{fname}.ci.npz", energy=mc.e_tot, ecore=energy_core, mocoeff=c, ci=mc.ci)


def run_pyci(fname, n_procs):
    """run pyci v0.6.0"""
    mf = fcidump.to_scf(f"{fname}.FCIDUMP", molpro_orbsym=True)
    nelec = mf.mol.nelectron
    n_up, n_dn = ((nelec // 2), (nelec // 2))
    pyci.set_num_threads(int(n_procs))
    ham = pyci.hamiltonian(f"{fname}.FCIDUMP")
    wfn = pyci.doci_wfn(ham.nbasis, n_up, n_dn)
    wfn.add_all_dets()

    # Solve
    op = pyci.sparse_op(ham, wfn)
    ev, cv = op.solve(n=1, tol=1.0e-9)

    data = np.load(f"{fname}.ci.npz")
    etot = data['ecore'] + ev[0]
    assert np.allclose(etot, data['energy'])
    # Save 1,2-RDMs
    rdm1, rdm2 = pyci.compute_rdms(wfn, cv[0])
    np.savez(f"{NAME}.ci.npz", energy=etot, coeff=cv[0], rdm1=rdm1, rdm2=rdm2)
    


NAME = '$output'
GEOM = """$geometry"""
CHARGE = $charge
MULT = $spinmult
nprocs = $nprocs
BASIS = """$basis_set1"""


# Run PySCF
run_casscf(GEOM, CHARGE, MULT, BASIS, NAME, unit='A')
# run_casscf(NAME)

# Run PyCI
run_pyci(NAME, nprocs)
