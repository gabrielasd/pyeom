"""Title: $title"""
import gqcpy

import numpy as np


# Utility functions
def RHF_calculation(atcoords, charge, basisname, fname):
    print(f"Calculating RHF for {fname}")
    # Build GQCP molecule
    if isinstance(atcoords, str):
        molecule = gqcpy.Molecule.ReadXYZ(atcoords, charge=charge)
    else:
        raise NotImplementedError(f'Molecule {type(atcoords)} not implemented')

    # Run RHF
    # Build the molecular Hamiltonian and the objective function
    basis = gqcpy.RSpinOrbitalBasis_d(molecule, basisname)    
    hamiltonian = gqcpy.FQMolecularHamiltonian(molecule)
    rsq_hamiltonian = basis.quantize(hamiltonian)
    objective = gqcpy.DiagonalRHFFockMatrixObjective_d(rsq_hamiltonian, 1.0e-5)  
    # Create a RHF SCF solver and solve the SCF equations
    N = molecule.numberOfElectrons()
    S = basis.quantize(gqcpy.OverlapOperator())
    environment = gqcpy.RHFSCFEnvironment_d.WithCoreGuess(N, rsq_hamiltonian, S) 
    # solver = gqcpy.RHFSCFSolver_d.Plain(threshold=1.0e-04, maximum_number_of_iterations=5000)
    solver = gqcpy.RHFSCFSolver_d.DIIS(threshold=1.0e-04, maximum_number_of_iterations=5000)  # the system is not converging very rapidly
    qc_structure = gqcpy.RHF_d.optimize(objective, solver, environment)

    # Transform the Hamiltonian to the MO basis
    rhf_parameters = qc_structure.parameters()
    C = rhf_parameters.expansion()
    basis.transform(C)
    rsq_hamiltonian_mo = rsq_hamiltonian.transformed(C)

    nuc_rep = gqcpy.NuclearRepulsionOperator(molecule.nuclearFramework()).value()
    rhf_energy = qc_structure.groundStateEnergy() + nuc_rep
    print("RHF energy: ", rhf_energy)
    # SCF stability analysis
    stability_matrices = rhf_parameters.calculateStabilityMatrices(rsq_hamiltonian_mo)
    stability_matrices.printStabilityDescription()

    print('Saving RHF results')
    np.savez(f"{fname}.scf.npz", energy=rhf_energy, nuc=nuc_rep, coeff=C.matrix())
    
    return N, rsq_hamiltonian_mo, basis


def OODOCI_calculation(N_P, basis, hamiltonian, fname):
    print(f"Start OODOCI for {fname}")
    K = basis.numberOfSpatialOrbitals()
    
    onv_basis = gqcpy.SeniorityZeroONVBasis(K, N_P)
    x0 = gqcpy.LinearExpansion_SeniorityZero.HartreeFock(onv_basis).coefficients()
    solver_davidson = gqcpy.EigenproblemSolver.Davidson(maximum_number_of_iterations=10000, convergence_threshold=1.0e-05)
    environment_davidson = gqcpy.CIEnvironment.Iterative(hamiltonian, onv_basis, x0)
    
    optimizer = gqcpy.DOCINewtonOrbitalOptimizer(onv_basis, solver_davidson, environment_davidson)
    optimizer.optimize(basis, hamiltonian)

    nuc_rep = np.load(f"{fname}.scf.npz")['nuc']
    eigenvals = optimizer.eigenvalue() + nuc_rep
    eigenvecs = optimizer.makeLinearExpansion().coefficients()
    wfn = optimizer.makeLinearExpansion()

    one_int = hamiltonian.core().parameters()
    two_int = hamiltonian.twoElectron().parameters()
    print('Saving OODOCI results')
    np.savez(f"{fname}.ham.npz", onemo=one_int, twomo=two_int, nuc=nuc_rep)
    
    return eigenvals, eigenvecs, wfn


# Set up parameters
NAME = '$output'
GEOM = '$geometry'
CHARGE = $charge
BASIS = '$basis_set1'


# Run calculations
nelec, sq_hamiltonian, spinor_basis = RHF_calculation(GEOM, CHARGE, BASIS, NAME)

N_P = nelec // 2  # number of electron pairs
eigenvals, eigenvecs, wfn = OODOCI_calculation(N_P, spinor_basis, sq_hamiltonian, NAME)

dm1a = wfn.calculateSpinResolved1DM().alpha.matrix()
dm1b = wfn.calculateSpinResolved1DM().beta.matrix()
dm2aa = wfn.calculateSpinResolved2DM().alphaAlpha().tensor()
dm2ab = wfn.calculateSpinResolved2DM().alphaBeta().tensor()
dm2ba = wfn.calculateSpinResolved2DM().betaAlpha().tensor()
dm2bb = wfn.calculateSpinResolved2DM().betaBeta().tensor()
rdm1 = [dm1a, dm1b]
rdm2 = [dm2aa, dm2ab, dm2ba, dm2bb]

np.savez(f"{NAME}.ci.npz", energy=eigenvals, coeff=eigenvecs, rdm1=rdm1, rdm2=rdm2)
