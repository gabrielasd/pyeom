"""Title: $title"""
import gqcpy

import numpy as np


# Utility functions
def _rhf(molecule, basis, guess=None):
    # Build the molecular Hamiltonian and the objective function
    N = molecule.numberOfElectrons()
    S = basis.quantize(gqcpy.OverlapOperator())

    hamiltonian = gqcpy.FQMolecularHamiltonian(molecule)
    rsq_hamiltonian = basis.quantize(hamiltonian)
    objective = gqcpy.DiagonalRHFFockMatrixObjective_d(rsq_hamiltonian, 1.0e-5) 

    # Create a RHF SCF solver and solve the SCF equations
    if guess == None:
        environment = gqcpy.GHFSCFEnvironment_d.WithCoreGuess(N, rsq_hamiltonian, S) 
    else:
        environment = gqcpy.GHFSCFEnvironment_d(N, rsq_hamiltonian, S, guess)
    # solver = gqcpy.RHFSCFSolver_d.Plain(threshold=1.0e-04, maximum_number_of_iterations=5000)
    solver = gqcpy.RHFSCFSolver_d.DIIS(threshold=1.0e-04, maximum_number_of_iterations=5000)  # the system is not converging very rapidly
    _qc_structure = gqcpy.RHF_d.optimize(objective, solver, environment)

    # Transform the Hamiltonian to the MO basis
    rhf_parameters = _qc_structure.parameters()
    C = rhf_parameters.expansion()
    basis.transform(C)
    _rsq_hamiltonian_mo = rsq_hamiltonian.transformed(C)
    
    return _qc_structure, _rsq_hamiltonian_mo


def calculateNewGuess(restricted_stability_matrices, rhf_qc_model):
    from scipy import linalg as la
    
    _stability_matrix = restricted_stability_matrices.internal()

    # diagonalize the stability matrix    
    eigenvalues, eigenvectors = np.linalg.eigh(_stability_matrix)

    # Extract the lowest eigenvector
    lowest_eigenvector = eigenvectors[:, 0]

    # Create the rotation matrix that rotates the coefficients to the lowest eigenvector
    occupied_orbitals =  rhf_qc_model.numberOfElectrons() // 2
    coeffs = rhf_qc_model.expansion().matrix()
    virtual_orbitals = int(len(coeffs[0]) - occupied_orbitals)
    
    K = np.zeros(((occupied_orbitals + virtual_orbitals), (occupied_orbitals + virtual_orbitals)))
 
    lowest_eigenvector = lowest_eigenvector.reshape((occupied_orbitals, virtual_orbitals))

    K[occupied_orbitals:, :occupied_orbitals] = -1 * lowest_eigenvector.T.conjugate()
    K[:occupied_orbitals, occupied_orbitals:] = lowest_eigenvector

    rotated_coefficients = coeffs @ la.expm(-K)

    return gqcpy.GTransformation_d(rotated_coefficients)


def RHF_calculation(atcoords, charge, basisname, fname):
    print(f"Calculating RHF for {fname}")
    # Build GQCP molecule
    if isinstance(atcoords, str):
        molecule = gqcpy.Molecule.ReadXYZ(atcoords, charge=charge)
    else:
        raise NotImplementedError(f'Molecule {type(atcoords)} not implemented')

    # Run RHF
    N = molecule.numberOfElectrons()
    basis = gqcpy.RSpinOrbitalBasis_d(molecule, basisname)
    qc_structure, rsq_hamiltonian_mo = _rhf(molecule, basis)
    rhf_parameters = qc_structure.parameters()

    # SCF stability analysis
    stability_matrices = rhf_parameters.calculateStabilityMatrices(rsq_hamiltonian_mo)
    stability_matrices.printStabilityDescription()
    if not stability_matrices.isInternallyStable():
        print("Solve RHF with new guess")
        new_guess = calculateNewGuess(stability_matrices, rhf_parameters)
        qc_structure, rsq_hamiltonian_mo = _rhf(molecule, basis, guess=new_guess)
        rhf_parameters = qc_structure.parameters()
        stability_matrices = rhf_parameters.calculateStabilityMatrices(rsq_hamiltonian_mo)
        stability_matrices.printStabilityDescription()
    
    nuc_rep = gqcpy.NuclearRepulsionOperator(molecule.nuclearFramework()).value()
    rhf_energy = qc_structure.groundStateEnergy() + nuc_rep
    C = rhf_parameters.expansion()
    print("RHF energy   = ", rhf_energy)

    print('Saving RHF results')
    np.savez(f"{fname}.scf.npz", energy=rhf_energy, nuc=nuc_rep, coeff=C.matrix())
    
    return N, rsq_hamiltonian_mo, basis


def OODOCI_calculation(N_P, basis, hamiltonian, fname):
    print(f"Start OODOCI for {fname}")
    K = basis.numberOfSpatialOrbitals()
    
    onv_basis = gqcpy.SeniorityZeroONVBasis(K, N_P)
    solver = gqcpy.EigenproblemSolver.Dense_d()
    environment = gqcpy.CIEnvironment.Dense(hamiltonian, onv_basis)
    
    optimizer = gqcpy.DOCINewtonOrbitalOptimizer(onv_basis, solver, environment)
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
