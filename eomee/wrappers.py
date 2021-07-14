# Inspired on scripts by David Kim < david.kim.91@gmail.com >

"""Scripts to generate the electron integrals and reduced density matrices.

Functions
---------

.. code-block:: python hartreefock(mol) # Runs RHF in PySCF.

    pyscf_rdms(nelec, one_int, two_int, energy_nuc, is_phys=True, maxcicle=100, tol=1e-8)
        # Solve FCI problem and compute the 1 and 2-reduced density matrices with PySCF.

    pyci_rdms(nelec, one_int, two_int, energy_nuc, is_phys=True, wfn_type="doci", symmetric=True, n=1, ncv=30, tol=1.0e-6)
        # Solve CI problem and compute the 1 and 2-reduced density matrices with PyCI.

(Warning: These scripts have not been tested)

"""

import numpy as np
from pyscf import scf, ao2mo, gto, fci
import pyci


def hartreefock(mol):
    """Run RHF using PySCF.

    Parameters
    ----------
    mol : pyscf.gto.mole.Mole
        PySCF molecule instance.

    Returns
    -------
    result : dict
        "el_energy"
            The electronic energy.
        "nuc_nuc_energy"
            The nuclear repulsion energy.
        "one_int"
            The tuple of the one-electron interal.
        "two_int"
            The tuple of the two-electron integral in Physicist's notation.

    Raises
    ------
    ValueError
        If given mol is not a PySCF Mole instance.

    """
    if not mol.__class__.__name__ == "Mole":
        raise ValueError("`mol` must be a `pyscf.gto.mole.Mole` instance.")

    # run hf
    hf = scf.RHF(mol)
    # energies
    energy_nuc = hf.energy_nuc()
    energy_tot = hf.kernel()  # HF is solved here
    energy_elec = energy_tot - energy_nuc

    # Get integrals (See pyscf.gto.moleintor.getints_by_shell for other types of integrals)
    # Transform from AO to MO basis and change 2-electron integrals to physicist's notation.
    mo_coeff = hf.mo_coeff  # mo_coeffs
    # get 1e integral
    one_int_ab = mol.intor("cint1e_nuc_sph") + mol.intor("cint1e_kin_sph")
    one_mo = mo_coeff.T.dot(one_int_ab).dot(mo_coeff)
    # get 2e integral (PySCF uses Chemist's notation)
    eri = ao2mo.full(mol, mo_coeff, verbose=0, intor="cint2e_sph")
    two_int = ao2mo.restore(1, eri, mol.nao_nr())
    two_mo = np.einsum("ijkl->ikjl", two_int)

    # results
    result = {
        "nelec": mol.nelec,
        "el_energy": energy_elec,
        "nuc_nuc_energy": energy_nuc,
        "one_int": (one_mo,),
        "two_int": (two_mo,),
    }
    return result


def pyscf_rdms(nelec, one_int, two_int, energy_nuc, is_phys=True, maxcicle=100, tol=1e-8):
    """ Solve FCI problem and compute the 1 and 2-reduced density matrices with PySCF.

    Parameters
    ----------
    nelec : tuple (int, int)
        Number of alpha and beta electrons.
    one_int : numpy npz file
        One lectron integrals in the molecular orbital basis.
    two_int : numpy npz file
        Two lectron integrals in the molecular orbital basis.
    energy_nuc : float
        Nuclear repulsion energy.
    is_phys : bool, default=True
        Format of the two-electron integrals. If not specified it is assumed the integrals are in
        physicist notation and will be transformed to chemist notation (internal PySCF format).
    maxcicle : int, default=100
        Total number of iterations.
    tol : float, default=1e-8
        Energy convergence tolerance.

    Returns
    -------
    result : dict
        "el_energy"
            FCI electronic energy.
        "nuc_nuc_energy"
            The nuclear repulsion energy.
        "dm1"
            The tuple of the spin-resolved 1-electron density matrix.
        "two_int"
            The tuple of the spin-resolved two-electron density matrix.
    """
    # attributes
    h = one_int
    v = two_int
    norb = h.shape[0]

    if is_phys:
        # Transform physisist notation <ij|kl> to chemist notation (ik|jl) used by PySCF
        v = np.einsum("ijkl->ikjl", v)

    #
    # Run FCI
    # Create a FCI (=FCISolver) object because FCI object offers more options to control the
    # calculation.
    #
    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = maxcicle
    cisolver.conv_tol = tol
    energy_fci, fcivec = cisolver.kernel(h, v, norb, nelec, ecore=energy_nuc)

    # Compute spin-resolved 1- and 2-electron density matrices.
    # For 2-electron density matrix:
    # dm2aa corresponds to alpha spin for both 1st electron and 2nd electron
    # dm2ab corresponds to alpha spin for 1st electron and beta spin for 2nd electron
    # dm2bb corresponds to beta spin for both 1st electron and 2nd electron
    #
    if nelec is int:
        na, nb = nelec, nelec
    else:
        na, nb = nelec
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s(fcivec, norb, (na, nb))
    # Swap 2-RDM indices due to PySCF convention
    # 1pdm[p,q] = < |q^{\dagger} p| > ==> dm1[q,p]
    # 2pdm[p,q,r,s] = < |p^{\dagger} r^{\dagger} s q| > ==> dm2[p,r,q,s]
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    dm2bb = np.einsum("ijkl->ikjl", dm2bb)

    # Build generalized RDMs:
    rdm1 = np.zeros((2 * norb, 2 * norb))
    rdm1[:norb, :norb] = dm1a
    rdm1[norb:, norb:] = dm1b
    rdm2 = np.zeros((2 * norb,) * 4)
    # alpha alpha alpha alpha
    rdm2[:norb, :norb, :norb, :norb] = dm2aa
    # beta beta beta beta
    rdm2[norb:, norb:, norb:, norb:] = dm2bb
    # alpha beta alpha beta
    rdm2[:norb, norb:, :norb, norb:] = dm2ab
    # beta alpha beta alpha
    rdm2[norb:, :norb, norb:, :norb] = dm2ab.transpose((1, 0, 3, 2))
    # alpha beta beta alpha
    rdm2[:norb, norb:, norb:, :norb] = -dm2ab.transpose((0, 1, 3, 2))
    # beta alpha alpha beta
    rdm2[norb:, :norb, :norb, norb:] = -dm2ab.transpose((1, 0, 2, 3))

    # results
    output = {
        "el_energy": energy_fci - energy_nuc,
        "nuc_nuc_energy": energy_nuc,
        "dm1": (rdm1,),
        "dm2": (rdm2,),
    }
    return output


def pyci_rdms(
    nelec,
    one_int,
    two_int,
    energy_nuc,
    is_phys=True,
    wfn_type="doci",
    symmetric=True,
    n=1,
    ncv=30,
    tol=1.0e-6,
):
    """ Solve CI problem and compute the 1 and 2-reduced density matrices with PyCI.

    Parameters
    ----------
    nelec : tuple (int, int)
        Number of alpha and beta electrons.
    one_int : numpy npz file
        One lectron integrals in the molecular orbital basis.
    two_int : numpy npz file
        Two lectron integrals in the molecular orbital basis.
    energy_nuc : float
        Nuclear repulsion energy.
    is_phys : bool, default=True
        Format of the two-electron integrals. If not specified it is assumed the integrals are in
        physicist notation.
    wfn_type : str, default="doci"
        Wavefunction type supported by PyCI.
    tol : float, default=1e-8
        Energy convergence tolerance
    symmetric : bool, default=True
    n : int, default=1
        Number of eigenstates to be solved. Default is the ground state.
    ncv : int, default=30
        Number of Lanczos vectors to use.
    tol : float, default=1.0e-6
        Energy convergence tolerance

    Returns
    -------
    result : dict
        "el_energy"
            FCI electronic energy.
        "nuc_nuc_energy"
            The nuclear repulsion energy.
        "dm1"
            The spin-resolved 1-electron density matrix.
        "two_int"
            The spin-resolved two-electron density matrix.
    """
    # attributes
    ecore = energy_nuc
    h = one_int
    if nelec is int:
        na, nb = nelec, nelec
    else:
        na, nb = nelec
    if is_phys:
        v = two_int
    else:
        # Transform chemist notation (ik|jl) to physisist notation <ij|kl> used by PyCI
        v = np.einsum("ijkl->ikjl", two_int)

    # Make Hamiltonian and wavefunction
    ham = pyci.hamiltonian(ecore, h, v)
    if wfn_type == "doci":
        wfn = pyci.doci_wfn(ham.nbasis, na, nb)
    elif wfn_type == "fci":
        wfn = pyci.fullci_wfn(ham.nbasis, na, nb)
    else:
        raise NotImplementedError("Wavefunction type unsupported by this script.")

    # Solve CI problem
    op = pyci.sparse_op(ham, wfn, symmetric=symmetric)
    es, cs = pyci.solve(op, n=n, ncv=ncv, tol=tol)

    # Compute spin-resolved 1- and 2-electron density matrices.
    rdm1, rdm2 = pyci.compute_rdms(wfn, cs[0])

    # results
    output = {
        "el_energy": es[0] - energy_nuc,
        "nuc_nuc_energy": energy_nuc,
        "dm1": (rdm1,),
        "dm2": (rdm2,),
    }
    return output


if __name__ == "__main__":
    mol = gto.Mole()
    mol.build(
        atom="He 0 0 0; H 0 0 0.9295",  # in Angstrom
        basis="sto-6g",
        symmetry=False,
        charge=1,
        spin=0,  # = 2S = spin_up - spin_down
        unit="angstrom",
    )

    # data = hartreefock(mol)
    # one_int = data["one_int"][0]
    # two_int = data["two_int"][0]
    # fcidata = pyscf_rdms(mol.nelec, one_int, two_int, data["nuc_nuc_energy"])
    # print(data["nuc_nuc_energy"])
    # print(data["el_energy"] + data["nuc_nuc_energy"])
    # np.save('pyscfhartreefock', np.array([data["el_energy"], data["nuc_nuc_energy"]]))
    # np.save('one_int', data["one_int"][0])
    # np.save('two_int', data["two_int"][0])
