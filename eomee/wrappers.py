# Inspired on scripts by David Kim < david.kim.91@gmail.com >

"""Scripts to generate the electron integrals and reduced density matrices.

Functions
---------

.. code-block:: python hartreefock(mol) # Runs RHF in PySCF.

    FullCIRDMs(nelec, one_int, two_int, ecore, is_phys=True, **kwargs).get_rdms()
        # Solve the FCI problem and compute the 1 and 2-reduced density matrices with PySCF.

    PyCIRDMs.from_integrals(nelec, one_int, two_int, ecore, wfn_type, **kwargs).get_rdms()
        # Solve the CI problem and compute the 1 and 2-reduced density matrices with PyCI.

(Warning: These scripts have not been tested)

"""

import numpy as np
from pyscf import scf, ao2mo, gto, fci, tools
import pyci


def hartreefock_pyscf(mol):
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
    # get 2e integral (PySCF uses Chemist's notation) and
    # transform to physisist notation.
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


class PyCIRDMs(object):
    """ Solve the CI problem and compute the 1 and 2-reduced density matrices with PyCI.

    Parameters
    ----------
    wfn : pyci.wavefunction
        A PyCI wavefunction instance.
    ham : pyci.hamiltonian
        A PyCI Hamiltonian instance.
    tol : float, default=1e-8
        Energy convergence tolerance.
    symmetric : bool, default=True
    n : int, default=4
        Number of eigenstates to be solved. Default is the ground state.
    ncv : int, default=4
        Number of Lanczos vectors to use.
    tol : float, default=1.0e-6
        Energy convergence tolerance.
    """

    def __init__(self, wfn, ham, **kwargs):
        if not isinstance(wfn, (pyci.doci_wfn, pyci.fullci_wfn, pyci.genci_wfn)):
            raise ValueError("`wfn` must be a PyCI wavefunction instance.")
        if not isinstance(ham, pyci.hamiltonian):
            raise ValueError("`wfn` must be a PyCI Hamiltonian instance.")
        # attributes
        self._wfn = wfn
        self._ham = ham
        if "symmetric" not in kwargs.keys():
            self._symmetric = True
        else:
            self._symmetric = kwargs["symmetric"]
        if "n" not in kwargs.keys():
            self._n = 1
        else:
            self._n = kwargs["n"]
        if "ncv" not in kwargs.keys():
            self._ncv = 4
        else:
            self._ncv = kwargs["ncv"]
        if "tol" not in kwargs.keys():
            self._tol = 1.0e-6
        else:
            self._tol = kwargs["tol"]

        # Solve CI problem
        op = pyci.sparse_op(self._ham, self._wfn, symmetric=self._symmetric)
        es, cs = pyci.solve(op, n=self._n, ncv=self._ncv, tol=self._tol)
        self._es = es
        self._cs = cs

    @classmethod
    def from_fcidump(cls, nelec, fname, wfn_type, **kwargs):
        """initialize class from a FCIDUMP file.

        Parameters
        ----------
        nelec : int, tuple or list of integers
            Number of alpha and beta electrons.
        fname : str
            Path to FCIDUMP file.
        wfn_type : str
            Wavefunction type supported by PyCI.

        Returns
        -------
        An instance of the class PyCIRDMs.

        Raises
        ------
        ValueError
            If `nelec` is not one of int, tuple or list of integers.
        NotImplementedError
            If `wfn_type` is not one of `doci` or `fci`.
        """
        if isinstance(nelec, int):
            na, nb = nelec, nelec
        elif isinstance(nelec, (tuple, list)):
            na, nb = nelec
        else:
            raise ValueError(f"`nelec` must be int, tuple or list; {nelec} was passed")
        # Make PyCI Hamiltonian and wavefunction
        ham = pyci.hamiltonian(fname)
        if wfn_type == "doci":
            wfn = pyci.doci_wfn(ham.nbasis, na, nb)
        elif wfn_type == "fci":
            wfn = pyci.fullci_wfn(ham.nbasis, na, nb)
        else:
            raise NotImplementedError("Wavefunction type unsupported by this script.")
        wfn.add_all_dets()
        return cls(wfn, ham, **kwargs)

    @classmethod
    def from_integrals(cls, nelec, one_int, two_int, ecore, wfn_type, **kwargs):
        """initialize class from electron integrals in npy format.

        Parameters
        ----------
        nelec : int, tuple or list of integers
            Number of alpha and beta electrons.
        h : (n,n) numpy array
            One lectron integrals in the molecular orbital basis.
        v : (n,n,n,n) numpy array
            Two lectron integrals in the molecular orbital basis.
        ecore : float
            Nuclear repulsion energy.
        wfn_type : str
            Wavefunction type supported by PyCI.
        is_phys : bool, default=True
            Format of the two-electron integrals. If not specified it is assumed the integrals are in
            physicist notation.

        Returns
        -------
        An instance of the class PyCIRDMs.

        Raises
        ------
        ValueError
            If `nelec` is not one of int, tuple or list of integers.
        """
        if isinstance(nelec, int):
            na, nb = nelec, nelec
        elif isinstance(nelec, (tuple, list)):
            na, nb = nelec
        else:
            raise ValueError(f"`nelec` must be int, tuple or list; {nelec} was passed")
        if "is_phys" not in kwargs.keys():
            is_phys = True
        else:
            is_phys = kwargs["is_phys"]
        if not is_phys:
            # Transform chemist notation (ik|jl) to physisist notation <ij|kl> used by PyCI
            two_int = np.einsum("ijkl->ikjl", two_int)
        # Make Hamiltonian and wavefunction
        ham = pyci.hamiltonian(ecore, one_int, two_int)
        if wfn_type == "doci":
            wfn = pyci.doci_wfn(ham.nbasis, na, nb)
        elif wfn_type == "fci":
            wfn = pyci.fullci_wfn(ham.nbasis, na, nb)
        else:
            raise NotImplementedError("Wavefunction type unsupported by this script.")
        wfn.add_all_dets()
        return cls(wfn, ham, **kwargs)

    def get_rdms(self):
        """Compute spin-resolved 1- and 2-electron density matrices."""
        return pyci.compute_rdms(self._wfn, self._cs[0])

    def get_outputs(self):
        """Retrieve energy and DMs from CI calculation.

        Returns
        -------
        result : dict
            "el_energy"
                FCI electronic energy.
            "nuc_nuc_energy"
                The nuclear repulsion energy.
            "dm1"
                The spin-resolved 1-electron density matrix.
            "dm2"
                The spin-resolved two-electron density matrix.
        """
        rdm1, rdm2 = self.get_rdms()
        # results
        output = {
            "el_energy": self._es[0] - self._ham.ecore,
            "nuc_nuc_energy": self._ham.ecore,
            "dm1": (rdm1,),
            "dm2": (rdm2,),
        }
        return output


class FullCIRDMs(object):
    """ Solve FCI problem and compute the 1 and 2-reduced density matrices with PySCF.

    Parameters
    ----------
    nelec : tuple (int, int)
        Number of alpha and beta electrons.
    h : (n,n) numpy array
        One lectron integrals in the molecular orbital basis.
    v : (n,n,n,n) numpy array
        Two lectron integrals in the molecular orbital basis.
    ecore : float
        Nuclear repulsion energy.
    is_phys : bool, default=False
        Format of the two-electron integrals. If not specified it is assumed the integrals are in
        chemist notation (internal PySCF format).
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
        "dm2"
            The tuple of the spin-resolved two-electron density matrix.
    """

    def __init__(self, nelec, h, v, ecore, **kwargs):
        """initialize class from electron integrals in npy format.

        Parameters
        ----------
        nelec : int or tuple (int, int)
            Number of alpha and beta electrons.
        h : (n,n) numpy array
            One lectron integrals in the molecular orbital basis.
        v : (n,n,n,n) numpy array
            Two lectron integrals in the molecular orbital basis.
        ecore : float
            Nuclear repulsion energy.
        is_phys : bool, default=False
            Format of the two-electron integrals. If not specified it is assumed the integrals are in
            chemist notation.
        """
        if "is_phys" not in kwargs.keys():
            is_phys = False
        else:
            is_phys = kwargs["is_phys"]
        if is_phys:
            # Transform physisist notation <ij|kl> to chemist notation (ik|jl) used by PySCF
            v = np.einsum("ijkl->ikjl", v)
        if "maxcicle" not in kwargs.keys():
            self._maxcicle = 100
        else:
            self._maxcicle = kwargs["maxcicle"]
        if "tol" not in kwargs.keys():
            self._tol = 1.0e-8
        else:
            self._tol = kwargs["tol"]
        self._h = h
        self._v = v
        self._norb = h.shape[0]
        self._nelec = nelec
        self._ecore = ecore
        #
        # Run FCI
        # Create a FCI (=FCISolver) object because FCI object offers more options to control the
        # calculation.
        #
        self._cisolver = fci.direct_spin1.FCI()
        self._cisolver.max_cycle = self._maxcicle
        self._cisolver.conv_tol = self._tol
        self._es, self._cs = self._cisolver.kernel(
            self._h, self._v, self._norb, self._nelec, ecore=self._ecore
        )

    @classmethod
    def from_fcidump(cls, fname, **kwargs):
        """initialize class from a FCIDUMP file.

        Parameters
        ----------
        fname : str
            Path to FCIDUMP file.

        Returns
        -------
        An instance of the class FullCIRDMs.
        """
        # Load integrlas from FCIDUMP as keys to a dict
        data = tools.fcidump.read(fname)
        # Two-electron integrals are loaded in PySCF internal format (chemist notation).
        if "is_phys" in kwargs.keys():
            kwargs["is_phys"] = False
        return cls(data["NELEC"], data["H1"], data["H2"], data["ECORE"], **kwargs)

    def get_rdms(self):
        """Compute spin-resolved 1- and 2-electron density matrices."""
        # FIXME: Only restricted HF is suported for int number of electrons
        if isinstance(self._nelec, int):
            na, nb = self._nelec // 2, self._nelec // 2
        elif isinstance(self._nelec, tuple):
            na, nb = self._nelec
        # Compute spin-resolved 1- and 2-electron density matrices.
        # For 2-electron density matrix:
        # dm2aa corresponds to alpha spin for both 1st electron and 2nd electron
        # dm2ab corresponds to alpha spin for 1st electron and beta spin for 2nd electron
        # dm2bb corresponds to beta spin for both 1st electron and 2nd electron
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = self._cisolver.make_rdm12s(
            self._cs, self._norb, (na, nb)
        )
        # Swap 2-RDM indices due to PySCF convention
        # 1pdm[p,q] = < |q^{\dagger} p| > ==> dm1[q,p]
        # 2pdm[p,q,r,s] = < |p^{\dagger} r^{\dagger} s q| > ==> dm2[p,r,q,s]
        dm2aa = np.einsum("ijkl->ikjl", dm2aa)
        dm2ab = np.einsum("ijkl->ikjl", dm2ab)
        dm2bb = np.einsum("ijkl->ikjl", dm2bb)

        # Build generalized RDMs:
        rdm1 = np.zeros((2 * self._norb, 2 * self._norb))
        rdm1[: self._norb, : self._norb] = dm1a
        rdm1[self._norb :, self._norb :] = dm1b
        rdm2 = np.zeros((2 * self._norb,) * 4)
        # alpha alpha alpha alpha
        rdm2[: self._norb, : self._norb, : self._norb, : self._norb] = dm2aa
        # beta beta beta beta
        rdm2[self._norb :, self._norb :, self._norb :, self._norb :] = dm2bb
        # alpha beta alpha beta
        rdm2[: self._norb, self._norb :, : self._norb, self._norb :] = dm2ab
        # beta alpha beta alpha
        rdm2[self._norb :, : self._norb, self._norb :, : self._norb] = dm2ab.transpose((1, 0, 3, 2))
        # alpha beta beta alpha
        rdm2[: self._norb, self._norb :, self._norb :, : self._norb] = -dm2ab.transpose(
            (0, 1, 3, 2)
        )
        # beta alpha alpha beta
        rdm2[self._norb :, : self._norb, : self._norb, self._norb :] = -dm2ab.transpose(
            (1, 0, 2, 3)
        )
        return rdm1, rdm2

    def get_outputs(self):
        """Retrieve energy and DMs from CI calculation.

        Returns
        -------
        result : dict
            "el_energy"
                FCI electronic energy.
            "nuc_nuc_energy"
                The nuclear repulsion energy.
            "dm1"
                The spin-resolved 1-electron density matrix.
            "dm2"
                The spin-resolved two-electron density matrix.
        """
        rdm1, rdm2 = self.get_rdms()
        # results
        output = {
            "el_energy": self._es - self._ecore,
            "nuc_nuc_energy": self._ecore,
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
    # Run RHF
    data = hartreefock_pyscf(mol)
    one_int = data["one_int"][0]
    two_int = data["two_int"][0]
    #
    # Running PySCF
    #
    fcidata = FullCIRDMs(
        mol.nelec, one_int, two_int, data["nuc_nuc_energy"], is_phys=True
    ).get_outputs()
    print(fcidata["el_energy"])
    #
    # Running PyCI
    #
    # FIXME: running PyCI from electron integrals arrays doesn't work
    fcidata = PyCIRDMs.from_integrals(
        mol.nelec, one_int, two_int, data["nuc_nuc_energy"], "fci", is_phys=True, n=1, ncv=1
    ).get_outputs()
    print(fcidata["el_energy"])
