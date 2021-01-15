"""Test eomee.ionization."""

import os
import glob
import eomee
from eomee.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import numpy as np
import csv


def check_inputs_symm(oneint, twoint, onedm, twodm):
    """Check symmetry of electron integrals and Density Matrices."""
    # Electron integrals and DMs symmetric permutations
    assert np.allclose(oneint, oneint.T)
    assert np.allclose(onedm, onedm.T)
    assert np.allclose(twoint, np.einsum("pqrs->rspq", twoint))
    assert np.allclose(twoint, np.einsum("pqrs->qpsr", twoint))
    assert np.allclose(twodm, np.einsum("pqrs->rspq", twodm))
    assert np.allclose(twodm, np.einsum("pqrs->qpsr", twodm))
    # Two-electron integrals  and 2DM antisymmetric permutations
    assert np.allclose(twoint, -np.einsum("pqrs->pqsr", twoint))
    assert np.allclose(twoint, -np.einsum("pqrs->qprs", twoint))
    assert np.allclose(twodm, -np.einsum("pqrs->pqsr", twodm))
    assert np.allclose(twodm, -np.einsum("pqrs->qprs", twodm))


def test_excitationeom_erpa_heh_sto3g():
    """Test Excitation ERPA for HeH+ (STO-3G)

    """
    # H2O
    nbasis = 7
    data = np.load("gaussian_fchk_energy.npy")
    # nuc_nuc = data[2]
    orb_sum = 2 * (
        -2.02628914e01
        - 1.20969737e00
        - 5.47964664e-01
        - 4.36527222e-01
        - 3.87586740e-01
    )
    print("Occ MOs summ", "\n", orb_sum)

    # nbasis = 4
    nuc_nuc = 0
    one_mo = np.load(find_datafiles("h2o_crawford_sto-3g_oneint.npy"))
    # one_mo = np.load(find_datafiles("h2_3-21g_oneint.npy"))
    # print(one_mo.shape)
    two_mo = np.load(find_datafiles("h2o_crawford_sto-3g_twoint.npy"))
    # two_mo = antisymmetrize(two_mo)
    one_dm, two_dm = hartreefock_rdms(nbasis, 5, 5)
    # check_inputs_symm(one_mo, two_mo, one_dm, two_dm)

    # Build Fock operator
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    # two_mo = symmetrize(spinize(two_mo))
    Fk = np.copy(one_mo)
    Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
    # Fk += np.einsum("piqi->pq", two_mo)
    # Fk -= np.einsum("piiq->pq", two_mo)
    # w, v = eig(Fk)
    # print(w)

    # Energy Fock operator
    # one_mo = spinize(one_mo)
    # two_mo = symmetrize(spinize(two_mo))
    print(
        "reference E_fock",
        "\n",
        (np.einsum("ij,ij", one_mo, one_dm) + np.einsum("ijkl,ijkl", two_mo, two_dm))
        + nuc_nuc,
    )

    # energy
    Fk_energy = np.einsum("pq, pq", Fk, one_dm)
    # Fk_energy = 2 * (Fk[0, 0] + Fk[1, 1])

    print()
    print("Diagonal of generated Fock matrix ", "\n", np.diag(Fk))
    print("obtained E_fock", "\n", Fk_energy)
    # print(orb_sum)
    # print(orb_sum - Fk_energy)

    one_mo_0 = Fk
    two_mo_0 = np.zeros_like(two_mo)
    dE = eomee.ExcitationEOM.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
    print("1st order correction", Fk_energy + dE + nuc_nuc)
    print(
        "reference E_fock",
        (
            np.einsum("ij,ij", one_mo, one_dm)
            + 0.5 * np.einsum("ijkl,ijkl", two_mo, two_dm)
        )
        + nuc_nuc,
    )
    # print(ecorr)
    # ecorr = eomee.DoubleElectronRemovalEOM.erpa(
    #     one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
    # )
    # print(ecorr)


def test_ph_rpa_h2_scuseria():
    """Test Excitation RPA for H2 (cc-pVDZ)
    RHF reference wfn.

    """
    files = glob.glob("scuseria/h2_sc_*")
    energies_1 = []
    for i, _ in enumerate(files, start=1):
        word = "h2_sc_{0}".format(i)
        print("Molecule, ", word)
        path = "scuseria/" + word + "/hartreefock_pyscf_energy.npy"
        data = np.load(path, allow_pickle=True, encoding="bytes",)
        nuc_nuc = data[2]

        one_mo = np.load("scuseria/{0}/{0}_cc-pvdz_oneint.npy".format(word))
        two_mo = np.load("scuseria/{0}/{0}_cc-pvdz_twoint.npy".format(word))
        # # # two_mo = antisymmetrize(two_mo)
        nbasis = one_mo.shape[0]
        one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

        # Build Fock operator
        one_mo = spinize(one_mo)
        two_mo = symmetrize(spinize(two_mo))
        Fk = np.copy(one_mo)
        Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
        # energy
        Fk_energy = np.einsum("pq, pq", Fk, one_dm)
        # print(
        #     "reference E_fock",
        #     (
        #         np.einsum("ij,ij", one_mo, one_dm)
        #         + 0.5 * np.einsum("ijkl,ijkl", two_mo, two_dm)
        #     )
        #     + nuc_nuc,
        # )

        one_mo_0 = Fk
        two_mo_0 = np.zeros_like(two_mo)
        dE = eomee.ExcitationEOM.erpa(
            one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
        )
        print("1st order correction", Fk_energy + dE + nuc_nuc)
        energy_1 = Fk_energy + dE + nuc_nuc
        energies_1.append(energy_1)
        # # # print(ecorr)
        # # # ecorr = eomee.DoubleElectronRemovalEOM.erpa(
        # # #     one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
        # # # )
        # # # print(ecorr)
    print("DONE")

    output = "energy1_h2_scuseria_nonlinearterm.csv"

    with open(output, "w") as output_file:
        energy_data = csv.writer(output_file, dialect="excel")
        fieldnames = [
            "Molecules",
            "HH bohr",
            "Energy_linear",
        ]
        energy_data.writerow(fieldnames)

        for i, _ in enumerate(files):
            molname = "h2_sc_{0}".format(i + 1)
            bond = "{:.1f}".format(1.0 + (i) * 0.1)
            energy_data.writerow((molname, bond, energies_1[i]))
    print("Finito")


# test_excitationeom_erpa_heh_sto3g()
test_ph_rpa_h2_scuseria()

