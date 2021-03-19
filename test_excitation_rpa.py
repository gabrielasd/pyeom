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

from scipy.integrate import fixed_quad as integrate

# from scipy.integrate import quad as integrate


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
    # energies_1 = []
    for i, _ in enumerate(files[:3], start=1):
        word = "h2_sc_{0}".format(i)
        print("Molecule, ", word)
        path = "scuseria/" + word + "/hartreefock_pyscf_energy.npy"
        data = np.load(path, allow_pickle=True, encoding="bytes",)
        nuc_nuc = data[2]

        one_mo = np.load("scuseria/{0}/{0}_cc-pvdz_oneint.npy".format(word))
        two_mo = np.load("scuseria/{0}/{0}_cc-pvdz_twoint.npy".format(word))
        # # two_mo = antisymmetrize(two_mo)
        nbasis = one_mo.shape[0]
        one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

        # Build Fock operator
        one_mo = spinize(one_mo)
        two_mo = symmetrize(spinize(two_mo))
        Fk = np.copy(one_mo)
        Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
        # energy
        Fk_energy = np.einsum("pq, pq", Fk, one_dm)
        print(
            "reference E_fock",
            (
                np.einsum("ij,ij", one_mo, one_dm)
                + np.einsum("ijkl,ijkl", two_mo, two_dm)
            )
            + nuc_nuc,
        )

        one_mo_0 = Fk
        two_mo_0 = np.zeros_like(two_mo)
        # dE = eomee.ExcitationEOM.erpa(
        #     one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
        # )
        dE = eomee.DoubleElectronAttachmentEOM.erpa(
            one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
        )
        print("1st order correction", Fk_energy + dE + nuc_nuc)
        # energy_1 = Fk_energy + dE + nuc_nuc
        # energies_1.append(energy_1)
    print("DONE")

    # output = "energy1_h2_scuseria_nonlinearterm.csv"

    # with open(output, "w") as output_file:
    #     energy_data = csv.writer(output_file, dialect="excel")
    #     fieldnames = [
    #         "Molecules",
    #         "HH bohr",
    #         "Energy_linear",
    #     ]
    #     energy_data.writerow(fieldnames)

    #     for i, _ in enumerate(files):
    #         molname = "h2_sc_{0}".format(i + 1)
    #         bond = "{:.1f}".format(1.0 + (i) * 0.1)
    #         energy_data.writerow((molname, bond, energies_1[i]))
    # print("Finito")


def test_ph_rpa_smallmol():
    """Test Excitation RPA for H2 (sto-3g)
    RHF reference wfn.

    """
    # energies_1 = []

    one_mo = np.load(find_datafiles("be_sto3g_oneint.npy"))
    two_mo = np.load(find_datafiles("be_sto3g_twoint.npy"))
    # one_mo = np.load(("smallmol/Be_0_1_3-21g_oneint.npy"))
    # two_mo = np.load(("smallmol/Be_0_1_3-21g_twoint.npy"))
    # one_mo = np.load(("vanaggelen/be_ccpvdz_oneint.npy"))
    # two_mo = np.load(("vanaggelen/be_ccpvdz_twoint.npy"))
    # ehf = -14.4868202421757  # -14.5723376309534
    # energyfile = "hartreefock_pyscf_energy.npy"
    # path = "smallmol/" + energyfile
    # data = np.load(path, allow_pickle=True, encoding="bytes",)
    nuc_nuc = 0  # data[2]
    # ehf = data[1]  # + nuc_nuc
    # print(ehf)

    # # two_mo = antisymmetrize(two_mo)
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, 2, 2)
    # print(nbasis)

    # # Evaluate particle-hole EOM
    # phrpa = eomee.ExcitationEOM(
    #     spinize(one_mo), antisymmetrize(spinize(two_mo)), one_dm, two_dm
    # )

    # # print(phrpa.rhs)
    # EPH, CPH = phrpa.solve_dense(orthog="asymmetric")
    # # print("E(phRPA) = ", np.amax(EPH))
    # # print(sorted(EPH))
    # # print()
    # lowest_e = np.sort(EPH[EPH > 0])
    # print(lowest_e)
    # # order = np.argsort(EPH)
    # # vals, vecs = EPH[order], CPH[order]
    # # print(vals)

    # Build Fock operator
    one_mo = spinize(one_mo)
    two_mo = symmetrize(spinize(two_mo))
    Fk = np.copy(one_mo)
    # Szabo: <i|h|j> + \sum_b <ib||jb>
    # Fk += np.einsum("piqi->pq", antisymmetrize(two_mo))
    #
    Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
    # e = np.linalg.eig(Fk)[0]
    # print(e)

    # Energies
    Fk_energy = np.einsum("pq, pq", Fk, one_dm)
    print(Fk_energy)
    print(
        "reference E_fock",
        (np.einsum("ij,ij", one_mo, one_dm) + np.einsum("ijkl,ijkl", two_mo, two_dm))
        + nuc_nuc,
    )
    ehf = (
        np.einsum("ij,ij", one_mo, one_dm)
        + 0.5 * np.einsum("ijkl,ijkl", two_mo, two_dm)
    ) + nuc_nuc
    print(ehf)

    one_mo_0 = Fk
    two_mo_0 = np.zeros_like(two_mo)
    dE = eomee.ExcitationEOM.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
    # dE = eomee.DoubleElectronAttachmentEOM.erpa(
    #     one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm
    # )
    # print("1st order correction", Fk_energy + dE + nuc_nuc)

    # E_1 - E_HF
    print("Ecorr", Fk_energy + dE - ehf)
    # print("Nonlinear term", dE)

    # # e_chf = eq77_pernal2018(one_mo, one_mo_0, two_mo, two_mo_0, one_dm, two_dm)
    # # print("Ecorr pernal", e_chf)
    # # print("DONE")


def test_ph_rpa_ccpvdz_vanaggelen():
    """Test Excitation RPA for H2 (sto-3g)
    RHF reference wfn.

    """
    one_mo = np.load(("vanaggelen/be_ccpvdz_oneint.npy"))
    two_mo = np.load(("vanaggelen/be_ccpvdz_twoint.npy"))
    # one_dm = np.load(("vanaggelen/be_ccpvdz_fci_onedm.npy"))
    # two_dm = np.load(("vanaggelen/be_ccpvdz_fci_twodm.npy"))
    # HF DMs
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, 2, 2)

    # Evaluate particle-hole EOM
    phrpa = eomee.ExcitationEOM(
        spinize(one_mo), antisymmetrize(spinize(two_mo)), one_dm, two_dm
    )

    # print(phrpa.rhs)
    EPH, CPH = phrpa.solve_dense()
    # print("E(phRPA) = ", np.amax(EPH))
    print(sorted(EPH))


def test_ph_rpa_chatterjee():
    """Test Excitation RPA for H2 (sto-3g)
    RHF reference wfn.

    """
    one_mo = np.load(("chatterjee/h2_aug-cc-pvtz_oneint.npy"))
    two_mo = np.load(("chatterjee/h2_aug-cc-pvtz_twoint.npy"))
    # one_mo = np.load(("chatterjee/lih_cc-pvtz_oneint.npy"))
    # two_mo = np.load(("chatterjee/lih_cc-pvtz_twoint.npy"))
    # one_mo = np.load(("chatterjee/Be_aug-cc-pvtz_oneint.npy"))
    # two_mo = np.load(("chatterjee/Be_aug-cc-pvtz_twoint.npy"))
    # HF DMs
    nbasis = one_mo.shape[0]
    one_dm, two_dm = hartreefock_rdms(nbasis, 2, 2)
    # print(nbasis)

    # Evaluate particle-hole EOM
    phrpa = eomee.ExcitationEOM(
        spinize(one_mo), antisymmetrize(spinize(two_mo)), one_dm, two_dm
    )

    # print(phrpa.rhs)
    EPH, CPH = phrpa.solve_dense()
    # print("E(phRPA) = ", np.amax(EPH))
    print(sorted(EPH))


def eq77_pernal2018(h_1, h_0, v_1, v_0, dm1, dm2):
    # FIXME: This isn't right either, returns positive
    # ecorr values: 0.3353199240867684

    # Size of dimensions
    n = h_0.shape[0]
    # H_1 - H_0
    dh = h_1 - h_0
    # V_1 - V_0
    dv = v_1 - v_0

    dm1_eye = np.einsum(
        "qr,ps->pqrs", np.eye(n), dm1, optimize=True
    )  # in our erpa "sq,pr->pqrs"
    linear = dm1_eye - np.einsum("ps,qr->pqrs", dm1, dm1, optimize=True)
    linear = 0.5 * np.einsum("pqrs,pqrs", linear, v_1, optimize=True)

    @np.vectorize
    def integrand(alpha):
        # Compute H^alpha
        h = alpha * dh
        h += h_0
        v = alpha * dv
        v += v_0
        # Antysymmetrize v_pqrs
        v = antisymmetrize(v)

        phrpa = eomee.ExcitationEOM(h, v, dm1, dm2)
        _, coeffs = phrpa.solve_dense()

        coeffs = coeffs.reshape(n ** 2, n, n)
        # Compute transition RDMs
        tdms = np.einsum("nij,pqij->npq", coeffs, phrpa._rhs.reshape(n, n, n, n))
        # Compute nonlinear energy term
        TT = np.zeros_like(dm2)
        for tv in tdms:
            TT += np.einsum("pr,qs->pqrs", tv, tv)  # in our erpa "ps,qr->pqrs"
        TT -= np.einsum("pr,qs->pqrs", tdms[0], tdms[0])
        return np.einsum("pqrs,pqrs", TT, v_1)

    # return -linear
    return 0.5 * integrate(integrand, 0, 1, n=50)[0] - linear
    # return integrate(integrand, 0, 1, limit=50, epsabs=1.49e-04, epsrel=1.49e-04)[0]


# test_excitationeom_erpa_heh_sto3g()
# test_ph_rpa_h2_scuseria()
test_ph_rpa_smallmol()
# test_ph_rpa_ccpvdz_vanaggelen()
# test_ph_rpa_chatterjee()


# Comments:
# =========
# Based on results for H2 STO-6G, taking HF as the reference at
# alpha = 0 and TDMs approximated from ph-RPA.
# Ecorr from our implemented adiabatic connection formula, Ecorr(phrpa),
# vs the results using Equation (7) from Tahir2019 implemented in
# the dity_tdhf.py script, Ecorr(phrpa)_tahir:
# Ecorr(phrpa) = -0.5018403045485365
# Ecorr(phrpa)_tahir =  -0.026114856279256027
# We greatly overestimate the correlation correction.
# The attemp at comparing vs an inpmlentation of Equation (82) from
# KPernal2018 (also in script dity_tdhf.py) hasn't gone well:
# Ecorr(phrpa)_pernal =  0.16366164714633863
# The implementation needs verification of several terms involving the
# two-electron integrals.
# Comparing only the nonlinear terms from our equation to the one from
# the current implementation of Pernal isn't usefull either:
# Our integrated nonlinear term = 1.0036806090970734
# Pernal's = 0.16366164714633863
# DONE:
# Try equation (77) from KPernal2018, which is writen in terms of the TDMs
# so I can use it with our ph-RPA implementation. Compare the result with the
# one I'm getting with eomee.ExcitationEOM.erpa, and the one from the
# the implementation of equation (82).
# Evaluating Ecorr I get:
# Ecorr 0.9909121973199415 (nuestro codigo)
# Ecorr pernal 0.3353199240867684 (Eq. 77 )
# TODO
# Evalua manualmente los terminos en la funcion eomee.ExcitationEOM.erpa
# para alpha=0 y alpha=1
# Comprueba el convenio de indices usado en eomee.ExcitationEOM.erpa respecto
# al de ExcitationEOM

