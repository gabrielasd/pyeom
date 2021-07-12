# This file is part of EOMEE.
#
# EOMEE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# EOMEE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

r"""Test eomee.ionization."""


from eomee import EOMExc

from eomee.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
    pickpositiveeig,
    pickeig
)

import numpy as np

import pytest


def get_tdm(cv, dm1, dm2, commutator=True):
    """Compute the transition RDMs. Two options are possible: to use the commutator of
    the excitation operators (commutator=True) or not.

    Parameters
    ----------
    cv : np.ndarray((n**2, n**2))
        Eigenvector matrix.
    dm1 : np.ndarray((n, n))
        Spin resolved 1-particle reduced density matrix.
    dm2 : np.ndarray((n, n, n, n))
        Spin resolved 2-particle reduced density matrix.
    commutator : bool, optional
        Form used to approximate the transition-RDMs, one of commutator (True) or no commutator
        (False), by default True

    Returns
    -------
    np.ndarray((n**2, n, n))
        1-particle transition density matrix.
    """
    nspins = dm1.shape[0]

    # Compute transition RDMs
    if commutator:
        # gamma_kl;n = \sum_ij < |[k^+ l, j^+ i]| > c_ij;n
        # c_ij;n (\delta_lj \gamma_ki - \delta_ik \gamma_jl)
        rdm_terms = np.einsum("lj,ki->klij", np.eye(nspins), dm1, optimize=True)
        rdm_terms -= np.einsum("ki,jl->klij", np.eye(nspins), dm1, optimize=True)
    else:
        # gamma_kl;n = \sum_ij < |k^+ l j^+ i| > c_ij;n
        # c_ij;n (\delta_lj \gamma_ki - \Gamma_kjil)
        rdm_terms = np.einsum("ki,lj->klij", dm1, np.eye(nspins), optimize=True)
        rdm_terms -= np.einsum("kjil->klij", dm2, optimize=True)

    cv = cv.reshape(cv.shape[0], nspins, nspins)
    tdms = np.einsum("mrs,pqrs->mpq", cv, rdm_terms)
    return tdms


def get_dm2_from_tdms(cv, one_dm, two_dm, comm=True):
    """Reconstruct the 2-RDM in terms of the 1-RDM for the ground state
    and the transition-RDMs from particle-hole eRPA.

    Parameters
    ----------
    cv : np.ndarray((n**2, n**2))
        Eigenvector matrix.
    one_dm : np.ndarray((n, n))
        Spin resolved 1-particle reduced density matrix.
    two_dm : np.ndarray((n, n, n, n))
        Spin resolved 2-particle reduced density matrix.
    comm : bool, optional
        Form used to approximate the transition-RDMs, one of commutator (True) or no commutator
        (False), by default True

    Returns
    -------
    np.ndarray((n, n, n, n))
        Reconstructed 2-RDM.
    """
    # Gamma_pqrs = < | p^+ q^+ s r | >
    #            = - < | p^+ q^+ r s | >
    #            = - \delta_qr * \gamma_ps
    #            + \gamma_pr * \gamma_qs
    #            + \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
    n = one_dm.shape[0]
    # \gamma_pr * \gamma_qs - \delta_qr * \gamma_ps
    dm2 = np.einsum("pr,qs->pqrs", one_dm, one_dm, optimize=True)
    dm2 -= np.einsum("qr,ps->pqrs", np.eye(n), one_dm, optimize=True)

    # Compute term involvin the tdms
    # \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
    tdms = get_tdm(cv, one_dm, two_dm, commutator=comm)

    tv = np.zeros_like(two_dm)
    for rdm in tdms:
        tv += np.einsum("pr,qs->pqrs", rdm, rdm, optimize=True)
    return dm2 + tv


def test_eomexc_neigs():
    """Check number of eigenvalues.

    """
    one_mo = np.arange(16, dtype=float).reshape(4, 4)
    two_mo = np.arange(16 * 16, dtype=float).reshape(4, 4, 4, 4)
    one_dm = np.zeros((4, 4), dtype=float)
    one_dm[0, 0], one_dm[2, 2] = 1.0, 1.0
    two_dm = np.einsum("pr,qs->pqrs", one_dm, one_dm)
    two_dm -= np.einsum("ps,qr->pqrs", one_dm, one_dm)

    eom = EOMExc(one_mo, two_mo, one_dm, two_dm)
    assert eom.neigs == 4 ** 2


@pytest.mark.parametrize(
    "filename, nparts, answer",
    [
    ("be_sto3g", (2, 2), 4.13573698),
    ("h2_hf_sto6g", (1, 1), 0.92954096),
    ("h2_3-21g", (1, 1), 0.56873365),
    ],
)
def test_eomexc(filename, nparts, answer):
    """Test lowest singlet excited state evaluated with ExcitationEOM for a RHF reference. The
    commutator form is used on the right-hand-side so this is equivalent to ph-RPA/TD-HF. PySCF's
    TDHF results are used as reference.

    cases 2 and 3: H-H bond 0.742 A

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)
    # Evaluate particle-hole EOM
    phrpa = EOMExc(spinize(one_mo), spinize(two_mo), one_dm, two_dm)
    ev, _ = phrpa.solve_dense(orthog="asymmetric")
    result = pickeig(ev)

    assert np.allclose(result[1], answer)


@pytest.mark.parametrize(
    "filename, nparts, ehf",
    [
    ("be_sto3g", (2, 2), -14.351880476202),
    ("h2_3-21g", (1, 1), -1.836099198341838),
    ("ne_321g", (5, 5), -127.803824528206),
    ("he_ccpvdz", (1, 1), -2.85516047724274),
    ],
)
def test_reconstructed_2rdm_phrpa(filename, nparts, ehf):
    """Evaluate ph-EOM (with commutator on right-hand-side).
    Approximate the transition RDM (using the commutator
    of the excitation operator) and reconstruct the 2-RDM.

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Evaluate particle-hole EOM.
    # Reconstruct 2RDM from T-RDM and check it adds to the right number
    # of electron pairs (normalization condition).
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    phrpa = EOMExc(one_mo, two_mo, one_dm, two_dm)
    w, cv = phrpa.solve_dense(orthog="asymmetric")
    _, pcv, _ = pickpositiveeig(w, cv)
    rdm2 = get_dm2_from_tdms(pcv, one_dm, two_dm, comm=True)
    assert np.allclose(np.einsum("ijij", rdm2), ((na + nb) * ((na + nb) - 1)))

    # Energy from HF RDMs
    energy1 = np.einsum("ij,ij", one_mo, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo, two_dm
    )
    assert np.allclose(energy1, ehf)
    # Energy from reconstructed RDMs
    energy2 = np.einsum("ij,ij", one_mo, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo, rdm2
    )
    print("E_HF", ehf, "E_2rdm", energy2)


# def test_excitationeom_erpa_heh_sto3g():
#     """Test Excitation ERPA for HeH+ (STO-3G)"""
#     nbasis = 2
#     one_mo = np.load(find_datafiles("heh+_sto3g_oneint.npy"))
#     one_mo = spinize(one_mo)
#     two_mo = np.load(find_datafiles("heh+_sto3g_twoint.npy"))
#     two_mo = symmetrize(spinize(two_mo))
#     two_mo = antisymmetrize(two_mo)
#     one_dm, two_dm = hartreefock_rdms(nbasis, 1, 1)

#     n = one_mo.shape[0]
#     aa = one_mo[:1, :1]
#     bb = one_mo[n // 2 : (n // 2 + 1), n // 2 : (n // 2 + 1)]
#     aaaa = two_mo[:1, :1, :1, :1]
#     abab = two_mo[:1, n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1)]
#     baba = two_mo[n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1), :1]
#     bbbb = two_mo[
#         n // 2 : (n // 2 + 1),
#         n // 2 : (n // 2 + 1),
#         n // 2 : (n // 2 + 1),
#         n // 2 : (n // 2 + 1),
#     ]
#     one_mo_0 = np.zeros_like(one_mo)
#     two_mo_0 = np.zeros_like(two_mo)
#     one_mo_0[:1, :1] = aa
#     one_mo_0[n // 2 : (n // 2 + 1), n // 2 : (n // 2 + 1)] = bb
#     two_mo_0[:1, :1, :1, :1] = aaaa
#     two_mo_0[:1, n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1)] = abab
#     two_mo_0[n // 2 : (n // 2 + 1), :1, n // 2 : (n // 2 + 1), :1] = baba
#     two_mo_0[
#         n // 2 : (n // 2 + 1),
#         n // 2 : (n // 2 + 1),
#         n // 2 : (n // 2 + 1),
#         n // 2 : (n // 2 + 1),
#     ] = bbbb

#     ecorr = EOMExc.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
#     print(ecorr)


@pytest.mark.parametrize(
    "filename, nparts, ehf",
    [
    ("be_sto3g", (2, 2), -14.351880476202),
    ("h2_3-21g", (1, 1), -1.836099198341838),
    ("ne_321g", (5, 5), -127.803824528206),
    ],
)
def test_phrpa_adiabaticconection(filename, nparts, ehf):
    """Test ground state energy correction through eRPA.

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Build Fock operator
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    Fk = np.copy(one_mo)
    Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)

    # Evaluate ERPA
    one_mo_0 = Fk
    two_mo_0 = np.zeros_like(two_mo)
    dE = EOMExc.erpa(
        one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm, orthog="asymmetric"
    )
    print("E_HF", ehf)
    print("E_erpa", np.einsum("pq, pq", Fk, one_dm) + dE)
