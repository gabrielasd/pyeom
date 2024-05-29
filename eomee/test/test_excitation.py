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


from eomee.excitation import EOMExc, eval_ecorr, _eval_W_alpha_constant_terms

from eomee.tools import (
    find_datafiles,
    spinize,
    antisymmetrize,
    hartreefock_rdms,
    make_gvbpp_hamiltonian,
    from_unrestricted,
)

from eomee.solver import pick_positive, _pick_singlets, _pick_multiplets, _pickeig

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
    ev, _ = phrpa.solve_dense(mode="nonsymm")
    idx = 3 # 1st singlet
    assert np.allclose(ev[idx], answer)


def test_eomexc_gvb_h2_631g():
    filename = 'h2_0.70_gvbpp_631g'
    ham = np.load(find_datafiles("{}.ham.npz".format(filename)))
    dms = np.load(find_datafiles("{}.dms.npz".format(filename)))
    gem_m = np.load(find_datafiles("{}.geminals.npy".format(filename)))
    one_mo = ham['onemo']
    two_mo = ham["twomo"]
    dm1_a = dms['dm1'][0]
    dm2_aa, dm2_ab = dms['dm2']
    
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_m, dm1_a)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo)
    v1 = spinize(two_mo)
    rdm1 = from_unrestricted([dm1_a, dm1_a])
    rdm2 = from_unrestricted([dm2_aa, dm2_ab, dm2_aa])
    n = one_mo.shape[0]
    rdm2[:n, n:, n:, :n] = -dm2_ab.transpose((0,1,3,2))
    rdm2[n:, :n, :n, n:] = -dm2_ab.transpose((1,0,2,3))

    # Evaluate particle-hole EOM
    singlets = [1.0297]
    triplets = [0.6729, 0.6729, 0.6729, 1.3819, 1.3819, 1.3819, 1.3819, 1.6211, 1.6211,
    1.6211, 1.6211, 2.2159, 2.2159, 2.2159, 2.2159, 2.4551, 2.4551, 2.4551, 2.4551]
    erpa = EOMExc(h0, v0, rdm1, rdm2)
    ev, cv = erpa.solve_dense(mode="nonsymm")
    # ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    ev_p, cv_p = ev, cv
    singlets_ev = _pick_singlets(ev_p, cv_p)[0]
    triplets_ev = _pick_multiplets(ev_p, cv_p)[0]

    assert np.allclose(singlets, singlets_ev, atol=1e-4)
    assert np.allclose(triplets, triplets_ev, atol=1e-4)

    # Evaluate particle-hole EOM
    singlets = [0.5879, 1.0384, 1.395, 1.8613, 2.1643]
    triplets = [0.4182, 0.4182, 0.4182, 0.8364, 0.8364, 0.8364, 1.3622, 1.3622,
    1.3622, 1.467,  1.467,  1.467, 1.9592, 1.9592, 1.9592]
    erpa = EOMExc(h1, v1, rdm1, rdm2)
    ev, cv = erpa.solve_dense(mode="nonsymm")
    # ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    ev_p, cv_p = ev, cv
    singlets_ev = _pick_singlets(ev_p, cv_p)[0]
    triplets_ev = _pick_multiplets(ev_p, cv_p)[0]

    assert np.allclose(singlets, singlets_ev, atol=1e-4)
    assert np.allclose(triplets, triplets_ev, atol=1e-4)


def test_eomexc_gvb_h2o_631g():
    filename = 'h2o_1.00_gvbpp_631g'
    ham = np.load(find_datafiles("{}.ham.npz".format(filename)))
    dms = np.load(find_datafiles("{}.dms.npz".format(filename)))
    one_mo = ham['onemo']
    two_mo = ham["twomo"]
    dm1_a = dms['rdm1'][0]
    dm2_aa, dm2_ab = dms['rdm2']
    h1 = spinize(one_mo)
    v1 = spinize(two_mo)
    rdm1 = from_unrestricted([dm1_a, dm1_a])
    rdm2 = from_unrestricted([dm2_aa, dm2_ab, dm2_aa])
    n = one_mo.shape[0]
    rdm2[:n, n:, n:, :n] = -dm2_ab.transpose((0,1,3,2))
    rdm2[n:, :n, :n, n:] = -dm2_ab.transpose((1,0,2,3))

    # Evaluate particle-hole EOM
    singlets = [0.33050385, 0.39671774, 0.43111097]
    triplets = [0.30006822, 0.36716561, 0.38073963]
    erpa = EOMExc(h1, v1, rdm1, rdm2)
    ev, cv = erpa.solve_dense(mode="nonsymm")
    # ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    ev_p, cv_p = ev, cv
    singlets_ev = _pick_singlets(ev_p, cv_p)[0]
    singlets_ev = _pickeig(singlets_ev, tol=0.001)[:3]
    triplets_ev = _pick_multiplets(ev_p, cv_p)[0]
    triplets_ev = _pickeig(triplets_ev, tol=0.001)[:3]

    assert np.allclose(singlets, singlets_ev, atol=1e-2)
    assert np.allclose(triplets, triplets_ev, atol=1e-2)


def test_eomexc_gvb_h2_631g():
    filename = 'h2_0.70_gvbpp_631g'
    ham = np.load(find_datafiles("{}.ham.npz".format(filename)))
    dms = np.load(find_datafiles("{}.dms.npz".format(filename)))
    gem_m = np.load(find_datafiles("{}.geminals.npy".format(filename)))
    one_mo = ham['onemo']
    two_mo = ham["twomo"]
    dm1_a = dms['dm1'][0]
    dm2_aa, dm2_ab = dms['dm2']
    
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_m, dm1_a)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo)
    v1 = spinize(two_mo)
    rdm1 = from_unrestricted([dm1_a, dm1_a])
    rdm2 = from_unrestricted([dm2_aa, dm2_ab, dm2_aa])
    n = one_mo.shape[0]
    rdm2[:n, n:, n:, :n] = -dm2_ab.transpose((0,1,3,2))
    rdm2[n:, :n, :n, n:] = -dm2_ab.transpose((1,0,2,3))

    # Evaluate particle-hole EOM
    singlets = [1.0297]
    triplets = [0.6729, 0.6729, 0.6729, 1.3819, 1.3819, 1.3819, 1.3819, 1.6211, 1.6211,
    1.6211, 1.6211, 2.2159, 2.2159, 2.2159, 2.2159, 2.4551, 2.4551, 2.4551, 2.4551]
    erpa = EOMExc(h0, v0, rdm1, rdm2)
    ev, cv = erpa.solve_dense(orthog="nonsymm")
    # ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    singlets_ev = _pick_singlets(ev, cv)[0]
    triplets_ev = _pick_multiplets(ev, cv)[0]

    assert np.allclose(singlets, singlets_ev, atol=1e-4)
    assert np.allclose(triplets, triplets_ev, atol=1e-4)

    # Evaluate particle-hole EOM
    singlets = [0.5879, 1.0384, 1.395, 1.8613, 2.1643]
    triplets = [0.4182, 0.4182, 0.4182, 0.8364, 0.8364, 0.8364, 1.3622, 1.3622,
    1.3622, 1.467,  1.467,  1.467, 1.9592, 1.9592, 1.9592]
    erpa = EOMExc(h1, v1, rdm1, rdm2)
    ev, cv = erpa.solve_dense(orthog="nonsymm")
    # ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    singlets_ev = _pick_singlets(ev, cv)[0]
    triplets_ev = _pick_multiplets(ev, cv)[0]

    assert np.allclose(singlets, singlets_ev, atol=1e-4)
    assert np.allclose(triplets, triplets_ev, atol=1e-4)


def test_eomexc_gvb_h2o_631g():
    filename = 'h2o_1.00_gvbpp_631g'
    ham = np.load(find_datafiles("{}.ham.npz".format(filename)))
    dms = np.load(find_datafiles("{}.dms.npz".format(filename)))
    one_mo = ham['onemo']
    two_mo = ham["twomo"]
    dm1_a = dms['rdm1'][0]
    dm2_aa, dm2_ab = dms['rdm2']
    h1 = spinize(one_mo)
    v1 = spinize(two_mo)
    rdm1 = from_unrestricted([dm1_a, dm1_a])
    rdm2 = from_unrestricted([dm2_aa, dm2_ab, dm2_aa])
    n = one_mo.shape[0]
    rdm2[:n, n:, n:, :n] = -dm2_ab.transpose((0,1,3,2))
    rdm2[n:, :n, :n, n:] = -dm2_ab.transpose((1,0,2,3))

    # Evaluate particle-hole EOM
    singlets = [0.33050385, 0.39671774, 0.43111097]
    triplets = [0.30006822, 0.36716561, 0.38073963]
    erpa = EOMExc(h1, v1, rdm1, rdm2)
    ev, cv = erpa.solve_dense(orthog="nonsymm")
    # ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    singlets_ev = _pick_singlets(ev, cv)[0]
    singlets_ev = _pickeig(singlets_ev, tol=0.001)[:3]
    triplets_ev = _pick_multiplets(ev, cv)[0]
    triplets_ev = _pickeig(triplets_ev, tol=0.001)[:3]

    assert np.allclose(singlets, singlets_ev, atol=1e-2)
    assert np.allclose(triplets, triplets_ev, atol=1e-2)


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
    w, cv = phrpa.solve_dense(mode="nonsymm")
    # _, pcv, _ = pickpositiveeig(w, cv)
    pcv = cv
    
    # Check that that the reconstructed 2RDM has the right normalization
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
    solution = eval_ecorr(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
    print("E_HF", ehf)
    print("E_erpa", np.einsum("pq, pq", Fk, one_dm) + solution)


def test_ac_gvb_h2_631g():
    """Test adiabatic connection for GVBPP H2 at 0.7 A.

    Numericla result for comparison generated with GAMCOR:
    E_geminal = -1.900367259303
    E_gvb = -1.14439982
    E_gvb_ac = -1.14877553
    alpha_indep_term = -0.151183973855072
    int_W_alpha = 0.142987997959857
    E_corr = -0.00437571

    """
    E_geminal = -1.900367259303
    E_gvb = -1.14439982
    E_gvb_ac = -1.14877553
    alpha_indep_term = -0.151183973855072
    int_W_alpha = 0.142987997959857
    E_corr = -0.00437571

    filename = 'h2_0.70_gvbpp_631g'
    ham = np.load(find_datafiles("{}.ham.npz".format(filename)))
    dms = np.load(find_datafiles("{}.dms.npz".format(filename)))
    gem_m = np.load(find_datafiles("{}.geminals.npy".format(filename)))
    one_mo = ham['onemo']
    two_mo = ham['twomo']
    nuc = ham['nuc']
    dm1_a = dms['dm1'][0]
    dm2_aa, dm2_ab = dms['dm2']
    
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_m, dm1_a)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo)
    v1 = spinize(two_mo)
    rdm1 = from_unrestricted([dm1_a, dm1_a])
    rdm2 = from_unrestricted([dm2_aa, dm2_ab, dm2_aa])
    n = one_mo.shape[0]
    rdm2[:n, n:, n:, :n] = -dm2_ab.transpose((0,1,3,2))
    rdm2[n:, :n, :n, n:] = -dm2_ab.transpose((1,0,2,3))

    # Evaluate particle-hole EOM
    solution = eval_ecorr(h0, v0, h1, v1, rdm1, rdm2)
    E1 = E_gvb + solution

    assert np.allclose(E1, E_gvb_ac, atol=1e-5)
    assert np.allclose(solution, E_corr, atol=1e-5)

    # result = E1_E0 - linear
    # assert np.allclose(result, int_W_alpha, atol=1e-5)

    linear = _eval_W_alpha_constant_terms(v1-v0, rdm1, rdm2, True, 1e-7)
    linear += np.einsum("ij,ij", h1-h0, rdm1) # add one-body term
    linear += 0.5 * np.einsum("pqrs,pqrs", v1-v0, rdm2) # remove term from <GVB|H^0|GVB>
    shift = 0.5 * np.einsum("ij,ij", spinize(two_mo_0_inter), rdm1)
    result = linear - shift
    assert np.allclose(result, alpha_indep_term, atol=1e-3)


def test_ac_gvb_h2o_631g():
    E_geminal = np.sum([-45.896106956249, -2.364978315966, -2.364988478617, -2.418131034466, -2.418028050155])
    E_gvb = -76.04690633
    E_gvb_ac = -76.11455506
    alpha_indep_term = -0.848565549040074
    int_W_alpha = 0.725397540381398
    E_corr = -0.06764874

    filename = 'h2o_1.00_gvbpp_631g'
    ham = np.load(find_datafiles("{}.ham.npz".format(filename)))
    dms = np.load(find_datafiles("{}.dms.npz".format(filename)))
    gem_m = np.load(find_datafiles("{}.geminals.npy".format(filename)))
    one_mo = ham['onemo']
    two_mo = ham['twomo']
    nuc = ham['nuc']
    dm1_a = dms['rdm1'][0]
    dm2_aa, dm2_ab = dms['rdm2']
    
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_m, dm1_a)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo)
    v1 = spinize(two_mo)
    rdm1 = from_unrestricted([dm1_a, dm1_a])
    rdm2 = from_unrestricted([dm2_aa, dm2_ab, dm2_aa])
    n = one_mo.shape[0]
    rdm2[:n, n:, n:, :n] = -dm2_ab.transpose((0,1,3,2))
    rdm2[n:, :n, :n, n:] = -dm2_ab.transpose((1,0,2,3))

    # Evaluate particle-hole EOM
    solution = eval_ecorr(h0, v0, h1, v1, rdm1, rdm2)
    E1 = E_gvb + solution

    assert np.allclose(E1, E_gvb_ac, atol=1e-2)
    assert np.allclose(solution, E_corr, atol=1e-2)

    linear = _eval_W_alpha_constant_terms(v1-v0, rdm1, rdm2, True, 1e-7)
    linear += np.einsum("ij,ij", h1-h0, rdm1) # add one-body term
    linear += 0.5 * np.einsum("pqrs,pqrs", v1-v0, rdm2) # remove term from <GVB|H^0|GVB>
    shift = 0.5 * np.einsum("ij,ij", spinize(two_mo_0_inter), rdm1)
    result = linear + shift
    assert np.allclose(result, alpha_indep_term, atol=1e-2)
