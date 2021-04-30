import eomee
from eomee.tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
    from_unrestricted,
)

import numpy as np
from scipy.linalg import svd


def get_tdm(cv, dm1, dm2, commutator=True):
    """ Compute the transition RDMs.
    Two options are possible, to use the commutator of
    the excitation operators (commutator=True) or not.

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


def solve_dense(rhs, lhs, tol=1.0e-7):
    """
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10
        rhs : numpy array
            Right-hand-side matrix
        lhs : numpy array
            Left-hand-side matrix

        """
    # Invert RHS matrix
    U, s, V = svd(rhs)
    s = s ** (-1)
    # Check singular value threshold
    s[s >= 1 / tol] = 0.0
    # rhs^(-1)
    S_inv = np.diag(s)
    rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))

    # Apply RHS^-1 * LHS and
    # run eigenvalue solver
    A = np.dot(rhs_inv, lhs)
    w, v = np.linalg.eig(A)
    # Return w (eigenvalues)
    #    and v (eigenvector column matrix -- so transpose it!)
    return np.real(w), np.real(v.T)


def pickeig(w):
    "adapted from PySCF TDSCF module"
    idx = np.where(w > 0.001 ** 2)[0]
    # get unique eigvals
    b = np.sort(w[idx])
    d = np.append(True, np.diff(b))
    TOL = 1e-6
    w = b[d > TOL]
    return w


phrpa = [
    ("be_sto3g", (2, 2), 0.22224316, 4.13573698),
    ("h2_hf_sto6g", (1, 1), 0.55493436, 0.92954096),
    ("h2_3-21g", (1, 1), 0.36008273, 0.56873365),
]


def test_phrpa_rhs_comm(filename, nparts, e1, e2):
    """Test Excitation EOM for RHF reference wfn.
    The commutator form is used on the right-hand-side.
    This is equivalent to ph-RPA/TD-HF.

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Evaluate particle-hole EOM
    phrpa = eomee.ExcitationEOM(
        spinize(one_mo), antisymmetrize(spinize(two_mo)), one_dm, two_dm
    )
    ev, cv = phrpa.solve_dense(orthog="asymmetric")
    result = pickeig(ev)

    assert np.allclose(result[0], e1)
    assert np.allclose(result[1], e2)


cis = [
    ("be_sto3g", (2, 2), 0.06550711, 0.23850668),
    ("h2_hf_sto6g", (1, 1), 0.58389584, 0.94711594),
    ("h2_3-21g", (1, 1), 0.38076394, 0.57817776),
]


def test_phrpa_rhs_nocomm(filename, nparts, e1, e2):
    """Test Excitation EOM for RHF reference wfn.
    The right-hand-side is without a commutator.
    This is equivalent to CIS.

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Make particle-hole EOM object
    phrpa = eomee.ExcitationEOM(
        spinize(one_mo), antisymmetrize(spinize(two_mo)), one_dm, two_dm
    )
    # Compute right-hand-side (no-commutator form)
    # M_klij = \gamma_kj \delta_li - \Gamma_kijl
    I = np.eye(2 * nbasis, dtype=one_mo.dtype)
    rhs = np.einsum("kj,li->klji", one_dm, I, optimize=True)
    rhs -= np.einsum("kijl->klji", two_dm, optimize=True)
    rhs = rhs.reshape((2 * nbasis) ** 2, (2 * nbasis) ** 2)
    # Solve eigenvalue problem
    ev, cv = solve_dense(rhs, phrpa.lhs)
    result = pickeig(ev)

    assert np.allclose(result[0], e1)
    assert np.allclose(result[1], e2)


def get_dm2_from_tdms(cv, one_dm, two_dm, comm=True):
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


def pickpositiveeig(w, cv):
    "adapted from PySCF TDSCF module"
    idx = np.where(w > 0.01 ** 2)[0]
    return w[idx], cv[idx], idx


normalization = [
    ("be_sto3g", (2, 2), -14.351880476202),
    ("h2_hf_sto6g", (1, 1), -1.8384342592562477),
    ("h2_3-21g", (1, 1), -1.836099198341838),
    ("ne_321g", (5, 5), -127.803824528206),
    ("he_ccpvdz", (1, 1), -2.85516047724274),
]


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

    # Evaluate particle-hole EOM
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    phrpa = eomee.ExcitationEOM(one_mo, antisymmetrize(two_mo), one_dm, two_dm)
    w, cv = phrpa.solve_dense(orthog="asymmetric")
    # Reconstruct 2RDM from T-RDM and
    # check it adds to the right number
    # of electron pairs
    _, pcv, _ = pickpositiveeig(w, cv)
    rdm2 = get_dm2_from_tdms(pcv, one_dm, two_dm, comm=True)

    def check_dm_normalization_condition(npart, twodm):
        assert np.allclose(np.einsum("ijij", twodm), (npart * (npart - 1)))

    check_dm_normalization_condition(na + nb, rdm2)

    # Energy from HF RDMs
    energy1 = np.einsum("ij,ij", one_mo, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo, two_dm
    )
    assert np.allclose(energy1, ehf)
    # Energy from reconstructed RDMs
    energy2 = np.einsum("ij,ij", one_mo, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo, rdm2
    )
    print("E_HF", ehf, "E_2rdm", energy2, "\n")


def test_reconstructed_2rdm_cis(filename, nparts, ehf):
    """Evaluate ph-EOM (without commutator on right-hand-side).
    Approximate the transition RDM (using the commutator
    of the excitation operator) and reconstruct the 2-RDM.

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Evaluate particle-hole EOM (rhs no-commutator form)
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    phrpa = eomee.ExcitationEOM(one_mo, antisymmetrize(two_mo), one_dm, two_dm)
    # M_klij = \gamma_kj \delta_li - \Gamma_kijl
    I = np.eye(2 * nbasis, dtype=one_mo.dtype)
    rhs = np.einsum("kj,li->klji", one_dm, I, optimize=True)
    rhs -= np.einsum("kijl->klji", two_dm, optimize=True)
    rhs = rhs.reshape((2 * nbasis) ** 2, (2 * nbasis) ** 2)
    # Solve eigenvalue problem
    w, cv = solve_dense(rhs, phrpa.lhs)

    # Reconstruct 2RDM from T-RDM and
    # check it adds to the right number
    # of electron pairs
    _, pcv, _ = pickpositiveeig(w, cv)
    rdm2 = get_dm2_from_tdms(pcv, one_dm, two_dm, comm=True)

    def check_dm_normalization_condition(npart, twodm):
        assert np.allclose(np.einsum("ijij", twodm), (npart * (npart - 1)))

    check_dm_normalization_condition(na + nb, rdm2)

    # Energy from HF RDMs
    energy1 = np.einsum("ij,ij", one_mo, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo, two_dm
    )
    assert np.allclose(energy1, ehf)
    # Energy from reconstructed RDMs
    energy2 = np.einsum("ij,ij", one_mo, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo, rdm2
    )
    print("E_HF", ehf, "E_2rdm", energy2, "\n")


def test_phrpa_adiabaticconection(filename, nparts, ehf):
    """Test energy correction from ERPA.
    The commutator form is used on the ph-EOM right-hand-side.

    """
    one_mo = np.load(find_datafiles("{}_oneint.npy".format(filename)))
    two_mo = np.load(find_datafiles("{}_twoint.npy".format(filename)))
    nbasis = one_mo.shape[0]
    na, nb = nparts
    one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)

    # Build Fock operator
    one_mo = spinize(one_mo)
    two_mo = symmetrize(spinize(two_mo))
    Fk = np.copy(one_mo)
    Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)

    # Evaluate ERPA
    one_mo_0 = Fk
    two_mo_0 = np.zeros_like(two_mo)
    dE = eomee.ExcitationEOM.erpa(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm)
    print("E_HF", ehf)
    print("E_erpa", np.einsum("pq, pq", Fk, one_dm) + dE)


for test in phrpa:
    test_phrpa_rhs_comm(*test)

for test in cis:
    test_phrpa_rhs_nocomm(*test)

for test in normalization:
    test_reconstructed_2rdm_phrpa(*test)

for test in normalization:
    test_reconstructed_2rdm_cis(*test)

for test in normalization:
    test_phrpa_adiabaticconection(*test)

