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

r"""Electron integral transformations from spatial to spin representation and Hartree-Fock RDMs."""


from os import path

import numpy as np


__all__ = [
    "spinize",
    "symmetrize",
    "antisymmetrize",
    "from_unrestricted",
    "hartreefock_rdms",
    "find_datafiles",
    "pickpositiveeig",
]

DIRPATH = path.join(path.dirname(__file__), "test/", "data/")


def find_datafiles(file_name):
    r""" """
    datapath = path.join(path.abspath(DIRPATH), file_name)
    return path.abspath(datapath)


def spinize(x):
    r"""
    Transform a two- or four- index array from spatial to spin representation.

    Parameters
    ----------
    x : np.ndarray(float(n, n)) or np.ndarray(float(n, n, n, n))
        Spatial representation array.

    Returns
    -------
    y : np.ndarray(float(m, m)) or np.ndarray(float(m, m, m, m))
        Spin representation array.

    """
    n = x.shape[0]
    m = n * 2
    if x.ndim == 2:
        y = np.zeros((m, m))
        y[:n, :n] = x
        y[n:, n:] = x
    elif x.ndim == 4:
        y = np.zeros((m, m, m, m))
        y[:n, :n, :n, :n] = x
        y[n:, n:, n:, n:] = x
        y[:n, n:, :n, n:] = x
        y[n:, :n, n:, :n] = x
    else:
        raise ValueError("Input must have ndim == 2 or ndim == 4")
    return y


def symmetrize(x):
    r"""
    Symmetrize a two- or four- index array in the spin representation.

    Parameters
    ----------
    x : np.ndarray(float(n, n)) or np.ndarray(float(n, n, n, n))
        Two- or four- index spin representation array.

    Returns
    -------
    y : np.ndarray(float(m, m)) or np.ndarray(float(m, m, m, m))
        Antisymmetrized two- or four- index spin representation array.

    """
    if x.ndim == 2:
        y = x + x.T
        y *= 0.5
    elif x.ndim == 4:
        y = x + x.transpose(1, 0, 3, 2)
        y += x.transpose(2, 3, 0, 1)
        y += x.transpose(3, 2, 1, 0)
        y *= 0.25
    else:
        raise ValueError("Input must have ndim == 2 or ndim == 4")
    return y


def antisymmetrize(x):
    r"""
    Antisymmetrize a four-index array in the spin representation.

    Parameters
    ----------
    x : np.ndarray(float(n, n, n, n))
        Four-index spin representation array.

    Returns
    -------
    y : np.ndarray(float(n, n, n, n))
        Antisymmetrized four-index spin representation array.

    """
    if x.ndim != 4:
        raise ValueError("Input must have ndim == 4")
    return x - x.transpose(0, 1, 3, 2)


def from_unrestricted(blocks):
    r"""
    Return a two- or four- index array in the spin representation from blocks.

    A two-index array is recontrcuted from blocks (a, b).
    A four-index array is recontrcuted from blocks (aa, ab, bb).

    Parameters
    ----------
    blocks : tuple of np.ndarray of length 2 or 3
        Blocks from which to reconstruct array.

    Returns
    -------
    y : np.ndarray(float(m, m)) or np.ndarray(float(m, m, m, m))
        Spin representation array.

    """
    if len(blocks) == 2:
        for b in blocks:
            if b.ndim != 2:
                raise ValueError("Input must have ndim == 2")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k))
        y[:n, :n] = blocks[0]
        y[n:, n:] = blocks[1]
    elif len(blocks) == 3:
        for b in blocks:
            if b.ndim != 4:
                raise ValueError("Input must have ndim == 4")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k, k, k))
        y[:n, :n, :n, :n] = blocks[0]
        y[:n, n:, :n, n:] = blocks[1]
        y[n:, :n, n:, :n] = blocks[1]
        y[n:, n:, n:, n:] = blocks[2]
    else:
        raise ValueError("Invalid input")
    return y


def hartreefock_rdms(nbasis, na, nb):
    r"""
    Return the 1- and 2- RDMs of the Hartree-Fock Slater determinant.

    Returns the RDMS in the antisymmetrized spin representation.

    Parameters
    ----------
    nbasis : int
        Number of spatial basis functions.
    na : int
        Number of alpha or spin-up electrons.
    nb : int
        Number of beta or spin-down electrons.

    Returns
    -------
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).

    """
    k = 2 * nbasis
    dm1 = np.zeros((k, k))
    for i in range(na):
        dm1[i, i] = 1.0
    for i in range(nbasis, nbasis + nb):
        dm1[i, i] = 1.0
    dm2 = np.kron(dm1, dm1).reshape(k, k, k, k)
    dm2 -= dm2.transpose(0, 1, 3, 2)
    return dm1, dm2


def make_doci_hamiltonian(one_mo, two_mo):
    """Build seniority zero Hamiltonian

    Parameters
    ----------
    one_mo : numpy.array
        one-electron integrals in MO basis; (K, K) matrix, where K is number of spatial orbitals.
    two_mo : numpy.array
        two-electron integrals in MO basis; (K, K, K, K) tensor, where K is number of spatial orbitals.

    Returns
    -------
    numpy.array
        one- and two- electron integrals corresponding to the seniority zero sector of the Hamiltonian operator
    """
    # DOCI Hamiltonian
    nbasis = one_mo.shape[0]
    one_mo_sen0 = np.zeros_like(one_mo)
    two_mo_sen0 = np.zeros_like(two_mo)
    for p in range(nbasis):
        one_mo_sen0[p, p] = one_mo[p, p]
        for q in range(nbasis):
            two_mo_sen0[p, p, q, q] = two_mo[p, p, q, q]
            two_mo_sen0[p, q, p, q] = two_mo[p, q, p, q]
            two_mo_sen0[p, q, q, p] = two_mo[p, q, q, p]
    return one_mo_sen0, two_mo_sen0


def make_doci_ham_spinized(one_mo, two_mo):
    # DOCI Hamiltonian
    n = one_mo.shape[0]
    m = 2*n
    one_int_aa = np.zeros((n,n))
    two_int_aaaa = np.zeros((n,n,n,n))
    two_int_abab = np.zeros((n,n,n,n))

    for p in range(n):
        one_int_aa[p, p] = one_mo[p, p]
    # aaaa
    for p in range(n):
        for q in range(n):
            #may need to exclude p==q
            two_int_aaaa[p, q, p, q] = two_mo[p, q, p, q] 
            two_int_aaaa[p, q, q, p] = two_mo[p, q, q, p]
    # abab        
    for p in range(n):
        for q in range(n):
            two_int_abab[p, p, q, q] = two_mo[p, p, q, q]
            two_int_abab[p, q, p, q] = two_mo[p, q, p, q] # p==q overwrites above
    
    _h = np.zeros((m, m))
    _h[:n, :n] = one_int_aa
    _h[n:, n:] = one_int_aa
    _v = np.zeros((m, m, m, m))
    # aaaa
    _v[:n, :n, :n, :n] = two_int_aaaa
    # bbbb
    _v[n:, n:, n:, n:] = two_int_aaaa
    # abab
    _v[:n, n:, :n, n:] = two_int_abab
    _v[n:, :n, n:, :n] = two_int_abab.transpose((1,0,3,2))
    return _h, _v


def make_spinized_fock_hamiltonian(one_mo, two_mo, one_dm):
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    # Build Fock operator
    Fk = np.copy(one_mo)
    Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
    one_mo_0 = Fk
    two_mo_0 = np.zeros_like(two_mo)
    return one_mo_0, two_mo_0


def make_gvbpp_hamiltonian(one_mo, two_mo, gem_matrix, dm1a):
    """Build GVB-PP Hamiltonian

    Parameters
    ----------
    one_mo : numpy.array
        one-electron integrals in MO basis; (K, K) matrix, where K is number of spatial orbitals.
    two_mo : numpy.array
        two-electron integrals in MO basis; (K, K, K, K) tensor, where K is number of spatial orbitals.
    gem_matrix : numpy.array
        Binary matrix asigning spatial orbitals to geminals; (K, M) matrix, where K is number of spatial 
        orbitals and M number of geminals.
    dm1a : numpy.array
        one-electron density matrix (alpha or beta spin block) in MO basis; (K, K) matrix, where K 
        is number of spatial orbitals.

    Returns
    -------
    (K, K) and (K, K, K, K) numpy.arrays
        one- and two- electron integrals corresponding to the GVB-PP Hamiltonian
    """
    k = one_mo.shape[0]
    assert k == gem_matrix.shape[0]
    n_gems = gem_matrix.shape[1]

    def fill_ham_inter(two_mo0, two_mo, set_i, set_j, dm1):
        for p in set_i:
            for q in set_i:
                for r in set_j:
                    # g_pqrs_aaaa
                    two_mo0[p, q] += dm1[r,r]*two_mo[p,r,q,r]
                    two_mo0[p, q] -= dm1[r,r]*two_mo[p,r,r,q]
                    # g_pqrs_abab
                    two_mo0[p, q] += dm1[r,r]*two_mo[p,r,q,r]
        return two_mo0

    def fill_ham_intra(one_mo0, two_mo0, one_mo, two_mo, set_i):
        for p in set_i:
            for q in set_i:
                one_mo0[p, q] = one_mo[p, q]
                for r in set_i:
                    for s in set_i:
                        two_mo0[p, q, r, s] = two_mo[p, q, r, s]
        return one_mo0, two_mo0  

    one_mo0 = np.zeros_like(one_mo)
    two_mo0 = np.zeros_like(two_mo)
    two_mo_inter = np.zeros_like(one_mo)
    for i in range(n_gems):
        gem_i = np.nonzero(gem_matrix.T[i])[0]
        one_mo_0, two_mo_0 = fill_ham_intra(one_mo0, two_mo0, one_mo, two_mo, gem_i)
        for j in range(n_gems):
            if j != i:
                gem_j = np.nonzero(gem_matrix.T[j])[0]
                two_mo_inter = fill_ham_inter(two_mo_inter, two_mo, gem_i, gem_j, dm1a)    

    return one_mo_0, two_mo_0, two_mo_inter

# FIXME: make pickpositiveeig return sorted w, c and remove sorting from pick_{singlet, triplet}
def pickpositiveeig(w, cv, tol=0.01):
    r"""
    Adapted from PySCF TDSCF module.

    """
    idx = np.where(w > tol ** 2)[0]
    return w[idx], cv[idx], idx


def picknonzeroeigs(w, cv, tol=0.01):
    r"""
    Prune out the GEVP solutions whose eigenvalues are close to zero as determined by the tolerance tol.

    """
    idx = np.where(np.abs(w) > tol ** 2)[0]
    return w[idx], cv[idx], idx


def pickeig(w, tol=0.001):
    "adapted from PySCF TDSCF module"
    idx = np.where(w > tol ** 2)[0]
    # get unique eigvals
    b = np.sort(w[idx])
    d = np.append(True, np.diff(b))
    TOL = 1e-6
    w = b[d > TOL]
    return w


def pick_singlets(eigvals, eigvecs):
    # sort ev and cv correspondingly
    idx = eigvals.argsort()
    b = eigvals[idx]
    eigvecs = eigvecs[idx]
    # start picking up singlets
    mask = np.append(True, np.diff(b)) > 1.e-7
    unique_eigs_idx = np.where(mask)[0]
    number_unique_eigs = np.diff(unique_eigs_idx)
    idx = np.where(number_unique_eigs == 1)[0]
    singlet_idx = unique_eigs_idx[idx]
    if unique_eigs_idx[-1] == len(eigvals)-1:
        singlet_idx = np.append(singlet_idx, unique_eigs_idx[-1])
    singlets_ev = b[singlet_idx]
    singlets_cv = eigvecs[singlet_idx]
    return singlets_ev, singlets_cv, singlet_idx


def pick_multiplets(eigvals, eigvecs):
    # sort ev and cv correspondingly
    idx = eigvals.argsort()
    b = eigvals[idx]
    eigvecs = eigvecs[idx]
    # start picking up triplets
    _, _, singlet_idx = pick_singlets(eigvals, eigvecs)
    triplets_ev = np.delete(b, singlet_idx)
    triplets_cv = np.delete(eigvecs, singlet_idx, axis=0)
    return triplets_ev, triplets_cv


class TDM():
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
    def __init__(self, cv, dm1, dm2) -> None:
        self._cv = cv
        self._dm1 = dm1
        self._dm2 = dm2
        self._n = dm1.shape[0]

    def get_tdm(self, op, comm=True):
        if op == 'hh':
            rdmterms = self.hh(commutator=comm)
        elif op == 'ph':
            rdmterms = self.ph(commutator=comm)
        elif op == 'pp':
            rdmterms = self.pp(commutator=comm)
        else:
            raise ValueError("`op` must be one of `ph`, `pp` or `hh`.")
        cv = self._cv.reshape(self._cv.shape[0], self._n, self._n)
        return np.einsum("mrs,pqsr->mpq", cv, rdmterms)

    def ph(self, commutator=True):
        if commutator:
            rdm_terms = np.einsum("li,kj->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("kj,il->klji", np.eye(self._n), self._dm1, optimize=True)
        else:
            rdm_terms = np.einsum("kj,li->klji", self._dm1, np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("kijl->klji", self._dm2, optimize=True)
        return rdm_terms

    def hh(self, commutator=True):
        if commutator:
            # < |[k^+ l^+, i j]| >
            # \delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}
            rdm_terms = np.einsum("ik,jl->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("il,jk->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            # - \delta_{i k} \left\{a^\dagger_{l} a_{j}\right\}
            # + \delta_{i l} \left\{a^\dagger_{k} a_{j}\right\}
            rdm_terms -= np.einsum("ik,jl->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms += np.einsum("il,jk->klji", np.eye(self._n), self._dm1, optimize=True)
            # - \delta_{j l} \left\{a^\dagger_{k} a_{i}\right\}
            # + \delta_{j k} \left\{a^\dagger_{l} a_{i}\right\}
            rdm_terms -= np.einsum("jl,ik->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms += np.einsum("jk,il->klji", np.eye(self._n), self._dm1, optimize=True)
        else:
            # gamma_kl;n = \sum_ij < |k^+ l^+ i j| > c_ij;n
            # c_ij;n \Gamma_klji
            rdm_terms = self._dm2
        return rdm_terms

    def pp(self, commutator=True):
        if commutator:
            #
            # < |[k l, i^+ j^+]| >
            #
            # \delta_{i l} \delta_{j k} -\delta_{i k} \delta_{j l}
            rdm_terms = np.einsum("il,jk->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("ik,jl->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            # + \delta_{i k} \left\{a^\dagger_{j} a_{l}\right\}
            # - \delta_{i l} \left\{a^\dagger_{j} a_{k}\right\}
            rdm_terms += np.einsum("ik,jl->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("il,jk->klji", np.eye(self._n), self._dm1, optimize=True)
            # + \delta_{j l} \left\{a^\dagger_{i} a_{k}\right\}
            # - \delta_{j k} \left\{a^\dagger_{i} a_{l}\right\}
            rdm_terms += np.einsum("jl,ik->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("jk,il->klji", np.eye(self._n), self._dm1, optimize=True)
        else:
            #
            # < |k l i^+ j^+| >
            #
            # M_klji = \delta_li \delta_kj - \delta_ki \delta_lj
            rdm_terms = np.einsum("li,kj->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("ki,lj->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            # M_klji += \delta_{ki} \gamma_{jl} - \delta_{kj} \gamma_{li}
            #        += \delta_{lj} \gamma_{ki} - \delta_{li} \gamma_{jk}
            rdm_terms += np.einsum("ki,lj->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("kj,li->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("li,kj->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms += np.einsum("lj,ki->klji", np.eye(self._n), self._dm1, optimize=True)
            # M_klji += \Gamma_klji
            rdm_terms += self._dm2
        return rdm_terms


def reconstruct_dm2(cv, dm1, dm2, op, comm=True):
    """Reconstruct the 2-RDM from the 1-RDM and transition-RDMs.

    Parameters
    ----------
    cv : np.ndarray((n**2, n**2))
        Eigenvector matrix.
    dm1 : np.ndarray((n, n))
        Spin resolved 1-particle reduced density matrix.
    dm2 : np.ndarray((n, n, n, n))
        Spin resolved 2-particle reduced density matrix.
    comm : bool, optional
        Form used to approximate the transition-RDMs, one of commutator (True) or no commutator
        (False), by default True

    Returns
    -------
    np.ndarray((n, n, n, n))
        Reconstructed 2-RDM.
    """
    n = dm1.shape[0]
    tv = np.zeros_like(dm2)
    if op == 'hh':
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = \sum_{n=0} < | p^+ q^+|N-2> <N-2|s r| >
        tdms = TDM(cv, dm1, dm2).get_tdm('hh', comm=comm)
        tv = np.zeros_like(dm2)
        for rdm in tdms:
            tv += np.einsum("pq,rs->pqrs", rdm, rdm, optimize=True)
        return tv/2
    elif op == 'pp':
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = \deta_ps \delta_rq - delta_pr \delta_sq
        #            + \delta_qs \gamma_pr - delta_qr \gamma_ps
        #            - \delta_ps \gamma_qr + delta_pr \gamma_qs
        #            + \sum_{n=0} < |s r|N+2> <N+2|p^+ q^+| >
        eye_eye = np.einsum("ps,rq->pqrs", np.eye(n), np.eye(n), optimize=True)
        eye_eye -= np.einsum("pr,sq->pqrs", np.eye(n), np.eye(n), optimize=True)
        eye_dm1 = np.einsum("qs,pr->pqrs", np.eye(n), dm1, optimize=True)
        eye_dm1 -= np.einsum("qr,ps->pqrs", np.eye(n), dm1, optimize=True)
        eye_dm1 -= np.einsum("ps,qr->pqrs", np.eye(n), dm1, optimize=True)
        eye_dm1 += np.einsum("pr,qs->pqrs", np.eye(n), dm1, optimize=True)
        linear_terms = eye_eye + eye_dm1
        # Compute term involvin the tdms
        # \sum_{n=0} < |s r|N+2> <N+2|p^+ q^+| >
        tdms = TDM(cv, dm1, dm2).get_tdm('pp', comm=comm)
        for rdm in tdms:
            tv += np.einsum("sr,qp->pqrs", rdm, rdm, optimize=True)
        return linear_terms + tv/2
    elif op == 'ph':
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = - < | p^+ q^+ r s | >
        #            = - \delta_qr * \gamma_ps + \gamma_pr * \gamma_qs
        #            + \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
        # \gamma_pr * \gamma_qs - \delta_qr * \gamma_ps
        linear_terms = np.einsum("pr,qs->pqrs", dm1, dm1, optimize=True)
        linear_terms -= np.einsum("qr,ps->pqrs", np.eye(n), dm1, optimize=True)
        # Compute term involvin the tdms
        # \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
        tdms = TDM(cv, dm1, dm2).get_tdm('ph', comm=comm)
        for rdm in tdms:
            tv += np.einsum("pr,qs->pqrs", rdm, rdm.T, optimize=True)
        return linear_terms + tv
    else:
        raise ValueError("`op` must be one of `ph`, `pp` or `hh`.")
    

def two_positivity_condition(matrix, nbasis, dm1, dm2):
    r""" P, Q and G (2-positivity) conditions for N-representability.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. One of `p`, `g` and `q`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).

    Returns
    -------
    np.ndarray(float(n, n, n, n))
        P, Q or G-matrix from the 2-positivity conditions.
    """
    m = 2 * nbasis
    I = np.eye(m)
    if matrix == 'p':
        # P-condition: P >= 0
        # P_pqrs = <\Psi|a^\dagger_p a^\dagger_q s r|\Psi>
        #        = \Gamma_pqrs
        return dm2   # P_matrix
    elif matrix == 'q':
        # Q-condition: Q >= 0
        # Q_pqrs = <\Psi|a_p a_q s^\dagger r^\dagger|\Psi>
        #        = \Gamma_pqrs + \delta_qs \delta_pr - \delta_ps \delta_qr
        #        - \delta_qs \gamma_pr + \delta_ps \gamma_qr + \delta_qr \gamma_ps - \delta_pr \gamma_qs
        # Q_matrix = dm2
        Q_matrix = (np.einsum('qs,pr->pqrs', I, I) -  np.einsum('ps,qr->pqrs', I, I))
        Q_matrix -= (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        Q_matrix += (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1)) 
        Q_matrix += dm2
        return Q_matrix
    elif matrix == 'g':
        # G-condition: G >= 0
        # G_pqrs = <\Psi|a^\dagger_p a_q s^\dagger r|\Psi>
        #        = \delta_qs \gamma_pr - \Gamma_psrq
        return np.einsum('qs,pr->pqrs', I, dm1) - dm2.transpose((0,3,2,1))
    else:
        raise ValueError(f'Matrix must be one of `p`, `q` and `g`, {matrix} given.')


def matrix_P_to_matrix(matrix, nbasis, dm1, dm2, matrixp=None):
    r"""Transform P-matrix into G or Q-matrices.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. Either `g` or `q`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).
    matrixg : ndarray((n, n, n, n)), optional
        If provided it is taken as the G-matrix, by default None

    Returns
    -------
    ndarray((n, n, n, n))
        The type of matrix requested by `matrix`.
    """
    m = 2 * nbasis
    I = np.eye(m)
    # P_pqrs = <\Psi|a^\dagger_p a^\dagger_q s r|\Psi>
    if matrixp is None:
        P_matrix = two_positivity_condition('p', nbasis, dm1, dm2)
    else:
        P_matrix = matrixp

    if matrix == 'g':
        # P_pqrs = \delta_qs \gamma_pr - <\Psi|a^\dagger_p a_s q^\dagger r|\Psi>
        # G_psrq = \delta_qs \gamma_pr - P_pqrs = G_pqrs
        return np.einsum('qs,pr->pqrs', I, dm1) - P_matrix
    elif matrix == 'q':
        # P_pqrs = <\Psi|a_p a_q s^\dagger r^\dagger|\Psi> - \delta_qs \delta_pr + \delta_ps \delta_qr
        #        + \delta_qs \gamma_pr - \delta_ps \gamma_qr - \delta_qr \gamma_ps + \delta_pr \gamma_qs
        # Q_pqrs = P_pqrs + \delta_qs \delta_pr - \delta_ps \delta_qr
        #        - \delta_qs \gamma_pr + \delta_ps \gamma_qr + \delta_qr \gamma_ps - \delta_pr \gamma_qs
        Q_matrix = P_matrix
        Q_matrix += (np.einsum('qs,pr->pqrs', I, I) -  np.einsum('ps,qr->pqrs', I, I))
        Q_matrix -= (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        Q_matrix += (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1)) 
        return Q_matrix    
    else:
        raise ValueError(f'Matrix must be `q` or `g`, {matrix} given.')


def matrix_G_to_matrix(matrix, nbasis, dm1, dm2, matrixg=None):
    r"""Transform G-matrix into P or Q-matrices.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. Either `p` or `q`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).
    matrixg : ndarray((n, n, n, n)), optional
        If provided it is taken as the G-matrix, by default None

    Returns
    -------
    ndarray((n, n, n, n))
        The type of matrix requested by `matrix`.
    """
    m = 2 * nbasis
    I = np.eye(m)
    # G_pqrs = <\Psi|a^\dagger_p a_q s^\dagger r|\Psi>
    if matrixg is None:
        G_matrix = two_positivity_condition('g', nbasis, dm1, dm2)
    else:
        G_matrix = matrixg

    if matrix == 'p':
        # G_pqrs = \delta_qs \gamma_pr - <\Psi|a^\dagger_p s^\dagger a_q r|\Psi>
        # P_psrq = \delta_qs \gamma_pr - G_pqrs = P_pqrs        
        return np.einsum('qs,pr->pqrs', I, dm1) - G_matrix
    elif matrix == 'q':
        # G_pqrs = \delta_qs \gamma_pr - P_psrq
        # P_psrq = <\Psi|a_p a_s q^\dagger r^\dagger|\Psi> - \delta_qs \delta_pr + \delta_pq \delta_sr
        #        + \delta_qs \gamma_pr - \delta_pq \gamma_sr - \delta_sr \gamma_pq + \delta_pr \gamma_qs
        # Q_psrq = (\delta_qs \gamma_pr - G_pqrs) + \delta_qs \delta_pr - \delta_pq \delta_sr
        #        - \delta_qs \gamma_pr + \delta_pq \gamma_sr + \delta_sr \gamma_pq - \delta_pr \gamma_qs
        Q_matrix = matrix_G_to_matrix('p', nbasis, dm1, dm2)   # P_pqrs
        Q_matrix += (np.einsum('qs,pr->pqrs', I, I) -  np.einsum('ps,qr->pqrs', I, I))
        Q_matrix -= (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        Q_matrix += (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1)) 
        return Q_matrix    
    else:
        raise ValueError(f'Matrix must be `q` or `p`, {matrix} given.')


def matrix_Q_to_matrix(matrix, nbasis, dm1, dm2, matrixq=None):
    r"""Transform Q-matrix into P or G-matrices.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. Either `p` or `g`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).
    matrixg : ndarray((n, n, n, n)), optional
        If provided it is taken as the G-matrix, by default None

    Returns
    -------
    ndarray((n, n, n, n))
        The type of matrix requested by `matrix`.
    """
    m = 2 * nbasis
    I = np.eye(m)
    # Q_pqrs = <\Psi|a_p a_q s^\dagger r^\dagger|\Psi>
    if matrixq is None:
        Q_matrix = two_positivity_condition('q', nbasis, dm1, dm2)
    else:
        Q_matrix = matrixq

    if matrix == 'p':
        # Q_pqrs = <\Psi|a^\dagger_p a^\dagger_q s r|\Psi> + \delta_qs \delta_pr - \delta_ps \delta_qr
        #        - \delta_qs \gamma_pr + \delta_ps \gamma_qr + \delta_qr \gamma_ps - \delta_pr \gamma_qs
        # P_pqrs = Q_pqrs - \delta_qs \delta_pr + \delta_ps \delta_qr
        #        + \delta_qs \gamma_pr - \delta_ps \gamma_qr - \delta_qr \gamma_ps + \delta_pr \gamma_qs
        P_matrix = Q_matrix
        P_matrix += (np.einsum('ps,qr->pqrs', I, I) - np.einsum('qs,pr->pqrs', I, I))
        P_matrix += (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        P_matrix -= (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1))
        return P_matrix
    elif matrix == 'g':
        # G_pqrs = \delta_qs \gamma_pr - P_psrq
        # P_pqrs = Q_pqrs - \delta_qs \delta_pr + \delta_ps \delta_qr
        #        + \delta_qs \gamma_pr - \delta_ps \gamma_qr - \delta_qr \gamma_ps + \delta_pr \gamma_qs
        return np.einsum('qs,pr->pqrs', I, dm1) - matrix_Q_to_matrix('p', nbasis, dm1, dm2)
    else:
        raise ValueError(f'Matrix must be `p` or `g`, {matrix} given.')


def brute_hherpa_lhs(h, v, dm1, dm2):
    """hole-hole ERPA left-hand side supermatrix

    Parameters
    ----------
    h : ndarray(n,n)
        spin resolved one-electron integrals
    v : ndarray(n,n,n,n)
        spin resolved two-electron integrals (<pq|rs>)
    dm1 : ndarray(n,n)
        1-RDM
    dm2 : ndarray(n,n,n,n)
        2-RDM
    """
    n = h.shape[0]
    I = np.eye(n, dtype=h.dtype)

    a = np.einsum("ik,lj->klji", h, dm1)
    a -= np.einsum("jk,li->klji", h, dm1)
    a += np.einsum("jl,ki->klji", h, dm1)
    a -= np.einsum("il,kj->klji", h, dm1)
    #
    b = np.einsum("ik,jp->ijkp", I, I)
    hdm1 = np.einsum("pq,lq->pl", h, dm1)
    a += np.einsum("pl,ijkp->klji", hdm1, b)
    a -= np.einsum("pl,ijkp->klji", h, b)
    b = np.einsum("ip,jk->ijpk", I, I)
    a -= np.einsum("pl,ijpk->klji", hdm1, b)
    a += np.einsum("pl,ijpk->klji", h, b)
    #
    b = np.einsum("il,jp->ijlp", I, I)
    hdm1 = np.einsum("pq,kq->pk", h, dm1)
    a += np.einsum("pk,ijlp->klji", h, b)
    a -= np.einsum("pk,ijlp->klji", hdm1, b)
    b = np.einsum("ip,jl->ijpl", I, I)
    a -= np.einsum("pk,ijpl->klji", h, b)
    a += np.einsum("pk,ijpl->klji", hdm1, b)
    ##
    # # b = np.einsum("pi,qj->pqij", I, I)
    # # vI = np.einsum("pqrs,pqij->rsij", v, b)
    # # b = np.einsum("kr,ls->klrs", I, I)
    # # a -= 0.5 * np.einsum("rsij,klrs->klji", vI, b)
    # # b = np.einsum("ks,lr->klsr", I, I)
    # # a += 0.5 * np.einsum("rsij,klsr->klji", vI, b)
    # # #
    # # b = np.einsum("pj,qi->pqji", I, I)
    # # vI = np.einsum("pqrs,pqji->rsji", v, b)
    # # b = np.einsum("kr,ls->klrs", I, I)
    # # a += 0.5 * np.einsum("rsji,klrs->klji", vI, b)
    # # b = np.einsum("ks,lr->klsr", I, I)
    # # a -= 0.5 * np.einsum("rsji,klsr->klji", vI, b)
    nu = v - v.transpose(1, 0, 2, 3)
    a -= np.einsum("ijkl->klji", nu)
    ##
    b = np.einsum("kr,ls->klrs", I, dm1)
    b -= np.einsum("ks,lr->klrs", I, dm1)
    b -= np.einsum("lr,ks->klrs", I, dm1)
    b += np.einsum("ls,kr->klrs", I, dm1)
    nu = v - v.transpose(1, 0, 2, 3)
    a += 0.5 * np.einsum("ijrs,klrs->klji", nu, b)
    #
    b = np.einsum("qj,pi->qpji", I, dm1)
    b -= np.einsum("qi,pj->qpji", I, dm1)
    b -= np.einsum("pj,qi->qpji", I, dm1)
    b += np.einsum("pi,qj->qpji", I, dm1)
    nu = v - v.transpose(0, 1, 3, 2)
    a += 0.5 * np.einsum("pqkl,qpji->klji", nu, b)
    ##
    b = np.einsum("pj,qr->pqjr", I, dm1)
    b -= np.einsum("qj,pr->pqjr", I, dm1)
    ii = np.einsum("ki,ls->klis", I, I)
    ii -= np.einsum("li,ks->klis", I, I)
    vb = np.einsum("pqrs,pqjr->sj", v, b)
    a += 0.5 * np.einsum("sj,klis->klji", vb, ii)
    #
    b = np.einsum("pj,qs->pqjs", I, dm1)
    b -= np.einsum("qj,ps->pqjs", I, dm1)
    ii = np.einsum("li,kr->lkir", I, I)
    ii -= np.einsum("ki,lr->lkir", I, I)
    vb = np.einsum("pqrs,pqjs->rj", v, b)
    a += 0.5 * np.einsum("rj,lkir->klji", vb, ii)
    #
    b = np.einsum("pi,qs->pqis", I, dm1)
    b -= np.einsum("qi,ps->pqis", I, dm1)
    ii = np.einsum("kj,lr->kljr", I, I)
    ii -= np.einsum("lj,kr->kljr", I, I)
    vb = np.einsum("pqrs,pqis->ri", v, b)
    a += 0.5 * np.einsum("ri,kljr->klji", vb, ii)
    #
    b = np.einsum("pi,qr->pqir", I, dm1)
    b -= np.einsum("qi,pr->pqir", I, dm1)
    ii = np.einsum("lj,ks->lkjs", I, I)
    ii -= np.einsum("kj,ls->lkjs", I, I)
    vb = np.einsum("pqrs,pqir->si", v, b)
    a += 0.5 * np.einsum("si,lkjs->klji", vb, ii)
    ##
    a -= 0.5 * np.einsum("jqks,qlsi->klji", v, dm2)
    a += 0.5 * np.einsum("jqrk,qlri->klji", v, dm2)
    a += 0.5 * np.einsum("iqks,qlsj->klji", v, dm2)
    a -= 0.5 * np.einsum("iqrk,qlrj->klji", v, dm2)
    #
    a += 0.5 * np.einsum("jqls,qksi->klji", v, dm2)
    a -= 0.5 * np.einsum("jqrl,qkri->klji", v, dm2)
    a -= 0.5 * np.einsum("iqls,qksj->klji", v, dm2)
    a += 0.5 * np.einsum("iqrl,qkrj->klji", v, dm2)
    #
    a += 0.5 * np.einsum("pjks,plsi->klji", v, dm2)
    a -= 0.5 * np.einsum("pjrk,plri->klji", v, dm2)
    a -= 0.5 * np.einsum("piks,plsj->klji", v, dm2)
    a += 0.5 * np.einsum("pirk,plrj->klji", v, dm2)
    #
    a -= 0.5 * np.einsum("pjls,pksi->klji", v, dm2)
    a += 0.5 * np.einsum("pjrl,pkri->klji", v, dm2)
    a += 0.5 * np.einsum("pils,pksj->klji", v, dm2)
    a -= 0.5 * np.einsum("pirl,pkrj->klji", v, dm2)
    ##
    vdm2 = np.einsum("iqrs,qlrs->il", v, dm2)
    a += 0.5 * np.einsum("kj,il->klji", I, vdm2)
    vdm2 = np.einsum("jqrs,qlrs->jl", v, dm2)
    a -= 0.5 * np.einsum("ki,jl->klji", I, vdm2)
    vdm2 = np.einsum("jqrs,qkrs->jk", v, dm2)
    a += 0.5 * np.einsum("li,jk->klji", I, vdm2)
    vdm2 = np.einsum("iqrs,qkrs->ik", v, dm2)
    a -= 0.5 * np.einsum("lj,ik->klji", I, vdm2)
    #
    vdm2 = np.einsum("pjrs,plrs->jl", v, dm2)
    a += 0.5 * np.einsum("ki,jl->klji", I, vdm2)
    vdm2 = np.einsum("pirs,plrs->il", v, dm2)
    a -= 0.5 * np.einsum("kj,il->klji", I, vdm2)
    vdm2 = np.einsum("pirs,pkrs->ik", v, dm2)
    a += 0.5 * np.einsum("lj,ik->klji", I, vdm2)
    vdm2 = np.einsum("pjrs,pkrs->jk", v, dm2)
    a -= 0.5 * np.einsum("li,jk->klji", I, vdm2)
    return a.reshape(n**2, n**2)


def eval_cation_dm2(eigvec, rdm1, rdm2):
    """Compute the 2RDM (spin-resolved) from the eigenstates of hole-hole ERPA.

    Parameters
    ----------
    eigvec : ndarray(n**2, n**2)
        coefficients of an eigenstate of the hole-hole ERPA equation.
    rdm1 : ndarray(n, n)
        one-electron reduced density matrix for the N-electron system (usually the neutral ground state).
    rdm2 : ndarray(n, n, n, n)
        two-electron reduced density matrix for the N-electron system (usually the neutral ground state).
    """
    cv = eigvec.reshape(rdm1.shape)
    temp = -np.einsum("pq,rs->pqrs", cv, cv) + np.einsum("pq,sr->pqrs", cv, cv)
    temp += (np.einsum("qp,rs->pqrs", cv, cv) - np.einsum("qp,sr->pqrs", cv, cv))
    
    cvdm1 = np.einsum("ru,us->rs", cv, rdm1)
    temp += (np.einsum("pq,rs->pqrs", cv, cvdm1) - np.einsum("pq,sr->pqrs", cv, cvdm1)) 
    
    cvdm1 = np.einsum("tr,ts->sr", cv, rdm1)
    temp -= (np.einsum("pq,sr->pqrs", cv, cvdm1) - np.einsum("pq,rs->pqrs", cv, cvdm1)) # -d_pq * d_tr * g_ts
    temp += (np.einsum("qp,sr->pqrs", cv, cvdm1) - np.einsum("qp,rs->pqrs", cv, cvdm1)) # d_qp * d_tr * g_ts

    cvcv = np.einsum("pt,tr->pr", cv, cv)
    temp += (np.einsum("pr,qs->pqrs", cvcv, rdm1) - np.einsum("ps,qr->pqrs", cvcv, rdm1))
    # cvcv = np.einsum("qt,tr->qr", cv, cv)
    temp -= (np.einsum("qr,ps->pqrs", cvcv, rdm1) - np.einsum("qs,pr->pqrs", cvcv, rdm1))
    temp += (np.einsum("rp,qs->pqrs", cvcv, rdm1) - np.einsum("rq,ps->pqrs", cvcv, rdm1)) # d_ru * d_up * g_qs
    temp -= (np.einsum("sp,qr->pqrs", cvcv, rdm1) - np.einsum("sq,pr->pqrs", cvcv, rdm1)) # -d_su * d_up * g_qr
    
    cvdm2 = np.einsum("tu,qusr->qtsr", cv, rdm2)
    temp -= np.einsum("pt,qtsr->pqrs", cv, cvdm2) # -d_pt * d_tu * G_qusr
    temp -= np.einsum("pv,qrvs->pqrs", cv, cvdm2) # -d_pv * d_ru * G_quvs
    temp += np.einsum("pv,qsvr->pqrs", cv, cvdm2) # d_pv * d_su * G_quvr
    temp += np.einsum("qt,ptsr->pqrs", cv, cvdm2) # d_qt * d_tu * G_pusr
    temp += np.einsum("qv,prvs->pqrs", cv, cvdm2) # d_qv * d_ru * G_puvs
    temp -= np.einsum("qv,psvr->pqrs", cv, cvdm2) # d_qv * d_su * G_puvr
    temp += np.einsum("tp,qtsr->pqrs", cv, cvdm2) # d_tp * d_tu * G_qusr
    temp -= np.einsum("tq,ptsr->pqrs", cv, cvdm2) # -d_tq * d_tu * G_pusr

    cvcv = np.einsum("pu,ru->pr", cv, cv)
    temp -= (np.einsum("pr,qs->pqrs", cvcv, rdm1) - np.einsum("ps,qr->pqrs", cv, cvdm1))
    # cvcv = np.einsum("qu,ru->qr", cv, cv)
    temp += (np.einsum("qr,ps->pqrs", cvcv, rdm1) - np.einsum("qs,pr->pqrs", cvcv, rdm1))
    
    cvdm2 = np.einsum("tu,qtsr->qusr", cv, rdm2)
    temp += np.einsum("pu,qusr->pqrs", cv, cvdm2) # d_pu * d_tu * G_qtsr
    temp += np.einsum("pv,qrvs->pqrs", cv, cvdm2) # d_pv * d_tr * G_qtvs
    temp -= np.einsum("pv,qsvr->pqrs", cv, cvdm2) # -d_pv * d_ts * G_qtvr
    temp -= np.einsum("qu,pusr->pqrs", cv, cvdm2) # -d_qu * d_tu * G_ptsr
    temp -= np.einsum("qv,prvs->pqrs", cv, cvdm2) # -d_qv * d_tr * G_ptvs
    temp += np.einsum("qv,psvr->pqrs", cv, cvdm2) # d_qv * d_ts * G_ptvr
    temp -= np.einsum("up,qusr->pqrs", cv, cvdm2) # -d_tu * d_up * G_qtsr
    temp += np.einsum("uq,pusr->pqrs", cv, cvdm2) #  d_tu * d_uq * G_ptsr

    cvdm1 = np.einsum("pv,qv->pq", cv, rdm1)
    temp += (np.einsum("pq,rs->pqrs", cvdm1, cv) - np.einsum("pq,sr->pqrs", cvdm1, cv)) # d_pv * d_rs * g_qv
    temp -= (np.einsum("qp,rs->pqrs", cvdm1, cv) - np.einsum("qp,sr->pqrs", cvdm1, cv)) # -d_qv * d_rs * g_pv
    
    cvdm1 = np.einsum("ru,us->rs", cv, rdm1)
    temp -= (np.einsum("qp,rs->pqrs", cv, cvdm1) - np.einsum("qp,sr->pqrs", cv, cvdm1)) # -d_qp * d_ru * g_us

    cvdm1 = np.einsum("wp,qw->pq", cv, rdm1)
    temp -= (np.einsum("rs,pq->pqrs", cv, cvdm1) - np.einsum("rs,qp->pqrs", cv, cvdm1)) # -d_rs * d_wp * g_qw
    temp += (np.einsum("sr,pq->pqrs", cv, cvdm1) - np.einsum("sr,qp->pqrs", cv, cvdm1)) #  d_sr * d_wp * g_qw

    cvdm2 = np.einsum("wp,quws->qups", cv, rdm2)
    temp += np.einsum("ru,qups->pqrs", cv, cvdm2) # d_ru * d_wp * G_quws
    temp -= np.einsum("ru,puqs->pqrs", cv, cvdm2) # -d_ru * d_wq * G_puws
    temp -= np.einsum("su,qupr->pqrs", cv, cvdm2) # -d_su * d_wp * G_quwr
    temp += np.einsum("su,puqr->pqrs", cv, cvdm2) #  d_su * d_wq * G_puwr
    temp -= np.einsum("tr,qtps->pqrs", cv, cvdm2) # -d_tr * d_wp * G_qtws
    temp += np.einsum("tr,ptqs->pqrs", cv, cvdm2) #  d_tr * d_wq * G_ptws
    temp += np.einsum("ts,qtpr->pqrs", cv, cvdm2) #  d_ts * d_wp * G_qtwr
    temp -= np.einsum("ts,ptqr->pqrs", cv, cvdm2) # -d_ts * d_wq * G_ptwr

    cvcv = np.einsum("tp,tr->pr", cv, cv)
    temp -= (np.einsum("pr,qs->pqrs", cvcv, rdm1) - np.einsum("ps,qr->pqrs", cv, cvdm1)) # -d_tp * d_tr * g_qs
    temp += (np.einsum("qr,ps->pqrs", cvcv, rdm1) - np.einsum("qs,pr->pqrs", cv, cvdm1)) #  d_tq * d_tr * g_ps

    return temp
    
    
