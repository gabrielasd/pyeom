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

r"""Spin Adapted Excitation EOM state class."""


import numpy as np

from scipy.integrate import fixed_quad

from eomee.excitation import EE, EEm
from eomee.tools import spinize, from_unrestricted
from eomee.solver import eig_pinvb, lowdin_svd, eig_pruneq_pinvb, pick_positive, eig_invb


__all__ = [
    "EES",
    "EET",
    "EESm",
    "EETm",
]


def _get_lhs_spin_blocks(lhs, n, k):
    lhs = lhs.reshape(n, n, n, n)
    A_aaaa = lhs[:k, :k, :k, :k]
    A_bbbb = lhs[k:, k:, k:, k:]
    A_aabb = lhs[:k, :k, k:, k:]
    A_bbaa = lhs[k:, k:, :k, :k]
    return (A_aaaa, A_bbbb, A_aabb, A_bbaa)


def _get_rhs_spin_blocks(rhs, n, k):
    rhs = rhs.reshape(n, n, n, n)
    M_aaaa = rhs[:k, :k, : k, : k]
    M_bbbb = rhs[ k:,  k:,  k:,  k:]
    M_aabb = rhs[: k, : k,  k:,  k:]
    M_bbaa = rhs[ k:,  k:, : k, : k]
    return (M_aaaa, M_bbbb, M_aabb, M_bbaa)


def _get_transition_rdm1(cv, metric, nabsis):
    if not cv.shape[0] == nabsis**2:
        raise ValueError(f"Coefficients vector has the wrong shape, expected {nabsis**2}, got {cv.shape[0]}.")
    cv = cv.reshape(nabsis, nabsis)
    rhs = metric.reshape(nabsis, nabsis, nabsis, nabsis)
    return np.einsum("pqrs,rs->pq", rhs, cv)


class EES(EE):
    r"""
    Spin-adapted particle-hole EOM for the singlet spin symmetry.
    
    The excitation operator is given by:

    .. math::
        \hat{Q}_k = \sum_{ij} { c_{ij} (a^{\dagger}_i  a_j + a^{\dagger}_{\bar{i}}  a_{\bar{j}})}

    The excited state wavefunctions and energies are obtained by solving the equation:
    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l + a^{\dagger}_{\bar{k}}  a_{\bar{l}}, \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l + a^{\dagger}_{\bar{k}}  a_{\bar{l}}, \hat{Q} \right] \Psi^{(N)}_0 \right>

    """
    def __init__(self, h, v, dm1, dm2):
        super().__init__(h, v, dm1, dm2)
        self._k = self._n // 2
        # Generalized particle-hole matrices
        self._lhs_ab = self._lhs
        self._rhs_ab = self._rhs
        # Spin-adapted particle-hole matrices
        self._lhs = self._compute_lhs_1()
        self._rhs = self._compute_rhs_1()
    

    @property
    def k(self):
        r"""
        Return the number of spatial orbital basis functions.

        Returns
        -------
        k : int
            Number of spatial orbital basis functions.

        """
        return self._k

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
        return (self._k) ** 2
    
    def _compute_lhs_1(self):
        A_aaaa, A_bbbb, A_aabb, A_bbaa = _get_lhs_spin_blocks(self._lhs, self._n, self._k)
        A = A_aaaa + A_bbbb + A_aabb + A_bbaa
        return 0.5 * A.reshape(self._k**2, self._k**2)

    def _compute_rhs_1(self):
        M_aaaa, M_bbbb, M_aabb, M_bbaa = _get_rhs_spin_blocks(self._rhs, self._n, self._k) 
        M = M_aaaa + M_bbbb + M_aabb + M_bbaa
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def compute_tdm1(self, coeffs):
        r"""
        Compute the transition RDMs for the singlet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a_q | \Psi^{(N)}_\lambda >

        The diagonal elements of this matrix are zero.

        Parameters
        ----------
        coeffs : np.ndarray(k**2)
            Coefficients vector for the lambda-th excited state.
        
        Returns
        -------
        tdm1 : np.ndarray(k,k)
            1-electron reduced transition RDMs.

        """
        return _get_transition_rdm1(coeffs, self.rhs, self._k)


class EET(EE):
    r"""
    Spin-adapted particle-hole EOM for the triplet spin symmetry.
    
    The excitation operator is given by:

    .. math::
        \hat{Q}_k = \sum_{ij} { c_{ij} (a^{\dagger}_i  a_j - a^{\dagger}_{\bar{i}}  a_{\bar{j}})}

    The excited state wavefunctions and energies are obtained by solving the equation:
    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l - a^{\dagger}_{\bar{k}}  a_{\bar{l}}, \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l - a^{\dagger}_{\bar{k}}  a_{\bar{l}}, \hat{Q} \right] \Psi^{(N)}_0 \right>

    """
    def __init__(self, h, v, dm1, dm2):
        super().__init__(h, v, dm1, dm2)
        self._k = self._n // 2
        # Generalized particle-hole matrices
        self._lhs_ab = self._lhs
        self._rhs_ab = self._rhs
        # Spin-adapted particle-hole matrices
        self._lhs = self._compute_lhs_30()
        self._rhs = self._compute_rhs_30()
    

    @property
    def k(self):
        r"""
        Return the number of spatial orbital basis functions.

        Returns
        -------
        k : int
            Number of spatial orbital basis functions.

        """
        return self._k

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
        return (self._k) ** 2
    
    def _compute_lhs_30(self):
        A_aaaa, A_bbbb, A_aabb, A_bbaa = _get_lhs_spin_blocks(self._lhs, self._n, self._k)
        A = A_aaaa + A_bbbb - A_aabb - A_bbaa
        return 0.5 * A.reshape(self._k**2, self._k**2)
    
    def _compute_rhs_30(self):
        M_aaaa, M_bbbb, M_aabb, M_bbaa = _get_rhs_spin_blocks(self._rhs, self._n, self._k) 
        M = M_aaaa + M_bbbb - M_aabb - M_bbaa
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def compute_tdm1(self, coeffs):
        r"""
        Compute the transition RDMs for the triplet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a_q | \Psi^{(N)}_\lambda >

        The diagonal elements of this matrix are zero.

        Parameters
        ----------
        coeffs : np.ndarray(k**2)
            Coefficients vector for the lambda-th excited state.
        
        Returns
        -------
        tdm1 : np.ndarray(k,k)
            1-electron reduced transition RDMs.

        """
        return _get_transition_rdm1(coeffs, self.rhs, self._k)


class EESm(EEm):
    r"""
    Singlet excited states for the particle-hole EOM without commutator on the metric matrix.

    """
    def __init__(self, h, v, dm1, dm2):
        super().__init__(h, v, dm1, dm2)
        self._k = self._n // 2
        # Generalized particle-hole matrices
        self._lhs_ab = self._lhs
        self._rhs_ab = self._rhs
        # Spin-adapted particle-hole matrices
        self._lhs = self._compute_lhs_30()
        self._rhs = self._compute_rhs_30()
    

    @property
    def k(self):
        r"""
        Return the number of spatial orbital basis functions.

        Returns
        -------
        k : int
            Number of spatial orbital basis functions.

        """
        return self._k

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
        return (self._k) ** 2
    
    def _compute_lhs_30(self):
        A_aaaa, A_bbbb, A_aabb, A_bbaa = _get_lhs_spin_blocks(self._lhs, self._n, self._k)
        A = A_aaaa + A_bbbb - A_aabb - A_bbaa
        return 0.5 * A.reshape(self._k**2, self._k**2)
    
    def _compute_rhs_30(self):
        M_aaaa, M_bbbb, M_aabb, M_bbaa = _get_rhs_spin_blocks(self._rhs, self._n, self._k) 
        M = M_aaaa + M_bbbb - M_aabb - M_bbaa
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def compute_tdm1(self, coeffs):
        r"""
        Compute the transition RDMs for the triplet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a_q | \Psi^{(N)}_\lambda >

        The diagonal elements of this matrix are zero.

        Parameters
        ----------
        coeffs : np.ndarray(k**2)
            Coefficients vector for the lambda-th excited state.
        
        Returns
        -------
        tdm1 : np.ndarray(k,k)
            1-electron reduced transition RDMs.

        """
        return _get_transition_rdm1(coeffs, self.rhs, self._k)


class EETm(EEm):
    r"""
    Triplet excited states for the particle-hole EOM without commutator on the metric matrix.

    """
    def __init__(self, h, v, dm1, dm2):
        super().__init__(h, v, dm1, dm2)
        self._k = self._n // 2
        # Generalized particle-hole matrices
        self._lhs_ab = self._lhs
        self._rhs_ab = self._rhs
        # Spin-adapted particle-hole matrices
        self._lhs = self._compute_lhs_30()
        self._rhs = self._compute_rhs_30()
    

    @property
    def k(self):
        r"""
        Return the number of spatial orbital basis functions.

        Returns
        -------
        k : int
            Number of spatial orbital basis functions.

        """
        return self._k

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
        return (self._k) ** 2
    
    def _compute_lhs_30(self):
        A_aaaa, A_bbbb, A_aabb, A_bbaa = _get_lhs_spin_blocks(self._lhs, self._n, self._k)
        A = A_aaaa + A_bbbb - A_aabb - A_bbaa
        return 0.5 * A.reshape(self._k**2, self._k**2)
    
    def _compute_rhs_30(self):
        M_aaaa, M_bbbb, M_aabb, M_bbaa = _get_rhs_spin_blocks(self._rhs, self._n, self._k) 
        M = M_aaaa + M_bbbb - M_aabb - M_bbaa
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def compute_tdm1(self, coeffs):
        r"""
        Compute the transition RDMs for the triplet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a_q | \Psi^{(N)}_\lambda >

        The diagonal elements of this matrix are zero.

        Parameters
        ----------
        coeffs : np.ndarray(k**2)
            Coefficients vector for the lambda-th excited state.
        
        Returns
        -------
        tdm1 : np.ndarray(k,k)
            1-electron reduced transition RDMs.

        """
        return _get_transition_rdm1(coeffs, self.rhs, self._k)


def _pherpa_linearterms(_n, _dh, _dv, _dm1):
    # Gamma_pqrs = < | p^+ q^+ s r | > = - < | p^+ q^+ r s | >
    #            = - \delta_qr * \gamma_ps
    #            + \gamma_pr * \gamma_qs
    #            + \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
    dm1_eye = np.einsum("qr,ps->pqrs", np.eye(_n), _dm1, optimize=True)
    # Compute linear term (eq. 19)
    # dh * \gamma + 0.5 * dv * (\gamma_pr * \gamma_qs - \delta_qr * \gamma_ps)
    _linear = np.einsum("pr,qs->pqrs", _dm1, _dm1, optimize=True) - dm1_eye
    _linear = np.einsum("pq,pq", _dh, _dm1, optimize=True) + 0.5 * np.einsum(
        "pqrs,pqrs", _dv, _linear, optimize=True
    )
    return _linear


def _truncate_dm1dm1_matrix(nspins, ij_d_occs, _dm1dm1, _eigtol):
    nt = nspins**2
    truncated = np.zeros_like(_dm1dm1)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > _eigtol
            cond2 = np.abs(ij_d_occs[rs]) > _eigtol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                truncated[p,r,q,s] = _dm1dm1[p,r,q,s]
    return truncated


def _truncate_eyedm1_matrix(nspins, ij_d_occs, _eyedm1, _eigtol):
    nt = nspins**2
    truncated = np.zeros_like(_eyedm1)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > _eigtol
            cond2 = np.abs(ij_d_occs[rs]) > _eigtol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                truncated[p,q,r,s] = _eyedm1[p,q,r,s]
    return truncated


def _truncate_rdm2_matrix(nspins, ij_d_occs, _rdm2, _eigtol):
    nt = nspins**2
    truncated = np.zeros_like(_rdm2)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > _eigtol
            cond2 = np.abs(ij_d_occs[rs]) > _eigtol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                truncated[p,r,q,s] = _rdm2[p,r,q,s]
    return truncated


def _alpha_independent_terms_rdm2_alpha(_dm1, _rhs, _summall, _eigtol):
    # (\gamma_pr * \gamma_qs - \delta_qr * \gamma_ps)
    _n = _dm1.shape[0]
    dm1dm1 = np.einsum("pr,qs->pqrs", _dm1, _dm1, optimize=True)
    dm1_eye = np.einsum("qr,ps->pqrs", np.eye(_n), _dm1, optimize=True)
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        dm1dm1  = _truncate_dm1dm1_matrix(_n, d_occs_ij, dm1dm1, _eigtol)
        dm1_eye  = _truncate_eyedm1_matrix(_n, d_occs_ij, dm1_eye, _eigtol)
    return (dm1dm1 - dm1_eye)


def _rdm2_a0(_rdm2, _rhs, _summall, _eigtol):
    _n = _rdm2.shape[0]
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        _rdm2  = _truncate_rdm2_matrix(_n, d_occs_ij, _rdm2, _eigtol)
    return _rdm2


def _get_pherpa_metric_matrix(dm1):
    # Compute ph-ERPA metric matrix
    # < |[p^+ q,s^+ r]| > = \delta_qs \gamma_pr - \delta_pr \gamma_sq
    _n = dm1.shape[0]
    _rdm_terms = np.einsum("qs,pr->pqrs", np.eye(_n), dm1, optimize=True)
    _rdm_terms -= np.einsum("pr,sq->pqrs", np.eye(_n), dm1, optimize=True)
    return _rdm_terms


def _sum_over_nstates_tdtd_matrices(_k, _dm1, coeffs, dmterms):
    # Compute transition RDMs (eq. 29)
    tdms = np.einsum("mrs,pqrs->mpq", coeffs.reshape(coeffs.shape[0], _k, _k), dmterms)
    # Compute nonlinear energy term
    _tv = np.zeros((_k, _k, _k, _k), dtype=_dm1.dtype)
    for tdm in tdms:
        _tv += np.einsum("pr,qs->pqrs", tdm, tdm.T, optimize=True)
    return _tv


def _eval_tdtd_alpha_mtx_from_erpa(erpa_gevp_type, h_l, v_l, dm1, dm2, invtol, solver_type):
    # Solve particle-hole ERPA equations at given perturbation strength alpha
    ph = erpa_gevp_type(h_l, v_l, dm1, dm2)
    ph._invtol = invtol
    cv = ph.solve_dense(mode=solver_type, normalize=True)[1]
    # norm = np.dot(cv, np.dot(ph.rhs, cv.T))
    # diag_n = np.diag(norm)
    # sqr_n = np.sqrt(np.abs(diag_n))
    # cv = (cv.T / sqr_n).T

    metric = ph.rhs.reshape(ph.k, ph.k, ph.k, ph.k)
    return 0.5 * _sum_over_nstates_tdtd_matrices(ph.k, dm1, cv, metric) # where is the 0.5 factor coming from?


def _eval_W_alpha_singlets(tdtd_singlets, dv):
    # f(alpha) = 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_term^{\alpha}_{pqrs}
    # Gamma_term = \sum_{n \in Singlets} tdm_0n tdm_n0
    tdtd = spinize(tdtd_singlets)
    energy = np.einsum("pqrs,pqrs", dv, tdtd, optimize=True)
    return 0.5 * energy


def _eval_W_alpha_triplets(tdtd_triplets, dv):
    # f(alpha) = 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_term^{\alpha}_{pqrs}
    # Gamma_term = \sum_{n \in Triplets} tdm_0n tdm_n0
    tdtd = from_unrestricted([tdtd_triplets, -tdtd_triplets, tdtd_triplets]) # tdtd_aa, tdtd_ab, tsts_bb
    energy = np.einsum("pqrs,pqrs", dv, tdtd, optimize=True)
    return 0.5 * energy


def _eval_W_alpha_constant_terms(dv, rdm1, rdm2, summall, invtol):
    # 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_terms_{pqrs}
    # Gamma_terms = (gamma_{pr} * gamma_{qs} + delta_{qr} * gamma_{ps})
    #             - Gamma_^{\alpha=0}
    n = rdm1.shape[0]
    rhs = _get_pherpa_metric_matrix(rdm1).reshape(n ** 2, n ** 2)
    temp = _alpha_independent_terms_rdm2_alpha(rdm1, rhs, summall, invtol)
    temp -= _rdm2_a0(rdm2, rhs, summall, invtol)
    return 0.5 * np.einsum("pqrs,pqrs", dv, temp, optimize=True)


def ac_integrand_pherpa(lam, h0, v0, dh, dv, dm1, dm2, summall=True, invtol=1.0e-7, solvertype="nonsymm"):
    """Compute the integrand of the adiabatic connection formulation.

    .. math::
    W(\alpha) = 0.5 \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) (\Gamma^{\alpha}_{pqrs} - \Gamma^{\alpha=0}_{pqrs})

    where :math:`\Gamma^{\alpha}_{pqrs}` is

    .. math::
    \Gamma^{\alpha}_{pqrs} = \gamma^{\alpha=0}_{pr} \gamma^{\alpha=0}_{qs} 
    + \sum_{\nu \in Singlets} \gamma^{\alpha;0 \nu}_{pr} \gamma^{\alpha;\nu 0}_{qs} 
    + \sum_{\nu \in Triplets} \gamma^{\alpha;0 \nu}_{pr} \gamma^{\alpha;\nu 0}_{qs} 
    - \delta_{ps} \gamma^{\alpha=0}_{qr}

    Parameters
    ----------
    lam : _type_
        _description_
    h0 : _type_
        _description_
    v0 : _type_
        _description_
    dh : _type_
        _description_
    dv : _type_
        _description_
    dm1 : _type_
        _description_
    dm2 : _type_
        _description_
    summall : bool, optional
        _description_, by default True
    invtol : _type_, optional
        _description_, by default 1.0e-7
    solvertype : str, optional
        _description_, by default "nonsymm"

    Returns
    -------
    _type_
        _description_
    """
    # Compute H^alpha
    h = lam * dh
    h += h0
    v = lam * dv
    v += v0

    # Eval TDMs at alpha from particle-hole singlet transitions and compute energy
    tdtd_aa = _eval_tdtd_alpha_mtx_from_erpa(EES, h, v, dm1, dm2, invtol, solvertype)
    energy = _eval_W_alpha_singlets(tdtd_aa, dv)

    # Eval TDMs at alpha from particle-hole triplets transitions and compute energy
    tdtd_aa = _eval_tdtd_alpha_mtx_from_erpa(EET, h, v, dm1, dm2, invtol, solvertype)    
    energy += _eval_W_alpha_triplets(tdtd_aa, dv)

    # Eval perturbation independent terms
    energy += _eval_W_alpha_constant_terms(dv, dm1, dm2, summall, invtol)

    return energy


def eval_ecorr(h_0, v_0, h_1, v_1, dm1, dm2, summ_all=True, inv_tol=1.0e-7, nint=5):
    """Compute the (dynamic) correlation energy from the adiabatic connection formulation and 
    particle-hole ERPA.

    .. math::
    E_corr = < \Psi^{\alpha=1}_0 | \hat{H} | \Psi^{\alpha=1}_0 > - < \Psi^{\alpha=0}_0 | \hat{H} | \Psi^{\alpha=0}_0 >
    = 0.5 \sum_{pqrs} \int_{0}_{1} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) (\Gamma^{\alpha}_{pqrs} - \Gamma^{\alpha=0}_{pqrs}) d \alpha

    where :math:`\Gamma^{\alpha}_{pqrs}` is

    .. math::
    \Gamma^{\alpha}_{pqrs} = \gamma^{\alpha=0}_{pr} \gamma^{\alpha=0}_{qs} 
    + \sum_{\nu !=0} \gamma^{\alpha;0 \nu}_{pr} \gamma^{\alpha;\nu 0}_{qs} 
    - \delta_{ps} \gamma^{\alpha=0}_{qr}

    Parameters
    ----------
    h_0 : np.ndarray((n, n))
        One-electron integrals for the reference Hamiltonian (at alpha=0).
    v_0 : np.ndarray((n, n, n, n))
        Two-electron integrals for the reference Hamiltonian (at alpha=0).
    h_1 : np.ndarray((n, n))
        One-electron integrals for the true Hamiltonian (at alpha=1).
    v_1 : np.ndarray((n, n, n, n))
        Two-electron integrals for the true Hamiltonian (at alpha=1).
    dm1 : np.ndarray((n, n))
        One-electron reduced density matrix for the reference wavefunction (at alpha=0).
    dm2 : np.ndarray((n, n, n, n))
        Two-electron reduced density matrix for the reference wavefunction (at alpha=0).
    summ_all : bool, optional
        Whether the sum over the two body terms is carried over all `p,q,r,s` indexes or not.
        If False, pairs of spin-orbitals that are not involved in any particle-hole excitation
        are excluded. Which pair to remove is determined by the diagonal elements of the ERPA 
        metric matrix. By default True.
    inv_tol : float, optional
        Tolerance for small singular values when solving the ERPA eigenvalue problem, 
        by default 1.0e-7.
    nint : int, optional
        Order of quadrature integration, by default 5.

    Returns
    -------
    _type_
        _description_
    """
    # H_1 - H_0
    dh = h_1 - h_0
    # V_1 - V_0
    dv = v_1 - v_0

    # Evaluate integrand function: W(alpha)
    @np.vectorize
    def ac_integrand(alpha):        
        return ac_integrand_pherpa(alpha, h_0, v_0, dh, dv, dm1, dm2, summall=summ_all, invtol=inv_tol)

    # integrate function
    return fixed_quad(ac_integrand, 0, 1, n=nint)[0]