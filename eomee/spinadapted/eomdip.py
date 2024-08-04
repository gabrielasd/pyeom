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

r"""Spin Adapted Double Ionization EOM state class."""


import numpy as np

from scipy.integrate import fixed_quad

from eomee.eomdip import DIP

from eomee.tools.erpahh import (
    _zeroth_order_rdm2,
    _get_hherpa_metric_matrix,
    _sum_over_nstates_tdtd_matrices,
)


__all__ = [
    "DIPS",
    "DIPT",
    "eval_ecorr",
]


def _get_lhs_spin_blocks(lhs, n, k):
    lhs = lhs.reshape(n, n, n, n)
    A_abab = lhs[:k, k:, :k, k:]
    A_baab = lhs[k:, :k, :k, k:]
    A_abba = lhs[:k, k:, k:, :k]
    A_baba = lhs[k:, :k, k:, :k]
    return (A_abab, A_baab, A_abba, A_baba)


def _get_rhs_spin_blocks(rhs, n, k):
    rhs = rhs.reshape(n, n, n, n)
    M_abab = rhs[:k, k:, :k, k:]
    M_baab = rhs[k:, :k, :k, k:]
    M_abba = rhs[:k, k:, k:, :k]
    M_baba = rhs[k:, :k, k:, :k]
    return (M_abab, M_baab, M_abba, M_baba)


def _get_transition_dm(cv, metric, nabsis):
    if not cv.shape[0] == nabsis ** 2:
        raise ValueError(
            f"Coefficients vector has the wrong shape, expected {nabsis**2}, got {cv.shape[0]}."
        )
    cv = cv.reshape(nabsis, nabsis)
    rhs = metric.reshape(nabsis, nabsis, nabsis, nabsis)
    return np.einsum("pqrs,rs->pq", rhs, cv)


class DIPS(DIP):
    r"""
    Spin-adapted hole-hole EOM for the singlet spin symmetry.
    
    The excitation operator is given by:

    .. math::
        \hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} - a_{\bar{i}} a_j)}

    The excited state wavefunctions and energies are obtained by solving the equation:

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} - a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} - a^\dagger_{\bar{k}} a^\dagger_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

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
        A_abab, A_baab, A_abba, A_baba = _get_lhs_spin_blocks(self._lhs, self._n, self._k)
        A = A_abab - A_baab - A_abba + A_baba
        return 0.5 * A.reshape(self._k ** 2, self._k ** 2)

    def _compute_rhs_1(self):
        M_abab, M_baab, M_abba, M_baba = _get_rhs_spin_blocks(self._rhs, self._n, self._k)
        M = M_abab - M_baab - M_abba + M_baba
        return 0.5 * M.reshape(self._k ** 2, self._k ** 2)

    def compute_tdm(self, coeffs):
        r"""
        Compute the transition RDMs for the singlet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a^\dagger_q | \Psi^{(N-2)}_\lambda >

        Parameters
        ----------
        coeffs : np.ndarray(k**2)
            Coefficients vector for the lambda-th double ionized state.
        
        Returns
        -------
        tdm : np.ndarray(k,k)
            transition RDMs between the reference (ground) state and the double ionized state.

        """
        return _get_transition_dm(coeffs, self.rhs, self._k)


class DIPT(DIP):
    r"""
    Spin-adapted hole-hole EOM for the triplet spin symmetry.
    
    The excitation operator is given by:

    .. math::
        \hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} + a_{\bar{i}} a_j)}

    The excited state wavefunctions and energies are obtained by solving the equation:

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} + a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} + a^\dagger_{\bar{k}} a^\dagger_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

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
        A_abab, A_baab, A_abba, A_baba = _get_lhs_spin_blocks(self._lhs, self._n, self._k)
        A = A_abab + A_baab + A_abba + A_baba
        return 0.5 * A.reshape(self._k ** 2, self._k ** 2)

    def _compute_rhs_30(self):
        M_abab, M_baab, M_abba, M_baba = _get_rhs_spin_blocks(self._rhs, self._n, self._k)
        M = M_abab + M_baab + M_abba + M_baba
        return 0.5 * M.reshape(self._k ** 2, self._k ** 2)

    def compute_tdm(self, coeffs):
        r"""
        Compute the transition RDMs for the triplet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a^\dagger_q | \Psi^{(N-2)}_\lambda >

        Parameters
        ----------
        coeffs : np.ndarray(k**2)
            Coefficients vector for the lambda-th double ionized state.
        
        Returns
        -------
        tdm : np.ndarray(k,k)
            transition RDMs between the reference (ground) state and the two-electron attached state.

        """
        return _get_transition_dm(coeffs, self.rhs, self._k)


### Beggins definition of utility functions that will be used to compute the residual correlation energy
### with the `eval_ecorr` function defined at the end of this file.
def _eval_tdtd_alpha_mtx_from_erpa(erpa_gevp_type, h_l, v_l, dm1, dm2, invtol, solver_type):
    # Solve the perturbation dependent particle-hole ERPA equations
    erpa = erpa_gevp_type(h_l, v_l, dm1, dm2)
    erpa._invtol = invtol
    cv = erpa.solve_dense(mode=solver_type, pick_posw=False, normalize=True)[1]
    metric = erpa.rhs.reshape(erpa.k, erpa.k, erpa.k, erpa.k)
    return _sum_over_nstates_tdtd_matrices(erpa.k, dm1, cv, metric)


def _eval_W_alpha_singlets(tdtd_singlets, dv):
    # f(alpha) = 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_term^{\alpha}_{pqrs}
    # where: Gamma_term = \sum_{n \in Singlets} tdm_0n tdm_n0
    # NOTE: the 0.5 factor in the singlet alpha-beta transition DM contributions is because each
    # contributes to G_abab and G_baba blocks of 2RDM. This factor wouldn't be necesary if we ommited the G_baba
    # block, or restricted the DIP transition operator to Q_pq with p>q.
    k = tdtd_singlets.shape[0]
    m = 2 * k
    hh_rdm2 = np.zeros((m, m, m, m), dtype=tdtd_singlets.dtype)
    hh_rdm2[:k, k:, :k, k:] = 0.5 * tdtd_singlets  # tdtd_ab
    hh_rdm2[k:, :k, k:, :k] = 0.5 * tdtd_singlets
    energy = np.einsum("pqrs,pqrs", dv, hh_rdm2, optimize=True)
    return 0.5 * energy


def _eval_W_alpha_triplets(tdtd_triplets, dv):
    # f(alpha) = 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_term^{\alpha}_{pqrs}
    # where: Gamma_term = \sum_{n \in Triplets} tdm_0n tdm_n0
    # NOTE: the 0.5 factor in the triplet alpha-beta transition DM contributions is because each
    # contributes to G_abab and G_baba blocks of 2RDM. This factor wouldn't be necesary if we ommited the G_baba
    # block, or restricted the DIP transition operator to Q_pq with p>q.
    k = tdtd_triplets.shape[0]
    m = 2 * k
    hh_rdm2 = np.zeros((m, m, m, m), dtype=tdtd_triplets.dtype)
    # 30 (fill the alpha,beta,alpha,beta blocks of the 2-RDM)
    hh_rdm2[:k, k:, :k, k:] += 0.5 * tdtd_triplets  # tdtd_ab
    hh_rdm2[k:, :k, k:, :k] += 0.5 * tdtd_triplets
    # 31 (fill the all alpha/beta block of the 2-RDM)
    hh_rdm2[:k, :k, :k, :k] = tdtd_triplets
    hh_rdm2[k:, k:, k:, k:] = tdtd_triplets
    energy = np.einsum("pqrs,pqrs", dv, hh_rdm2, optimize=True)
    return 0.5 * energy


def _eval_W_alpha_constant_terms(dv, rdm1, rdm2, summall, invtol):
    # f = -0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_^{\alpha=0}_{pqrs}
    n = rdm1.shape[0]
    metric = _get_hherpa_metric_matrix(rdm1).reshape(n ** 2, n ** 2)
    dm2_0 = _zeroth_order_rdm2(rdm2, metric, summall, invtol)
    return -0.5 * np.einsum("pqrs,pqrs", dv, dm2_0, optimize=True)


### End of utility functions


def ac_integrand_hherpa(
    lam, h0, v0, dh, dv, dm1, dm2, summall=True, invtol=1.0e-7, solvertype="nonsymm"
):
    """Compute the integrand of the adiabatic connection formulation.

    .. math::
    W(\alpha) = 0.5 \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) (\Gamma^{\alpha}_{pqrs} - \Gamma^{\alpha=0}_{pqrs})

    """
    # Compute H^alpha
    h = lam * dh
    h += h0
    v = lam * dv
    v += v0

    # Eval TDMs at alpha from particle-hole singlet transitions and compute energy
    tdtd_aa = _eval_tdtd_alpha_mtx_from_erpa(DIPS, h, v, dm1, dm2, invtol, solvertype)
    energy = _eval_W_alpha_singlets(tdtd_aa, dv)

    # Eval TDMs at alpha from particle-hole triplets transitions and compute energy
    tdtd_aa = _eval_tdtd_alpha_mtx_from_erpa(DIPT, h, v, dm1, dm2, invtol, solvertype)
    energy += _eval_W_alpha_triplets(tdtd_aa, dv)

    # Eval perturbation independent terms
    energy += _eval_W_alpha_constant_terms(dv, dm1, dm2, summall, invtol)

    return energy


def eval_ecorr(h_0, v_0, h_1, v_1, dm1, dm2, summ_all=True, inv_tol=1.0e-7, nint=5):
    """Compute the residual correlation energy from the adiabatic connection formulation and hole-hole ERPA.

    .. math::

        E_{corr} &= < \Psi^{\\alpha=1}_0 | \hat{H} | \Psi^{\\alpha=1}_0 > - < \Psi^{\\alpha=0}_0 | \hat{H} | \Psi^{\\alpha=0}_0 >
        
        &= 0.5 \sum_{pqrs} \int_{0}^{1} (v^{\\alpha=1}_{pqrs} - v^{\\alpha=0}_{prqs}) (\Gamma^{\\alpha}_{pqrs} - \Gamma^{\\alpha=0}_{pqrs}) d \\alpha

    where :math:`\Gamma^{\\alpha}_{pqrs}` is

    .. math::

        \Gamma^{\\alpha}_{pqrs} = \sum_{\\nu !=0} \gamma^{\\alpha;0 \\nu}_{pq} \gamma^{\\alpha;\\nu 0}_{sr}

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

    """
    # One-body perturbation
    dh = h_1 - h_0
    # Two-body perturbation
    dv = v_1 - v_0

    # Evaluate integrand function: W(alpha)
    @np.vectorize
    def ac_integrand(alpha):
        return ac_integrand_hherpa(
            alpha, h_0, v_0, dh, dv, dm1, dm2, summall=summ_all, invtol=inv_tol
        )

    # integrate function
    return fixed_quad(ac_integrand, 0, 1, n=nint)[0]
