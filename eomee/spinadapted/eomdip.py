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

from eomee.eomdip import DIP, DIPm
from eomee.tools import pickpositiveeig, spinize, from_unrestricted
from eomee.solver import nonsymmetric, svd_lowdin, eig_pinv


__all__ = [
    "DIPS",
    "DIPT",
]


# class DIPSA(DIP):
#     r"""
#     Excitation EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} \mp a_{\bar{i}} a_j)}`.

#     .. math::
#         \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
#         = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

#     """
#     def __init__(self, h, v, dm1, dm2):
#         super().__init__(h, v, dm1, dm2)
#         self._k = self._n // 2
#         self._lhs_sb = self._get_lhs_spinblocks()
#         self._rhs_sb = self._get_rhs_spinblocks()
#         self._lhs1 = self._compute_lhs_1()
#         self._rhs1 = self._compute_rhs_1()
#         self._lhs3 = self._compute_lhs_30()
#         self._rhs3 = self._compute_rhs_30()
    

#     @property
#     def k(self):
#         r"""
#         Return the number of spatial orbital basis functions.

#         Returns
#         -------
#         k : int
#             Number of spatial orbital basis functions.

#         """
#         return self._k

#     @property
#     def neigs(self):
#         r"""
#         Return the size of the eigensystem.

#         Returns
#         -------
#         neigs : int
#             Size of eigensystem.

#         """
#         # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
#         return (self._k) ** 2
    
#     def _get_lhs_spinblocks(self):
#         lhs = self._lhs.reshape(self._n, self._n, self._n, self._n)
#         A_abab = lhs[:self._k, self._k:, :self._k, self._k:]
#         A_baab = lhs[self._k:, :self._k, :self._k, self._k:]
#         A_abba = lhs[:self._k, self._k:, self._k:, :self._k]    
#         A_baba = lhs[self._k:, :self._k, self._k:, :self._k]
#         return (A_abab, A_baab, A_abba, A_baba)

#     def _get_rhs_spinblocks(self):
#         rhs = self._rhs.reshape(self._n, self._n, self._n, self._n)
#         M_abab = rhs[:self._k, self._k:, :self._k, self._k:]
#         M_baab = rhs[self._k:, :self._k, :self._k, self._k:]
#         M_abba = rhs[:self._k, self._k:, self._k:, :self._k]    
#         M_baba = rhs[self._k:, :self._k, self._k:, :self._k]
#         return (M_abab, M_baab, M_abba, M_baba)
    
#     def _compute_lhs_1(self):
#         A_abab, A_baab, A_abba, A_baba = self._lhs_sb
#         A = A_abab - A_baab - A_abba + A_baba
#         return 0.5 * A.reshape(self._k**2, self._k**2)
    
#     def _compute_lhs_30(self):
#         A_abab, A_baab, A_abba, A_baba = self._lhs_sb
#         A = A_abab + A_baab + A_abba + A_baba 
#         return 0.5 * A.reshape(self._k**2, self._k**2)

#     def _compute_rhs_1(self):
#         M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
#         M = M_abab - M_baab - M_abba + M_baba
#         return 0.5 * M.reshape(self._k**2, self._k**2)
    
#     def _compute_rhs_30(self):
#         M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
#         M = M_abab + M_baab + M_abba + M_baba
#         return 0.5 * M.reshape(self._k**2, self._k**2)
    
#     def solve_dense(self, tol=1.0e-10, mode="nonsymm", err="ignore", mult=1):
#         r"""
#         Solve the EOM eigenvalue system.

#         Parameters
#         ----------
#         tol : float, optional
#             Tolerance for small singular values. Default: 1.0e-10
#         mode : str, optional
#             Specifies whether a symmetric or nonsymmetric method is used to solve the GEVP.
#             Default is `nonsymm` in which the inverse of the right hand side matrix is taken.
#         err : ("warn" | "ignore" | "raise")
#             What to do if a divide-by-zero floating point error is raised.
#             Default behavior is to ignore divide by zero errors.
#         mult : int, optional
#             State multiplicity. Singlet (1) ot triplet (3) states. Default: 1

#         Returns
#         -------
#         w : np.ndarray((m,))
#             Eigenvalue array (m eigenvalues).
#         v : np.ndarray((m, n))
#             Eigenvector matrix (m eigenvectors).

#         """
#         modes = {'nonsymm': nonsymmetric, 'symm': svd_lowdin, 'qtrunc': eig_pinv}
#         if not isinstance(tol, float):
#             raise TypeError("Argument tol must be a float")        
#         try:
#             _solver = modes[mode]
#         except KeyError:
#             print(
#                 "Invalid mode parameter. Valid options are nonsymm, symm or qtrunc."
#             )
#         if mult == 1:
#             lhs = self._lhs1
#             rhs = self._rhs1
#         elif mult == 3:
#             lhs = self._lhs3
#             rhs = self._rhs3
#         else:
#             raise ValueError("Invalid state multiplicity. Valid options are 1 or 3.")
#         w, v = _solver(lhs, rhs, tol=tol, err=err)
#         return np.real(w), np.real(v)
    

#     @classmethod
#     def erpa(cls, h_0, v_0, h_1, v_1, dm1, dm2, solver="nonsymm", eigtol=1.e-7, mult=1, nint=5, dm1ac=True):
#         r"""
#         Compute the ERPA correlation energy for the operator.

#         """
#         # Size of dimensions
#         n = h_0.shape[0]
#         # H_1 - H_0
#         dh = h_1 - h_0
#         # V_1 - V_0
#         dv = v_1 - v_0
        
#         # Compute ERPA correction energy
#         # Nonlinear term (eq. 19 integrand)        
#         function = Integrandhh(cls, h_0, v_0, dh, dv, dm1, dm2)
#         if mult == 1:
#             params = (solver, eigtol, True, dm1ac)
#             alphadep=  fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
#         elif mult == 3:
#             params = (solver, eigtol, False, dm1ac)
#             alphadep =  fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
#         elif mult == 13:
#             params = (solver, eigtol, True, dm1ac)
#             alphadep =  fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
#             params = (solver, eigtol, False, dm1ac)
#             alphadep += fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
#         else:
#             raise ValueError("Invalid mult parameter. Valid options are 1, 3 or 13.")
#         if not dm1ac:
#             linear = 0.
#         else:
#             linear = _hherpa_linearterms(dh, dm1)
#         ecorr = linear + alphadep

#         output = {}
#         output["ecorr"] = ecorr
#         output["linear"] = linear
#         output["error"] = None

#         return output
    
#     @classmethod
#     def erpa_ecorr(cls, h_0, v_0, h_1, v_1, dm1, dm2, solver="nonsymm", eigtol=1.e-7, summall=True, mult=1, nint=5, dm1ac=True):
#         r"""
#         Compute the ERPA correlation energy for the operator.

#         .. math::
#         E_corr = (E^{\alpha=1} - E^{\alpha=0}) - (< \Psi^{\alpha=0}_0 | \hat{H} | \Psi^{\alpha=0}_0 > - E^{\alpha=0})
#         = 0.5 \sum_{pqrs} \int_{0}_{1} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) \Gamma^{\alpha}_{pqrs} d \alpha
#         - 0.5 \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) \Gamma^{\alpha=0}_{pqrs}

#         where :math:`\Gamma^{\alpha}_{pqrs}` is

#         .. math::
#         \Gamma^{\alpha}_{pqrs} = \sum_{\nu =0} \gamma^{\alpha;0 \nu}_{pq} \gamma^{\alpha;\nu 0}_{rs}
#         """
#         # Size of dimensions
#         n = h_0.shape[0]
#         # H_1 - H_0
#         dh = h_1 - h_0
#         # V_1 - V_0
#         dv = v_1 - v_0

#         integrand = Integrandhh(cls, h_0, v_0, dh, dv, dm1, dm2)
#         if mult == 1:
#             params = (solver, eigtol, True, dm1ac)
#             alphadep=  fixed_quad(integrand.vfunc, 0, 1, args=params, n=nint)[0]
#         elif mult == 3:
#             params = (solver, eigtol, False, dm1ac)
#             alphadep =  fixed_quad(integrand.vfunc, 0, 1, args=params, n=nint)[0]
#         elif mult == 13:
#             params = (solver, eigtol, True, dm1ac)
#             alphadep =  fixed_quad(integrand.vfunc, 0, 1, args=params, n=nint)[0]
#             params = (solver, eigtol, False, dm1ac)
#             alphadep +=  fixed_quad(integrand.vfunc, 0, 1, args=params, n=nint)[0]
#         else:
#             raise ValueError("Invalid mult parameter. Valid options are 1, 3 or 13.")
        
#         rhs = Integrandhh.eval_dmterms(n, dm1).reshape(n ** 2, n ** 2)
#         temp = _rdm2_a0(n, dm2, rhs, summall, eigtol)
#         alphaindep = -0.5 * np.einsum("pqrs,pqrs", dv, temp, optimize=True)
#         ecorr = alphadep + alphaindep

#         output = {}
#         output["ecorr"] = ecorr
#         output["linear"] = alphaindep
#         output["error"] = None

#         return output


def _hherpa_linearterms(_dh, _dm1):
    # dh * \gamma
    return np.einsum("pq,pq", _dh, _dm1, optimize=True)


def _rdm2_a0(_n, _rdm2, _rhs, _summall, _eigtol):
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        _rdm2  = truncate_rdm2_matrix(_n, d_occs_ij, _rdm2, _eigtol)
    return _rdm2


def truncate_rdm2_matrix(nspins, ij_d_occs, _rdm2, _eigtol):
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
                truncated[p,q,r,s] = _rdm2[p,q,r,s]
    return truncated


class Integrandhh:
    r"""Compute adiabatic connection integrand."""
    def __init__(self, method, h0, v0, dh, dv, dm1, dm2):
        self.h_0 = h0
        self.v_0 = v0
        self.dh = dh
        self.dv = dv
        # TODO: Check that method is DIP
        self.dm1 = dm1
        self.dm2 = dm2
        self.method = method
        self.vfunc = np.vectorize(self.eval_integrand) 
    
    @staticmethod
    def eval_dmterms(_n, _dm1):
        #FIXME: This functions returns the generalized hole-hole RHS
        # not the spin-adapted one. It is left here because its used to 
        # compute the alpha-independent terms in the classmethodsa bove.

        # Compute RDM terms of transition RDM
        # Commutator form: < |[p+q+,s r]| >
        # \delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}
        _rdm_terms = np.einsum("ik,jl->klji", np.eye(_n), np.eye(_n), optimize=True)
        _rdm_terms -= np.einsum("il,jk->klji", np.eye(_n), np.eye(_n), optimize=True)
        # - \delta_{i k} \left\{a^\dagger_{l} a_{j}\right\}
        # + \delta_{i l} \left\{a^\dagger_{k} a_{j}\right\}
        _rdm_terms -= np.einsum("ik,jl->klji", np.eye(_n), _dm1, optimize=True)
        _rdm_terms += np.einsum("il,jk->klji", np.eye(_n), _dm1, optimize=True)
        # - \delta_{j l} \left\{a^\dagger_{k} a_{i}\right\}
        # + \delta_{j k} \left\{a^\dagger_{l} a_{i}\right\}
        _rdm_terms -= np.einsum("jl,ik->klji", np.eye(_n), _dm1, optimize=True)
        _rdm_terms += np.einsum("jk,il->klji", np.eye(_n), _dm1, optimize=True)
        return _rdm_terms
    
    @staticmethod
    def eval_alphadependent_2rdmterms(_k, _dm1, coeffs, dmterms):
        # Compute transition RDMs (eq. 29)
        tdms = np.einsum("mrs,pqrs->mpq", coeffs.reshape(coeffs.shape[0], _k, _k), dmterms)
        # Compute nonlinear energy term
        _tv = np.zeros((_k, _k, _k, _k), dtype=_dm1.dtype)
        for tdm in tdms:
            _tv += np.einsum("pq,rs->pqrs", tdm, tdm, optimize=True)
        return _tv

    def eval_integrand(self, alpha, gevps, tol, singlets, _dm1ac):
        """Compute integrand."""
        # Compute H^alpha
        h = alpha * self.dh
        h += self.h_0
        v = alpha * self.dv
        v += self.v_0
        m = h.shape[0]
        k = m // 2
        # Solve EOM equations
        hh = self.method(h, v, self.dm1, self.dm2)

        if singlets:
            w, c = hh.solve_dense(tol=tol, mode=gevps, mult=1)
            metric = hh._rhs1
        else:
            w, c = hh.solve_dense(tol=tol, mode=gevps, mult=3)
            metric = hh._rhs3
        
        cv_p = pickpositiveeig(w, c)[1]
        norm = np.dot(cv_p, np.dot(metric, cv_p.T))
        diag_n = np.diag(norm)
        idx = np.where(diag_n > 0)[0]  # Remove eigenvalues with negative norm
        sqr_n = np.sqrt(diag_n[idx])
        c = (cv_p[idx].T / sqr_n).T

        # Compute transition RDMs energy contribution (eq. 29)
        metric = metric.reshape(k, k, k, k)
        tdtd_ab = Integrandhh.eval_alphadependent_2rdmterms(k, self.dm1, c, metric)
        hh_rdm2 = np.zeros((m,m,m,m), dtype=self.dm2.dtype)
        # NOTE: the 0.5 factor in the singlet and triplet alpha-beta transition DM contributions is because each
        # contributes to G_abab and G_baba blocks of 2RDM. This factor wouldn't be necesary if we ommited the G_baba
        # block, or restricted the DIP transition operator to Q_pq with p>q.
        if singlets:
            hh_rdm2[:k, k:, :k, k:] = 0.5 * tdtd_ab
            hh_rdm2[k:, :k, k:, :k] = 0.5 * tdtd_ab
        else:
            # 30
            hh_rdm2[:k, k:, :k, k:] += 0.5 * tdtd_ab
            hh_rdm2[k:, :k, k:, :k] += 0.5 * tdtd_ab
            # 31
            hh_rdm2[:k, :k, :k, :k] = tdtd_ab
            hh_rdm2[k:, k:, k:, k:] = tdtd_ab
        result = 0.5 * np.einsum("pqrs,pqrs", self.dv, hh_rdm2, optimize=True)

        if not _dm1ac:
            # Evaluate ERPA 1RDM^alpha
            nparts = np.trace(self.dm1)
            hh_rdm1 = np.einsum('pqrq->pr', hh_rdm2) / (nparts-1)
            result += np.einsum('pq,pq->', self.dh, hh_rdm1)
        
        return result     


# class DIP2SA(DIPm):
#     r"""
#     Excitation EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} \mp a_{\bar{i}} a_j)}`.

#     .. math::
#         \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
#         = \Delta_{k} \left< \Psi^{(N)}_0 \middle| a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l \hat{Q} \middle| \Psi^{(N)}_0 \right>

#     """
#     def __init__(self, h, v, dm1, dm2):
#         super().__init__(h, v, dm1, dm2)
#         self._k = self._n // 2
#         self._lhs_sb = self._get_lhs_spinblocks()
#         self._rhs_sb = self._get_rhs_spinblocks()
#         self._lhs1 = self._compute_lhs_1()
#         self._rhs1 = self._compute_rhs_1()
#         self._lhs3 = self._compute_lhs_30()
#         self._rhs3 = self._compute_rhs_30()
    

#     @property
#     def k(self):
#         r"""
#         Return the number of spatial orbital basis functions.

#         Returns
#         -------
#         k : int
#             Number of spatial orbital basis functions.

#         """
#         return self._k

#     @property
#     def neigs(self):
#         r"""
#         Return the size of the eigensystem.

#         Returns
#         -------
#         neigs : int
#             Size of eigensystem.

#         """
#         # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
#         return (self._k) ** 2
    
#     def _get_lhs_spinblocks(self):
#         lhs = self._lhs.reshape(self._n, self._n, self._n, self._n)
#         A_abab = lhs[:self._k, self._k:, :self._k, self._k:]
#         A_baab = lhs[self._k:, :self._k, :self._k, self._k:]
#         A_abba = lhs[:self._k, self._k:, self._k:, :self._k]    
#         A_baba = lhs[self._k:, :self._k, self._k:, :self._k]
#         return (A_abab, A_baab, A_abba, A_baba)

#     def _get_rhs_spinblocks(self):
#         rhs = self._rhs.reshape(self._n, self._n, self._n, self._n)
#         M_abab = rhs[:self._k, self._k:, :self._k, self._k:]
#         M_baab = rhs[self._k:, :self._k, :self._k, self._k:]
#         M_abba = rhs[:self._k, self._k:, self._k:, :self._k]    
#         M_baba = rhs[self._k:, :self._k, self._k:, :self._k]
#         return (M_abab, M_baab, M_abba, M_baba)
    
#     def _compute_lhs_1(self):
#         A_abab, A_baab, A_abba, A_baba = self._lhs_sb
#         A = A_abab - A_baab - A_abba + A_baba
#         return 0.5 * A.reshape(self._k**2, self._k**2)
    
#     def _compute_lhs_30(self):
#         A_abab, A_baab, A_abba, A_baba = self._lhs_sb
#         A = A_abab + A_baab + A_abba + A_baba 
#         return 0.5 * A.reshape(self._k**2, self._k**2)

#     def _compute_rhs_1(self):
#         M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
#         M = M_abab - M_baab - M_abba + M_baba
#         return 0.5 * M.reshape(self._k**2, self._k**2)
    
#     def _compute_rhs_30(self):
#         M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
#         M = M_abab + M_baab + M_abba + M_baba
#         return 0.5 * M.reshape(self._k**2, self._k**2)
    
#     def solve_dense(self, tol=1.0e-10, mode="nonsymm", err="ignore", mult=1):
#         r"""
#         Solve the EOM eigenvalue system.

#         Parameters
#         ----------
#         tol : float, optional
#             Tolerance for small singular values. Default: 1.0e-10
#         mode : str, optional
#             Specifies whether a symmetric or nonsymmetric method is used to solve the GEVP.
#             Default is `nonsymm` in which the inverse of the right hand side matrix is taken.
#         err : ("warn" | "ignore" | "raise")
#             What to do if a divide-by-zero floating point error is raised.
#             Default behavior is to ignore divide by zero errors.
#         mult : int, optional
#             State multiplicity. Singlet (1) ot triplet (3) states. Default: 1

#         Returns
#         -------
#         w : np.ndarray((m,))
#             Eigenvalue array (m eigenvalues).
#         v : np.ndarray((m, n))
#             Eigenvector matrix (m eigenvectors).

#         """
#         modes = {'nonsymm': nonsymmetric, 'symm': svd_lowdin, 'qtrunc': eig_pinv}
#         if not isinstance(tol, float):
#             raise TypeError("Argument tol must be a float")        
#         try:
#             _solver = modes[mode]
#         except KeyError:
#             print(
#                 "Invalid mode parameter. Valid options are nonsymm, symm or qtrunc."
#             )
#         if mult == 1:
#             lhs = self._lhs1
#             rhs = self._rhs1
#         elif mult == 3:
#             lhs = self._lhs3
#             rhs = self._rhs3
#         else:
#             raise ValueError("Invalid state multiplicity. Valid options are 1 or 3.")
#         w, v = _solver(lhs, rhs, tol=tol, err=err)
#         return np.real(w), np.real(v)


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
    if not cv.shape[0] == nabsis**2:
        raise ValueError(f"Coefficients vector has the wrong shape, expected {nabsis**2}, got {cv.shape[0]}.")
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
        return 0.5 * A.reshape(self._k**2, self._k**2)

    def _compute_rhs_1(self):
        M_abab, M_baab, M_abba, M_baba = _get_rhs_spin_blocks(self._rhs, self._n, self._k) 
        M = M_abab - M_baab - M_abba + M_baba
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def compute_tdm(self, coeffs):
        r"""
        Compute the transition RDMs for the singlet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a^\dagger_q | \Psi^{(N-2)}_\lambda >

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
        return 0.5 * A.reshape(self._k**2, self._k**2)

    def _compute_rhs_30(self):
        M_abab, M_baab, M_abba, M_baba = _get_rhs_spin_blocks(self._rhs, self._n, self._k) 
        M = M_abab + M_baab + M_abba + M_baba
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def compute_tdm(self, coeffs):
        r"""
        Compute the transition RDMs for the singlet excitations.

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a^\dagger_q | \Psi^{(N-2)}_\lambda >

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
        return _get_transition_dm(coeffs, self.rhs, self._k)
