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

from eomee.doubleelectronaff import EOMDEA, EOMDEA2
from eomee.tools import picknonzeroeigs
from eomee.solver import nonsymmetric, svd_lowdin, eig_pinv


__all__ = [
    "DEASA",
]


class DEASA(EOMDEA):
    r"""
    EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a^\dagger_i  a^\dagger_{\bar{j}} \mp a^\dagger_{\bar{i}} a^\dagger_j)}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a_k  a_{\bar{l}} \mp a_{\bar{k}} a_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a_k  a_{\bar{l}} \mp a_{\bar{k}} a_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

    """
    def __init__(self, h, v, dm1, dm2):
        super().__init__(h, v, dm1, dm2)
        self._k = self._n // 2
        self._lhs_sb = self._get_lhs_spinblocks()
        self._rhs_sb = self._get_rhs_spinblocks()
        self._lhs1 = self._compute_lhs_1()
        self._rhs1 = self._compute_rhs_1()
        self._lhs3 = self._compute_lhs_30()
        self._rhs3 = self._compute_rhs_30()
    

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
    
    def _get_lhs_spinblocks(self):
        lhs = self._lhs.reshape(self._n, self._n, self._n, self._n)
        A_abab = lhs[:self._k, self._k:, :self._k, self._k:]
        A_baab = lhs[self._k:, :self._k, :self._k, self._k:]
        A_abba = lhs[:self._k, self._k:, self._k:, :self._k]    
        A_baba = lhs[self._k:, :self._k, self._k:, :self._k]
        return (A_abab, A_baab, A_abba, A_baba)

    def _get_rhs_spinblocks(self):
        rhs = self._rhs.reshape(self._n, self._n, self._n, self._n)
        M_abab = rhs[:self._k, self._k:, :self._k, self._k:]
        M_baab = rhs[self._k:, :self._k, :self._k, self._k:]
        M_abba = rhs[:self._k, self._k:, self._k:, :self._k]    
        M_baba = rhs[self._k:, :self._k, self._k:, :self._k]
        return (M_abab, M_baab, M_abba, M_baba)
    
    def _compute_lhs_1(self):
        A_abab, A_baab, A_abba, A_baba = self._lhs_sb
        A = A_abab - A_baab - A_abba + A_baba
        return 0.5 * A.reshape(self._k**2, self._k**2)
    
    def _compute_lhs_30(self):
        A_abab, A_baab, A_abba, A_baba = self._lhs_sb
        A = A_abab + A_baab + A_abba + A_baba 
        return 0.5 * A.reshape(self._k**2, self._k**2)

    def _compute_rhs_1(self):
        M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
        M = M_abab - M_baab - M_abba + M_baba
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def _compute_rhs_30(self):
        M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
        M = M_abab + M_baab + M_abba + M_baba
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def solve_dense(self, tol=1.0e-10, mode="nonsymm", err="ignore", mult=1):
        r"""
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10
        mode : str, optional
            Specifies whether a symmetric or nonsymmetric method is used to solve the GEVP.
            Default is `nonsymm` in which the inverse of the right hand side matrix is taken.
        err : ("warn" | "ignore" | "raise")
            What to do if a divide-by-zero floating point error is raised.
            Default behavior is to ignore divide by zero errors.
        mult : int, optional
            State multiplicity. Singlet (1) ot triplet (3) states. Default: 1

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        modes = {'nonsymm': nonsymmetric, 'symm': svd_lowdin, 'qtrunc': eig_pinv}
        if not isinstance(tol, float):
            raise TypeError("Argument tol must be a float")        
        try:
            _solver = modes[mode]
        except KeyError:
            print(
                "Invalid mode parameter. Valid options are nonsymm, symm or qtrunc."
            )
        if mult == 1:
            lhs = self._lhs1
            rhs = self._rhs1
        elif mult == 3:
            lhs = self._lhs3
            rhs = self._rhs3
        else:
            raise ValueError("Invalid state multiplicity. Valid options are 1 or 3.")
        w, v = _solver(lhs, rhs, tol=tol, err=err)
        return np.real(w), np.real(v) 
    
    @classmethod
    def erpa(cls, h_0, v_0, h_1, v_1, dm1, dm2, solver="nonsymm", eigtol=1.e-7, mult=1, nint=5, dm1ac=True):
        r"""
        Compute the ERPA correlation energy for the operator.

        """
        # Size of dimensions
        n = h_0.shape[0]
        # H_1 - H_0
        dh = h_1 - h_0
        # V_1 - V_0
        dv = v_1 - v_0
        
        if not dm1ac:
            raise NotImplementedError("dm1ac=False not implemented yet.")
        else:
            linear = _pperpa_linearterms(dh, dv, dm1)

        # Compute ERPA correction energy (eq. 19 integrand) 
        function = IntegrandPP(cls, h_0, v_0, dh, dv, dm1, dm2)
        if mult == 1:
            params = (solver, eigtol, True, dm1ac)
            alphadep=  fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
        elif mult == 3:
            params = (solver, eigtol, False, dm1ac)
            alphadep =  fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
        elif mult == 13:
            params = (solver, eigtol, True, dm1ac)
            alphadep =  fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
            params = (solver, eigtol, False, dm1ac)
            alphadep += fixed_quad(function.vfunc, 0, 1, args=params, n=nint)[0]
        else:
            raise ValueError("Invalid mult parameter. Valid options are 1, 3 or 13.")  
        ecorr = linear + alphadep    

        output = {}
        output["ecorr"] = ecorr
        output["linear"] = linear
        output["error"] = None

        return output


def _pperpa_linearterms(_dh, _dv, _dm1):
    _n = _dm1.shape[0]
    # \delta_pr * \gamma_qs
    eye_dm1 = np.einsum("pr,qs->pqrs", np.eye(_n), _dm1, optimize=True)
    # \delta_qs * \gamma_pr
    dm1_eye = np.einsum("pr,qs->pqrs", _dm1, np.eye(_n), optimize=True)

    # Compute linear term (eq. 19)
    # dh * \gamma + 0.5 * dv * (\delta_pr * \gamma_qs + \delta_qs * \gamma_pr - \delta_ps * \gamma_qr
    #                           - \delta_qr * \gamma_ps - \delta_pr * \delta_qs + \delta_ps * \delta_qr)
    _linear = (
        eye_dm1
        - np.transpose(eye_dm1, axes=(0, 1, 3, 2))
        + dm1_eye
        - np.transpose(dm1_eye, axes=(0, 1, 3, 2))
        - np.einsum("pr,qs->pqrs", np.eye(_n), np.eye(_n), optimize=True)
        + np.einsum("ps,qr->pqrs", np.eye(_n), np.eye(_n), optimize=True)
    )
    _linear = np.einsum("pq,pq", _dh, _dm1, optimize=True) + 0.5 * np.einsum(
        "pqrs,pqrs", _dv, _linear, optimize=True
    )
    return _linear


def eval_alphadependent_2dmterms(_dm1, evecs, dmterms):
    _k = _dm1.shape[0]//2
    # Compute transition RDMs (eq. 29)
    tdms = np.einsum("mrs,pqrs->mpq", evecs.reshape(evecs.shape[0], _k, _k), dmterms)
    # Compute nonlinear energy term
    _tv = np.zeros((_k, _k, _k, _k), dtype=_dm1.dtype)
    for tdm in tdms:
        _tv += np.einsum("pq,rs->pqrs", tdm, tdm, optimize=True)
    return _tv


def eval_tdmterms(_dm1):
    # Note: This functions returns the generalized particle-particle RHS
    # not the spin-adapted one.
    n = _dm1.shape[0]
    # Compute inmutable terms in (eq. 35)
    # #
    # rdm_terms = (
    #     np.einsum("li,kj->klji", np.eye(n), np.eye(n), optimize=True)
    #     - np.einsum("ki,lj->klji", np.eye(n), np.eye(n), optimize=True)
    #     + eye_dm1
    #     - np.transpose(eye_dm1, axes=(0, 1, 3, 2))
    #     + dm1_eye
    #     - np.transpose(dm1_eye, axes=(0, 1, 3, 2))
    #     + dm2
    # )
    # #
    # \delta_{i l} \delta_{j k} -\delta_{i k} \delta_{j l}
    _rdm_terms = np.einsum("il,jk->klji", np.eye(n), np.eye(n), optimize=True)
    _rdm_terms -= np.einsum("ik,jl->klji", np.eye(n), np.eye(n), optimize=True)
    # + \delta_{i k} \left\{a^\dagger_{j} a_{l}\right\}
    # - \delta_{i l} \left\{a^\dagger_{j} a_{k}\right\}
    _rdm_terms += np.einsum("ik,jl->klji", np.eye(n), _dm1, optimize=True)
    _rdm_terms -= np.einsum("il,jk->klji", np.eye(n), _dm1, optimize=True)
    # + \delta_{j l} \left\{a^\dagger_{i} a_{k}\right\}
    # - \delta_{j k} \left\{a^\dagger_{i} a_{l}\right\}
    _rdm_terms += np.einsum("jl,ik->klji", np.eye(n), _dm1, optimize=True)
    _rdm_terms -= np.einsum("jk,il->klji", np.eye(n), _dm1, optimize=True)
    return _rdm_terms


class IntegrandPP:
    r"""Compute adiabatic connection integrand."""
    def __init__(self, method, h0, v0, dh, dv, dm1, dm2):
        self.h_0 = h0
        self.v_0 = v0
        self.dh = dh
        self.dv = dv
        # TODO: Check that method is DEA
        self.dm1 = dm1
        self.dm2 = dm2
        self.method = method
        self.vfunc = np.vectorize(self.eval_integrand) 

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
        pp = self.method(h, v, self.dm1, self.dm2)

        if singlets:
            w, c = pp.solve_dense(tol=tol, mode=gevps, mult=1)
            metric = pp._rhs1
        else:
            w, c = pp.solve_dense(tol=tol, mode=gevps, mult=3)
            metric = pp._rhs3
        
        cv_p = picknonzeroeigs(w, c)[1]
        norm = np.dot(cv_p, np.dot(metric, cv_p.T))
        diag_n = np.diag(norm)
        idx = np.where(diag_n > 0)[0]  # Remove eigenvalues with negative norm
        sqr_n = np.sqrt(diag_n[idx])
        c = (cv_p[idx].T / sqr_n).T

        # Compute transition RDMs energy contribution (eq. 29)
        metric = metric.reshape(k, k, k, k)
        tdtd_ab = eval_alphadependent_2dmterms(self.dm1, c, metric)
        pp_rdm2 = np.zeros((m,m,m,m), dtype=self.dm2.dtype)
        if singlets:
            pp_rdm2[:k, k:, :k, k:] = 0.5 * tdtd_ab
            pp_rdm2[k:, :k, k:, :k] = 0.5 * tdtd_ab
        else:
            # 30
            pp_rdm2[:k, k:, :k, k:] += 0.5 * tdtd_ab
            pp_rdm2[k:, :k, k:, :k] += 0.5 * tdtd_ab
            # 31
            pp_rdm2[:k, :k, :k, :k] = tdtd_ab
            pp_rdm2[k:, k:, k:, k:] = tdtd_ab
        result = 0.5 * np.einsum("pqrs,pqrs", self.dv, pp_rdm2, optimize=True)

        if not _dm1ac:
            # # Evaluate ERPA 1RDM^alpha
            # nparts = np.trace(self.dm1)
            # hh_rdm1 = np.einsum('pqrq->pr', pp_rdm2) / (nparts-1)
            # result += np.einsum('pq,pq->', self.dh, hh_rdm1)
            raise NotImplementedError("1RDM not implemented yet")
        
        return result 


class DEA2SA(EOMDEA2):
    r"""
    EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a^\dagger_i  a^\dagger_{\bar{j}} \mp a^\dagger_{\bar{i}} a^\dagger_j)}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a_k  a_{\bar{l}} \mp a_{\bar{k}} a_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| a_k  a_{\bar{l}} \mp a_{\bar{k}} a_l \hat{Q} \middle| \Psi^{(N)}_0 \right>

    """
    def __init__(self, h, v, dm1, dm2):
        super().__init__(h, v, dm1, dm2)
        self._k = self._n // 2
        self._lhs_sb = self._get_lhs_spinblocks()
        self._rhs_sb = self._get_rhs_spinblocks()
        self._lhs1 = self._compute_lhs_1()
        self._rhs1 = self._compute_rhs_1()
        self._lhs3 = self._compute_lhs_30()
        self._rhs3 = self._compute_rhs_30()
    

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
    
    def _get_lhs_spinblocks(self):
        lhs = self._lhs.reshape(self._n, self._n, self._n, self._n)
        A_abab = lhs[:self._k, self._k:, :self._k, self._k:]
        A_baab = lhs[self._k:, :self._k, :self._k, self._k:]
        A_abba = lhs[:self._k, self._k:, self._k:, :self._k]    
        A_baba = lhs[self._k:, :self._k, self._k:, :self._k]
        return (A_abab, A_baab, A_abba, A_baba)

    def _get_rhs_spinblocks(self):
        rhs = self._rhs.reshape(self._n, self._n, self._n, self._n)
        M_abab = rhs[:self._k, self._k:, :self._k, self._k:]
        M_baab = rhs[self._k:, :self._k, :self._k, self._k:]
        M_abba = rhs[:self._k, self._k:, self._k:, :self._k]    
        M_baba = rhs[self._k:, :self._k, self._k:, :self._k]
        return (M_abab, M_baab, M_abba, M_baba)
    
    def _compute_lhs_1(self):
        A_abab, A_baab, A_abba, A_baba = self._lhs_sb
        A = A_abab - A_baab - A_abba + A_baba
        return 0.5 * A.reshape(self._k**2, self._k**2)
    
    def _compute_lhs_30(self):
        A_abab, A_baab, A_abba, A_baba = self._lhs_sb
        A = A_abab + A_baab + A_abba + A_baba 
        return 0.5 * A.reshape(self._k**2, self._k**2)

    def _compute_rhs_1(self):
        M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
        M = M_abab - M_baab - M_abba + M_baba
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def _compute_rhs_30(self):
        M_abab, M_baab, M_abba, M_baba = self._rhs_sb 
        M = M_abab + M_baab + M_abba + M_baba
        return 0.5 * M.reshape(self._k**2, self._k**2)
    
    def solve_dense(self, tol=1.0e-10, mode="nonsymm", err="ignore", mult=1):
        r"""
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10
        mode : str, optional
            Specifies whether a symmetric or nonsymmetric method is used to solve the GEVP.
            Default is `nonsymm` in which the inverse of the right hand side matrix is taken.
        err : ("warn" | "ignore" | "raise")
            What to do if a divide-by-zero floating point error is raised.
            Default behavior is to ignore divide by zero errors.
        mult : int, optional
            State multiplicity. Singlet (1) ot triplet (3) states. Default: 1

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        modes = {'nonsymm': nonsymmetric, 'symm': svd_lowdin, 'qtrunc': eig_pinv}
        if not isinstance(tol, float):
            raise TypeError("Argument tol must be a float")        
        try:
            _solver = modes[mode]
        except KeyError:
            print(
                "Invalid mode parameter. Valid options are nonsymm, symm or qtrunc."
            )
        if mult == 1:
            lhs = self._lhs1
            rhs = self._rhs1
        elif mult == 3:
            lhs = self._lhs3
            rhs = self._rhs3
        else:
            raise ValueError("Invalid state multiplicity. Valid options are 1 or 3.")
        w, v = _solver(lhs, rhs, tol=tol, err=err)
        return np.real(w), np.real(v)
