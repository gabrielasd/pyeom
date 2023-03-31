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

from eomee.doubleionization import EOMDIP, EOMDIP2
from eomee.tools import pickpositiveeig, spinize, from_unrestricted
from eomee.solver import nonsymmetric, svd_lowdin, eig_pinv
from eomee.base import EOMState


__all__ = [
    "DIPSA",
]


class DIPSA(EOMDIP):
    r"""
    Excitation EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} \mp a_{\bar{i}} a_j)}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

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
        
        # Compute ERPA correction energy
        # Nonlinear term (eq. 19 integrand)        
        function = Integrandhh(cls, h_0, v_0, dh, dv, dm1, dm2)
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
        if not dm1ac:
            linear = 0.
        else:
            linear = _hherpa_linearterms(dh, dm1)
        ecorr = linear + alphadep

        output = {}
        output["ecorr"] = ecorr
        output["linear"] = linear
        output["error"] = None

        return output


def _hherpa_linearterms(_dh, _dm1):
    # dh * \gamma
    return np.einsum("pq,pq", _dh, _dm1, optimize=True)


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


class DIP2SA(EOMDIP2):
    r"""
    Excitation EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} \mp a_{\bar{i}} a_j)}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l \hat{Q} \middle| \Psi^{(N)}_0 \right>

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


class DIPSAidxs(EOMState):
    r"""
    Excitation EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} (a_i  a_{\bar{j}} \mp a_{\bar{i}} a_j)}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l , \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[a^\dagger_k  a^\dagger_{\bar{l}} \mp a^\dagger_{\bar{k}} a^\dagger_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

    """
    def __init__(self, h, v, dm1, dm2, mult=1):
        if not isinstance(mult, int):
            raise TypeError("Argument mult must be an integer")
        if mult not in [1, 3]:
            raise ValueError("Invalid state multiplicity. Valid options are 1 or 3.")
        self._m = mult
        self._k = h.shape[0] // 2
        super().__init__(h, v, dm1, dm2)

    @property
    def neigs(self):
        r""" """
        return self._k ** 2
    
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

    def _lhs_ab(self):
        k = self._k
        M = np.zeros((k,k,k,k), dtype=self._h.dtype)
        for k_i in range(k):
            for l_i in range(k):
                for j_i in range(k):
                    for i_i in range(k):
                        params = (k, self._h, self._v, self._dm1, self._dm2, M, (k_i, l_i, j_i, i_i), 'ab')
                        M += _factor2terms_lhs(*params)
                        M += _factor1terms_lhs(*params) 
        
        M -= self._v[:k, k:, :k, k:]    # v_abab
        M += M.transpose(1, 0, 3, 2)
        return M

    def _lhs_aa(self):
        k = self._k
        M = np.zeros((k,k,k,k), dtype=self._h.dtype)
        for k_i in range(k):
            for l_i in range(k):
                for j_i in range(k):
                    for i_i in range(k):
                        params = (k, self._h, self._v, self._dm1, self._dm2, M, (k_i, l_i, j_i, i_i), 'aa')
                        # M = _factor2terms_lhs(*params)
                        M = _factor1terms_lhs(*params)     
        M -= self._v[:k, :k, :k, :k]    # v_aaaa
        M += M.transpose(1, 0, 3, 2)
        return M
    
    def _rhs_ab(self):
        k = self._k
        B = np.zeros((k,k,k,k), dtype=self._h.dtype)
        for k_i in range(k):
            for l_i in range(k):
                for j_i in range(k):
                    for i_i in range(k):
                        B = _terms_rhs(k, self._dm1, B, (k_i, l_i, j_i, i_i), 'ab')
        return B

    def _compute_lhs(self):
        r"""
        Compute

        .. math::

            A_{klji} = 
        """
        A_abab = self._lhs_ab()
        if self._m == 1:
            A = A_abab - A_abab.transpose(0, 1, 3, 2)
        else:
            A = A_abab + A_abab.transpose(0, 1, 3, 2)
        return A.reshape(self._k**2, self._k**2)
    
    def _compute_rhs(self):
        r"""
        Compute :math:`M_{klji} = `
        """
        M_abab = self._rhs_ab()
        if self._m == 1:
            M = M_abab - M_abab.transpose(0, 1, 3, 2)
        else:
            M = M_abab + M_abab.transpose(0, 1, 3, 2)
        return M.reshape(self._k**2, self._k**2)


def _factor2terms_lhs(nb, h, v, d1, d2, M, idxs, blok):
    k_mo, l_mo, j_mo, i_mo = idxs
    I = np.eye(2*nb, dtype=h.dtype)

    if blok == 'ab':
        k_idx, j_idx = k_mo, j_mo
        l_idx, i_idx = l_mo + nb, i_mo + nb
    elif blok == 'aa':
        k_idx, j_idx = k_mo, j_mo
        l_idx, i_idx = l_mo, i_mo
    else:
        raise ValueError("Invalid block. Valid options are 'ab' or 'aa'")

    M[k_mo, l_mo, j_mo, i_mo] += (
        h[i_idx, l_idx] * I[j_idx, k_idx] -
        h[i_idx, l_idx] * d1[j_idx, k_idx] +
        h[i_idx, k_idx] * d1[j_idx, l_idx] -
        h[i_idx, k_idx] * I[i_idx, j_idx]
    )

    M[k_mo, l_mo, j_mo, i_mo] += (
        d1[l_idx, :] @ h[:, j_idx] * I[k_idx, i_idx] -
        d1[k_idx, :] @ h[:, j_idx] * I[l_idx, i_idx]
    )

    M[k_mo, l_mo, j_mo, i_mo] += (
        np.sum(v[i_idx, :, :, k_idx] * d1 * I[l_idx, j_idx]) -
        np.sum(v[i_idx, :, :, l_idx] * d1 * I[k_idx, j_idx])
    )

    M[k_mo, l_mo, j_mo, i_mo] += (
        np.sum(v[j_idx, :, :, k_idx] * d2[:, l_idx, :,  i_idx]) -
        np.sum(v[j_idx, :, :, l_idx] * d2[:, k_idx, :,  i_idx])
    )
    return 2 * M


def _factor1terms_lhs(nb, h, v, d1, d2, M, idxs, blok):
    k_mo, l_mo, j_mo, i_mo = idxs
    I = np.eye(2*nb, dtype=d1.dtype)

    if blok == 'ab':
        k_idx, j_idx = k_mo, j_mo
        l_idx, i_idx = l_mo + nb, i_mo + nb
    elif blok == 'aa':
        k_idx, j_idx = k_mo, j_mo
        l_idx, i_idx = l_mo, i_mo
    else:
        raise ValueError("Invalid block. Valid options are 'ab' or 'aa'")
    
    M[k_mo, l_mo, j_mo, i_mo] += (
        v[j_idx, i_idx, l_idx, :] @ d1[k_idx, :] -
        v[j_idx, i_idx, k_idx, :] @ d1[l_idx, :] +
        2 *  v[:, j_idx, k_idx, l_idx] @ d1[:, i_idx]
    )
    M[k_mo, l_mo, j_mo, i_mo] += (
        np.sum(v[:, j_idx, :, :] * d2[:, l_idx, :, :] * I[k_idx, i_idx]) -
        np.sum(v[:, j_idx, :, :] * d2[:, k_idx, :, :] * I[l_idx, i_idx])
    )
    return M


def _terms_rhs(nb, d1, B, idxs, blok):
    k_mo, l_mo, j_mo, i_mo = idxs
    I = np.eye(2*nb, dtype=d1.dtype)

    if blok == 'ab':
        k_idx, j_idx = k_mo, j_mo
        l_idx, i_idx = l_mo + nb, i_mo + nb
    elif blok == 'aa':
        k_idx, j_idx = k_mo, j_mo
        l_idx, i_idx = l_mo, i_mo
    else:
        raise ValueError("Invalid block. Valid options are 'ab' or 'aa'")
   
    B[k_mo, l_mo, j_mo, i_mo] += (
        I[i_idx, k_idx] * I[j_idx, l_idx] -
        I[i_idx, l_idx] * I[j_idx, k_idx] -
        I[i_idx, k_idx] * d1[j_idx, l_idx] +
        I[i_idx, l_idx] * d1[j_idx, k_idx] -
        I[j_idx, l_idx] * d1[i_idx, k_idx] +
        I[j_idx, k_idx] * d1[i_idx, l_idx]
    )
    return B