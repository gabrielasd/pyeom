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

from eomee.doubleionization import EOMDEA, EOMDEA2
from eomee.tools import pickpositiveeig, spinize, from_unrestricted
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
