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

r"""Equations-of-motion state base class."""


from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from scipy.linalg import eig, svd
from scipy.sparse.linalg import eigs

from .solver import nonsymmetric, svd_lowdin

from .tools import antisymmetrize


__all__ = [
    "EOMBase",
]


class EOMState(metaclass=ABCMeta):
    r"""
    Equations-of-motion state abstract base class.

    Overwrite neigs, _compute_lhs, _compute_rhs.

    """

    def __init__(self, h, v, dm1, dm2):
        r"""
        Initialize an EOMState instance.

        Parameters
        ----------
        h : np.ndarray((n, n))
            1-particle integral array. :math:`n` corresponds to the number of spin-orbitals.
        v : np.ndarray((n, n, n, n))
            2-particle integral array in physicist's notation. :math:`n` corresponds to the number
            of spin-orbitals.
        dm1 : np.ndarray((n, n))
            1-particle reduced density matrix. A spin resolved 1-RDM is used.
        dm2 : np.ndarray((n, n, n, n))
            2-particle reduced density matrix. A spin resolved 2-RDM is used.

        """
        self.verify_integrals(h, v)
        self.verify_rdms(dm1, dm2)
        if not h.shape[0] == dm2.shape[0]:
            raise ValueError(
                "Electron integrals and density matrices must have equal number of spin-orbitals"
            )
        # Basic system attributes
        self._n = h.shape[0]
        self._h = h
        self._v = antisymmetrize(v)
        self._dm1 = dm1
        self._dm2 = dm2
        # Compute arrays for generalized eigenvalue problem
        # _compute_{lhs|rhs} are overwritten by EOMState subclasses
        self._lhs = self._compute_lhs()
        self._rhs = self._compute_rhs()

    @property
    def n(self):
        r"""
        Return the number of orbital basis functions.

        Returns
        -------
        n : int
            Number of orbital basis functions.

        """
        return self._n

    @abstractproperty
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        raise NotImplementedError("Subclasses must overwrite this property")

    @property
    def h(self):
        r"""
        Return the 1-particle integral array.

        Returns
        -------
        h : np.ndarray((n, n))
            1-particle integral array.

        """
        return self._h

    @property
    def v(self):
        r"""
        Return the 2-particle integral array.

        Returns
        -------
        v : np.ndarray((n, n, n, n))
            2-particle integral array.

        """
        return self._v

    @property
    def dm1(self):
        r"""
        Return the 1-particle reduced density matrix.

        Returns
        -------
        dm1 : np.ndarray((n, n))
            1-particle reduced density matrix.

        """
        return self._dm1

    @property
    def dm2(self):
        r"""
        Return the 2-particle reduced density matrix.

        Returns
        -------
        dm2 : np.ndarray((n, n, n, n))
            2-particle reduced density matrix.

        """
        return self._dm2

    @property
    def lhs(self):
        r"""
        Return the left-hand-side operator matrix.

        Returns
        -------
        lhs : np.ndarray((n, n))
            Left-hand-side operator matrix.

        """
        return self._lhs

    @property
    def rhs(self):
        r"""
        Return the right-hand-side operator matrix.

        Returns
        -------
        rhs : np.ndarray((n, n))
            Right-hand-side operator matrix.

        """
        return self._rhs

    def verify_integrals(self, h, v):
        """Check the type and shape of the 1- and 2-electron integrals.

        Parameters
        ----------
        h : np.ndarray((n, n))
            1-particle integral array.
        v : np.ndarray((n, n, n, n))
            2-particle integral array.

        Raises
        ------
        TypeError
            The 1- and 2-electron integrals must be numpy arrays.
        ValueError
            The one-electron integrals must be a square matrix.
            Two-electron integrals must have four equivalent dimensions.
            The number of spin-orbitals between electron integrals don't match.
        """
        if not (isinstance(h, np.ndarray) and isinstance(v, np.ndarray)):
            raise TypeError("The 1- and 2-electron integrals must be numpy arrays.")
        if not (h.ndim == 2 and h.shape[0] == h.shape[1]):
            raise ValueError("The one-electron integrals must be a square matrix.")
        if not (v.ndim == 4 and v.shape == (v.shape[0],) * 4):
            raise ValueError("Two-electron integrals must have four equivalent dimensions.")
        if not h.shape[0] == v.shape[0]:
            raise ValueError("The number of spin-orbitals between electron integrals don't match.")

    def verify_rdms(self, dm1, dm2):
        """Check the type, shape and symmetry of the 1- and 2-particle reduced density matrices.

        Parameters
        ----------
        dm1 : np.ndarray((n, n))
            1-particle reduced density matrix.
        dm2 : np.ndarray((n, n, n, n))
            2-particle reduced density matrix.

        Raises
        ------
        TypeError
            The 1- and 2-particle reduced density matrices must be numpy arrays.
        ValueError
            The 1-particle reduced density matrix must be a square matrix.
            The 2-particle reduced density matrix must have four equivalent dimensions.
            The number of spin-orbitals between density matrices don't match.
            One/Two-particle density matrix does not satisfy the symmetric permutations.
            2-particle density matrix does not satisfy the asymmetric permutations.
        """
        if not (isinstance(dm1, np.ndarray) and isinstance(dm2, np.ndarray)):
            raise TypeError("The 1- and 2-particle reduced density matrices must be numpy arrays.")
        if not (dm1.ndim == 2 and dm1.shape[0] == dm1.shape[1]):
            raise ValueError("The 1-particle reduced density matrix must be a square matrix.")
        if not (dm2.ndim == 4 and dm2.shape == (dm2.shape[0],) * 4):
            raise ValueError(
                "The 2-particle reduced density matrix must have four equivalent dimensions."
            )
        if not dm1.shape[0] == dm2.shape[0]:
            raise ValueError("The number of spinorbitals between density matrices don't match.")
        # Symmetric permutations:
        onedm_symm = np.allclose(dm1, dm1.T)
        twodm_symm = np.all(
            [
                np.allclose(dm2, dm2.transpose(2, 3, 0, 1)),
                np.allclose(dm2, dm2.transpose(1, 0, 3, 2)),
            ]
        )
        symmetries = {"1": onedm_symm, "2": twodm_symm}
        for number, symm in symmetries.items():
            if not symm:
                raise ValueError(
                    f"{number}-particle density matrix does not satisfy the symmetric permutations."
                )
        # Two-reduced density matrix antisymmetric permutations:
        twodm_asymm = np.all(
            [
                np.allclose(dm2, -dm2.transpose(0, 1, 3, 2)),
                np.allclose(dm2, -dm2.transpose(1, 0, 2, 3)),
            ]
        )
        if not twodm_asymm:
            raise ValueError(
                "2-particle density matrix does not satisfy the asymmetric permutations."
            )

    def solve_dense(self, tol=1.0e-10, mode="nonsymm", err="ignore", *args, **kwargs):
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

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        modes = {'nonsymm': nonsymmetric, 'symm': svd_lowdin}
        if not isinstance(tol, float):
            raise TypeError("Argument tol must be a float")        
        try:
            _solver = modes[mode]
        except KeyError:
            print(
                "Invalid mode parameter. Valid options are nonsymm or symm."
            )
        
        w, v = _solver(self._lhs, self._rhs, tol=tol, err=err)
        return np.real(w), np.real(v)

    def solve_sparse(self, eigvals=6, tol=1.0e-10, err="ignore", *args, **kwargs):
        r"""
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        eigvals : int, optional
            Number of eigenpairs to find. Must be smaller than N-1.
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10
        err : ("warn" | "ignore" | "raise")
            What to do if a divide-by-zero floating point error is raised.
            Default behavior is to ignore divide by zero errors.

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        if not isinstance(tol, float):
            raise TypeError("Argument tol must be a float")
        # Invert RHS matrix
        # RHS matrix SVD
        U, s, V = svd(self._rhs)
        # Check singular value threshold
        with np.errstate(divide=err):
            s = s ** (-1)
        s[s >= 1 / tol] = 0.0
        # S^(-1)
        S_inv = np.diag(s)
        # rhs^(-1)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
        # Apply RHS^-1 * LHS
        A = np.dot(rhs_inv, self._lhs)
        # Run scipy `linalg.eigs` eigenvalue solver
        w, v = eigs(A, k=eigvals, which="SR", *args, **kwargs)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)

    @abstractmethod
    def _compute_lhs(self):
        r"""
        Compute the left-hand-side operator matrix.

        """
        raise NotImplementedError("Subclasses must overwrite this method")

    @abstractmethod
    def _compute_rhs(self):
        r"""
        Compute the right-hand-side operator matrix.

        """
        raise NotImplementedError("Subclasses must overwrite this method")
