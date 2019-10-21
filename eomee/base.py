"""
Equations-of-motion state base class.

"""


from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.linalg import eig, svd
from scipy.sparse.linalg import eigs


__all__ = [
    'EOMState',
]


class EOMState(metaclass=ABCMeta):
    """
    Equations-of-motion state abstract base class.

    Overwrite neigs, _compute_lhs, _compute_rhs.

    """

    def __init__(self, h, v, dm1, dm2):
        """
        Initialize an EOMState instance.

        Parameters
        ----------
        h : np.ndarray((n, n))
            1-particle integral array.
        v : np.ndarray((n, n, n, n))
            2-particle integral array.
        dm1 : np.ndarray((n, n))
            1-particle reduced density matrix.
        dm2 : np.ndarray((n, n, n, n))
            2-particle reduced density matrix.

        """
        # Basic system attributes
        if not (isinstance(h, np.ndarray) and h.ndim == 2):
            raise ValueError('One-particle integrals should be a 2-dimensional '
                             'numpy array')
        if not (isinstance(v, np.ndarray) and v.ndim == 4):
            raise ValueError('Two-particle integrals should be a 4-dimensional '
                             'numpy array')
        if not (isinstance(dm1, np.ndarray) and dm1.ndim == 2):
            raise ValueError('One-particle reduced density matrix should be a '
                             '2-dimensional numpy array')
        if not (isinstance(dm2, np.ndarray) and dm2.ndim == 4):
            raise ValueError('Two-particle reduced density matrix should be a '
                             '2-dimensional numpy array')
        self._n = h.shape[0]
        self._h = h
        self._v = v
        self._dm1 = dm1
        self._dm2 = dm2
        # Compute arrays for generalized eigenvalue problem
        # _compute_{lhs|rhs} are overwritten by EOMState subclasses
        self._lhs = self._compute_lhs()
        self._rhs = self._compute_rhs()

    @property
    def n(self):
        """
        Return the number of orbital basis functions.

        Returns
        -------
        n : int
            Number of orbital basis functions.

        """
        return self._n

    @abstractproperty
    def neigs(self):
        """
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        raise NotImplementedError("Subclasses must overwrite this property")

    @property
    def h(self):
        """
        Return the 1-particle integral array.

        Returns
        -------
        h : np.ndarray((n, n))
            1-particle integral array.

        """
        return self._h

    @property
    def v(self):
        """
        Return the 2-particle integral array.

        Returns
        -------
        v : np.ndarray((n, n, n, n))
            2-particle integral array.

        """
        return self._v

    @property
    def dm1(self):
        """
        Return the 1-particle reduced density matrix.

        Returns
        -------
        dm1 : np.ndarray((n, n))
            1-particle reduced density matrix.

        """
        return self._dm1

    @property
    def dm2(self):
        """
        Return the 2-particle reduced density matrix.

        Returns
        -------
        dm2 : np.ndarray((n, n, n, n))
            2-particle reduced density matrix.

        """
        return self._dm2

    @property
    def lhs(self):
        """
        Return the left-hand-side operator matrix.

        Returns
        -------
        lhs : np.ndarray((n, n))
            Left-hand-side operator matrix.

        """
        return self._lhs

    @property
    def rhs(self):
        """
        Return the right-hand-side operator matrix.

        Returns
        -------
        rhs : np.ndarray((n, n))
            Right-hand-side operator matrix.

        """
        return self._rhs

    def solve_dense(self, tol=1.0e-10, *args, **kwargs):
        """
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        if not isinstance(tol, float):
            raise TypeError('Argument tol must be a float')
        # Invert RHS matrix
        # RHS matrix SVD
        U, s, V = svd(self._rhs)
        # Check singular value threshold
        s = s ** (-1)
        s[s >= 1 / tol] = 0.
        # S^(-1)
        S_inv = np.diag(s)
        # rhs^(-1)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
        # Apply RHS^-1 * LHS
        A = np.dot(rhs_inv, self._lhs)
        # Run scipy `linalg.eig` eigenvalue solver
        w, v = eig(A, *args, **kwargs)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)

    def solve_sparse(self, eigvals=6, tol=1.0e-10, *args, **kwargs):
        """
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        eigvals : int, optional
            Number of eigenpairs to find. Must be smaller than N-1.
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        if not isinstance(tol, float):
            raise TypeError('Argument tol must be a float')
        # Invert RHS matrix
        # RHS matrix SVD
        U, s, V = svd(self._rhs)
        # Check singular value threshold
        s = s ** (-1)
        s[s >= 1 / tol] = 0.
        # S^(-1)
        S_inv = np.diag(s)
        # rhs^(-1)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
        # Apply RHS^-1 * LHS
        A = np.dot(rhs_inv, self._lhs)
        # Run scipy `linalg.eigs` eigenvalue solver
        w, v = eigs(A, k=eigvals, which='SR', *args, **kwargs)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)

    @abstractmethod
    def _compute_lhs(self):
        """
        Compute the left-hand-side operator matrix.

        """
        raise NotImplementedError("Subclasses must overwrite this method")

    @abstractmethod
    def _compute_rhs(self):
        """
        Compute the right-hand-side operator matrix.

        """
        raise NotImplementedError("Subclasses must overwrite this method")
