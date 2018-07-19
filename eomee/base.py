"""
Equations-of-motion state base class.

"""


from abc import ABCMeta, abstractmethod, abstractproperty

from scipy.linalg import eig


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

    def solve(self, eigvals=None):
        """
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        eigvals : (2,), optional
            Range of eigenpairs to find, by ascending eigenvalue, denoted (lo, hi).
            E.g., (0, 3) would find the 3 eigenpairs with the lowest eigenvalues.
            If `None`, all eigenpairs are found (default).

        Returns
        -------
        w : np.ndarray((m,))
            Eigenvalue array (m eigenvalues).
        v : np.ndarray((m, n))
            Eigenvector matrix (m eigenvectors).

        """
        # Run scipy `linalg.eig` eigenvalue solver
        w, v = eig(self._lhs, b=self._rhs)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return w, v.T

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
