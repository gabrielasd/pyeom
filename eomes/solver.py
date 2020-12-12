"""
Dense solver module.

"""


import numpy as np
from scipy.linalg import eig, svd


__all__ = [
    "dense",
]


def dense(lhs, rhs, tol=1.0e-7, orthog="symmetric", *args, **kwargs):
    """
        Solve the EOM eigenvalue system.

        Parameters
        ----------
        tol : float, optional
            Tolerance for small singular values. Default: 1.0e-10
        orthog : str, optional
            Matrix orthogonalization method. Default is symmetric orthogonalization
            in which the inverse square root of the right hand side matrix is taken.
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
    U, s, V = svd(rhs)

    if orthog == "symmetric":
        # Apply inverse square root to eigvals of RHS
        s = s ** (-0.5)
        # Check singular value threshold
        s[s >= 1 / tol] = 0.0
        # Transform back to RHS^(-0.5)
        S_inv = np.diag(s)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
        # Apply RHS^-0.5 * LHS * RHS^-0.5
        A = np.dot(rhs_inv, np.dot(lhs, rhs_inv))
        # Run scipy `linalg.eig` eigenvalue solver
        w, v = eig(A, *args, **kwargs)
        # Transform back to original eigenvector matrix
        v = np.dot(rhs_inv, v)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)
    elif orthog == "asymmetric":
        # Check singular value threshold
        s = s ** (-1)
        s[s >= 1 / tol] = 0.0
        # S^(-1)
        S_inv = np.diag(s)
        # rhs^(-1)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))
        # Apply RHS^-1 * LHS
        A = np.dot(rhs_inv, lhs)
        # Run scipy `linalg.eig` eigenvalue solver
        w, v = eig(A, *args, **kwargs)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)
    else:
        raise ValueError(
            "Invalid orthogonalization parameter. Valid options are "
            "symmetric or asymmetric."
        )
