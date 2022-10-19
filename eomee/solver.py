r"""Dense solvers for Generalized Eigenvalue Problems."""
import numpy as np
from scipy.linalg import eig, svd, pinv, eigh
from numpy.lib.scimath import sqrt as csqrt


def nonsymmetric(lhs, rhs, tol=1.0e-10, err="ignore"):
    r"""
    Solve the EOM eigenvalue system.

    Given the generalized eigenvalue problem:

    .. math::
        \mathbf{A} x = w \mathbf{B} x

    transform it into an estandard one with:

    .. math::
        \mathbf{B}^{-1} \mathbf{A} x = w x

    Parameters
    ----------
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
    # print(f'Using nonsymm solver.')
    # Invert RHS matrix
    U, s, V = svd(rhs)
    with np.errstate(divide=err):
        s = s ** (-1)
    s[s >= 1 / tol] = 0.0
    S_inv = np.diag(s)  # S^(-1)
    rhs_inv = np.dot(U, np.dot(S_inv, V))   # rhs^(-1)
    A = np.dot(rhs_inv, lhs)    # Apply RHS^-1 * LHS
    # Run scipy `linalg.eig` eigenvalue solver
    w, v = eig(A)
    # if np.any(np.iscomplex(w)):
    #     print(f'Warning: complex eigenvalues found.')
    # Return w (eigenvalues)
    #    and v (eigenvector column matrix -- so transpose it!)
    return w, v.T


def svd_lowdin(lhs, rhs, tol=1.0e-10, err="ignore"):
    r"""
    Solve the EOM eigenvalue system with symmetric orthogonalization.

    Given the generalized eigenvalue problem:

    .. math::
        \mathbf{A} x = w \mathbf{B} x

    transform it into an estandard one with Lowdin orthogonalization:

    .. math::
        \mathbf{B}^{-1/2} \mathbf{A} \mathbf{B}^{-1/2} \mathbf{B}^{1/2} x \\
        = w \mathbf{B}^{-1/2} \mathbf{B} \mathbf{B}^{-1/2} \mathbf{B}^{1/2} x \\
        \mathbf{A}^{'} x^{'} = w x^{'}

    Parameters
    ----------
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
    # Invert RHS matrix
    U, s, V = svd(rhs)
    # Apply inverse square root to eigvals of RHS
    with np.errstate(divide=err):
        s = s ** (-0.5)
    # Check singular value threshold
    s[s >= 1 / tol] = 0.0
    # Transform back to RHS^(-0.5)
    S_inv = np.diag(s)
    rhs_inv = np.dot(U, np.dot(S_inv, V))
    # Apply RHS^-0.5 * LHS * RHS^-0.5
    A = np.dot(rhs_inv, np.dot(lhs, rhs_inv))
    # Run scipy `linalg.eig` eigenvalue solver
    w, v = eig(A)
    # Transform back to original eigenvector matrix
    v = np.dot(rhs_inv, v)
    # Return w (eigenvalues)
    #    and v (eigenvector column matrix -- so transpose it!)
    return w, v.T


def lowdin(lhs, rhs, tol=1.0e-10, err="ignore"):
    r"""
    Solve the EOM eigenvalue system with symmetric orthogonalization.

    Parameters
    ----------
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
    # assert np.allclose(rhs, rhs.T)
    # assert np.allclose(lhs, lhs.T)
    w, v = eigh(rhs)
    w[w < 0.] = 0.
    with np.errstate(divide=err):
        inv_sqrt_w = w ** (-0.5)
    inv_sqrt_w[inv_sqrt_w > 1./tol] = 0.
    ort_m = np.dot(v, np.dot(np.diag(inv_sqrt_w), v.T))
    Hm = np.dot(ort_m.T, np.dot(lhs, ort_m))
    # assert np.allclose(Hm, Hm.T)
    w, v = eigh(Hm)
    v = np.dot(ort_m, v)
    return w, v.T


def lowdin_complex(lhs, rhs, tol=1.0e-10):
    r"""
    Solve the EOM eigenvalue system with symmetric orthogonalization.

    Given the generalized eigenvalue problem:

    .. math::
        \mathbf{A} x = w \mathbf{B} x

    where :math:`\mathbf{B}` is a symmetric and indefinite matrix, transform it 
    into an estandard eigenvalue problem with symmetric orthogonalization. 
    The orthogonalization matrix :math:`\mathbf{B}^{-1/2}` will be complex valued.

    Parameters
    ----------
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
    # assert np.allclose(rhs, rhs.T)
    # assert np.allclose(lhs, lhs.T)
    w, v = eigh(rhs)
    sqrt_w = csqrt(w)
    inv_sqrt_w = pinv(np.diag(sqrt_w), rcond=tol)
    ort_m = np.dot(v, np.dot(inv_sqrt_w, v.T))
    Hm = np.dot(ort_m.T, np.dot(lhs, ort_m))
    # assert np.allclose(Hm, Hm.T)
    w, v= eigh(Hm)
    v = np.dot(ort_m, v)
    if np.any(np.iscomplex(w)):
        print(f'Warning: complex eigenvalues found.')
    return w, v.T


def safe_eig(h, s, tol=1e-10, err=None):
    from functools import reduce
    seig, t = eigh(s)
    try:
        w, v = eig(h, s)
    except np.linalg.LinAlgError:
        idx = np.abs(seig) >= tol
        t = t[:,idx] * (1/seig[idx])
        if t.size > 0:
            heff = reduce(np.dot, (t.T.conj(), h, t))
            w, v = eig(heff)
            v = np.dot(t, v)
        else:
            w = np.zeros((0,))
            v = t
    return w, v.T


def nonsymmetric2(lhs, rhs, eps=1.0e-2, tol=1.0e-10, err=None):
    # print(f'Using nonsymm2 solver, eps {eps}')
    def zeroing_rows_and_cols(h, s, lindep):
        seig = np.diag(s)
        idx = np.abs(seig) < lindep
        t = np.ones_like(seig)
        t[idx] = 0.
        T = np.diag(t)
        A = T@h@T
        B = T@s@T
        return A, B
    lhs, rhs = zeroing_rows_and_cols(lhs, rhs, eps)
    S_inv = pinv(rhs, rcond=tol)
    A = np.dot(S_inv, lhs)
    # Run scipy `linalg.eig` eigenvalue solver
    w, v = eig(A)
    # if np.any(np.iscomplex(w)):
    #     print(f'Warning: complex eigenvalues found.')
    # Return w (eigenvalues)
    #    and v (eigenvector column matrix -- so transpose it!)
    return w, v.T


def prunned(lhs, rhs, eps=1.0e-2, tol=1.0e-10, err=None):
    # print(f'Using prunned solver, eps {eps}')
    def zeroing_rows_and_cols(h, s, lindep):
        seig = np.diag(s)
        # seig = eigh(s)[0]
        idx = np.abs(seig) < lindep
        t = np.ones_like(seig)
        t[idx] = 0.
        T = np.diag(t)
        A = T@h@T
        B = T@s@T
        return A, B
    lhs, rhs = zeroing_rows_and_cols(lhs, rhs, eps)
    s, U = np.linalg.eigh(rhs)
    idx = np.where(np.abs(s) > tol)[0]
    B = np.dot(U[:, idx].T, np.dot(rhs, U[:, idx]))
    A = np.dot(U[:, idx].T, np.dot(lhs, U[:, idx]))
    w, v = eig(A, B)
    v = np.dot(U[:, idx], v)
    if np.any(np.iscomplex(w)):
        print(f'Warning: complex eigenvalues found.')
    # Return w (eigenvalues)
    #    and v (eigenvector column matrix -- so transpose it!)
    return w, v.T
