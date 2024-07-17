r"""Dense solvers for Generalized Eigenvalue Problems."""
import numpy as np
from scipy.linalg import eig, svd, pinv, eigh
from numpy.lib.scimath import sqrt as csqrt


INV_THRESHOLD = 1e-7
EIG_THRESHOLD = 1e-2
IMAG_THRESHOLD = 1e-4


def eig_invb(lhs, rhs, tol=1.0e-10, err="ignore"):
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
    # Invert RHS matrix
    U, s, V = svd(rhs)
    with np.errstate(divide=err):
        s = s ** (-1)
    s[s >= 1 / tol] = 0.0
    S_inv = np.diag(s)  # S^(-1)
    rhs_inv = np.dot(U, np.dot(S_inv, V))  # rhs^(-1)
    A = np.dot(rhs_inv, lhs)  # Apply RHS^-1 * LHS
    # Run scipy `linalg.eig` eigenvalue solver
    w, v = eig(A)
    # if len(np.iscomplex(w)) != 0:
    #     print(f'Warning: complex eigenvalues found.')

    # Return w (eigenvalues)
    #    and v (eigenvector column matrix -- so transpose it!)
    return w, v.T


def _zeroing_rows_and_cols(h, s, lindep):
    # HARD CODED
    # lindep = 1.0e-2
    seig = np.diag(s)
    idx = np.abs(seig) < lindep
    t = np.ones_like(seig)
    t[idx] = 0.0
    T = np.diag(t)
    A = T @ h @ T
    B = T @ s @ T
    return A, B


def eig_pinvb(lhs, rhs, tol=1.0e-10):
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

    Returns
    -------
    w : np.ndarray((m,))
        Eigenvalue array (m eigenvalues).
    v : np.ndarray((m, n))
        Eigenvector matrix (m eigenvectors).

    """
    # lhs, rhs = _zeroing_rows_and_cols(lhs, rhs, tol)
    S_inv = pinv(rhs, rcond=tol)
    A = np.dot(S_inv, lhs)
    # Run scipy `linalg.eig` eigenvalue solver
    w, v = eig(A)
    # if np.any(np.iscomplex(w)):
    #     print(f'Warning: complex eigenvalues found.')
    return w, v.T


def _pruneQ(L, R, tol):
    # Fid the eigenvalues of the metric matrix smaller than a tolerance assuming
    # it is a Hermitian matrix.
    s, _U = eigh(R)
    _idx = np.where(np.abs(s) > tol)[0]
    _B = np.dot(_U[:, _idx].T, np.dot(R, _U[:, _idx]))
    _A = np.dot(_U[:, _idx].T, np.dot(L, _U[:, _idx]))
    return (_A, _B, _U, _idx)


def eig_pruneq(lhs, rhs, tol=1.0e-10, err=None):
    # Remove configurations with small amplitudes in the metric matrix from the
    # transition operator.
    A, B, U, idx = _pruneQ(lhs, rhs, tol)
    w, v = eig(A, B)
    # Transform back to original eigenvector matrix
    v = np.dot(U[:, idx], v)
    return w, v.T


def eig_pruneq_pinvb(lhs, rhs, tol=1.0e-10, err=None):
    A, B, U, idx = _pruneQ(lhs, rhs, tol)
    S_inv = pinv(B, rcond=tol)
    A = np.dot(S_inv, A)
    # Run scipy `linalg.eig` eigenvalue solver
    w, v = eig(A)
    # Transform back to original eigenvector matrix
    v = np.dot(U[:, idx], v)
    # if np.any(np.iscomplex(w)):
    #     print(f'Warning: complex eigenvalues found.')
    return w, v.T


def lowdin_svd(lhs, rhs, tol=1.0e-10, err="ignore"):
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
    w[w < 0.0] = 0.0
    with np.errstate(divide=err):
        inv_sqrt_w = w ** (-0.5)
    inv_sqrt_w[inv_sqrt_w > 1.0 / tol] = 0.0
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
    w, v = eigh(Hm)
    v = np.dot(ort_m, v)
    if len(np.iscomplex(w)) != 0:
        print(f"Warning: complex eigenvalues found.")
    return w, v.T


def pick_positive(ev, cv, cutoff):
    r"""
    Remove the GEVP solutions whose eigenvalues are negative as determined by the tolerance tol.
    If complex eigenvalues are found, they will be removed from the final solution set.

    """
    idx = np.where(ev > cutoff ** 2)[0]
    ev, cv = ev[idx], cv[idx]

    # Check solutions with imaginary component
    real_indices = np.where(ev.imag < IMAG_THRESHOLD)[0]
    nimag = len(ev) - len(real_indices)
    if nimag != 0:
        print(
            f"""Warning: {nimag} complex eigenvalues found. 
              These will be removed from the final solution set."""
        )

    # Remove complex eigenvalues and corresponding eigenvectors
    real_ev = ev[real_indices]
    real_cv = cv[real_indices]

    return np.real(real_ev), np.real(real_cv)


def pick_nonzero(ev, cv, cutoff):
    r"""
    Remove the GEVP solutions whose eigenvalues are close to zero as determined by the tolerance tol.
    If complex eigenvalues are found, they will be removed from the final solution set.

    """
    idx = np.where(np.abs(ev) > cutoff ** 2)[0]
    ev, cv = ev[idx], cv[idx]

    # Check solutions with imaginary component
    real_indices = np.where(ev.imag < IMAG_THRESHOLD)[0]
    nimag = len(ev) - len(real_indices)
    if nimag != 0:
        print(
            f"""Warning: {nimag} complex eigenvalues found. 
              These will be removed from the final solution set."""
        )

    # Remove complex eigenvalues and corresponding eigenvectors
    real_ev = ev[real_indices]
    real_cv = cv[real_indices]

    return real_ev, real_cv


def _pickeig(w, tol=0.001):
    "Adapted from PySCF TDSCF module"
    idx = np.where(w > tol ** 2)[0]
    # get unique eigvals
    b = np.sort(w[idx])
    d = np.append(True, np.diff(b))
    TOL = 1e-6
    w = b[d > TOL]
    return w


def _pick_singlets(eigvals, eigvecs):
    # sort ev and cv correspondingly
    idx = eigvals.argsort()
    b = eigvals[idx]
    eigvecs = eigvecs[idx]
    # start picking up singlets
    mask = np.append(True, np.diff(b)) > 1.0e-7
    unique_eigs_idx = np.where(mask)[0]
    number_unique_eigs = np.diff(unique_eigs_idx)
    idx = np.where(number_unique_eigs == 1)[0]
    singlet_idx = unique_eigs_idx[idx]
    if unique_eigs_idx[-1] == len(eigvals) - 1:
        singlet_idx = np.append(singlet_idx, unique_eigs_idx[-1])
    singlets_ev = b[singlet_idx]
    singlets_cv = eigvecs[singlet_idx]
    return singlets_ev, singlets_cv, singlet_idx


def _pick_multiplets(eigvals, eigvecs):
    # sort ev and cv correspondingly
    idx = eigvals.argsort()
    b = eigvals[idx]
    eigvecs = eigvecs[idx]
    # start picking up triplets
    _, _, singlet_idx = _pick_singlets(eigvals, eigvecs)
    triplets_ev = np.delete(b, singlet_idx)
    triplets_cv = np.delete(eigvecs, singlet_idx, axis=0)
    return triplets_ev, triplets_cv
