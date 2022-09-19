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

r"""Electron integral transformations from spatial to spin representation and Hartree-Fock RDMs."""


from os import path

import numpy as np


__all__ = [
    "spinize",
    "symmetrize",
    "antisymmetrize",
    "from_unrestricted",
    "hartreefock_rdms",
    "find_datafiles",
    "pickpositiveeig",
]

DIRPATH = path.join(path.dirname(__file__), "test/", "data/")


def find_datafiles(file_name):
    r""" """
    datapath = path.join(path.abspath(DIRPATH), file_name)
    return path.abspath(datapath)


def spinize(x):
    r"""
    Transform a two- or four- index array from spatial to spin representation.

    Parameters
    ----------
    x : np.ndarray(float(n, n)) or np.ndarray(float(n, n, n, n))
        Spatial representation array.

    Returns
    -------
    y : np.ndarray(float(m, m)) or np.ndarray(float(m, m, m, m))
        Spin representation array.

    """
    n = x.shape[0]
    m = n * 2
    if x.ndim == 2:
        y = np.zeros((m, m))
        y[:n, :n] = x
        y[n:, n:] = x
    elif x.ndim == 4:
        y = np.zeros((m, m, m, m))
        y[:n, :n, :n, :n] = x
        y[n:, n:, n:, n:] = x
        y[:n, n:, :n, n:] = x
        y[n:, :n, n:, :n] = x
    else:
        raise ValueError("Input must have ndim == 2 or ndim == 4")
    return y


def symmetrize(x):
    r"""
    Symmetrize a two- or four- index array in the spin representation.

    Parameters
    ----------
    x : np.ndarray(float(n, n)) or np.ndarray(float(n, n, n, n))
        Two- or four- index spin representation array.

    Returns
    -------
    y : np.ndarray(float(m, m)) or np.ndarray(float(m, m, m, m))
        Antisymmetrized two- or four- index spin representation array.

    """
    if x.ndim == 2:
        y = x + x.T
        y *= 0.5
    elif x.ndim == 4:
        y = x + x.transpose(1, 0, 3, 2)
        y += x.transpose(2, 3, 0, 1)
        y += x.transpose(3, 2, 1, 0)
        y *= 0.25
    else:
        raise ValueError("Input must have ndim == 2 or ndim == 4")
    return y


def antisymmetrize(x):
    r"""
    Antisymmetrize a four-index array in the spin representation.

    Parameters
    ----------
    x : np.ndarray(float(n, n, n, n))
        Four-index spin representation array.

    Returns
    -------
    y : np.ndarray(float(n, n, n, n))
        Antisymmetrized four-index spin representation array.

    """
    if x.ndim != 4:
        raise ValueError("Input must have ndim == 4")
    return x - x.transpose(0, 1, 3, 2)


def from_unrestricted(blocks):
    r"""
    Return a two- or four- index array in the spin representation from blocks.

    A two-index array is recontrcuted from blocks (a, b).
    A four-index array is recontrcuted from blocks (aa, ab, bb).

    Parameters
    ----------
    blocks : tuple of np.ndarray of length 2 or 3
        Blocks from which to reconstruct array.

    Returns
    -------
    y : np.ndarray(float(m, m)) or np.ndarray(float(m, m, m, m))
        Spin representation array.

    """
    if len(blocks) == 2:
        for b in blocks:
            if b.ndim != 2:
                raise ValueError("Input must have ndim == 2")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k))
        y[:n, :n] = blocks[0]
        y[n:, n:] = blocks[1]
    elif len(blocks) == 3:
        for b in blocks:
            if b.ndim != 4:
                raise ValueError("Input must have ndim == 4")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k, k, k))
        y[:n, :n, :n, :n] = blocks[0]
        y[:n, n:, :n, n:] = blocks[1]
        y[n:, :n, n:, :n] = blocks[1]
        y[n:, n:, n:, n:] = blocks[2]
    else:
        raise ValueError("Invalid input")
    return y


def hartreefock_rdms(nbasis, na, nb):
    r"""
    Return the 1- and 2- RDMs of the Hartree-Fock Slater determinant.

    Returns the RDMS in the antisymmetrized spin representation.

    Parameters
    ----------
    nbasis : int
        Number of spatial basis functions.
    na : int
        Number of alpha or spin-up electrons.
    nb : int
        Number of beta or spin-down electrons.

    Returns
    -------
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).

    """
    k = 2 * nbasis
    dm1 = np.zeros((k, k))
    for i in range(na):
        dm1[i, i] = 1.0
    for i in range(nbasis, nbasis + nb):
        dm1[i, i] = 1.0
    dm2 = np.kron(dm1, dm1).reshape(k, k, k, k)
    dm2 -= dm2.transpose(0, 1, 3, 2)
    return dm1, dm2


def pickpositiveeig(w, cv, tol=0.01):
    r"""
    Adapted from PySCF TDSCF module.

    """
    idx = np.where(w > tol ** 2)[0]
    return w[idx], cv[idx], idx


def pickeig(w, tol=0.001):
    "adapted from PySCF TDSCF module"
    idx = np.where(w > tol ** 2)[0]
    # get unique eigvals
    b = np.sort(w[idx])
    d = np.append(True, np.diff(b))
    TOL = 1e-6
    w = b[d > TOL]
    return w


def two_positivity_condition(matrix, nbasis, dm1, dm2):
    r""" P, Q and G (2-positivity) conditions for N-representability.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. One of `p`, `g` and `q`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).

    Returns
    -------
    np.ndarray(float(n, n, n, n))
        P, Q or G-matrix from the 2-positivity conditions.
    """
    m = 2 * nbasis
    I = np.eye(m)
    if matrix == 'p':
        # P-condition: P >= 0
        # P_pqrs = <\Psi|a^\dagger_p a^\dagger_q s r|\Psi>
        #        = \Gamma_pqrs
        return dm2   # P_matrix
    elif matrix == 'q':
        # Q-condition: Q >= 0
        # Q_pqrs = <\Psi|a_p a_q s^\dagger r^\dagger|\Psi>
        #        = \Gamma_pqrs + \delta_qs \delta_pr - \delta_ps \delta_qr
        #        - \delta_qs \gamma_pr + \delta_ps \gamma_qr + \delta_qr \gamma_ps - \delta_pr \gamma_qs
        Q_matrix = dm2
        Q_matrix += (np.einsum('qs,pr->pqrs', I, I) -  np.einsum('ps,qr->pqrs', I, I))
        Q_matrix -= (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        Q_matrix += (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1)) 
        return Q_matrix
    elif matrix == 'g':
        # G-condition: G >= 0
        # G_pqrs = <\Psi|a^\dagger_p a_q s^\dagger r|\Psi>
        #        = \delta_qs \gamma_pr - \Gamma_psrq
        return np.einsum('qs,pr->pqrs', I, dm1) - dm2.transpose((0,3,2,1))
    else:
        raise ValueError(f'Matrix must be one of `p`, `q` and `g`, {matrix} given.')


def matrix_P_to_matrix(matrix, nbasis, dm1, dm2, matrixp=None):
    r"""Transform P-matrix into G or Q-matrices.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. Either `g` or `q`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).
    matrixg : ndarray((n, n, n, n)), optional
        If provided it is taken as the G-matrix, by default None

    Returns
    -------
    ndarray((n, n, n, n))
        The type of matrix requested by `matrix`.
    """
    m = 2 * nbasis
    I = np.eye(m)
    # P_pqrs = <\Psi|a^\dagger_p a^\dagger_q s r|\Psi>
    if matrixp is None:
        P_matrix = two_positivity_condition('p', nbasis, dm1, dm2)
    else:
        P_matrix = matrixp

    if matrix == 'g':
        # P_pqrs = \delta_qs \gamma_pr - <\Psi|a^\dagger_p a_s q^\dagger r|\Psi>
        # G_psrq = \delta_qs \gamma_pr - P_pqrs = G_pqrs
        return np.einsum('qs,pr->pqrs', I, dm1) - P_matrix
    elif matrix == 'q':
        # P_pqrs = <\Psi|a_p a_q s^\dagger r^\dagger|\Psi> - \delta_qs \delta_pr + \delta_ps \delta_qr
        #        + \delta_qs \gamma_pr - \delta_ps \gamma_qr - \delta_qr \gamma_ps + \delta_pr \gamma_qs
        # Q_pqrs = P_pqrs + \delta_qs \delta_pr - \delta_ps \delta_qr
        #        - \delta_qs \gamma_pr + \delta_ps \gamma_qr + \delta_qr \gamma_ps - \delta_pr \gamma_qs
        Q_matrix = P_matrix
        Q_matrix += (np.einsum('qs,pr->pqrs', I, I) -  np.einsum('ps,qr->pqrs', I, I))
        Q_matrix -= (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        Q_matrix += (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1)) 
        return Q_matrix    
    else:
        raise ValueError(f'Matrix must be `q` or `g`, {matrix} given.')


def matrix_G_to_matrix(matrix, nbasis, dm1, dm2, matrixg=None):
    r"""Transform G-matrix into P or Q-matrices.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. Either `p` or `q`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).
    matrixg : ndarray((n, n, n, n)), optional
        If provided it is taken as the G-matrix, by default None

    Returns
    -------
    ndarray((n, n, n, n))
        The type of matrix requested by `matrix`.
    """
    m = 2 * nbasis
    I = np.eye(m)
    # G_pqrs = <\Psi|a^\dagger_p a_q s^\dagger r|\Psi>
    if matrixg is None:
        G_matrix = two_positivity_condition('g', nbasis, dm1, dm2)
    else:
        G_matrix = matrixg

    if matrix == 'p':
        # G_pqrs = \delta_qs \gamma_pr - <\Psi|a^\dagger_p s^\dagger a_q r|\Psi>
        # P_psrq = \delta_qs \gamma_pr - G_pqrs = P_pqrs        
        return np.einsum('qs,pr->pqrs', I, dm1) - G_matrix
    elif matrix == 'q':
        # G_pqrs = \delta_qs \gamma_pr - P_psrq
        # P_psrq = <\Psi|a_p a_s q^\dagger r^\dagger|\Psi> - \delta_qs \delta_pr + \delta_pq \delta_sr
        #        + \delta_qs \gamma_pr - \delta_pq \gamma_sr - \delta_sr \gamma_pq + \delta_pr \gamma_qs
        # Q_psrq = (\delta_qs \gamma_pr - G_pqrs) + \delta_qs \delta_pr - \delta_pq \delta_sr
        #        - \delta_qs \gamma_pr + \delta_pq \gamma_sr + \delta_sr \gamma_pq - \delta_pr \gamma_qs
        Q_matrix = matrix_G_to_matrix('p', nbasis, dm1, dm2)   # P_pqrs
        Q_matrix += (np.einsum('qs,pr->pqrs', I, I) -  np.einsum('ps,qr->pqrs', I, I))
        Q_matrix -= (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        Q_matrix += (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1)) 
        return Q_matrix    
    else:
        raise ValueError(f'Matrix must be `q` or `p`, {matrix} given.')


def matrix_Q_to_matrix(matrix, nbasis, dm1, dm2, matrixq=None):
    r"""Transform Q-matrix into P or G-matrices.

    Parameters
    ----------
    matrix : str
        Type of matrix to be evaluated. Either `p` or `g`.
    nbasis : int
        Number of spatial basis functions.
    dm1 : np.ndarray(float(n, n))
        One-electron reduced density matrix in the spin representation.
    dm2 : np.ndarray(float(n, n, n, n))
        Two-electron reduced density matrix in the spin representation
        (antisymmetrized).
    matrixg : ndarray((n, n, n, n)), optional
        If provided it is taken as the G-matrix, by default None

    Returns
    -------
    ndarray((n, n, n, n))
        The type of matrix requested by `matrix`.
    """
    m = 2 * nbasis
    I = np.eye(m)
    # Q_pqrs = <\Psi|a_p a_q s^\dagger r^\dagger|\Psi>
    if matrixq is None:
        Q_matrix = two_positivity_condition('q', nbasis, dm1, dm2)
    else:
        Q_matrix = matrixq

    if matrix == 'p':
        # Q_pqrs = <\Psi|a^\dagger_p a^\dagger_q s r|\Psi> + \delta_qs \delta_pr - \delta_ps \delta_qr
        #        - \delta_qs \gamma_pr + \delta_ps \gamma_qr + \delta_qr \gamma_ps - \delta_pr \gamma_qs
        # P_pqrs = Q_pqrs - \delta_qs \delta_pr + \delta_ps \delta_qr
        #        + \delta_qs \gamma_pr - \delta_ps \gamma_qr - \delta_qr \gamma_ps + \delta_pr \gamma_qs
        P_matrix = Q_matrix
        P_matrix += (np.einsum('ps,qr->pqrs', I, I) - np.einsum('qs,pr->pqrs', I, I))
        P_matrix += (np.einsum('qs,pr->pqrs', I, dm1) +  np.einsum('pr,qs->pqrs', I, dm1))
        P_matrix -= (np.einsum('ps,qr->pqrs', I, dm1) +  np.einsum('qr,ps->pqrs', I, dm1))
        return P_matrix
    elif matrix == 'g':
        # G_pqrs = \delta_qs \gamma_pr - P_psrq
        # P_pqrs = Q_pqrs - \delta_qs \delta_pr + \delta_ps \delta_qr
        #        + \delta_qs \gamma_pr - \delta_ps \gamma_qr - \delta_qr \gamma_ps + \delta_pr \gamma_qs
        return np.einsum('qs,pr->pqrs', I, dm1) - matrix_Q_to_matrix('p', nbasis, dm1, dm2)
    else:
        raise ValueError(f'Matrix must be `p` or `g`, {matrix} given.')
