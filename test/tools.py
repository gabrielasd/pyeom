"""
Functions for electron integrals transformation from spatial to spin representation
and Hartree-Fock reduced density calculation.
"""
# Author: Michael Risher
from os import path

import numpy as np


__all__ = [
    "spinize",
    "symmetrize",
    "antisymmetrize",
    "from_unrestricted",
    "hartreefock_rdms",
    "find_datafiles",
]

DIRPATH = path.join(path.dirname(__file__), "data/")


def find_datafiles(file_name):
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
