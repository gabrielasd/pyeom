"""Test src.load."""

import numpy as np
from src import solver
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import pytest


def test_solver_dense_orthogonalization():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8


def test_solver_dense_tolerance():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8

