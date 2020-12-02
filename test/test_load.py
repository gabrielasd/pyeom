"""Test src.load."""

import numpy as np
from src import load
from .tools import (
    find_datafiles,
    spinize,
    symmetrize,
    antisymmetrize,
    hartreefock_rdms,
)

import pytest


def test_load_parse_inputfile():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8


def test_load_check_inputs():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8


def test_load_parsedparams():
    """Check the one-body teerms of the ionization potential
    equations of motion are correct.

    """
    nbasis = 2
    one_mo = np.load(find_datafiles("h2_hf_sto6g_oneint.npy"))
    one_mo = spinize(one_mo)

    # assert abs(aval1[-1] - ip[-1]) < 1e-8
