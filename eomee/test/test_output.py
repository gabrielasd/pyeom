"""Test eomee.load."""

import os
import glob
import numpy as np
from eomee import output
import pytest


class MockParsedParams:
    """Mocking load module class ParsedParams
    """

    def __init__(self):
        self.nparts = 4
        self.oneint_file = "oneint_file"
        self.twoint_file = "twoint_file"
        self.dm1_file = "dm1_file"
        self.dm2_file = "dm2_file"
        self.eom = "ip"
        self.get_tdm = False
        self.orthog = "symm"
        self.tol = 0.1
        self.roots = None


def example_dumpinputs():
    """Define three test cases to test the output module.
    Case 1: Do not print transition energies to the output file or
    store the TDMs.
    Case 2: Do not print transition energies to the output file.
    Store the TDMs.
    Case 3: Print transition energies to the output file but do not
    store the TDMs.
    Case 4: Print transition energies to the output file and store the TDMs.

    Yields:
        tuple: Input parameters for test_output_dump:
        filename, params, excen, coeffs, tdms, num_files, num_outlines
    """
    # Case 1: No roots, no TDMs
    test_params1 = MockParsedParams()
    # Case 2: TDMs, no roots
    test_params2 = MockParsedParams()
    test_params2.roots = 1
    # Case 3: roots, no TDMs

    for case in [
        (
            "example_output.in",
            test_params1,
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 2.0], [1.0, 2.0]]),
            None,
            2,
            17,
        ),
        (
            "example_output.in",
            test_params1,
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 2.0], [1.0, 2.0]]),
            np.zeros((2, 2, 2)),
            3,
            17,
        ),
        (
            "example_output.in",
            test_params2,
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 2.0], [1.0, 2.0]]),
            None,
            2,
            21,
        ),
    ]:
        yield case


@pytest.mark.parametrize(
    "filename, params, excen, coeffs, tdms, num_files, num_outlines",
    example_dumpinputs(),
)
def test_output_dump(filename, params, excen, coeffs, tdms, num_files, num_outlines):
    """Check that the content of the output file and the number
    of output files generated are correct.
    """
    output.dump(filename, params, excen, coeffs, tdms=tdms)
    files = sorted(glob.glob("example_output*"))
    with open("example_output.out", "r") as outfile:
        filelines = outfile.readlines()

    assert len(files) == num_files
    assert len(filelines) == num_outlines

    for file in files:
        os.system("rm {}".format(file))


@pytest.mark.parametrize(
    "excen, method, roots",
    [(np.asarray([1.0, 2.0]), "eom", 1), (np.asarray([0.0, 1.0, 2.0]), "ip", 3)],
)
def test_output_get_roots_valueerror(excen, method, roots):
    #     filename, params, excen, coeffs, tdms = None
    with pytest.raises(ValueError):
        output.get_roots(excen, method, roots)
