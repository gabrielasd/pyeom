"""Test eomes.load."""

import os
import glob
from eomes import main
from test.tools import find_datafiles

import pytest


@pytest.mark.parametrize(
    "filename, numfiles",
    [
        ("test_input1", 2),
        ("test_input2", 3),
        ("test_input3", 2),
        ("test_input4", 2),
        ("test_input5", 2),
    ],
)
def test_run_main(filename, numfiles):
    inputfile = find_datafiles(filename + ".in")
    os.system("python eomes/main.py {0}".format(inputfile))
    files = sorted(glob.glob(filename + "*"))
    assert len(files) == numfiles

    for file in files:
        os.system("rm {}".format(file))
