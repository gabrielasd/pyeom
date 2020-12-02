"""
Equations-of-motion state base class.

"""


import os, sys
import re

import numpy as np


__all__ = [
    "parse_inputfile",
    "check_inputs",
    "ParsedParams",
]


def parse_inputfile(filename):
    with open(filename, "r") as ifile:
        content = ifile.read()
    # Make list of lines in ifile skipping white lines.
    # Based on stackoverflow post:
    # how-to-delete-all-blank-lines-in-the-file-with-the-help-of-python
    lines = [line for line in content.split("\n") if line.strip()]
    kwargs = {l.split(":")[0].strip(): l.split(":")[1].strip() for l in lines}
    return ParsedParams(kwargs)


def check_inputs(params):
    if not (isinstance(params, ParsedParams)):
        raise TypeError("chec_inputs argument must be a ParsedParams instance.")

    # check numbers
    if not isinstance(params.nparts, (int, tuple)):
        raise TypeError(
            "Number of electrons must be given as an integer or tuple of integers."
        )
    if not isinstance(params.tol, float):
        raise TypeError(
            "The tolerance value for matrix inversion must be given as a float."
        )

    # Check integrals files
    integrals = {"one": params.oneint_file, "two": params.twoint_file}
    for number, filename in integrals.items():
        if not os.path.isfile(filename):
            raise ValueError(
                "Cannot find the {}-electron integrals at {}."
                "".format(number, os.path.abspath(filename))
            )
    # Check density matrix files
    densities = {"one": params.dm1_file, "two": params.dm2_file}
    for number, filename in densities.items():
        if not os.path.isfile(filename):
            raise ValueError(
                "Cannot find the {}-electron reduced density matrix at {}."
                "".format(number, os.path.abspath(filename))
            )

    # check matrix orthogonalization method
    if params.orthog not in ["symmetric", "asymmetric"]:
        raise ValueError("Orthogonalization method must be `symmetric` or `asymmetric`")

    # check EOM method options
    eom_type = ["ip", "ea", "exc", "dip", "dea"]
    if params.eom not in eom_type:
        raise ValueError(
            "Equation-of-motion method must be one of `ip`, `ea`, `exc`, "
            "`dip` and `dea`."
        )
    if not isinstance(params.get_tdm, bool):
        raise TypeError(
            "Flag for transition density matrix computation must "
            "be given as a boolean."
        )


class ParsedParams:
    def __init__(self, content):
        # Assign numbers
        nparts = content["nelec"]
        nparts = re.findall("\d", nparts)
        if len(nparts) == 1:
            self.nparts = int(content["nelec"])
        elif len(nparts) == 2:
            self.nparts = (int(nparts[0]), int(nparts[1]))
        else:
            raise TypeError(
                """Number of electrons must be given as an integer,
                two comma separated integers or a tuple of integers"""
            )
        if "tol" not in content:
            self.tol = 1.0e-7
        else:
            self.tol = float(content["tol"])
        # self.nspino = int(content["nspino"])

        # Assign files
        self.oneint_file = content["oneint_file"]
        self.twoint_file = content["twoint_file"]
        self.dm1_file = content["dm1_file"]
        self.dm2_file = content["dm2_file"]

        # Assign solver options
        self.orthog = content["orthog"]

        # Assign EOM options
        self.eom = content["eom"]
        if "get_tdm" not in content:
            self.get_tdm = False
        elif content["get_tdm"] not in ["True", "False"]:
            raise ValueError("`get_tdm` must be `True` or `False`")
        else:
            self.get_tdm = bool(content["get_tdm"])


# if __name__ == "__main__":
#     filename = "../examples/file1.txt"
#     parameters = parse_inputfile(filename)
#     # print(parameters.nparts)
#     check_inputs(parameters)

