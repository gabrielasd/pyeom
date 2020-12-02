"""
Equations-of-motion state base class.

"""


import os, sys

import numpy as np


__all__ = [
    "dump",
]


def dump(filename, params, excen, coeffs, tdms=None):
    output = ""
    output += "# Number of electrons\n"
    output += "nelec: {:d}\n".format(sum(params.nparts))
    output += "# Electron Integrals files\n"
    output += "oneint_file: {}\n".format(params.oneint_file)
    output += "twoint_file: {}\n".format(params.twoint_file)
    output += "# Density Matrix files\n"
    output += "dm1_file: {}\n".format(params.dm1_file)
    output += "dm2_file: {}\n".format(params.dm2_file)
    output += "# Equation-of-motion parameters\n"
    output += "eom: {}\n".format(params.eom)
    output += "get_tdm: {}\n".format(params.get_tdm)
    output += "# Generalized eigenvalue solver parameters\n"
    output += "orthog: {}\n".format(params.orthog)
    output += "tol: {}\n".format(params.tol)

    dir_path = os.path.dirname(os.path.realpath(filename))
    cwd = os.getcwd()
    os.chdir(dir_path)

    filename = os.path.basename(filename)
    filename = filename.split(".")[0]
    npzfile = "{0}_excen_coeffs".format(filename)
    np.savez(npzfile, excen=excen, coeffs=coeffs)
    if not tdms == None:
        npyfile = "{0}_tdms".format(filename)
        np.save(npyfile, tdms)

    filename += ".out"
    with open(filename, "w") as ofile:
        ofile.write(output)

    os.chdir(cwd)
