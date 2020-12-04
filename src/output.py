"""
Output module.

"""


import os

import numpy as np


__all__ = [
    "dump",
]


def get_roots(excen, method, roots):
    """[summary]

    Args:
        excen ([type]): [description]
        method ([type]): [description]
        roots ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if method == "ip":
        header = "Index, Ionization potential\n"
    elif method == "ea":
        header = "Index, Electron affinity\n"
    elif method == "exc":
        header = "Index, Excitation energy\n"
    elif method == "dip":
        header = "Index, Double ionization potential\n"
    elif method == "dea":
        header = "Index, Double Electron affinity\n"
    else:
        raise ValueError("Invalid EOM method: {}".format(method))

    # FIXME:
    # number of roots cant be > than number of eigenvalues
    excen[excen < 0] = 0.0
    none0_id = np.flatnonzero(excen)
    output = ""
    output += header
    for idx in none0_id[:roots]:
        output += "{0}, {1:.5f}\n".format(idx, excen[idx])
    return output


def dump(filename, params, excen, coeffs, tdms=None):
    """[summary]

    Args:
        filename ([type]): [description]
        params ([type]): [description]
        excen ([type]): [description]
        coeffs ([type]): [description]
        tdms ([type], optional): [description]. Defaults to None.
        roots ([type], optional): [description]. Defaults to None.
    """
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
    output += "# Number of roots to print\n"
    output += "roots: {}\n".format(params.roots)
    output += "\n"

    if params.roots is not None:
        output += "# Total number of eigenvalues {}\n".format(len(excen))
        output += "Printing {} selected energies\n".format(params.roots)
        output += get_roots(excen, params.eom, params.roots)

    dir_path = os.path.dirname(os.path.realpath(filename))
    cwd = os.getcwd()
    os.chdir(dir_path)

    filename = os.path.basename(filename)
    filename = filename.split(".")[0]
    npzfile = "{0}_excen_coeffs".format(filename)
    np.savez(npzfile, excen=excen, coeffs=coeffs)
    if not tdms is None:
        npyfile = "{0}_tdms".format(filename)
        np.save(npyfile, tdms)

    filename += ".out"
    with open(filename, "w") as ofile:
        ofile.write(output)

    os.chdir(cwd)
