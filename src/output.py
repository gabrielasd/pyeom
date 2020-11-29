"""
Equations-of-motion state base class.

"""


import os, sys

import numpy as np


__all__ = [
    "dump",
]


def dump(filename, exce, coeffs, tdms=None):
    npofile = "{0}_exce_coeffs".format(filename)
    np.save(npofile, [exce, coeffs])
    if not tdms == None:
        npofile = "{0}_tdms".format(filename)
        np.save(npofile, tdms)

    # with open(filename, "w") as ofile:
    #     pass
