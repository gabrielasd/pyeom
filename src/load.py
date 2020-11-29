"""
Equations-of-motion state base class.

"""


import os, sys

import numpy as np


__all__ = ["parse_inputfile", "check_inputs", "ParsedParams",]


def parse_inputfile(filename):
    with open(filename, 'r') as ifile:
        content = ifile.read()
    lines = [l for l in content.split("\n")]
    content = {option: value, for option, value in line.split() for line in lines}
    return ParsedParams(content)    


def check_inputs():
    pass


class ParsedParams:
    def __init__(self, content):
        self.nparts = content["nparts"]
        self.oneint_file = content["oneint_file"]
        self.twoint_file = content["twoint_file"]
        self.dm1_file = content["dm1_file"]
        self.dm2_file = content["dm2_file"]
        self.eom = content["eom"]
        self.orthog = content["orthog"]
        self.tol = content["tol"]
