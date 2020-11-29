"""
Equations-of-motion state base class.

"""

import numpy as np


__all__ = [
    "WfnRDMs",
]


class WfnRDMs:
    def __init__(self, nparts, dm1_file=str, dm2_file=str):
        self._nparts = nparts
        self.assign_rdms(dm1_file, dm2_file)
        self._nspino = self._dm1.shape[0]

    def verify_rdms(self, h, v):
        pass

    def assign_rdms(self, dm1_file=str, dm2_file=str):
        self._dm1 = np.load(dm1_file)
        self._dm2 = np.load(dm2_file)
        self.verify_rdms(self._dm1, self._dm2)

    @property
    def dm1(self):
        return self._dm1

    @property
    def dm2(self):
        return self._dm2

    @property
    def nspino(self):
        return self._nspino

    @property
    def nparts(self):
        return self._nparts
