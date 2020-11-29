"""
Equations-of-motion state base class.

"""

import numpy as np


__all__ = [
    "ElectronIntegrals",
]


class ElectronIntegrals:
    def __init__(self, oneint_file=str, twoint_file=str):
        self.load_integrals(oneint_file, twoint_file)
        self._nspino = self._h.shape[0]

    def verify_integrals(self, h, v):
        pass

    def load_integrals(self, oneint_file=str, twoint_file=str):
        self._h = np.load(oneint_file)
        self._v = np.load(twoint_file)
        self.verify_integrals(self._h, self._v)

    @property
    def h(self):
        return self._h

    @property
    def v(self):
        return self._v

    @property
    def nspino(self):
        return self._nspino
