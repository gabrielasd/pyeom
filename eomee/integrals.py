"""
Electron integrlas module.

"""

import re
import numpy as np


__all__ = [
    "ElectronIntegrals",
]


class ElectronIntegrals:
    """[summary]"""

    def __init__(self, oneint_file, twoint_file):
        self.load_integrals(oneint_file, twoint_file)
        self.verify_integrals(self._h, self._v)
        self._nspino = self._h.shape[0]

    @property
    def h(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._h

    @property
    def v(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._v

    @property
    def nspino(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._nspino

    def verify_integrals(self, h, v):
        """[summary]

        Args:
            h ([type]): [description]
            v ([type]): [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """

        if not (h.ndim == 2 and h.shape[0] == h.shape[1]):
            raise ValueError("One-electron integrals must be a two-dimensional square matrix")
        if not (v.ndim == 4 and v.shape == (v.shape[0],) * 4):
            raise ValueError("Two-electron integrals must be a square matrix of matrix")
        if not h.shape[0] == v.shape[0]:
            raise ValueError("Number of spinorbitals between electron integrals don't match")
        # Check integrals symmetry
        # Symmetric permutations
        oneint_symm = np.allclose(h, h.T)
        twoint_symm = np.all(
            [
                np.allclose(v, v.transpose(2, 3, 0, 1)),
                np.allclose(v, v.transpose(1, 0, 3, 2)),
            ]
        )
        symmetries = {"one": oneint_symm, "two": twoint_symm}
        for number, symm in symmetries.items():
            if not symm:
                raise ValueError(
                    "{}-electron integrals do not satisfy symmetric permutations".format(number)
                )
        # Two-electron integrals antisymmetric permutations
        twoint_asymm = np.all(
            [
                np.allclose(v, -v.transpose(0, 1, 3, 2)),
                np.allclose(v, -v.transpose(1, 0, 2, 3)),
            ]
        )
        if not twoint_asymm:
            raise ValueError("Two-electron integrals do not satisfy the asymmetric permutations")

    def load_integralfile(self, int_file):
        """[summary]

        Args:
            int_file (str): .npy integrals file.
        """
        match = re.search(r"\.npy$", int_file)
        if not match:
            raise ValueError("The electron integral must be a .npy file:{0} given".format(int_file))
        return np.load(int_file)

    def load_integrals(self, oneint_file, twoint_file):
        """[summary]

        Args:
            oneint_file ([type]): [description]
            twoint_file ([type]): [description]

        Raises:
            TypeError: [description]
        """
        integrals = {"one": (oneint_file), "two": twoint_file}
        for number, integral in integrals.items():
            if isinstance(integral, str):
                temp = self.load_integralfile(integral)
            elif isinstance(integral, np.ndarray):
                temp = integral
            else:
                raise TypeError(
                    "{0}-electron integral must be a .npy file or numpy array: {1} given".format(
                        number, integral
                    )
                )
            if number == "one":
                self._h = temp
            else:
                self._v = temp
