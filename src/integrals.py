"""
Electron integrlas module.

"""

import numpy as np


__all__ = [
    "ElectronIntegrals",
]


class ElectronIntegrals:
    """[summary]

    """

    def __init__(self, oneint_file=str, twoint_file=str):
        self.load_integrals(oneint_file, twoint_file)
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
            TypeError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """

        # FIXME: Add symmetry check
        if not (isinstance(h, np.ndarray) and isinstance(v, np.ndarray)):
            raise TypeError("Electron integrals must be given as a numpy array")
        elif not (h.ndim == 2 and h.shape[0] == h.shape[1]):
            raise ValueError(
                "One-electron integrals must be a two-dimensional square matrix"
            )
        elif not (v.ndim == 4 and v.shape == (h.shape[0],) * 4):
            raise ValueError("Two-electron integrals must be a square matrix of matrix")
        if not h.shape[0] == v.shape[0]:
            raise ValueError(
                "Number of spinorbitals between electron integrals don't match"
            )
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
                    "{}-electron integrlas do not satisfy symmetric permutations".format(
                        number
                    )
                )
        # Two-electron integrals antisymmetric permutations
        twoint_asymm = np.all(
            [
                np.allclose(v, -v.transpose(0, 1, 3, 2)),
                np.allclose(v, -v.transpose(1, 0, 2, 3)),
            ]
        )
        if not twoint_asymm:
            raise ValueError(
                "Two-electron integrlas do not satisfy the asymmetric permutations"
            )

    def load_integrals(self, oneint_file=str, twoint_file=str):
        """[summary]

        Args:
            oneint_file ([type], optional): [description]. Defaults to str.
            twoint_file ([type], optional): [description]. Defaults to str.
        """
        self._h = np.load(oneint_file)
        self._v = np.load(twoint_file)
        self.verify_integrals(self._h, self._v)
