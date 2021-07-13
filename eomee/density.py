"""Reduced density matrices module."""

import numpy as np


__all__ = [
    "WfnRDMs",
]


class WfnRDMs:
    """[summary]"""

    def __init__(self, nparts, dm1_file=str, dm2_file=str):
        if isinstance(nparts, int):
            self._nparts = nparts
        else:
            self._nparts = sum(nparts)
        self.assign_rdms(dm1_file, dm2_file)
        self._nspino = self._dm1.shape[0]

    @property
    def dm1(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._dm1

    @property
    def dm2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._dm2

    @property
    def nspino(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._nspino

    @property
    def nparts(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._nparts

    def verify_rdms(self, dm1, dm2):
        """[summary]

        Args:
            dm1 ([type]): [description]
            dm2 ([type]): [description]

        Raises:
            TypeError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """
        # Check DMs
        # This check makes no sence given how the files are loaded with assign_rdms
        # if not (isinstance(dm1, np.ndarray) and isinstance(dm2, np.ndarray)):
        #     raise TypeError("Density matrices must be given as a numpy array")
        if not (dm1.ndim == 2 and dm1.shape[0] == dm1.shape[1]):
            raise ValueError("One-reduced density matrix must be a two-dimensional square matrix")
        if not (dm2.ndim == 4 and dm2.shape == (dm2.shape[0],) * 4):
            raise ValueError("Two-reduced density matrix must be a square matrix of matrix")
        if not dm1.shape[0] == dm2.shape[0]:
            raise ValueError("Number of spinorbitals between density matrices don't match")

        # Check DMs symmetry
        # Symmetric permutations
        onedm_symm = np.allclose(dm1, dm1.T)
        twodm_symm = np.all(
            [
                np.allclose(dm2, dm2.transpose(2, 3, 0, 1)),
                np.allclose(dm2, dm2.transpose(1, 0, 3, 2)),
            ]
        )
        symmetries = {"one": onedm_symm, "two": twodm_symm}
        for number, symm in symmetries.items():
            if not symm:
                raise ValueError(
                    "{}-RDM does not satisfy the symmetric permutations".format(number)
                )
        # Two-reduced density matrix antisymmetric permutations
        twodm_asymm = np.all(
            [
                np.allclose(dm2, -dm2.transpose(0, 1, 3, 2)),
                np.allclose(dm2, -dm2.transpose(1, 0, 2, 3)),
            ]
        )
        if not twodm_asymm:
            raise ValueError("Two-RDM does not satisfy the asymmetric permutations")

        # Check normalization condition of one- and two-DMs
        if not np.trace(dm1) == self._nparts:
            raise ValueError(
                "Expected normalization value of one-reduced density matrix {}, got {}"
                "".format(self._nparts, np.trace(dm1))
            )
        dm2 = dm2.reshape(dm2.shape[0] ** 2, dm2.shape[0] ** 2)
        norm = self._nparts * (self._nparts - 1)
        if not np.trace(dm2) == norm:
            raise ValueError(
                "Expected normalization value of two-reduced density matrix {}, got {}"
                "".format(norm, np.trace(dm2))
            )

    def assign_rdms(self, dm1_file=str, dm2_file=str):
        """[summary]

        Args:
            dm1_file ([type], optional): [description]. Defaults to str.
            dm2_file ([type], optional): [description]. Defaults to str.
        """
        self._dm1 = np.load(dm1_file)
        self._dm2 = np.load(dm2_file)
        self.verify_rdms(self._dm1, self._dm2)
