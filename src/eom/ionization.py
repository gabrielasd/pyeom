"""
Electron removal Equations-of-motion.

"""


import numpy as np

from .base import EOMBase


__all__ = [
    "EOMIP",
]


class EOMIP(EOMBase):
    r"""
    Ionization EOM state for operator Q = \sum_n { c_n a_n }.

    .. math::
        \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m
        \left[ \hat{H}, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>\\
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m
        \hat{Q} \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        """
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}}
        return self._n

    def _compute_lhs(self):
        r"""
        Compute

        .. math::

            A = \sum_q { -h_{nq} \gamma_{mq} }
                - 0.5 \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqrs} }.

        """
        # A_mn = -h_nq \gamma_mq - 0.5 <v_nqrs> \Gamma_mqrs
        a = np.dot(self._dm1, self._h)
        b = np.tensordot(self._dm2, self._v, ((1, 2, 3), (1, 2, 3)))
        b *= 0.5
        b += a
        return -b

    def _compute_rhs(self):
        r"""
        Compute :math:`M = \sum_n { \gamma_{mn} }`.

        """
        # M_mn = \gamma_mn
        return np.copy(self._dm1)

    def compute_tdm(self, coeffs):
        r"""
        Compute

        .. math::
            T_m = \sum_n { \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m
            \hat{Q} \middle| \Psi^{(N)}_0 \right> c_{n} }.

        """
        return np.einsum("mn,nl->ml", self._dm1, coeffs)
