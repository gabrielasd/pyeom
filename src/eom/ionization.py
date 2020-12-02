import numpy as np

from .base import EOMBase


__all__ = [
    "EOMIP",
    "IonizationDoubleCommutator",
    "IonizationAntiCommutator",
]


class EOMIP(EOMBase):
    """
    Ionization EOM state for operator Q = \sum_n { c_n a_n }.
    .. math::
        \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m \left[ \hat{H}, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m \hat{Q} \middle| \Psi^{(N)}_0 \right>

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
        """
        Compute A = \sum_q { -h_{nq} \gamma_{mq} }
                  - 0.5 \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqrs} }.

        """
        # A_mn = -h_nq \gamma_mq - 0.5 <v_nqrs> \Gamma_mqrs
        a = np.dot(self._dm1, self._h)
        b = np.tensordot(self._dm2, self._v, ((1, 2, 3), (1, 2, 3)))
        b *= 0.5
        b += a
        return -b

    def _compute_rhs(self):
        """
        Compute M = \sum_n { \gamma_{mn} }.

        """
        # M_mn = \gamma_mn
        return np.copy(self._dm1)

    def compute_tdm(self, coeffs):
        """
        Compute .

        """
        # M_mn = \gamma_mn
        # return np.copy(self._dm1)
        pass


class IonizationDoubleCommutator(EOMBase):
    """
    Ionization EOM state for operator Q = \sum_n { c_n a_n }.
    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_m, \left[ \hat{H}, \hat{Q} \right] \right] \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_m, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>

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
        """
        Compute A = h_{nm} -2 \sum_q { h_{nq} \gamma_{mq} }
                  + \sum_{qs} { \left< nq||ms \right> \gamma_{qs} }
                  + \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqsr} }.

        """
        # A_mn = h_mn - 2 * \gamma_mq * h_qn
        a = np.copy(self._h)
        a -= 2 * np.dot(self._dm1, self._h)
        # A_mn += <v_msnq> \gamma_sq - \Gamma_mqrs <v_nqrs>
        a += np.einsum("msnq,sq", self._v, self._dm1, optimize=True)
        a -= np.einsum("mqrs,nqrs", self._dm2, self._v, optimize=True)
        return a

    def _compute_rhs(self):
        """
        Compute M = 2 \gamma_{mn} - \delta_{nm}.

        """
        # M_mn = 2 * \gamma_mn - \delta_mn
        m = 2 * np.copy(self._dm1)
        m -= np.eye(self._n, dtype=self._dm1.dtype)
        return m


class IonizationAntiCommutator(EOMBase):
    """
    Ionization EOM state for operator Q = \sum_n { c_n a_n }.
    .. math::
        \left< \Psi^{(N)}_0 \middle| \Big\{ a^{\dagger}_m, \left[ \hat{H}, \hat{Q} \right] \Big\} \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| \Big\{[a^{\dagger}_m, \hat{Q} \Big\} \middle| \Psi^{(N)}_0 \right>

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
        """
        Compute A = -h_{nm} + \sum_{qr} { \left< qn||mr \right> \gamma_{qr} }.

        """
        # A_mn = -h_mn
        a = -np.copy(self._h)
        # A_mn += <v_qnmr> \gamma_qr
        a += np.einsum("qnmr,qr", self._v, self._dm1, optimize=True)
        return a

    def _compute_rhs(self):
        """
        Compute M = \delta_{nm}.

        """
        # M_mn = \delta_mn
        m = np.eye(self._n, dtype=self._dm1.dtype)
        return m
