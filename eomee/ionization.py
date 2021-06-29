# This file is part of EOMEE.
#
# EOMEE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# EOMEE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

r"""Ionization Potential EOM state class."""


import numpy as np

from .base import EOMState


__all__ = [
    "EOMIP",
    "EOMIPDoubleCommutator",
    "EOMIPAntiCommutator",
]


class EOMIP(EOMState):
    r"""
    Ionization EOM state for operator :math:`\hat{Q}_k = \sum_n { c_n a_n }`.

    .. math::

        \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m \left[ \hat{H}, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m \hat{Q} \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        r"""
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

            A_{mn} = \sum_q { -h_{nq} \gamma_{mq} } - 0.5 \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqrs} }.

        """
        # A_mn = -h_nq \gamma_mq - 0.5 <v_nqrs> \Gamma_mqrs
        a = np.dot(self._dm1, self._h)
        b = np.tensordot(self._dm2, self._v, ((1, 2, 3), (1, 2, 3)))
        b *= 0.5
        b += a
        return -b

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = \gamma_{mn}`.

        """
        # M_mn = \gamma_mn
        return np.copy(self._dm1)


class EOMIPDoubleCommutator(EOMState):
    r"""
    Ionization EOM state for operator :math:`\hat{Q}_k = \sum_n { c_n a_n }`.

    .. math::

        \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_m, \left[ \hat{H}, \hat{Q} \right] \right] \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_m, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        r"""
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

            A_{mn} = h_{nm} -2 \sum_q { h_{nq} \gamma_{mq} } + \sum_{qs} { \left< nq||ms \right> \gamma_{qs} }
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
        r"""
        Compute :math:`M_{mn} = 2 \gamma_{mn} - \delta_{nm}`.

        """
        # M_mn = 2 * \gamma_mn - \delta_mn
        m = 2 * np.copy(self._dm1)
        m -= np.eye(self._n, dtype=self._dm1.dtype)
        return m


class EOMIPAntiCommutator(EOMState):
    r"""
    Ionization EOM state for operator :math:`\hat{Q}_k = \sum_n { c_n a_n }`.

    .. math::

        \left< \Psi^{(N)}_0 \middle| \Big\{ a^{\dagger}_m, \left[ \hat{H}, \hat{Q} \right] \Big\} \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| \Big\{a^{\dagger}_m, \hat{Q} \Big\} \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        r"""
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
        Compute :math:`A_{mn} = -h_{nm} + \sum_{qr} { \left< qn||mr \right> \gamma_{qr} }`.

        """
        # A_mn = -h_mn
        a = -np.copy(self._h)
        # A_mn += <v_qnmr> \gamma_qr
        a += np.einsum("qnmr,qr", self._v, self._dm1, optimize=True)
        return a

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = \delta_{mn}`.

        """
        # M_mn = \delta_mn
        m = np.eye(self._n, dtype=self._dm1.dtype)
        return m
