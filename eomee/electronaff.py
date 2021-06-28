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

r"""Electron affinity EOM state class."""


import numpy as np

from .base import EOMState


__all__ = [
    "EOMEA",
    "EOMEADoubleCommutator",
    "EOMEAAntiCommutator",
]


class EOMEA(EOMState):
    r"""
    Electron Affinities EOM states for operator :math:`Q = \sum_n { c_n a^{\dagger}_n }`.

    .. math::

        \sum_{n} \left< \Psi^{(N)}_{0} \middle| a_{m} \left[ \hat{H},a^{\dagger}_n \right] \middle| \Psi^{(N)}_{0} \right> c_{n;k}
        &= \Delta_{k} \sum_{n} \left< \Psi^{(N)}_{0} \middle| a_{m}a^{\dagger}_n \middle| \Psi^{(N)}_{0} \right> c_{n;k}

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

            A = h_mn - \sum_p { h_{pn} \gamma_{pm} }
            - 2 \sum_{pqs} { v_pqsn \Gamma_pqsm }
            + 4 \sum_{qs} { v_mqns \gamma_qs}.

        """
        # A_mn = h_mn - h_pn \gamma_pm - 0.5 v_pqsn \Gamma_pqsm
        #      + v_mqns \gamma_qs
        # A_mn = h_mn + v_mqns \gamma_qs
        #      - ( h_pn \gamma_pm + 0.5 * v_pqsn \Gamma_pqsm )
        a = np.copy(self._h)
        a += np.tensordot(self._v, self._dm1, axes=((1, 3), (0, 1)))
        a -= np.dot(self._dm1, self._h)
        a -= 0.5 * np.tensordot(self._dm2, self._v, axes=((0, 1, 2), (0, 1, 2)))
        return a

    def _compute_rhs(self):
        r"""
        Compute :math:`M = \sum_n { \delta_nm - \gamma_{nm} }`.

        """
        # M_mn = \delta_mn - \gamma_mn
        m = np.eye(self._n)
        m -= self._dm1
        return m


class EOMEAAntiCommutator(EOMState):
    r"""
    Electron Affinities EOM states for operator :math:`Q = \sum_n { c_n a^{\dagger}_n }`.

    .. math::

        \sum_{n} \left< \Psi^{(N)}_{0} \middle| \Big\{ a_{m}, \left[ \hat{H},a^{\dagger}_n \right]\Big\} \middle| \Psi^{(N)}_{0} \right> c_{n;k}
        &= \Delta_{k} \sum_{n} \left< \Psi^{(N)}_{0} \middle| \Big\{a_{m},a^{\dagger}_n \Big\} \middle| \Psi^{(N)}_{0} \right> c_{n;k}

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
        Compute :math:`A = h_mn + \sum_{qr} { v_mqnr \gamma_qr}`.

        """
        # A_mn = h_mn + <v_mqnr> \gamma_qr
        a = np.copy(self._h)
        a += np.einsum("mqnr,qr->mn", self._v, self._dm1)
        return a

    def _compute_rhs(self):
        r"""
        Compute :math:`M = \sum_n { \delta_nm }`.

        """
        # M_mn = \delta_mn
        m = np.eye(self._n)
        return m


class EOMEADoubleCommutator(EOMState):
    r"""
    Electron Affinities EOM states for operator :math:`Q = \sum_n { c_n a^{\dagger}_n }`.

    .. math::

        \sum_{n} \left< \Psi^{(N)}_{0} \middle| \left[ a_{m}, \left[ \hat{H},a^{\dagger}_n \right]\right] \middle| \Psi^{(N)}_{0} \right> c_{n;k}
        &= \Delta_{k} \sum_{n} \left< \Psi^{(N)}_{0} \middle| \left[a_{m},a^{\dagger}_n \right] \middle| \Psi^{(N)}_{0} \right> c_{n;k}

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

            A = h_mn - 2\sum_p { h_{pn} \gamma_{pm} }
            + \sum_{pqr} { v_pqrn \Gamma_pqmr }
            + \sum_{qs} { v_mqns \gamma_qs}.

        """
        # A_mn = h_mn + <v_mqnr> \gamma_qr
        a = np.copy(self._h)
        a += np.einsum("mqnr,qr->mn", self._v, self._dm1)
        # A_mn -= 2 h_pn \gamma_pm
        a -= 2 * np.einsum(
            "pm,pn->mn",
            self._dm1,
            self._h,
        )
        # A_mn += <v_pqrn> \Gamma_pqmr
        #      -= <v_pqnr> \Gamma_pqmr
        a -= 2 * np.einsum(
            "pqmr,pqnr->mn",
            self._dm2,
            self._v,
        )
        return a

    def _compute_rhs(self):
        r"""
        Compute :math:`M = \sum_n { \delta_nm - 2 \gamma_{nm} }`

        """
        # M_mn = \delta_mn - 2 \gamma_mn
        m = np.eye(self._n)
        m -= 2 * self._dm1
        return m
