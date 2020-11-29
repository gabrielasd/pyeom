import numpy as np

from eomee.base import EOMState


__all__ = [
    'ElectronAffinitiesEOM1',
    'ElectronAffinitiesEOM2',
    'ElectronAffinitiesEOM3',
]


class ElectronAffinitiesEOM1(EOMState):
    """
    Electron Affinities EOM states for operator Q = \sum_n { c_n a^{\dagger}_n }.

    .. math::
        \sum_{n} \left< \Psi^{(N)}_{0} \middle| a_{m} \left[ \hat{H},a^{\dagger}_n \right] \middle| \Psi^{(N)}_{0} \right> c_{n;k}
        &= \Delta_{k} \sum_{n} \left< \Psi^{(N)}_{0} \middle| a_{m}a^{\dagger}_n \middle| \Psi^{(N)}_{0} \right> c_{n;k}

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
        Compute A = h_mn - \sum_p { h_{pn} \gamma_{pm} }
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
        """
        Compute M = \sum_n { \delta_nm - \gamma_{nm} }.

        """
        # M_mn = \delta_mn - \gamma_mn
        m = np.eye(self._n)
        m -= self._dm1
        return m


class ElectronAffinitiesEOM2(EOMState):
    """
    Electron Affinities EOM states for operator Q = \sum_n { c_n a^{\dagger}_n }.

    .. math::
        \sum_{n} \left< \Psi^{(N)}_{0} \middle| \Big\{ a_{m}, \left[ \hat{H},a^{\dagger}_n \right]\Big\} \middle| \Psi^{(N)}_{0} \right> c_{n;k}
        &= \Delta_{k} \sum_{n} \left< \Psi^{(N)}_{0} \middle| \Big\{a_{m},a^{\dagger}_n \Big\} \middle| \Psi^{(N)}_{0} \right> c_{n;k}

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
        Compute A = h_mn + \sum_{qr} { v_mqnr \gamma_qr}.

        """
        # A_mn = h_mn + <v_mqnr> \gamma_qr
        a = np.copy(self._h)
        a += np.einsum('mqnr,qr->mn', self._v, self._dm1)
        return a

    def _compute_rhs(self):
        """
        Compute M = \sum_n { \delta_nm }.

        """
        # M_mn = \delta_mn
        m = np.eye(self._n)
        return m


class ElectronAffinitiesEOM3(EOMState):
    """
    Electron Affinities EOM states for operator Q = \sum_n { c_n a^{\dagger}_n }.

    .. math::
        \sum_{n} \left< \Psi^{(N)}_{0} \middle| \left[ a_{m}, \left[ \hat{H},a^{\dagger}_n \right]\right] \middle| \Psi^{(N)}_{0} \right> c_{n;k}
        &= \Delta_{k} \sum_{n} \left< \Psi^{(N)}_{0} \middle| \left[a_{m},a^{\dagger}_n \right] \middle| \Psi^{(N)}_{0} \right> c_{n;k}

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
        Compute A = h_mn - 2\sum_p { h_{pn} \gamma_{pm} }
                  + \sum_{pqr} { v_pqrn \Gamma_pqmr }
                  + \sum_{qs} { v_mqns \gamma_qs}.

        """
        # A_mn = h_mn + <v_mqnr> \gamma_qr
        a = np.copy(self._h)
        a += np.einsum('mqnr,qr->mn', self._v, self._dm1)
        # A_mn -= 2 h_pn \gamma_pm
        a -= 2 * np.einsum('pm,pn->mn', self._dm1, self._h, )
        # A_mn += <v_pqrn> \Gamma_pqmr
        #      -= <v_pqnr> \Gamma_pqmr
        a -= 2 * np.einsum('pqmr,pqnr->mn', self._dm2, self._v, )
        return a

    def _compute_rhs(self):
        """
        Compute M = \sum_n { \delta_nm - 2 \gamma_{nm} }

        """
        # M_mn = \delta_mn - 2 \gamma_mn
        m = np.eye(self._n)
        m -= 2 * self._dm1
        return m
