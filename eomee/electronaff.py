import numpy as np

from eomee.base import EOMState


__all__ = [
    'ElectronAffinitiesEOM1',
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
        Compute A = h_mn + \sum_p { -h_{pn} \gamma_{pm} }
                  + \sum_{pqs} { v_pqns \Gamma_pqsm - v_pqsn \Gamma_pqsm }
                  + 2 \sum_{qs} { v_mqns \gamma_qs + v_qmns \gamma_qs}. 

        """
        # A_mn = h_mn - h_pn \gamma_pm + v_pqns \Gamma_pqsm - v_pqsn \Gamma_pqsm
        #      + 2 v_mqns \gamma_qs - 2 v_qmns \gamma_qs
        #      = h_mn - \gamma_mp h_pn + \Gamma_pqsm v_pqns - \Gamma_pqsm v_pqsn
        #      + 2 v_mqns \gamma_qs - 2 v_qmns \gamma_qs
        a = self._h
        a -= np.dot(self._dm1, self._h)
        a += np.tensordot(self._dm2, self._v, axes=((0, 1, 2), (0, 1, 3)))
        a -= np.tensordot(self._dm2, self._v, axes=((0, 1, 2), (0, 1, 2)))
        b = np.tensordot(self._v, self._dm1, axes=((1, 3), (0, 1)))
        b *= 2
        a += b
        b = np.tensordot(self._v, self._dm1, axes=((0, 3), (0, 1)))
        b *= 2
        a -= b
        return b

    def _compute_rhs(self):
        """
        Compute M = \sum_n { \delta_mn - \gamma_{nm} }.

        """
        # M_mn = \delta_mn - \gamma_mn
        m = np.eye(self._n)
        m -= self._dm1
        return m