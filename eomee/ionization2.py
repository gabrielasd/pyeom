import numpy as np

from eomee.base import EOMState


__all__ = [
    'IonizationDoubleCommutator',
]


class IonizationDoubleCommutator(EOMState):
    """
    Ionization EOM state for operator Q = \sum_n { c_n a_n }.

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
        # A_mn += v^T_msnq \gamma_sq - \Gamma^T_mqrs v_nqrs
        a += np.einsum('msnq,sq', self._v, self._dm1, optimize=True)
        a -= np.einsum('mqrs,nqrs', self._dm2, self._v, optimize=True)
        return a

    def _compute_rhs(self):
        """
        Compute M = 2 \gamma_{mn} - \delta_{nm}.

        """
        # M_mn = 2 * \gamma_mn - \delta_mn
        m = 2 * np.copy(self._dm1)
        m -= np.eye(self._n, dtype=self._dm1.dtype)
        return m
