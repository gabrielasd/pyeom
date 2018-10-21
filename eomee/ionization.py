import numpy as np

from eomee.base import EOMState


__all__ = [
    'IonizationEOMState',
    ]


class IonizationEOMState(EOMState):
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
        Compute A = \sum_q { -h_{nq} \gamma_{mq} }
                  - 2 \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqrs} }.

        """
        # A_mn = -h_nq \gamma_mq + 0.5 (v_pnrs \Gamma_mprs - v_nqrs \Gamma_mqrs)
        #      = -h_nq \gamma_mq + 0.5 (v_qnrs - v_nqrs) \Gamma_mqrs
        #      = -h_nq \gamma_mq - 2 <v_nqrs> \Gamma_mqrs
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
