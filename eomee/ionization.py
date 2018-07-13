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
                  + 2 \sum_{qrs} { v_{qnrs} \Gamma_{mqrs} }.

        """
        # A_mn = -h_nq \gamma_mq + 2 v_qnrs \Gamma_mqrs
        #      = -\gamma_mq h_qn + 2 \Gamma_mqrs v_qnrs
        a = np.dot(self._dm1, self._h)
        b = np.tensordot(self._dm2, self._v, axes=((3, 2, 1), (3, 2, 0)))
        b *= 2
        b -= a
        return b

    def _compute_rhs(self):
        """
        Compute M = \sum_n { \gamma_{mn} }.

        """
        # M_mn = \gamma_mn
        return np.copy(self._dm1)
