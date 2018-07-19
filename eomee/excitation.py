import numpy as np

from eomee.base import EOMState


__all__ = [
    'ExcitationEOM',
    ]


class ExcitationEOM(EOMState):
    """
    Ionization EOM state for operator Q = \sum_ij { c_ij a^{\dagger}_i  a_j}.

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
        # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
        return self._n**2

    def _compute_lhs(self):
        """
        Compute A = h_li \gamma_kj + h_jk \gamma_il
                  - \sum_{pq} { h_pq \delta_li \delta_jp \gamma_kq }
                  - \sum_{pq} { h_pq \delta_qi \delta_jk \gamma_pl }
                  + \sum_{pq} { v_lpiq \Gamma_kpjq + v_jpkq \Gamma_iplq }
                  + 2 \sum_{ps} { v_plis \Gamma_pkjs + v_pjks \Gamma_pils }
                  - \sum_{pr} { v_plri \Gamma_pkjr + v_pjrk \Gamma_pilr }
                  + \sum_{rs} { v_jlrs \Gamma_iksr - v_ljrs \Gamma_iksr }
                  + \sum_{pq} { v_ikpq \Gamma_ljpq - v_kipq \Gamma_ljpq }
                  - \sum_{pqrs} { v_{pqrs} \delta_jp \delta_li \Gamma_{kqrs} }
                  - \sum_{pqrs} { v_{pqrs} \delta_ri \delta_kj \Gamma_{pqls} }
                  + \sum_{pqrs} { v_{pqrs} \delta_jq \delta_il \Gamma_{pksr} }
                  + \sum_{pqrs} { v_{pqrs} \delta_is \delta_jk \Gamma_{pqlr} }

        """
        dm1h = np.dot(self._dm1, self._h)
        I = np.eye(self._n)

        # A_klij = h_li \gamma_kj + h_jk \gamma_il
        #        = \gamma_kj h_li + h_kj \gamma_li
        b = np.kron(self._dm1, self._h)
        b += np.kron(self._h, self._dm1)
        b = b.reshape(self._n, self._n, self._n, self._n).transpose(0, 1, 3, 2)
        # A_klij -= h_jq \delta_li \gamma_kq + h_pi \delta_jk \gamma_pl
        #        -= \gamma_kq h_qj \delta_li + \delta_kj \gamma_lp h_pi
        a = np.kron(dm1h, I)
        a += np.kron(I, dm1h)
        a = a.reshape(self._n, self._n, self._n, self._n).transpose(0, 1, 3, 2)
        b -= a
        # A_klij += v_lpiq \Gamma_kpjq + v_jpkq \Gamma_iplq
        #        += \Gamma_kpjq v_lpiq + v_kqjp \Gamma_lqip
        a = np.tensordot(self._dm2, self._v, axes=([1,3], [1, 3]))
        a += np.tensordot(self._v, self._dm2, axes=([1,3], [1, 3]))
        a = a.transpose(0, 2, 3, 1)
        b += a
        # A_klij += 2 * (v_plis \Gamma_pkjs + v_pjks \Gamma_pils )
        #        += 2 * ( \Gamma_pkjs v_plis + v_kspj \Gamma_lspi)
        a = np.tensordot(self._dm2, self._v, axes=([0,3], [0, 3]))
        a += np.tensordot(self._v, self._dm2, axes=([1,2], [1, 2]))
        a = a.transpose(0, 2, 3, 1)
        a *= 2
        b += a
        # A_klij -= v_plri \Gamma_pkjr + v_pjrk \Gamma_pilr
        #        -= \Gamma_pkjr v_plri + v_rkpj \Gamma_lrpi
        a = np.tensordot(self._dm2, self._v, axes=([0,3], [0, 2]))
        a += np.tensordot(self._v, self._dm2, axes=([1,2], [1, 2]))
        a = a.transpose(0, 2, 3, 1)
        b -= a
        # A_klij += v_jlrs \Gamma_iksr - v_ljrs \Gamma_iksr
        #        += \Gamma_iksr v_jlrs - \Gamma_iksr v_ljrs
        a = np.tensordot(self._dm2, self._v, axes=([2,3], [3, 2])).transpose(1, 3, 0, 2)
        a -= np.tensordot(self._dm2, self._v, axes=([2,3], [3, 2])).transpose(1, 2, 0, 3)
        b += a
        # A_klij += v_ikpq \Gamma_ljpq - v_kipq \Gamma_ljpq
        a = np.tensordot(self._v, self._dm2, axes=([2,3], [2, 3])).transpose(1, 2, 0, 3)
        a -= np.tensordot(self._v, self._dm2, axes=([2,3], [2, 3])).transpose(0, 2, 1, 3)
        b += a
        # A_klij -= v_jqrs \delta_li \Gamma_kqrs + v_pqis \delta_kj \Gamma_pqls
        #        -= (\Gamma_kqrs v_jqrs) \delta_li + \delta_kj (\Gamma_pqls v_pqis)
        a = np.dot(self._dm2, self._v, axes=([1, 2, 3], [1, 2, 3]))
        a = np.kron(a, I).reshape(self._n, self._n, self._n, self._n)
        a = a.transpose(0, 1, 3, 2)
        b -= a
        a = np.dot(self._dm2, self._v, axes=([0, 1, 3], [0, 1, 3]))
        a = np.kron(I, a).reshape(self._n, self._n, self._n, self._n)
        a = a.transpose(0, 1, 3, 2)
        b -= a
        # A_klij += v_pjrs \delta_il \Gamma_pksr + v_pqri \delta_jk \Gamma_pqlr
        #        += (\Gamma_pksr v_pjrs) \delta_li + \delta_kj (\Gamma_pqlr v_pqri)
        a = np.dot(self._dm2, self._v, axes=([0, 2, 3], [0, 2, 3]))
        a = np.kron(a, I).reshape(self._n, self._n, self._n, self._n)
        a = a.transpose(0, 1, 3, 2)
        b += a
        a = np.dot(self._dm2, self._v, axes=([0, 1, 3], [0, 1, 2]))
        a = np.kron(I, a).reshape(self._n, self._n, self._n, self._n)
        a = a.transpose(0, 1, 3, 2)
        b += a
        return b.reshape(self._n**2, self._n**2)

    def _compute_rhs(self):
        """
        Compute M = \sum_ij { \gamma_{kj} \delta_{li} - \delta_{jk} \gamma_{il} }.

        """
        I = np.eye(self._n)

        # M_klij = \gamma_kj \delta_li - \delta_jk \gamma_il
        #        = \gamma_kj \delta_li - \delta_kj \gamma_li
        m = np.kron(self._dm1, I).reshape(self._n, self._n, self._n, self._n)
        n -= np.kron(I, self._dm1).reshape(self._n, self._n, self._n, self._n)
        m = m.transpose(0, 1, 3, 2)
        return m.reshape(self._n**2, self._n**2)
