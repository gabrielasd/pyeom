import numpy as np

from eomee.base import EOMState


__all__ = [
    'ExcitationEOM',
]


class ExcitationEOM(EOMState):
    """
    Excitation EOM state for operator Q = \sum_{ij} { c_{ij} a^{\dagger}_i  a_j}.
    .. math::
        \left< \Psi^{(N)}_0 \middle| [a^{\dagger}_k  a_l, [\hat{H}, \hat{Q}]] \middle| \Psi^{(N)}_0 \right>
        &= \omega_{\lambda} \left< \Psi^{(N)}_0 \middle| [a^{\dagger}_k a_l \hat{Q}] \Psi^{(N)}_0 \right>

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
        Compute A = h_{li} \gamma_{kj} + h_{jk} \gamma_{il}
                  - \sum_q { h_{jq} \delta_{il} \gamma_{kq}}
                  - \sum_q { h_{qi} \delta_{jk} \gamma_{ql}}
                  + \sum_{qs} { \left< lq||si \right> \Gamma_{kqsj} }
                  + \sum_{qs} { \left< jq||sk \right>  \Gamma_{iqsl} }
                  - 0.5 \sum_{qrs} { \delta_{il} \left< jq||rs \right> \Gamma_{kqrs} }
                  - 0.5 \sum_{pqs} { \delta_{jk} \left< pq||si \right> \Gamma_{pqsl}}
                  + 0.5 \sum_{pq} { \left< pq||ik\right>  \Gamma_{pqlj} }
                  + 0.5 \sum_{rs} { \left< jl||rs \right> \Gamma_{kirs} }

        """
        hdm1 = np.dot(self._h, self._dm1)
        I = np.eye(self._n, dtype=self._h.dtype)

        # A_klij = h_li \gamma_kj + h_jk \gamma_il
        b = np.einsum('li,kj->klji', self.h, self.dm1, optimize=True)
        b += np.einsum('jk,il->klji', self.h, self.dm1, optimize=True)
        # A_klij -= ( \delta_il h_jq \gamma_qk + \delta_jk h_iq \gamma_ql )
        b -= np.einsum('il,jk->klji', I, hdm1, optimize=True)
        b -= np.einsum('jk,il->klji', I, hdm1, optimize=True)
        # A_klij += <v_lqsi> \Gamma_kqsj
        b += np.einsum('lqsi,kqsj->klji', self.v, self.dm2, optimize=True)
        # b += 0.5 * np.einsum('lpsi,kpsj->klij', self.v, self.dm2)
        # A_klij += <v_jqsk> \Gamma_iqsl
        b += np.einsum('jqsk,iqsl->klji', self.v, self.dm2, optimize=True)
        # b += 0.5 * np.einsum('jpsk,ipsl->klij', self.v, self.dm2)
        # A_klij += 0.5 ( <v_pqik> \Gamma_pqlj )
        b += 0.5 * np.einsum('pqik,pqlj->klji', self.v, self.dm2, optimize=True)
        # A_klij -= 0.5 ( <v_ljrs> \Gamma_kirs )
        b -= 0.5 * np.einsum('ljrs,kirs->klji', self.v, self.dm2, optimize=True)
        # A_klij -= 0.5 ( \delta_il <v_jqrs> \Gamma_kqrs )
        vdm2 = np.einsum('jqrs,kqrs->jk', self.v, self.dm2, optimize=True)
        b -= 0.5 * np.einsum('il,jk->klji', I, vdm2, optimize=True)
        # A_klij -= 0.5 ( \delta_jk <v_pqsi> \Gamma_pqsl )
        vdm2 = np.einsum('pqsi,pqsl->il', self.v, self.dm2, optimize=True)
        b -= 0.5 * np.einsum('jk,il->klji', I, vdm2, optimize=True)
        return b.reshape(self._n**2, self._n**2)

    def _compute_rhs(self):
        """
        Compute M = \gamma_{kj} \delta_{li} - \Gamma_{kijl}.

        """
        I = np.eye(self._n, dtype=self._h.dtype)

        # M_klij = \gamma_kj \delta_li - \Gamma_kijl
        m = np.einsum('kj,li->klji', self.dm1, I, optimize=True)
        m -= np.einsum('kijl->klji', self.dm2, optimize=True)
        return m.reshape(self._n**2, self._n**2)
