import numpy as np

from eomee.base import EOMState


__all__ = [
    'ExcitationEOM',
    ]


class ExcitationEOM(EOMState):
    """
    Excitation EOM state for operator Q = \sum_ij { c_ij a^{\dagger}_i  a_j}.
    <\Psi| [a^{\dagger}_k  a_l, [H, Q]] |\Psi> = \omega <\Psi| [a^{\dagger}_k  a_l, Q] |\Psi>

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
        Compute A = - \sum_{ij,q} { h_jq \delta_il \gamma_kq}
                  - \sum_{ij,p} { h_pi \delta_jk \gamma_pl}
                  + \sum_{ij} { h_li \gamma_kj}
                  + \sum_{ij} { h_jk \gamma_il}
                  + 0.5 \sum_{ij,qrs} { \delta_il <jq||rs> \Gamma_kqrs}
                  + 0.5 \sum_{ij,pqs} { \delta_jk <pq||is> \Gamma_pqsl}
                  + 0.5 \sum_{ij,pq} { <pq||ik> \Gamma_pqlj}
                  + 0.5 \sum_{ij,qs} { <lq||si> \Gamma_kqsj}
                  + 0.5 \sum_{ij,ps} { <lp||si> \Gamma_kpsj}
                  + 0.5 \sum_{ij,qs} { <jq||sk> \Gamma_iqsl}
                  + 0.5 \sum_{ij,rs} { <jl||rs> \Gamma_ikrs}
                  + 0.5 \sum_{ij,ps} { <jp||sk> \Gamma_ipsl}

        """
        hdm1 = np.dot(self._h, self._dm1)
        I = np.eye(self._n, dtype=self._h.dtype)

        # A_klij = -\delta_il [h_jq \gamma_qk]_jk - \delta_jk [h_ip \gamma_pl]_il 
        #        = - [\delta_il hgamma_jk]_ijlk - [\delta_jk hgamma_il]_jikl
        b = -np.einsum('il,jk->klij', I, hdm1)
        b -= np.einsum('jk,il->klij', I, hdm1)
        # A_klij += [h_li \gamma_kj]_lkij + [h_jk \gamma_il]_jikl
        b += np.einsum('li,kj->klij', self.h, self.dm1)
        b += np.einsum('jk,il->klij', self.h, self.dm1)
        # A_klij -= 0.5 * [\delta_il (v_jqrs \Gamma_kqrs)_jk]_ijlk       
        vdm2 = np.einsum('jqrs,kqrs->jk', self.v, self.dm2)
        b -= 0.5 * np.einsum('il,jk->klij', I, vdm2)
        # A_klij -= 0.5 * [\delta_jk (v_pqsi \Gamma_pqsl)_il]_jikl
        vdm2 = np.einsum('pqis,pqsl->il', self.v, self.dm2)
        b -= 0.5 * np.einsum('jk,il->klij', I, vdm2)
        # A_klij += 0.5 * (v_pqik \Gamma_pqlj)_iklj
        b += 0.5 * np.einsum('pqik,pqlj->klij', self.v, self.dm2)
        # A_klij += 0.5 * (v_lqsi \Gamma_kqsj)_likj
        #         + 0.5 * (v_lpsi \Gamma_kpsj)_likj
        b += 0.5 * np.einsum('lqsi,kqsj->klij', self.v, self.dm2)
        b += 0.5 * np.einsum('lpsi,kpsj->klij', self.v, self.dm2)
        # A_klij += 0.5 * (v_jqsk \Gamma_iqsl)_jkil
        b += 0.5 * np.einsum('jqsk,iqsl->klij', self.v, self.dm2)
        # A_klij -= 0.5 * (v_jlrs \Gamma_ikrs)_jlik
        b -= 0.5 * np.einsum('jlrs,ikrs->klij', self.v, self.dm2)
        # A_klij += 0.5 * (v_jpsk \Gamma_ipsl)_jkil
        b += 0.5 * np.einsum('jpsk,ipsl->klij', self.v, self.dm2)
        return b.reshape(self._n**2, self._n**2)

    def _compute_rhs(self):
        """
        Compute M = \sum_ij { \gamma_{kj} \delta_{li} - \delta_{jk} \gamma_{il} }.

        """
        I = np.eye(self._n, dtype=self._h.dtype)

        # M_klij = [\delta_li \gamma_kj]_lkij
        m = np.einsum('li,kj->klij', I, self.dm1)
        return m.reshape(self._n**2, self._n**2)
