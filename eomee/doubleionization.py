import numpy as np

from eomee.base import EOMState

__all__ = [
    "DoubleElectronRemovalEOM"
]


class DoubleElectronRemovalEOM(EOMState):
    """

    """

    @property
    def neigs(self):
        return self._n**2

    def _compute_lhs(self):
        """
        Compute A = 2 [h_{il} \delta_{jk} - h_{il} \gamma_{kj} + h_{ik} \gamma_{lj} - h_{ik} \delta_{jl}]
                  + 2[\sum_{q} (h_{jq} \gamma_{lq} \delta_{ik} - h_{jq} \gamma_{kq} \delta_{il}) ]
                  + \nu_{jikl} + \sum_{r} (\nu_{jilr} \gamma_{kr} - \nu_{jikr} \gamma_{lr})
                  + 2 \sum_{q} \nu_{qjkl} \gamma_{qi}
                  + 2 [\sum_{qr} (\nu_{iqrk} \gamma_{qr} \delta_{lj} + \nu_{iqlr} \gamma_{qr} \delta_{kj})]
                  + 2 [\sum_{qr} (\nu_{jqrk} \Gamma_{qlri} + \nu_{jqlr} \Gamma_{qkri})]
                  + \sum_{qrs} (\nu_{qjrs} \Gamma_{qlrs} \delta_{ki} + \nu_{jqrs} \Gamma_{qkrs} \delta_{li})
        """

        I = np.eye(self._n, dtype=self._h.dtype)

        # A_klji = 2 h_il \delta_jk - 2 h_il \gamma^T_jk + 2 h_ik \gamma^T_jl - 2 h_ik \delta_jl
        a = np.einsum('il,jk->klji', self._h, I)
        a -= np.einsum('il,jk->klji', self._h, self._dm1)
        a += np.einsum('ik,jl->klji', self._h, self._dm1)
        a -= np.einsum('ik,jl->klji', self._h, I)
        # A_klji += 2 \gamma_lq h^T_qj \delta^T_ki - 2 \gamma_kq h^T_qj \delta^T_li
        dm1h = np.einsum('ab,bc->ac', self._dm1, self._h)
        a += np.einsum('lj,ki->klji', dm1h, I)
        a -= np.einsum('kj,li->klji', dm1h, I)
        a *= 2
        # A_klji += v^T_klji
        a += self._v
        # A_klji += (v_jilr \gamma_kr - v_jikr \gamma_lr) + 2 v_qjkl \gamma_qi
        a += np.einsum('jilr,kr->klji', self._v, self._dm1)
        a -= np.einsum('jikr,lr->klji', self._v, self._dm1)
        a += 2 * np.einsum('qjkl,qi->klji', self._v, self._dm1)
        # A_klji += 2 (v_iqrk \gamma_qr \delta_lj - v^T_iqrl \gamma_qr \delta_kj)
        vdm1 = np.einsum('abcd,bc->ad', self._v, self._dm1)
        a += 2 * np.einsum('ik,lj->klji', vdm1, I)
        a -= 2 * np.einsum('il,kj->klji', vdm1, I)
        # A_klji += 2 (v_jqrk \Gamma_qlri + v_jqlr \Gamma_qkri)
        a += 2 * np.einsum('jqrk,qlri->klji', self._v, self._dm2)
        a += 2 * np.einsum('jqlr,qkri->klji', self._v, self._dm2)
        # A_klji += v_qjrs \Gamma_qlrs \delta_ki - v^T_qjrs \Gamma_qkrs \delta_li
        vdm2 = np.einsum('abcd,aecd->be', self._v, self._dm2)
        a += np.einsum('jl,ki->klji', vdm2, I)
        a -= np.einsum('jk,li->klji', vdm2, I)
        return a.reshape(self._n**2, self._n**2)

    def _compute_rhs(self):
        """
        Compute M = \sum_ij {\Gamma_klji} c_ij
        """

        # M_klji = \Gamma_klji
        m = self._dm2
        return m.reshape(self._n**2, self._n**2)
