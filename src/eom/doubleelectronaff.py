import numpy as np

from eomee.base import EOMState

__all__ = ["DoubleElectronAttachmentEOM"]


class DoubleElectronAttachmentEOM(EOMState):
    """

    """

    @property
    def neigs(self):
        return self._n ** 2

    def _compute_lhs(self):
        """
        Compute A = 2 (h_{li} \delta_{kj} - h_{ki} \delta_{lj})
                  + 2 (h_{ki} \gamma_{jl} - h_{li} \gamma_{jk})
                  + 2 \sum_{p} (h_{pi} \gamma_{pk} \delta_{lj} + h_{pj} \gamma_{pl} \delta_{ki})
                  + \left< lk||ij \right>
                  + \sum_{q} (\left< ql||ij \right> \gamma_{qk} - \left< qk||ij \right> \gamma_{ql})
                  + 2 \sum_{r} \left< lk||jr \right> \gamma_{ir}
                  + 2 \sum_{qr} \gamma_{qr}(\left< ql||jr \right> \delta_{ki} - \left< qk||jr \right> \delta_{li})
                  + 2 \sum_{qr} (\left< ql||ir \right> \Gamma_{qjrk} - \left< qk||ir \right> \Gamma_{qjrl})
                  + \sum_{pqr} \left< pq||jr \right> (\Gamma_{pqrk} \delta_{li} - \Gamma_{pqrl} \delta_{ki})
        """

        I = np.eye(self._n, dtype=self._h.dtype)

        # A_klji = 2 (h_li \delta_kj - h_ki \delta_lj)
        #       += 2 (h_ki \gamma_lj - h_li \gamma_kj)
        a = np.einsum("kj,li->klji", I, self._h)
        a -= np.einsum("ki,lj->klji", self._h, I)
        a += np.einsum("ki,lj->klji", self._h, self._dm1)
        a -= np.einsum("kj,li->klji", self._dm1, self._h)
        # A_klji += 2 (h_ip \gamma_pk \delta_lj + h_jp \gamma_pl \delta_ki)
        hdm1 = np.einsum("ab,bc->ac", self._h, self._dm1)
        a += np.einsum("ik,lj->klji", hdm1, I)
        a += np.einsum("jl,ki->klji", hdm1, I)
        # A_klji += 2 <v_lkjr> \gamma_ir
        a += np.einsum("lkjr,ir->klji", self._v, self._dm1)
        # A_klji += 2 (<v_qljr> \gamma_qr \delta_ki - <v_qkjr> \gamma_qr \delta_li)
        vdm1 = np.einsum("abcd,ad->bc", self._v, self._dm1)
        a += np.einsum("lj,ki->klji", vdm1, I)
        a -= np.einsum("kj,li->klji", vdm1, I)
        # A_klji += 2 (<v_qlir> \Gamma_qjrk - <v_qkir> \Gamma_qjrl)
        a += np.einsum("qlir,qjrk->klji", self._v, self._dm2)
        a -= np.einsum("qkir,qjrl->klji", self._v, self._dm2)
        a *= 2
        # A_klji += <v_klji>
        a += self._v
        # A_klji += <v_qlij> \gamma_qk - <v_qkij> \gamma_ql
        a += np.einsum("qlij,qk->klji", self._v, self._dm1)
        a -= np.einsum("qkij,ql->klji", self._v, self._dm1)
        # A_klji += <v_pqjr> \Gamma_pqrk \delta_li - <v_pqjr> \Gamma_pqrl \delta_ki
        #         = -<v_pqrj> \Gamma_pqrk \delta_li + <v_pqrj> \Gamma_pqrl \delta_ki
        vdm2 = np.einsum("abcd,abce->de", self._v, self._dm2)
        a -= np.einsum("jk,li->klji", vdm2, I)
        a += np.einsum("jl,ki->klji", vdm2, I)
        return a.reshape(self._n ** 2, self._n ** 2)

    def _compute_rhs(self):
        """
        Compute M = \Gamma_{ijlk}
                  + \delta_{li} \delta_{kj} - \delta_{ki} \delta_{lj}
                  + \delta_{ki} \gamma_{jl} - \delta_{kj} \gamma_{li}
                  + \delta_{lj} \gamma_{ki} - \delta_{li} \gamma_{jk}
        """
        I = np.eye(self._n, dtype=self._h.dtype)
        # M_klji = \delta_li \delta_kj - \delta_ki \delta_lj
        m = np.einsum("li,kj->klji", I, I)
        m -= np.einsum("ki,lj->klji", I, I)
        # M_klji += \delta_{ki} \gamma_{jl} - \delta_{kj} \gamma_{li}
        #        += \delta_{lj} \gamma_{ki} - \delta_{li} \gamma_{jk}
        m += np.einsum("ki,lj->klji", I, self._dm1)
        m -= np.einsum("kj,li->klji", I, self._dm1)
        m -= np.einsum("li,kj->klji", I, self._dm1)
        m += np.einsum("lj,ki->klji", I, self._dm1)
        # M_klji += \Gamma_klji
        m += self._dm2
        return m.reshape(self._n ** 2, self._n ** 2)
