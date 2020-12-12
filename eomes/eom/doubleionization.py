"""
Double electron removal Equations-of-motion.

"""


import numpy as np

from scipy.integrate import fixed_quad

from .base import EOMBase

__all__ = ["EOMDIP"]


class EOMDIP(EOMBase):
    r"""
    Double Ionization EOM state for operator
    :math:`Q = \sum_{ij} { c_{ij} a_i a_j}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| a^{\dagger}_k a^{\dagger}_l
        \left[ \hat{H}, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| a^{\dagger}_k a^{\dagger}_l
        \hat{Q} \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        return self._n ** 2

    def _compute_lhs(self):
        r"""
        Compute

        .. math::
            A = 2 ( h_{il} \delta_{jk} - h_{il} \gamma_{kj}
                + h_{ik} \gamma_{lj} - h_{ik} \delta_{jl} )\\
                + 2 ( \sum_q { h_{jq} \gamma_{lq} \delta_{ik}
                - h_{jq} \gamma_{kq} \delta_{il} } )\\
                + \left< ji||kl \right> + \sum_r { \left< ji||lr \right> \gamma_{kr}
                - \left< ji||kr \right> \gamma_{lr} }\\
                + 2 \sum_q \left< qj||kl \right> \gamma_{qi}\\
                + 2 ( \sum_{qr} { \left< iq||rk \right> \gamma_{qr} \delta_{lj}
                + \left< iq||lr \right> \gamma_{qr} \delta_{kj} })\\
                + 2 ( \sum_{qr} { \left< jq||rk \right> \Gamma_{qlri}
                + \left< jq||lr \right> \Gamma_{qkri} })\\
                + \sum_{qrs} { \left< qj||rs \right> \Gamma_{qlrs} \delta_{ki}
                + \left< jq||rs \right> \Gamma_{qkrs} \delta_{li} }\\
        """

        I = np.eye(self._n, dtype=self._h.dtype)

        # A_klji = 2 h_il \delta_jk - 2 h_il \gamma_jk + 2 h_ik \gamma_jl - 2 h_ik \delta_jl
        a = np.einsum("il,jk->klji", self._h, I, optimize=True)
        a -= np.einsum("il,jk->klji", self._h, self._dm1, optimize=True)
        a += np.einsum("ik,jl->klji", self._h, self._dm1, optimize=True)
        a -= np.einsum("ik,jl->klji", self._h, I, optimize=True)
        # A_klji += 2 \gamma_lq h_qj \delta_ki - 2 \gamma_kq h_qj \delta_li
        dm1h = np.einsum("ab,bc->ac", self._dm1, self._h, optimize=True)
        a += np.einsum("lj,ki->klji", dm1h, I, optimize=True)
        a -= np.einsum("kj,li->klji", dm1h, I, optimize=True)
        a *= 2
        # A_klji += <v_klji>
        a += self._v
        # A_klji += <v_jilr> \gamma_kr - <v_jikr> \gamma_lr + 2 <v_qjkl> \gamma_qi
        a += np.einsum("jilr,kr->klji", self._v, self._dm1, optimize=True)
        a -= np.einsum("jikr,lr->klji", self._v, self._dm1, optimize=True)
        a += 2 * np.einsum("qjkl,qi->klji", self._v, self._dm1, optimize=True)
        # A_klji += 2 ( <v_iqrk> \gamma_qr \delta_lj - <v_iqrl> \gamma_qr \delta_kj )
        vdm1 = np.einsum("abcd,bc->ad", self._v, self._dm1, optimize=True)
        a += 2 * np.einsum("ik,lj->klji", vdm1, I, optimize=True)
        a -= 2 * np.einsum("il,kj->klji", vdm1, I, optimize=True)
        # A_klji += 2 ( <v_jqrk> \Gamma_qlri + <v_jqlr> \Gamma_qkri )
        a += 2 * np.einsum("jqrk,qlri->klji", self._v, self._dm2, optimize=True)
        a += 2 * np.einsum("jqlr,qkri->klji", self._v, self._dm2, optimize=True)
        # A_klji += <v_qjrs> \Gamma_qlrs \delta_ki - <v_qjrs> \Gamma_qkrs \delta_li
        vdm2 = np.einsum("abcd,aecd->be", self._v, self._dm2, optimize=True)
        a += np.einsum("jl,ki->klji", vdm2, I, optimize=True)
        a -= np.einsum("jk,li->klji", vdm2, I, optimize=True)
        return a.reshape(self._n ** 2, self._n ** 2)

    def _compute_rhs(self):
        r"""
        Compute :math:`M = \Gamma_{klji}`.
        """

        # M_klji = \Gamma_klji
        m = np.copy(self._dm2)
        return m.reshape(self._n ** 2, self._n ** 2)

    def compute_tdm(self, coeffs):
        r"""
        Compute

        .. math::
            \gamma_{kl} = \sum_{ij}{c_{ij} \left< \Psi^{(N)}_0 \middle|
            a^{\dagger}_k a^{\dagger}_l a_i a_j \middle|
            \Psi^{(N)}_0 \right>}

        """
        coeffs = coeffs.reshape(self._n ** 2, self._n, self._n)
        return np.einsum(
            "nij,pqij->npq",
            coeffs,
            self._rhs.reshape(self._n, self._n, self._n, self._n),
        )
