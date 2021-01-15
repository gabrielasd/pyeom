import numpy as np

# from scipy.integrate import fixed_quad
from scipy.integrate import quad as integrate

from eomee.base import EOMState
from eomee.tools import antisymmetrize

__all__ = ["DoubleElectronRemovalEOM"]


class DoubleElectronRemovalEOM(EOMState):
    """
    Double Ionization EOM state for operator Q = \sum_{ij} { c_{ij} a_i a_j}.
    .. math::
        \left< \Psi^{(N)}_0 \middle| a^{\dagger}_k a^{\dagger}_l \left[ \hat{H}, \hat{Q} \right] \middle| \Psi^{(N)}_0 \right>
        &= \Delta_k \left< \Psi^{(N)}_0 \middle| a^{\dagger}_k a^{\dagger}_l \hat{Q} \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        return self._n ** 2

    def _compute_lhs(self):
        """
        Compute A = 2 ( h_{il} \delta_{jk} - h_{il} \gamma_{kj} + h_{ik} \gamma_{lj} - h_{ik} \delta_{jl} )
                  + 2 ( \sum_q { h_{jq} \gamma_{lq} \delta_{ik} - h_{jq} \gamma_{kq} \delta_{il} } )
                  + \left< ji||kl \right> + \sum_r { \left< ji||lr \right> \gamma_{kr} - \left< ji||kr \right> \gamma_{lr} }
                  + 2 \sum_q \left< qj||kl \right> \gamma_{qi}
                  + 2 ( \sum_{qr} { \left< iq||rk \right> \gamma_{qr} \delta_{lj} + \left< iq||lr \right> \gamma_{qr} \delta_{kj} })
                  + 2 ( \sum_{qr} { \left< jq||rk \right> \Gamma_{qlri} + \left< jq||lr \right> \Gamma_{qkri} })
                  + \sum_{qrs} { \left< qj||rs \right> \Gamma_{qlrs} \delta_{ki} + \left< jq||rs \right> \Gamma_{qkrs} \delta_{li} }
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
        """
        Compute M = \Gamma_{klji}
        """

        # M_klji = \Gamma_klji
        m = np.copy(self._dm2)
        return m.reshape(self._n ** 2, self._n ** 2)

    @classmethod
    def erpa(cls, h_0, v_0, h_1, v_1, dm1, dm2, nint=50, *args, **kwargs):
        """
        Compute the ERPA correlation energy for the operator.

        """
        # Size of dimensions
        n = h_0.shape[0]
        # H_1 - H_0
        dh = h_1 - h_0
        # V_1 - V_0
        dv = v_1 - v_0

        # Linear term (eq. 20)
        # dh_pq * \gamma_pq
        linear = np.einsum("pq,pq", dh, dm1, optimize=True)

        # Nonlinear term (eq. 20 integrand)
        @np.vectorize
        def nonlinear(alpha):
            # Compute H^alpha
            h = alpha * dh
            h += h_0
            v = alpha * dv
            v += v_0
            # Antysymmetrize v_pqrs
            v = antisymmetrize(v)
            # Solve EOM equations
            c = (
                cls(h, v, dm1, dm2)
                .solve_dense(*args, **kwargs)[1]
                .reshape(n ** 2, n, n)
            )
            # Compute transition RDMs (eq. 32)
            # \gamma_m;pq = c_m;rs * \Gamma_pqrs
            rdms = np.einsum("mrs,pqrs->mpq", c, dm2)
            # Compute nonlinear energy term
            # dv_pqrs * {sum_{m}{\gamma_m;ps * \gamma_m;qr}}_pqrs
            tv = np.zeros_like(dm2)
            for rdm in rdms:
                tv += np.einsum("ps,qr->pqrs", rdm, rdm, optimize=True)
            return np.einsum("pqrs,pqrs", dv, tv, optimize=True)

        # Compute ERPA correlation energy (eq. 20)
        # return linear + 0.5 * fixed_quad(nonlinear, 0, 1, n=nint)[0]
        return (
            linear
            - 0.5
            * integrate(nonlinear, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.49e-04)[
                0
            ]
        )
