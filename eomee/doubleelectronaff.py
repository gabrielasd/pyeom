# This file is part of EOMEE.
#
# EOMEE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# EOMEE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

r"""Double electron affinity EOM state class."""


import numpy as np

from scipy.integrate import quad as integrate

from .base import EOMState
from .tools import antisymmetrize, pickpositiveeig


__all__ = ["EOMDEA", "EOMDEA_2"]


class EOMDEA(EOMState):
    r"""
    Double electron  attachment EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} a^{\dagger}_i a^{\dagger}_j}`.

    .. math::

        \left< \Psi^{(N)}_0 \middle| \left[a_k a_l, \left[ \hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_k \left< \Psi^{(N)}_0 \middle| [a_k a_l, \hat{Q}] \middle| \Psi^{(N)}_0 \right>
    """

    @property
    def neigs(self):
        r""" """
        return self._n ** 2

    def _compute_lhs(self):
        r"""
        Compute

        .. math::
            A_{klji} = 2 (h_{li} \delta_{kj} - h_{ki} \delta_{lj}) + 2 (h_{ki} \gamma_{jl} - h_{li} \gamma_{jk})
            + 2 \sum_{p} (h_{pi} \gamma_{pk} \delta_{lj} + h_{pj} \gamma_{pl} \delta_{ki}) + \left< lk||ij \right>
            + \sum_{q} (\left< ql||ij \right> \gamma_{qk} - \left< qk||ij \right> \gamma_{ql})
            + 2 \sum_{qr} \gamma_{qr}(\left< ql||jr \right> \delta_{ki} - \left< qk||jr \right> \delta_{li})
            + 2 \sum_{qr} (\left< ql||ir \right> \Gamma_{qjrk} - \left< qk||ir \right> \Gamma_{qjrl})
            + \sum_{pqr} \left< pq||jr \right> (\Gamma_{pqrk} \delta_{li} - \Gamma_{pqrl} \delta_{ki})
            + 2 \sum_{r} \left< lk||jr \right> \gamma_{ir}s

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
        r"""
        Compute

        .. math::

            M_{klji} = 2\delta_{li} \delta_{kj} - 2\delta_{li} \gamma_{jk} - 2\delta_{kj} \gamma_{li}

            - \delta_{i k} \delta_{j l} + \delta_{i k} \left\{a^\dagger_{j} a_{l}\right\} + \delta_{i l} \delta_{j k} - \delta_{i l} \left\{a^\dagger_{j} a_{k}\right\} - \delta_{j k} \left\{a^\dagger_{i} a_{l}\right\} + \delta_{j l} \left\{a^\dagger_{i} a_{k}\right\}

        """
        I = np.eye(self._n, dtype=self._h.dtype)
        # # M_klji = 2\delta_li \delta_kj
        # m = 2*np.einsum("li,kj->klji", I, I)
        # # M_klji -= \delta_{ki} \gamma_{jl} - \delta_{kj} \gamma_{li}
        # m -= 2*np.einsum("li,kj->klji", I, self._dm1)
        # m -= 2*np.einsum("kj,li->klji", I, self._dm1)

        # M_klji = \delta_{i l} \delta_{j k} -\delta_{i k} \delta_{j l}
        m = np.einsum("il,jk->klji", I, I, optimize=True) - np.einsum("ik,jl->klji", I, I, optimize=True)
        # M_klji = + \delta_{i k} \gamma_{j l} - \delta_{i l} \gamma_{j k}
        m += np.einsum("ik,jl->klji", I, self._dm1, optimize=True) - np.einsum("il,jk->klji", I, self._dm1, optimize=True)
        # M_klji = + \delta_{j l} \gamma_{i k} - \delta_{j k} \gamma_{i l}
        m += np.einsum("jl,ik->klji", I, self._dm1, optimize=True) - np.einsum("jk,il->klji", I, self._dm1, optimize=True)

        return m.reshape(self._n ** 2, self._n ** 2)


    @classmethod
    def erpa(cls, h_0, v_0, h_1, v_1, dm1, dm2, nint=50, *args, **kwargs):
        r"""
        Compute the ERPA correlation energy for the operator.

        """
        # Size of dimensions
        n = h_0.shape[0]
        # H_1 - H_0
        dh = h_1 - h_0
        # V_1 - V_0
        dv = v_1 - v_0
        # \delta_pr * \gamma_qs
        eye_dm1 = np.einsum("pr,qs->pqrs", np.eye(n), dm1, optimize=True)
        # \delta_qs * \gamma_pr
        dm1_eye = np.einsum("pr,qs->pqrs", dm1, np.eye(n), optimize=True)

        # Compute inmutable terms in (eq. 35)
        # There is a sign error in the equation. It
        # should be:
        # = \delta_pi * \delta_qj - \delta_pj * \delta_qi
        # - \delta_pi * \gamma_qj - \delta_qj * \gamma_pi
        # + \delta_pj * \gamma_qi + \delta_qi * \gamma_pj
        # However, considering that Michael's notation and the one I
        # used on the pp-EOM differ by a sign (<|p q j+ i+ |> = -<|p q i+ j+ |>)
        # bellow's expression matches the signs in (eq. 35)
        # Subindices order will be "pqsr" to match the "klji" notation I used
        # #
        # rdm_terms = (
        #     np.einsum("li,kj->klji", np.eye(n), np.eye(n), optimize=True)
        #     - np.einsum("ki,lj->klji", np.eye(n), np.eye(n), optimize=True)
        #     + eye_dm1
        #     - np.transpose(eye_dm1, axes=(0, 1, 3, 2))
        #     + dm1_eye
        #     - np.transpose(dm1_eye, axes=(0, 1, 3, 2))
        #     + dm2
        # )
        # #
        # \delta_{i l} \delta_{j k} -\delta_{i k} \delta_{j l}
        rdm_terms = np.einsum("il,jk->klji", np.eye(n), np.eye(n), optimize=True)
        rdm_terms -= np.einsum("ik,jl->klji", np.eye(n), np.eye(n), optimize=True)
        # + \delta_{i k} \left\{a^\dagger_{j} a_{l}\right\}
        # - \delta_{i l} \left\{a^\dagger_{j} a_{k}\right\}
        rdm_terms += np.einsum("ik,jl->klji", np.eye(n), dm1, optimize=True)
        rdm_terms -= np.einsum("il,jk->klji", np.eye(n), dm1, optimize=True)
        # + \delta_{j l} \left\{a^\dagger_{i} a_{k}\right\}
        # - \delta_{j k} \left\{a^\dagger_{i} a_{l}\right\}
        rdm_terms += np.einsum("jl,ik->klji", np.eye(n), dm1, optimize=True)
        rdm_terms -= np.einsum("jk,il->klji", np.eye(n), dm1, optimize=True)

        # Nonlinear term (eq. 19 integrand)
        def nonlinear(alpha):
            r""" """
            # Compute H^alpha
            h = alpha * dh
            h += h_0
            v = alpha * dv
            v += v_0
            # Solve EOM equations
            # c = cls(h, v, dm1, dm2).solve_dense(*args, **kwargs)[1].reshape(n ** 2, n, n)
            w, c = cls(h, v, dm1, dm2).solve_dense(*args, **kwargs)
            _, c, _ = pickpositiveeig(w, c)
            # Compute transition RDMs (eq. 35)
            rdms = np.einsum("mrs,pqsr->mpq", c.reshape(c.shape[0], n, n), rdm_terms)
            # Compute nonlinear energy term
            tv = np.zeros_like(dm2)
            for rdm in rdms:
                tv += np.einsum("sr,qp->pqrs", rdm, rdm, optimize=True)
            return np.einsum("pqrs,pqrs", dv, tv/2, optimize=True)

        # Compute linear term (eq. 19)
        # dh * \gamma + 0.5 * dv * (\delta_pr * \gamma_qs + \delta_qs * \gamma_pr - \delta_ps * \gamma_qr
        #                           - \delta_qr * \gamma_ps - \delta_pr * \delta_qs + \delta_ps * \delta_qr)
        linear = (
            eye_dm1
            - np.transpose(eye_dm1, axes=(0, 1, 3, 2))
            + dm1_eye
            - np.transpose(dm1_eye, axes=(0, 1, 3, 2))
            - np.einsum("pr,qs->pqrs", np.eye(n), np.eye(n), optimize=True)
            + np.einsum("ps,qr->pqrs", np.eye(n), np.eye(n), optimize=True)
        )
        linear = np.einsum("pq,pq", dh, dm1, optimize=True) + 0.5 * np.einsum(
            "pqrs,pqrs", dv, linear, optimize=True
        )

        # Compute ERPA correlation energy (eq. 19)
        return (
            linear
            + 0.5 * integrate(nonlinear, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.)[0]
        )
        # return linear


class EOMDEA_2(EOMState):
    r"""
    Double electron  attachment EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} a^{\dagger}_i a^{\dagger}_j}`.

    .. math::

        \left< \Psi^{(N)}_0 \middle| \left[a_k a_l, \left[ \hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_k \left< \Psi^{(N)}_0 \middle| a_k a_l \hat{Q} \middle| \Psi^{(N)}_0 \right>
    """

    @property
    def neigs(self):
        r""" """
        return self._n ** 2

    def _compute_lhs(self):
        r"""
        Compute

        .. math::
            A_{klji} = 2 (h_{li} \delta_{kj} - h_{ki} \delta_{lj}) + 2 (h_{ki} \gamma_{jl} - h_{li} \gamma_{jk})
            + 2 \sum_{p} (h_{pi} \gamma_{pk} \delta_{lj} + h_{pj} \gamma_{pl} \delta_{ki}) + \left< lk||ij \right>
            + \sum_{q} (\left< ql||ij \right> \gamma_{qk} - \left< qk||ij \right> \gamma_{ql})
            + 2 \sum_{qr} \gamma_{qr}(\left< ql||jr \right> \delta_{ki} - \left< qk||jr \right> \delta_{li})
            + 2 \sum_{qr} (\left< ql||ir \right> \Gamma_{qjrk} - \left< qk||ir \right> \Gamma_{qjrl})
            + \sum_{pqr} \left< pq||jr \right> (\Gamma_{pqrk} \delta_{li} - \Gamma_{pqrl} \delta_{ki})
            + 2 \sum_{r} \left< lk||jr \right> \gamma_{ir}s

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
        r"""
        Compute

        .. math::

            M_{klji} = \Gamma_{ijlk} + \delta_{li} \delta_{kj} - \delta_{ki} \delta_{lj} + \delta_{ki} \gamma_{jl}
            - \delta_{kj} \gamma_{li} + \delta_{lj} \gamma_{ki} - \delta_{li} \gamma_{jk}

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