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

r"""Excitation EOM state class."""


import numpy as np

from scipy.integrate import quad as integrate

from .base import EOMState
from .tools import antisymmetrize, pickpositiveeig


__all__ = [
    "EOMExc",
]


class EOMExc(EOMState):
    r"""
    Excitation EOM state for operator :math:`\hat{Q}_k = \sum_{ij} { c_{ij} a^{\dagger}_i  a_j}`.

    .. math::
        \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l, \left[\hat{H}, \hat{Q} \right]\right] \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[ a^{\dagger}_k a_l, \hat{Q} \right] \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}} * n_{\text{basis}}
        return self._n ** 2

    def _compute_lhs(self):
        r"""
        Compute

        .. math::

            A_{klji} = h_{li} \gamma_{kj} + h_{jk} \gamma_{il} - \sum_q { h_{jq} \delta_{il} \gamma_{kq}}
            - \sum_q { h_{qi} \delta_{jk} \gamma_{ql}} + \sum_{qs} { \left< lq||si \right> \Gamma_{kqsj} }
            + \sum_{qs} { \left< jq||sk \right>  \Gamma_{iqsl} }
            - 0.5 \sum_{qrs} { \delta_{il} \left< jq||rs \right> \Gamma_{kqrs} }
            - 0.5 \sum_{pqs} { \delta_{jk} \left< pq||si \right> \Gamma_{pqsl} }
            + 0.5 \sum_{pq} { \left< pq||ik\right>  \Gamma_{pqlj} }
            + 0.5 \sum_{rs} { \left< jl||rs \right> \Gamma_{kirs} }

        """
        hdm1 = np.dot(self._h, self._dm1)
        I = np.eye(self._n, dtype=self._h.dtype)

        # A_klij = h_li \gamma_kj + h_jk \gamma_il
        b = np.einsum("li,kj->klji", self.h, self.dm1, optimize=True)
        b += np.einsum("jk,il->klji", self.h, self.dm1, optimize=True)
        # A_klij -= ( \delta_il h_jq \gamma_qk + \delta_jk h_iq \gamma_ql )
        b -= np.einsum("il,jk->klji", I, hdm1, optimize=True)
        b -= np.einsum("jk,il->klji", I, hdm1, optimize=True)
        # A_klij += <v_lqsi> \Gamma_kqsj
        b += np.einsum("lqsi,kqsj->klji", self.v, self.dm2, optimize=True)
        # b += 0.5 * np.einsum('lpsi,kpsj->klij', self.v, self.dm2)
        # A_klij += <v_jqsk> \Gamma_iqsl
        b += np.einsum("jqsk,iqsl->klji", self.v, self.dm2, optimize=True)
        # b += 0.5 * np.einsum('jpsk,ipsl->klij', self.v, self.dm2)
        # A_klij += 0.5 ( <v_pqik> \Gamma_pqlj )
        b += 0.5 * np.einsum("pqik,pqlj->klji", self.v, self.dm2, optimize=True)
        # A_klij -= 0.5 ( <v_ljrs> \Gamma_kirs )
        b -= 0.5 * np.einsum("ljrs,kirs->klji", self.v, self.dm2, optimize=True)
        # A_klij -= 0.5 ( \delta_il <v_jqrs> \Gamma_kqrs )
        vdm2 = np.einsum("jqrs,kqrs->jk", self.v, self.dm2, optimize=True)
        b -= 0.5 * np.einsum("il,jk->klji", I, vdm2, optimize=True)
        # A_klij -= 0.5 ( \delta_jk <v_pqsi> \Gamma_pqsl )
        vdm2 = np.einsum("pqsi,pqsl->il", self.v, self.dm2, optimize=True)
        b -= 0.5 * np.einsum("jk,il->klji", I, vdm2, optimize=True)
        return b.reshape(self._n ** 2, self._n ** 2)

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{klji} = \gamma_{kj} \delta_{li} - \delta_{kj} \gamma_{li}`.

        """
        I = np.eye(self._n, dtype=self._h.dtype)

        # # M_klij = \gamma_kj \delta_li - \Gamma_kijl
        # # No commutator form
        # m = np.einsum("kj,li->klji", self.dm1, I, optimize=True)
        # m -= np.einsum("kijl->klji", self.dm2, optimize=True)

        # Commutator form
        m = np.einsum("li,kj->klji", I, self.dm1, optimize=True)
        m -= np.einsum("kj,il->klji", I, self.dm1, optimize=True)
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

        # Gamma_pqrs = < | p^+ q^+ s r | > = - < | p^+ q^+ r s | >
        #            = - \delta_qr * \gamma_ps
        #            + \gamma_pr * \gamma_qs
        #            + \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
        dm1_eye = np.einsum("qr,ps->pqrs", np.eye(n), dm1, optimize=True)
        # Compute linear term (eq. 19)
        # dh * \gamma + 0.5 * dv * (\gamma_pr * \gamma_qs - \delta_qr * \gamma_ps)
        linear = np.einsum("pr,qs->pqrs", dm1, dm1, optimize=True) - dm1_eye
        linear = np.einsum("pq,pq", dh, dm1, optimize=True) + 0.5 * np.einsum(
            "pqrs,pqrs", dv, linear, optimize=True
        )

        # Compute RDM terms of transition RDM
        # Commutator form: < |[p+q,s+r]| >
        # \delta_qs \gamma_pr - \delta_pr \gamma_sq
        rdm_terms = np.einsum("qs,pr->pqrs", np.eye(n), dm1, optimize=True)
        rdm_terms -= np.einsum("pr,sq->pqrs", np.eye(n), dm1, optimize=True)
        # @np.vectorize
        # Nonlinear term (eq. 19 integrand)
        def nonlinear(alpha):
            r""" """
            # Compute H^alpha
            h = alpha * dh
            h += h_0
            v = alpha * dv
            v += v_0
            # # Antysymmetrize v_pqrs
            # v = antisymmetrize(v)
            # Solve EOM equations
            w, c = cls(h, v, dm1, dm2).solve_dense(*args, **kwargs)
            _, c, _ = pickpositiveeig(w, c)
            # Compute transition RDMs (eq. 29)
            rdms = np.einsum("mrs,pqrs->mpq", c.reshape(c.shape[0], n, n), rdm_terms)
            # Compute nonlinear energy term
            tv = np.zeros_like(dm2)
            for rdm in rdms:
                tv += np.einsum("pr,qs->pqrs", rdm, rdm, optimize=True)
            return np.einsum("pqrs,pqrs", dv, tv, optimize=True)

        # Compute ERPA correlation energy (eq. 19)
        return (
            linear
            + 0.5 * integrate(nonlinear, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.49e-04)[0]
        )