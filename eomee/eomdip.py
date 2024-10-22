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

r"""Double electron removal EOM state class."""


import numpy as np

from scipy.integrate import quad as integrate
from scipy.integrate import quadrature as fixed_quad

from .base import EOMState
from .solver import pick_positive, _pick_singlets, _pick_multiplets


__all__ = [
    "DIP",
    "DIPm",
]


class DIP(EOMState):
    r"""Doubly Ionized state class.

    This class implements the extended random phase approximation method for the two-electron removal
    operator (hole-hole ERPA).

    :math:`| \Psi^{(N-2)}_\lambda > = \hat{Q}^{-2}_\lambda | \Psi^{(N)}_0 >`

    defined by the single electron transition operator :math:`\hat{Q}^{-2}_\lambda = \sum_{ij} { c_{ij;\lambda} a_i  a_j}`
    where the indices run over all spin-orbitlas.

    The double ionization energies (:math:`\Delta_\lambda = E^{(N-2)}_\lambda - E^(N)_0`) and wavefunction satisfy:

    .. math::

        &\mathbf{A} \mathbf{C}_\lambda = \Delta_\lambda \mathbf{U} \mathbf{C}_\lambda

        A_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a^{\dagger}_l, \left[\hat{H}, a_j  a_i \right]\right] \middle| \Psi^{(N)}_0 \right>

        U_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k a^{\dagger}_l, a_j  a_i \right] \middle| \Psi^{(N)}_0 \right>

    The :math:`n^2 \times n^2` matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` are defined in terms of the ground state's 
    reduced density matrices.

    Example
    -------
    >>> erpahh = eomee.DIP(h, v, dm1, dm2)
    >>> erpahh.neigs # number of solutions
    >>> erpahh.lhs # left-hand-side matrix
    >>> # Solve the generalized eigenvalue problem
    >>> erpahh.solve_dense()

    """

    @property
    def neigs(self):
        r""" """
        return self._n ** 2

    def _compute_lhs(self):
        r"""
        Compute

        .. math::

            A_{klji} = 2 ( h_{il} \delta_{jk} - h_{il} \gamma_{kj} + h_{ik} \gamma_{lj} - h_{ik} \delta_{jl} )
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
        # FIX: Add missing symmetric permutation terms
        a = a + a.transpose(1, 0, 3, 2)
        return 0.5 * a.reshape(self._n ** 2, self._n ** 2)

    def _compute_rhs(self):
        r"""
        Compute

        .. math::

            M_klji = delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}\\
                - \delta_{i k} \gamma_{lj} + \delta_{i l} \gamma_{kj}\\
                - \delta_{j l} \gamma_{ki} + \delta_{j k} \gamma_{li}
        
        """

        # M_klji = _klji
        # \delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}
        m = np.einsum("ik,jl->klji", np.eye(self._n), np.eye(self._n), optimize=True)
        m -= np.einsum("il,jk->klji", np.eye(self._n), np.eye(self._n), optimize=True)
        # - \delta_{i k} \left\{a^\dagger_{l} a_{j}\right\}
        # + \delta_{i l} \left\{a^\dagger_{k} a_{j}\right\}
        m -= np.einsum("ik,jl->klji", np.eye(self._n), self._dm1, optimize=True)
        m += np.einsum("il,jk->klji", np.eye(self._n), self._dm1, optimize=True)
        # - \delta_{j l} \left\{a^\dagger_{k} a_{i}\right\}
        # + \delta_{j k} \left\{a^\dagger_{l} a_{i}\right\}
        m -= np.einsum("jl,ik->klji", np.eye(self._n), self._dm1, optimize=True)
        m += np.einsum("jk,il->klji", np.eye(self._n), self._dm1, optimize=True)
        return m.reshape(self._n ** 2, self._n ** 2)

    def normalize_eigvect(self, coeffs):
        r"""
        Normalize coefficients vector.

        Make the solutions orthonormal with respect to the metric matrix U:
        .. math::
        \mathbf{c}^T \mathbf{U} \mathbf{c} = 1

        Parameters
        ----------
        coeffs : np.ndarray(n**2)
            Coefficients vector for the lambda-th excited state.
        
        Returns
        -------
        coeffs : np.ndarray(n**2)
            Normalized coefficients vector for the lambda-th excited state.

        """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        norm_factor = np.dot(coeffs, np.dot(self.rhs, coeffs.T))
        sqr_n = np.sqrt(np.abs(norm_factor))
        return (coeffs.T / sqr_n).T
    
    def compute_tdm(self, coeffs):
        r"""
        Compute the transition RDMs

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a^\dagger_q | \Psi^{(N-2)}_\lambda >

        Parameters
        ----------
        coeffs : np.ndarray(n**2)
            Coefficients vector for the lambda-th double ionized state.
        
        Returns
        -------
        tdm : np.ndarray(n,n)
            transition RDMs between the reference (ground) state and the double ionized state.

        """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        coeffs = coeffs.reshape(self._n, self._n)
        rhs = self.rhs.reshape(self.n,self.n,self.n,self.n)
        return np.einsum("pqrs,rs->pq", rhs, coeffs)
    
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

        linear = _hherpa_linearterms(dh, dm1)

        # Nonlinear term (integrand)        
        function = WrappNonlinear(cls, h_0, v_0, dh, dv, dm1, dm2)
        # nonlinear, abserr = integrate(function, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.49e-04)
        nonlinear, abserr = fixed_quad(function, 0, 1, tol=1.49e-04, maxiter=5, vec_func=False)
        # Compute ERPA correction energy
        ecorr = linear + 0.5 * nonlinear

        output = {}
        output["ecorr"] = ecorr
        output["linear"] = linear
        output["error"] = abserr

        return output


def _hherpa_linearterms(_dh, _dm1):
    # Linear term (eq. 20)
    # dh_pq * \gamma_pq
    _linear = np.einsum("pq,pq", _dh, _dm1, optimize=True)
    return _linear


class WrappNonlinear:
    r"""Compute adiabatic connection integrand."""
    def __init__(self, method, h0, v0, dh, dv, dm1, dm2):
        self.h_0 = h0
        self.v_0 = v0
        self.dh = dh
        self.dv = dv
        # TODO: Check that method is EOMDIP
        self.dm1 = dm1
        self.dm2 = dm2
        self.method = method
    
    @staticmethod
    def eval_tdmterms(_n, _dm1):
        # Compute RDM terms of transition RDM
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = \sum_{n=0} < | p^+ q^+ |N-2> <N-2|s r| >
        #
        # Commutator form: < |[p+ q+, i j]| >
        # \delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}
        _rdm_terms = np.einsum("ik,jl->klji", np.eye(_n), np.eye(_n), optimize=True)
        _rdm_terms -= np.einsum("il,jk->klji", np.eye(_n), np.eye(_n), optimize=True)
        # - \delta_{i k} \left\{a^\dagger_{l} a_{j}\right\}
        # + \delta_{i l} \left\{a^\dagger_{k} a_{j}\right\}
        _rdm_terms -= np.einsum("ik,jl->klji", np.eye(_n), _dm1, optimize=True)
        _rdm_terms += np.einsum("il,jk->klji", np.eye(_n), _dm1, optimize=True)
        # - \delta_{j l} \left\{a^\dagger_{k} a_{i}\right\}
        # + \delta_{j k} \left\{a^\dagger_{l} a_{i}\right\}
        _rdm_terms -= np.einsum("jl,ik->klji", np.eye(_n), _dm1, optimize=True)
        _rdm_terms += np.einsum("jk,il->klji", np.eye(_n), _dm1, optimize=True)
        return _rdm_terms
    
    @staticmethod
    def eval_nonlinearterms(_n, _dm1, coeffs, rdmterms):
        # Compute transition RDMs (eq. 32)
        # \gamma_m;pq = c_m;ji * < |p+q+ji| >
        rdms = np.einsum("mji,pqij->mpq", coeffs.reshape(coeffs.shape[0], _n, _n), rdmterms)
        _tv = np.zeros((_n, _n, _n, _n), dtype=_dm1.dtype)
        for rdm in rdms:
            tv += np.einsum("pq,rs->pqrs", rdm, rdm, optimize=True)
        return _tv

    def __call__(self, alpha, singlet=True, *args, **kwargs):
        """Compute integrand."""
        # Compute H^alpha
        h = alpha * self.dh
        h += self.h_0
        v = alpha * self.dv
        v += self.v_0
        n = h.shape[0]
        # Solve EOM equations
        hh = self.method(h, v, self.dm1, self.dm2)
        w, c = hh.solve_dense(*args, **kwargs)
        ev_p, cv_p, _ = pick_positive(w, c)
        if singlet:
            s_cv= _pick_singlets(ev_p, cv_p)[1]
            norm = np.dot(s_cv, np.dot(hh.rhs, s_cv.T))
            diag_n = np.diag(norm)
            idx = np.where(diag_n > 0)[0]  # Remove eigenvalues with negative norm
            sqr_n = np.sqrt(diag_n[idx])
            c = (s_cv[idx].T / sqr_n).T
        else:
            raise NotImplementedError("Only singlets are implemented")
        # Compute transition RDMs (eq. 32)
        rdm_terms = WrappNonlinear.eval_tdmterms(n, self.dm1)
        tv = WrappNonlinear.eval_nonlinearterms(n, self.dm1, c, rdm_terms)
        # Compute nonlinear energy term
        # dv_pqrs * {sum_{m}{\gamma_m;pq * \gamma_m;rs}}_pqrs        
        return np.einsum("pqrs,pqrs", self.dv, tv/2, optimize=True)


class DIPm(DIP):
    r"""
    Double Ionization EOM class without commutator in the right-hand side of the equation.

    The double ionization energies (:math:`\Delta_\lambda = E^{(N-2)}_\lambda - E^(N)_0`) and :math:`\lambda`th
    state of the :math:`(N-2)`-electron system are determined by solving the matrix equation:

    .. math::

        &\mathbf{A} \mathbf{C}_{k} = \Delta_{k} \mathbf{U} \mathbf{C}_{k}

        A_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a^{\dagger}_l, \left[\hat{H}, a_j  a_i \right]\right] \middle| \Psi^{(N)}_0 \right>

        U_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| a^{\dagger}_k a^{\dagger}_l a_j  a_i \middle| \Psi^{(N)}_0 \right>

    """

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{klji} = \Gamma_{klji}`

        """
        m = np.copy(self._dm2)
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

        # Linear term (eq. 20)
        # dh_pq * \gamma_pq
        linear = np.einsum("pq,pq", dh, dm1, optimize=True)

        # Nonlinear term (eq. 20 integrand)
        @np.vectorize
        def nonlinear(alpha):
            r""" """
            # Compute H^alpha
            h = alpha * dh
            h += h_0
            v = alpha * dv
            v += v_0
            # Solve EOM equations
            c = cls(h, v, dm1, dm2).solve_dense(*args, **kwargs)[1].reshape(n ** 2, n, n)
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
            - 0.5 * integrate(nonlinear, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.49e-04)[0]
        )


# Alias the hole-hole EOM equation
ERPAhh = DIP
