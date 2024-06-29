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

# from scipy.integrate import quad as integrate
from scipy.integrate import quadrature as integrate
from scipy.integrate import fixed_quad

from .base import EOMState
from .solver import pick_positive, _pick_singlets


__all__ = [
    "EE",
    "EEm",
]


def _get_lhs_particlehole_erpa(h, v, dm1, dm2):
    n = h.shape[0]
    hdm1 = np.dot(h, dm1)
    I = np.eye(n, dtype=h.dtype)

    # A_klij = h_li \gamma_kj + h_jk \gamma_il
    b = np.einsum("li,kj->klji", h,  dm1, optimize=True)
    b += np.einsum("jk,il->klji",  h,  dm1, optimize=True)
    # A_klij -= ( \delta_il h_jq \gamma_qk + \delta_jk h_iq \gamma_ql )
    b -= np.einsum("il,jk->klji", I, hdm1, optimize=True)
    b -= np.einsum("jk,il->klji", I, hdm1, optimize=True)
    # A_klij += <lq||si> \Gamma_kqsj
    b += np.einsum("lqsi,kqsj->klji",  v,  dm2, optimize=True)
    # A_klij += <jq||sk> \Gamma_iqsl
    b += np.einsum("jqsk,iqsl->klji",  v,  dm2, optimize=True)
    # A_klij += 0.5 ( <pq||ik> \Gamma_pqlj )
    b += 0.5 * np.einsum("pqik,pqlj->klji",  v,  dm2, optimize=True)
    # A_klij -= 0.5 ( <lj||rs> \Gamma_kirs )
    b -= 0.5 * np.einsum("ljrs,kirs->klji",  v,  dm2, optimize=True)
    # A_klij -= 0.5 ( \delta_il <jq||rs> \Gamma_kqrs )
    vdm2 = np.einsum("jqrs,kqrs->jk",  v,  dm2, optimize=True)
    b -= 0.5 * np.einsum("il,jk->klji", I, vdm2, optimize=True)
    # A_klij -= 0.5 ( \delta_jk <pq||si> \Gamma_pqsl )
    vdm2 = np.einsum("pqsi,pqsl->il", v,  dm2, optimize=True)
    b -= 0.5 * np.einsum("jk,il->klji", I, vdm2, optimize=True)
    return b


def _get_rhs_particlehole_erpa(dm1):
    # Commutator form
    n = dm1.shape[0]
    I = np.eye(n, dtype=dm1.dtype)
    m = np.einsum("li,kj->klji", I, dm1, optimize=True)
    m -= np.einsum("kj,il->klji", I, dm1, optimize=True)
    return m


class EE(EOMState):
    r"""Electron excitated state class.

    This class implements the extended random phase approximation method ([ERPA]_).
    The wavefunction of the :math:`\lambda`th excited state can be approximated as linear combination
    of single electron excited configurations produced from the ground state wavefunction:
    state wavefunction appliying a linear combination of single electron excitation operators, :math:`\hat{Q}^{0}_\lambda`:

    .. math::

        \Psi^{(N)}_\lambda > = \sum_{ij} c_{ij;\lambda} a^{\dagger}_i  a_j | \Psi^{(N)}_0 >

    where :math:`a^{\dagger}_i` and :math:`a_i` are the creation and annihilation operators, respectively
    and the indices :math:`ij` denote arbitrary spin-orbitals.

    The excitation energies (:math:`\Delta_\lambda = E^{(N)}_\lambda - E^(N)_0`) and wavefunction are
    determined by the following eigenvalue problem:

    .. math::

        &\mathbf{A} \mathbf{C}_{k} = \Delta_{k} \mathbf{U} \mathbf{C}_{k}

        A_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l, \left[\hat{H}, a^{\dagger}_j  a_i \right]\right] \middle| \Psi^{(N)}_0 \right>

        U_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| \left[ a^{\dagger}_k a_l, a^{\dagger}_j  a_i \right] \middle| \Psi^{(N)}_0 \right>

    where the matrices :math:`\mathbf{A}` and :math:`\mathbf{U}` can be defined in terms of the ground state's
    one- and two-electron density matrices. The matrices are :math:`n^2 \times n^2` matrices for an :math:`n`
    spin-orbital basis. Correspondingly, there will be :math:`n^2` solution if matrix diagonalization is applied.
    The eigenvalues obtained come in (mirrored) pairs; positive transition correspond to excitation
    energies :math:`\Delta_\lambda > 0` and negative ones to de-excitations :math:`\Delta_\lambda < 0`.

    .. [EKT] Chatterjee, K., & Pernal, K. (2012). The Journal of Chemical Physics, 137(20), 204109.

    Example
    -------
    >>> erpa = eomee.EE(h, v, dm1, dm2)
    >>> erpa.neigs # number of solutions
    >>> erpa.lhs # left-hand-side matrix
    >>> # Solve the ERPA matrix equation
    >>> erpa.solve_dense()

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
        Amtx = _get_lhs_particlehole_erpa(self.h, self.v, self.dm1, self.dm2)
        return Amtx.reshape(self._n ** 2, self._n ** 2)

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{klji} = \gamma_{kj} \delta_{li} - \delta_{kj} \gamma_{li}`.

        """
        Umtx = _get_rhs_particlehole_erpa(self.dm1)
        return Umtx.reshape(self._n ** 2, self._n ** 2)
    
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
    
    def compute_tdm1(self, coeffs):
        r"""
        Compute the transition RDMs

        .. math::
        \gamma^{0 \lambda}_{pq} = < \Psi^{(N)}_0 | a^\dagger_p a_q | \Psi^{(N)}_\lambda >

        The diagonal elements of this matrix are zero.

        Parameters
        ----------
        coeffs : np.ndarray(n**2)
            Coefficients vector for the lambda-th excited state.
        
        Returns
        -------
        tdm1 : np.ndarray(n,n)
            1-electron reduced transition RDMs.

        """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        coeffs = coeffs.reshape(self._n, self._n)
        rhs = self.rhs.reshape(self.n,self.n,self.n,self.n)
        return np.einsum("pqrs,rs->pq", rhs, coeffs)

    @classmethod
    def erpa(cls, h_0, v_0, h_1, v_1, dm1, dm2, solver="nonsymm", eigtol=1.e-7, singl=True, nint=5):
        r"""
        Compute the ERPA correlation energy for the operator.

        """
        # Size of dimensions
        n = h_0.shape[0]
        # H_1 - H_0
        dh = h_1 - h_0
        # V_1 - V_0
        dv = v_1 - v_0
        
        linear = _pherpa_linearterms(n, dh, dv, dm1)

        # Compute ERPA correction energy
        # Nonlinear term (eq. 19 integrand)        
        function = _IntegrandPh(cls, h_0, v_0, dh, dv, dm1, dm2)
        params = (solver, eigtol, singl)
        nonlinear, abserr = integrate(function.vfunc, 0, 1, args=params, tol=1.49e-04, maxiter=nint, vec_func=True)
        ecorr = linear + 0.5 * nonlinear

        output = {}
        output["ecorr"] = ecorr
        output["linear"] = linear
        output["error"] = abserr

        return output


def _pherpa_linearterms(_n, _dh, _dv, _dm1):
    # Gamma_pqrs = < | p^+ q^+ s r | > = - < | p^+ q^+ r s | >
    #            = - \delta_qr * \gamma_ps
    #            + \gamma_pr * \gamma_qs
    #            + \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
    dm1_eye = np.einsum("qr,ps->pqrs", np.eye(_n), _dm1, optimize=True)
    # Compute linear term (eq. 19)
    # dh * \gamma + 0.5 * dv * (\gamma_pr * \gamma_qs - \delta_qr * \gamma_ps)
    _linear = np.einsum("pr,qs->pqrs", _dm1, _dm1, optimize=True) - dm1_eye
    _linear = np.einsum("pq,pq", _dh, _dm1, optimize=True) + 0.5 * np.einsum(
        "pqrs,pqrs", _dv, _linear, optimize=True
    )
    return _linear


def _truncate_dm1dm1_matrix(nspins, ij_d_occs, _dm1dm1, _eigtol):
    nt = nspins**2
    truncated = np.zeros_like(_dm1dm1)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > _eigtol
            cond2 = np.abs(ij_d_occs[rs]) > _eigtol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                truncated[p,r,q,s] = _dm1dm1[p,r,q,s]
    return truncated


def _truncate_eyedm1_matrix(nspins, ij_d_occs, _eyedm1, _eigtol):
    nt = nspins**2
    truncated = np.zeros_like(_eyedm1)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > _eigtol
            cond2 = np.abs(ij_d_occs[rs]) > _eigtol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                truncated[p,q,r,s] = _eyedm1[p,q,r,s]
    return truncated


def _truncate_rdm2_matrix(nspins, ij_d_occs, _rdm2, _eigtol):
    nt = nspins**2
    truncated = np.zeros_like(_rdm2)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > _eigtol
            cond2 = np.abs(ij_d_occs[rs]) > _eigtol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                truncated[p,r,q,s] = _rdm2[p,r,q,s]
    return truncated


def _alpha_independent_terms_rdm2_alpha(_dm1, _rhs, _summall, _eigtol):
    # (\gamma_pr * \gamma_qs - \delta_qr * \gamma_ps)
    _n = _dm1.shape[0]
    dm1dm1 = np.einsum("pr,qs->pqrs", _dm1, _dm1, optimize=True)
    dm1_eye = np.einsum("qr,ps->pqrs", np.eye(_n), _dm1, optimize=True)
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        dm1dm1  = _truncate_dm1dm1_matrix(_n, d_occs_ij, dm1dm1, _eigtol)
        dm1_eye  = _truncate_eyedm1_matrix(_n, d_occs_ij, dm1_eye, _eigtol)
    return (dm1dm1 - dm1_eye)


def _rdm2_a0(_rdm2, _rhs, _summall, _eigtol):
    _n = _rdm2.shape[0]
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        _rdm2  = _truncate_rdm2_matrix(_n, d_occs_ij, _rdm2, _eigtol)
    return _rdm2


class _IntegrandPh:
    r"""Compute adiabatic connection integrand."""
    def __init__(self, method, h0, v0, dh, dv, dm1, dm2):
        self.h_0 = h0
        self.v_0 = v0
        self.dh = dh
        self.dv = dv
        # TODO: Check that method is EE
        self.dm1 = dm1
        self.dm2 = dm2
        self.method = method
        self.vfunc = np.vectorize(self.eval_integrand) 
    
    @staticmethod
    def eval_dmterms(_n, _dm1):
        # Compute RDM terms of transition RDM
        # Commutator form: < |[p+q,s+r]| >
        # \delta_qs \gamma_pr - \delta_pr \gamma_sq
        _rdm_terms = np.einsum("qs,pr->pqrs", np.eye(_n), _dm1, optimize=True)
        _rdm_terms -= np.einsum("pr,sq->pqrs", np.eye(_n), _dm1, optimize=True)
        return _rdm_terms
    
    @staticmethod
    def eval_alphadependent_terms(_n, _dm1, coeffs, dmterms):
        # Compute transition RDMs (eq. 29)
        tdms = np.einsum("mrs,pqrs->mpq", coeffs.reshape(coeffs.shape[0], _n, _n), dmterms)
        # Compute nonlinear energy term
        _tv = np.zeros((_n, _n, _n, _n), dtype=_dm1.dtype)
        for tdm in tdms:
            _tv += np.einsum("pr,qs->pqrs", tdm, tdm, optimize=True)
        return _tv

    def eval_integrand(self, alpha, gevps, tol, singlets):
        """Compute integrand."""
        # Compute H^alpha
        h = alpha * self.dh
        h += self.h_0
        v = alpha * self.dv
        v += self.v_0
        # Size of dimensions
        n = h.shape[0]
        # Solve EOM equations
        ph = self.method(h, v, self.dm1, self.dm2)
        w, c = ph.solve_dense(tol=tol, mode=gevps)
        # ev_p, cv_p, _ = pick_positive(w, c)
        ev_p, cv_p = w, c
        if singlets:
            s_cv= _pick_singlets(ev_p, cv_p)[1]
            norm = np.dot(s_cv, np.dot(ph.rhs, s_cv.T))
            diag_n = np.diag(norm)
            sqr_n = np.sqrt(np.abs(diag_n))
            c = (s_cv.T / sqr_n).T
        else:
            raise NotImplementedError("Triplets not implemented yet.")
        
        # Compute transition RDMs energy contribution (eq. 29)
        rdm_terms = IntegrandPh.eval_dmterms(n, self.dm1)
        tdtd = IntegrandPh.eval_alphadependent_terms(n, self.dm1, c, rdm_terms)
        return np.einsum("pqrs,pqrs", self.dv, tdtd, optimize=True)


class EEm(EE):
    r"""Electron excitated state class.

    The excitation energies (:math:`\Delta_\lambda = E^{(N)}_\lambda - E^(N)_0`) and :math:`\lambda`th
    state of the :math:`(N)`-electron wavefunction are determined by the following eigenvalue problem:

    .. math::

        &\mathbf{A} \mathbf{C}_{k} = \Delta_{k} \mathbf{U} \mathbf{C}_{k}

        A_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_k  a_l, \left[\hat{H}, a^{\dagger}_j  a_i \right]\right] \middle| \Psi^{(N)}_0 \right>

        U_{kl,ij} &= \left< \Psi^{(N)}_0 \middle| a^{\dagger}_k a_l a^{\dagger}_j  a_i  \middle| \Psi^{(N)}_0 \right>

    This is the matrix form of the double commutator EOM equation for the single electron excitation
    operator (same one as in the :class:`EE` class) but without a commutator on the right-hand side matrix
    :math:`\mathbf{U}` (mixed doublecommutator EOM form, EEm). The spectra of the eigenvalues obtained
    only contain positive values.

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

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{klji} = \gamma_kj \delta_li - \Gamma_kijl`.

        """
        I = np.eye(self._n, dtype=self._h.dtype)

        # No commutator form
        m = np.einsum("kj,li->klji", self.dm1, I, optimize=True)
        m -= np.einsum("kijl->klji", self.dm2, optimize=True)
        return m.reshape(self._n ** 2, self._n ** 2)


def _get_pherpa_metric_matrix(dm1):
    # Compute ph-ERPA metric matrix
    # < |[p^+ q,s^+ r]| > = \delta_qs \gamma_pr - \delta_pr \gamma_sq
    _n = dm1.shape[0]
    _rdm_terms = np.einsum("qs,pr->pqrs", np.eye(_n), dm1, optimize=True)
    _rdm_terms -= np.einsum("pr,sq->pqrs", np.eye(_n), dm1, optimize=True)
    return _rdm_terms


def _sum_over_nstates_tdtd_matrices(_n, _dm1, coeffs, dmterms):
    # Compute transition RDMs (eq. 29)
    tdms = np.einsum("mrs,pqrs->mpq", coeffs.reshape(coeffs.shape[0], _n, _n), dmterms)
    # Compute nonlinear energy term
    _tv = np.zeros((_n, _n, _n, _n), dtype=_dm1.dtype)
    for tdm in tdms:
        _tv += np.einsum("pr,qs->pqrs", tdm, tdm.T, optimize=True)
    return _tv


def _eval_tdtd_alpha_mtx_from_erpa(erpa_gevp_type, h_l, v_l, dm1, dm2, invtol, solver_type):
    # (1) Solve particle-hole ERPA equations at given perturbation strength alpha
    ph = erpa_gevp_type(h_l, v_l, dm1, dm2)
    ph._invtol = invtol
    ev, cv = ph.solve_dense(mode=solver_type, normalize=False)
    # (2) Pick the solutions that correspond to singlet transitions
    cv_s= _pick_singlets(ev, cv)[1]
    # (3) Normalize the eigenstates before computing the transition RDMs
    norm = np.dot(cv_s, np.dot(ph.rhs, cv_s.T))
    diag_n = np.diag(norm)
    sqr_n = np.sqrt(np.abs(diag_n))
    cv = (cv_s.T / sqr_n).T

    metric = ph.rhs.reshape(ph.n, ph.n, ph.n, ph.n)
    return _sum_over_nstates_tdtd_matrices(ph.n, dm1, cv, metric)


def _eval_W_alpha_singlets(tdtd_singlets, dv):
    # f(alpha) = 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_term^{\alpha}_{pqrs}
    # Gamma_term = \sum_{n \in Singlets} tdm_0n tdm_n0
    energy = np.einsum("pqrs,pqrs", dv, tdtd_singlets, optimize=True)
    return 0.5 * energy


def _eval_W_alpha_constant_terms(dv, rdm1, rdm2, summall, invtol):
    # 0.5 * \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) Gamma_terms_{pqrs}
    # Gamma_terms = (gamma_{pr} * gamma_{qs} + delta_{qr} * gamma_{ps})
    #             - Gamma_^{\alpha=0}
    n = rdm1.shape[0]
    rhs = _get_pherpa_metric_matrix(rdm1).reshape(n ** 2, n ** 2)
    temp = _alpha_independent_terms_rdm2_alpha(rdm1, rhs, summall, invtol)
    temp -= _rdm2_a0(rdm2, rhs, summall, invtol)
    return 0.5 * np.einsum("pqrs,pqrs", dv, temp, optimize=True)


def ac_integrand_pherpa(lam, h0, v0, dh, dv, dm1, dm2, summall=True, invtol=1.0e-7, solvertype="nonsymm"):
    """Compute the integrand of the adiabatic connection formulation.

    .. math::
    W(\alpha) = 0.5 \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) (\Gamma^{\alpha}_{pqrs} - \Gamma^{\alpha=0}_{pqrs})

    where :math:`\Gamma^{\alpha}_{pqrs}` is

    .. math::
    \Gamma^{\alpha}_{pqrs} = \gamma^{\alpha=0}_{pr} \gamma^{\alpha=0}_{qs} 
    + \sum_{\nu \in Singlets} \gamma^{\alpha;0 \nu}_{pr} \gamma^{\alpha;\nu 0}_{qs} 
    + \sum_{\nu \in Triplets} \gamma^{\alpha;0 \nu}_{pr} \gamma^{\alpha;\nu 0}_{qs} 
    - \delta_{ps} \gamma^{\alpha=0}_{qr}

    Parameters
    ----------
    lam : _type_
        _description_
    h0 : _type_
        _description_
    v0 : _type_
        _description_
    dh : _type_
        _description_
    dv : _type_
        _description_
    dm1 : _type_
        _description_
    dm2 : _type_
        _description_
    summall : bool, optional
        _description_, by default True
    invtol : _type_, optional
        _description_, by default 1.0e-7
    solvertype : str, optional
        _description_, by default "nonsymm"

    Returns
    -------
    _type_
        _description_
    """
    # Compute H^alpha
    h = lam * dh
    h += h0
    v = lam * dv
    v += v0

    # Eval TDMs at alpha from particle-hole singlet transitions and compute energy
    tdtd = _eval_tdtd_alpha_mtx_from_erpa(EE, h, v, dm1, dm2, invtol, solvertype)
    energy = _eval_W_alpha_singlets(tdtd, dv)

    # Eval perturbation independent terms
    energy += _eval_W_alpha_constant_terms(dv, dm1, dm2, summall, invtol)

    return energy


def eval_ecorr(h_0, v_0, h_1, v_1, dm1, dm2, summ_all=True, inv_tol=1.0e-7, nint=5):
    """Compute the (dynamic) correlation energy from the adiabatic connection formulation and 
    particle-hole ERPA.

    .. math::
    E_corr = < \Psi^{\alpha=1}_0 | \hat{H} | \Psi^{\alpha=1}_0 > - < \Psi^{\alpha=0}_0 | \hat{H} | \Psi^{\alpha=0}_0 >
    = 0.5 \sum_{pqrs} \int_{0}_{1} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{prqs}) (\Gamma^{\alpha}_{pqrs} - \Gamma^{\alpha=0}_{pqrs}) d \alpha

    where :math:`\Gamma^{\alpha}_{pqrs}` is

    .. math::
    \Gamma^{\alpha}_{pqrs} = \gamma^{\alpha=0}_{pr} \gamma^{\alpha=0}_{qs} 
    + \sum_{\nu !=0} \gamma^{\alpha;0 \nu}_{pr} \gamma^{\alpha;\nu 0}_{qs} 
    - \delta_{ps} \gamma^{\alpha=0}_{qr}

    Parameters
    ----------
    h_0 : np.ndarray((n, n))
        One-electron integrals for the reference Hamiltonian (at alpha=0).
    v_0 : np.ndarray((n, n, n, n))
        Two-electron integrals for the reference Hamiltonian (at alpha=0).
    h_1 : np.ndarray((n, n))
        One-electron integrals for the true Hamiltonian (at alpha=1).
    v_1 : np.ndarray((n, n, n, n))
        Two-electron integrals for the true Hamiltonian (at alpha=1).
    dm1 : np.ndarray((n, n))
        One-electron reduced density matrix for the reference wavefunction (at alpha=0).
    dm2 : np.ndarray((n, n, n, n))
        Two-electron reduced density matrix for the reference wavefunction (at alpha=0).
    summ_all : bool, optional
        Whether the sum over the two body terms is carried over all `p,q,r,s` indexes or not.
        If False, pairs of spin-orbitals that are not involved in any particle-hole excitation
        are excluded. Which pair to remove is determined by the diagonal elements of the ERPA 
        metric matrix. By default True.
    inv_tol : float, optional
        Tolerance for small singular values when solving the ERPA eigenvalue problem, 
        by default 1.0e-7.
    nint : int, optional
        Order of quadrature integration, by default 5.

    Returns
    -------
    _type_
        _description_
    """
    # H_1 - H_0
    dh = h_1 - h_0
    # V_1 - V_0
    dv = v_1 - v_0

    # Evaluate integrand function: W(alpha)
    @np.vectorize
    def ac_integrand(alpha):        
        return ac_integrand_pherpa(alpha, h_0, v_0, dh, dv, dm1, dm2, summall=summ_all, invtol=inv_tol)

    # integrate function
    return fixed_quad(ac_integrand, 0, 1, n=nint)[0]

# Alias for the particle-hole EOM equation
ERPA = EE
