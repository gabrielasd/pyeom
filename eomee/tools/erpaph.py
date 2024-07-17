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

r"""Utility functions to compute the residual correlation energy through AC-ERPA (particle-hole)."""


import numpy as np


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


def _perturbed_rdm2_constant_terms(_dm1, _rhs, _summall, _eigtol):
    # (\gamma_pr * \gamma_qs - \delta_qr * \gamma_ps)
    _n = _dm1.shape[0]
    dm1dm1 = np.einsum("pr,qs->pqrs", _dm1, _dm1, optimize=True)
    dm1_eye = np.einsum("qr,ps->pqrs", np.eye(_n), _dm1, optimize=True)
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        dm1dm1  = _truncate_dm1dm1_matrix(_n, d_occs_ij, dm1dm1, _eigtol)
        dm1_eye  = _truncate_eyedm1_matrix(_n, d_occs_ij, dm1_eye, _eigtol)
    return (dm1dm1 - dm1_eye)


def _zeroth_order_rdm2(_rdm2, _rhs, _summall, _eigtol):
    _n = _rdm2.shape[0]
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        _rdm2  = _truncate_rdm2_matrix(_n, d_occs_ij, _rdm2, _eigtol)
    return _rdm2


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
