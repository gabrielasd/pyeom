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

r"""Utility functions to compute the residual correlation energy through AC-ERPA (hole-hole)."""


import numpy as np


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


def _zeroth_order_rdm2(_rdm2, _rhs, _summall, _eigtol):
    _n = _rdm2.shape[0]
    if not _summall:
        d_occs_ij = np.diag(_rhs)
        _rdm2  = _truncate_rdm2_matrix(_n, d_occs_ij, _rdm2, _eigtol)
    return _rdm2


def _get_hherpa_metric_matrix(dm1):
    # Compute hh-ERPA metric matrix
    # \delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}
    n = dm1.shape[0]
    _rdm_terms = np.einsum("ik,jl->klji", np.eye(n), np.eye(n), optimize=True)
    _rdm_terms -= np.einsum("il,jk->klji", np.eye(n), np.eye(n), optimize=True)
    # - \delta_{i k} \left\{a^\dagger_{l} a_{j}\right\}
    # + \delta_{i l} \left\{a^\dagger_{k} a_{j}\right\}
    _rdm_terms -= np.einsum("ik,jl->klji", np.eye(n), dm1, optimize=True)
    _rdm_terms += np.einsum("il,jk->klji", np.eye(n), dm1, optimize=True)
    # - \delta_{j l} \left\{a^\dagger_{k} a_{i}\right\}
    # + \delta_{j k} \left\{a^\dagger_{l} a_{i}\right\}
    _rdm_terms -= np.einsum("jl,ik->klji", np.eye(n), dm1, optimize=True)
    _rdm_terms += np.einsum("jk,il->klji", np.eye(n), dm1, optimize=True)
    return _rdm_terms


def _sum_over_nstates_tdtd_matrices(_n, _dm1, coeffs, dmterms):
    tdms = np.einsum("mrs,pqrs->mpq", coeffs.reshape(coeffs.shape[0], _n, _n), dmterms)
    # Compute nonlinear energy term
    _tv = np.zeros((_n, _n, _n, _n), dtype=_dm1.dtype)
    for tdm in tdms:
        _tv += np.einsum("pq,rs->pqrs", tdm, tdm, optimize=True)
    return _tv
