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

r"""TDMs"""

import numpy as np


def check_dm_normalization(npart, twodm):
    assert np.allclose(np.einsum("ijij", twodm), (npart * (npart - 1)))
    # print('Calc', np.einsum("ijij", twodm), 'expect', npart * (npart - 1))

def check_rdm2_symmetry(old_rdm2, new_rdm2):
    # (0,1,2,3)
    axes = [(1,0,2,3), (0,1,3,2),(2,3,1,0),(3,2,0,1)]
    for axs in axes:
        assert np.allclose(old_rdm2, -new_rdm2.transpose(*axs))
    assert np.allclose(old_rdm2, new_rdm2.transpose(1,0,3,2))


class TDM():
    """Compute the transition RDMs. Two options are possible: to use the commutator of
    the excitation operators (commutator=True) or not.

    Parameters
    ----------
    cv : np.ndarray((n**2, n**2))
        Eigenvector matrix.
    dm1 : np.ndarray((n, n))
        Spin resolved 1-particle reduced density matrix.
    dm2 : np.ndarray((n, n, n, n))
        Spin resolved 2-particle reduced density matrix.
    commutator : bool, optional
        Form used to approximate the transition-RDMs, one of commutator (True) or no commutator
        (False), by default True

    Returns
    -------
    np.ndarray((n**2, n, n))
        1-particle transition density matrix.
    """
    def __init__(self, cv, dm1, dm2) -> None:
        self._cv = cv
        self._dm1 = dm1
        self._dm2 = dm2
        self._n = dm1.shape[0]

    def get_tdm(self, op, comm=True):
        if op == 'hh':
            rdmterms = self.hh(commutator=comm)
        elif op == 'ph':
            rdmterms = self.ph(commutator=comm)
        elif op == 'pp':
            rdmterms = self.pp(commutator=comm)
        else:
            raise ValueError("`op` must be one of `ph`, `pp` or `hh`.")
        cv = self._cv.reshape(self._cv.shape[0], self._n, self._n)
        return np.einsum("mrs,pqsr->mpq", cv, rdmterms)

    def ph(self, commutator=True):
        if commutator:
            rdm_terms = np.einsum("li,kj->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("kj,il->klji", np.eye(self._n), self._dm1, optimize=True)
        else:
            rdm_terms = np.einsum("kj,li->klji", self._dm1, np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("kijl->klji", self._dm2, optimize=True)
        return rdm_terms

    def hh(self, commutator=True):
        if commutator:
            # < |[k^+ l^+, i j]| >
            # \delta_{i k} \delta_{j l} - \delta_{i l} \delta_{j k}
            rdm_terms = np.einsum("ik,jl->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("il,jk->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            # - \delta_{i k} \left\{a^\dagger_{l} a_{j}\right\}
            # + \delta_{i l} \left\{a^\dagger_{k} a_{j}\right\}
            rdm_terms -= np.einsum("ik,jl->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms += np.einsum("il,jk->klji", np.eye(self._n), self._dm1, optimize=True)
            # - \delta_{j l} \left\{a^\dagger_{k} a_{i}\right\}
            # + \delta_{j k} \left\{a^\dagger_{l} a_{i}\right\}
            rdm_terms -= np.einsum("jl,ik->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms += np.einsum("jk,il->klji", np.eye(self._n), self._dm1, optimize=True)
        else:
            # gamma_kl;n = \sum_ij < |k^+ l^+ i j| > c_ij;n
            # c_ij;n \Gamma_klji
            rdm_terms = self._dm2
        return rdm_terms

    def pp(self, commutator=True):
        if commutator:
            #
            # < |[k l, i^+ j^+]| >
            #
            # \delta_{i l} \delta_{j k} -\delta_{i k} \delta_{j l}
            rdm_terms = np.einsum("il,jk->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("ik,jl->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            # + \delta_{i k} \left\{a^\dagger_{j} a_{l}\right\}
            # - \delta_{i l} \left\{a^\dagger_{j} a_{k}\right\}
            rdm_terms += np.einsum("ik,jl->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("il,jk->klji", np.eye(self._n), self._dm1, optimize=True)
            # + \delta_{j l} \left\{a^\dagger_{i} a_{k}\right\}
            # - \delta_{j k} \left\{a^\dagger_{i} a_{l}\right\}
            rdm_terms += np.einsum("jl,ik->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("jk,il->klji", np.eye(self._n), self._dm1, optimize=True)
        else:
            #
            # < |k l i^+ j^+| >
            #
            # M_klji = \delta_li \delta_kj - \delta_ki \delta_lj
            rdm_terms = np.einsum("li,kj->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            rdm_terms -= np.einsum("ki,lj->klji", np.eye(self._n), np.eye(self._n), optimize=True)
            # M_klji += \delta_{ki} \gamma_{jl} - \delta_{kj} \gamma_{li}
            #        += \delta_{lj} \gamma_{ki} - \delta_{li} \gamma_{jk}
            rdm_terms += np.einsum("ki,lj->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("kj,li->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms -= np.einsum("li,kj->klji", np.eye(self._n), self._dm1, optimize=True)
            rdm_terms += np.einsum("lj,ki->klji", np.eye(self._n), self._dm1, optimize=True)
            # M_klji += \Gamma_klji
            rdm_terms += self._dm2
        return rdm_terms


def reconstruct_dm2(cv, dm1, dm2, op, comm=True):
    """Reconstruct the 2-RDM in terms of the 1-RDM for the ground state
    and the transition-RDMs.

    Parameters
    ----------
    cv : np.ndarray((n**2, n**2))
        Eigenvector matrix.
    dm1 : np.ndarray((n, n))
        Spin resolved 1-particle reduced density matrix.
    dm2 : np.ndarray((n, n, n, n))
        Spin resolved 2-particle reduced density matrix.
    comm : bool, optional
        Form used to approximate the transition-RDMs, one of commutator (True) or no commutator
        (False), by default True

    Returns
    -------
    np.ndarray((n, n, n, n))
        Reconstructed 2-RDM.
    """
    n = dm1.shape[0]
    tv = np.zeros_like(dm2)
    if op == 'hh':
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = \sum_{n=0} < | p^+ q^+|N-2> <N-2|s r| >
        tdms = TDM(cv, dm1, dm2).get_tdm('hh', comm=comm)
        tv = np.zeros_like(dm2)
        for rdm in tdms:
            tv += np.einsum("pq,rs->pqrs", rdm, rdm, optimize=True)
        return tv/2
    elif op == 'pp':
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = \deta_ps \delta_rq - delta_pr \delta_sq
        #            + \delta_qs \gamma_pr - delta_qr \gamma_ps
        #            - \delta_ps \gamma_qr + delta_pr \gamma_qs
        #            + \sum_{n=0} < |s r|N+2> <N+2|p^+ q^+| >
        eye_eye = np.einsum("ps,rq->pqrs", np.eye(n), np.eye(n), optimize=True)
        eye_eye -= np.einsum("pr,sq->pqrs", np.eye(n), np.eye(n), optimize=True)
        eye_dm1 = np.einsum("qs,pr->pqrs", np.eye(n), dm1, optimize=True)
        eye_dm1 -= np.einsum("qr,ps->pqrs", np.eye(n), dm1, optimize=True)
        eye_dm1 -= np.einsum("ps,qr->pqrs", np.eye(n), dm1, optimize=True)
        eye_dm1 += np.einsum("pr,qs->pqrs", np.eye(n), dm1, optimize=True)
        linear_terms = eye_eye + eye_dm1
        # Compute term involvin the tdms
        # \sum_{n=0} < |s r|N+2> <N+2|p^+ q^+| >
        tdms = TDM(cv, dm1, dm2).get_tdm('pp', comm=comm)
        for rdm in tdms:
            tv += np.einsum("sr,qp->pqrs", rdm, rdm, optimize=True)
        return linear_terms + tv/2
    elif op == 'ph':
        # Gamma_pqrs = < | p^+ q^+ s r | >
        #            = - < | p^+ q^+ r s | >
        #            = - \delta_qr * \gamma_ps + \gamma_pr * \gamma_qs
        #            + \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
        # \gamma_pr * \gamma_qs - \delta_qr * \gamma_ps
        linear_terms = np.einsum("pr,qs->pqrs", dm1, dm1, optimize=True)
        linear_terms -= np.einsum("qr,ps->pqrs", np.eye(n), dm1, optimize=True)
        # Compute term involvin the tdms
        # \sum_{n!=0} (\gamma_pr;0n * \gamma_qs;n0)
        tdms = TDM(cv, dm1, dm2).get_tdm('ph', comm=comm)
        for rdm in tdms:
            # tv += np.einsum("pr,qs->pqrs", rdm, rdm, optimize=True)
            tv += np.einsum("pr,qs->pqrs", rdm, rdm.T, optimize=True)
        return linear_terms + tv
    else:
        raise ValueError("`op` must be one of `ph`, `pp` or `hh`.")


def pick_eigvals(ws, cv=None, sign=None, mult=None, tol=0.01, unique=False):
    if sign is None:
        idx = np.arange(len(ws))
    elif sign == 'p':
        idx = np.where(ws > tol ** 2)[0]
    elif sign == 'n':
        idx = np.where(-ws > tol ** 2)[0]

    if cv is None:
        ws = ws[idx]
        if unique:
            # get unique eigvals
            b = np.sort(ws)
            d = np.append(True, np.diff(b))
            TOL = 1e-6
            ws = b[d > TOL]
    else:
        ws, cv = ws[idx], cv[idx] # shorter or full spectrum solutions
        if mult == 1:
            # https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
            _,ids, counts = np.unique(ws.round(decimals=4), return_index=True, return_counts=True)
            # print('idx, counts', list(zip(idx, counts)))
            idx = np.sort(ids[counts==1])
            # print('idxs', idx)
            ws, cv = ws[idx], cv[idx]
    return ws, cv


def test_pick_eigvals():
    a = np.array([1, 2, 1, 0])
    c = np.arange(4*2).reshape(4,2)
    ev, cv = pick_eigvals(a, cv=c, sign='p', mult=1, tol=0.01, unique=False)
    assert np.allclose([2], ev)
    assert np.allclose(c[1], cv)
    a = np.array([1, -2, 1, -3])
    c = np.arange(4*2).reshape(4,2)
    ev, cv = pick_eigvals(a, cv=c, sign='n', mult=1, tol=0.01, unique=False)
    assert np.allclose([-2, -3], ev)
    assert np.allclose(c[[1,-1]], cv)
    ev, cv = pick_eigvals(a, cv=c, sign=None, mult=None, tol=0.01, unique=False)
    assert np.allclose(a, ev)
    assert np.allclose(c, cv)
# test_pick_eigvals()


def reconstruct_dm1(coeffs, metric):
    tdms = np.einsum('nj,ij->ni', coeffs, metric)
    gamma = np.zeros_like(metric)
    for tdm in tdms:
        gamma += np.einsum('p,q->pq', tdm, tdm)
    return gamma


def reconstruct_dm2_hh(coeffs, dmterms):
    # Reconstruct 2-RDM from hh-TDMs and check trace
    n = dmterms.shape[0]
    coeffs = coeffs.reshape(len(coeffs), n, n)
    tdms = np.einsum("msr,pqsr->mpq", coeffs, dmterms) #
    tv = np.zeros_like(dmterms) #dm2
    for rdm in tdms:
        tv += np.einsum("pq,rs->pqrs", rdm, rdm, optimize=True)
    return tv