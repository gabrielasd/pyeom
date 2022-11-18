"""Title: $title"""
import os

import numpy as np

from scipy.integrate import fixed_quad

import pyci

from eomee.excitation import EOMExc, WrappNonlinear,  _pherpa_linearterms
from eomee.tools import spinize, pickpositiveeig, TDM
from eomee.solver import nonsymmetric as solver_d
# from eomee.solver import nonsymmetric2 as solver_d2
from eomee.solver import eig_pinv as solver_d3 


def fill_ham_inter(two_mo0, two_mo, set_i, set_j, dm1):
    for p in set_i:
        for q in set_i:
            for r in set_j:
                # g_pqrs_aaaa
                two_mo0 [p, q]+= dm1[r,r]*two_mo[p,r,q,r]
                two_mo0[p, q] -= dm1[r,r]*two_mo[p,r,r,q]
                # g_pqrs_abab
                two_mo0[p, q] += dm1[r,r]*two_mo[p,r,q,r]
    return two_mo0


def fill_ham_intra(one_mo0, two_mo0, one_mo, two_mo, set_i):
    for p in set_i:
        for q in set_i:
            one_mo0[p, q] = one_mo[p, q]
            for r in set_i:
                for s in set_i:
                    two_mo0[p, q, r, s] = two_mo[p, q, r, s]
    return one_mo0, two_mo0


def make_gvbpp_ham_zero(one_mo, two_mo, gem_matrix, dm1a):
    k = one_mo.shape[0]
    assert k == gem_matrix.shape[0]
    n_gems = gem_matrix.shape[1]    

    one_mo0 = np.zeros_like(one_mo)
    two_mo0 = np.zeros_like(two_mo)
    two_mo_inter = np.zeros_like(one_mo)
    for i in range(n_gems):
        gem_i = np.nonzero(gem_matrix.T[i])[0]
        one_mo_0, two_mo_0 = fill_ham_intra(one_mo0, two_mo0, one_mo, two_mo, gem_i)
        for j in range(n_gems):
            if j != i:
                gem_j = np.nonzero(gem_matrix.T[j])[0]
                two_mo_inter = fill_ham_inter(two_mo_inter, two_mo, gem_i, gem_j, dm1a)    

    return one_mo_0, two_mo_0, two_mo_inter


def from_spins(blocks):
    r"""
    Return a two- or four- index array in the spin representation from blocks.

    A two-index array is recontrcuted from blocks (a, b).
    A four-index array is recontrcuted from blocks (aa, ab, ba, bb).

    """
    if len(blocks) == 2:
        for b in blocks:
            if b.ndim != 2:
                raise ValueError("Input must have ndim == 2")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k))
        y[:n, :n] = blocks[0]
        y[n:, n:] = blocks[1]
    elif len(blocks) == 4:
        for b in blocks:
            if b.ndim != 4:
                raise ValueError("Input must have ndim == 4")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k, k, k))
        y[:n, :n, :n, :n] = blocks[0]
        y[:n, n:, :n, n:] = blocks[1]
        y[n:, :n, n:, :n] = blocks[2]
        y[n:, n:, n:, n:] = blocks[3]
        y[:n, n:, n:, :n] = -blocks[1].transpose((0,1,3,2))
        y[n:, :n, :n, n:] = -blocks[1].transpose((1,0,2,3))
    else:
        raise ValueError("Invalid input")
    return y


def get_singlets(eigvals, eigvecs):
    # sort ev and cv correspondingly
    idx = eigvals.argsort()
    b = eigvals[idx]
    eigvecs = eigvecs[idx]
    # start picking up singlets
    mask = np.append(True, np.diff(b)) > 1.e-7
    unique_eigs_idx = np.where(mask)[0]
    number_unique_eigs = np.diff(unique_eigs_idx)
    idx = np.where(number_unique_eigs == 1)[0]
    singlet_idx = unique_eigs_idx[idx]
    if unique_eigs_idx[-1] == len(eigvals)-1:
        singlet_idx = np.append(singlet_idx, unique_eigs_idx[-1])
    singlets_ev = b[singlet_idx]
    singlets_cv = eigvecs[singlet_idx]
    return singlets_ev, singlets_cv, singlet_idx

def get_triplets(eigvals, eigvecs):
    # sort ev and cv correspondingly
    idx = eigvals.argsort()
    b = eigvals[idx]
    eigvecs = eigvecs[idx]
    # start picking up triplets
    _, _, singlet_idx = get_singlets(eigvals, eigvecs)
    triplets_ev = np.delete(b, singlet_idx)
    triplets_cv = np.delete(eigvecs, singlet_idx, axis=0)
    return triplets_ev, triplets_cv


def build_ph_gevp(h0, v0, h1, v1, rdm1, rdm2, alpha):
    # Compute H^alpha
    dh= h1-h0
    dv= v1-v0
    h = alpha * dh
    h += h0
    v = alpha * dv
    v += v0
    return EOMExc(h, v, rdm1, rdm2)


def load_accepted_pairs(fname):
    npairs = os.popen(f"grep 'Reduced\ to' {fname}").read()
    npairs = int(npairs.split()[-1])
    with open(fname, 'r') as f:
        content = f.read()
    last = content.split('Accepted pairs read:\n')[-1]
    accepted = last.split('\n')[:npairs]
    fn = lambda _row,x: int(_row.split()[x].strip()) -1 
    accepted = [[fn(row,0), fn(row,1)] for row in accepted]
    return accepted, npairs


def truncate_w_alpha_exch_accepted_pairs(_nbasis, _dv, _dm1dm1, _eyedm1, pairs):
    alpha_indep = 0
    # Only contributions from same spin components
    # alpha alpha alpha alpha
    _dv_aa = _dv[:_nbasis, :_nbasis, :_nbasis, :_nbasis]
    _dm1dm1_aa = _dm1dm1[:_nbasis, :_nbasis, :_nbasis, :_nbasis]
    _eyedm1_aa = _eyedm1[:_nbasis, :_nbasis, :_nbasis, :_nbasis]
    for p, q in pairs:
        for r, s in pairs:
            alpha_indep += 0.5 * _dv_aa[p,q,r,s] * _dm1dm1_aa[p,q,r,s]
            alpha_indep += 0.5 * _dv_aa[q,p,r,s] * _dm1dm1_aa[q,p,r,s]
            alpha_indep += 0.5 * _dv_aa[p,q,s,r] * _dm1dm1_aa[p,q,s,r]
            alpha_indep += 0.5 * _dv_aa[q,p,s,r] * _dm1dm1_aa[q,p,s,r]
            alpha_indep -= 0.5 * _dv_aa[p,q,r,s] * _eyedm1_aa[p,q,r,s]
            alpha_indep -= 0.5 * _dv_aa[q,p,r,s] * _eyedm1_aa[q,p,r,s]
            alpha_indep -= 0.5 * _dv_aa[p,q,s,r] * _eyedm1_aa[p,q,s,r]
            alpha_indep -= 0.5 * _dv_aa[q,p,s,r] * _eyedm1_aa[q,p,s,r]
    alpha_indep *= 2
    return alpha_indep


def truncate_w_alpha_exchange_terms(_nbasis, rhs, _dv, _dm1dm1, _eyedm1, tol):
    alpha_indep = 0
    nspins = 2*_nbasis
    nt = nspins**2
    ij_d_occs = np.diag(rhs)
    for pq in range(nt):
        for rs in range(nt):
            cond1 = np.abs(ij_d_occs[pq]) > tol
            cond2 = np.abs(ij_d_occs[rs]) > tol
            if cond1 and cond2:
                p = pq//nspins
                q = pq%nspins
                r = rs//nspins
                s = rs%nspins
                alpha_indep += _dv[p,q,r,s] * _dm1dm1[p,q,r,s]
                alpha_indep -= _dv[p,q,r,s] * _eyedm1[p,q,r,s]
    return 0.5*alpha_indep


def _erpa_tdms(erpa, _dm1, _dm2, _eigsol, _eig_tol, singl=True):
    solve_gevp = {'nonsymm': solver_d, 'qtrunc': solver_d3}[_eigsol]
    ev, cv = solve_gevp(erpa.lhs, erpa.rhs, tol=_eig_tol)
    ev, cv = np.real(ev), np.real(cv)
    pev, pcv, _ = pickpositiveeig(ev, cv)
    ###
    if singl:
        s_cv = get_singlets(pev, pcv)[1]
        norm = np.dot(s_cv, np.dot(erpa.rhs, s_cv.T))
        diag_n = np.diag(norm)
        sqr_n = np.sqrt(np.abs(diag_n))
        new_cv = s_cv.T / sqr_n
        pcv = new_cv.T
    else:
        pcv = get_triplets(pev, pcv)[1]
    ###
    tdms = TDM(pcv, _dm1, _dm2).get_tdm('ph', comm=True)
    _tdtd = np.zeros_like(_dm2)
    for rdm in tdms:
        _tdtd += np.einsum("pr,qs->pqrs", rdm, rdm.T, optimize=True)
    return _tdtd


def omega_alpha_summ(erpa, _dv, dm1, dm2, _eigsol, eig_tol):
    tdtd = _erpa_tdms(erpa, dm1, dm2, _eigsol, eig_tol, singl=True)
    w_alpha = 0.5*np.einsum("pqrs,pqrs", _dv, tdtd)
    tdtd = _erpa_tdms(erpa, dm1, dm2, _eigsol, eig_tol, singl=False)
    w_alpha += 0.5*np.einsum("pqrs,pqrs", _dv, tdtd)
    return w_alpha


def int_w_alpha_gauss(_n, _h0, _v0, _h1, _v1, rdm1, rdm2, gpairs, dv, exc_terms, _eigsol, eig_tol=1.e-7):
    dm1dm1_ex, eyedm1_ex = exc_terms
    @np.vectorize
    def _func(alpha):
        _erpa = build_ph_gevp(_h0, _v0, _h1, _v1, rdm1, rdm2, alpha)
        w_l = omega_alpha_summ(_erpa, dv, rdm1, rdm2, _eigsol, eigtol)
        if gpairs is not None:
            exch = truncate_w_alpha_exch_accepted_pairs(_n, dv, dm1dm1_ex, eyedm1_ex, gpairs)
        else:
            exch = truncate_w_alpha_exchange_terms(_n, _erpa.rhs, dv, dm1dm1_ex, eyedm1_ex, eig_tol)
        return w_l + exch
    
    return fixed_quad(_func, 0, 1, n=5)[0]


def run_acph(NAME, operator, eigs, eigtol):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    two_mo = data["twomo"]
    nucnuc = data["nuc"]
    nbasis = one_mo.shape[0]
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.gvb.npz"):
        raise ValueError(f"{NAME}.gvb.npz not found")
    data = np.load(f"{NAME}.gvb.npz")
    energy_gvb = data["energy"]
    dm1aa, dm1ab = data['rdm1']
    rdm1 = from_spins([dm1aa, dm1ab])
    dm2aaaa, dm2abab = data['rdm2']
    dm2baba = dm2abab.transpose((1,0,3,2))
    rdm2 = from_spins([dm2aaaa, dm2abab, dm2baba, dm2aaaa])
    
    nparts = int(np.ceil(np.trace(rdm1)))
    npairs = nparts // 2
    # FIXME: this is a hack to get the number of elgeminals right for H2 STO-6G
    ngems = npairs + 1     # considering fictitious geminals

    print('Load Geminals data')
    if not os.path.isfile("gvb_geminals.dat"):
        raise ValueError(f"gvb_geminals.dat not found")
    index_m = np.loadtxt("gvb_geminals.dat", dtype=int)
    assert index_m.shape[0] == nbasis
    assert index_m[nbasis-1, 1] == ngems
    index_m -= 1
    gem_mtrix = np.zeros((nbasis, ngems))
    for n,g in index_m:
        gem_mtrix[n, g] = 1.0

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    one_mo_0, two_mo_0, two_mo_0_inter = make_gvbpp_ham_zero(one_mo, two_mo, gem_mtrix, dm1aa)
    h0 = spinize(one_mo_0) 
    h0 += spinize(two_mo_0_inter)
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)
    # dh= h1-h0
    dv= v1-v0

    dm1dm1_ex = np.einsum("ps,qr->pqrs", rdm1, rdm1)
    eyedm1_ex = np.einsum("ps,qr->pqrs", rdm1, np.eye(rdm1.shape[0]))
    ph = build_ph_gevp(h0, v0, h1, v1, rdm1, rdm2, 0)
    pairs, _  = load_accepted_pairs("gammcor.out")
    # aindep_exch = truncate_w_alpha_exchange_terms(nbasis, ph.rhs, dv, dm1dm1_ex, eyedm1_ex, eigtol)
    aindep_exch = truncate_w_alpha_exch_accepted_pairs(nbasis, dv, dm1dm1_ex, eyedm1_ex, pairs)
    terms = [dm1dm1_ex, eyedm1_ex]
    ecorr = int_w_alpha_gauss(nbasis, h0, v0, h1, v1, rdm1, rdm2, pairs, dv, terms, eigs, eigtol)
    etot = energy_gvb + ecorr

    # Save EOM results
    np.savez(f"{NAME}.ac{operator}{eigs}.npz", energy=etot, ecorr=ecorr, ctnt=aindep_exch, integ=None, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-5


run_acph(NAME, 'ph', 'qtrunc', eigtol) #'safe', 'nonsymm
