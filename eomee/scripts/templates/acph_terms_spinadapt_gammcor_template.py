"""Title: $title"""
import os

import numpy as np

from scipy.integrate import quadrature as gauss

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


def omega_alpha_spinadapted(cv_aa, dvv, metric):
    nb = metric.shape[0]
    cv = cv_aa.reshape(cv_aa.shape[0], nb, nb)
    tdms_aa =  np.einsum("mrs,pqsr->mpq", cv, metric)
    tdtd_aa = np.zeros_like(metric)
    for rdm in tdms_aa:
        tdtd_aa += np.einsum("pr,qs->pqrs", rdm, rdm.T, optimize=True)
    tdtd_aa = 0.5 * tdtd_aa
    tdtd = spinize(tdtd_aa)
    dv_tdm = np.einsum("pqrs,pqrs", dvv, tdtd)
    return dv_tdm


def spin_adapt_singles(nbasis, lhs, rhs):
    nspins = 2 * nbasis
    lhs = lhs.reshape(nspins, nspins, nspins, nspins)
    rhs = rhs.reshape(nspins, nspins, nspins, nspins)
    A_aaaa = lhs[:nbasis, :nbasis, :nbasis, :nbasis]
    A_bbbb = lhs[nbasis:, nbasis:, nbasis:, nbasis:]
    A_aabb = lhs[:nbasis, :nbasis, nbasis:, nbasis:]
    A_bbaa = lhs[nbasis:, nbasis:, :nbasis, :nbasis]
    M_aaaa = rhs[:nbasis, :nbasis, :nbasis, :nbasis]
    M_bbbb = rhs[nbasis:, nbasis:, nbasis:, nbasis:]
    M_aabb = rhs[:nbasis, :nbasis, nbasis:, nbasis:]
    M_bbaa = rhs[nbasis:, nbasis:, :nbasis, :nbasis]
    A_sa = A_aaaa + A_bbbb + A_aabb + A_bbaa
    M_sa = M_aaaa + M_bbbb + M_aabb + M_bbaa
    A_sa = 0.5*A_sa.reshape(nbasis**2, nbasis**2)
    M_sa = 0.5*M_sa.reshape(nbasis**2, nbasis**2)
    return A_sa, M_sa


def W_alpha(la, _h0, _v0, _h1, _v1, rdm1, rdm2, _eigs, tol, singlet):
    solvers = {'nonsymm': solver_d, 'qtrunc': solver_d3} #, 'nonsymm2': solver_d2
    solver = solvers[_eigs]
    dvv = _v1 - _v0
    erpa = build_ph_gevp(_h0, _v0, _h1, _v1, rdm1, rdm2, la)
    nbasis = erpa.n // 2
    _lhs, _rhs = spin_adapt_singles(nbasis, erpa.lhs, erpa.rhs)
    ev, cv = solver(_lhs, _rhs, tol=tol)
    ev, cv = np.real(ev), np.real(cv)
    ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    if singlet:
        norm = np.dot(cv_p, np.dot(_rhs, cv_p.T))
        diag_n = np.diag(norm)
        sqr_n = np.sqrt(np.abs(diag_n)) #sqr_n = np.sqrt(diag_n)
        new_cv = cv_p.T / sqr_n
        cv_p = new_cv.T
        _rhs = _rhs.reshape(nbasis, nbasis, nbasis, nbasis)
        w_l = omega_alpha_spinadapted(cv_p, dvv, _rhs)
    else:
        raise NotImplementedError
    return w_l


def run_acph(NAME, operator, eigs, eigtol, alpha):
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
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)
    # min, max, step = alpha
    # walpha_sing = W_alpha(h0, v0, h1, v1, rdm1, rdm2, alpha, eigs, eigtol, singlet=True)
    # walpha_trip = W_alpha(h0, v0, h1, v1, rdm1, rdm2, alpha, eigs, eigtol, singlet=False)
    
    # int_wa_s = np.trapz(walpha_sing, dx=step)
    # int_wa_t = np.trapz(walpha_trip, dx=step)
    arg_s = (h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, True)
    # args_t = (h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, False)
    int_wa_s = gauss(W_alpha, 0, 1, args=arg_s, tol=1.e-4, maxiter=5, vec_func=False)[0]
    # int_wa_t = gauss(W_alpha, 0, 1, args=args_t, tol=1.e-4, maxiter=5, vec_func=False)[0]
    nonlinear = int_wa_s #+ int_wa_t
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc

    # Save EOM results
    np.savez(f"{NAME}.ac{operator}{eigs}.npz", energy=etot, ecorr=ecorr, ctnt=linear, integ=nonlinear, intega=int_wa_s, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-5
alpha = [0.0, 1.1, 0.1]


run_acph(NAME, 'ph', 'qtrunc', eigtol, alpha) #'safe', 'nonsymm
