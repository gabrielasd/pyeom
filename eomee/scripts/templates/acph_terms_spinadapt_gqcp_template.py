"""Title: $title"""
import os

import numpy as np

from scipy.integrate import quadrature as gauss

import pyci

from eomee.excitation import EOMExc, _pherpa_linearterms
from eomee.tools import spinize, pickpositiveeig, from_unrestricted
from eomee.solver import nonsymmetric as solver_d
# from eomee.solver import nonsymmetric2 as solver_d2
from eomee.solver import eig_pinv as solver_d3


def ham0_spinized(one_mo, two_mo):
    # DOCI Hamiltonian
    n = one_mo.shape[0]
    m = 2*n
    one_int_aa = np.zeros((n,n))
    two_int_aaaa = np.zeros((n,n,n,n))
    two_int_abab = np.zeros((n,n,n,n))

    for p in range(n):
        one_int_aa[p, p] = one_mo[p, p]
    # aaaa
    for p in range(n):
        for q in range(n):
            #may need to exclude p==q
            two_int_aaaa[p, q, p, q] = two_mo[p, q, p, q] 
            two_int_aaaa[p, q, q, p] = two_mo[p, q, q, p]
    # abab        
    for p in range(n):
        for q in range(n):
            two_int_abab[p, p, q, q] = two_mo[p, p, q, q]
            two_int_abab[p, q, p, q] = two_mo[p, q, p, q] # p==q overwrites above
    
    _h = np.zeros((m, m))
    _h[:n, :n] = one_int_aa
    _h[n:, n:] = one_int_aa
    _v = np.zeros((m, m, m, m))
    # aaaa
    _v[:n, :n, :n, :n] = two_int_aaaa
    # bbbb
    _v[n:, n:, n:, n:] = two_int_aaaa
    # abab
    _v[:n, n:, :n, n:] = two_int_abab
    _v[n:, :n, n:, :n] = two_int_abab.transpose((1,0,3,2))
    return _h, _v


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


def build_ph_gevp(h0, v0, h1, v1, rdm1, rdm2, alpha):
    # Compute H^alpha
    dh= h1-h0
    dv= v1-v0
    h = alpha * dh
    h += h0
    v = alpha * dv
    v += v0
    return EOMExc(h, v, rdm1, rdm2)


def spin_adapt_00(nbasis, lhs, rhs):
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


def omega_alpha_spinadapted_00(cv_aa, dvv, metric):
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


def spin_adapt_30(nbasis, lhs, rhs):
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
    A_sa = A_aaaa + A_bbbb - A_aabb - A_bbaa
    M_sa = M_aaaa + M_bbbb - M_aabb - M_bbaa
    A_sa = 0.5*A_sa.reshape(nbasis**2, nbasis**2)
    M_sa = 0.5*M_sa.reshape(nbasis**2, nbasis**2)
    return A_sa, M_sa


def omega_alpha_spinadapted_30(cv_aa, dvv, metric):
    nb = metric.shape[0]
    cv = cv_aa.reshape(cv_aa.shape[0], nb, nb)
    tdms_aa =  np.einsum("mrs,pqsr->mpq", cv, metric)
    tdtd_aa = np.zeros_like(metric)
    for rdm in tdms_aa:
        tdtd_aa += np.einsum("pr,qs->pqrs", rdm, rdm.T, optimize=True)
    tdtd_aa = 0.5 * tdtd_aa
    tdtd_ab = -tdtd_aa
    tdtd = from_unrestricted([tdtd_aa, tdtd_ab, tdtd_aa])
    dv_tdm = np.einsum("pqrs,pqrs", dvv, tdtd)
    return dv_tdm


def W_alpha(la, _h0, _v0, _h1, _v1, rdm1, rdm2, _eigs, tol, singlet):
    solvers = {'nonsymm': solver_d, 'qtrunc': solver_d3} #, 'nonsymm2': solver_d2
    solver = solvers[_eigs]
    dvv = _v1 - _v0
    erpa = build_ph_gevp(_h0, _v0, _h1, _v1, rdm1, rdm2, la)
    nbasis = erpa.n // 2

    if singlet:
        _lhs, _rhs = spin_adapt_00(nbasis, erpa.lhs, erpa.rhs)
    else:
        _lhs, _rhs = spin_adapt_30(nbasis, erpa.lhs, erpa.rhs)

    ev, cv = solver(_lhs, _rhs, tol=tol)
    ev, cv = np.real(ev), np.real(cv)
    _, cv_p, _ = pickpositiveeig(ev, cv)
    norm = np.dot(cv_p, np.dot(_rhs, cv_p.T))
    diag_n = np.diag(norm)
    sqr_n = np.sqrt(np.abs(diag_n))
    new_cv = cv_p.T / sqr_n
    cv_p = new_cv.T
    _rhs = _rhs.reshape(nbasis, nbasis, nbasis, nbasis)
    
    if singlet:
        w_l = omega_alpha_spinadapted_00(cv_p, dvv, _rhs)
    else:
        w_l = omega_alpha_spinadapted_30(cv_p, dvv, _rhs)
        
    return w_l

def run_acph(NAME, operator, eigs, eigtol, alpha):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    two_mo = np.einsum('ijkl->ikjl', data["twomo"])
    nucnuc = data["nuc"]
    nbasis = one_mo.shape[0]
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    rdm1 = from_spins(data['rdm1'])
    dm2aa, dm2ab, dm2ba, dm2bb = data['rdm2'] # transform 2-RDMs to our notation <|p*q*sr|>=\Gamma_pqrs
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    dm2ba = np.einsum("ijkl->ikjl", dm2ba)
    dm2bb = np.einsum("ijkl->ikjl", dm2bb)
    rdm2 = from_spins([dm2aa, dm2ab, dm2ba, dm2bb])

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    h0, v0 = ham0_spinized(one_mo, two_mo)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)     
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)
    # min, max, step = alpha
    # lpath = np.arange(min, max, step)
    # walpha_sing = []
    # walpha_trip = []
    # for l in lpath:
    #     val1 = W_alpha(l, h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, True)
    #     walpha_sing.append(val1)
    #     val2 = W_alpha(l, h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, False)
    #     walpha_trip.append(val2)
    # int_wa_s = np.trapz(walpha_sing, dx=step)
    # int_wa_t = np.trapz(walpha_trip, dx=step)

    arg_s = (h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, True)
    args_t = (h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, False)
    int_wa_s = gauss(W_alpha, 0, 1, args=arg_s, tol=1.e-4, maxiter=5, vec_func=False)[0]
    int_wa_t = gauss(W_alpha, 0, 1, args=args_t, tol=1.e-4, maxiter=5, vec_func=False)[0]
    nonlinear = int_wa_s + int_wa_t
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc

    # Save EOM results
    np.savez(f"{NAME}.ac{operator}{eigs}.npz", energy=etot, ecorr=ecorr, ctnt=linear, integ=nonlinear, intega=int_wa_s, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-5
alpha = [0.0, 1.0, 0.1]


run_acph(NAME, 'ph', 'qtrunc', eigtol, alpha) #'safe', 'nonsymm, 'qtrunc'
