"""Title: $title"""
import os

import numpy as np

from scipy.integrate import quadrature as gauss

from eomee.excitation import EOMExc, _pherpa_linearterms
from eomee.tools import spinize, pickpositiveeig, from_unrestricted, antisymmetrize, hartreefock_rdms
from eomee.solver import nonsymmetric as solver_d
# from eomee.solver import nonsymmetric2 as solver_d2
from eomee.solver import eig_pinv as solver_d3


def make_spinized_fock_hamiltonian(one_mo, two_mo, one_dm):
    one_mo = spinize(one_mo)
    two_mo = spinize(two_mo)
    # Build Fock operator
    Fk = np.copy(one_mo)
    Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
    one_mo_0 = Fk
    two_mo_0 = np.zeros_like(two_mo)
    return one_mo_0, two_mo_0


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


def run_acph(NAME, nelec, operator, eigs, eigtol):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    two_mo = np.einsum('ijkl->ikjl', data["twomo"])
    nucnuc = data["nuc"]
    nbasis = one_mo.shape[0]
    
    na = nelec // 2
    nb = nelec // 2
    rdm1, rdm2 = hartreefock_rdms(nbasis, na, nb)

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    h0, v0 = make_spinized_fock_hamiltonian(one_mo, two_mo, rdm1)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)     
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)

    arg_s = (h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, True)
    args_t = (h0, v0, h1, v1, rdm1, rdm2, eigs, eigtol, False)
    int_wa_s = gauss(W_alpha, 0, 1, args=arg_s, tol=1.e-4, maxiter=5, vec_func=False)[0]
    int_wa_t = gauss(W_alpha, 0, 1, args=args_t, tol=1.e-4, maxiter=5, vec_func=False)[0]
    nonlinear = int_wa_s + int_wa_t
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc

    # Save EOM results
    np.savez(f"{NAME}.rpaac{operator}{eigs}.npz", energy=etot, ecorr=ecorr, ctnt=linear, integ=nonlinear, intega=int_wa_s, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
NELEC = $nelec
eigtol = 1.0e-5


run_acph(NAME, NELEC, 'ph', 'qtrunc', eigtol) #'safe', 'nonsymm, 'qtrunc'
