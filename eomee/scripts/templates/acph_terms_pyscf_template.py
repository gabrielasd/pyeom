"""Title: $title"""
import os

import numpy as np

from scipy.integrate import quadrature as gauss

import pyci

from iodata import load_one

from eomee.excitation import EOMExc, _pherpa_linearterms
from eomee.tools import spinize, hartreefock_rdms, antisymmetrize, pickpositiveeig, TDM
from eomee.solver import nonsymmetric as solver_d
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


def omega_alpha(cv_aa, dvv, dm1, dm2):
    tdms = TDM(cv_aa, dm1, dm2).get_tdm('ph', comm=True)
    tdtd = np.zeros_like(dm2)
    for rdm in tdms:
        tdtd += np.einsum("pr,qs->pqrs", rdm, rdm.T, optimize=True)
    dv_tdm = np.einsum("pqrs,pqrs", dvv, tdtd)
    return dv_tdm


def build_ph_gevp(h0, v0, h1, v1, rdm1, rdm2, alpha):
    # Compute H^alpha
    dh= h1-h0
    dv= v1-v0
    h = alpha * dh
    h += h0
    v = alpha * dv
    v += v0
    return EOMExc(h, v, rdm1, rdm2)


def W_alpha(la, _h0, _v0, _h1, _v1, rdm1, rdm2, _eigs, tol, singlet):
    solvers = {'nonsymm': solver_d, 'qtrunc': solver_d3} #, 'nonsymm2': solver_d2
    solver = solvers[_eigs]
    dvv = _v1 - _v0
    erpa = build_ph_gevp(_h0, _v0, _h1, _v1, rdm1, rdm2, la)
    ev, cv = solver(erpa.lhs, erpa.rhs, tol=tol)
    ev, cv = np.real(ev), np.real(cv)
    ev_p, cv_p, _ = pickpositiveeig(ev, cv)
    if singlet:
        s_cv= get_singlets(ev_p, cv_p)[1]
        norm = np.dot(s_cv, np.dot(erpa.rhs, s_cv.T))
        diag_n = np.diag(norm)
        sqr_n = np.sqrt(np.abs(diag_n))
        new_cv = s_cv.T / sqr_n
        cv_s = new_cv.T
        w_l = omega_alpha(cv_s, dvv, rdm1, rdm2)
    else:
        cv_t = get_triplets(ev_p, cv_p)[1]
        w_l = omega_alpha(cv_t, dvv, rdm1, rdm2)
    return w_l


def run_acph(NAME, operator, solver, eigtol):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    nucnuc = ham.ecore
    nbasis = ham.nbasis

    if not os.path.isfile(f"{NAME}.molden"):
        raise ValueError(f'f"{NAME}.molden" not found')
    mol = load_one(f"{NAME}.molden")
    nelec = int(mol.nelec)
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

    arg_s = (h0, v0, h1, v1, rdm1, rdm2, solver, eigtol, True)
    args_t = (h0, v0, h1, v1, rdm1, rdm2, solver, eigtol, False)
    int_wa_s = gauss(W_alpha, 0, 1, args=arg_s, tol=1.e-4, maxiter=5, vec_func=False)[0]
    int_wa_t = gauss(W_alpha, 0, 1, args=args_t, tol=1.e-4, maxiter=5, vec_func=False)[0]
    nonlinear = int_wa_s + int_wa_t
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc

    np.savez(f"{NAME}.ac{operator}{solver}.npz", energy=etot, ecorr=ecorr, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7
eps = 1.0e-7
alpha = [0.0, 1.0, 0.1]


run_acph(NAME, 'ph', 'nonsymm', eigtol)
