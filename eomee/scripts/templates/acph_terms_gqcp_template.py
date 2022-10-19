"""Title: $title"""
import os

import numpy as np

from iodata import load_one

import pyci

from eomee.excitation import EOMExc, WrappNonlinear,  _pherpa_linearterms
from eomee.tools import spinize, pickpositiveeig, TDM
from eomee.solver import nonsymmetric as solver_d
from eomee.solver import nonsymmetric2 as solver_d2


def make_doci_hamiltonian(one_mo, two_mo):
    nbasis = one_mo.shape[0]
    one_mo_sen0 = np.zeros_like(one_mo)
    two_mo_sen0 = np.zeros_like(two_mo)
    for p in range(nbasis):
        one_mo_sen0[p, p] = one_mo[p, p]
        for q in range(nbasis):
            two_mo_sen0[p, p, q, q] = two_mo[p, p, q, q]
            two_mo_sen0[p, q, p, q] = two_mo[p, q, p, q]
            two_mo_sen0[p, q, q, p] = two_mo[p, q, q, p]
    return one_mo_sen0, two_mo_sen0


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
            two_int_abab[p, q, p, q] = two_mo[p, q, p, q] #may need to exclude p==q
    
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


def get_pyci_rdms(nparts, nuc_rep, h, g):
    ham = pyci.hamiltonian(nuc_rep, h, g)
    wfn = pyci.doci_wfn(ham.nbasis, *nparts)
    wfn.add_all_dets()

    op = pyci.sparse_op(ham, wfn)
    ev, cv = op.solve(n=1, tol=1.0e-9)
    d1, d2 = pyci.compute_rdms(wfn, cv[0])
    dm1, dm2 = pyci.spinize_rdms(d1, d2)
    return pyci.spinize_rdms(d1, d2)


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


def omega_alpha_singlet(cv_aa, dvv, dm1, dm2):
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


def W_alpha(h0, v0, h1, v1, rdm1, rdm2, lpath, _eigs, tol, singlet=True):
    solvers = {'nonsymm': solver_d, 'nonsymm2': solver_d2}
    solver = solvers[_eigs]
    integrand = []
    min, max, step = lpath
    lpath = np.arange(min, max, step)
    dvv = v1- v0    
    for l in lpath:
        erpa = build_ph_gevp(h0, v0, h1, v1, rdm1, rdm2, l)
        ev, cv = solver(erpa.lhs, erpa.rhs, tol=tol)
        ev, cv = np.real(ev), np.real(cv)
        ev_p, cv_p, _ = pickpositiveeig(ev, cv)
        if singlet:
            s_cv= get_singlets(ev_p, cv_p)[1]
            norm = np.dot(s_cv, np.dot(erpa.rhs, s_cv.T))
            diag_n = np.diag(norm)
            sqr_n = np.sqrt(diag_n)
            new_cv = s_cv.T / sqr_n
            cv_s = new_cv.T
            w_l = omega_alpha_singlet(cv_s, dvv, rdm1, rdm2)
        else:
            cv_t = get_triplets(ev_p, cv_p)[1]
            w_l = omega_alpha_triplet(cv_t, dvv, rdm1, rdm2)
        integrand.append(w_l)
    return integrand

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
    one_mo_0, two_mo_0 = make_doci_hamiltonian(one_mo, two_mo)
    h0 = spinize(one_mo_0) 
    v0 = spinize(two_mo_0)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo)     
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)
    min, max, step = alpha
    walpha_sing = W_alpha(h0, v0, h1, v1, rdm1, rdm2, alpha, eigs, eigtol, singlet=True)

    int_wa_s = np.trapz(walpha_sing, dx=step)
    nonlinear = int_wa_s
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc

    # Save EOM results
    np.savez(f"{NAME}.ac{operator}{eigs}.npz", energy=etot, ecorr=ecorr, ctnt=linear, integ=nonlinear, intega=int_wa_s, abserr=None)
    print('')


NAME = '$output'
CHARGE = $charge
MULT = $spinmult
eigtol = 1.0e-7
# eps = 1.0e-2
alpha = [0.0, 1.0, 0.1]


run_acph(NAME, 'ph', 'nonsymm', eigtol, alpha) #'safe', 'nonsymm
