"""Title: $title"""
import os

import numpy as np

from iodata import load_one

import pyci

from eomee.excitation import EOMExc, WrappNonlinear,  _pherpa_linearterms
from eomee.tools import spinize, pickpositiveeig, TDM
from eomee.solver import nonsymmetric as solver_d
from eomee.solver import nonsymmetric2 as solver_d2


method = {'fci': pyci.fullci_wfn, 'doci': pyci.doci_wfn}


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


def get_pyci_rdms(nparts, nuc_rep, h, g):
    ham = pyci.hamiltonian(nuc_rep, h, g)
    wfn = pyci.doci_wfn(ham.nbasis, *nparts)
    wfn.add_all_dets()

    op = pyci.sparse_op(ham, wfn)
    ev, cv = op.solve(n=1, tol=1.0e-9)
    d1, d2 = pyci.compute_rdms(wfn, cv[0])
    dm1, dm2 = pyci.spinize_rdms(d1, d2)
    return pyci.spinize_rdms(d1, d2)


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


def run_pyci(n_up, n_dn, fname, n_procs, nsol=1, wfn_type=pyci.fullci_wfn):
    """run pyci v0.6.0"""
    pyci.set_num_threads(int(n_procs))
    ham = pyci.hamiltonian(f"{fname}.FCIDUMP")
    wfn = wfn_type(ham.nbasis, n_up, n_dn)
    wfn.add_all_dets()

    # Solve
    op = pyci.sparse_op(ham, wfn)
    ev, cv = op.solve(n=nsol, tol=1.0e-9)
    return ev, cv, wfn


def run_acph(NAME, operator, eigs, eigtol, alpha):
    # Get electron integrals in MO format
    print('Load Hamiltonian')
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    one_mo = ham.one_mo 
    two_mo = ham.two_mo
    nucnuc = ham.ecore
    
    print('Load CI')
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    rdm1, rdm2 = pyci.spinize_rdms(data['rdm1'], data['rdm2'])

    # Evaluate AC-ERPA (DIP)
    print('Run AC-ERPA (particle-hole)')
    if operator != 'ph':
        raise ValueError('Invalid operator.')
    one_mo_0, two_mo_0 = make_doci_hamiltonian(one_mo, two_mo)
    h1 = spinize(one_mo) 
    v1 = spinize(two_mo) 
    h0 = spinize(one_mo_0) 
    v0 = spinize(two_mo_0)
    dh= h1-h0
    dv= v1-v0
    energy = np.einsum('ij,ji', h0, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)

    linear = _pherpa_linearterms(h1.shape[0], dh, dv, rdm1)
    min, max, step = alpha
    alphas = np.arange(min, max, step)
    # function = WrappNonlinear(EOMExc, h0, v0, dh, dv, rdm1, rdm2)

    # if solver == 'nonsymm':
    #     dE_da = [function(alpha, mode=solver, tol=eigtol) for alpha in alphas]
    # elif solver in ['nonsymm2', 'prunned']:
    #     assert epsilon is not None
    #     dE_da = [function(alpha, mode=solver, tol=eigtol, eps=epsilon) for alpha in alphas]
    # else:
    #     raise ValueError('Invalid eigensolver, must be `nonsymm`, `nonsymm2` or `prunned`.')
    walpha_sing = W_alpha(h0, v0, h1, v1, rdm1, rdm2, alpha, eigs, eigtol, singlet=True)
    int_wa_s = np.trapz(walpha_sing, dx=step)
    nonlinear = int_wa_s
    ecorr = linear + 0.5 * nonlinear
    etot = energy + ecorr + nucnuc

    np.savez(f"{NAME}.ac{operator}{eigs}.npz", energy=etot, ecorr=ecorr, ctnt=linear, integ=nonlinear, intega=int_wa_s, abserr=None)
    print('')


NAME = '$output'
NELEC = $nelec
CHARGE = $charge
MULT = $spinmult
WFN = '$lot'.lower()
nprocs = $nprocs
eigtol = 1.0e-7
# eps = 1.0e-2 #1.0e-7
alpha = [0.0, 1.0, 0.1]
n_a = NELEC // 2


# Run HCI
eigenvals, eigenvecs, pyci_wfn = run_pyci(n_a, n_a, NAME, nprocs, wfn_type=method[WFN])
energy = eigenvals[0]
print("HCI energy", energy)

# Compute HCI RDMs
rdm1, rdm2 = pyci.compute_rdms(pyci_wfn, eigenvecs[0])
# Save HCI data
np.savez(f"{NAME}.ci.npz", energy=eigenvals, coeff=eigenvecs, dets=pyci_wfn.to_occ_array(), rdm1=rdm1, rdm2=rdm2)


run_acph(NAME, 'ph', 'nonsymm', eigtol, alpha) #'safe', 'nonsymm', prunned
