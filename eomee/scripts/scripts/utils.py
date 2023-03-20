import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
from glob import glob

import numpy as np

from scipy import constants as const
ANGSTROM = const.physical_constants['Bohr radius'][0] / const.angstrom # Bohr radius in Angstrom
BOHR = 1
b2a = ANGSTROM / BOHR

import pyci

from eomee.doubleionization import EOMDIP
from eomee.spinadapted.particlehole import EOMExcSA
from eomee.spinadapted.holehole import DIPSA, Integrandhh, _hherpa_linearterms
from eomee.tools import hartreefock_rdms, make_spinized_fock_hamiltonian, make_doci_ham_spinized, make_gvbpp_hamiltonian
from eomee.tools import spinize, pickpositiveeig, from_unrestricted
from eomee.solver import nonsymmetric as solver_d




def load_rhf_data(nparts, NAME):
    # Load Hamiltonian
    if not os.path.isfile(f'{NAME}.ci.npz'):
        raise ValueError(f'{NAME}.ci.npz not found')
    ham = pyci.hamiltonian(f"{NAME}.FCIDUMP")
    rdm1, rdm2 = hartreefock_rdms(ham.nbasis,*nparts)
    return (ham.one_mo, ham.two_mo, ham.ecore, rdm1, rdm2)


def load_fci_rdms(NAME):
    # Load Hamiltonian
    if not os.path.isfile(f'{NAME}.FCIDUMP'):
        raise ValueError(f'{NAME}.FCIDUMP not found')
    data = np.load(f"{NAME}.ci.npz")
    dm1, dm2 = pyci.spinize_rdms(data['rdm1'], data['rdm2'])
    return (dm1, dm2)


def load_doci_data(NAME):
    # Load Hamiltonian
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    ham = np.load(f"{NAME}.ham.npz")
    twomo = np.einsum('ijkl->ikjl', ham["twomo"])
    # Load CI RDMs
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    dm2aa, dm2ab, dm2ba, dm2bb = data['rdm2']  # transform 2-RDMs to our notation <|p*q*sr|>=\Gamma_pqrs
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    dm2bb = np.einsum("ijkl->ikjl", dm2bb)
    rdm1 = from_unrestricted(data['rdm1'])
    rdm2 = from_unrestricted([dm2aa, dm2ab, dm2bb])
    n = ham["onemo"].shape[0]
    rdm2[:n, n:, n:, :n] = - dm2ab.transpose(0, 1, 3, 2)
    rdm2[n:, :n, :n, n:] = - dm2ab.transpose(1, 0, 2, 3)
    return (ham["onemo"], twomo, rdm1, rdm2)


def load_gvbpp_data(NAME):
    # Load Hamiltonian
    if not os.path.isfile(f'{NAME}.ham.npz'):
        raise ValueError(f'{NAME}.ham.npz not found')
    ham = np.load(f"{NAME}.ham.npz")
    # Load CI RDMs
    if not os.path.isfile(f"{NAME}.gvb.npz"):
        raise ValueError(f"{NAME}.gvb.npz not found")
    data = np.load(f"{NAME}.gvb.npz")
    dm2aa, dm2ab = data['rdm2']
    rdm1 = from_unrestricted(data['rdm1'])
    rdm2 = from_unrestricted([dm2aa, dm2ab, dm2aa])
    n = ham["onemo"].shape[0]
    rdm2[:n, n:, n:, :n] = - dm2ab.transpose(0, 1, 3, 2)
    rdm2[n:, :n, :n, n:] = - dm2ab.transpose(1, 0, 2, 3)
    return (ham["onemo"], ham["twomo"], rdm1, rdm2)


def load_purvis_rdms(NAME):
    # Load CI RDMs
    if not os.path.isfile(f"{NAME}.ci.npz"):
        raise ValueError(f"{NAME}.ci.npz not found")
    data = np.load(f"{NAME}.ci.npz")
    dm2aa, dm2ab, dm2bb = data['rdm2']  # transform 2-RDMs to our notation <|p*q*sr|>=\Gamma_pqrs
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    dm2bb = np.einsum("ijkl->ikjl", dm2bb)
    rdm1 = from_unrestricted(data['rdm1'])
    rdm2 = from_unrestricted([dm2aa, dm2ab, dm2bb])
    n = dm2aa.shape[0]
    rdm2[:n, n:, n:, :n] = - dm2ab.transpose(0, 1, 3, 2)
    rdm2[n:, :n, :n, n:] = - dm2ab.transpose(1, 0, 2, 3)
    return (rdm1, rdm2)


def alpha_ham_terms(oneint, twoint, one_dm, wfn, gem_m=None):
    if wfn == 'rhf':
        h0, v0 = make_spinized_fock_hamiltonian(oneint, twoint, one_dm)
    elif wfn == 'doci':
        h0, v0 = make_doci_ham_spinized(oneint, twoint)
    elif wfn == 'gvb':
        k = oneint.shape[0]
        oneint_0, twoint_0, twoint_0_inter = make_gvbpp_hamiltonian(oneint, twoint, gem_m, one_dm[:k, :k])
        h0 = spinize(oneint_0) + spinize(twoint_0_inter)
        v0 = spinize(twoint_0)
    else:
        raise ValueError('wfn must be rhf, doci or gvb')
    h1 = spinize(oneint) 
    v1 = spinize(twoint)
    dh= h1-h0
    dv= v1-v0
    return (h0, v0, dh, dv)


def build_hh_gevp(oneint, twoint, rdm1, rdm2, alpha, wfn):
    h0, v0, dh, dv = alpha_ham_terms(oneint, twoint, rdm1, wfn)
    # Compute H^alpha
    h = alpha * dh
    h += h0
    v = alpha * dv
    v += v0
    return EOMDIP(h, v, rdm1, rdm2)


def normalize_vectors(cv, metric):    
    norm = np.dot(cv, np.dot(metric, cv.T))
    diag_n = np.diag(norm)
    # print('diag_n = ', diag_n)
    idx = np.where(diag_n > 0)[0]  # Remove eigenvalues with negative norm
    sqr_n = np.sqrt(diag_n[idx])
    return (cv[idx].T / sqr_n).T


def ph_nocommutator_metric(dm1, dm2):
    m = dm1.shape[0]
    k = m//2
    I = np.eye(m)
    rhs = np.einsum("kj,li->klji",  dm1, I, optimize=True)
    rhs -= np.einsum("kijl->klji",  dm2, optimize=True)
    M_aaaa = rhs[:k, :k, :k, :k]
    M_bbbb = rhs[k:, k:, k:, k:]
    M_aabb = rhs[:k, :k, k:, k:]
    M_bbaa = rhs[k:, k:, :k, :k]
    M1 = M_aaaa + M_bbbb + M_aabb + M_bbaa
    M3 = M_aaaa + M_bbbb - M_aabb - M_bbaa
    return 0.5*M1, 0.5*M3


def hh_nocommutator_metric(dm2):
    k = dm2.shape[0]//2
    M_abab = dm2[:k, k:, :k, k:]
    M_baab = dm2[k:, :k, :k, k:]
    M_abba = dm2[:k, k:, k:, :k]    
    M_baba = dm2[k:, :k, k:, :k]
    M1 = M_abab - M_baab - M_abba + M_baba
    M3 = M_abab + M_baab + M_abba + M_baba
    return 0.5*M1, 0.5*M3


def phrpa_dm2_alpha(h, v, dm1, dm2, _tol=1.e-7, metric=None):
    erpa = EOMExcSA(h, v, dm1, dm2)
    w, c = erpa.solve_dense(tol=_tol, mode='qtrunc', mult=1)
    c = pickpositiveeig(w, c)[1]
    pcv00 = normalize_vectors(c, erpa._rhs1)
    w, c = erpa.solve_dense(tol=_tol, mode='qtrunc', mult=3)
    c = pickpositiveeig(w, c)[1]
    pcv30 = normalize_vectors(c, erpa._rhs3)
    m = erpa.n
    k = m // 2
    # 00
    if metric is None:
        dmterms = erpa._rhs1.reshape(k,k,k,k)
    tdms = np.einsum("mrs,pqrs->mpq", pcv00.reshape(pcv00.shape[0], k, k), dmterms)
    tv = np.zeros((k, k, k, k))
    for tdm in tdms:
        tv += np.einsum("pr,qs->pqrs", tdm, tdm, optimize=True)
    tdtd = spinize(tv)
    # 30
    if metric is None:
        dmterms = erpa._rhs3.reshape(k,k,k,k)
    tdms = np.einsum("mrs,pqrs->mpq", pcv30.reshape(pcv30.shape[0], k, k), dmterms)
    tv = np.zeros((k, k, k, k))
    for tdm in tdms:
        tv += np.einsum("pq,rs->pqrs", tdm, tdm, optimize=True)
    tdtd += from_unrestricted([tv, -tv, tv]) # aa, ab, bb
    dm = np.einsum("pr,qs->pqrs", dm1, dm1, optimize=True)
    dm -= np.einsum("qr,ps->pqrs", np.eye(m), dm1, optimize=True)
    return dm + tdtd


def hhrpa_dm2_alpha(h, v, dm1, dm2, _tol=1.e-7, metric=None):
    erpa = DIPSA(h, v, dm1, dm2)
    w, c = erpa.solve_dense(tol=_tol, mode='qtrunc', mult=1)
    c = pickpositiveeig(w, c)[1]
    pcv00 = normalize_vectors(c, erpa._rhs1)
    w, c = erpa.solve_dense(tol=_tol, mode='qtrunc', mult=3)
    c = pickpositiveeig(w, c)[1]
    pcv30 = normalize_vectors(c, erpa._rhs3)
    m = erpa.n
    k = m // 2
    tdtd = np.zeros((m,m,m,m))
    # 00
    if metric is None:
        dmterms = erpa._rhs1.reshape(k,k,k,k)
    else:
        dmterms = metric[0]
    tdms = np.einsum("mrs,pqrs->mpq", pcv00.reshape(pcv00.shape[0], k, k), dmterms)
    tv = np.zeros((k, k, k, k))
    for tdm in tdms:
        tv += np.einsum("pq,rs->pqrs", tdm, tdm, optimize=True)
    tdtd[:k, k:, :k, k:] = tv/2
    tdtd[k:, :k, k:, :k] = tv/2
    # 30
    if metric is None:
        dmterms = erpa._rhs3.reshape(k,k,k,k)
    else:
        dmterms = metric[1]
    tdms = np.einsum("mrs,pqrs->mpq", pcv30.reshape(pcv30.shape[0], k, k), dmterms)
    tv = np.zeros((k, k, k, k))
    for tdm in tdms:
        tv += np.einsum("pq,rs->pqrs", tdm, tdm, optimize=True)
    tdtd[:k, k:, :k, k:] += tv/2
    tdtd[k:, :k, k:, :k] += tv/2
    # 31
    tdtd[:k, :k, :k, :k] = tv
    tdtd[k:, k:, k:, k:] = tv
    return tdtd


def richerm2database(NAME):
    eigenvals = np.load(f"{NAME}_evals.npy")
    eigenvecs = np.load(f"{NAME}_evecs.npy")
    rdm1 =  np.load(f"{NAME}_1dm.npy")
    rdm2 = np.load(f"{NAME}_2dm.npy")
    # NAME = NAME.lower()
    np.savez(f"{NAME}.ci.npz", energy=eigenvals, coeff=eigenvecs, rdm1=rdm1, rdm2=rdm2)
