
import numpy as np

from eomee import EOMExc, EOMDEA, EOMDIP, EOMDIP2, EOMExc0, EOMDEA_2, EOMIP, EOMIPDoubleCommutator, EOMEA, EOMEADoubleCommutator
from eomee.tools import hartreefock_rdms, spinize, antisymmetrize, pickpositiveeig, from_unrestricted
from .rhotransition import check_rdm2_symmetry, reconstruct_dm2


method = {'pp': EOMDEA, 'ph': EOMExc, 'hh': EOMDIP, 'hh2': EOMDIP2, 'ph2': EOMExc0, 'pp2': EOMDEA_2,
'ip': EOMIP, 'ip2': EOMIPDoubleCommutator, 'ea': EOMEA, 'ea2': EOMEADoubleCommutator}


def make_doci_hamiltonian(one_mo, two_mo):
    """Build seniority zero Hamiltonian

    Parameters
    ----------
    one_mo : numpy.array
        one-electron integrals in MO basis; (K, K) matrix, where K is number of spatial orbitals.
    two_mo : numpy.array
        two-electron integrals in MO basis; (K, K, K, K) tensor, where K is number of spatial orbitals.

    Returns
    -------
    numpy.array
        one- and two- electron integrals corresponding to the seniority zero sector of the Hamiltonian operator
    """
    # DOCI Hamiltonian
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


def wrap_pyci(nparts, nuc_rep, h, g, ref, ns=1):
    import pyci

    wfntype = {'doci': pyci.doci_wfn, 'fci': pyci.fullci_wfn, }
    ham = pyci.hamiltonian(nuc_rep, h, g)
    if ref == 'hf':
        wfn = pyci.fullci_wfn(ham.nbasis, *nparts)
        wfn.add_hartreefock_det()
    elif ref in ['doci', 'fci']:
        wfn = wfntype[ref](ham.nbasis, *nparts)
        wfn.add_all_dets()
    else:
        raise ValueError('wrong wavefunction type.')
    op = pyci.sparse_op(ham, wfn)
    ev, cv = op.solve(n=ns, tol=1.0e-9)
    d1, d2 = pyci.compute_rdms(wfn, cv[0])
    dm1, dm2 = pyci.spinize_rdms(d1, d2)
    output = {
        'ev': ev,
        'cv': cv,
        'dm1': dm1,
        'dm2': dm2,
    }
    return output


def build_gevp(operator, nparts, one_mo, two_mo, one_dm, two_dm, restricted=True, wfntype='hf'):
    if wfntype in ['rhf', 'uhf']:
        wfntype = 'hf'
    if restricted:
        spatial2spin = spinize
    else:
        spatial2spin = from_unrestricted
    na, nb = nparts
    if one_dm is None and two_dm is None:
        assert wfntype == 'hf'
        nbasis = one_mo.shape[0]
        one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)
    
    # one_mo = spatial2spin(one_mo)
    # two_mo = spatial2spin(two_mo)
    if wfntype in ['hf', 'fci']:
        h = spatial2spin(one_mo)
        v = spatial2spin(two_mo)
    if wfntype == 'doci':
        onemo, twomo = make_doci_hamiltonian(one_mo, two_mo)
        h = spatial2spin(onemo)
        v = spatial2spin(twomo)

    return method[operator](h, v, one_dm, two_dm)

def run_eom(operator, nparts, one_mo, two_mo, one_dm=None, two_dm=None, restricted=True, wfn='hf', tol=1.0e-7, orthog="nonsymm"):
    """Evaluate ERPA (with commutator on right-hand-side).

    """
    # if restricted:
    #     spatial2spin = spinize
    # else:
    #     spatial2spin = from_unrestricted
    # na, nb = nparts
    # if one_dm is None and two_dm is None:
    #     nbasis = one_mo.shape[0]
    #     one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)
    
    # one_mo = spatial2spin(one_mo)
    # two_mo = spatial2spin(two_mo)
    # erpa = method[operator](one_mo, two_mo, one_dm, two_dm)
    # Solve particle-hole ERPA.
    erpa = build_gevp(operator, nparts, one_mo, two_mo, one_dm, two_dm, wfntype=wfn, restricted=restricted)
    w, cv = erpa.solve_dense(mode=orthog, tol=tol)

    return {
        'h': erpa.h, #one_mo
        'v': erpa.v, #two_mo
        'dm1': erpa.dm1, #one_dm
        'dm2': erpa.dm2, #two_dm
        'w': w,
        'cv': cv,
        'gevp': erpa
    }


def run_erpa(nparts, one_mo, two_mo, one_dm=None, two_dm=None, nucnuc=0, operator='ph', restricted=True, comm=True, solver_tol=1.0e-7):
    """Evaluate ph-ERPA (with commutator on right-hand-side).

    Approximate the transition RDM (using the commutator of the transition operator) and reconstruct the 2-RDM.

    """
    na, nb = nparts
    eom = run_eom(nparts, one_mo, two_mo, one_dm=one_dm, two_dm=two_dm, operator=operator, restricted=restricted, tol=solver_tol)    

    # Reconstruct 2RDM from T-RDM and check it adds to the right number of electron pairs (normalization condition).
    _, pcv, _ = pickpositiveeig(eom['w'], eom['cv'])
    erpa_rdm2 = reconstruct_dm2(pcv, eom['dm1'], eom['dm2'], operator, comm=comm)
    # assert np.allclose(np.einsum("ijij", erpa_rdm2), ((na + nb) * ((na + nb) - 1)))
    # print('Reconstructed 2-RDm normalization condition\n', np.einsum("ijij", erpa_rdm2), ((na + nb) * ((na + nb) - 1)))

    # Energy from reconstructed 2-RDMs
    energy = np.einsum("ij,ij", eom['h'], eom['dm1']) + 0.5 * np.einsum("ijkl,ijkl", eom['v'], erpa_rdm2)
    return energy + nucnuc


def run_erpa_ac(nparts, one_mo, two_mo, one_dm=None, two_dm=None, nucnuc=0, operator='ph', restricted=True, wfn='hf', solver_tol=1.0e-7):
    """Evaluate ERPA-AC (with commutator on right-hand-side)."""
    if restricted:
        spatial2spin = spinize
    else:
        spatial2spin = from_unrestricted
    
    na, nb = nparts
    if one_dm is None and two_dm is None:
        if not wfn == 'hf':
            raise NotImplementedError('The one- and two-electron RDMs must be provided')
        nbasis = one_mo.shape[0]
        one_dm, two_dm = hartreefock_rdms(nbasis, na, nb)
    
    if wfn == 'doci':
        one_mo_0, two_mo_0 = make_doci_hamiltonian(one_mo, two_mo)
        one_mo = spatial2spin(one_mo) # in spin-resolved form
        two_mo = spatial2spin(two_mo) # in spin-resolved form
        one_mo_0 = spatial2spin(one_mo_0) # in spin-resolved form
        two_mo_0 = spatial2spin(two_mo_0) # in spin-resolved form
        energy = np.einsum("ij,ij", one_mo_0, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo_0, two_dm
        )
    elif wfn == 'fci':
        one_mo_0, two_mo_0 = one_mo, two_mo
        one_mo = spatial2spin(one_mo) # in spin-resolved form
        two_mo = spatial2spin(two_mo) # in spin-resolved form
        one_mo_0 = spatial2spin(one_mo_0) # in spin-resolved form
        two_mo_0 = spatial2spin(two_mo_0) # in spin-resolved form
        energy = np.einsum("ij,ij", one_mo_0, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo_0, two_dm
        )
    else:
        one_mo = spatial2spin(one_mo)
        two_mo = spatial2spin(two_mo)
        # Build Fock operator
        Fk = np.copy(one_mo)
        Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
        one_mo_0 = Fk
        two_mo_0 = np.zeros_like(two_mo)
        energy = np.einsum("pq, pq", Fk, one_dm)
    print('energy at alpha 0', energy + nucnuc)
    
    # Evaluate ERPA-AC
    dE = method[operator].erpa(
        one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm, orthog="asymmetric", tol=solver_tol
    )

    return energy + dE + nucnuc


def overlap(operator, dm1, dm2):
    n = dm1.shape[0]
    I = np.eye(n, dtype=dm1.dtype)

    if operator in ['ph', 'ph2']:
        m = np.einsum("kj,li->klji", dm1, I, optimize=True)
        m -= np.einsum("kijl->klji", dm2, optimize=True)
    elif operator in ['hh', 'hh2']:
        m = dm2
    elif operator in ['pp', 'pp2']:
        # M_klji = \delta_li \delta_kj - \delta_ki \delta_lj
        m = np.einsum("li,kj->klji", I, I)
        m -= np.einsum("ki,lj->klji", I, I)
        # M_klji += \delta_{ki} \gamma_{jl} - \delta_{kj} \gamma_{li}
        #        += \delta_{lj} \gamma_{ki} - \delta_{li} \gamma_{jk}
        m += np.einsum("ki,lj->klji", I, dm1)
        m -= np.einsum("kj,li->klji", I, dm1)
        m -= np.einsum("li,kj->klji", I, dm1)
        m += np.einsum("lj,ki->klji", I, dm1)
        # M_klji += \Gamma_klji
        m += dm2
    else:
        raise ValueError('Wrong operator.')
    
    return m.reshape(n ** 2, n ** 2)


def brute_hherpa_lhs(h, v, dm1, dm2):
    """hole-hole ERPA left-hand side supermatrix

    Parameters
    ----------
    h : ndarray(n,n)
        spin resolved one-electron integrals
    v : ndarray(n,n,n,n)
        spin resolved two-electron integrals (<pq|rs>)
    dm1 : ndarray(n,n)
        1-RDM
    dm2 : ndarray(n,n,n,n)
        2-RDM
    """
    n = h.shape[0]
    I = np.eye(n, dtype=h.dtype)

    a = np.einsum("ik,lj->klji", h, dm1)
    a -= np.einsum("jk,li->klji", h, dm1)
    a += np.einsum("jl,ki->klji", h, dm1)
    a -= np.einsum("il,kj->klji", h, dm1)
    #
    b = np.einsum("ik,jp->ijkp", I, I)
    hdm1 = np.einsum("pq,lq->pl", h, dm1)
    a += np.einsum("pl,ijkp->klji", hdm1, b)
    a -= np.einsum("pl,ijkp->klji", h, b)
    b = np.einsum("ip,jk->ijpk", I, I)
    a -= np.einsum("pl,ijpk->klji", hdm1, b)
    a += np.einsum("pl,ijpk->klji", h, b)
    #
    b = np.einsum("il,jp->ijlp", I, I)
    hdm1 = np.einsum("pq,kq->pk", h, dm1)
    a += np.einsum("pk,ijlp->klji", h, b)
    a -= np.einsum("pk,ijlp->klji", hdm1, b)
    b = np.einsum("ip,jl->ijpl", I, I)
    a -= np.einsum("pk,ijpl->klji", h, b)
    a += np.einsum("pk,ijpl->klji", hdm1, b)
    ##
    # # b = np.einsum("pi,qj->pqij", I, I)
    # # vI = np.einsum("pqrs,pqij->rsij", v, b)
    # # b = np.einsum("kr,ls->klrs", I, I)
    # # a -= 0.5 * np.einsum("rsij,klrs->klji", vI, b)
    # # b = np.einsum("ks,lr->klsr", I, I)
    # # a += 0.5 * np.einsum("rsij,klsr->klji", vI, b)
    # # #
    # # b = np.einsum("pj,qi->pqji", I, I)
    # # vI = np.einsum("pqrs,pqji->rsji", v, b)
    # # b = np.einsum("kr,ls->klrs", I, I)
    # # a += 0.5 * np.einsum("rsji,klrs->klji", vI, b)
    # # b = np.einsum("ks,lr->klsr", I, I)
    # # a -= 0.5 * np.einsum("rsji,klsr->klji", vI, b)
    nu = v - v.transpose(1, 0, 2, 3)
    a -= np.einsum("ijkl->klji", nu)
    ##
    b = np.einsum("kr,ls->klrs", I, dm1)
    b -= np.einsum("ks,lr->klrs", I, dm1)
    b -= np.einsum("lr,ks->klrs", I, dm1)
    b += np.einsum("ls,kr->klrs", I, dm1)
    nu = v - v.transpose(1, 0, 2, 3)
    a += 0.5 * np.einsum("ijrs,klrs->klji", nu, b)
    #
    b = np.einsum("qj,pi->qpji", I, dm1)
    b -= np.einsum("qi,pj->qpji", I, dm1)
    b -= np.einsum("pj,qi->qpji", I, dm1)
    b += np.einsum("pi,qj->qpji", I, dm1)
    nu = v - v.transpose(0, 1, 3, 2)
    a += 0.5 * np.einsum("pqkl,qpji->klji", nu, b)
    ##
    b = np.einsum("pj,qr->pqjr", I, dm1)
    b -= np.einsum("qj,pr->pqjr", I, dm1)
    ii = np.einsum("ki,ls->klis", I, I)
    ii -= np.einsum("li,ks->klis", I, I)
    vb = np.einsum("pqrs,pqjr->sj", v, b)
    a += 0.5 * np.einsum("sj,klis->klji", vb, ii)
    #
    b = np.einsum("pj,qs->pqjs", I, dm1)
    b -= np.einsum("qj,ps->pqjs", I, dm1)
    ii = np.einsum("li,kr->lkir", I, I)
    ii -= np.einsum("ki,lr->lkir", I, I)
    vb = np.einsum("pqrs,pqjs->rj", v, b)
    a += 0.5 * np.einsum("rj,lkir->klji", vb, ii)
    #
    b = np.einsum("pi,qs->pqis", I, dm1)
    b -= np.einsum("qi,ps->pqis", I, dm1)
    ii = np.einsum("kj,lr->kljr", I, I)
    ii -= np.einsum("lj,kr->kljr", I, I)
    vb = np.einsum("pqrs,pqis->ri", v, b)
    a += 0.5 * np.einsum("ri,kljr->klji", vb, ii)
    #
    b = np.einsum("pi,qr->pqir", I, dm1)
    b -= np.einsum("qi,pr->pqir", I, dm1)
    ii = np.einsum("lj,ks->lkjs", I, I)
    ii -= np.einsum("kj,ls->lkjs", I, I)
    vb = np.einsum("pqrs,pqir->si", v, b)
    a += 0.5 * np.einsum("si,lkjs->klji", vb, ii)
    ##
    a -= 0.5 * np.einsum("jqks,qlsi->klji", v, dm2)
    a += 0.5 * np.einsum("jqrk,qlri->klji", v, dm2)
    a += 0.5 * np.einsum("iqks,qlsj->klji", v, dm2)
    a -= 0.5 * np.einsum("iqrk,qlrj->klji", v, dm2)
    #
    a += 0.5 * np.einsum("jqls,qksi->klji", v, dm2)
    a -= 0.5 * np.einsum("jqrl,qkri->klji", v, dm2)
    a -= 0.5 * np.einsum("iqls,qksj->klji", v, dm2)
    a += 0.5 * np.einsum("iqrl,qkrj->klji", v, dm2)
    #
    a += 0.5 * np.einsum("pjks,plsi->klji", v, dm2)
    a -= 0.5 * np.einsum("pjrk,plri->klji", v, dm2)
    a -= 0.5 * np.einsum("piks,plsj->klji", v, dm2)
    a += 0.5 * np.einsum("pirk,plrj->klji", v, dm2)
    #
    a -= 0.5 * np.einsum("pjls,pksi->klji", v, dm2)
    a += 0.5 * np.einsum("pjrl,pkri->klji", v, dm2)
    a += 0.5 * np.einsum("pils,pksj->klji", v, dm2)
    a -= 0.5 * np.einsum("pirl,pkrj->klji", v, dm2)
    ##
    vdm2 = np.einsum("iqrs,qlrs->il", v, dm2)
    a += 0.5 * np.einsum("kj,il->klji", I, vdm2)
    vdm2 = np.einsum("jqrs,qlrs->jl", v, dm2)
    a -= 0.5 * np.einsum("ki,jl->klji", I, vdm2)
    vdm2 = np.einsum("jqrs,qkrs->jk", v, dm2)
    a += 0.5 * np.einsum("li,jk->klji", I, vdm2)
    vdm2 = np.einsum("iqrs,qkrs->ik", v, dm2)
    a -= 0.5 * np.einsum("lj,ik->klji", I, vdm2)
    #
    vdm2 = np.einsum("pjrs,plrs->jl", v, dm2)
    a += 0.5 * np.einsum("ki,jl->klji", I, vdm2)
    vdm2 = np.einsum("pirs,plrs->il", v, dm2)
    a -= 0.5 * np.einsum("kj,il->klji", I, vdm2)
    vdm2 = np.einsum("pirs,pkrs->ik", v, dm2)
    a += 0.5 * np.einsum("lj,ik->klji", I, vdm2)
    vdm2 = np.einsum("pjrs,pkrs->jk", v, dm2)
    a -= 0.5 * np.einsum("li,jk->klji", I, vdm2)
    return a.reshape(n**2, n**2)


def hherpa_ac(one_mo, two_mo, one_dm, two_dm, nucnuc, tol=1.0e-7, wfn='hf'): 
    from scipy.integrate import quad as integrate
    from src.scripts.tools_erpa import solve_dense, solve_lowdin   
    if wfn == 'doci':
        one_mo_0, two_mo_0 = make_doci_hamiltonian(one_mo, two_mo)
        one_mo = spinize(one_mo) # in spin-resolved form
        two_mo = spinize(two_mo) # in spin-resolved form
        one_mo_0 = spinize(one_mo_0) # in spin-resolved form
        two_mo_0 = spinize(two_mo_0) # in spin-resolved form
        energy = np.einsum("ij,ij", one_mo_0, one_dm) + 0.5 * np.einsum(
        "ijkl,ijkl", two_mo_0, two_dm
        )
    else:
        one_mo = spinize(one_mo)
        two_mo = spinize(two_mo)
        # Build Fock operator
        Fk = np.copy(one_mo)
        Fk += np.einsum("piqj,ij->pq", antisymmetrize(two_mo), one_dm)
        one_mo_0 = Fk
        two_mo_0 = np.zeros_like(two_mo)
        energy = np.einsum("pq, pq", Fk, one_dm)    
    def ac(h_0, v_0, h, v, dm1, dm2, tol, nint=50):
        dv = v - v_0
        dh = h - h_0
        n = h.shape[0]
        # dh_pq * \gamma_pq
        linear = np.einsum("pq,pq", dh, dm1, optimize=True)
        # rdm terms for evaluating transition DM
        rdm_terms = dm2
        @np.vectorize
        def nonlinear(alpha):
            # Compute H^alpha
            h = alpha * dh
            h += h_0
            v = alpha * dv
            v += v_0
            # Antysymmetrize v_pqrs
            # Solve EOM equations
            lhs = brute_hherpa_lhs(h, v, dm1, dm2)
            rhs = dm2.reshape(n**2,n**2)
            w, c = solve_lowdin(lhs, rhs, tol=tol)
            _, c, _ = pickpositiveeig(w, c)
            # Compute transition RDMs
            # \gamma_m;pq = c_m;ji * < |p+q+ji| >
            rdms = np.einsum("mji,pqij->mpq", c.reshape(c.shape[0], n, n), rdm_terms)
            tv = np.zeros_like(dm2)
            for rdm in rdms:
                tv += np.einsum("pq,rs->pqrs", rdm, rdm, optimize=True)
            # Compute nonlinear energy term
            # dv_pqrs * {sum_{m}{\gamma_m;pq * \gamma_m;rs}}_pqrs
            return np.einsum("pqrs,pqrs", dv, tv, optimize=True)

        # Compute ERPA correlation energy
        return (
            linear
            + 0.5 * integrate(nonlinear, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.49e-04)[0]
        )
    dE = ac(one_mo_0, two_mo_0, one_mo, two_mo, one_dm, two_dm, tol)
    return energy + dE + nucnuc
