"""Esto es una copia (modificada) del script tdhf.py que tenia en la
carpeta: /home/gabrielasd/Documents/chapoteando/qmbasics/ci
El script original corresponde a 2018. """

from horton import *

import numpy as np
# from scipy.integrate import quad as integrate
# from scipy.integrate import quadrature as integrate
from scipy.integrate import fixed_quad as integrate

import pprint


def do_rhf(system, basis, nocc):

    # Make RHF function
    # Puse las siguientes 37 lineas de codigo dentro de
    # una funcion para hacer el script algo mas flexible.
    if isinstance(system, tuple) and (len(system) == 3):
        name, coords, nuc = system
        mol = IOData(title=name)
        # mol.title = name
        mol.coordinates = np.array(coords)
        mol.numbers = np.array(nuc)
    elif isinstance(system, IOData):
        mol = system


    # Shut up HORTON
    log.set_level(0)

    # SCF restricted HF
    obasis = get_gobasis(mol.coordinates, mol.numbers, basis)
    lf = DenseLinalgFactory(obasis.nbasis)
    S = obasis.compute_overlap(lf)
    T = obasis.compute_kinetic(lf)
    V = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    ee = obasis.compute_electron_repulsion(lf)

    exp_alpha = lf.create_expansion()
    guess_core_hamiltonian(S, T, V, exp_alpha)
    external = {"nn": compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        RTwoIndexTerm(T, "kin"),
        RDirectTerm(ee, "hartree"),
        RExchangeTerm(ee, "x_hf"),
        RTwoIndexTerm(V, "ne"),
    ]
    ham = REffHam(terms, external)
    occ_model = AufbauOccModel(nocc)
    scf_solver = PlainSCFSolver(1e-6)
    scf_solver(ham, lf, S, occ_model, exp_alpha)

    # Transform orbitals from AO to MO
    # TODO
    # Write a separate function that handles the AO to MO integral transformation
    # in ordr not to depend from HORTON for this, and in order for me to lear how
    # to do it.
    one = T.copy()
    one.iadd(V)
    two = ee
    (one_mo,), (two_mo,) = transform_integrals(one, two, "tensordot", exp_alpha)

    return (one_mo, two_mo, exp_alpha.energies)


def spinize_asymmetrize_twoint(dim, two_mo):
    twomo_phys = np.zeros((dim, dim, dim, dim))
    # print twomo_phys.shape
    # print range(twomo_phys.shape[0])
    for p in range(dim):
        for q in range(dim):
            for r in range(dim):
                for s in range(dim):
                    # All same spin
                    if (p // k == r // k == q // k == s // k == 0) or (
                        p // k == r // k == q // k == s // k == 1
                    ):
                        twomo_phys[p, q, r, s] = (
                            two_mo._array[p % k, q % k, r % k, s % k]
                            - two_mo._array[p % k, q % k, s % k, r % k]
                        )
                    elif (p // k == r // k == 0 and q // k == s // k == 1) or (
                        p // k == r // k == 1 and q // k == s // k == 0
                    ):
                        twomo_phys[p, q, r, s] = two_mo._array[
                            p % k, q % k, r % k, s % k
                        ]
                    elif (p // k == s // k == 0 and q // k == r // k == 1) or (
                        p // k == s // k == 1 and q // k == r // k == 0
                    ):
                        twomo_phys[p, q, r, s] = -two_mo._array[
                            p % k, q % k, s % k, r % k
                        ]
    return twomo_phys


def get_tdhf_matrix(nel, dim, emos, twomo_phys, alpha=1):
    # TDHF
    # A_ia,jb = F_ab \delta_ij - F_ij \delta_ab + <aj||ib>
    #         = [(F I)_aibj]^T(2143) - (F I)_iajb + [v_ajib]^T(3124)
    #         = (F I)_iajb - (F I)_iajb + v_iajb
    # B_ia,jb = <ab||ij>
    # Fock matrix is diagonal in the restricted MO basis
    # F_mn \delta_mn = F_mm = epsilon_m
    # A_ia,jb = (epsilon_a - epsilon_i) \delta_ab \delta_ij + <aj||ib>
    # In the definition of the CIS Hamiltonian dimensions below
    # I used "nel*2*virt" instead of the variable "dim" already defined so
    # that I don't forget where the dimensions of CIS came from.
    # TODO
    # Clean inecessary variables
    # -------Version de 2018------------
    # F = np.zeros((dim,dim))
    # F[:k,:k] = np.diag(exp_alpha.energies)
    # F[k:,k:] = np.diag(exp_alpha.energies)
    # Id = np.eye(dim)
    # virt = k - nocc
    # ----------------------------------
    # dim: nspins, k: nocc_alpha + nvir_alpha, nel: electrons
    F = np.zeros((dim, dim))
    k = dim // 2  # nocc_alpha + nvir_alpha
    nocc_alpha = nel // 2
    nvir_alpha = k - nocc_alpha
    F[:k, :k] = np.diag(emos)
    F[k:, k:] = np.diag(emos)
    Id = np.eye(dim)

    A = np.zeros((nel * 2 * nvir_alpha, nel * 2 * nvir_alpha))
    B = np.zeros((nel * 2 * nvir_alpha, nel * 2 * nvir_alpha))
    # In the definition of the matrices F, Id and twomo_phys, the matrix
    # elements between occupied and virtual orbitals of the same spin are grouped
    # together in sections of the matrix. It is so because of the ordering I assumed
    # for the spinorbitals in the wfn - all alpha spinorbitals first. Therefore, the
    # arrangement of the matrix elements differs from the one when all he occupied
    # orbitals go first (alphas and betas) and then all the virtuals, which is the ordering
    # I think Joshua Goings is using.
    # Then to make the loops for the evaluation of the CIS matrix elements work properly
    # I had to define a list of indexed molecular spinorbitals, and from those group into
    # separate lists the indexes of occupied and virtual spinorbitals. However, I need to
    # find a cleaner way to do this than with the appending finction used.
    mos = [m for m in range(dim)]
    occs = mos[:k][:nocc_alpha]
    occs.append(mos[k:][:nocc_alpha][0])
    virts = mos[:k][nocc_alpha:]
    virts.append(mos[k:][nocc_alpha:][0])
    # print occs
    # print virts

    for i, io in enumerate(occs):
        for a, av in enumerate(virts):
            I = i * (2 * nvir_alpha) + a
            for j, jo in enumerate(occs):
                for b, bv in enumerate(virts):
                    J = j * (2 * nvir_alpha) + b
                    A[I, J] = (
                        F[av, bv] * Id[io, jo]
                        - F[io, jo] * Id[av, bv]
                        + alpha*twomo_phys[av, jo, io, bv]
                    )
                    B[I, J] = alpha*twomo_phys[av, bv, io, jo]
    return A, B


def ecorr_tahir(w,A):
    """Equation (7) in DOI:10.1103/PhysRevB.99.195149"""
    w[w < 0] = 0.0
    ecorr = sum(w) - np.trace(A)
    return 0.25 * ecorr

def ecorr_pernal(nel, nspin, emos, two_int_spin,nint=50):
    # Equation (82) in DOI: 10.1002/qua.25462
    # E_c,HF = 2 int^{1}_0 dalpha {\sum_ijab \sum_v (Y_via-X_via)(Y_vjb-X_vjb)<ij|ab>}
    #        - int^{1}_0 da I_ij I_ab <ij|ab>
    # TODO: prove derivation from (77)
    nocc_alpha = nel//2
    nmo = nspin // 2 # nocc_alpha + nvir_alpha = alpha_spins
    mos = list(range(nmo))
    occs = mos[:nocc_alpha]
    occs += [occ + nmo for occ in occs]
    virts = mos[nocc_alpha:]
    virts += [vir + nmo for vir in virts]

    term2 = 0
    # FIXME: It is strange that all these terms are 0.
    # I may be selecting the wrong ones.
    for io in occs:
        for av in virts:
            term2 += two_int_spin[io,io,av,av]
    # print "2do termino", term2
    @np.vectorize
    def integrand(a):
        # print "Value of alpha", a
        A, B = get_tdhf_matrix(nel, dim, emos, two_int_spin, alpha=a)
        # Solve TDHF matrix equation
        M = np.bmat([[A, B], [-B, -A]])
        _, CTD = np.linalg.eig(M)
        CTD = np.real(CTD.T)
        # Compute TDM
        occxvirt = A.shape[0]
        tv = np.zeros((nel,nel,nspin-nel,nspin-nel))
        for v, C_v in enumerate(CTD):
            X_via = C_v[0,:occxvirt]
            Y_via = C_v[0,occxvirt:]
            YX = (Y_via-X_via).reshape(nel,dim-nel)
            tv += np.einsum('ia,jb->ijab', YX, YX)
        # Build integrand
        # FIXME: I'm not sure either if I'm building correctly
        # this term. These should be the [occ,occ,virt,virt] segments
        # from the complete <pq||rs> matrix.
        two_int_ijab = np.zeros((nel,nel,nspin-nel,nspin-nel))
        for i, io in enumerate(occs):
            for j, jo in enumerate(occs):
                for a, av in enumerate(virts):
                    for b, bv in enumerate(virts):
                       two_int_ijab[i,j,a,b] = two_int_spin[io, jo, av, bv]
        term1 = np.einsum('ijab,ijab', tv, two_int_ijab, optimize=True)
        return term1 - term2

    # Evaluate integral over alpha
    # FIXME: There may be a 0.5 or 0.25 factor (or other) that I'm not getting right
    # in this equation, as Pernal uses two-electron integrals formated as <pq|rs> while
    # I use <pq||rs>.
    #------Integration algorithms----------
    # QAGS from the Fortran library QUADPACK
    # return integrate(integrand, 0, 1, limit=nint, epsabs=1.49e-04, epsrel=1.49e-04)
    # fixed-tolerance Gaussian quadrature.
    # return integrate(integrand, 0, 1, maxiter=nint, tol=1.49e-04, rtol=1.49e-04)
    # fixed-order Gaussian quadrature
    return integrate(integrand, 0, 1, n=nint)


if __name__ == "__main__":
    # Molecule definition
    # TODO
    # All this initial part (molecular definition and SCf calculation)
    # is independent form the CIS calculation. I have to modify this function
    # so that it only needs to receives the information it required to run the CIS
    # algorithm.
    # # -----------HeH----------------
    # basis = "sto-3g"
    # name = "HeH+" #"H2O"
    # nuc = [2, 1]
    # nel = 2 #10
    # bond_lenght = 0.9295 * angstrom
    # coords = [[0.0, 0.0, 0.0], [0.0, 0.0, bond_lenght]]
    # heh = (name, coords, nuc)
    #-----------H2----------------
    xyz = 'smallmol/h2.xyz'
    nel = 2
    mol = IOData.from_file(xyz)

    # Evalua RHF y obten integrales y energias de MOs
    basis = "sto-6g"
    nocc = nel // 2
    one_int, two_int, emos = do_rhf(mol, basis, nocc)
    # print two_mo._array.shape
    # print
    # print obasis.nbasis
    # print exp_alpha.energies[nocc:]
    # print emos

    # Transform two-electron integrals to physisist notation (<pq||rs>)
    # TODO
    # Make this integral notation trasnformation a separate function in an external
    # file that gets called by cis.py
    # -------Version de 2018------------
    # dim = 2 * obasis.nbasis
    # k = obasis.nbasis
    # ----------------------------------
    # Crea <pq||rs> "transformando" las integrales bielectronicas
    # de HORTON de la base de MOs a espin orbitales y antisimetrizandolas
    k = one_int._array.shape[0]
    dim = 2 * k
    two_int = spinize_asymmetrize_twoint(dim, two_int)
    A, B = get_tdhf_matrix(nel, dim, emos, two_int)
    # print A
    # # Solve CIS matrix equation
    # ECIS, CCIS = np.linalg.eig(A)
    # Solve CIS matrix equation
    M = np.bmat([[A, B], [-B, -A]])
    ETD, CTD = np.linalg.eig(M)
    # print "E(TDHF) = ", np.amax(ETD)
    # print sorted(ETD)
    # print "E(CIS) = ", np.amax(ECIS)
    # print "E(CIS) = ", np.amin(ECIS[:18])
    # print sorted(ECIS)
    # my_printer = pprint.PrettyPrinter(width=7)
    # my_printer.pprint(B)
    # ecorr_cis = ecorr_tahir(ECIS,A)
    # print "Ecorr(CIS) = ", ecorr_cis
    ecorr_phrpa = ecorr_tahir(ETD,A)
    print "Ecorr(phrpa) = ", ecorr_phrpa


    #------------ K. Pernal--------------
    # FIXME: It is not returning the right value and
    # it is not because of the integration algorithm used.
    E_cHF = ecorr_pernal(nel, dim, emos, two_int, nint=50)
    print "Ecorr(pernal) = ", E_cHF[0]



