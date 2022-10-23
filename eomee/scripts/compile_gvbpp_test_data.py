
import sys  
sys.path.insert(0, '../')

import os

from glob import glob

import numpy as np
np.set_printoptions(precision=4)

from src.scripts.tools_erpa import from_spins

from eomee.tools import spinize


def load_files(_dir):
    cwd = os.getcwd()
    os.chdir(_dir)
    _mol = _dir.split('/')[1]
    basis = _dir.split('/')[-1].split('_')[-1]
    NAME = f'{_mol}_gvbpp_{basis}'
    ham = np.load(f'{NAME}.ham.npz')
    rdms = np.load(f'{NAME}.dms.npz')
    geminals = np.load(f'{NAME}.geminals.npy')
    os.chdir(cwd)
    
    return ham, rdms, geminals


def save_files(_dir, fname, ngems):
    cwd = os.getcwd()
    os.chdir(_dir)
    _mol = _dir.split('/')[1]
    basis = _dir.split('/')[-1].split('_')[-1]
    NAME = f'{fname}'

    os.system(f'cp {NAME}.ham.npz {_mol}_gvbpp_{basis}.ham.npz')
    data = np.load(f"{NAME}.gvb.npz")
    dm1aa, _ = data['rdm1']
    dm2aaaa, dm2abab = data['rdm2']
    np.savez(f'{_mol}_gvbpp_{basis}.dms.npz', rdm1=[dm1aa], rdm2=[dm2aaaa,dm2abab])

    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    nbasis = one_mo.shape[0]

    index_m = np.loadtxt(f"gvb_geminals.dat", dtype=int)
    assert index_m.shape[0] == nbasis
    assert index_m[-1, 1] == ngems
    index_m -= 1
    gem_mtrix = np.zeros((nbasis, ngems))
    for n,g in index_m:
        gem_mtrix[n, g] = 1.0
    np.save(f'{_mol}_gvbpp_{basis}.geminals.npy', gem_mtrix)
    os.chdir(cwd)


def read_data_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    while lines[-1] == "":
        del lines[-1]
    return lines


def get_gammcor_hamiltonian(integ_files):
    assert len(integ_files) == 2
    el1_file = integ_files[0]
    el2_file = integ_files[1]
    assert el1_file.strip('.dat') == 'gvb_1el_integ'
    assert el2_file.strip('.dat') == 'gvb_2el_integ'

    print("Loading electron integrals...")
    el1_lines = read_data_file(el1_file)
    el2_lines = read_data_file(el2_file)

    nbasis = int(el1_lines[-1].split()[0])
    oneint = np.zeros((nbasis, nbasis))
    twoint = np.zeros((nbasis, nbasis, nbasis, nbasis))

    for p in range(nbasis):
        for q in range(nbasis):
            indx = p * nbasis + q
            oneint[p, q] = float(el1_lines[indx].split()[-1])
    
    for p in range(nbasis):
        for q in range(nbasis):
            for r in range(nbasis):
                for s in range(nbasis):
                    indx = p * nbasis * nbasis * nbasis + q * nbasis * nbasis + r * nbasis + s
                    twoint[p, q, r, s] = float(el2_lines[indx].split()[-1])
    # Loaded integrals are in chemist notation. Transform to physicist
    twoint = twoint.transpose(0, 2, 1, 3)
    print("DONE")
    
    return oneint, twoint


def fill_inter(two_mo0, two_mo, set_i, set_j, dm1):
    for p in set_i:
        for q in set_i:
            for r in set_j:
                # g_pqrs_aaaa
                two_mo0[p, q] += dm1[r,r]*two_mo[p,r,q,r]
                two_mo0[p, q] -= dm1[r,r]*two_mo[p,r,r,q]
                # g_pqrs_abab
                two_mo0[p, q] += dm1[r,r]*two_mo[p,r,q,r]
    return two_mo0


def fill_intra(one_mo0, two_mo0, one_mo, two_mo, set_i):
    for p in set_i:
        for q in set_i:
            one_mo0[p, q] = one_mo[p, q]
            for r in set_i:
                for s in set_i:
                    two_mo0[p, q, r, s] = two_mo[p, q, r, s]
    return one_mo0, two_mo0


def get_geminal_i_hamiltonian(gemi, one_mo, two_mo, gem_matrix, dm1a):
    n_gems = gem_matrix.shape[1]    
    one_mo0 = np.zeros_like(one_mo)
    two_mo0 = np.zeros_like(two_mo)
    two_mo_inter = np.zeros_like(one_mo)

    gem_i = np.nonzero(gem_matrix.T[gemi])[0]
    one_mo_0, two_mo_0 = fill_intra(one_mo0, two_mo0, one_mo, two_mo, gem_i)
    for j in range(n_gems):
        if j != gemi:
            gem_j = np.nonzero(gem_matrix.T[j])[0]
            two_mo_inter = fill_inter(two_mo_inter, two_mo, gem_i, gem_j, dm1a)   

    return one_mo_0 + two_mo_inter, two_mo_0


def make_gvbpp_hamiltonian(one_mo, two_mo, gem_matrix, dm1a):
    k = one_mo.shape[0]
    assert k == gem_matrix.shape[0]
    n_gems = gem_matrix.shape[1]    

    one_mo0 = np.zeros_like(one_mo)
    two_mo0 = np.zeros_like(two_mo)
    two_mo_inter = np.zeros_like(one_mo)
    for i in range(n_gems):
        gem_i = np.nonzero(gem_matrix.T[i])[0]
        one_mo_0, two_mo_0 = fill_intra(one_mo0, two_mo0, one_mo, two_mo, gem_i)
        for j in range(n_gems):
            if j != i:
                gem_j = np.nonzero(gem_matrix.T[j])[0]
                two_mo_inter = fill_inter(two_mo_inter, two_mo, gem_i, gem_j, dm1a)    

    return one_mo_0, two_mo_0, two_mo_inter


def load_geminals_coeffs(fname):
    with open(fname, 'r') as f:
        content = f.read()
    last = content.split('Printout of final geminals')[-1]
    with open('temp.out', 'w') as f:
        f.write(last)
    content = os.popen(f"grep 'Orbital,\ geminal' temp.out").read()
    os.system('rm temp.out')
    lines = content.split('\n')
    while lines[-1] == '':
        lines = lines[:-1]
    coeffs = [float(line.split()[-2]) for line in lines]
    return coeffs


def load_geminals_matrix(fname):
    index_m = np.loadtxt(fname, dtype=int)
    nbasis = index_m.shape[0]
    ngems = index_m[nbasis-1, 1]
    index_m -= 1
    gem_mtrix = np.zeros((nbasis, ngems))
    for n,g in index_m:
        gem_mtrix[n, g] = 1.0
    
    return nbasis, ngems, gem_mtrix


def make_gvbpp_rdms(gem_matrix, coeffs):
    def fill_aaaa(gamma, set_i, set_j, coeff):
        # Fill inter geminal terms
        for p in set_i:
            for q in set_j:
                gamma[p,q,p,q] = coeff[p]**2 * coeff[q]**2
                gamma[q,p,q,p] = gamma[p,q,p,q]
                gamma[p,q,q,p] = -gamma[p,q,p,q]
                gamma[q,p,p,q] = -gamma[p,q,p,q]
        return gamma
    
    def fill_abab_intra(gamma, set_i, coeff):
        # Fill intra geminal terms
        for p in set_i:
            gamma[p,p,p,p] = coeff[p] * coeff[p]
            for q in set_i:
                if p != q:
                    gamma[p,p,q,q] = coeff[p] * coeff[q]
        return gamma
    
    def fill_abab_inter(gamma, set_i, set_j, coeff):
        # Fill inter geminal terms
        for p in set_i:
            for q in set_j:
                gamma[p,q,p,q] = coeff[p]**2 * coeff[q]**2   # 
                gamma[q,p,q,p] = gamma[p,q,p,q]
        return gamma
    
    k = len(coeffs)
    assert k == gem_matrix.shape[0]
    n_gems = gem_matrix.shape[1]

    dm1_a = np.diag(np.square(coeffs))

    dm2_aa = np.zeros((k, k, k, k))
    dm2_ab = np.zeros((k, k, k, k))
    for i in range(n_gems):
        gem_i = np.nonzero(gem_matrix.T[i])[0]
        dm2_ab = fill_abab_intra(dm2_ab, gem_i, coeffs)
        for j in range(i+1, n_gems):            
            gem_j = np.nonzero(gem_matrix.T[j])[0]
            dm2_aa = fill_aaaa(dm2_aa, gem_i, gem_j, coeffs)
            dm2_ab = fill_abab_inter(dm2_ab, gem_i, gem_j, coeffs)
    return dm1_a, dm2_aa, dm2_ab


def compile_raw_gammcor_data(nelec, molname, fp_data, fname=None):
    cwd = os.getcwd()
    os.chdir(fp_data)
    content = os.popen("grep 'SEVERE' " +  f"{molname}.out").read()
    content = os.popen(f"grep 'Final\ GVB-PP\ energy' {molname}.out").read()
    egvb = float(content.split()[-1])        
    
    # check GammCor output: normal termination and get energies
    content = os.popen(f"grep 'EGVB+ENuc' gammcor.out").read()
    etot = float(content.split()[-1])
    ecorr = float(content.split()[-2])
    content = os.popen(f"grep 'Nuclear\ repulsion' gammcor.out").read()
    nuc_rep = float(content.split()[-1])
    
    # Load RDMS and integrals files and build numpy arrays
    data = glob('*.dat')
    if len(data) != 3:
        raise ValueError(f"Missing data files in {fp_data}")
    el_files = ['gvb_1el_integ.dat', 'gvb_2el_integ.dat']
    one_int, two_int = get_gammcor_hamiltonian(el_files)
    nb, ngems, gem_matrix = load_geminals_matrix('gvb_geminals.dat')
    assert nb == one_int.shape[0]
    # WARNING: Does not work for inactive orbitals
    if 2 * ngems == nelec:
        coeffs = load_geminals_coeffs(f'{molname}.out')
    else:
        npairs = ngems - 1
        ncoeffs = 2*npairs
        coeffs = np.zeros(nb)
        coeffs[:ncoeffs] = load_geminals_coeffs(f'{molname}.out')
    dm1, dm2aa, dm2ab = make_gvbpp_rdms(gem_matrix, coeffs)
    rdm1 = [dm1, dm1]
    rdm2 = [dm2aa, dm2ab]    

    np.savez(f"{fname}.ham.npz", onemo=one_int, twomo=two_int, nuc=nuc_rep)
    np.savez(f"{fname}.gvb.npz", energy=egvb, coeff=coeffs, dets=None, rdm1=rdm1, rdm2=rdm2)
    np.savez(f"{fname}.gammcor.npz", energy=etot, ecorr=ecorr, abserr=None)
    os.chdir(cwd)


def test_dms(ham, rdms, nel, answer):
    nel = float(nel)
    h = spinize(ham["onemo"])
    v = spinize(ham["twomo"])
    dm1a = rdms["rdm1"][0]
    dm2aa, dm2ab = rdms["rdm2"]
    rdm1 = from_spins([dm1a, dm1a])
    # rdm2 = from_spins([dm2aa, dm2ab, dm2ab.transpose((1,0,3,2)), dm2aa])
    
    n = ham["onemo"].shape[0]
    assert dm1a.shape[0] == n
    assert dm2aa.shape[0] == n
    assert dm2ab.shape[0] == n    

    k = 2 * n
    y = np.zeros((k, k, k, k))
    y[:n, :n, :n, :n] = dm2aa
    y[:n, n:, :n, n:] = dm2ab
    y[n:, :n, n:, :n] = dm2ab
    y[n:, n:, n:, n:] = dm2aa
    y[:n, n:, n:, :n] = -dm2ab.transpose((0,1,3,2))
    y[n:, :n, :n, n:] = -dm2ab.transpose((1,0,2,3))

    assert np.allclose(dm1a, dm1a.T)
    assert np.allclose(dm2aa, dm2aa.transpose(1, 0, 3, 2))
    assert np.allclose(dm2aa, -dm2aa.transpose(1, 0, 2, 3))

    n_up = np.trace(dm1a)
    npats = 2 * n_up
    assert np.allclose(n_up, nel // 2)
    new_dm1 = np.einsum("piqi->pq", y) / (npats - 1)
    assert np.allclose(dm1a, new_dm1[:n, :n])
    pairs = np.einsum("pipi", y)
    assert np.allclose(pairs/2, npats*(npats-1)/2)

    energy_el_gvb = np.einsum('ij,ji', h, rdm1) + 0.5 * np.einsum('ijkl,ijkl', v, y)
    assert np.allclose(energy_el_gvb, answer)


def test_hamiltonian_terms(ham, rdms, gem_mtrix, one_el, intra_g, intrer_g):
    Egeminals_el = intra_g + intrer_g
    Egeminals_tot_el = one_el + Egeminals_el
    one_mo = ham["onemo"]
    two_mo = ham["twomo"]
    # nuc = ham["nuc"]
    dm1a = rdms["rdm1"][0]
    dm2aa, dm2ab = rdms["rdm2"]
    rdm1 = from_spins([dm1a, dm1a])
    rdm2 = from_spins([dm2aa, dm2ab, dm2ab.transpose((1,0,3,2)), dm2aa])
    
    one_mo_0, two_mo_0, two_mo_0inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_mtrix, dm1a)
    h = spinize(one_mo_0) 
    v_intra = spinize(two_mo_0)
    v_inter = spinize(two_mo_0inter)
    energy_one_body = np.einsum('ij,ji', h, rdm1)
    energy_two_body_intra = 0.5 * np.einsum('ijkl,ijkl', v_intra, rdm2)
    energy_two_body_inter = np.einsum('ij,ji', v_inter, rdm1)
    gvb_elel = energy_two_body_intra + energy_two_body_inter - 0.5 * energy_two_body_inter
    energy = energy_one_body + gvb_elel

    assert np.allclose(energy_one_body, one_el)
    assert np.allclose(energy_two_body_intra, intra_g)
    assert np.allclose(0.5*energy_two_body_inter, intrer_g)
    assert np.allclose(gvb_elel, Egeminals_el)
    assert np.allclose(energy, Egeminals_tot_el)


def test_gvbpp_e0(ham, rdms, gem_mtrix, geminals_e, answer):
    # ham, rdms, gem_mtrix = load_files(dir)
    one_mo = ham["onemo"]
    two_mo = ham["twomo"]
    # # nuc = ham["nuc"]
    dm1a = rdms["rdm1"][0]
    dm2aa, dm2ab = rdms["rdm2"]
    rdm1 = from_spins([dm1a, dm1a])
    rdm2 = from_spins([dm2aa, dm2ab, dm2ab.transpose((1,0,3,2)), dm2aa])
    
    # Geminal i energy
    for indx, val in enumerate(geminals_e):
        one_mo_i, two_mo_i = get_geminal_i_hamiltonian(indx, one_mo, two_mo, gem_mtrix, dm1a)
        h = spinize(one_mo_i) 
        v = spinize(two_mo_i)
        energy_one_body = np.einsum('ij,ji', h, rdm1)
        energy_two_body = 0.5 * np.einsum('ijkl,ijkl', v, rdm2)
        energy1 = energy_one_body + energy_two_body
        assert np.allclose(energy1, val)
    
    # E0 = E_g0 + E_g1
    one_int_0, two_int_0, two_int_0_inter = make_gvbpp_hamiltonian(one_mo, two_mo, gem_mtrix, dm1a) 
    h0 = spinize(one_int_0) + spinize(two_int_0_inter)
    v0 = spinize(two_int_0)
    energy_one_body = np.einsum('ij,ji', h0, rdm1)
    energy_two_body = 0.5 * np.einsum('ijkl,ijkl', v0, rdm2)
    energy = energy_one_body + energy_two_body
    print(energy, sum(geminals_e))
    print(energy_one_body, energy_two_body)
    assert np.allclose(energy, sum(geminals_e))
    # E_GVB = E0 - 0.5 v_pqrs_inter
    v_inter = spinize(two_int_0_inter)
    energy_el_gvb = energy - 0.5 * np.einsum('ij,ji', v_inter, rdm1)    
    assert np.allclose(energy_el_gvb, answer)


# geom = 'h2o_1.00'
# mol = geom.split('_')[0]
# basis = 'sto6g'
# f_name = f'h2o_q000_m01_k00_sp_gvbpp_{basis}'
# folder = f'gammcor/{geom}/{f_name}'
# ngems = 6


def compile_h2_631g():
    geom = 'h2_0.70'
    mol = geom.split('_')[0]
    basis = '631g'
    f_name = f'h2_q000_m01_k00_sp_gvbpp_{basis}'
    folder = f'gammcor/{geom}/{f_name}'
    nelec = 2
    ngems = 2

    compile_raw_gammcor_data(nelec, mol, folder, fname=f_name)
    save_files(folder, f_name, ngems)

    E_one_el = -2.53479735
    E_intra_g = 0.63443009
    E_intrer_g = 0.00000000
    E_el = -1.900367259303
    E_geminals = [-1.900367259303]
    hamilt, dms, geminals = load_files(folder)
    test_dms(hamilt, dms, nelec, E_el)
    test_gvbpp_e0(hamilt, dms, geminals, E_geminals, E_el)
    test_hamiltonian_terms(hamilt, dms, geminals, E_one_el, E_intra_g, E_intrer_g)


def compile_h2o_sto6g():
    geom = 'h2o_1.00'
    mol = geom.split('_')[0]
    basis = '631g'
    f_name = f'h2o_q000_m01_k00_sp_gvbpp_{basis}'
    folder = f'gammcor/{geom}/{f_name}'
    nelec = 10
    ngems = 6

    compile_raw_gammcor_data(nelec, mol, folder, fname=f_name)
    save_files(folder, f_name, ngems)

    E_one_el = -122.18362661
    E_intra_g = 7.94911566
    E_intrer_g = 30.20656255 -0.82042349
    E_el = -84.84837189
    E_geminals = [-45.896106956249, -2.364978315966, -2.364988478617, -2.418131034466, -2.418028050155]
    hamilt, dms, geminals = load_files(folder)
    test_dms(hamilt, dms, nelec, E_el)
    test_hamiltonian_terms(hamilt, dms, geminals, E_one_el, E_intra_g, E_intrer_g)
    test_gvbpp_e0(hamilt, dms, geminals, E_geminals, E_el)
    


# compile_h2_631g()
compile_h2o_sto6g()
