import os, sys
import re 
from glob import glob

from string import Template

from argparse import ArgumentParser

from sys import stderr

import numpy as np


# Initialize command line argument parser
parser = ArgumentParser()


# Specify positional arguments and options
parser.add_argument("-r", action="store_true", default=False, help="submit calculations")
parser.add_argument("-i", action="store_true", default=False, help="Write an job script")
parser.add_argument("-c", action="store_true", default=False, help="Process raw GammCor files")
parser.add_argument(
        "program", type=str, choices=["gammc"], help="Name of the program in the python script."
    )
parser.add_argument("fnames", nargs="*", help="Path to existing input file(s).")
parser.add_argument("-z", "--znucl", type=int, default=False, help="Nuclear charge.")
parser.add_argument("-q", "--charge", type=int, default=0, help="Charge.")
parser.add_argument("-t", "--temp", type=str, default=False, help="Template file.")
parser.add_argument("-ta", "--tact", type=float, default=0.998, help="Threshold for active space.")
parser.add_argument("-tsa", "--tselact", type=int, default=3, help="Threshold for selecteing active space.")


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


def get_gammcor_rdms(rdm_files):
    assert len(rdm_files) == 3

    dm1_file = rdm_files[0]
    dm2_aa_file, dm2_ab_file = rdm_files[1], rdm_files[2]
    assert dm1_file.strip('.dat') == 'gvb_1rdm'
    assert dm2_aa_file.split('.')[0] == 'gvb_2rdm_aaaa'
    assert dm2_ab_file.split('.')[0] == 'gvb_2rdm_abab'

    print("Loading density matrices...")
    dm1_lines = read_data_file(dm1_file)
    dm2_aa_lines = read_data_file(dm2_aa_file)
    dm2_ab_lines = read_data_file(dm2_ab_file)

    nbasis = int(dm1_lines[-1].split()[0])
    onedm = np.zeros((nbasis, nbasis))
    twodm_aa = np.zeros((nbasis, nbasis, nbasis, nbasis))
    twodm_ab = np.zeros((nbasis, nbasis, nbasis, nbasis))

    # 1-RDM
    for p in range(nbasis):
        indx = p * nbasis + p
        onedm[p, p] = float(dm1_lines[indx].split()[-1])
    # 2-RDM alpha alpha alpha alpha
    for p in range(nbasis):
        for q in range(nbasis):
            for r in range(nbasis):
                for s in range(nbasis):
                    indx = p * nbasis * nbasis * nbasis + q * nbasis * nbasis + r * nbasis + s
                    twodm_aa[p, q, r, s] = float(dm2_aa_lines[indx].split()[-1])
    # 2-RDM alpha beta alpha beta
    for p in range(nbasis):
        for q in range(nbasis):
            for r in range(nbasis):
                for s in range(nbasis):
                    indx = p * nbasis * nbasis * nbasis + q * nbasis * nbasis + r * nbasis + s
                    twodm_ab[p, q, r, s] = float(dm2_ab_lines[indx].split()[-1])
    print("DONE")

    return onedm, twodm_aa, twodm_ab


def load_geminals_matrix(fname):
    index_m = np.loadtxt(fname, dtype=int)
    nbasis = index_m.shape[0]
    ngems = index_m[nbasis-1, 1]
    index_m -= 1
    gem_mtrix = np.zeros((nbasis, ngems))
    for n,g in index_m:
        gem_mtrix[n, g] = 1.0
    
    return nbasis, ngems, gem_mtrix


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


def submit_serial_job(args):
    program = args.program.lower()
    if program != 'gammc':
        raise NotImplementedError(f"Input for program {program} not supported.")
    folders = sorted(args.fnames)

    for index, fp_job in enumerate(folders):
        print(f"Submit {index} {fp_job}")
        
        # get base directory
        base_database = os.getcwd()
        
        os.chdir(fp_job)
        if not os.path.isfile('input.inp'):
            raise FileNotFoundError(f'No input file found in {fp_job}')
        
        dalton_outs = glob('*.tar.gz')
        if len(dalton_outs) == 0:
            raise FileNotFoundError(f'No output .tar file found in {fp_job}')
        elif len(dalton_outs) > 1:
            raise ValueError(f'More than one output .tar file found in {fp_job}')
        os.system(f"tar -xf {dalton_outs[0]}")
        
        # run job
        os.system(f"gammcor > gammcor.out")
        os.system("sleep 3")

        os.chdir(base_database)


def process_gammcor(args):
    program = args.program.lower()
    if program != 'gammc':
        raise NotImplementedError(f"Verifications for program {program} not supported.")

    dalton_aborted = []    
    gammcor_aborted = []
    gammcor_missfiles = []

    # get base directory
    base_database = os.getcwd()
    folders = args.fnames
    for index, fp_job in enumerate(folders):
        print(f"Input {index} {fp_job}")
        # get folder & filename
        folder, subdir = os.path.split(fp_job)
        molname = subdir.split('_')[0].lower()     
        os.chdir(fp_job)

        # check Dalton finished normally
        content = os.popen("grep 'SEVERE' " +  f"{molname}.out").read()
        if content is not "":
            print(f"SEVERE ERROR in {fp_job}")
            dalton_aborted.append(fp_job)
            os.chdir(base_database)
            continue
        
        content = os.popen(f"grep 'Final\ GVB-PP\ energy' {molname}.out").read()
        print("The content is: ", content)
        if "NOT CONVERGED" in content:
            print(f"{fp_job}: GVB-PP did not converge")
            os.chdir(base_database)
            continue
        egvb = float(content.split()[-1])        
        
        # check GammCor output: normal termination and get energies
        content = os.popen(f"grep 'EGVB+ENuc' gammcor.out").read()
        if content is None:
            print(f"Abnormal GammCor termination in {fp_job}")
            gammcor_aborted.append(fp_job)
            os.chdir(base_database)
            continue
        etot = float(content.split()[-1])
        ecorr = float(content.split()[-2])
        content = os.popen(f"grep 'Nuclear\ repulsion' gammcor.out").read()
        nuc_rep = float(content.split()[-1])
        
        # Load RDMS and integrals files and build numpy arrays
        data = glob('*.dat')
        if len(data) != 6:
            print(f"Missing data files in {fp_job}")
            gammcor_missfiles.append(fp_job)
            os.chdir(base_database)
            continue
        el_files = ['gvb_1el_integ.dat', 'gvb_2el_integ.dat']
        one_int, two_int = get_gammcor_hamiltonian(el_files)
        # dm_files = ['gvb_1rdm.dat', 'gvb_2rdm_aaaa.dat', 'gvb_2rdm_abab.dat']
        # dm1, dm2aa, dm2ab = get_gammcor_rdms(dm_files)
        nb, ngems, gem_matrix = load_geminals_matrix('gvb_geminals.dat')
        assert nb == one_int.shape[0]
        # FIXME: figure rignt way to asign coeffs to geminals
        coeffs = np.zeros(nb)
        # ninactive = 0
        # stop = 2 * (ngems -1 - ninactive)
        # coeffs[:ninactive] = 1.0
        # coeffs[ninactive:stop] = load_geminals_coeffs(f'{molname}.out')
        stop = 2 * (ngems - 1 )
        coeffs[:stop] = load_geminals_coeffs(f'{molname}.out')
        ## coeffs = load_geminals_coeffs(f'{molname}.out')
        dm1, dm2aa, dm2ab = make_gvbpp_rdms(gem_matrix, coeffs)
        rdm1 = [dm1, dm1]
        rdm2 = [dm2aa, dm2ab]    

        np.savez(f"{subdir}.ham.npz", onemo=one_int, twomo=two_int, nuc=nuc_rep)
        np.savez(f"{subdir}.gvb.npz", energy=egvb, coeff=coeffs, dets=None, rdm1=rdm1, rdm2=rdm2)
        np.savez(f"{subdir}.gammcor.npz", energy=etot, ecorr=ecorr, abserr=None)
        
        os.chdir(base_database)
    
    print("-"*8)
    if len(dalton_aborted) > 0:
        print(f"{len(dalton_aborted)} Dalton jobs aborted")
        for job in dalton_aborted:
            print(job)
    elif len(gammcor_aborted) > 0:
        print("-"*8)
        print(f"{len(gammcor_aborted)} GammCor jobs aborted")
        for job in gammcor_aborted:
            print(job)
    elif len(gammcor_missfiles) > 0:
        print("-"*8)
        print(f"{len(gammcor_missfiles)} GammCor missing .dat files")
        for job in gammcor_missfiles:
            print(job)


def make_input(args):
    program = args.program.lower()
    if program != 'gammc':
        raise NotImplementedError(f"Input for program {program} not supported.")
    if not args.znucl % 2 == 0:
        raise ValueError("Only even number of electrons supported.")  
    params = {"znucl": args.znucl, "charge": args.charge, 'ThrAct': args.tact, 'ThrSelAct': args.tselact}

    with open(args.temp, 'r') as f:
        content = f.read()
    template = Template(content)    

    folders = args.fnames
    for index, fp_job in enumerate(folders):
        print(f"Input {index} {fp_job}")

        # get base directory
        base_database = os.getcwd()
        os.chdir(fp_job)

        string = template.substitute(params)
        # write input file
        with open('input.inp', 'w') as f:
            f.write(string)
        
        os.chdir(base_database)


if __name__ == "__main__":

    # Parse arguments
    args = parser.parse_args()

    # Exit if no commands are specified
    if not (args.r or args.i or args.c):
        print("No command specified. Exiting...", file=stderr)
        exit(1)

    # Run specified command(s)
    if args.r:
        submit_serial_job(args)
    if args.i:
        if not args.znucl:
            raise ValueError("The nuclear charge must be specified.")
        make_input(args)
    if args.c:
        process_gammcor(args)

    # Exit successfully
    exit(0)