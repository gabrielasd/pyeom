import sys
import os
from glob import glob
import numpy as np


def dump_hamiltonian(subdir):
    # dump integrals in chemist notation and indexing counts from 1
    NAME = subdir.split('/')[-1]
    # fp_data = f'{subdir}/{NAME}'
    data = np.load(f"{NAME}.ham.npz")
    one_mo = data["onemo"]
    two_mo = data["twomo"]
    nbasis = one_mo.shape[0]
    
    with open('doci_1el_integ.dat', 'w') as f:
        for i in range(nbasis):
            for j in range(nbasis):
                f.write(f"{i+1} {j+1} {one_mo[i,j]}\n")
    
    with open('doci_2el_integ.dat', 'w') as f:
        for i in range(nbasis):
            for j in range(nbasis):
                for k in range(nbasis):
                    for l in range(nbasis):
                        f.write(f"{i+1} {j+1} {k+1} {l+1} {two_mo[i,j,k,l]}\n")

def dump_rdms(subdir):
    NAME = subdir.split('/')[-1]
    # fp_data = f'{subdir}/{NAME}'
    data = np.load(f"{NAME}.ci.npz")
    dm1a, _ = data["rdm1"]
    dm2aa, dm2ab, _, _ = data['rdm2'] # transform 2-RDMs to our notation <|p*q*sr|>=\Gamma_pqrs
    dm2aa = np.einsum("ijkl->ikjl", dm2aa)
    dm2ab = np.einsum("ijkl->ikjl", dm2ab)
    nbasis = dm1a.shape[0]
    assert dm2aa.shape == (nbasis, nbasis, nbasis, nbasis)
    assert dm2ab.shape == (nbasis, nbasis, nbasis, nbasis)

    with open('doci_1rdm.dat', 'w') as f:
        for i in range(nbasis):
            for j in range(nbasis):
                f.write(f"{i+1} {j+1} {dm1a[i,j]}\n")
    
    with open('doci_2rdm_aaaa.dat', 'w') as f:
        for i in range(nbasis):
            for j in range(nbasis):
                for k in range(nbasis):
                    for l in range(nbasis):
                        f.write(f"{i+1} {j+1} {k+1} {l+1} {dm2aa[i,j,k,l]}\n")
    
    with open('doci_2rdm_abab.dat', 'w') as f:
        for i in range(nbasis):
            for j in range(nbasis):
                for k in range(nbasis):
                    for l in range(nbasis):
                        f.write(f"{i+1} {j+1} {k+1} {l+1} {dm2ab[i,j,k,l]}\n")


def dump_energies(subdir):
    NAME = subdir.split('/')[-1]
    # fp_data = f'{subdir}/{NAME}'
    data = np.load(f"{NAME}.scf.npz")
    ehf = data["energy"]
    enuc = data["nuc"]
    data = np.load(f"{NAME}.ci.npz")
    e0 = data["energy"]
    with open('gqcp_energies.dat', 'w') as f:
        f.write(f"ERHF+ENUC, EDOCI+ENUC, ENUC, {ehf:.8f}, {e0:.8f}, {enuc:.8f}")


if __name__ == "__main__":
    # use example:
    # python3 dump_gqcp_2_gammcor.py h2o_1.00/0008_q000_m01_k00_sp_oodoci_sto3g
    fp = sys.argv[1]

    os.chdir(fp)
    dump_hamiltonian(fp)
    dump_rdms(fp)
    dump_energies(fp)