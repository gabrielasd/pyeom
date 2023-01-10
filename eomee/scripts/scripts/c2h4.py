import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
from glob import glob

from scipy import constants as const
ANGSTROM = const.physical_constants['Bohr radius'][0] / const.angstrom # Bohr radius in Angstrom
BOHR = 1
b2a = ANGSTROM / BOHR


def c2h4_mol(ang1, ang2):
    RCC=1.336376
    RCH=1.085103695
    HCC=121.6944
    if ang2 is None:
        ang2=180+ang1
    geometry="""
    C
    C  1 RCC
    H  1 RCH 2 HCC
    H  1 RCH 2 HCC 3 180.
    H3 2 RCH 1 HCC 3 Angle1
    H4 2 RCH 1 HCC 3 Angle2
    """
    geometry=geometry.replace("RCC", str(RCC))
    geometry=geometry.replace("RCH", str(RCH))
    geometry=geometry.replace("HCC", str(HCC))
    geometry=geometry.replace("Angle1", str(ang1))
    geometry=geometry.replace("Angle2", str(ang2))
    return geometry


def make_geom(dirname, angles): 
    """Make a z-mat geometry file for C2H2 molecule.""" 
    if not os.path.exists(dirname):
        os.makedirs(dirname)     
    # os.system(f'mkdir {dirname}')
    os.chdir(dirname)
    for deg in angles:
        geom = c2h4_mol(deg, None)
        f=open(f'{dirname}_{deg:.1f}.zmat',"w")
        f.write(geom)
        f.close()
    os.chdir('..')


def zmat2xyz(dname):
    from pyscf import gto
    os.chdir(dname)
    subfolders = glob(f'{dname}*')

    def _to_xyz(fname, natoms, atoms, coords):        
        with open(fname, 'w') as fout:
            fout.write(f'{natoms}\n')
            fout.write(f'\n')
            for atom, xyz in zip(atoms, coords):
                x, y, z = xyz * b2a
                fout.write(f'{atom} {x:.5f} {y:.5f} {z:.5f}\n')

    for folder in subfolders:
        os.chdir(folder)
        # Build PySCF molecule
        fname = f'{folder}.zmat'
        geom = open(fname).read()
        mol = gto.Mole()
        mol.atom = geom
        mol.charge = 0
        mol.spin = 0
        mol.unit = 'Angstrom'
        mol.build()
        coords = mol.atom_coords()  # in Bohr
        natoms = len(coords)
        atoms = mol.elements
        fname = fname.replace("zmat", 'xyz')
        _to_xyz(fname, natoms, atoms, coords)
        os.chdir('..')
    os.chdir('..')


def make_mol_dirs(dname):
    os.chdir(dname)
    molfiles = glob(f'{dname}*.zmat')
    # molfiles = sorted(mols, key=lambda job: float(job.split('_')[-1].strip('.mol'))) #sort by index at file end

    for mol in molfiles:
        # Make molecule folder
        fname =  mol.strip('.zmat')
        if not os.path.exists(fname):
            os.makedirs(fname)        
        # move geometry file    
        os.system(f'mv {mol} {fname}/.')
    os.chdir('..')


if __name__ == "__main__":
    prefix = 'c2h4'   
    Angl=[0.001, 10., 15., 20., 30, 45., 60., 75., 80., 82., 83., 84., 86., 88., 89., 89.999]
    # Angl=[0.001]
    aunits = False

    # # Make geometry
    make_geom(prefix, Angl)
    make_mol_dirs(prefix)

    zmat2xyz(prefix)
