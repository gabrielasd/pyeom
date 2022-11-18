import sys
sys.path.insert(0, '../../numdata/work_data/')
from glob import glob
import os
import numpy as np
from string import Template


def make_geom(atnames, dirname, dists, basis, aunits, template_name): 
    """Make a geometry file 2(H2) asymmetric chain molecule.
    MolFactory chain_asymm: 'ch2'.
    """
    from src.scripts.generate_xyz import MolFactory

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)

    params = { "basis_set1": basis}
    with open(template_name, 'r') as f:
        content = f.read()
    template = Template(content)

    for b in dists:
        bintra = 2.0
        geom = MolFactory.from_shape(atnames, 'ch2', [bintra, b], bohr=aunits)
        geom.to_xyz(f'temp.xyz')

        with open('temp.xyz', 'r') as f:
            content = f.read()
        number, title, geometry = content.split('\n', 2)
        params['GEOM'] = geometry
        string = template.substitute(params)
        os.system('rm temp.xyz')

        # write input file    
        with open(f'{dirname}_{b:.2f}.mol', 'w') as f:
            f.write(string)
    os.chdir('..')


def make_mol_dirs(dirname):
    os.chdir(dirname)
    molfiles = glob(f'{dirname}*.mol')
    # molfiles = sorted(mols, key=lambda job: float(job.split('_')[-1].strip('.mol'))) #sort by index at file end

    for mol in molfiles:
        # Make molecule folder
        fname =  mol.strip('.mol')
        if not os.path.exists(fname):
            os.makedirs(fname)        
        # move geometry file    
        os.system(f'mv {mol} {fname}/.')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['H', 'H', 'H', 'H']
    prefix = 'h4_achain'   
    bonds = np.arange(1.0, 7.2, 0.2)
    aunits = True
    basisset = '3-21G'  
    template_path = f'../../templates/H4ach.mol'

    # Make geometry
    make_geom(atms, prefix, bonds, basisset, aunits, template_path)

    make_mol_dirs(prefix)
