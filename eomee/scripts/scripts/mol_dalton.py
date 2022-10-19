import sys
from glob import glob
import os
import numpy as np
from string import Template


def make_geom(dirname, dists, basis, template_name): 
    """Make a .mol file.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        # os.system(f'mkdir {dirname}')
    os.chdir(dirname)

    params = { "basis_set1": basis}
    with open(template_name, 'r') as f:
        content = f.read()
    template = Template(content)

    for bond in dists:
        params['blength'] = f'{bond:.11f}'
        string = template.substitute(params)

        # write input file    
        with open(f'{dirname}_{bond:.2f}.mol', 'w') as f:
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
    # prefix = 'h2o'   
    # bonds = np.arange(0.7, 4.4, 0.1)
    # basisset = 'cc-pVDZ'
    # template_path = f'../../templates/h2o.mol'
    # prefix = 'n2'   
    # bonds = np.arange(0.7, 4.4, 0.1)
    # basisset = 'cc-pVDZ'
    # template_path = f'../../templates/n2.mol'
    prefix = 'h2'   
    bonds = np.arange(0.4, 5.1, 0.1)
    basisset = 'STO-6G' #'6-31G'
    template_path = f'../../templates/H2.mol'

    # Make geometry
    make_geom(prefix, bonds, basisset, template_path)

    make_mol_dirs(prefix)
