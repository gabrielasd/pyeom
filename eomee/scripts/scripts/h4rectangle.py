import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
import numpy as np


def make_geom(atnames, dirname, dists, aunits): 
    """Make a geometry file 2(H2) rectangle molecule.
    Paldus P4 model.

    MolFactory rectangle: 'rc'.
    """
    from src.scripts.generate_xyz import MolFactory       
    os.system(f'mkdir {dirname}')
    os.chdir(dirname)
    for b in dists:
        bintra = 1.058 #2.0
        geom = MolFactory.from_shape(atnames, 'rc', [bintra, b], bohr=aunits)
        geom.to_xyz(f'{dirname}_{b:.2f}.xyz')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['H', 'H', 'H', 'H']
    prefix = 'h4_rec'   
    # bond = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0] 
    # bond = np.arange(1.0, 7.2, 0.2)
    aunits = False #True
    bond = [0.6, 0.8, 0.9, 0.95, 1.0, 1.03, 1.058, 1.08, 1.1, 1.15, 1.2, 1.4, 1.6, 1.8] # Angstrom

    # Make geometry
    make_geom(atms, prefix, bond, aunits)
