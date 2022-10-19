import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
import numpy as np


def make_geom(atnames, dirname, dists, aunits): 
    """Make a geometry file 2(H2) chain molecule.
    MolFactory chain: 'ch'.
    """
    from src.scripts.generate_xyz import MolFactory       
    os.system(f'mkdir {dirname}')
    os.chdir(prefix)
    for b in dists:
        geom = MolFactory.from_shape(atnames, 'ch', [b], bohr=aunits)
        geom.to_xyz(f'{dirname}_{b:.2f}.xyz')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['H', 'H', 'H', 'H']
    prefix = 'h4_chain'   
    bond = np.arange(0.5, 5.1, 0.1)
    aunits = False

    # Make geometry
    make_geom(atms, prefix, bond, aunits)
