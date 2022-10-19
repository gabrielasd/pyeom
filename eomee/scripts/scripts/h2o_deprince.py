import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
import numpy as np


def make_geom(atnames, dirname, dists, ang, aunits): 
    """Make a geometry file for H2O molecule.
    MolFactory isosceles: 'iso'.
    """
    from src.scripts.generate_xyz import MolFactory       
    os.system(f'mkdir {dirname}')
    os.chdir(dirname)
    for b in dists:
        geom = MolFactory.from_shape(atnames, 'iso', [b], angle=ang, bohr=aunits)
        geom.to_xyz(f'{dirname}_{b:.2f}.xyz')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['O', 'H', 'H']
    prefix = 'h2o'   
    bond = np.arange(0.7, 4.4, 0.1)
    angle = 104.5
    aunits = False

    # Make geometry
    make_geom(atms, prefix, bond, angle, aunits)
