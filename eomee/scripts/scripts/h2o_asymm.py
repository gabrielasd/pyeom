import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
import numpy as np


def make_geom(atnames, dirname, fixb, dists, ang, aunits): 
    """Make a geometry file for H2O molecule.
    MolFactory isosceles: 'iso'.
    """
    from src.scripts.generate_xyz import MolFactory       
    os.system(f'mkdir {dirname}')
    os.chdir(prefix)
    for b in dists:
        geom = MolFactory.from_shape(atnames, 'ast', [fixb, b], angle=ang, bohr=aunits)
        geom.to_xyz(f'{dirname}_{b:.2f}.xyz')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['O', 'H', 'H']
    prefix = 'h2o_asymm'   
    bonds = np.arange(0.7, 4.4, 0.1)
    fixbond = 1.00
    angle = 104.5
    aunits = False

    # Make geometry
    make_geom(atms, prefix, fixbond, bonds, angle, aunits)
