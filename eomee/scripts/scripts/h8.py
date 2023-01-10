import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
import numpy as np


def make_geom(atnames, dirname, dists, aunits): 
    """Make a geometry file H8 molecule, distorted octagon.

    MolFactory distorted_octagon: 'oc'.
    """
    from src.scripts.generate_xyz import MolFactory
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)
    for b in dists:
        bintra = 2.0
        geom = MolFactory.from_shape(atnames, 'oc', [bintra, b], bohr=aunits)
        geom.to_xyz(f'{dirname}_{b:.2f}.xyz')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['H',] * 8
    prefix = 'h8'   
    bond = [0., 0.0001, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.5, 1.0]
    # bond = [1.0]
    aunits = True

    # Make geometry
    make_geom(atms, prefix, bond, aunits)
