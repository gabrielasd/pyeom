import sys  
sys.path.insert(0, '../numdata/work_data/')

import os
import numpy as np


def make_geom(atnames, dirname, dists, aunits): 
    """Make a geometry file for diatomic linear molecule LiH.
    MolFactory line: 'ln'.
    """
    from src.scripts.generate_xyz import MolFactory       
    os.system(f'mkdir {dirname}')
    os.chdir(prefix)
    for b in dists:
        geom = MolFactory.from_shape(atnames, 'ln', [b], bohr=aunits)
        geom.to_xyz(f'{dirname}_{b:.2f}.xyz')
    os.chdir('..')


if __name__ == "__main__":
    atms = ['F', 'F']
    prefix = 'f2'   
    # bond = np.arange(0.4, 5.1, 0.1)
    bond = [2.134528652, 2.534752774, 2.668160815, 2.801568856, 2.934976897, 3.068384937,
        3.335201019, 4.002241223, 4.669281426, 5.33632163, 6.670402038, 8.004482445]
    aunits = True

    # Make geometry
    make_geom(atms, prefix, bond, aunits)


