import sys
from glob import glob
import os
import numpy as np
from string import Template


# def make_geom(dirname, dists, basis, template_name): 
#     """Make a .mol file."""
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)
#     os.chdir(dirname)

#     params = { "basis_set1": basis}
#     with open(template_name, 'r') as f:
#         content = f.read()
#     template = Template(content)

#     for bond in dists:
#         params['blength'] = f'{bond:.11f}'
#         string = template.substitute(params)

#         # write input file    
#         with open(f'{dirname}_{bond:.2f}.mol', 'w') as f:
#             f.write(string)
#     os.chdir('..')


def c2h4_mol(ang1, ang2):
    """Make a z-mat geometry for C2H2 molecule."""
    RCC=1.336376
    RCH=1.085103695
    HCC=121.6944
    if ang2 is None:
        ang2=180+ang1
    _params = { "RCC": RCC, "RCH": RCH, "HCC": HCC, "Angle1": ang1, "Angle2": ang2}
    return _params


def beh2_mol(RHH, RZ):
    """Make a cartesian geometry for BeH2 molecule."""
    _params = { "RHH": RHH, "RZ": RZ}
    return _params


def base_mol(bond):
    _params = { "blength": bond}
    return _params


def make_geom(dirname, geom_param, basis, template_name): 
    """Make a .mol file."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)

    with open(template_name, 'r') as f:
        content = f.read()
    template = Template(content)

    for x in geom_param:
        # params = base_mol(x)
        params = c2h4_mol(x, None)
        params['basis_set1'] = basis
        string = template.substitute(params)
        
        with open(f'{dirname}_{x:.2f}.mol', 'w') as f:
            f.write(string)
    os.chdir('..')


def make_geom2(dirname, basis, template_name, geom_param1, geom_param2): 
    """Make a .mol file."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)

    with open(template_name, 'r') as f:
        content = f.read()
    template = Template(content)

    for x,y in zip(geom_param1, geom_param2):
        params = beh2_mol(x, y)
        params['basis_set1'] = basis
        string = template.substitute(params)
        
        y = y / 0.529177
        with open(f'{dirname}_{y:.1f}.mol', 'w') as f:
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
    # basisset = '6-31G'  #'cc-pVDZ'
    # template_path = f'../../templates/h2o.mol' #f'../common/h2o.mol'  #
    # prefix = 'n2'   
    # bonds = np.arange(0.7, 4.4, 0.1)
    # basisset = 'cc-pVDZ'
    # template_path = f'../../templates/n2.mol'
    # prefix = 'h2'   
    # bonds = np.arange(0.4, 5.1, 0.1)
    # basisset = 'STO-6G' #'6-31G'
    # template_path = f'../../templates/H2.mol'
    # prefix = 'c2h4'   
    # Angl=[0.001, 10., 15., 20., 30, 45., 60., 75., 80., 82., 83., 84., 86., 88., 89., 89.999]
    # basisset = '6-31G' #'Ahlrichs-VDZ'
    # template_path = f'../../templates/c2h4.mol'
    # prefix = 'f2'
    # bonds = [2.134528652, 2.534752774, 2.668160815, 2.801568856, 2.934976897, 3.068384937,
    #     3.335201019, 4.002241223, 4.669281426, 5.33632163, 6.670402038, 8.004482445]
    # bonds = [i * 0.529177 for i in bonds]  # convert to Angstrom
    # basisset = 'Ahlrichs-VDZ'
    # template_path = f'../../templates/f2.mol'
    prefix = 'beh2'
    rhh = [2.54, 2.08, 1.62, 1.39, 1.275, 1.16, 0.93, 0.70, 0.70]
    rhh = [i * 0.52917721 for i in rhh]  # convert to Angstrom
    rz = [0.0, 1.0, 2.0, 2.5, 2.75, 3.0, 3.5, 4.0, 6.0]
    rz = [i * 0.52917721 for i in rz]  # convert to Angstrom
    basisset = 'cc-pVDZ'
    template_path = f'../../templates/beh2.mol'

    # Make geometry
    # make_geom(prefix, bonds, basisset, template_path)
    # make_geom(prefix, Angl, basisset, template_path)
    make_geom2(prefix, basisset, template_path, rhh, rz)

    make_mol_dirs(prefix)
