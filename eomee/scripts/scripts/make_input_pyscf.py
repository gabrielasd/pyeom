import os
from glob import glob
from string import Template
import basis_set_exchange as bse


#############
# Edit lines bellow (file names, template path, folders)
#
q = 0
mult = 1
prefix = 'h8_chain' #'f2' #"c2h4"
method = "casscf" #"rhf" # 
subdir = f'0001_q000_m01_k00_sp_{method}'
basis =  'Ahlrichs-VDZ' #'6-31G' #
elements = ['H'] # ['F'] #['C', 'H']
basisname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")
subdir += f'_{basisname}'
# get basis set
if basis.lower().startswith("ahlrichs"):
    basis = basis.lower().replace("ahlrichs-", "ahlrichs ")
basis_set = bse.get_basis(
    basis, fmt="nwchem", elements=elements, header=False,
)
template_name = 'pyscf_casscf_guess.py' # 'pyscf_casscf.py' # 'pyscf_geom.py' # 
guessdir = '../{}'
params = { "output": subdir, "charge": q, "basis_set1": basis_set, "spinmult": mult, "nprocs": 2, "guess": ""}
TEMPLPATH = '../templates'
geomext = 'xyz' #'zmat'


# mols = glob(f'{prefix}*')
folders = glob(f'{prefix}*')
folders = sorted(folders, key=lambda job: float(job.split('_')[-1])) #sort by index at file end
#############


with open(f'{TEMPLPATH}/{template_name}', 'r') as f:
    content = f.read()
template = Template(content)

for inx, folder in enumerate(folders):
    if not os.path.isfile(f'{folder}/{folder}.{geomext}'):
        raise FileNotFoundError(f'Coordinate file {folder}/{folder}.{geomext} not found.')
    if geomext == 'zmat':
        with open(f'{folder}/{folder}.{geomext}', 'r') as f:
            geometry = f.read()
    else:
        geometry = f'../{folder}.xyz'
    
    params['title'] = f'{folder}/{subdir}'
    params['geometry'] = geometry
    if inx > 0:
        params['guess'] = f'../../{folders[inx-1]}'  # guess from previous job, assumes sorted folders
    string = template.substitute(params)
    # Make job folder
    fp_job = f'{folder}/{subdir}'
    if not os.path.exists(fp_job):
        os.makedirs(fp_job)   
    # write input file    
    with open(f'{folder}/{subdir}/{subdir}.py', 'w') as f:
        f.write(string)
