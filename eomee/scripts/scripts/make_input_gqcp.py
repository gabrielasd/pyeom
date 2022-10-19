import os
from glob import glob
from string import Template


prefix = 'h2' #'h2o' # 'lih'  # 'h4_achain' #  'h2o_asymm' #'h4_chain' #
subdir = f'0001_q000_m01_k00_sp_oodoci'
q = 0
basis = 'sto-6g' #'6-31G' # 'sto-3g' #'cc-pvdz' #'3-21G'# 
basisname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")
subdir += f'_{basisname}'
template_name = 'gqcp_template.py'
params = { "output": subdir, "charge": q, "basis_set1": basis }


mols = glob(f'{prefix}*')
folders = sorted(mols, key=lambda job: float(job.split('_')[-1])) #sort by index at file end

with open(f'../templates/{template_name}', 'r') as f:
    content = f.read()
template = Template(content)

for folder in folders:
    if not os.path.isfile(f'{folder}/{folder}.xyz'):
        raise FileNotFoundError(f'Coordinate file {folder}/{folder}.xyz not found.')
    
    params['title'] = f'{folder}/{subdir}'
    params['geometry'] = f'../{folder}.xyz'    
    string = template.substitute(params)
    # Make job folder
    fp_job = f'{folder}/{subdir}'
    if not os.path.exists(fp_job):
        os.makedirs(fp_job)   
    # write input file    
    with open(f'{folder}/{subdir}/{subdir}.py', 'w') as f:
        f.write(string)
