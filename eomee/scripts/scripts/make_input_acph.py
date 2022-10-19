from glob import glob
from string import Template


prefix = 'h2o'
subdir = '0008_q000_m01_k00_sp_doci_ccpvdz'
q = 0
m = 1
template_name = 'acph_template.py'
params = { "output": subdir, "charge": q, "spinmult": m }


mols = glob(f'{prefix}*')
folders = sorted(mols, key=lambda job: float(job.split('_')[-1])) #sort by index at file end

with open(f'../templates/{template_name}', 'r') as f:
    content = f.read()
template = Template(content)

for folder in folders[:1]:
    params['title'] = f'{folder}/{subdir}'
    string = template.substitute(params)
    with open(f'{folder}/{subdir}/{subdir}_ac.py', 'w') as f:
        f.write(string)
