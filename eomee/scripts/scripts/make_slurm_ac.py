from glob import glob
from string import Template


prefix = 'h2o'
subdir = '0008_q000_m01_k00_sp_doci_ccpvdz'
ddhhmm = '00-05:00'
cpus = 4
mem = '20G'
account = 'def-ayers'
template_name = 'slurm_ac.sh'
params = { "name": subdir, "time": ddhhmm, "memory": mem, "n_proc": cpus, "account": account }


mols = glob(f'{prefix}*')
folders = sorted(mols, key=lambda job: float(job.split('_')[-1])) #sort by index at file end

with open(f'../templates/{template_name}', 'r') as f:
    content = f.read()
template = Template(content)

for folder in folders[:1]:
    string = template.substitute(params)
    with open(f'{folder}/{subdir}/{subdir}_ac.sh', 'w') as f:
        f.write(string)
