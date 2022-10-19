import os, sys 
from glob import glob


header = """ &FCI NORB=10,NELEC=14,MS2=0,
 ORBSYM = 1,1,1,1,1,1,1,1,1,1,
 ISYM=1
 &END
"""

prefix = 'N2'
subdir = '0007_q000_m01_k00_sp_oodoci'
pattern = 'END\n'

mols = glob(f'{prefix}*')
folders = sorted(mols, key=lambda job: float(job.split('_')[-1])) #sort by index at file end

for folder in folders:
    with open(f'{folder}/{subdir}/{folder}.FCIDUMP', 'r') as f:
        content = f.read()
    
    body = content.split(pattern)[1]

    with open(f'{folder}/{subdir}/{subdir}.FCIDUMP', 'w') as f:
        f.write(header)
        f.write(body)
        