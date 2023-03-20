import os
# from glob import glob
from utils import richerm2database


acdata = os.getenv('ACDATA')
s = 1
q = 0
prefix = f'B_{q}_{s}'
subdir = f'{prefix}'
basis =  'aug-cc-pVDZ'


# name = f'{acdata}/richerm/{subdir}/{prefix}_{basis}'
# richerm2database(name)


# os.system(f'mv {acdata}/richerm/{subdir} {acdata}/richerm/{prefix}_{basis}')
