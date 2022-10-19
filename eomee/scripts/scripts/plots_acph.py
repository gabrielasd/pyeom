import os

from matplotlib import pyplot
import matplotlib as mpl

import numpy as np

from iodata import load_one


def compile_table_phac(folders):
    folders = sorted(folders)

    for index, fp_job in enumerate(folders):
        # get folder & filename
        _, subdir = os.path.split(fp_job)
        rhf = load_one(f"{subdir}.molden")
        ci = np.load(f"{subdir}.ci.npz")
        ac = np.load(f"{subdir}.acphnonsymm.npz")
    return {
        'rhf': rhf['energy'],
        'doci': ci['energy'],
        'ph-AC': ac['energy'],
    }
