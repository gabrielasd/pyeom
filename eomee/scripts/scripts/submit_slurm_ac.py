from glob import glob
import os


prefix = 'h2o'
subdir = '0008_q000_m01_k00_sp_doci_ccpvdz'
mols = glob(f'{prefix}*')
folders = sorted(mols, key=lambda job: float(job.split('_')[-1])) #sort by index at file end


for folder in folders[:1]:
    fp_slurm = f'{folder}/{subdir}/{subdir}_ac.sh'
    # get base directory
    base_database = os.getcwd()
    
    # get folder & filename
    folder, fn_slurm = os.path.split(fp_slurm)
    if folder != "":
        os.chdir(folder)
    # submit job
    os.system(f"sbatch {fn_slurm}")
    os.system("sleep 3")

    os.chdir(base_database)
