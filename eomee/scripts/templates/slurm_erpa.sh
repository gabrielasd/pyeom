#!/bin/sh

#SBATCH --time=${time}
#SBATCH --mem=${memory}
#SBATCH --cpus-per-task=${n_proc}
#SBATCH --account=${account}
#SBATCH --job-name=${name}
#SBATCH --output=slurm-${name}.out

# ERPA -> Exit 0 if successful
python3 ${name}_ac.py  && exit 0

# Exit 1 if unsuccessful
exit 1