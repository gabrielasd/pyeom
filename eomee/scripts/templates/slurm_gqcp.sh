#!/bin/sh

#SBATCH --time=${time}
#SBATCH --mem=${memory}G
#SBATCH --cpus-per-task=${n_proc}
#SBATCH --account=${account}
#SBATCH --job-name=${name}
#SBATCH --output=slurm-${name}-%j.out

module load singularity

# OODOCI -> Exit 0 if successful
singularity exec -e -B /home -B /project -B /scratch ~/bin/GQCP/gqcp.sif python3 ${name}.py  && exit 0

# Exit 1 if unsuccessful
exit 1