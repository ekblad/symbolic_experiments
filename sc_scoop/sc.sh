#!/bin/bash -l
#SBATCH --job-name=symb_class
#SBATCH --nodes=3
#SBATCH -n 96
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH -p high           # Queue name "normal"
#SBATCH -t 999999        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=wdlynch@ucdavis.edu # address for email notification
#SBATCH --mail-type=ALL # email at Begin and End of job
#SBATCH --array=0-20%1

GDIR=/group/hermangrp
export PATH=$GDIR/miniconda3/bin:$PATH

hosts=$(srun bash -c hostname)

source activate py37
python -m scoop --host $hosts -v sc.py $SLURM_ARRAY_TASK_ID
