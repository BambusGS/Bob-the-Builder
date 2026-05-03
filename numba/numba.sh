g!/usr/bin/env bash

#BSUB -J numba
#BSUB -q c02613
#BSUB -W  5
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16384MB]"
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o numba_all_%J.out
#BSUB -e numba_all_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
echo "Hello from numba"

time python3 -u numba.py 20

