#!/usr/bin/env bash

#BSUB -J numba
#BSUB -q c02613
#BSUB -W 30 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model == XeonE5_2660v3]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o numba_%J.out
#BSUB -e numba_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
echo "Hello from numba"

time python3 -u numba_analysis.py 10

