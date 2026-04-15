#!/usr/bin/env bash

#BSUB -J run
#BSUB -q c02613
#BSUB -W 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model == XeonE5_2660v3]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o run_%J.out
#BSUB -e run_%J.err

python3 -u numba_analysis.py
