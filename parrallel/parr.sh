#!/usr/bin/env bash

#BSUB -J parr
#BSUB -q hpc
#BSUB -W 30 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3024MB]"
#BSUB -n 12
#BSUB -o parr_%J.out
#BSUB -e parr_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
echo "Hello from orig"

time python3 -u parralel.py 20 1
time python3 -u parralel.py 20 2
time python3 -u parralel.py 20 3
time python3 -u parralel.py 20 4
time python3 -u parralel.py 20 5
time python3 -u parralel.py 20 6
time python3 -u parralel.py 20 7
time python3 -u parralel.py 20 8
time python3 -u parralel.py 20 9
time python3 -u parralel.py 20 10
time python3 -u parralel.py 20 11
time python3 -u parralel.py 20 12

