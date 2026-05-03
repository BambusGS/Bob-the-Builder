#!/bin/sh
#BSUB -q c02613
#BSUB -J sim9
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=16GB]" 
#BSUB -o output_%J.out

module load cuda/11.8
time python3 simulate_prob9.py 4571
