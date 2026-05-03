#!/bin/sh
#BSUB -q gpuv100
#BSUB -J sim10
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -R "rusage[mem=16GB]" 
#BSUB -o output_%J.out
#BSUB -e error_%J.err

module load cuda/11.8

time python3 simulate_prob10.py 20