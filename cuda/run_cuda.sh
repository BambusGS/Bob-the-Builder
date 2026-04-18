#!/usr/bin/env bash

#BSUB -J cuda
#BSUB -q c02613
#BSUB -W 00:30 
#BSUB -n 4
#BSUB -R "rusage[mem=4069MB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o LOG/cuda_%J.out
#BSUB -e LOG/cuda_%J.err

# commands explanation:
# -J : job name
# -q : queue name
# -W : walltime limit in minutes
# -n : number of cores
# -R : resource requirements memory per core
# -R : resource requirements to run on a single host
# -o : output file
# -e : error file
# -N for job completion -B for job begin #BSUB -N

# nodestat -F

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

lscpu

# Run the time command, input N, and pipe err to stdout for logging
python3 -u cuda.py 4571 false 2>&1
# python3 -u cuda.py 10 true 2>&1

# nsys profile -o cuda_profile python3 -u cuda.py 10 false 2>&1
# nsys stats cuda_profile.nsys-rep

# sleep 1 # wait for the files to be written

# rm -rf cuda_profile.nsys-rep
# rm -rf cuda_profile.sqlite