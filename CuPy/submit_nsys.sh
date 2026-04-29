#!/bin/sh
#BSUB -q c02613
#BSUB -J profile_and_read
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:15
#BSUB -R "rusage[mem=16GB]"
#BSUB -o prof_output_%J.out

module load cuda/11.8


nsys profile --trace=cuda,osrt --force-overwrite true -o my_new_profile_${LSB_JOBID} python3 simulate_prob10.py 20


nsys stats my_new_profile_${LSB_JOBID}.nsys-rep