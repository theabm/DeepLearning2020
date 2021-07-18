#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00

cd $PBS_O_WORKDIR

~/.conda/envs/dl/bin/python train.py --dataroot ./datasets/maps --which_direction BtoA --num_epochs 200 --batchSize 1
~/.conda/envs/dl/bin/python test.py --dataroot ./datasets/maps --which_direction BtoA --num_epochs 200 --batchSize 1

