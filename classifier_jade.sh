#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# big partition
#SBATCH --partition=small

# set max wallclock time
#SBATCH --time=24:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=ag803@sussex.ac.uk

arguments_path="jade_bash.txt"
LINES=`wc --lines < $arguments_path`

for i in $(seq ${SLURM_ARRAY_TASK_ID} 10 $LINES); do
        ARGS=`head -$i $arguments_path | tail -1`
		
		python classifier.py --train --seed 2345 --dataset dvs_gesture --num-epochs 100 --hidden-size 256 256 --hidden-recurrent False True --hidden-model alif alif --hidden-input-sparsity 0.1 0.1 --hidden-recurrent-sparsity 0.01 $ARGS
		
done
