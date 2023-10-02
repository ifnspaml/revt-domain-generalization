#!/bin/bash

##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

##DEBUG -----
##SBATCH --partition=debug
##SBATCH --time=00:20:00

##NORMAL -----
#SBATCH --partition=gpu,gpub
#SBATCH --time=7-00:00:00
#SBATCH --exclude=gpu[01-05,07]

## Name of the job script
#SBATCH --job-name=fcn_r50_d8_Benchmark

## Name of the log file
#SBATCH --output=fcn_r50_d8_Benchmark-%j.out

cd ~/work/transformer-domain-generalization || return
module load comp/gcc/11.2.0
##source activate transformer-domain-generalization
source activate transformer-domain-generalization

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

srun python ./tools/benchmark.py local_configs/ResNet/50/fcn_r50-d8_512x1024_40k_cityscapes.py