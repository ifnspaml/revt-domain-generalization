#!/bin/bash

##GENERAL -----
#SBATCH --job-name=segFormer_mergeTest
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=experiment_encoder_3-%j.out

##DEBUG -----
##SBATCH --partition=debug
##SBATCH --time=00:20:00

##NORMAL -----
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --exclude=gpu[04,05,02,03,06]

# Load modules
module load comp/gcc/11.2.0
source activate transformer-domain-generalization

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# Execute programs
script_dir=$(pwd)
cd ~/work/transformer-domain-generalization || return

srun python ./tools/test_reparametrization_3d.py\
 ./local_configs/segformer/B5/segformer.b5.512x512.gta2cs.40k.batch2.py\
 "slurm_script_runs/SegFormerB5/gta_train_split/test/re_param/ternary_plot/res_array_decoder_3_cs_continue.npy"\
 --launcher="slurm"\
 --gpu-collect\
 --eval-options efficient_test='True'\
 --weights-filter "backbone.*"\
 --options dist_params.backend='nccl' dist_params.port=29584\
 --resume-from "slurm_script_runs/SegFormerB5/gta_train_split/test/re_param/ternary_plot/res_array_decoder_3_cs.npy"\
 --checkpoints\
    "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_a/latest.pth" \
    "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_b/latest.pth" \
    "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_c/latest.pth"