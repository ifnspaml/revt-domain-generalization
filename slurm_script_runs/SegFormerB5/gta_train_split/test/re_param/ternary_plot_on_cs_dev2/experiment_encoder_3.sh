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
#SBATCH --partition=debug
#SBATCH --time=00:20:00

##NORMAL -----
##SBATCH --partition=gpu
##SBATCH --time=8-00:00:00
##SBATCH --exclude=gpu[04,05,02,03,06]

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

port=$(comm -23 <(seq 20000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
srun python ./tools/test_reparametrization_3d.py\
 ./local_configs/segformer/B5/segformer.b5.512x512.gta2cs.40k.batch2.py\
 "slurm_script_runs/SegFormerB5/gta_train_split/test/re_param/ternary_plot_on_cs_dev2/res_array_decoder_3_cs_val.npy"\
 --launcher="slurm"\
 --gpu-collect\
 --eval-options efficient_test='True'\
 --weights-filter "backbone.*"\
 --options dist_params.backend='nccl' dist_params.port=${port}\
        data.test.type="CityscapesDataset"\
        dataset_type="CityscapesDataset"\
        data_root="data/"\
        data.test.data_root="data/"\
        data.test.img_dir="cityscapes/leftImg8bit/train"\
        data.test.ann_dir="cityscapes/gtFine/train"\
        data.test.split="cs_splits/val_split.txt"\
 --checkpoints\
    "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_a/latest.pth" \
    "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_b/latest.pth" \
    "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_c/latest.pth"