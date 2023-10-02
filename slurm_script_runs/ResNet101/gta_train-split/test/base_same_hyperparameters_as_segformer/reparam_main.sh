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
##SBATCH --exclude=gpu[04,02,05,01]

######################################################################################
### Parameters:

## Name of the job script
#SBATCH --job-name=Dlv3+_r101_gta_train-split_base_same_hyperparams

## Name of the log file
#SBATCH --output=test_abc-%j.out

## The config that is used for training and testing

## The config that is used for training and testing
main_config="./local_configs/ResNet/101/deeplabv3plus_r101.b5.512x512.gta2cs.40k.batch2.py"

## The name of the working dir where the results are saved. at the end the prefix "_[a..z]" is added.
declare -a work_dirs=("./work_dirs/DLv3plusR101/deeplabv3plus.r101.512x512.gta2cs.60k.adamw_lr0_00006_a/iter_40000.pth"
                      "./work_dirs/DLv3plusR101/deeplabv3plus.r101.512x512.gta2cs.60k.adamw_lr0_00006_b/iter_40000.pth"
                      "./work_dirs/DLv3plusR101/deeplabv3plus.r101.512x512.gta2cs.60k.adamw_lr0_00006_c/iter_40000.pth")

work_dir_start_char="a"

test_cityscapes=true
test_cityscapes_val=true
test_bdd=true
test_mapillary=true
test_acdc=true
test_kitti=true
test_synthia=true
test_gta_val=true
test_synthia_val=true

######################################################################################

script_dir=$(pwd)
cd ~/work/transformer-domain-generalization || return
source slurm_script_runs/utils/slurm_init.sh
source slurm_script_runs/utils/declare_test_dataset.sh
source slurm_script_runs/utils/test.sh

## Test reparameterized models
for dataset_idx in "${!dataset_names[@]}"; do

  for checkpoint_idx in "${!work_dirs[@]}"; do
    letter=$(echo "$((suffix + checkpoint_idx))" | xxd -p -r)
    log_file="${script_dir}/test_${letter}_on_${dataset_names[dataset_idx]}.log"
    test_dataset "${main_config}"\
           "${work_dirs[checkpoint_idx]}"\
           "${dataset_types[dataset_idx]}"\
           "${dataset_data_root[dataset_idx]}"\
           "${dataset_img_dir[dataset_idx]}"\
           "${dataset_ann_dir[dataset_idx]}"\
           "${data_test_split[dataset_idx]}"\
           "$log_file"
  done
done