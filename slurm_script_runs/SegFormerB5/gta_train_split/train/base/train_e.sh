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
#SBATCH --exclude=gpu[04,02]

######################################################################################
### Parameters:

## Name of the job script
#SBATCH --job-name=SegFo_B5_gta-train_base_encoder-soup_e

## Name of the log file
#SBATCH --output=train_e-%j.out

max_iters=40000

## The config that is used for training and testing
main_config="./local_configs/segformer/B5/segformer.b5.512x512.gta2cs.40k.batch2.py"

## The name of the working dir where the results are saved. at the end the prefix "_[a..z]" is added.
work_dir_prefix="./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base"

## Number of models that are trained and tested
num_models=1

## Defines the starting value for the working dir prefix
work_dir_start_char="e"

## Defines the start seed. The first model is trained wth this value. After this the seed is inceased by one for the next model
start_seed=16

## Defines on which dataset the models are trained
train_dataset="gta_train"
#train_dataset="gta_full"
#train_dataset="synthia"

## Defines if the models are evaluated on the target domains after training
test_models=true

## Defines the target domains on which the models are evaluated after training
test_cityscapes=true
test_bdd=false
test_mapillary=false
test_acdc=false
test_kitti=false
test_synthia=false
test_gta_val=false

override=false

######################################################################################

# Load modules
module load comp/gcc/11.2.0
source activate transformer-domain-generalization

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

# create work_dirs names and seeds
declare -a work_dirs=()
declare -a seeds=()
suffix=$(printf '%x\n' "'$work_dir_start_char")
echo -e "Names of the work_dirs:"
for (( i=0; i<num_models; i++ ))
do
    letter=$(echo "$((suffix + i))" | xxd -p -r)
    work_dirs+=("${work_dir_prefix}_${letter}")
    seeds+=($((start_seed + i)))
    echo -e "   ${work_dirs[i]}"
done

# Define variables for train dataset
data_train_split=""
if [ "$train_dataset" = "gta_full" ]; then
    data_train_type="GTADataset"
    data_train_data_root="data/gta/"
    data_train_img_dir="images"
    data_train_ann_dir="labels"

    echo -e "Train Dataset:\n   GTA Full (Train+Val)"
elif [ "$train_dataset" = "gta_train" ]; then
    data_train_type="GTADataset"
    data_train_data_root="data/"
    data_train_img_dir="gta/images"
    data_train_ann_dir="gta/labels"
    data_train_split="data.train.split=\"gta_splits/train_split.txt\""

    echo -e "Train Dataset:\n   GTA Train"
else
    data_train_type="SynthiaDataset"
    data_train_data_root="data/synthia/"
    data_train_img_dir="RGB"
    data_train_ann_dir="GT/LABELS/LABELS"

    echo -e "Train Dataset:\n   Synthia"
fi

# Define variables for Test dataset
declare -a dataset_names=()
declare -a dataset_types=()
declare -a dataset_data_root=()
declare -a dataset_img_dir=()
declare -a dataset_ann_dir=()
declare -a data_test_split=()

if [ "$test_cityscapes" = true ] ; then
    dataset_names+=("Cityscapes")
    dataset_types+=("CityscapesDataset")
    dataset_data_root+=("data/cityscapes/")
    dataset_img_dir+=("leftImg8bit/val")
    dataset_ann_dir+=("gtFine/val")
    data_test_split+=("")
fi
if [ "$test_bdd" = true ] ; then
    dataset_names+=("bdd")
    dataset_types+=("BDD100kDataset")
    dataset_data_root+=("data/bdd100k/")
    dataset_img_dir+=("images/val")
    dataset_ann_dir+=("labels/val")
    data_test_split+=("")
fi
if [ "$test_mapillary" = true ] ; then
    dataset_names+=("mapillary")
    dataset_types+=("MapillaryDataset")
    dataset_data_root+=("data/mapillary/")
    dataset_img_dir+=("validation/ColorImage")
    dataset_ann_dir+=("segmentation_trainid/validation/Segmentation")
    data_test_split+=("")
fi
if [ "$test_acdc" = true ] ; then
    dataset_names+=("acdc")
    dataset_types+=("ACDCDataset")
    dataset_data_root+=("data/acdc/")
    dataset_img_dir+=("rgb_anon/val")
    dataset_ann_dir+=("gt/val")
    data_test_split+=("")
fi
if [ "$test_kitti" = true ] ; then
    dataset_names+=("kitti")
    dataset_types+=("KITTI2015Dataset")
    dataset_data_root+=("data/kitti/")
    dataset_img_dir+=("images/validation")
    dataset_ann_dir+=("labels/validation")
    data_test_split+=("")
fi
if [ "$test_synthia" = true ] ; then
    dataset_names+=("synthia")
    dataset_types+=("SynthiaDataset")
    dataset_data_root+=("data/synthia/")
    dataset_img_dir+=("RGB")
    dataset_ann_dir+=("GT/LABELS/LABELS")
    data_test_split+=("")
fi
if [ "$test_gta_val" = true ] ; then
    dataset_names+=("gta_val")
    dataset_types+=("GTADataset")
    dataset_data_root+=("data/")
    dataset_img_dir+=("gta/images")
    dataset_ann_dir+=("gta/labels")

    data_test_split+=("data.test.split=\"gta_splits/val_split.txt\"")
fi
echo -e "Test Datasets:"
for dataset_idx in "${!dataset_names[@]}"; do
    echo -e "   ${dataset_names[dataset_idx]}"
done

# Execute programs
script_dir=$(pwd)
cd ~/work/transformer-domain-generalization || return

## train the models
for work_dirs_idx in "${!work_dirs[@]}"; do
  letter=$(echo "$((suffix + work_dirs_idx))" | xxd -p -r)
  if [[ -f "${work_dirs[work_dirs_idx]}/latest.pth" ]] && [[ ! -f "${work_dirs[work_dirs_idx]}/iter_${max_iters}.pth" ]] && [[ "$override" = false ]]; then
      # Resume Training
      port=$(comm -23 <(seq 20000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
      srun --output "${script_dir}/train_${letter}.log"\
       python ./tools/train.py\
       "${main_config}"\
        --seed ${seeds[work_dirs_idx]}\
        --launcher="slurm"\
        --resume-from "${work_dirs[work_dirs_idx]}/latest.pth"\
        --options work_dir="${work_dirs[work_dirs_idx]}"\
        runner.work_dir="${work_dirs[work_dirs_idx]}"\
        data.train.type="$data_train_type"\
        data.train.data_root="$data_train_data_root"\
        $(echo -n "$data_train_split")\
        data.train.img_dir="$data_train_img_dir"\
        data.train.ann_dir="$data_train_ann_dir"\
        data.test.type="${dataset_types[0]}"\
        dataset_type="${dataset_types[0]}"\
        data_root="${dataset_data_root[0]}"\
        data.test.data_root="${dataset_data_root[0]}"\
        data.test.img_dir="${dataset_img_dir[0]}"\
        data.test.ann_dir="${dataset_ann_dir[0]}"\
        dist_params.backend='nccl' dist_params.port=${port}
  elif [[ ! -f "${work_dirs[work_dirs_idx]}/iter_${max_iters}.pth" ]] || [[ "$override" = true ]]; then
      # Start training
      port=$(comm -23 <(seq 20000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
      srun --output "${script_dir}/train_${letter}.log"\
       python ./tools/train.py\
       "${main_config}"\
        --seed ${seeds[work_dirs_idx]}\
        --launcher="slurm"\
        --options work_dir="${work_dirs[work_dirs_idx]}"\
        runner.work_dir="${work_dirs[work_dirs_idx]}"\
        data.train.type="$data_train_type"\
        data.train.data_root="$data_train_data_root"\
        $(echo -n "$data_train_split")\
        data.train.img_dir="$data_train_img_dir"\
        data.train.ann_dir="$data_train_ann_dir"\
        data.test.type="${dataset_types[0]}"\
        dataset_type="${dataset_types[0]}"\
        data_root="${dataset_data_root[0]}"\
        data.test.data_root="${dataset_data_root[0]}"\
        data.test.img_dir="${dataset_img_dir[0]}"\
        data.test.ann_dir="${dataset_ann_dir[0]}"\
        dist_params.backend='nccl' dist_params.port=${port}
  fi
done

## evaluate the models
if [ "$test_models" = true ] ; then
    for dataset_idx in "${!dataset_names[@]}"; do
      echo -e "Test all on ${dataset_names[dataset_idx]}"
      for checkpoint_idx in "${!work_dirs[@]}"; do
        letter=$(echo "$((suffix + checkpoint_idx))" | xxd -p -r)
        log_file="${script_dir}/test_${letter}_on_${dataset_names[dataset_idx]}.log"
        isInFile=$( (cat "${log_file}" || true) | grep -c "mIoU")
        if [ $isInFile -eq 0 ]; then
            echo -e "Test all on ${dataset_names[dataset_idx]}" >> "${log_file}"
            echo -e "Test for ${checkpoint_idx}:"
            echo -e "Test for ${checkpoint_idx}:" >> "${log_file}"
            srun --output "${log_file}"\
             python ./tools/test.py\
             "${main_config}"\
             "${work_dirs[checkpoint_idx]}/latest.pth"\
             --gpu-collect\
             --eval-options efficient_test='True'\
             --no-progress-bar\
             --options \
                data.test.type="${dataset_types[dataset_idx]}"\
                dataset_type="${dataset_types[dataset_idx]}"\
                data_root="${dataset_data_root[dataset_idx]}"\
                data.test.data_root="${dataset_data_root[dataset_idx]}"\
                data.test.img_dir="${dataset_img_dir[dataset_idx]}"\
                data.test.ann_dir="${dataset_ann_dir[dataset_idx]}"\
                $(echo -n "${data_test_split[dataset_idx]}")
            echo -e "___________________________________________________"
        fi
      done
    done
fi