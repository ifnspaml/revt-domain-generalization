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
#SBATCH --job-name=SegFo_B2_gta-train_different2_encoder-soup_abc

## Name of the log file
#SBATCH --output=test_abc-%j.out

## The config that is used for training and testing
main_config="./local_configs/segformer/B2/segformer.b2.512x512.gta2cs.40k.batch2_pixmix_bilateralfilter.py"

## The name of the working dir where the results are saved. at the end the prefix "_[a..z]" is added.
declare -a work_dirs=("./work_dirs/gta_train/segformer_b2/segformer.b2.512x512.gta2cs.40k.batch2_bilateral_a/latest.pth"
                      "./work_dirs/gta_train/segformer_b2/segformer.b2.512x512.gta2cs.40k.batch2_pixmix_a/latest.pth"
                      "./work_dirs/gta_train/segformer_b2/segformer.b2.512x512.gta2cs.40k.batch2_pixmix_bilateralfilter_a/latest.pth")

declare -a merge_orders=("0;1;2" "1;2;0" "2;0;1")

work_dir_start_char="a"

## Defines the target domains on which the models are evaluated after training
test_cityscapes=true
test_cityscapes_val=true
test_bdd=true
test_mapillary=true
test_acdc=true
test_kitti=true
test_synthia=true
test_gta_val=true
test_synthia_val=true

declare -a regexes=("backbone.*")

######################################################################################

# Load modules
module load comp/gcc/11.2.0
source activate transformer-domain-generalization

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

suffix=$(printf '%x\n' "'$work_dir_start_char")


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
if [ "$test_cityscapes_val" = true ] ; then
    dataset_names+=("Cityscapes_val")
    dataset_types+=("CityscapesDataset")
    dataset_data_root+=("data/")
    dataset_img_dir+=("cityscapes/leftImg8bit/train")
    dataset_ann_dir+=("cityscapes/gtFine/train")
    data_test_split+=("data.test.split=\"cs_splits/val_split.txt\"")
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
if [ "$test_synthia_val" = true ] ; then
    dataset_names+=("synthia_val")
    dataset_types+=("SynthiaDataset")
    dataset_data_root+=("data/")
    dataset_img_dir+=("synthia/RGB")
    dataset_ann_dir+=("synthia/GT/LABELS/LABELS")
    data_test_split+=("data.test.split=\"synthia_splits/val_split.txt\"")
fi
echo -e "Test Datasets:"
for dataset_idx in "${!dataset_names[@]}"; do
    echo -e "   ${dataset_names[dataset_idx]}"
done

declare -a checkpoints_merged=()
for i in "${!merge_orders[@]}"; do
    rand_str=$(echo $RANDOM | md5sum | head -c 20; echo;)
    checkpoints_merged+=("./work_dirs/model_reparametrization/tmp_$rand_str.pth")
done

# Execute programs
script_dir=$(pwd)
cd ~/work/transformer-domain-generalization || return

## Re-Param
for regex_idx in "${!regexes[@]}"; do
  echo -e "Test for Regex: ${regexes[regex_idx]}"
  for i in "${!merge_orders[@]}"; do
    IFS=";" read -r -a order <<< "${merge_orders[i]}"
    checkpoints_str=""
    for j in "${!order[@]}"; do
        checkpoints_str+=" ${work_dirs["${order[j]}"]}"
    done
    srun python tools/model_reparametrization.py\
     "${checkpoints_merged[i]}"\
     --checkpoints\
        $(echo -n "$checkpoints_str")\
     --weights-filter "${regexes[regex_idx]}"
    echo -e "Finished Re-Param ${i}"
  done

  ## evaluate the models
  for dataset_idx in "${!dataset_names[@]}"; do
    echo -e "Test all on ${dataset_names[dataset_idx]}"
    for checkpoint_idx in "${!checkpoints_merged[@]}"; do
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
           "${checkpoints_merged[checkpoint_idx]}"\
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
done

for i in "${!checkpoints_merged[@]}"; do
    rm "${checkpoints_merged[i]}"
done