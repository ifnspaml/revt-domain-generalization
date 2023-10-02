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
##SBATCH --exclude=gpu[04,02]

######################################################################################
### Parameters:

## Name of the job script
#SBATCH --job-name=SegFo_B5_gta-trainset_base_encoder-soup_abc

## Name of the log file
#SBATCH --output=test_abc-%j.out

## The config that is used for training and testing
main_config="./local_configs/segformer/B5/segformer.b5.512x512.gta2cs.40k.batch2.py"

## The name of the working dir where the results are saved. at the end the prefix "_[a..z]" is added.
declare -a work_dirs=("./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_a/latest.pth"
                      "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_b/latest.pth"
                      "./work_dirs/gta_train_split/segformer.b5.512x512.gta2cs.40k.batch2_base_c/latest.pth")

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

declare -a regexes=('(backbone\.patch_embed1\.proj\.weight)|(backbone\.patch_embed1\.proj\.bias)|(backbone\.patch_embed2\.proj\.weight)|(backbone\.patch_embed2\.proj\.bias)|(backbone\.patch_embed3\.proj\.weight)|(backbone\.patch_embed3\.proj\.bias)|(backbone\.patch_embed4\.proj\.weight)|(backbone\.patch_embed4\.proj\.bias)|(backbone\.block1\.0\.attn\.q\.weight)|(backbone\.block1\.0\.attn\.q\.bias)|(backbone\.block1\.0\.attn\.kv\.weight)|(backbone\.block1\.0\.attn\.kv\.bias)|(backbone\.block1\.0\.attn\.proj\.weight)|(backbone\.block1\.0\.attn\.proj\.bias)|(backbone\.block1\.0\.attn\.sr\.weight)|(backbone\.block1\.0\.attn\.sr\.bias)|(backbone\.block1\.0\.mlp\.fc1\.weight)|(backbone\.block1\.0\.mlp\.fc1\.bias)|(backbone\.block1\.0\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block1\.0\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block1\.0\.mlp\.fc2\.weight)|(backbone\.block1\.0\.mlp\.fc2\.bias)|(backbone\.block1\.1\.attn\.q\.weight)|(backbone\.block1\.1\.attn\.q\.bias)|(backbone\.block1\.1\.attn\.kv\.weight)|(backbone\.block1\.1\.attn\.kv\.bias)|(backbone\.block1\.1\.attn\.proj\.weight)|(backbone\.block1\.1\.attn\.proj\.bias)|(backbone\.block1\.1\.attn\.sr\.weight)|(backbone\.block1\.1\.attn\.sr\.bias)|(backbone\.block1\.1\.mlp\.fc1\.weight)|(backbone\.block1\.1\.mlp\.fc1\.bias)|(backbone\.block1\.1\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block1\.1\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block1\.1\.mlp\.fc2\.weight)|(backbone\.block1\.1\.mlp\.fc2\.bias)|(backbone\.block1\.2\.attn\.q\.weight)|(backbone\.block1\.2\.attn\.q\.bias)|(backbone\.block1\.2\.attn\.kv\.weight)|(backbone\.block1\.2\.attn\.kv\.bias)|(backbone\.block1\.2\.attn\.proj\.weight)|(backbone\.block1\.2\.attn\.proj\.bias)|(backbone\.block1\.2\.attn\.sr\.weight)|(backbone\.block1\.2\.attn\.sr\.bias)|(backbone\.block1\.2\.mlp\.fc1\.weight)|(backbone\.block1\.2\.mlp\.fc1\.bias)|(backbone\.block1\.2\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block1\.2\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block1\.2\.mlp\.fc2\.weight)|(backbone\.block1\.2\.mlp\.fc2\.bias)|(backbone\.block2\.0\.attn\.q\.weight)|(backbone\.block2\.0\.attn\.q\.bias)|(backbone\.block2\.0\.attn\.kv\.weight)|(backbone\.block2\.0\.attn\.kv\.bias)|(backbone\.block2\.0\.attn\.proj\.weight)|(backbone\.block2\.0\.attn\.proj\.bias)|(backbone\.block2\.0\.attn\.sr\.weight)|(backbone\.block2\.0\.attn\.sr\.bias)|(backbone\.block2\.0\.mlp\.fc1\.weight)|(backbone\.block2\.0\.mlp\.fc1\.bias)|(backbone\.block2\.0\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block2\.0\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block2\.0\.mlp\.fc2\.weight)|(backbone\.block2\.0\.mlp\.fc2\.bias)|(backbone\.block2\.1\.attn\.q\.weight)|(backbone\.block2\.1\.attn\.q\.bias)|(backbone\.block2\.1\.attn\.kv\.weight)|(backbone\.block2\.1\.attn\.kv\.bias)|(backbone\.block2\.1\.attn\.proj\.weight)|(backbone\.block2\.1\.attn\.proj\.bias)|(backbone\.block2\.1\.attn\.sr\.weight)|(backbone\.block2\.1\.attn\.sr\.bias)|(backbone\.block2\.1\.mlp\.fc1\.weight)|(backbone\.block2\.1\.mlp\.fc1\.bias)|(backbone\.block2\.1\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block2\.1\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block2\.1\.mlp\.fc2\.weight)|(backbone\.block2\.1\.mlp\.fc2\.bias)|(backbone\.block2\.2\.attn\.q\.weight)|(backbone\.block2\.2\.attn\.q\.bias)|(backbone\.block2\.2\.attn\.kv\.weight)|(backbone\.block2\.2\.attn\.kv\.bias)|(backbone\.block2\.2\.attn\.proj\.weight)|(backbone\.block2\.2\.attn\.proj\.bias)|(backbone\.block2\.2\.attn\.sr\.weight)|(backbone\.block2\.2\.attn\.sr\.bias)|(backbone\.block2\.2\.mlp\.fc1\.weight)|(backbone\.block2\.2\.mlp\.fc1\.bias)|(backbone\.block2\.2\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block2\.2\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block2\.2\.mlp\.fc2\.weight)|(backbone\.block2\.2\.mlp\.fc2\.bias)|(backbone\.block2\.3\.attn\.q\.weight)|(backbone\.block2\.3\.attn\.q\.bias)|(backbone\.block2\.3\.attn\.kv\.weight)|(backbone\.block2\.3\.attn\.kv\.bias)|(backbone\.block2\.3\.attn\.proj\.weight)|(backbone\.block2\.3\.attn\.proj\.bias)|(backbone\.block2\.3\.attn\.sr\.weight)|(backbone\.block2\.3\.attn\.sr\.bias)|(backbone\.block2\.3\.mlp\.fc1\.weight)|(backbone\.block2\.3\.mlp\.fc1\.bias)|(backbone\.block2\.3\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block2\.3\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block2\.3\.mlp\.fc2\.weight)|(backbone\.block2\.3\.mlp\.fc2\.bias)|(backbone\.block2\.4\.attn\.q\.weight)|(backbone\.block2\.4\.attn\.q\.bias)|(backbone\.block2\.4\.attn\.kv\.weight)|(backbone\.block2\.4\.attn\.kv\.bias)|(backbone\.block2\.4\.attn\.proj\.weight)|(backbone\.block2\.4\.attn\.proj\.bias)|(backbone\.block2\.4\.attn\.sr\.weight)|(backbone\.block2\.4\.attn\.sr\.bias)|(backbone\.block2\.4\.mlp\.fc1\.weight)|(backbone\.block2\.4\.mlp\.fc1\.bias)|(backbone\.block2\.4\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block2\.4\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block2\.4\.mlp\.fc2\.weight)|(backbone\.block2\.4\.mlp\.fc2\.bias)|(backbone\.block2\.5\.attn\.q\.weight)|(backbone\.block2\.5\.attn\.q\.bias)|(backbone\.block2\.5\.attn\.kv\.weight)|(backbone\.block2\.5\.attn\.kv\.bias)|(backbone\.block2\.5\.attn\.proj\.weight)|(backbone\.block2\.5\.attn\.proj\.bias)|(backbone\.block2\.5\.attn\.sr\.weight)|(backbone\.block2\.5\.attn\.sr\.bias)|(backbone\.block2\.5\.mlp\.fc1\.weight)|(backbone\.block2\.5\.mlp\.fc1\.bias)|(backbone\.block2\.5\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block2\.5\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block2\.5\.mlp\.fc2\.weight)|(backbone\.block2\.5\.mlp\.fc2\.bias)|(backbone\.block3\.0\.attn\.q\.weight)|(backbone\.block3\.0\.attn\.q\.bias)|(backbone\.block3\.0\.attn\.kv\.weight)|(backbone\.block3\.0\.attn\.kv\.bias)|(backbone\.block3\.0\.attn\.proj\.weight)|(backbone\.block3\.0\.attn\.proj\.bias)|(backbone\.block3\.0\.attn\.sr\.weight)|(backbone\.block3\.0\.attn\.sr\.bias)|(backbone\.block3\.0\.mlp\.fc1\.weight)|(backbone\.block3\.0\.mlp\.fc1\.bias)|(backbone\.block3\.0\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.0\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.0\.mlp\.fc2\.weight)|(backbone\.block3\.0\.mlp\.fc2\.bias)|(backbone\.block3\.1\.attn\.q\.weight)|(backbone\.block3\.1\.attn\.q\.bias)|(backbone\.block3\.1\.attn\.kv\.weight)|(backbone\.block3\.1\.attn\.kv\.bias)|(backbone\.block3\.1\.attn\.proj\.weight)|(backbone\.block3\.1\.attn\.proj\.bias)|(backbone\.block3\.1\.attn\.sr\.weight)|(backbone\.block3\.1\.attn\.sr\.bias)|(backbone\.block3\.1\.mlp\.fc1\.weight)|(backbone\.block3\.1\.mlp\.fc1\.bias)|(backbone\.block3\.1\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.1\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.1\.mlp\.fc2\.weight)|(backbone\.block3\.1\.mlp\.fc2\.bias)|(backbone\.block3\.2\.attn\.q\.weight)|(backbone\.block3\.2\.attn\.q\.bias)|(backbone\.block3\.2\.attn\.kv\.weight)|(backbone\.block3\.2\.attn\.kv\.bias)|(backbone\.block3\.2\.attn\.proj\.weight)|(backbone\.block3\.2\.attn\.proj\.bias)|(backbone\.block3\.2\.attn\.sr\.weight)|(backbone\.block3\.2\.attn\.sr\.bias)|(backbone\.block3\.2\.mlp\.fc1\.weight)|(backbone\.block3\.2\.mlp\.fc1\.bias)|(backbone\.block3\.2\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.2\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.2\.mlp\.fc2\.weight)|(backbone\.block3\.2\.mlp\.fc2\.bias)|(backbone\.block3\.3\.attn\.q\.weight)|(backbone\.block3\.3\.attn\.q\.bias)|(backbone\.block3\.3\.attn\.kv\.weight)|(backbone\.block3\.3\.attn\.kv\.bias)|(backbone\.block3\.3\.attn\.proj\.weight)|(backbone\.block3\.3\.attn\.proj\.bias)|(backbone\.block3\.3\.attn\.sr\.weight)|(backbone\.block3\.3\.attn\.sr\.bias)|(backbone\.block3\.3\.mlp\.fc1\.weight)|(backbone\.block3\.3\.mlp\.fc1\.bias)|(backbone\.block3\.3\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.3\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.3\.mlp\.fc2\.weight)|(backbone\.block3\.3\.mlp\.fc2\.bias)|(backbone\.block3\.4\.attn\.q\.weight)|(backbone\.block3\.4\.attn\.q\.bias)|(backbone\.block3\.4\.attn\.kv\.weight)|(backbone\.block3\.4\.attn\.kv\.bias)|(backbone\.block3\.4\.attn\.proj\.weight)|(backbone\.block3\.4\.attn\.proj\.bias)|(backbone\.block3\.4\.attn\.sr\.weight)|(backbone\.block3\.4\.attn\.sr\.bias)|(backbone\.block3\.4\.mlp\.fc1\.weight)|(backbone\.block3\.4\.mlp\.fc1\.bias)|(backbone\.block3\.4\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.4\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.4\.mlp\.fc2\.weight)|(backbone\.block3\.4\.mlp\.fc2\.bias)|(backbone\.block3\.5\.attn\.q\.weight)|(backbone\.block3\.5\.attn\.q\.bias)|(backbone\.block3\.5\.attn\.kv\.weight)|(backbone\.block3\.5\.attn\.kv\.bias)|(backbone\.block3\.5\.attn\.proj\.weight)|(backbone\.block3\.5\.attn\.proj\.bias)|(backbone\.block3\.5\.attn\.sr\.weight)|(backbone\.block3\.5\.attn\.sr\.bias)|(backbone\.block3\.5\.mlp\.fc1\.weight)|(backbone\.block3\.5\.mlp\.fc1\.bias)|(backbone\.block3\.5\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.5\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.5\.mlp\.fc2\.weight)|(backbone\.block3\.5\.mlp\.fc2\.bias)|(backbone\.block3\.6\.attn\.q\.weight)|(backbone\.block3\.6\.attn\.q\.bias)|(backbone\.block3\.6\.attn\.kv\.weight)|(backbone\.block3\.6\.attn\.kv\.bias)|(backbone\.block3\.6\.attn\.proj\.weight)|(backbone\.block3\.6\.attn\.proj\.bias)|(backbone\.block3\.6\.attn\.sr\.weight)|(backbone\.block3\.6\.attn\.sr\.bias)|(backbone\.block3\.6\.mlp\.fc1\.weight)|(backbone\.block3\.6\.mlp\.fc1\.bias)|(backbone\.block3\.6\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.6\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.6\.mlp\.fc2\.weight)|(backbone\.block3\.6\.mlp\.fc2\.bias)|(backbone\.block3\.7\.attn\.q\.weight)|(backbone\.block3\.7\.attn\.q\.bias)|(backbone\.block3\.7\.attn\.kv\.weight)|(backbone\.block3\.7\.attn\.kv\.bias)|(backbone\.block3\.7\.attn\.proj\.weight)|(backbone\.block3\.7\.attn\.proj\.bias)|(backbone\.block3\.7\.attn\.sr\.weight)|(backbone\.block3\.7\.attn\.sr\.bias)|(backbone\.block3\.7\.mlp\.fc1\.weight)|(backbone\.block3\.7\.mlp\.fc1\.bias)|(backbone\.block3\.7\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.7\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.7\.mlp\.fc2\.weight)|(backbone\.block3\.7\.mlp\.fc2\.bias)|(backbone\.block3\.8\.attn\.q\.weight)|(backbone\.block3\.8\.attn\.q\.bias)|(backbone\.block3\.8\.attn\.kv\.weight)|(backbone\.block3\.8\.attn\.kv\.bias)|(backbone\.block3\.8\.attn\.proj\.weight)|(backbone\.block3\.8\.attn\.proj\.bias)|(backbone\.block3\.8\.attn\.sr\.weight)|(backbone\.block3\.8\.attn\.sr\.bias)|(backbone\.block3\.8\.mlp\.fc1\.weight)|(backbone\.block3\.8\.mlp\.fc1\.bias)|(backbone\.block3\.8\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.8\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.8\.mlp\.fc2\.weight)|(backbone\.block3\.8\.mlp\.fc2\.bias)|(backbone\.block3\.9\.attn\.q\.weight)|(backbone\.block3\.9\.attn\.q\.bias)|(backbone\.block3\.9\.attn\.kv\.weight)|(backbone\.block3\.9\.attn\.kv\.bias)|(backbone\.block3\.9\.attn\.proj\.weight)|(backbone\.block3\.9\.attn\.proj\.bias)|(backbone\.block3\.9\.attn\.sr\.weight)|(backbone\.block3\.9\.attn\.sr\.bias)|(backbone\.block3\.9\.mlp\.fc1\.weight)|(backbone\.block3\.9\.mlp\.fc1\.bias)|(backbone\.block3\.9\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.9\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.9\.mlp\.fc2\.weight)|(backbone\.block3\.9\.mlp\.fc2\.bias)|(backbone\.block3\.10\.attn\.q\.weight)|(backbone\.block3\.10\.attn\.q\.bias)|(backbone\.block3\.10\.attn\.kv\.weight)|(backbone\.block3\.10\.attn\.kv\.bias)|(backbone\.block3\.10\.attn\.proj\.weight)|(backbone\.block3\.10\.attn\.proj\.bias)|(backbone\.block3\.10\.attn\.sr\.weight)|(backbone\.block3\.10\.attn\.sr\.bias)|(backbone\.block3\.10\.mlp\.fc1\.weight)|(backbone\.block3\.10\.mlp\.fc1\.bias)|(backbone\.block3\.10\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.10\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.10\.mlp\.fc2\.weight)|(backbone\.block3\.10\.mlp\.fc2\.bias)|(backbone\.block3\.11\.attn\.q\.weight)|(backbone\.block3\.11\.attn\.q\.bias)|(backbone\.block3\.11\.attn\.kv\.weight)|(backbone\.block3\.11\.attn\.kv\.bias)|(backbone\.block3\.11\.attn\.proj\.weight)|(backbone\.block3\.11\.attn\.proj\.bias)|(backbone\.block3\.11\.attn\.sr\.weight)|(backbone\.block3\.11\.attn\.sr\.bias)|(backbone\.block3\.11\.mlp\.fc1\.weight)|(backbone\.block3\.11\.mlp\.fc1\.bias)|(backbone\.block3\.11\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.11\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.11\.mlp\.fc2\.weight)|(backbone\.block3\.11\.mlp\.fc2\.bias)|(backbone\.block3\.12\.attn\.q\.weight)|(backbone\.block3\.12\.attn\.q\.bias)|(backbone\.block3\.12\.attn\.kv\.weight)|(backbone\.block3\.12\.attn\.kv\.bias)|(backbone\.block3\.12\.attn\.proj\.weight)|(backbone\.block3\.12\.attn\.proj\.bias)|(backbone\.block3\.12\.attn\.sr\.weight)|(backbone\.block3\.12\.attn\.sr\.bias)|(backbone\.block3\.12\.mlp\.fc1\.weight)|(backbone\.block3\.12\.mlp\.fc1\.bias)|(backbone\.block3\.12\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.12\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.12\.mlp\.fc2\.weight)|(backbone\.block3\.12\.mlp\.fc2\.bias)|(backbone\.block3\.13\.attn\.q\.weight)|(backbone\.block3\.13\.attn\.q\.bias)|(backbone\.block3\.13\.attn\.kv\.weight)|(backbone\.block3\.13\.attn\.kv\.bias)|(backbone\.block3\.13\.attn\.proj\.weight)|(backbone\.block3\.13\.attn\.proj\.bias)|(backbone\.block3\.13\.attn\.sr\.weight)|(backbone\.block3\.13\.attn\.sr\.bias)|(backbone\.block3\.13\.mlp\.fc1\.weight)|(backbone\.block3\.13\.mlp\.fc1\.bias)|(backbone\.block3\.13\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.13\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.13\.mlp\.fc2\.weight)|(backbone\.block3\.13\.mlp\.fc2\.bias)|(backbone\.block3\.14\.attn\.q\.weight)|(backbone\.block3\.14\.attn\.q\.bias)|(backbone\.block3\.14\.attn\.kv\.weight)|(backbone\.block3\.14\.attn\.kv\.bias)|(backbone\.block3\.14\.attn\.proj\.weight)|(backbone\.block3\.14\.attn\.proj\.bias)|(backbone\.block3\.14\.attn\.sr\.weight)|(backbone\.block3\.14\.attn\.sr\.bias)|(backbone\.block3\.14\.mlp\.fc1\.weight)|(backbone\.block3\.14\.mlp\.fc1\.bias)|(backbone\.block3\.14\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.14\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.14\.mlp\.fc2\.weight)|(backbone\.block3\.14\.mlp\.fc2\.bias)|(backbone\.block3\.15\.attn\.q\.weight)|(backbone\.block3\.15\.attn\.q\.bias)|(backbone\.block3\.15\.attn\.kv\.weight)|(backbone\.block3\.15\.attn\.kv\.bias)|(backbone\.block3\.15\.attn\.proj\.weight)|(backbone\.block3\.15\.attn\.proj\.bias)|(backbone\.block3\.15\.attn\.sr\.weight)|(backbone\.block3\.15\.attn\.sr\.bias)|(backbone\.block3\.15\.mlp\.fc1\.weight)|(backbone\.block3\.15\.mlp\.fc1\.bias)|(backbone\.block3\.15\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.15\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.15\.mlp\.fc2\.weight)|(backbone\.block3\.15\.mlp\.fc2\.bias)|(backbone\.block3\.16\.attn\.q\.weight)|(backbone\.block3\.16\.attn\.q\.bias)|(backbone\.block3\.16\.attn\.kv\.weight)|(backbone\.block3\.16\.attn\.kv\.bias)|(backbone\.block3\.16\.attn\.proj\.weight)|(backbone\.block3\.16\.attn\.proj\.bias)|(backbone\.block3\.16\.attn\.sr\.weight)|(backbone\.block3\.16\.attn\.sr\.bias)|(backbone\.block3\.16\.mlp\.fc1\.weight)|(backbone\.block3\.16\.mlp\.fc1\.bias)|(backbone\.block3\.16\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.16\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.16\.mlp\.fc2\.weight)|(backbone\.block3\.16\.mlp\.fc2\.bias)|(backbone\.block3\.17\.attn\.q\.weight)|(backbone\.block3\.17\.attn\.q\.bias)|(backbone\.block3\.17\.attn\.kv\.weight)|(backbone\.block3\.17\.attn\.kv\.bias)|(backbone\.block3\.17\.attn\.proj\.weight)|(backbone\.block3\.17\.attn\.proj\.bias)|(backbone\.block3\.17\.attn\.sr\.weight)|(backbone\.block3\.17\.attn\.sr\.bias)|(backbone\.block3\.17\.mlp\.fc1\.weight)|(backbone\.block3\.17\.mlp\.fc1\.bias)|(backbone\.block3\.17\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.17\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.17\.mlp\.fc2\.weight)|(backbone\.block3\.17\.mlp\.fc2\.bias)|(backbone\.block3\.18\.attn\.q\.weight)|(backbone\.block3\.18\.attn\.q\.bias)|(backbone\.block3\.18\.attn\.kv\.weight)|(backbone\.block3\.18\.attn\.kv\.bias)|(backbone\.block3\.18\.attn\.proj\.weight)|(backbone\.block3\.18\.attn\.proj\.bias)|(backbone\.block3\.18\.attn\.sr\.weight)|(backbone\.block3\.18\.attn\.sr\.bias)|(backbone\.block3\.18\.mlp\.fc1\.weight)|(backbone\.block3\.18\.mlp\.fc1\.bias)|(backbone\.block3\.18\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.18\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.18\.mlp\.fc2\.weight)|(backbone\.block3\.18\.mlp\.fc2\.bias)|(backbone\.block3\.19\.attn\.q\.weight)|(backbone\.block3\.19\.attn\.q\.bias)|(backbone\.block3\.19\.attn\.kv\.weight)|(backbone\.block3\.19\.attn\.kv\.bias)|(backbone\.block3\.19\.attn\.proj\.weight)|(backbone\.block3\.19\.attn\.proj\.bias)|(backbone\.block3\.19\.attn\.sr\.weight)|(backbone\.block3\.19\.attn\.sr\.bias)|(backbone\.block3\.19\.mlp\.fc1\.weight)|(backbone\.block3\.19\.mlp\.fc1\.bias)|(backbone\.block3\.19\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.19\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.19\.mlp\.fc2\.weight)|(backbone\.block3\.19\.mlp\.fc2\.bias)|(backbone\.block3\.20\.attn\.q\.weight)|(backbone\.block3\.20\.attn\.q\.bias)|(backbone\.block3\.20\.attn\.kv\.weight)|(backbone\.block3\.20\.attn\.kv\.bias)|(backbone\.block3\.20\.attn\.proj\.weight)|(backbone\.block3\.20\.attn\.proj\.bias)|(backbone\.block3\.20\.attn\.sr\.weight)|(backbone\.block3\.20\.attn\.sr\.bias)|(backbone\.block3\.20\.mlp\.fc1\.weight)|(backbone\.block3\.20\.mlp\.fc1\.bias)|(backbone\.block3\.20\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.20\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.20\.mlp\.fc2\.weight)|(backbone\.block3\.20\.mlp\.fc2\.bias)|(backbone\.block3\.21\.attn\.q\.weight)|(backbone\.block3\.21\.attn\.q\.bias)|(backbone\.block3\.21\.attn\.kv\.weight)|(backbone\.block3\.21\.attn\.kv\.bias)|(backbone\.block3\.21\.attn\.proj\.weight)|(backbone\.block3\.21\.attn\.proj\.bias)|(backbone\.block3\.21\.attn\.sr\.weight)|(backbone\.block3\.21\.attn\.sr\.bias)|(backbone\.block3\.21\.mlp\.fc1\.weight)|(backbone\.block3\.21\.mlp\.fc1\.bias)|(backbone\.block3\.21\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.21\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.21\.mlp\.fc2\.weight)|(backbone\.block3\.21\.mlp\.fc2\.bias)|(backbone\.block3\.22\.attn\.q\.weight)|(backbone\.block3\.22\.attn\.q\.bias)|(backbone\.block3\.22\.attn\.kv\.weight)|(backbone\.block3\.22\.attn\.kv\.bias)|(backbone\.block3\.22\.attn\.proj\.weight)|(backbone\.block3\.22\.attn\.proj\.bias)|(backbone\.block3\.22\.attn\.sr\.weight)|(backbone\.block3\.22\.attn\.sr\.bias)|(backbone\.block3\.22\.mlp\.fc1\.weight)|(backbone\.block3\.22\.mlp\.fc1\.bias)|(backbone\.block3\.22\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.22\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.22\.mlp\.fc2\.weight)|(backbone\.block3\.22\.mlp\.fc2\.bias)|(backbone\.block3\.23\.attn\.q\.weight)|(backbone\.block3\.23\.attn\.q\.bias)|(backbone\.block3\.23\.attn\.kv\.weight)|(backbone\.block3\.23\.attn\.kv\.bias)|(backbone\.block3\.23\.attn\.proj\.weight)|(backbone\.block3\.23\.attn\.proj\.bias)|(backbone\.block3\.23\.attn\.sr\.weight)|(backbone\.block3\.23\.attn\.sr\.bias)|(backbone\.block3\.23\.mlp\.fc1\.weight)|(backbone\.block3\.23\.mlp\.fc1\.bias)|(backbone\.block3\.23\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.23\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.23\.mlp\.fc2\.weight)|(backbone\.block3\.23\.mlp\.fc2\.bias)|(backbone\.block3\.24\.attn\.q\.weight)|(backbone\.block3\.24\.attn\.q\.bias)|(backbone\.block3\.24\.attn\.kv\.weight)|(backbone\.block3\.24\.attn\.kv\.bias)|(backbone\.block3\.24\.attn\.proj\.weight)|(backbone\.block3\.24\.attn\.proj\.bias)|(backbone\.block3\.24\.attn\.sr\.weight)|(backbone\.block3\.24\.attn\.sr\.bias)|(backbone\.block3\.24\.mlp\.fc1\.weight)|(backbone\.block3\.24\.mlp\.fc1\.bias)|(backbone\.block3\.24\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.24\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.24\.mlp\.fc2\.weight)|(backbone\.block3\.24\.mlp\.fc2\.bias)|(backbone\.block3\.25\.attn\.q\.weight)|(backbone\.block3\.25\.attn\.q\.bias)|(backbone\.block3\.25\.attn\.kv\.weight)|(backbone\.block3\.25\.attn\.kv\.bias)|(backbone\.block3\.25\.attn\.proj\.weight)|(backbone\.block3\.25\.attn\.proj\.bias)|(backbone\.block3\.25\.attn\.sr\.weight)|(backbone\.block3\.25\.attn\.sr\.bias)|(backbone\.block3\.25\.mlp\.fc1\.weight)|(backbone\.block3\.25\.mlp\.fc1\.bias)|(backbone\.block3\.25\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.25\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.25\.mlp\.fc2\.weight)|(backbone\.block3\.25\.mlp\.fc2\.bias)|(backbone\.block3\.26\.attn\.q\.weight)|(backbone\.block3\.26\.attn\.q\.bias)|(backbone\.block3\.26\.attn\.kv\.weight)|(backbone\.block3\.26\.attn\.kv\.bias)|(backbone\.block3\.26\.attn\.proj\.weight)|(backbone\.block3\.26\.attn\.proj\.bias)|(backbone\.block3\.26\.attn\.sr\.weight)|(backbone\.block3\.26\.attn\.sr\.bias)|(backbone\.block3\.26\.mlp\.fc1\.weight)|(backbone\.block3\.26\.mlp\.fc1\.bias)|(backbone\.block3\.26\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.26\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.26\.mlp\.fc2\.weight)|(backbone\.block3\.26\.mlp\.fc2\.bias)|(backbone\.block3\.27\.attn\.q\.weight)|(backbone\.block3\.27\.attn\.q\.bias)|(backbone\.block3\.27\.attn\.kv\.weight)|(backbone\.block3\.27\.attn\.kv\.bias)|(backbone\.block3\.27\.attn\.proj\.weight)|(backbone\.block3\.27\.attn\.proj\.bias)|(backbone\.block3\.27\.attn\.sr\.weight)|(backbone\.block3\.27\.attn\.sr\.bias)|(backbone\.block3\.27\.mlp\.fc1\.weight)|(backbone\.block3\.27\.mlp\.fc1\.bias)|(backbone\.block3\.27\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.27\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.27\.mlp\.fc2\.weight)|(backbone\.block3\.27\.mlp\.fc2\.bias)|(backbone\.block3\.28\.attn\.q\.weight)|(backbone\.block3\.28\.attn\.q\.bias)|(backbone\.block3\.28\.attn\.kv\.weight)|(backbone\.block3\.28\.attn\.kv\.bias)|(backbone\.block3\.28\.attn\.proj\.weight)|(backbone\.block3\.28\.attn\.proj\.bias)|(backbone\.block3\.28\.attn\.sr\.weight)|(backbone\.block3\.28\.attn\.sr\.bias)|(backbone\.block3\.28\.mlp\.fc1\.weight)|(backbone\.block3\.28\.mlp\.fc1\.bias)|(backbone\.block3\.28\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.28\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.28\.mlp\.fc2\.weight)|(backbone\.block3\.28\.mlp\.fc2\.bias)|(backbone\.block3\.29\.attn\.q\.weight)|(backbone\.block3\.29\.attn\.q\.bias)|(backbone\.block3\.29\.attn\.kv\.weight)|(backbone\.block3\.29\.attn\.kv\.bias)|(backbone\.block3\.29\.attn\.proj\.weight)|(backbone\.block3\.29\.attn\.proj\.bias)|(backbone\.block3\.29\.attn\.sr\.weight)|(backbone\.block3\.29\.attn\.sr\.bias)|(backbone\.block3\.29\.mlp\.fc1\.weight)|(backbone\.block3\.29\.mlp\.fc1\.bias)|(backbone\.block3\.29\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.29\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.29\.mlp\.fc2\.weight)|(backbone\.block3\.29\.mlp\.fc2\.bias)|(backbone\.block3\.30\.attn\.q\.weight)|(backbone\.block3\.30\.attn\.q\.bias)|(backbone\.block3\.30\.attn\.kv\.weight)|(backbone\.block3\.30\.attn\.kv\.bias)|(backbone\.block3\.30\.attn\.proj\.weight)|(backbone\.block3\.30\.attn\.proj\.bias)|(backbone\.block3\.30\.attn\.sr\.weight)|(backbone\.block3\.30\.attn\.sr\.bias)|(backbone\.block3\.30\.mlp\.fc1\.weight)|(backbone\.block3\.30\.mlp\.fc1\.bias)|(backbone\.block3\.30\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.30\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.30\.mlp\.fc2\.weight)|(backbone\.block3\.30\.mlp\.fc2\.bias)|(backbone\.block3\.31\.attn\.q\.weight)|(backbone\.block3\.31\.attn\.q\.bias)|(backbone\.block3\.31\.attn\.kv\.weight)|(backbone\.block3\.31\.attn\.kv\.bias)|(backbone\.block3\.31\.attn\.proj\.weight)|(backbone\.block3\.31\.attn\.proj\.bias)|(backbone\.block3\.31\.attn\.sr\.weight)|(backbone\.block3\.31\.attn\.sr\.bias)|(backbone\.block3\.31\.mlp\.fc1\.weight)|(backbone\.block3\.31\.mlp\.fc1\.bias)|(backbone\.block3\.31\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.31\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.31\.mlp\.fc2\.weight)|(backbone\.block3\.31\.mlp\.fc2\.bias)|(backbone\.block3\.32\.attn\.q\.weight)|(backbone\.block3\.32\.attn\.q\.bias)|(backbone\.block3\.32\.attn\.kv\.weight)|(backbone\.block3\.32\.attn\.kv\.bias)|(backbone\.block3\.32\.attn\.proj\.weight)|(backbone\.block3\.32\.attn\.proj\.bias)|(backbone\.block3\.32\.attn\.sr\.weight)|(backbone\.block3\.32\.attn\.sr\.bias)|(backbone\.block3\.32\.mlp\.fc1\.weight)|(backbone\.block3\.32\.mlp\.fc1\.bias)|(backbone\.block3\.32\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.32\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.32\.mlp\.fc2\.weight)|(backbone\.block3\.32\.mlp\.fc2\.bias)|(backbone\.block3\.33\.attn\.q\.weight)|(backbone\.block3\.33\.attn\.q\.bias)|(backbone\.block3\.33\.attn\.kv\.weight)|(backbone\.block3\.33\.attn\.kv\.bias)|(backbone\.block3\.33\.attn\.proj\.weight)|(backbone\.block3\.33\.attn\.proj\.bias)|(backbone\.block3\.33\.attn\.sr\.weight)|(backbone\.block3\.33\.attn\.sr\.bias)|(backbone\.block3\.33\.mlp\.fc1\.weight)|(backbone\.block3\.33\.mlp\.fc1\.bias)|(backbone\.block3\.33\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.33\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.33\.mlp\.fc2\.weight)|(backbone\.block3\.33\.mlp\.fc2\.bias)|(backbone\.block3\.34\.attn\.q\.weight)|(backbone\.block3\.34\.attn\.q\.bias)|(backbone\.block3\.34\.attn\.kv\.weight)|(backbone\.block3\.34\.attn\.kv\.bias)|(backbone\.block3\.34\.attn\.proj\.weight)|(backbone\.block3\.34\.attn\.proj\.bias)|(backbone\.block3\.34\.attn\.sr\.weight)|(backbone\.block3\.34\.attn\.sr\.bias)|(backbone\.block3\.34\.mlp\.fc1\.weight)|(backbone\.block3\.34\.mlp\.fc1\.bias)|(backbone\.block3\.34\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.34\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.34\.mlp\.fc2\.weight)|(backbone\.block3\.34\.mlp\.fc2\.bias)|(backbone\.block3\.35\.attn\.q\.weight)|(backbone\.block3\.35\.attn\.q\.bias)|(backbone\.block3\.35\.attn\.kv\.weight)|(backbone\.block3\.35\.attn\.kv\.bias)|(backbone\.block3\.35\.attn\.proj\.weight)|(backbone\.block3\.35\.attn\.proj\.bias)|(backbone\.block3\.35\.attn\.sr\.weight)|(backbone\.block3\.35\.attn\.sr\.bias)|(backbone\.block3\.35\.mlp\.fc1\.weight)|(backbone\.block3\.35\.mlp\.fc1\.bias)|(backbone\.block3\.35\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.35\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.35\.mlp\.fc2\.weight)|(backbone\.block3\.35\.mlp\.fc2\.bias)|(backbone\.block3\.36\.attn\.q\.weight)|(backbone\.block3\.36\.attn\.q\.bias)|(backbone\.block3\.36\.attn\.kv\.weight)|(backbone\.block3\.36\.attn\.kv\.bias)|(backbone\.block3\.36\.attn\.proj\.weight)|(backbone\.block3\.36\.attn\.proj\.bias)|(backbone\.block3\.36\.attn\.sr\.weight)|(backbone\.block3\.36\.attn\.sr\.bias)|(backbone\.block3\.36\.mlp\.fc1\.weight)|(backbone\.block3\.36\.mlp\.fc1\.bias)|(backbone\.block3\.36\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.36\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.36\.mlp\.fc2\.weight)|(backbone\.block3\.36\.mlp\.fc2\.bias)|(backbone\.block3\.37\.attn\.q\.weight)|(backbone\.block3\.37\.attn\.q\.bias)|(backbone\.block3\.37\.attn\.kv\.weight)|(backbone\.block3\.37\.attn\.kv\.bias)|(backbone\.block3\.37\.attn\.proj\.weight)|(backbone\.block3\.37\.attn\.proj\.bias)|(backbone\.block3\.37\.attn\.sr\.weight)|(backbone\.block3\.37\.attn\.sr\.bias)|(backbone\.block3\.37\.mlp\.fc1\.weight)|(backbone\.block3\.37\.mlp\.fc1\.bias)|(backbone\.block3\.37\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.37\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.37\.mlp\.fc2\.weight)|(backbone\.block3\.37\.mlp\.fc2\.bias)|(backbone\.block3\.38\.attn\.q\.weight)|(backbone\.block3\.38\.attn\.q\.bias)|(backbone\.block3\.38\.attn\.kv\.weight)|(backbone\.block3\.38\.attn\.kv\.bias)|(backbone\.block3\.38\.attn\.proj\.weight)|(backbone\.block3\.38\.attn\.proj\.bias)|(backbone\.block3\.38\.attn\.sr\.weight)|(backbone\.block3\.38\.attn\.sr\.bias)|(backbone\.block3\.38\.mlp\.fc1\.weight)|(backbone\.block3\.38\.mlp\.fc1\.bias)|(backbone\.block3\.38\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.38\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.38\.mlp\.fc2\.weight)|(backbone\.block3\.38\.mlp\.fc2\.bias)|(backbone\.block3\.39\.attn\.q\.weight)|(backbone\.block3\.39\.attn\.q\.bias)|(backbone\.block3\.39\.attn\.kv\.weight)|(backbone\.block3\.39\.attn\.kv\.bias)|(backbone\.block3\.39\.attn\.proj\.weight)|(backbone\.block3\.39\.attn\.proj\.bias)|(backbone\.block3\.39\.attn\.sr\.weight)|(backbone\.block3\.39\.attn\.sr\.bias)|(backbone\.block3\.39\.mlp\.fc1\.weight)|(backbone\.block3\.39\.mlp\.fc1\.bias)|(backbone\.block3\.39\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block3\.39\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block3\.39\.mlp\.fc2\.weight)|(backbone\.block3\.39\.mlp\.fc2\.bias)|(backbone\.block4\.0\.attn\.q\.weight)|(backbone\.block4\.0\.attn\.q\.bias)|(backbone\.block4\.0\.attn\.kv\.weight)|(backbone\.block4\.0\.attn\.kv\.bias)|(backbone\.block4\.0\.attn\.proj\.weight)|(backbone\.block4\.0\.attn\.proj\.bias)|(backbone\.block4\.0\.mlp\.fc1\.weight)|(backbone\.block4\.0\.mlp\.fc1\.bias)|(backbone\.block4\.0\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block4\.0\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block4\.0\.mlp\.fc2\.weight)|(backbone\.block4\.0\.mlp\.fc2\.bias)|(backbone\.block4\.1\.attn\.q\.weight)|(backbone\.block4\.1\.attn\.q\.bias)|(backbone\.block4\.1\.attn\.kv\.weight)|(backbone\.block4\.1\.attn\.kv\.bias)|(backbone\.block4\.1\.attn\.proj\.weight)|(backbone\.block4\.1\.attn\.proj\.bias)|(backbone\.block4\.1\.mlp\.fc1\.weight)|(backbone\.block4\.1\.mlp\.fc1\.bias)|(backbone\.block4\.1\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block4\.1\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block4\.1\.mlp\.fc2\.weight)|(backbone\.block4\.1\.mlp\.fc2\.bias)|(backbone\.block4\.2\.attn\.q\.weight)|(backbone\.block4\.2\.attn\.q\.bias)|(backbone\.block4\.2\.attn\.kv\.weight)|(backbone\.block4\.2\.attn\.kv\.bias)|(backbone\.block4\.2\.attn\.proj\.weight)|(backbone\.block4\.2\.attn\.proj\.bias)|(backbone\.block4\.2\.mlp\.fc1\.weight)|(backbone\.block4\.2\.mlp\.fc1\.bias)|(backbone\.block4\.2\.mlp\.dwconv\.dwconv\.weight)|(backbone\.block4\.2\.mlp\.dwconv\.dwconv\.bias)|(backbone\.block4\.2\.mlp\.fc2\.weight)|(backbone\.block4\.2\.mlp\.fc2\.bias)')

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