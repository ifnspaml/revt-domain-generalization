# transformer-domain-generalization

This repository provides the official PyTorch implementation of the following paper:
> [**A Re-Parameterized Vision Transformer (ReVT) for Domain-Generalized Semantic Segmentation**](https://arxiv.org/abs/2308.13331)<br>

> **Abstract:** 
*The task of semantic segmentation requires a model to assign semantic labels to each pixel of an image. However, the performance of such models degrades when deployed in an unseen domain with different data distributions compared to the training domain. We present a new augmentation-driven approach to domain generalization for semantic segmentation using a re-parameterized vision transformer (ReVT) with weight averaging of multiple models after training. We evaluate our approach on several benchmark datasets and achieve state-of-the-art mIoU performance of 47.3% (prior art: 46.3%) for small models and of 50.1% (prior art: 47.8%) for midsized models on commonly used benchmark datasets. At the same time, our method requires fewer parameters and reaches a higher frame rate than the best prior art. It is also easy to implement and, unlike network ensembles, does not add any computational complexity during inference.*<br>

## Getting Started

Our code is heavily derived from [SegFormer](https://github.com/NVlabs/SegFormer/) (NeurIPS 2021). If you use this code in your research, please also cite their work.

### Requirements

We ran our experiments on a GPU cluster driven by slurm. Therefore, all our experiment scripts are slurm scripts. Nevertheless, it should be possible to reproduce our results without slurm by only small changes.

Other requirements are:
- Nvidia GPU with CUDA
- Install [Anaconda](https://www.anaconda.com/)
- GCC 11.2.0

### Installation

Clone this repository into <User_Home_dir>/work/.
```
cd ~/
mkdir ./work
cd work
git clone revt-domain-generalization
cd revt-domain-generalization
```

Install all packages in the requirements.txt by calling the following command:
```
conda create --name transformer-domain-generalization --file ./requirements.txt
```

### Datasets
We used the dataset structure from [SegFormer](https://github.com/NVlabs/SegFormer/), which is based on [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).
All datasets are implemented as a `Custom Datasets`, which have the following structure:

```none
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val

```

For more detailed information please refer to [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) on how to structure the datasets.

We use the following datasets:

- Synthetic Images:
  - [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) 
  - [SYNTHIA](https://synthia-dataset.net/downloads/)

- Real-World Images:
  - [Cityscapes](https://www.cityscapes-dataset.com/downloads/)
  - [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5)
  - [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
  - [KITTI](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)
  - [ACDC](https://acdc.vision.ee.ethz.ch/download)

- PixMix Images:
  - [fractals](https://drive.google.com/file/d/1qC2gIUx9ARU7zhgI4IwGD3YcFhm8J4cA/view?usp=sharing) from [PixMix](https://github.com/andyzoujm/pixmix#user-content-mixing-set) Repository

### Download Pretrained Weights on Imagenet
We used the pretrained weights offered by the [SegFormer](https://github.com/NVlabs/SegFormer/) repository.
Please download the weights and put them into the folder: ```./pretrained```

## Usage

### Reproduce Experiments with Provided Slurm Scripts:
All included slurm-scripts work out of the box by calling:
```
cd ~/work/revt-domain-generalization/<location-of-slurm-script>
sbatch ./<name-of-slurm-script>
```

The slurm scripts are all located in ``./slurm_scripts_runs/``.

The structure is as follows:
```
├── slurm_scripts_runs
│   ├── templates
│   │   ├── test.sh
│   │   ├── train.sh
│   │   ├── test_ensemble.sh
│   │   ├── test_reparam.sh
│   ├── <Model_Type1>
│   │   ├── <Train-DatasetA>
│   │   │   ├── test
│   │   │   │   ├── standard
│   │   │   │   │   ├── <Test_Method_I>
│   │   │   │   │   ├── ...
│   │   │   │   ├── ensemble
│   │   │   │   │   ├── Ensemble(1,1,1)
│   │   │   │   │   ├── Ensemble(1,4,6)
│   │   │   │   │   ├── ...
│   │   │   │   ├── re_param
│   │   │   │   │   ├── ReVT(1,1,1)
│   │   │   │   │   ├── ReVT(1,4,6)
│   │   │   │   │   ├── ...
│   │   │   ├── train
│   │   │   │   ├── <Method_I>
│   │   │   │   ├── <Method_II>
│   │   │   │   ├── ...
│   │   ├── <Train-DatasetB>
│   │   │   ├── ...
│   │   ├── ...
│   ├── <Model_Type2>
│   │   ├── <Train-DatasetA>
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...

```
For example:
- An experiment with the SegFormer architecture with the MiT-B5 encoder, trained on GTA train_split with PixMix as augmentation is in the folder: ``./slurm_script_runs/SegFormerB5/gta_train_split/train/pixmix``

- The corresponding ReVT experiment script (ReVT {PixMix, PixMix, PixMix}) is located under: ``slurm_script_runs/SegFormerB5/gta_train_split/test/re_param/pixmix``


### Use Reparameterization Script
The Script for applying ReVT on checkpoints is implemented in a generic way and should therefore work on most torch models.
The only requirement is, that the checkpoints are at the top level dict objects, which have the key "state_dict".
All weights in the "state_dict" that match the specified regex are averaged.

You can merge an arbitrary number of checkpoints by:
```
python ./tools/model_reparameterization <Destination_File> --checkpoints <list of checkpoint files> --cpu-only --weights-filter <Regex>
```
Example:
The following example applies ReVT to three baseline models (A, B, C), to create a combined checkpoint file.
All weights in the state_dict that match the regex "backbone.*" are averaged. All others weights are taken over by the first given checkpointfile (A)
```
python ./tools/model_reparameterization.py ./work_dir/ReVT_B5/BaselineABC.pth\
         --checkpoints\
            ./work_dir/SegFormerB5/gta_dev/BaselineA.pth\
            ./work_dir/SegFormerB5/gta_dev/BaselineB.pth\
            ./work_dir/SegFormerB5/gta_dev/BaselineC.pth\
         --cpu-only --weights-filter "backbone.*"
```

### Trained weights
The weights can be obtained from the following links:

| Model Type   | Train Dataset       | Method       | mIoU Cityscapes | mIoU BDD | Link                                                                                 |
|--------------|---------------------|--------------|-----------------|----------|--------------------------------------------------------------------------------------|
| Segformer B2 | GTA train split     | Baseline     | 41.73%          | 38.77%   | will be added soon                                      |
|              |                     | ReVT {1,4,6} | 46.27%          | 43.29%   | will be added soon                                      |
|              |                     | ReVT {4,5,6} | 45.55%          | 43.43%   | will be added soon                                      |
|              | SYNTHIA train split | Baseline     | 39.71%          | 29.76%   | will be added soon                                      |
|              |                     | ReVT {1,4,6} | 40.91%          | 34.53%   | will be added soon                                      |
|              |                     | ReVT {4,5,6} | 41.09%          | 35.18%   | will be added soon                                      |
| Segformer B3 | GTA train split     | Baseline     | 43.92%          | 42.96%   | will be added soon                                      |
|              |                     | ReVT {1,4,6} | 48.33%          | 48.17%   | will be added soon                                      |
|              |                     | ReVT {4,5,6} | 47.95%          | 48.26%   | will be added soon                                      |
|              | SYNTHIA train split | Baseline     | 42.43%          | 33.33%   | will be added soon                                      |
|              |                     | ReVT {1,4,6} | 44.97%          | 38.65%   | will be added soon                                      |
|              |                     | ReVT {4,5,6} | 45.26%          | 38.73%   | will be added soon                                      |
| Segformer B5 | GTA train split     | Baseline     | 45.31%          | 43.32%   | https://drive.google.com/drive/folders/1clnJptm58PrLEGooB2cyUpbdqU0_d8Oq?usp=sharing |
|              |                     | ReVT {1,4,6} | 49.96%          | 48.01%   | https://drive.google.com/drive/folders/1MuvBVyNveIxs5v011AdIp_i4dp5HU4sd?usp=sharing |
|              |                     | ReVT {4,5,6} | 49.55%          | 48.11%   | https://drive.google.com/drive/folders/1hhmNDwGRAd_F9Rb198L2pPaDn0rlf3Ec?usp=sharing |
|              | SYNTHIA train split | Baseline     | 45.07%          | 35.19%   | will be added soon                                      |
|              |                     | ReVT {1,4,6} | 46.28%          | 40.30%   | will be added soon                                      |
|              |                     | ReVT {4,5,6} | 45.08%          | 39.62%   | will be added soon                                      |
