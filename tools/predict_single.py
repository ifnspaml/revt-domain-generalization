import sys, os
sys.path.append(os.getcwd())

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
import math
from mmseg.models import build_segmentor
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


def predict(model, input):

    prediction = model.module.encode_decode(input['img'][0], input['img_metas'])
    print(prediction)
    return prediction

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('image', help='name of the image file')
    parser.add_argument(
        'checkpoint', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    args = parser.parse_args()
    return args

def load_image(model, img, cfg):
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    return data


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    #load image
    input = load_image(model, args.image, cfg)

    prediction = predict(model, input)

if __name__ == '__main__':
    main()

