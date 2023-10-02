import sys, os
sys.path.append(os.getcwd())

import argparse
from mmseg.models import build_segmentor
from mmcv.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()
    return args

def model_summary(model):
    print(model)
    print("Complete: ", sum(p.numel() for p in model.parameters()))
    #print("Backbone: ", sum(p.numel() for p in model.backbone.parameters()))
    #print("Head: ", sum(p.numel() for p in model.decode_head.parameters()))

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    #checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model_summary(model)


if __name__ == '__main__':
    main()

