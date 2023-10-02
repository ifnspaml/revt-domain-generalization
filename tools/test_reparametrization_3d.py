import sys, os
sys.path.append(os.getcwd())

import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, load_state_dict
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from model_reparametrization import model_reparametrization
from IPython import embed
import numpy as np
import os.path

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('outfile', type=str, help='name of the output config file')
    parser.add_argument('--resume-from', default=None,
                        type=str, help='if set, the calculation is resumed for this outfile')
    parser.add_argument('--checkpoints',
        type=str,
        nargs='+',
        help='path to the checkpoint files. This Must be exact 3 checkpoints')
    parser.add_argument(
        '--weights-filter',
        type=str,
        default=".+",
        help='regex for filter the weight-types. Only matching model weights are re-parametrized')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--reverse-order', action='store_true', help='begin at the end')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    print("init distributed")
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print("create Dataloader")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    assert isinstance(args.checkpoints, list) and len(args.checkpoints)==3, "Please give exact 3 checkpoint files"
    checkpoints = args.checkpoints
    print("build model")
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    result_dict = []
    beta_last = 0
    gamma_last = 0
    if args.resume_from is not None and os.path.isfile(args.resume_from):
        print(f"Resume from {args.resume_from}")
        result_dict = np.load(args.resume_from, allow_pickle=True).tolist()
        _, beta_last, gamma_last, _ = result_dict[-1]
        beta_last = int(round(beta_last * 18))
        gamma_last = int(round(gamma_last * 18))

    beta_range = range(0, 19)
    gamma_range = range(0, 19)
    if args.reverse_order:
        beta_range = range(18,-1, -1)
        gamma_range = range(18,-1, -1)
    for nominator_beta in beta_range:
        for nominator_gamma in gamma_range:
            if nominator_beta + nominator_gamma > 18:
                continue

            if 100*nominator_beta + nominator_gamma <= 100*beta_last + gamma_last and not args.reverse_order:
                continue
            elif 100*nominator_beta + nominator_gamma >= 100*beta_last + gamma_last and args.reverse_order:
                continue
            print(f"{100*nominator_beta + nominator_gamma} < {100*beta_last + gamma_last}")
            alpha = (18-nominator_beta-nominator_gamma)/18
            beta = nominator_beta/18
            gamma = nominator_gamma/18

            mIoU_values = []
            print(f"evaluate alpha: {alpha}, beta: {beta}, gamma: {gamma}")
            alpha_beta_gamma = [alpha, beta, gamma]
            for i in range(3):
                print(f"    with head {['A', 'B', 'C'][i]}")
                checkpoint = create_new_checkpoint([checkpoints[i%3], checkpoints[(i+1)%3], checkpoints[(i+2)%3]]
                                      , alpha_beta_gamma[i%3], alpha_beta_gamma[(i+1)%3], alpha_beta_gamma[(i+2)%3], args)
                metrics = evaluate_checkpoint(cfg, args, distributed, data_loader, dataset, checkpoint)
                mIoU_value = metrics["mIoU"]
                mIoU_values.append(mIoU_value)
            result_dict.append([alpha, beta, gamma, mIoU_values])
            np.save(args.outfile, np.array(result_dict))

def create_new_checkpoint(checkpoints, alpha, beta, gamma, args):
    return model_reparametrization(checkpoints, weights_filter=args.weights_filter, cpu_only=False, factors=[alpha, beta, gamma])
    #torch.save(new_checkpoint, "./work_dirs/model_reparametrization/tmp_3d.pth")

def evaluate_checkpoint(cfg, args, distributed, data_loader, dataset, checkpoint):
    print("load checkpoint")
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_state_dict(model, checkpoint["state_dict"], False, None)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    print("prepare model for distributed training")
    if not distributed:
        model = MMDataParallel(model.cuda(), device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, progress_bar=False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test, progress_bar=False)

    print("start test")
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            return dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
