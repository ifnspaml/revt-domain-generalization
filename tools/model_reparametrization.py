import torch
import copy
import argparse
from progress.bar import Bar
import re


def model_reparametrization(checkpoint_files, factors=None, cpu_only=False, weights_filter=".+", operation="mean", alpha=0.9, debug=True, warnings=True):
    assert isinstance(checkpoint_files, list) and len(checkpoint_files) > 0\
        , "checkpoint_files must be a non empty list"

    # load checkpoints
    for i, e in enumerate(checkpoint_files):
        if isinstance(e, str):
            if cpu_only:
                checkpoint_files[i] = torch.load(e, map_location=torch.device('cpu'))
            else:
                checkpoint_files[i] = torch.load(e)

    # check if all checkpoints are from the same model
    if warnings:
        for i, e in enumerate(checkpoint_files):
            assert e["state_dict"].keys() == checkpoint_files[0]["state_dict"].keys()\
                , f'The {i}. checkpoint file does not match with the first checkpoint file'

    res = copy.deepcopy(checkpoint_files[0])
    max_len = len([key_name for key_name in checkpoint_files[0]["state_dict"].keys() if \
               isinstance(checkpoint_files[0]["state_dict"][key_name], torch.FloatTensor)\
               and re.match(weights_filter, key_name)])

    #print(checkpoint_files[0]["state_dict"].keys())

    matched_key_list = []
    with Bar('Processing', max=max_len) as bar:
        for key_name in checkpoint_files[0]["state_dict"].keys():
            if isinstance(checkpoint_files[0]["state_dict"][key_name], torch.FloatTensor)\
                    and re.match(weights_filter, key_name):
                matched_key_list.append(key_name)
                if factors==None:
                    l = []
                    for i, e in enumerate(checkpoint_files):
                        l.append(e["state_dict"][key_name])
                    if operation == "mean":
                        res["state_dict"][key_name] = torch.mean(torch.stack(l), dim=0)
                    elif operation == "min":
                        res["state_dict"][key_name] = torch.min(torch.stack(l), dim=0).values
                    elif operation == "compression":
                        if alpha == -1:
                            print("harmonic mean")
                            l_pow = []
                            epsillon = 1e-7
                            for e in l:
                                l_pow.append(torch.where(e == 0, torch.pow(e+epsillon, alpha), torch.pow(e, alpha)))
                            mean = torch.mean(torch.stack(l_pow), dim=0)
                            res["state_dict"][key_name] = torch.where(mean == 0, torch.pow(mean+epsillon, 1/alpha), torch.pow(mean, 1/alpha))
                        else:
                            l_pow = []
                            for e in l:
                                l_pow.append(torch.where(e < 0, -1 * torch.pow(-e, alpha), torch.pow(e, alpha)))
                            mean = torch.mean(torch.stack(l_pow), dim=0)
                            res["state_dict"][key_name] = torch.where(mean < 0, -1 * torch.pow(-mean, 1/alpha), torch.pow(mean, 1/alpha))
                    elif operation == "trimean":
                        median = torch.median(torch.stack(l), dim=0).values
                        lower_half_list = []
                        upper_half_list = []
                        for l_i in l:
                            lower_half_list.append(torch.where(median<l_i, torch.nan, l_i))
                            upper_half_list.append(torch.where(median>l_i, torch.nan, l_i))
                        first_quartile = torch.nanmedian(torch.stack(lower_half_list)).values
                        third_quartile = torch.nanmedian(torch.stack(upper_half_list)).values
                        trimean = (first_quartile + 2*median)
                        res["state_dict"][key_name] = trimean
                    else:
                        raise Exception(f"Operation '{operation}' is not supported")
                else:
                    assert len(factors) == len(checkpoint_files)\
                        , "the length of the factors list is not equal to the length of the checkpoint_files list"
                    # normalize factors
                    factors = (torch.tensor(factors, dtype=torch.float64)/torch.sum(torch.tensor(factors, dtype=torch.float64))).tolist()
                    l = []
                    for i, e in enumerate(checkpoint_files):
                        l.append(factors[i] * e["state_dict"][key_name])
                    res["state_dict"][key_name] = torch.sum(torch.stack(l), dim=0)
                bar.next()

    if debug:
        print(f"The regex {weights_filter} matched for the layers: {matched_key_list}")
    return res

def parse_args():
    parser = argparse.ArgumentParser(
        description='reparametrization of checkpoint files')
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        help='path to the checkpoint files')
    parser.add_argument(
        '--factors',
        type=float,
        nargs='+',
        default=None,
        help='The factors for calulating the mean')
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='when set uses only cpu')
    parser.add_argument(
        '--no-warnings',
        action='store_true',
        help='ignore errors when the checkpoints do not match exactly')
    parser.add_argument(
        '--weights-filter',
        type=str,
        default=".+",
        help='regex for filter the weight-types. Only matching model weights are re-parametrized')
    parser.add_argument(
        '--operation',
        type=str,
        default="mean",
        help='mean or min. Defines which operation is used to merge the weights')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.9,
        help='Only importnat in combination with operator compression')

    parser.add_argument('outfile', type=str, help='name of the output config file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    checkpoints = args.checkpoints if isinstance(args.checkpoints, list) else [args.checkpoints]

    factors = None if args.factors is None or len(args.factors) == 0 else args.factors
    new_checkpoint = model_reparametrization(checkpoints, factors=factors, weights_filter=args.weights_filter, cpu_only=args.cpu_only, operation=args.operation, alpha=args.alpha, warnings=(not args.no_warnings))
    torch.save(new_checkpoint, args.outfile)

if __name__ == '__main__':
    main()