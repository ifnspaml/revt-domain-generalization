from builtins import NotImplementedError

import torch
import argparse
import re



def load_checkpoints(checkpoint_files, cpu_only=False, warnings=True):
    assert isinstance(checkpoint_files, list) and len(checkpoint_files) > 1 \
        , "checkpoint_files must be a non empty list with len greater 1"

    # load checkpoints
    for i, e in enumerate(checkpoint_files):
        if isinstance(e, str):
            if cpu_only:
                checkpoint_files[i] = torch.load(e, map_location=torch.device('cpu'))
            else:
                checkpoint_files[i] = torch.load(e)

    if warnings:
        # check if all checkpoints are from the same model
        for i, e in enumerate(checkpoint_files):
            assert e["state_dict"].keys() == checkpoint_files[0]["state_dict"].keys() \
                , f'The {i}. checkpoint file does not match with the first checkpoint file'

    return checkpoint_files


def get_distance_regex(checkpoints, weights_filter=".+", l = [], use_l2_norm=False, use_msd=False):
    all_pairs_idx = [(idx_a, idx_b) for idx_a, _ in enumerate(checkpoints) for idx_b in
                     list(range(0, len(checkpoints)))[idx_a + 1:]]

    #print(weights_filter)
    for key_name in checkpoints[0]["state_dict"].keys():
        if isinstance(checkpoints[0]["state_dict"][key_name], torch.FloatTensor)\
                and re.match(weights_filter, key_name):
            for i, e in enumerate(checkpoints):
                if len(l)-1 < i:
                    l.append(torch.tensor([]))
                l[i] = torch.hstack((l[i],e["state_dict"][key_name].flatten()))
    if use_l2_norm:
        def l2Norm(a,b):
            return torch.norm(a-b)
        f = l2Norm
    elif use_msd:
        def msd(a,b):
            return torch.norm(a-b) / a.flatten().shape[0]
        f = msd
    else:
        f = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    output = []
    for a, b in all_pairs_idx:
        output.append(f(l[a], l[b]))
    output = torch.mean(torch.stack(output), dim=0)
    #print(l[0].shape[0])
    return output, l

def parse_args():
    parser = argparse.ArgumentParser(
        description='shows distance of different blocks')
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        help='path to the checkpoint files')
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='when set uses only cpu')
    parser.add_argument(
        '--l2-norm',
        action='store_true',
        help='use L2 Norm instead of cosine similarity')
    parser.add_argument(
        '--msd',
        action='store_true',
        help='use mean squared displacement instead of cosine similarity')
    parser.add_argument(
        '--model-type',
        default='segformerb5',
        help='segformer or resnet50')
    parser.add_argument(
        '--no-warnings',
        action='store_true',
        help='ignore errors when the checkpoints do not match exactly')
    parser.add_argument(
        '--regex',
        type=str,
        nargs='+',
        help='')
    args = parser.parse_args()
    return args

def get_block_regexes_segformerb5(depth = [3, 6, 40, 3]):
    regexes = []
    block_names = []
    for i, d in enumerate(depth):
        for j in range(-1, d+1):
            regex = f"backbone\\.block{i+1}\\.{j}\\..*"
            block_name = f"Block{i+1}.{j}"
            if j == -1:
                regex = f"backbone.*patch_embed{i+1}\\..*"
                block_name = f"PatchEmbed{i + 1}"
            elif j == d:
                regex = f"backbone\\.norm{i + 1}\\..*"
                block_name = f"Norm{i + 1}"

            regexes.append(regex)
            block_names.append(block_name)
    return regexes, block_names


def get_block_regexes_all():
    regexes = [".*"]
    return regexes, ["complete_model"]


def get_block_regexes_backbone():
    regexes = ["backbone.*"]
    return regexes, ["backbone_only"]


def get_block_regexes_resnet(depth):
    regexes = ["backbone\\.stem\\..*"]
    block_names = ["Stem"]
    for i, d in enumerate(depth):
        for j in range(0, d):
            regex = f"backbone\\.layer{i+1}\\.{j}\\..*"
            block_name = f"Block{i+1}.{j}"
            regexes.append(regex)
            block_names.append(block_name)
    print(regexes)
    return regexes, block_names


def get_block_regexes_resnet50(depth=[3, 4, 6, 3]):
    return get_block_regexes_resnet(depth)


def get_block_regexes_resnet101(depth=[3, 4, 23, 3]):
    return get_block_regexes_resnet(depth)


def get_cosine_similarity_blockwise(checkpoints, cpu_only=False, model_type="segformerb5"):
    assert isinstance(checkpoints, list) and len(checkpoints), "please give exact 2 checkpoints"

    if model_type.lower() == "segformerb5":
        regexes, block_names = get_block_regexes_segformerb5()
    elif model_type.lower() == "resnet50":
        regexes, block_names = get_block_regexes_resnet50()
    elif model_type.lower() == "resnet101":
        regexes, block_names = get_block_regexes_resnet101()
    elif model_type.lower() == "all":
        regexes, block_names = get_block_regexes_all()
    elif model_type.lower() == "backbone":
        regexes, block_names = get_block_regexes_backbone()
    else:
        raise NotImplementedError(f"Model type: {model_type.lower()} is not implemented yet")
    checkpoints = load_checkpoints(checkpoints, cpu_only=cpu_only)

    cosine_similarity = []
    for i, r in enumerate(regexes):
        cosine_similarity, _ = get_distance_regex(checkpoints, weights_filter=r)



def main():
    args = parse_args()
    assert isinstance(args.checkpoints, list) and len(args.checkpoints), "please give exact 2 checkpoints"

    if args.regex is not None and len(args.regex) > 0:
        regexes, block_names = args.regex, ["regex"]
    elif args.model_type.lower() == "segformerb5":
        regexes, block_names = get_block_regexes_segformerb5()
    elif args.model_type.lower() == "resnet50":
        regexes, block_names = get_block_regexes_resnet50()
    elif args.model_type.lower() == "resnet101":
        regexes, block_names = get_block_regexes_resnet101()
    elif args.model_type.lower() == "all":
        regexes, block_names = get_block_regexes_all()
    elif args.model_type.lower() == "backbone":
        regexes, block_names = get_block_regexes_backbone()
    else:
        raise NotImplementedError(f"Model type: {args.model_type.lower()} is not implemented yet")
    checkpoints = load_checkpoints(args.checkpoints, cpu_only=args.cpu_only, warnings=(not args.no_warnings))
    regex_pyramid = "("
    l = []
    for i, r in enumerate(regexes):
        regex_pyramid = regex_pyramid[0:-1] + "|" + r + ")"

        #l = []
        #print(f"{block_names[i]}: ")
        cosine_similarity, l = get_distance_regex(checkpoints, weights_filter=r, l=l, use_l2_norm=args.l2_norm, use_msd=args.msd)
        #cosine_similarity = round((1 - float(cosine_similarity))*1000, 2)
        #print(f"{i},{cosine_similarity},{block_names[i]}")
        print(f"{cosine_similarity}")

if __name__ == '__main__':
    main()