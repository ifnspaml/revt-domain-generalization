import torch
import mmcv
from mmseg.utils.running_mean_std import RunningStats
from mmseg.ops import resize
import os
import re
import copy


def get_feature_mean(model, data_loader, progress_bar=True):
    model.eval()
    dataset = data_loader.dataset
    stats = RunningStats()
    if progress_bar:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if i == 0:
            print(data["img"][0].shape)
            if len(data["img"][0].shape) == 4:
                _, c, h, w = data["img"][0].shape
            else:  # 3
                c, h, w = data["img"][0].shape

        if data["img"][0].dim() == 3:
            img = data["img"][0][None,].cuda()
        else:
            img = data["img"][0].cuda()
        print(img.shape)

        with torch.no_grad():
            img = resize(img, size=(h, w), mode='bilinear', align_corners=False)
            batched_features = model.module.extract_feat(img)

            batch_size = img.size(0)
            for n in range(batch_size):
                if isinstance(batched_features, list):
                    print(batched_features[0].shape)
                    features = batched_features[0][n].flatten()
                    for level in batched_features[1:]:
                        features = torch.hstack([features.flatten(), level[n].flatten()])
                else:
                    features = batched_features[n].flatten()

                stats.push(features)

        if progress_bar:
            batch_size = img.size(0)
            for _ in range(batch_size):
                prog_bar.update()

    return stats.mean(), stats.standard_deviation()


def get_class_weight_relevance(model, data_loader, class_idx, progress_bar=True):
    # pass
    # class_idx

    model.train()
    dataset = data_loader.dataset
    stats = RunningStats()
    if progress_bar:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):

        """if data["img"][0].dim() == 3:
            img = data["img"][0][None,].cuda()
        else:
            img = data["img"][0].cuda()
        print(img.shape)"""

        data["gt_semantic_seg"][0] = torch.where(data["gt_semantic_seg"][0] == class_idx, data["gt_semantic_seg"][0],
                                                 torch.tensor([255], dtype=torch.uint8))
        with torch.enable_grad():
            loss = model(return_loss=True, img=data["img"][0], img_metas=data["img_metas"],
                         gt_semantic_seg=data["gt_semantic_seg"][0].unsqueeze(dim=0).type(torch.long))
            loss["decode.loss_seg"].backward()

        if progress_bar:
            prog_bar.update()

    grad = None
    for name, param in model.named_parameters():
        if "backbone" not in name:
            continue

        if grad is None:
            grad = param.grad.flatten()
        else:
            grad = torch.hstack((grad.flatten(), param.grad.flatten()))

    return torch.abs(grad)


def parse_checkpoint_mIoUs(checkpoint_results):
    mIoUs = []
    for res_file in checkpoint_results:
        file = ""
        with open(res_file) as f:
            file = f.readlines()
        #print(file)

        regex = "\\|[a-z\. ]*\\|[0-9\. ]*\\|[0-9\. ]*\\|"
        findings =[]
        for line in file:
            findings.extend(re.findall(regex, line))

        findings = [float(f.replace(" ", "")[1:-1].split("|")[1]) for f in findings]
        mIoUs.append(findings)
    return mIoUs


def get_classwise_checkpoint_weights(checkpoint_mIoUs):
    mIoU_sum = []
    mIoU_best = []
    for c_idx, checkpoint_mIoU in enumerate(checkpoint_mIoUs):
        for class_idx, class_mIoU in enumerate(checkpoint_mIoU):
            if class_idx >= len(mIoU_sum):
                mIoU_sum.append(class_mIoU)
                mIoU_best.append(class_mIoU)
            else:
                mIoU_sum[class_idx] += class_mIoU
                if class_mIoU > mIoU_best[class_idx]:
                    mIoU_best[class_idx] = class_mIoU

    result = []
    num_classes = len(checkpoint_mIoUs)
    for checkpoint_mIoU in checkpoint_mIoUs:
        res_c = []
        for class_idx, class_mIoU in enumerate(checkpoint_mIoU):
            if mIoU_sum[class_idx] == 0:
                res_c.append(1.0/num_classes)
            else:
                res_c.append(class_mIoU / mIoU_sum[class_idx])
                #res_c.append(float(class_mIoU >= mIoU_best[class_idx]))
        result.append(res_c)

    print(result)
    return result


def get_classwise_weight_relevance(model, data_loader, tmp_dir, use_existing_files=False, progress_bar=True, norm_files=False):
    num_classes = 19
    grad_sum = None
    for i in range(num_classes):
        print(f"Get grad for class {i} of {num_classes}")
        if use_existing_files:
            grad = torch.load(os.path.join(tmp_dir, f'grad_{i}.pt'))
        else:
            model.zero_grad()
            grad = get_class_weight_relevance(model, data_loader, i, progress_bar=progress_bar)
            torch.save(grad, os.path.join(tmp_dir, f'grad_{i}.pt'))

        if grad_sum is None:
            grad_sum = grad.flatten()
        else:
            grad_sum = torch.sum(torch.stack([grad_sum, grad]), dim=0)

    torch.save(grad_sum, os.path.join(tmp_dir, f'grad_sum.pt'))

    # normalize grad_files
    print(f"Normalize grad-files")
    prog_bar = mmcv.ProgressBar(num_classes)

    res_files = []
    for i in range(num_classes):
        if norm_files:
            grad = torch.load(os.path.join(tmp_dir, f'grad_{i}.pt'))
            norm_grad = torch.div(grad, grad_sum)
            norm_grad = torch.where(norm_grad.cuda().isnan(), torch.tensor([1.0/num_classes], dtype=torch.float).cuda(), norm_grad.cuda())
            #print(f"Class {i}: {norm_grad[7000000]}")

            torch.save(norm_grad, os.path.join(tmp_dir, f'grad_{i}.pt'))
        res_files.append(os.path.join(tmp_dir, f'grad_{i}.pt'))

        prog_bar.update()
    return res_files


def _reparam_classwise(model, checkpoint_files, checkpoints_weights, weight_class_distribution_files):
    # load checkpoints
    for i, e in enumerate(checkpoint_files):
        if isinstance(e, str):
            checkpoint_files[i] = torch.load(e)

    # num_classes = len(weight_class_distribution_files)
    res = copy.deepcopy(checkpoint_files[0])
    eval_name = list(res["state_dict"].keys())[50]
    print(eval_name)
    print(checkpoint_files[0]["state_dict"][eval_name])
    print(checkpoint_files[1]["state_dict"][eval_name])
    print(checkpoint_files[2]["state_dict"][eval_name])

    for checkpoint_idx, checkpoint in enumerate(checkpoint_files):
        for class_idx, weights in enumerate(zip(weight_class_distribution_files, checkpoints_weights[checkpoint_idx])):
            idx_offset = 0
            weight_file, class_checkpoint_weight = weights
            classwise_weight = torch.load(weight_file)
            if torch.isnan(classwise_weight).any():
                print("Warning: classwise_weight contains NaNs")
            #print(f"classwise_weight shape: {classwise_weight.shape}")

            for name, param in model.module.named_parameters():
                if "backbone" not in name:
                    continue

                param_len = param.flatten().shape[0]
                #print(param_len, idx_offset)

                #print(classwise_weight[idx_offset:idx_offset + param_len].shape)
                reshaped_classwise_weight = torch.reshape(classwise_weight[idx_offset:idx_offset + param_len],
                                                          param.shape)
                idx_offset += param_len
                checkpoint_weights = checkpoint["state_dict"][name]

                # Reset weight if first value
                if checkpoint_idx == 0 and class_idx == 0:
                    #print(f"set to zero '{eval_name}'")
                    res["state_dict"][name] = torch.zeros_like(param)
                res["state_dict"][name] += reshaped_classwise_weight.cuda() * checkpoint_weights.cuda() * class_checkpoint_weight
                #res["state_dict"][name] += reshaped_classwise_weight.cuda() * class_checkpoint_weight
        """sum = 0
        for k in res["state_dict"].keys():
            if "backbone" not in k:
                continue

            sum = sum + torch.sum(res["state_dict"][k] - 1.0)
    print(f"Sum is: {sum}")"""
    print(res["state_dict"][eval_name])
    return res


def classwise_reparameterization(model, data_loader, tmp_dir, checkpoint_files, checkpoint_result_files,
                                 progress_bar=True):
    files = get_classwise_weight_relevance(model, data_loader, tmp_dir, use_existing_files=True,
                                           progress_bar=progress_bar)
    mIoUs = parse_checkpoint_mIoUs(checkpoint_result_files)
    checkpoint_weights = get_classwise_checkpoint_weights(mIoUs)

    res = _reparam_classwise(model, checkpoint_files, checkpoints_weights=checkpoint_weights,
                             weight_class_distribution_files=files)
    torch.save(res, os.path.join(tmp_dir, 'result_checkpoint.pth'))
