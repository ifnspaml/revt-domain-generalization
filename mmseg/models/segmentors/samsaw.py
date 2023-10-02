##copied from: https://github.com/leolyj/SAN-SAW/blob/2964f913c05c6a98b44248e5ae0879fe4de1b1e1/graphs/models/deeplab50_ClassINW.py#L125
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
affine_par = True
import torch.utils.model_zoo as model_zoo
from math import ceil
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import torch.utils.model_zoo as model_zoo
import kmeans1d
import time
import numpy as np
#import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
affine_par = True

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ..builder import SEGMENTORS
from .base import BaseSegmentor


class SAW(nn.Module):
    def __init__(self, args, dim, relax_denom=0, classifier=None, work=False):
        super(SAW, self).__init__()
        self.work = work
        self.selected_classes = args.selected_classes
        self.C = len(args.selected_classes)
        self.dim = dim
        self.i = torch.eye(self.C, self.C).cuda()
        self.reversal_i = torch.ones(self.C, self.C).triu(diagonal=1).cuda()
        self.classify = classifier
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            print("Note relax_denom == 0!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom


    def get_mask_matrix(self):
        return self.i, self.reversal_i, self.margin, self.num_off_diagonal

    def get_covariance_matrix(self, x, eye=None):
        eps = 1e-5
        B, C, H, W = x.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        if eye is None:
            eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B

    def instance_whitening_loss(self, x, eye, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x, eye=eye)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss
    def sort_with_idx(self, x, idx,weights):
        b,c,_,_ = x.size()
        after_sort = torch.zeros_like(x)
        weights = F.sigmoid(weights)
        for i in range(b):

            for k in range(int(c / self.C)):
                for j in range(self.C):
                    channel_id = idx[self.selected_classes[j]][k]
                    wgh = weights[self.selected_classes[j]][channel_id]
                    after_sort[i][self.C*k+j][:][:] = wgh * x[i][channel_id][:][:]

        return after_sort

    def forward(self, x):
        if self.work:
            weights_keys = self.classify.state_dict().keys()

            selected_keys_classify = []

            for key in weights_keys:
                if "weight" in key:
                    selected_keys_classify.append(key)

            for key in selected_keys_classify:
                weights_t = self.classify.state_dict()[key]

            classsifier_weights = abs(weights_t.squeeze())
            _,index = torch.sort(classsifier_weights, descending=True,dim=1)
            f_map_lst = []
            B, channel_num, H, W = x.shape
            x = self.sort_with_idx(x,index,classsifier_weights)

            for i in range(int(channel_num/self.C)):
                group = x[:,self.C*i:self.C*(i+1),:,:]
                f_map_lst.append(group)

            eye, mask_matrix, margin, num_remove_cov = self.get_mask_matrix()
            SAW_loss = torch.FloatTensor([0]).cuda()

            for i in range(int(channel_num / self.C)):
                loss = self.instance_whitening_loss(f_map_lst[i], eye, mask_matrix, margin, num_remove_cov)
                SAW_loss = SAW_loss+loss
        else:
            SAW_loss = torch.FloatTensor([0]).cuda()



        return SAW_loss


class SAN(nn.Module):

    def __init__(self, inplanes, selected_classes=None):
        super(SAN, self).__init__()
        self.margin = 0
        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.selected_classes = selected_classes
        self.CFR_branches = nn.ModuleList()
        for i in selected_classes:
            self.CFR_branches.append(
                nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.mask_matrix = None

    def cosine_distance(self, obs, centers):
        obs_norm = obs / obs.norm(dim=1, keepdim=True)
        centers_norm = centers / centers.norm(dim=1, keepdim=True)
        cos = torch.matmul(obs_norm, centers_norm.transpose(1, 0))
        return 1 - cos

    def l2_distance(self, obs, centers):
        dis = ((obs.unsqueeze(dim=1) - centers.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
        return dis

    def _kmeans_batch(self, obs: torch.Tensor, k: int, distance_function,batch_size=0, thresh=1e-5, norm_center=False):

        # k x D
        centers = obs[torch.randperm(obs.size(0))[:k]].clone()
        history_distances = [float('inf')]
        if batch_size == 0:
            batch_size = obs.shape[0]
        while True:
            # (N x D, k x D) -> N x k
            segs = torch.split(obs, batch_size)
            seg_center_dis = []
            seg_center_ids = []
            for seg in segs:
                distances = distance_function(seg, centers)
                center_dis, center_ids = distances.min(dim=1)
                seg_center_ids.append(center_ids)
                seg_center_dis.append(center_dis)

            obs_center_dis_mean = torch.cat(seg_center_dis).mean()
            obs_center_ids = torch.cat(seg_center_ids)
            history_distances.append(obs_center_dis_mean.item())
            diff = history_distances[-2] - history_distances[-1]
            if diff < thresh:
                if diff < 0:
                    warnings.warn("Distance diff < 0, distances: " + ", ".join(map(str, history_distances)))
                break
            for i in range(k):
                obs_id_in_cluster_i = obs_center_ids == i
                if obs_id_in_cluster_i.sum() == 0:
                    continue
                obs_in_cluster = obs.index_select(0, obs_id_in_cluster_i.nonzero().squeeze())
                c = obs_in_cluster.mean(dim=0)
                if norm_center:
                    c /= c.norm()
                centers[i] = c
        return centers, history_distances[-1]

    def kmeans(self, obs: torch.Tensor, k: int, distance_function=l2_distance, iter=20, batch_size=0, thresh=1e-5, norm_center=False):

        best_distance = float("inf")
        best_centers = None
        for i in range(iter):
            if batch_size == 0:
                batch_size == obs.shape[0]
            centers, distance = self._kmeans_batch(obs, k,
                                              norm_center=norm_center,
                                              distance_function=distance_function,
                                              batch_size=batch_size,
                                              thresh=thresh)
            if distance < best_distance:
                best_centers = centers
                best_distance = distance
        return best_centers, best_distance

    def product_quantization(self, data, sub_vector_size, k, **kwargs):
        centers = []
        for i in range(0, data.shape[1], sub_vector_size):
            sub_data = data[:, i:i + sub_vector_size]
            sub_centers, _ = self.kmeans(sub_data, k=k, **kwargs)
            centers.append(sub_centers)
        return centers

    def data_to_pq(self, data, centers):
        assert (len(centers) > 0)
        assert (data.shape[1] == sum([cb.shape[1] for cb in centers]))

        m = len(centers)
        sub_size = centers[0].shape[1]
        ret = torch.zeros(data.shape[0], m,
                          dtype=torch.uint8,
                          device=data.device)
        for idx, sub_vec in enumerate(torch.split(data, sub_size, dim=1)):
            dis = self.l2_distance(sub_vec, centers[idx])
            ret[:, idx] = dis.argmin(dim=1).to(dtype=torch.uint8)
        return ret

    def train_product_quantization(self, data, sub_vector_size, k, **kwargs):
        center_list = self.product_quantization(data, sub_vector_size, k, **kwargs)
        pq_data = self.data_to_pq(data, center_list)
        return pq_data, center_list

    def _gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def pq_distance_book(self, pq_centers):
        assert (len(pq_centers) > 0)

        pq = torch.zeros(len(pq_centers),
                         len(pq_centers[0]),
                         len(pq_centers[0]),
                         device=pq_centers[0].device)
        for ci, center in enumerate(pq_centers):
            for i in range(len(center)):
                dis = self.l2_distance(center[i:i + 1, :], center)
                pq[ci, i] = dis
        return pq

    def Regional_Normalization(self, region_mask, x):
        masked = x*region_mask
        RN_feature_map = self.IN(masked)
        return RN_feature_map

    def asymmetric_table(self, query, centers):
        m = len(centers)
        sub_size = centers[0].shape[1]
        ret = torch.zeros(
            query.shape[0], m, centers[0].shape[0],
            device=query.device)
        assert (query.shape[1] == sum([cb.shape[1] for cb in centers]))
        for i, offset in enumerate(range(0, query.shape[1], sub_size)):
            sub_query = query[:, offset: offset + sub_size]
            ret[:, i, :] = self.l2_distance(sub_query, centers[i])
        return ret

    def asymmetric_distance_slow(self, asymmetric_tab, pq_data):
        ret = torch.zeros(asymmetric_tab.shape[0], pq_data.shape[0])
        for i in range(asymmetric_tab.shape[0]):
            for j in range(pq_data.shape[0]):
                dis = 0
                for k in range(pq_data.shape[1]):
                    sub_dis = asymmetric_tab[i, k, pq_data[j, k].item()]
                    dis += sub_dis
                ret[i, j] = dis
        return ret

    def asymmetric_distance(self, asymmetric_tab, pq_data):
        pq_db = pq_data.long()
        dd = [torch.index_select(asymmetric_tab[:, i, :], 1, pq_db[:, i]) for i in range(pq_data.shape[1])]
        return sum(dd)

    def pq_distance(self, obj, centers, pq_disbook):
        ret = torch.zeros(obj.shape[0], centers.shape[0])
        for obj_idx, o in enumerate(obj):
            for ct_idx, c in enumerate(centers):
                for i, (oi, ci) in enumerate(zip(o, c)):
                    ret[obj_idx, ct_idx] += pq_disbook[i, oi.item(), ci.item()]
        return ret

    def set_class_mask_matrix(self, normalized_map):

        b,c,h,w = normalized_map.size()
        var_flatten = torch.flatten(normalized_map)


        try:  # kmeans1d clustering setting for RN block
            clusters, centroids = kmeans1d.cluster(var_flatten,5, 3)
            num_category = var_flatten.size()[0] - clusters.count(0)  # 1: class-region, 2~5: background
            _, indices = torch.topk(var_flatten, k=int(num_category))
            mask_matrix = torch.flatten(torch.zeros(b, c, h, w).cuda())
            mask_matrix[indices] = 1
        except:
            mask_matrix = torch.ones(var_flatten.size()[0]).cuda()

        mask_matrix = mask_matrix.view(b, c, h, w)

        return mask_matrix

    def forward(self, x, masks):
        outs=[]
        idx = 0
        masks = F.softmax(masks,dim=1)
        for i in self.selected_classes:
            mask = torch.unsqueeze(masks[:,i,:,:],1)
            mid = x * mask
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out,_ = torch.max(mid,dim=1, keepdim=True)
            atten = torch.cat([avg_out,max_out,mask],dim=1)
            atten = self.sigmoid(self.CFR_branches[idx](atten))
            out = mid*atten
            heatmap = torch.mean(out, dim=1, keepdim=True)

            class_region = self.set_class_mask_matrix(heatmap)
            out = self.Regional_Normalization(class_region,out)
            outs.append(out)
        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        # self.IN = None
        # if IN:
        #     self.IN = nn.InstanceNorm2d(planes*4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # if self.IN is not None:
        #
        #     out = self.IN(out)
        out = self.relu(out)

        return out


class class_in_block(nn.Module):

    def __init__(self, inplanes, classin_classes=None):
        super(class_in_block, self).__init__()

        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.classin_classes = classin_classes
        self.branches = nn.ModuleList()
        for i in classin_classes:
            self.branches.append(
                nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, masks):
        outs=[]
        idx = 0
        masks = F.softmax(masks,dim=1)
        for i in self.classin_classes:
            mask = torch.unsqueeze(masks[:,i,:,:],1)
            mid = x * mask
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out,_ = torch.max(mid,dim=1, keepdim=True)
            atten = torch.cat([avg_out,max_out,mask],dim=1)
            atten = self.sigmoid(self.branches[idx](atten))
            out = mid*atten
            out = self.IN(out)
            outs.append(out)
        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

@SEGMENTORS.register_module()
class ResNetMulti(BaseSegmentor):
    def __init__(self,args, layers, num_classes, block=Bottleneck, pretrained=None, train_cfg=None, test_cfg=None):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()

        self.classifier_1 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.classifier_2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.in1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # self.layer1_class_in = class_in_block(inplanes=256, classin_classes=args.selected_classes)
        # self.layer2_class_in = class_in_block(inplanes=512, classin_classes=args.selected_classes)
        self.SAN_stage_1 = SAN(inplanes=256, selected_classes=args.selected_classes)
        self.SAN_stage_2 = SAN(inplanes=512, selected_classes=args.selected_classes)
        self.SAW_stage_1 = SAW(args, dim=256, relax_denom=2.0, classifier=self.classifier_1)
        self.SAW_stage_2 = SAW(args, dim=512, relax_denom=2.0, classifier=self.classifier_2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def print_weights(self):
        weights_keys = self.layer1_pred.state_dict().keys()
        for key in weights_keys:
            if "num_batches_tracked" in key:
                continue
            weights_t = self.layer1_pred.state_dict()[key].numpy()
        return weights_t

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        #print(len(img))
        x = img[0]
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_1_ori = x
        x1 = self.classifier_1(x.detach())
        x = self.SAN_stage_1(x,x1)
        x_1_ined = x

        saw_loss_lay1 = self.SAW_stage_1(x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        x = self.layer2(x)
        x_2_ori = x
        x2 = self.classifier_2(x.detach())
        x = self.SAN_stage_2(x, x2)
        x_2_ined = x

        saw_loss_lay2 = self.SAW_stage_2(x)
        x2 = F.interpolate(x2, size=input_size, mode='bilinear', align_corners=True)

        x = self.layer3(x)
        x3 = self.layer5(x)
        x3 = F.interpolate(x3, size=input_size, mode='bilinear', align_corners=True)

        x4 = self.layer4(x)
        x4 = self.layer6(x4)
        x4 = F.interpolate(x4, size=input_size, mode='bilinear', align_corners=True)

        return x4, x3, x2, x1, x_2_ori, x_2_ined, x_1_ori, x_1_ined, saw_loss_lay2, saw_loss_lay1

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.in1)
        b.append(self.layer1)
        b.append(self.SAN_stage_1)
        b.append(self.layer2)
        b.append(self.SAN_stage_2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.classifier_1)
        b.append(self.classifier_2)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]


def Res50_ClassINW(args,num_classes=21, pretrained=True):
    model = ResNetMulti(args, [3, 4, 6, 3], num_classes, block=Bottleneck)

    if pretrained:


        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)

        # restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
        # # restore_from = './pretrained_model/GTA5_source.pth'
        # saved_state_dict = torch.load(restore_from)
        #
        # new_params = model.state_dict().copy()
        # for i in saved_state_dict:
        #     i_parts = i.split('.')
        #     if not i_parts[1] == 'layer5':
        #         new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        # model.load_state_dict(new_params)
    return model

if __name__ == '__main__':
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], 19)
    restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
    # restore_from = './pretrained_model/GTA5_source.pth'
    saved_state_dict = torch.load(restore_from)
    # for i in saved_state_dict:
    #     print("i:",i)
    #     i_parts = i.split('.')
    #     print(i_parts[0],i_parts[1])

    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        print("i:",i)
        i_parts = i.split('.')


    for i in new_params:
        print("i_new:",i)
        i_parts = i.split('.')

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    model.load_state_dict(new_params)
