from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss
from lib.model.utils.mulScaleBlock import MulScaleBlockBlock3DChannels
from lib.model.utils.mulScaleTime import MulScaleBlockBlock3DChannels
from lib.model.utils.mulScaleTC import MulScaleBlockTC

import numpy as np
import math
import pdb
import time

# 生成良好的边界框
DEBUG = False

class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din, out_scores=False):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512

        self.anchor_scales = cfg.ANCHOR_SCALES  # anchor尺度
        self.feat_stride = cfg.FEAT_STRIDE[0]  # 特征步长， 记录图像经过特征图缩小的尺度
        self.out_scores = out_scores
        self.mask_upsample_rate = 1 # 上采样

        # define the convrelu layers processing input feature map
        # 3x3x3滑动窗口集中特征信息 在每个滑动窗口位置，我们同时预测多个region proposals，其中每个位置的最大可能建议的数量表示为k
        # 空间上 H/16 x W/16 的特征 下采样到 1x1。最后输出 512xL/8x1x1
        # self.RCNN_attention = NONLocalBlock3D(self.din, inter_channels=self.din)
        # self.mulScaleBlock = MulScaleBlockBlock3DChannels(self.din, 512, stride=1, downsample=None)
        # self.mulScaleBlock = MulScaleBlockBlock3DChannels(self.din, self.din, stride=1, downsample=None, frame=self.frame)
        # self.mulScaleBlock = MulScaleBlockTC(self.din, self.din, stride=1, downsample=None,
        #                                                   frame=self.frame)

        self.RPN_Conv1 = nn.Conv3d(self.din, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True)
        self.RPN_Conv2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True)
        self.RPN_output_pool = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                            stride=(1, 2, 2))  # 空间维度上下采样  则输出是(2, 512, 96, 1, 1)符合论文要求

        # define bg/fg classifcation score layer 每个proposal是行为还是背景二分类分数
        # 对每个anchor都要进行背景或前景的分类得分， 个数就是尺度个数乘以2
        self.nc_score_out = len(self.anchor_scales) * 2  # 2(bg/fg) * 10 (anchors)
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0)  # 生成每个anchor的前景和背景得分

        # define anchor twin offset prediction layer
        self.nc_twin_out = len(self.anchor_scales) * 2  # 2(coords) * 10 (anchors)
        self.RPN_twin_pred = nn.Conv3d(512, self.nc_twin_out, 1, 1, 0)  # 生成每个anchor的坐标偏移量

        # define proposal layer 生成候选区域， 包括对anchor的一系列得分排序，锚框修正操作，最终返回符合要求的rois
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.out_scores)   # 已经经过非极大值抑制进行筛选

        # define anchor target layer 将anchor对应GT. 生成anchor标签和边界框回归目标
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales)

        self.rpn_loss_cls = 0
        self.rpn_loss_twin = 0
        self.rpn_loss_mask = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
            input_shape[4]
        )
        return x

    def forward(self, base_feat, gt_twins):
        batch_size = base_feat.size(0)  # 特征的第一维
        print("base_feat", base_feat.size())

        # return feature map after conv relu layer 首先调用RPN_Conv对提取的主干网络特征进行卷积并通过ReLU激活。
        # mulScale = self.mulScaleBlock(base_feat)
        # print("mulScale", mulScale.size())
        rpn_conv1 = F.relu(self.RPN_Conv1(base_feat), inplace=True)
        print("rpn_conv1", rpn_conv1.size())
        rpn_conv2 = F.relu(self.RPN_Conv2(rpn_conv1), inplace=True)
        print("rpn_conv2", rpn_conv2.size())

        rpn_output_pool = self.RPN_output_pool(rpn_conv2)  # (1,512,16,1,1)
        print("rpn_output_pool",rpn_output_pool.size())

        # get rpn classification score
        # 调用RPN_cls_score对rpn_output_pool进行11卷积得到rpn classification score 为计算每个anchor的前景背景得分，需要将特征变形，
        # 将特征输入Softmax中获得相应类别rpn_cls_prob。
        rpn_cls_score = self.RPN_cls_score(rpn_output_pool)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        # 用softmax函数得到概率
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        # 前景背景分类， 2个参数
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor twins
        # 调用RPN_twin_pred对rpn_output_pool进行11卷积得到anchor坐标修正值。
        # 将边框修正值和类别预测输入到RPN_proposal中获取对应的rois。
        rpn_twin_pred = self.RPN_twin_pred(rpn_output_pool)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # rois = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))
        # rpn及paoposalLayer生成的rois会返回到tdRCNN的forward中，对RPN网络生成的候选区域进行后续的池化和分类及回归计算。
        if self.out_scores:
            rois, rois_score = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))
        else:
            rois = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_twin = 0
        self.rpn_loss_mask = 0
        self.rpn_label = None

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_twins is not None
            # rpn_data = [label_targets, twin_targets, twin_inside_weights, twin_outside_weights]
            # label_targets: (batch_size, 1, A * length, height, width)
            # twin_targets: (batch_size, A*2, length, height, width), the same as twin_inside_weights and twin_outside_weights
            # anchor目标
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_twins))

            # compute classification loss
            # permute(多维数组，[维数的组合]) 改变维数
            # 返回rpn网络判断的anchor前后景分数
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, 2)
            self.rpn_label = rpn_data[0].view(batch_size, -1)  # 返回每个anchor属于前景还是后景的ground truth

            # 在调用分类器之前， 分类器输入输出都是一维， view就是将多维tensor展平成一维
            rpn_keep = Variable(self.rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(self.rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)   # 交叉熵 RPN生成的边界框被正确分类为前景/背景的比例
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_twin_targets, rpn_twin_inside_weights, rpn_twin_outside_weights = rpn_data[1:]

            # compute twin regression loss
            # 在训练计算边框误差时有用， 仅对未超出图像边界的anchor有用
            rpn_twin_inside_weights = Variable(rpn_twin_inside_weights)
            rpn_twin_outside_weights = Variable(rpn_twin_outside_weights)
            # 返回每个anchor对应的事实偏移值
            rpn_twin_targets = Variable(rpn_twin_targets)

            self.rpn_loss_twin = _smooth_l1_loss(rpn_twin_pred, rpn_twin_targets, rpn_twin_inside_weights,
                                                rpn_twin_outside_weights, sigma=3, dim=[1, 2, 3, 4])         #  平滑的L1损失函数 预测回归系数与目标回归系数之间的距离

        if self.out_scores:
            return rois, rois_score, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask
        else:
            return rois, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)   # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RPN_Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_twin_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self.init_weights()

    def generate_mask_label(self, gt_twins, feat_len):
        """
        gt_twins will be (batch_size, n, 3), where each gt will be (x1, x2, class_id)
        # feat_len is the length of mask-task features, self.feat_stride * feat_len = video_len
        # according: self.feat_stride, and upsample_rate
        # mask will be (batch_size, feat_len), -1 -- ignore, 1 -- fg, 0 -- bg
        """
        batch_size = gt_twins.size(0)
        mask_label = torch.zeros(batch_size, feat_len).type_as(gt_twins)
        for b in range(batch_size):
            single_gt_twins = gt_twins[b]
            single_gt_twins[:, :2] = (single_gt_twins[:, :2] / self.feat_stride).int()
            twins_start = single_gt_twins[:, 0]
            _, indices = torch.sort(twins_start)
            single_gt_twins = torch.index_select(single_gt_twins, 0, indices).long().cpu().numpy()

            starts = np.minimum(np.maximum(0, single_gt_twins[:, 0]), feat_len - 1)
            ends = np.minimum(np.maximum(0, single_gt_twins[:, 1]), feat_len)
            for x in zip(starts, ends):
                mask_label[b, x[0]:x[1] + 1] = 1

        return mask_label