import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from lib.model.tdcnn.tdcnn import _TDCNN
import math
from functools import partial
from lib.model.utils.config import cfg

__all__ = [
    'drsn', 'drsn10', 'drsn18', 'drsn34', 'drsn50', 'drsn101',
    'drsn152', 'drsn200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.shrinkage = Shrinkage(planes, gap_size=(1, 1, 1))

        self.residual_function = nn.Sequential(
            conv3x3x3(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            conv3x3x3(planes, planes * BasicBlock.expansion),
            nn.BatchNorm3d(planes * BasicBlock.expansion),
            self.shrinkage
            )

        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != BasicBlock.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x    # [2, 64, 64, 28, 28]
        x = self.gap(x)  # [2, 64, 1, 1, 1]

        x=x.view(-1, x.size(1))
        # x = torch.flatten(x, 1)  # [2, 64]

        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        # print('x', x.size())
        # print('x_abs', x_abs.size())

        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.shrinkage = Shrinkage(planes, gap_size=(1, 1, 1))

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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
        out = self.relu(out)

        return out

class drsn(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(drsn, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def drsn10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = drsn(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def drsn18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = drsn(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def drsn34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = drsn(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def drsn50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = drsn(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def drsn101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = drsn(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def drsn152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = drsn(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def drsn200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = drsn(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

class drsn_tdcnn(_TDCNN):
    def __init__(self, depth=34, pretrained=False):
        #self.model_path = '/home/agwang/Deeplearning/pytorch_dir/pretrainedmodels/resnet-34-kinetics-cpu.pth'
        self.pretrained = pretrained
        self.depth = depth
        self.shortcut_type = 'A' if depth in [18, 34] else 'B'
        self.dout_base_model = 256 if depth in [18, 34] else 1024
        # self.model_path = 'data/pretrained_model/ucf101-caffe.pth'
        self.model_path = 'data/pretrained_model/save_200.pth'.format(depth)
        self.dout_top_model = 512 if depth in [18, 34] else 2048
        _TDCNN.__init__(self)

    def _init_modules(self):
        net_str = "drsn34{}(sample_size=112, sample_duration={}, shortcut_type=\'{}\')".format(self.depth, int(cfg.TRAIN.LENGTH[0]), self.shortcut_type)
        drsn = eval(net_str)
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)['state_dict']
            # parallel unpack (model = nn.DataParallel(model, device_ids=None))
            drsn.load_state_dict({k[7:] : v for k,v in state_dict.items() if k[7:] in drsn.state_dict()})

        # Using , shape(1,256,96,7,7) for resnet34
        self.RCNN_base = nn.Sequential(drsn.conv1, drsn.bn1,drsn.relu,
                        drsn.maxpool,drsn.layer1,drsn.layer2,drsn.layer3)
        # Using
        self.RCNN_top = nn.Sequential(drsn.layer4)

        # Fix blocks:
        # TODO: fix blocks optionally
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

        # not using the last maxpool layer,
        self.RCNN_cls_score = nn.Linear(self.dout_top_model, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(self.dout_top_model, 2 * self.n_classes)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm3d') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode, FIXED_BLOCKS=0
            self.RCNN_base.eval()
            if cfg.RESNET.FIXED_BLOCKS < 1:
                self.RCNN_base[4].train()
            if cfg.RESNET.FIXED_BLOCKS < 2:
                self.RCNN_base[5].train()
            if cfg.RESNET.FIXED_BLOCKS < 3:
                self.RCNN_base[6].train()

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm3d') != -1:
                m.eval()

        self.RCNN_base.apply(set_bn_eval)
        self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc6 = self.RCNN_top(pool5).mean(4).mean(3).mean(2)
        return fc6
