import  torch
import torch.nn as nn
from lib.model.tdcnn.tdcnn import _TDCNN
import math


# M为池化层
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

# 所有3d卷积核大小为3x3x3步长为1
# 池化层的第一个数是时间深度，如果设置成1的话，也就是在单独的每帧上面进行池化，
# 如果大于1的话，那么就是在时间轴上，也就是多帧之间进行池化，前者是有利于在初始阶段保留时间特征，
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    maxpool_count = 0
    # 池化层共五个，除了第一个是1*2*2 其余的为2*2*2 最后一个padding=(0, 1, 1)
    for v in cfg:
        if v == 'M':
            maxpool_count += 1
            if maxpool_count == 1:
                layers += [nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))]
            elif maxpool_count == 5:
                layers += [nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))]
            else:
                layers += [nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v   # 512

    return nn.Sequential(*layers)

class C3D(nn.Module):
    """
    The C3D network as described in [1].
        References
        ----------
       [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
       Proceedings of the IEEE international conference on computer vision. 2015.
    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __init__(self):
        super(C3D, self).__init__()
        self.features = make_layers(cfg['A'], batch_norm=False)
        self.classifier = nn.Sequential(
            nn.Linear(512*1*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
            nn.Linear(4096, 487),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class c3d_tdcnn(_TDCNN):
    def __init__(self, pretrained = False):
        self.model_path = 'data/pretrained_model/activitynet_iter_30000_3fps-caffe.pth'
        self.dout_base_model = 512   # 利用Backbone网络提取的特征图深度
        self.frame = 16
        self.pretrained = pretrained
        _TDCNN.__init__(self,)

    # 将Backbone网络c3d进行切分，定义为tdcnn的组块，分别定义RCNN_base、RCNN_top、RCNN_cls_score、RCNN_twin_pre
    def _init_modules(self):
        c3d = C3D()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            c3d.load_state_dict({k:v for k, v in state_dict.items() if k in c3d.state_dict()})

        # Using conv1 -> conv5b, not using the last maxpool
        self.RCNN_base = nn.Sequential(*list(c3d.features._modules.values())[:-1])
        # Using fc6
        self.RCNN_top = nn.Sequential(*list(c3d.classifier._modules.values())[:-4])
        # Fix the layers before pool2:
        for layer in range(6):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False


        # not using the last maxpool layer 给出单独的分类层
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(4096, 2*self.n_classes)

    # 将最终得到的兴趣区域特征展平，输入到全连接层中
    def _head_to_tail(self, pool5):
        # 将最后一个池化层展平
        pool5_flat = pool5.view(pool5.size(0), -1)
        # 将展平的pool5输入到全连接层
        fc6 = self.RCNN_top(pool5_flat)

        return fc6