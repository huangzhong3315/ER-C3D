import torch
import torch.nn as nn
from lib.model.utils.mulScaleBlock import MulScaleBlockBlock3DChannels

def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class MulScaleBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes=None, stride=1, downsample=None, frame=None):
        super(MulScaleBlock, self).__init__()

        norm_layer = nn.BatchNorm3d
        scale_width = int(frame / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.mulScaleBlock = MulScaleBlockBlock3DChannels(inplanes, planes, stride=1, downsample=None)

        self.conv1_2_1 = conv3x3x3(inplanes, planes)
        self.bn1_2_1 = norm_layer(planes)
        self.conv1_2_2 = conv3x3x3(inplanes, planes)
        self.bn1_2_2 = norm_layer(planes)
        self.conv1_2_3 = conv3x3x3(inplanes, planes)
        self.bn1_2_3 = norm_layer(planes)
        self.conv1_2_4 = conv3x3x3(inplanes, planes)
        self.bn1_2_4 = norm_layer(planes)

        self.conv2_2_1 = conv3x3x3(inplanes, planes)
        self.bn2_2_1 = norm_layer(planes)
        self.conv2_2_2 = conv3x3x3(inplanes, planes)
        self.bn2_2_2 = norm_layer(planes)
        self.conv2_2_3 = conv3x3x3(inplanes, planes)
        self.bn2_2_3 = norm_layer(planes)
        self.conv2_2_4 = conv3x3x3(inplanes, planes)
        self.bn2_2_4 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 2)
        print("sp_x", sp_x[0].size())

        ##########################################################

        out_C_0 = self.mulScaleBlock(sp_x[0])
        out_C_1 = self.mulScaleBlock(sp_x[1])
        out_C_2 = self.mulScaleBlock(sp_x[2])
        out_C_3 = self.mulScaleBlock(sp_x[3])
        print("out_C_0", out_C_0.size())

        ##########################################################

        ##########################################################
        out_1_1 = self.conv1_2_1(out_C_0)
        out_1_1 = self.bn1_2_1(out_1_1)

        out_1_1_relu = self.relu(out_1_1)
        out_1_2 = self.conv1_2_2(out_1_1_relu + out_C_1)
        out_1_2 = self.bn1_2_2(out_1_2)

        out_1_2_relu = self.relu(out_1_2)
        out_1_3 = self.conv1_2_3(out_1_2_relu + out_C_2)
        out_1_3 = self.bn1_2_3(out_1_3)

        out_1_3_relu = self.relu(out_1_3)
        out_1_4 = self.conv1_2_4(out_1_3_relu + out_C_3)
        out_1_4 = self.bn1_2_4(out_1_4)

        output_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=2)
        ############################################################

        ############################################################
        # out_2_1 = self.conv2_2_1(out_C_0)
        # out_2_1 = self.bn2_2_1(out_2_1)
        # out_2_1_relu = self.relu(out_2_1)
        #
        # out_2_2 = self.conv2_2_2(out_2_1_relu + out_C_1)
        # out_2_2 = self.bn2_2_2(out_2_2)
        # out_2_2_relu = self.relu(out_2_2)
        #
        # out_2_3 = self.conv2_2_3(out_2_2_relu + out_C_2)
        # out_2_3 = self.bn2_2_3(out_2_3)
        # out_2_3_relu = self.relu(out_2_3)
        #
        # out_2_4 = self.conv2_2_4(out_2_3_relu + out_C_3)
        # out_2_4 = self.bn2_2_4(out_2_4)
        #
        # output_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=2)

        out_2_1 = self.conv2_2_1(out_C_3)
        out_2_1 = self.bn2_2_1(out_2_1)
        out_2_1_relu = self.relu(out_2_1)

        out_2_2 = self.conv2_2_2(out_2_1_relu + out_C_2)
        out_2_2 = self.bn2_2_2(out_2_2)
        out_2_2_relu = self.relu(out_2_2)

        out_2_3 = self.conv2_2_3(out_2_2_relu + out_C_1)
        out_2_3 = self.bn2_2_3(out_2_3)
        out_2_3_relu = self.relu(out_2_3)

        out_2_4 = self.conv2_2_4(out_2_3_relu + out_C_0)
        out_2_4 = self.bn2_2_4(out_2_4)

        output_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=2)
        #################################################################


        out = output_1 + output_2
        print("out", out.size())
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        print("out", out.size())
        out = self.relu(out)

        return out

class MulScaleBlockTC(MulScaleBlock):
    def __init__(self, inplanes, planes=None, stride=1, downsample=None, frame=None):
        super(MulScaleBlockTC, self).__init__(inplanes,
                                              planes=planes,
                                              stride=stride,
                                              downsample=downsample,
                                              frame=frame)


if __name__ == '__main__':

    img = torch.randn(2, 512, 64, 8, 8)
    net = MulScaleBlockTC(512, 512, stride=1, downsample=None, frame=64)
    out = net(img)
    print(out.size())














##############################################################
# out_1_1 = self.conv1_2_1(sp_x[0])
# out_1_1 = self.bn1_2_1(out_1_1)
# out_1_1_relu = self.relu(out_1_1)
# out_1_2 = self.conv1_2_2(out_1_1_relu + sp_x[1])
#
# out_1_2 = self.bn1_2_2(out_1_2)
# out_1_2_relu = self.relu(out_1_2)
# out_1_3 = self.conv1_2_3(out_1_2_relu + sp_x[2])
# out_1_3 = self.bn1_2_3(out_1_3)
# out_1_3_relu = self.relu(out_1_3)
# out_1_4 = self.conv1_2_4(out_1_3_relu + sp_x[3])
# out_1_4 = self.bn1_2_4(out_1_4)
# output_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=2)
# print("output_1", output_1.size())
#
# out_2_1 = self.conv2_2_1(sp_x[0])
# out_2_1 = self.bn2_2_1(out_2_1)
# out_2_1_relu = self.relu(out_2_1)
# out_2_2 = self.conv2_2_2(out_2_1_relu + sp_x[1])
# out_2_2 = self.bn2_2_2(out_2_2)
# out_2_2_relu = self.relu(out_2_2)
# out_2_3 = self.conv2_2_3(out_2_2_relu + sp_x[2])
# out_2_3 = self.bn2_2_3(out_2_3)
# out_2_3_relu = self.relu(out_2_3)
# out_2_4 = self.conv2_2_4(out_2_3_relu + sp_x[3])
# out_2_4 = self.bn2_2_4(out_2_4)
# output_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=2)

#########################################################################