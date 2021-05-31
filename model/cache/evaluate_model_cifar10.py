import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, ResBlock, SSPBlock2, SSPBlock3
from .block import RKBlock2, ArkBlock
from .cifar10 import *


class CIFAR10Module_ARK_evaluate(nn.Module):
    def __init__(
            self,
            args,
            genotype,
            layers=3,
            num_classes=10,
            init_channel=16,
            norm_type="b",
            downsample_type="r",
            a21=-1.0,
            b10=0.5,
            a_logic=False,
            b_logic=True):
        super(CIFAR10Module_ARK_evaluate, self).__init__()
        channel = init_channel
        self.args = args
        self.conv = conv3x3(3, channel)
        self.num_op = len(genotype)
        self.blocks = nn.ModuleList()

        for index in range(self.num_op):
            self.blocks.append(self.dict_to_model(
                channel, genotype[index], layers, norm_type, a21, b10, a_logic, b_logic))
            if index != self.num_op - 1:
                if index < 2:
                    self.blocks.append(
                        self._subsample(
                            channel, channel,
                            stride=1,
                            norm_type=norm_type,
                            block_type=downsample_type).cuda())
                else:
                    self.blocks.append(
                        self._subsample(
                            channel, channel * 2,
                            norm_type=norm_type,
                            block_type=downsample_type).cuda())
                    channel *= 2

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, num_classes)

    def _subsample(
            self,
            inplanes,
            planes,
            stride=2,
            norm_type="b",
            block_type="r",
            a21=-1.0,
            b10=0.5,
            a_logic=False,
            b_logic=True):
        downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            norm(planes, norm_type=norm_type)
        )
        # only supports the ResBlock
        if block_type == "r":
            return ResBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type)
        return BasicBlock(inplanes, planes, stride=stride, downsample=downsample, norm_type=norm_type, a21=a21, b10=b10, a_logic=a_logic, b_logic=b_logic)

    def forward(self, x):
        out = self.conv(x)
        for index in range(len(self.blocks)):
            out = self.blocks[index](out)
            # print(out.shape)
        # import ipdb; ipdb.set_trace()
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def dict_to_model(self,
                      channel,
                      index,
                      layers,
                      norm_type,
                      a21,
                      b10,
                      a_logic,
                      b_logic):
        if index == 'conv_block':
            return nn.Sequential(
            *[ConvBlock(channel, channel,
                        norm_type=norm_type,
                        a21=a21, b10=b10,
                        a_logic=a_logic,
                        b_logic=b_logic).cuda() for _ in range(layers)]
            )
        elif index == 'resblock':
            return nn.Sequential(
            *[ResBlock(channel, channel,
                       norm_type=norm_type,
                       a21=a21, b10=b10,
                       a_logic=a_logic,
                       b_logic=b_logic).cuda() for _ in range(layers)]
            )
        elif index == 'SSPBlock2':
            return nn.Sequential(
            *[SSPBlock2(channel, channel,
                        norm_type=norm_type,
                        a21=a21, b10=b10,
                        a_logic=a_logic,
                        b_logic=b_logic).cuda() for _ in range(layers)]
            )
        elif index == 'SSPBlock3':
            return nn.Sequential(
            *[SSPBlock3(channel, channel,
                        norm_type=norm_type,
                        a21=a21, b10=b10,
                        a_logic=a_logic,
                        b_logic=b_logic).cuda() for _ in range(layers)]
            )
        elif index == 'RKBlock2':
            return nn.Sequential(
            *[RKBlock2(channel, channel,
                       norm_type=norm_type,
                       a21=a21, b10=b10,
                       a_logic=a_logic,
                       b_logic=b_logic).cuda() for _ in range(layers)]
            )
        elif index == 'ArkBlock':
            return nn.Sequential(
            *[ArkBlock(channel, channel,
                       norm_type=norm_type,
                       a21=a21, b10=b10,
                       a_logic=a_logic,
                       b_logic=b_logic).cuda() for _ in range(layers)])
