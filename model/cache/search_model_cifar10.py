import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, ResBlock, SSPBlock2, SSPBlock3
from .block import RKBlock2, ArkBlock
from .cifar10 import *


class CIFAR10Module_ARK_search(nn.Module):
    def __init__(
            self,
            args,
            layers=3,
            num_classes=10,
            init_channel=16,
            norm_type="b",
            downsample_type="r",
            a21=-1.0,
            b10=0.5,
            a_logic=False,
            b_logic=True):
        super(CIFAR10Module_ARK_search, self).__init__()
        channel = init_channel
        self.args = args
        self.conv = conv3x3(3, channel)
        self.arch_param = Variable(
            1e-3 * torch.randn(self.args.depth, 6).cuda(),
            requires_grad=True).cuda()
        self.num_op = len(self.arch_param)
        self.blocks = nn.ModuleList()
        for index in range(self.num_op):
            # self.blocks.append(nn.Sequential()),
            self.blocks.append(nn.Sequential(
                *[ConvBlock(channel, channel,
                            norm_type=norm_type,
                            a21=a21, b10=b10,
                            a_logic=a_logic,
                            b_logic=b_logic).cuda() for _ in range(layers)]
            )),
            self.blocks.append(nn.Sequential(
                *[ResBlock(channel, channel,
                               norm_type=norm_type,
                               a21=a21, b10=b10,
                               a_logic=a_logic,
                               b_logic=b_logic).cuda() for _ in range(layers)]
            )),
            self.blocks.append(nn.Sequential(
                *[SSPBlock2(channel, channel,
                            norm_type=norm_type,
                            a21=a21, b10=b10,
                            a_logic=a_logic,
                            b_logic=b_logic).cuda() for _ in range(layers)]
            )),
            self.blocks.append(nn.Sequential(
                *[SSPBlock3(channel, channel,
                            norm_type=norm_type,
                            a21=a21, b10=b10,
                            a_logic=a_logic,
                            b_logic=b_logic).cuda() for _ in range(layers)]
            )),
            self.blocks.append(nn.Sequential(
                *[RKBlock2(channel, channel,
                               norm_type=norm_type,
                               a21=a21, b10=b10,
                               a_logic=a_logic,
                               b_logic=b_logic).cuda() for _ in range(layers)]
            )),
            self.blocks.append(nn.Sequential(
                *[ArkBlock(channel, channel,
                               norm_type=norm_type,
                               a21=a21, b10=b10,
                               a_logic=a_logic,
                               b_logic=b_logic).cuda() for _ in range(layers)]))
            if index != len(self.arch_param) - 1:
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
        # outs = [out]
        for index in range(self.num_op):
            modules = [self.blocks[7 * index](out), self.blocks[7 * index + 1](out),
                       self.blocks[7 * index +
                                   2](out), self.blocks[7 * index + 3](out),
                       self.blocks[7 * index +
                                   4](out), self.blocks[7 * index + 5](out)]
            out = self._weighted_sum(
                modules, F.softmax(self.arch_param, 1)[index])
            if index != self.num_op - 1:
                out = self.blocks[7 * index + 6](out)
            # print(out.shape)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _weighted_sum(self, input, weight):
        for index in range(len(input)):
            if index == 0:
                out = input[index] * weight[index]
            else:
                out += input[index] * weight[index]
        return out

    def genotypes(self):
        operations = [
            'conv_block',
            'resblock',
            'SSPBlock2',
            'SSPBlock3',
            'RKBlock2',
            'ArkBlock']
        structure = []
        maximum_indices = self.arch_param.argmax(1)
        for index in range(len(self.arch_param)):
            # import ipdb; ipdb.set_trace()
            structure.append(operations[maximum_indices[index]])
        return structure

    def new(self):
        model_new = CIFAR10Module_ARK_search(args=self.args,
                                             layers=self.args.block,
                                             num_classes=self.args.classes,
                                             init_channel=self.args.init_channel,
                                             norm_type=self.args.norm,
                                             downsample_type="r",
                                             a21=-1.0,
                                             b10=0.5,
                                             a_logic=False,
                                             b_logic=True).cuda()
        for x, y in zip(model_new.arch_param, self.arch_param):
            x.data.copy_(y.data)
        return model_new

    # def loss(self):
    #     return nn.CrossEntropyLoss()
    def _loss(self, input, target):
        logits = self.forward(input)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        return criterion(logits, target)

    def _TRADES_loss(self, adv_logits, natural_logits, target, beta):
        # Based on the repo TREADES: https://github.com/yaodongyu/TRADES
        batch_size = len(target)
        criterion_kl = nn.KLDivLoss(size_average=False).cuda()
        loss_natural = nn.CrossEntropyLoss(
            reduction='mean')(natural_logits, target)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                        F.softmax(natural_logits, dim=1))
        loss = loss_natural + beta * loss_robust
        return loss
