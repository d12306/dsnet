import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, ResBlock, SSPBlock2, SSPBlock3
from .block import RKBlock2, ArkBlock
from .cifar10 import *


class CIFAR10Module_ARK_Dirichlet(nn.Module):
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
        super(CIFAR10Module_ARK_Dirichlet, self).__init__()
        channel = init_channel
        self.args = args
        self.conv = conv3x3(3, channel)
        self.blocks = nn.ModuleList()
        self.mask = None
        self.arch_param = [Variable(
            1e-3 * torch.randn(self.args.num_op, ).cuda(), 
            requires_grad=True)
        torch.nn.init.kaiming_normal_(self.arch_param, mode='fan_in')
        for index in range(self.args.depth):
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
            if not self.args.is_quick:
                if index != self.args.depth - 1:
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
            else:
                if index != self.args.depth - 1:
                    self.blocks.append(
                        self._subsample(
                            channel, channel * 2,
                            norm_type=norm_type,
                            block_type=downsample_type).cuda())
                    channel *= 2
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, num_classes)

    def process_step_vector(self, x, method, mask, tau=None):
        if method == 'softmax':
            output = F.softmax(x, dim=-1)
        elif method == 'dirichlet':
            if self.args.is_score_function:
                output = torch.distributions.dirichlet.Dirichlet(
                    F.elu(x) + 1).sample()
            else:
                output = torch.distributions.dirichlet.Dirichlet(
                    F.elu(x) + 1).rsample()
        elif method == 'gumbel':
            output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)
        # import ipdb; ipdb.set_trace()
        # assert output.sum() == 1
        if mask is None:
            return output
        else:
            output_pruned = torch.zeros_like(output)
            output_pruned[mask] = output[mask]
            output_pruned /= output_pruned.sum()
            assert (output_pruned[~mask] == 0.0).all()
            return output_pruned

    def process_step_matrix(self, x, method, mask, tau=None):
        weights = []
        if mask is None:
            for line in x:
                weights.append(self.process_step_vector(
                    line, method, None, tau))
        else:
            for i, line in enumerate(x):
                weights.append(self.process_step_vector(
                    line, method, mask[i], tau))
        return torch.stack(weights)


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
            return ResBlock(inplanes,
                            planes,
                            stride=stride,
                            downsample=downsample,
                            norm_type=norm_type)
        return BasicBlock(inplanes,
                          planes,
                          stride=stride,
                          downsample=downsample,
                          norm_type=norm_type,
                          a21=a21,
                          b10=b10,
                          a_logic=a_logic,
                          b_logic=b_logic)

    def forward(self, x, arch_param_test=None):
        out = self.conv(x)
        if arch_param_test == None:
            if self.args.is_fixed:
                arch_param = torch.ones_like(self.arch_param).cuda() / self.args.num_op
            elif self.args.is_softmax:
                arch_param = self.process_step_matrix(
                    self.arch_param, 'softmax', self.mask)
            elif self.args.is_gumbel:
                arch_param = self.process_step_matrix(
                    self.arch_param, 'gumbel', self.mask)
            else:
                arch_param = self.process_step_matrix(
                    self.arch_param, 'dirichlet', self.mask)
            for index in range(self.args.depth):
                modules = [self.blocks[7 * index](out), self.blocks[7 * index + 1](out),
                        self.blocks[7 * index +
                                    2](out), self.blocks[7 * index + 3](out),
                        self.blocks[7 * index +
                                    4](out), self.blocks[7 * index + 5](out)]
                out = self._weighted_sum(
                    modules, arch_param[index])
                if index != self.args.depth - 1:
                    out = self.blocks[7 * index + 6](out)
                print(out.shape)
        else:
            for index in range(self.args.depth):
                modules = [self.blocks[7 * index](out), self.blocks[7 * index + 1](out),
                        self.blocks[7 * index +
                                    2](out), self.blocks[7 * index + 3](out),
                        self.blocks[7 * index +
                                    4](out), self.blocks[7 * index + 5](out)]
                out = self._weighted_sum(
                    modules, arch_param_test[index])
                if index != self.args.depth - 1:
                    out = self.blocks[7 * index + 6](out)
                # print(out.shape)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def calculate_arch_param_test(self):
        if self.args.is_fixed:
            arch_param = torch.ones_like(self.arch_param).cuda() / self.args.num_op
        elif self.args.is_softmax:
            arch_param = self.process_step_matrix(
                self.arch_param, 'softmax', self.mask)
        elif self.args.is_gumbel:
            arch_param = self.process_step_matrix(
                self.arch_param, 'gumbel', self.mask)
        else:
            arch_param = self.process_step_matrix(
                self.arch_param, 'dirichlet', self.mask)
        return arch_param

    def _weighted_sum(self, input, weight):
        for index in range(len(input)):
            if index == 0:
                out = input[index] * \
                    weight[index]
            else:
                out += input[index] * \
                    weight[index]
        return out