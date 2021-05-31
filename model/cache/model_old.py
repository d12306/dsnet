import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, ResBlock, SSPBlock2, SSPBlock3
from .block import RKBlock2, ArkBlock
from .cifar10 import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            in_features, out_features).cuda())
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        identity = torch.eye(adj.size(0)).cuda()
        adj = adj + identity
        temp = torch.sum(adj, 1).pow(-1)
        temp[temp != temp] = 0
        D = torch.diag(temp)
        adj = torch.mm(D, adj)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class CIFAR10Module_ARK_Adaptive(nn.Module):
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
        super(CIFAR10Module_ARK_Adaptive, self).__init__()
        channel = init_channel
        self.args = args
        self.conv = conv3x3(3, channel)
        self.blocks = nn.ModuleList()
        self.mask = None
        # self.sigma = self.args.sigma
        if self.args.is_softmax or self.args.is_gumbel:
            self.arch_param = Variable(
            1e-3 * torch.randn(self.args.depth, self.args.num_op).cuda(), 
            requires_grad=True)
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
        self.fc1 = nn.Linear(self.args.graph_hidden_dim, 1)
        self.GCN = GraphConvolution(
            self.args.graph_hidden_dim, self.args.graph_hidden_dim)
        self.get_predictors()

    def get_predictors(self):
        self.transformations = []
        shape1 = self.blocks[0][0].conv1.weight.shape[1] * self.blocks[0][0].conv1.weight.shape[2] *self.blocks[0][0].conv1.weight.shape[3]
        self.transformation1 = nn.Linear(shape1 * 2, self.args.graph_hidden_dim).cuda()
        self.transformations.append(self.transformation1)

        shape2 = self.blocks[7][0].conv1.weight.shape[1] * self.blocks[7][0].conv1.weight.shape[2] *self.blocks[7][0].conv1.weight.shape[3]
        self.transformation2 = nn.Linear(shape2 * 2, self.args.graph_hidden_dim).cuda()
        self.transformations.append(self.transformation2)

        shape3 = self.blocks[14][0].conv1.weight.shape[1] * self.blocks[14][0].conv1.weight.shape[2] *self.blocks[14][0].conv1.weight.shape[3]
        self.transformation3 = nn.Linear(shape3 * 2, self.args.graph_hidden_dim).cuda()
        self.transformations.append(self.transformation3)

        shape4 = self.blocks[21][0].conv1.weight.shape[1] * self.blocks[21][0].conv1.weight.shape[2] *self.blocks[21][0].conv1.weight.shape[3]
        self.transformation4 = nn.Linear(shape4 * 2, self.args.graph_hidden_dim).cuda()
        self.transformations.append(self.transformation4)

        shape5 = self.blocks[28][0].conv1.weight.shape[1] * self.blocks[28][0].conv1.weight.shape[2] *self.blocks[28][0].conv1.weight.shape[3]
        self.transformation5 = nn.Linear(shape5 * 2, self.args.graph_hidden_dim).cuda()
        self.transformations.append(self.transformation5)


        self.transformation_back = []
        self.transformation_back1 = nn.Linear(self.args.graph_hidden_dim, 
                                        self.args.init_channel).cuda()
        self.transformation_back.append(self.transformation_back1)

        self.transformation_back2 = nn.Linear(self.args.graph_hidden_dim, 
                                            2 * self.args.init_channel).cuda()
        self.transformation_back.append(self.transformation_back2)

        self.transformation_back3 = nn.Linear(self.args.graph_hidden_dim, 
                                            4 * self.args.init_channel).cuda()
        self.transformation_back.append(self.transformation_back3)

        self.transformation_back4 = nn.Linear(self.args.graph_hidden_dim, 
                                            8*self.args.init_channel).cuda()
        self.transformation_back.append(self.transformation_back4)

        self.transformation_back5 = nn.Linear(self.args.graph_hidden_dim, 
                                            16 * self.args.init_channel).cuda()
        self.transformation_back.append(self.transformation_back5)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def graph_reasoning(self):
        graph_node = []
        distribution_weight = []
        for index in range(self.args.depth):
            # for index_op in range(self.args.num_op):
            weight_this_layer = []
            weight1 = torch.cat([torch.mean(self.blocks[7 * index][0].conv1.weight, 1).view(-1),
            torch.mean(self.blocks[7 * index][0].conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight1)

            weight2 = torch.cat([torch.mean(self.blocks[7 * index + 1][0].conv1.weight, 1).view(-1),
            torch.mean(self.blocks[7 * index + 1][0].conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight2)

            weight3 = torch.cat([torch.mean(self.blocks[7 * index + 2][0].block.conv1.weight, 1).view(-1),
            torch.mean(self.blocks[7 * index + 2][0].block.conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight3)
            # import ipdb; ipdb.set_trace()
            weight4 = torch.cat([torch.mean(self.blocks[7 * index + 3][0].block.conv1.weight, 1).view(-1),
            torch.mean(self.blocks[7 * index + 3][0].block.conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight4)

            weight5 = torch.cat([torch.mean(self.blocks[7 * index + 4][0].conv1.weight, 1).view(-1),
            torch.mean(self.blocks[7 * index + 4][0].conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight5)

            weight6 = torch.cat([torch.mean(self.blocks[7 * index + 5][0].conv1.weight, 1).view(-1),
            torch.mean(self.blocks[7 * index + 5][0].conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight6)

            input_for_transformation = torch.stack(weight_this_layer)
            output_for_transformation = self.transformations[index](input_for_transformation)
            graph_node.append(output_for_transformation)
        graph_node = torch.stack(graph_node)
        graph_node = graph_node.view(-1, self.args.graph_hidden_dim)
        # 5, 6, 128
        # import ipdb; ipdb.set_trace()
        graph_left = graph_node.unsqueeze(0).repeat(self.args.num_op * self.args.depth, 1, 1)
        graph_right = graph_node.unsqueeze(1).repeat(1, self.args.num_op * self.args.depth, 1)
        cross_graph = F.softmax(-torch.sum(
            (graph_left - graph_right)**2, dim=2) / (2.0 * self.sigma), 1)#
        # cross_graph = torch.sigmoid(
        #     self.fc1(torch.abs(graph_left - graph_right))).squeeze()
        enhanced_feat = self.GCN(graph_node, cross_graph)
        for index in range(self.args.depth):
            weight_this_layer = self.transformation_back[index](enhanced_feat[6 * index: 6 * index + 6])
            weight_this_layer = F.softmax(weight_this_layer / (2.0 * self.sigma), 0).view(-1)
            distribution_weight.append(weight_this_layer)
            #import ipdb; ipdb.set_trace()
        return distribution_weight


    def process_step_vector(self, x, method, mask, tau=5):
        if method == 'softmax':
            output = F.softmax(x, dim=0)
        elif method == 'dirichlet':
            if self.args.is_score_function:
                output = torch.distributions.dirichlet.Dirichlet(
                    F.elu(x) + 1).sample()
            else:
                output = torch.distributions.dirichlet.Dirichlet(
                    F.elu(x) + 1).rsample()
        elif method == 'gumbel':
            output = F.gumbel_softmax(x, tau=10, hard=False, dim=0)
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
                graph_attention_weights = torch.ones(self.args.depth,
                 self.args.num_op).cuda() / self.args.num_op
            elif self.args.is_softmax:
                graph_attention_weights = self.process_step_matrix(
                    self.arch_param, 'softmax', self.mask)
            elif self.args.is_gumbel:
                graph_attention_weights = self.process_step_matrix(
                    self.arch_param, 'gumbel', self.mask)
            else:
                # print('mark1')
                graph_attention_weights = self.graph_reasoning()
                # print('mark2')
            for index in range(self.args.depth):
                modules = [self.blocks[7 * index](out), self.blocks[7 * index + 1](out),
                        self.blocks[7 * index +
                                    2](out), self.blocks[7 * index + 3](out),
                        self.blocks[7 * index +
                                    4](out),
                                    self.blocks[7 * index +
                                    5](out)]
                out = self._weighted_sum(
                    modules, graph_attention_weights[index])
                if self.noise:
                    out = out + 0.1 * torch.std(out) * torch.randn_like(out)
                if index != self.args.depth - 1:
                    out = self.blocks[7 * index + 6](out)
                if self.noise:
                    out = out + 0.1 * torch.std(out) * torch.randn_like(out)
                # print(out.shape)
            # print('mark3')
        else:
            for index in range(self.args.depth):
                modules = [self.blocks[7 * index](out), self.blocks[7 * index + 1](out),
                        self.blocks[7 * index +
                                    2](out), self.blocks[7 * index + 3](out),
                        self.blocks[7 * index +
                                    4](out),
                                    self.blocks[7 * index +
                                    5](out)]
                out = self._weighted_sum(
                    modules, arch_param_test[index])
                if self.noise:
                    out = out + 0.1 * torch.std(out) * torch.randn_like(out)
                if index != self.args.depth - 1:
                    out = self.blocks[7 * index + 6](out)
                if self.noise:
                    out = out + 0.1 * torch.std(out) * torch.randn_like(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def calculate_arch_param_test(self):
        if self.args.is_fixed:
            graph_attention_weights = torch.ones(self.args.depth,
                self.args.num_op).cuda() / self.args.num_op
        elif self.args.is_softmax:
            graph_attention_weights = self.process_step_matrix(
                self.arch_param, 'softmax', self.mask)
        elif self.args.is_gumbel:
            graph_attention_weights = self.process_step_matrix(
                self.arch_param, 'gumbel', self.mask)
        else:
            graph_attention_weights = self.graph_reasoning()
        return graph_attention_weights

    def _weighted_sum(self, input, weight):
        for index in range(len(input)):
            if index == 0:
                out = input[index] * \
                    weight[index]#.view(1, -1, 1, 1)
            else:
                out += input[index] * \
                    weight[index]#.view(1, -1, 1, 1)
        return out
    
    def set_noise_true(self):
        self.noise=True

    def set_noise_false(self):
        self.noise=False
