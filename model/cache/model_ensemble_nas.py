import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .block import conv3x3, conv1x1, norm
from .block import BasicBlock, ResBlock, SSPBlock2, SSPBlock3
from .block import RKBlock2, ArkBlock
from .cifar10 import *
from .efficientnet import MBConvBlock
from .densenet import _DenseBlock


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
        # self.temperature = nn.Parameter(torch.randn(1), requires_grad=True).cuda()
        # self.sigma = self.args.sigma
        if args.is_normal:
            if self.args.is_softmax or self.args.is_gumbel or args.is_diri:
                # normal distribution.
                self.arch_param = Variable(
                1e-3 * torch.randn(self.args.depth, self.args.num_op).cuda(), 
                requires_grad=True)
        elif args.is_uniform:
            if self.args.is_softmax or self.args.is_gumbel or args.is_diri:
                # normal distribution.
                self.arch_param = Variable(
                1e-3 * torch.rand(self.args.depth, self.args.num_op).cuda(), 
                requires_grad=True)
        elif args.is_log_normal:
            if self.args.is_softmax or self.args.is_gumbel or args.is_diri:
                # normal distribution.
                self.arch_param = Variable(
                1e-3 * torch.rand(self.args.depth,
                self.args.num_op).log_normal_().cuda(), 
                requires_grad=True)
        elif args.is_exponential:
            if self.args.is_softmax or self.args.is_gumbel or args.is_diri:
                # normal distribution.
                self.arch_param = Variable(
                1e-3 * torch.rand(self.args.depth,
                self.args.num_op).exponential_().cuda(), 
                requires_grad=True)
        elif args.is_geometric:
            if self.args.is_softmax or self.args.is_gumbel or args.is_diri:
                # normal distribution.
                self.arch_param = Variable(
                1e-3 * torch.rand(self.args.depth,
                self.args.num_op).geometric_(0.5).cuda(), 
                requires_grad=True)
        elif args.is_trained:
            if self.args.is_softmax or self.args.is_gumbel or args.is_diri:
                # normal distribution.
                temp_dis = torch.Tensor([[-3.8046e-02, -4.0694e-02,  1.8031e-01, -5.9434e-02],
                                        [ 3.0606e-02,  3.3249e-04,  1.5231e-02, -2.3586e-02],
                                        [ 5.6491e-02, -2.2726e-02, -8.9184e-02,  1.0278e-01],
                                        [ 5.5741e-04, -2.5018e-02,  4.2670e-02,  3.5855e-02],
                                        [ 2.9131e-02, -2.3679e-02,  4.5026e-02, -1.9753e-03],
                                        [ 2.9233e-02, -5.6670e-02, -2.2855e-02,  1.0332e-01],
                                        [-3.2084e-02, -9.9476e-02,  8.6322e-02,  8.0716e-02],
                                        [ 3.2483e-02, -7.4212e-02,  6.6345e-02,  4.0340e-02],
                                        [ 7.5314e-02, -8.5606e-02,  4.5782e-02, -3.2868e-03],
                                        [ 9.0318e-02, -1.0605e-01,  2.4147e-02,  3.2868e-02],
                                        [ 1.2734e-01, -8.2794e-02, -1.3688e-01,  1.2983e-01],
                                        [ 1.3050e-01, -1.1210e-01,  3.1331e-02,  7.6262e-02],
                                        [ 2.3352e-01, -1.6725e-01,  6.8595e-02, -3.9627e-02],
                                        [ 1.3590e-01, -1.7111e-01, -2.1120e-02,  1.1846e-01],
                                        [ 1.1671e+00, -1.2097e+00, -2.7491e-01, -2.9023e-01]])
                self.arch_param = Variable(temp_dis.cuda(), requires_grad=True)

        for index in range(self.args.depth):
            # self.blocks.append(nn.Sequential(
            #     *[ResBlock(channel, channel,
            #                    norm_type=norm_type,
            #                    a21=a21, b10=b10,
            #                    a_logic=a_logic,
            #                    b_logic=b_logic).cuda() for _ in range(layers)]
            # )),
            self.blocks.append(
                nn.Sequential(
                *[MBConvBlock(channel, channel,
                               expand_ratio=2, stride=1, kernel_size=3).cuda() for _ in range(layers)]
            )),
            self.blocks.append(
                nn.Sequential(
                *[MBConvBlock(channel, channel,
                               expand_ratio=2, stride=1, kernel_size=3).cuda() for _ in range(layers)]
            )),
            self.blocks.append(
                nn.Sequential(
                *[MBConvBlock(channel, channel,
                               expand_ratio=2, stride=1, kernel_size=3).cuda() for _ in range(layers)]
            )),
            self.blocks.append(
                nn.Sequential(
                *[MBConvBlock(channel, channel,
                               expand_ratio=2, stride=1, kernel_size=3).cuda() for _ in range(layers)]
            )),
            # self.blocks.append(
            #      nn.Sequential(
            #     *[_DenseBlock(
            #     num_layers=2,
            #     num_input_features=channel,
            #     bn_size=4,
            #     growth_rate=channel,
            #     drop_rate=0,
            #     memory_efficient=False).cuda() for _ in range(layers)]
            # )),
            # self.blocks.append(nn.Sequential(
            #     *[ArkBlock(channel, channel,
            #                    norm_type=norm_type,
            #                    a21=a21, b10=b10,
            #                    a_logic=a_logic,
            #                    b_logic=b_logic).cuda() for _ in range(layers)]))
            if (index + 1) == int(2/3*self.args.depth):
            # if index != self.args.depth - 1:
                self.blocks.append(
                    self._subsample(
                        channel, channel * 2,
                        norm_type=norm_type,
                        block_type=downsample_type).cuda())
                channel = 2 * channel
            if (index + 1) == int(1/3*self.args.depth):
                self.blocks.append(
                    self._subsample(
                        channel, channel * self.args.factor,
                        norm_type=norm_type,
                        block_type=downsample_type).cuda())
                channel = channel * self.args.factor

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, num_classes)
        # self.fc1 = nn.Linear(self.args.graph_hidden_dim, 1)
        self.GCN = GraphConvolution(
            self.args.graph_hidden_dim, self.args.graph_hidden_dim)
        self.graph_bn = torch.nn.BatchNorm1d(self.args.graph_hidden_dim)
        # self.get_predictors()
        #self.init_param()

    def init_param(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def get_predictors(self):
        self.transformations = nn.ModuleList()
        self.transformation_back = nn.ModuleList()
        for index in range(self.args.depth):
            if index < int(1/3*self.args.depth):
                bias = 0
            elif index < int(2/3*self.args.depth):
                bias = 1
            else:
                bias = 2
            ################ for the resnet block.
            shape1 = self.blocks[index * self.args.num_op + bias][0].conv1.weight.shape[0] \
                * self.blocks[index * self.args.num_op + bias][0].conv1.weight.shape[2] \
                * self.blocks[index * self.args.num_op + bias][0].conv1.weight.shape[3]
            transformation1 = nn.Linear(shape1 * 2, self.args.graph_hidden_dim).cuda()
            self.transformations.append(transformation1)

            ############### for the efficient net block.
            shape1 = self.blocks[index * self.args.num_op + bias + 1][0].pre_layer.conv1.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 1][0].pre_layer.conv1.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 1][0].pre_layer.conv1.weight.shape[3] 
            
            shape2 = self.blocks[index * self.args.num_op + bias + 1][0].se.conv1.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 1][0].se.conv1.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 1][0].se.conv1.weight.shape[3]
            
            shape3 = self.blocks[index * self.args.num_op + bias + 1][0].se.conv2.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 1][0].se.conv2.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 1][0].se.conv2.weight.shape[3]

            shape4 = self.blocks[index * self.args.num_op + bias + 1][0].linear.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 1][0].linear.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 1][0].linear.weight.shape[3]

            shape5 = self.blocks[index * self.args.num_op + bias + 1][0].dw.conv1.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 1][0].dw.conv1.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 1][0].dw.conv1.weight.shape[3] 

            transformation1 = nn.Linear(shape1 + shape2 + shape3 + shape4 + shape5, 
            self.args.graph_hidden_dim).cuda()
            self.transformations.append(transformation1)

            ############## for the dense-net block.
            shape1 = self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv1.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv1.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv1.weight.shape[3] 

            shape2 = self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv2.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv2.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv2.weight.shape[3] 

            shape3 = self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv1.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv1.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv1.weight.shape[3] 

            shape4 = self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv2.weight.shape[0] * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv2.weight.shape[2]  * \
            self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv2.weight.shape[3] 

            transformation1 = nn.Linear(shape1 + shape2 + shape3 + shape4, 
            self.args.graph_hidden_dim).cuda()
            self.transformations.append(transformation1)


            ############### for the RK block.
            shape1 = self.blocks[index * self.args.num_op + bias + 3][0].conv1.weight.shape[0] \
                * self.blocks[index * self.args.num_op + bias + 3][0].conv1.weight.shape[2] \
                * self.blocks[index * self.args.num_op + bias + 3][0].conv1.weight.shape[3]
            transformation1 = nn.Linear(shape1 * 2, self.args.graph_hidden_dim).cuda()
            self.transformations.append(transformation1)
            transformation_back1 = nn.Linear(self.args.graph_hidden_dim, 
                                        1).cuda()
            self.transformation_back.append(transformation_back1)


            # if index < int(1/3*self.args.depth):
            #     shape1 = self.blocks[0][0].conv1.weight.shape[1] \
            #     * self.blocks[0][0].conv1.weight.shape[2] \
            #     * self.blocks[0][0].conv1.weight.shape[3]
            #     transformation1 = nn.Linear(shape1 * 2, self.args.graph_hidden_dim).cuda()
            #     self.transformations.append(transformation1)
            #     transformation_back1 = nn.Linear(self.args.graph_hidden_dim, 
            #                                 1).cuda()
            #     self.transformation_back.append(transformation_back1)
            # elif index < int(2/3*self.args.depth):
            #     shape1 = self.blocks[int(1/3*self.args.depth) * self.args.num_op + 1][0].conv1.weight.shape[1] \
            #     * self.blocks[int(1/3*self.args.depth) * self.args.num_op + 1][0].conv1.weight.shape[2] \
            #     * self.blocks[int(1/3*self.args.depth) * self.args.num_op + 1][0].conv1.weight.shape[3]
            #     transformation1 = nn.Linear(shape1 * 2, self.args.graph_hidden_dim).cuda()
            #     self.transformations.append(transformation1)
            #     transformation_back1 = nn.Linear(self.args.graph_hidden_dim, 
            #                                 1).cuda()
            #     self.transformation_back.append(transformation_back1)
            # else:
            #     shape1 = self.blocks[int(2/3*self.args.depth) * self.args.num_op + 2][0].conv1.weight.shape[1] \
            #     * self.blocks[int(2/3*self.args.depth) * self.args.num_op + 2][0].conv1.weight.shape[2] \
            #     * self.blocks[int(2/3*self.args.depth) * self.args.num_op + 2][0].conv1.weight.shape[3]
            #     transformation1 = nn.Linear(shape1 * 2, self.args.graph_hidden_dim).cuda()
            #     self.transformations.append(transformation1)
            #     transformation_back1 = nn.Linear(self.args.graph_hidden_dim, 
            #                                 1).cuda()
            #     self.transformation_back.append(transformation_back1)

    def set_temperature(self, sigma):
        self.temperature = sigma

    def graph_reasoning(self):
        graph_node = []
        distribution_weight = []
        for index in range(self.args.depth):
            # for index_op in range(self.args.num_op):
            weight_this_layer = []
            if index < int(1/3*self.args.depth):
                bias = 0
            elif index < int(2/3*self.args.depth):
                bias = 1
            else:
                bias = 2
            ############## for the resent block.
            weight3 = torch.cat([torch.mean(self.blocks[self.args.num_op * index + bias][0].conv1.weight, 1).view(-1),
            torch.mean(self.blocks[self.args.num_op * index + bias][0].conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight3)
            ############## for the efficient-net block.
            weight4 = torch.cat([torch.mean(self.blocks[self.args.num_op * index + bias + 1][0].pre_layer.conv1.weight, 1).view(-1),
            torch.mean(self.blocks[self.args.num_op * index + bias + 1][0].se.conv1.weight, 1).view(-1)])
            weight4 = torch.cat([weight4, 
             torch.mean(self.blocks[self.args.num_op * index + bias + 1][0].se.conv2.weight, 1).view(-1)])
            weight4 = torch.cat([weight4, 
             torch.mean(self.blocks[self.args.num_op * index + bias + 1][0].linear.weight, 1).view(-1)])
            weight4 = torch.cat([weight4, 
             torch.mean(self.blocks[self.args.num_op * index + bias + 1][0].dw.conv1.weight, 1).view(-1)])
            weight_this_layer.append(weight4)
            ############## for the densenet block.
            weight5 = torch.cat([torch.mean(self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv1.weight, 1).view(-1),
            torch.mean(self.blocks[index * self.args.num_op + bias + 2][0].denselayer1.conv2.weight, 1).view(-1)])
            weight5 = torch.cat([weight5, 
            torch.mean(self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv1.weight, 1).view(-1)])
            weight5 = torch.cat([weight5,
            torch.mean(self.blocks[index * self.args.num_op + bias + 2][0].denselayer2.conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight5)
            ############# for the RK block.
            weight6 = torch.cat([torch.mean(self.blocks[self.args.num_op * index + bias + 3][0].conv1.weight, 1).view(-1),
            torch.mean(self.blocks[self.args.num_op * index + bias + 3][0].conv2.weight, 1).view(-1)])
            weight_this_layer.append(weight6)


            output_for_transformation = []
            output_for_transformation.append(self.transformations[self.args.num_op * index](weight_this_layer[0]))
            output_for_transformation.append(self.transformations[self.args.num_op * index + 1](weight_this_layer[1]))
            output_for_transformation.append(self.transformations[self.args.num_op * index + 2](weight_this_layer[2]))
            output_for_transformation.append(self.transformations[self.args.num_op * index + 3](weight_this_layer[3]))
            graph_node.append(F.relu(self.graph_bn(torch.stack(output_for_transformation))))
        # print(graph_node)
        graph_node = torch.stack(graph_node)
        graph_node = graph_node.view(-1, self.args.graph_hidden_dim)
        # 5, 6, 128
        # import ipdb; ipdb.set_trace()
        graph_left = graph_node.unsqueeze(0).repeat(self.args.num_op * self.args.depth, 1, 1)
        graph_right = graph_node.unsqueeze(1).repeat(1, self.args.num_op * self.args.depth, 1)
        cross_graph = F.softmax(-torch.sum(
            (graph_left - graph_right)**2, dim=2) / (2.0 * self.sigma), 1)
        enhanced_feat = self.GCN(graph_node, cross_graph)
        for index in range(self.args.depth):
            weight_this_layer = self.transformation_back[index](enhanced_feat[
                self.args.num_op * index : self.args.num_op * index + self.args.num_op])
            weight_this_layer = F.softmax(weight_this_layer / (2.0 * self.sigma), 0).view(-1)
            distribution_weight.append(weight_this_layer)
        # print(distribution_weight)
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
            # Sample soft categorical using reparametrization trick:
            # import ipdb; ipdb; ipdb.set_trace()
            output = F.gumbel_softmax(x, tau=self.temperature, hard=False, dim=0)
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
            elif self.args.is_diri:
                graph_attention_weights = self.process_step_matrix(
                self.arch_param, 'dirichlet', self.mask)
            else:
                # print('mark1')
                graph_attention_weights = self.graph_reasoning()
                # print('mark2')
            for index in range(self.args.depth):
                if index < int(1/3*self.args.depth):
                    bias = 0
                elif index < int(2/3*self.args.depth):
                    bias = 1
                else:
                    bias = 2
                modules = [self.blocks[self.args.num_op * index + bias](out), 
                        self.blocks[self.args.num_op * index + bias + 1](out),
                        self.blocks[self.args.num_op * index + bias + 2](out),
                        self.blocks[self.args.num_op * index + bias + 3](out)]
                out = self._weighted_sum(
                    modules, graph_attention_weights[index])
                ### debug #########
                # out = graph_attention_weights[index][0] * self.blocks[self.args.num_op * index + bias](out) + \
                # graph_attention_weights[index][1] * self.blocks[self.args.num_op * index + bias + 1](out) + \
                # graph_attention_weights[index][2] * self.blocks[self.args.num_op * index + bias + 2](out) + \
                # graph_attention_weights[index][3] * self.blocks[self.args.num_op * index + bias + 3](out)
                ##### end ####
                if index == int(1/3*self.args.depth) - 1 or index == int(2/3*self.args.depth) - 1:
                    out = self.blocks[self.args.num_op * (index + 1) + bias](out)
                # print(out.shape)
            # print('mark3')
        else:
            for index in range(self.args.depth):
                if index < int(1/3*self.args.depth):
                    bias = 0
                elif index < int(2/3*self.args.depth):
                    bias = 1
                else:
                    bias = 2
                modules = [self.blocks[self.args.num_op * index + bias](out), 
                        self.blocks[self.args.num_op * index + bias + 1](out),
                        self.blocks[self.args.num_op * index + bias + 2](out),
                        self.blocks[self.args.num_op * index + bias + 3](out)]
                out = self._weighted_sum(
                    modules, arch_param_test[index])
                if index == int(1/3*self.args.depth) - 1 or index == int(2/3*self.args.depth) - 1:
                    out = self.blocks[self.args.num_op * (index + 1) + bias](out)
                # print(out.shape)
            # print('mark3')
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
        elif self.args.is_diri:
            graph_attention_weights = self.process_step_matrix(
                self.arch_param, 'dirichlet', self.mask)
        else:
            graph_attention_weights = self.graph_reasoning()
        return graph_attention_weights

    def _weighted_sum(self, input, weight):
        out = 0.
        for index in range(len(input)):
            out = out + input[index] * weight[index]#.view(1, -1, 1, 1)
        return out