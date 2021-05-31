
"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1128.03385v1
"""

import torch
import torch.nn.utils.spectral_norm as snorm
import torch.nn as nn

class SIR_ResNet18(nn.Module):
    def __init__(self, wd=0.1, it=1,theta=1.0, num_classes=10):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))

        self.conv2 =  nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False))

        self.conv2_d =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False))  for i in range(it))
        
        self.conv3 =  nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False))

        self.conv3_d = nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)) for i in range(it))

        self.conv4 =  nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False))

        self.d4 = nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            )

        self.conv5 =  nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False))

        self.conv5_d = nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)) for i in range(it))

        self.conv6 =  nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))

        self.d6 = nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
            )

        self.conv7 =  nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))

        self.conv7_d =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)) for i in range(it))

        self.conv8 =  nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False))

        self.d8 = nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            )
# )

        self.conv9 =  nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False))

        self.conv9_d =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)) for i in range(it))
        self.theta =theta
        self.final_bn = nn.GroupNorm(1,128)
        self.final_relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.graph=False
        self.wd = wd
        self.it = it

    def forward(self, x):
        wd = self.wd
        it = self.it
        theta = self.theta
        for i in range(it):
            self.conv2_d[i][2].weight.data = self.conv2[2].weight.data
            self.conv2_d[i][0].weight.data = self.conv2[0].weight.data
            self.conv2_d[i][0].bias.data = self.conv2[0].bias.data
            self.conv2_d[i][5].weight.data = self.conv2[5].weight.data
            self.conv2_d[i][3].weight.data = self.conv2[3].weight.data
            self.conv2_d[i][3].bias.data = self.conv2[3].bias.data
            self.conv3_d[i][2].weight.data = self.conv3[2].weight.data
            self.conv3_d[i][0].weight.data = self.conv3[0].weight.data
            self.conv3_d[i][0].bias.data = self.conv3[0].bias.data
            self.conv3_d[i][5].weight.data = self.conv3[5].weight.data
            self.conv3_d[i][3].weight.data = self.conv3[3].weight.data
            self.conv3_d[i][3].bias.data = self.conv3[3].bias.data
            self.conv5_d[i][2].weight.data = self.conv5[2].weight.data
            self.conv5_d[i][0].weight.data = self.conv5[0].weight.data
            self.conv5_d[i][0].bias.data = self.conv5[0].bias.data
            self.conv5_d[i][5].weight.data = self.conv5[5].weight.data
            self.conv5_d[i][3].weight.data = self.conv5[3].weight.data
            self.conv5_d[i][3].bias.data = self.conv5[3].bias.data
            self.conv7_d[i][2].weight.data = self.conv7[2].weight.data
            self.conv7_d[i][0].weight.data = self.conv7[0].weight.data
            self.conv7_d[i][0].bias.data = self.conv7[0].bias.data
            self.conv7_d[i][5].weight.data = self.conv7[5].weight.data
            self.conv7_d[i][3].weight.data = self.conv7[3].weight.data
            self.conv7_d[i][3].bias.data = self.conv7[3].bias.data
            self.conv9_d[i][2].weight.data = self.conv9[2].weight.data
            self.conv9_d[i][0].weight.data = self.conv9[0].weight.data
            self.conv9_d[i][0].bias.data = self.conv9[0].bias.data
            self.conv9_d[i][5].weight.data = self.conv9[5].weight.data
            self.conv9_d[i][3].weight.data = self.conv9[3].weight.data
            self.conv9_d[i][3].bias.data = self.conv9[3].bias.data
        x1 = self.conv1(x)
        t2 = self.conv2(x1)
        x2 = x1 + t2
        with torch.set_grad_enabled(True):
            loss_y = torch.norm(x2-x1-(1-theta)*t2-theta*self.conv2_d[0](x2))**2
            temp = loss_y
            f_y = torch.autograd.grad(loss_y, x2, retain_graph=self.graph, create_graph=self.graph)
            x2 = x2 - wd*f_y[0]
        wd = self.wd
        for i in range(it-1):
            # if flag == False:
            #     break
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x2-x1-(1-theta)*t2-theta*self.conv2_d[i+1](x2))**2
                f_y = torch.autograd.grad(loss_y, x2, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x2 = x2 - wd*f_y[0]
            wd = self.wd
        t3 = self.conv3(x2)
        x3 = x2 + t3
        wd = self.wd
        with torch.set_grad_enabled(True):
            loss_y = torch.norm(x3-x2-(1-theta)*t3-theta*self.conv3_d[0](x3))**2
            # loss_y = torch.norm(x3-x2-self.conv3_d[0](x3))**2
            temp = loss_y
            f_y = torch.autograd.grad(loss_y, x3, retain_graph=self.graph, create_graph=self.graph)
            x3 = x3 - wd*f_y[0]
        wd = self.wd
        # print(torch.norm(x2),torch.norm(loss_y),torch.norm(f_y[0]))
        for i in range(it-1):
            # if flag == False:
            #     break
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x3-x2-(1-theta)*t3-theta*self.conv3_d[i+1](x3))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x3, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x3 = x3 - wd*f_y[0]
            wd = self.wd
        x4 = self.conv4(x3) + self.d4(x3)
        # x4.requires_grad=True
        # x4.retain_grad=True
        t5 = self.conv5(x4)
        x5 = x4 + t5
        wd = self.wd
        with torch.set_grad_enabled(True):
            loss_y = torch.norm(x5-x4-(1-theta)*t5-theta*self.conv5_d[0](x5))**2
            # loss_y = torch.norm(x5-x3-self.conv5_d[0](x5))**2
            temp = loss_y
            f_y = torch.autograd.grad(loss_y, x5, retain_graph=self.graph, create_graph=self.graph)
            flag = False
            x5 = x5 - wd*f_y[0]
        wd = self.wd
        for i in range(it-1):
            # if flag == False:
            #     break
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x5-x4-(1-theta)*t5-theta*self.conv5_d[i+1](x5))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x5, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x5 = x5 - wd*f_y[0]
            wd = self.wd
        x6 = self.conv6(x5) + self.d6(x5)
        t7 = self.conv7(x6)
        x7 = x6 + t7
        wd = self.wd
        with torch.set_grad_enabled(True):
            loss_y = torch.norm(x7-x6-(1-theta)*t7-theta*self.conv7_d[0](x7))**2
            temp = loss_y
            f_y = torch.autograd.grad(loss_y, x7, retain_graph=self.graph, create_graph=self.graph)
            flag = False
            x7 = x7 - wd*f_y[0]
        wd = self.wd
        for i in range(it-1):
            # if flag == False:
            #     break
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x7-x6-(1-theta)*t7-theta*self.conv7_d[i+1](x7))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x7, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x7 = x7 - wd*f_y[0]
            wd = self.wd
        x8 = self.conv8(x7) + self.d8(x7)
        # x8.requires_grad=True
        # x8.retain_grad=True
        t9 = self.conv9(x8)
        output = x8 + t9
        wd = self.wd
        with torch.set_grad_enabled(True):
            loss_y = torch.norm(output-x8-(1-theta)*t9+theta*self.conv9_d[0](output))**2
            temp = loss_y
            f_y = torch.autograd.grad(loss_y, output, retain_graph=self.graph, create_graph=self.graph)
            flag = False
            output = output - wd*f_y[0]
        wd = self.wd
        for i in range(it-1):
            # if flag == False:
            #     break
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(output-x8-(1-theta)*t9+theta*self.conv9_d[i+1](output))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, output, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                output = output - wd*f_y[0]
            wd = self.wd
        output = self.final_relu(self.final_bn(output))
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class SIR_ResNet34(nn.Module):
    def __init__(self, wd=0.1, it=1,theta=1.0, num_classes=10):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))

        self.conv2 =  nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False))

        self.conv2_d =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False))  for i in range(it))
        
        self.conv3 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)) for j in range(2))

        self.conv3_d = nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)) for i in range(it)) for j in range(2))


        self.conv4 =  nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False))

        self.d4 = nn.Sequential(
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            )

        self.conv5 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False))  for j in range(3))

        self.conv5_d = nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)) for i in range(it)) for j in range(3))

        self.conv6 =  nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))

        self.d6 = nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
            )


        self.conv7 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)) for j in range(5))

        self.conv7_d =  nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)) for i in range(it)) for j in range(5))

        self.conv8 =  nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False))

        self.d8 = nn.Sequential(
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            )
# )

        self.conv9 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)) for i in range(2))

        self.conv9_d =  nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)) for i in range(it)) for i in range(2))
        self.theta =theta
        self.final_bn = nn.GroupNorm(1,128)
        self.final_relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.graph=False
        self.wd = wd
        self.it = it

    def forward(self, x):
        wd = self.wd
        it = self.it
        theta = self.theta
        for i in range(it):
            self.conv2_d[i][2].weight.data = self.conv2[2].weight.data
            self.conv2_d[i][0].weight.data = self.conv2[0].weight.data
            self.conv2_d[i][0].bias.data = self.conv2[0].bias.data
            self.conv2_d[i][5].weight.data = self.conv2[5].weight.data
            self.conv2_d[i][3].weight.data = self.conv2[3].weight.data
            self.conv2_d[i][3].bias.data = self.conv2[3].bias.data
            for j in range(2):
                self.conv3_d[j][i][2].weight.data = self.conv3[j][2].weight.data
                self.conv3_d[j][i][0].weight.data = self.conv3[j][0].weight.data
                self.conv3_d[j][i][0].bias.data = self.conv3[j][0].bias.data
                self.conv3_d[j][i][5].weight.data = self.conv3[j][5].weight.data
                self.conv3_d[j][i][3].weight.data = self.conv3[j][3].weight.data
                self.conv3_d[j][i][3].bias.data = self.conv3[j][3].bias.data
            for j in range(3):
                self.conv5_d[j][i][2].weight.data = self.conv5[j][2].weight.data
                self.conv5_d[j][i][0].weight.data = self.conv5[j][0].weight.data
                self.conv5_d[j][i][0].bias.data = self.conv5[j][0].bias.data
                self.conv5_d[j][i][5].weight.data = self.conv5[j][5].weight.data
                self.conv5_d[j][i][3].weight.data = self.conv5[j][3].weight.data
                self.conv5_d[j][i][3].bias.data = self.conv5[j][3].bias.data
            for j in range(5):
                self.conv7_d[j][i][2].weight.data = self.conv7[j][2].weight.data
                self.conv7_d[j][i][0].weight.data = self.conv7[j][0].weight.data
                self.conv7_d[j][i][0].bias.data = self.conv7[j][0].bias.data
                self.conv7_d[j][i][5].weight.data = self.conv7[j][5].weight.data
                self.conv7_d[j][i][3].weight.data = self.conv7[j][3].weight.data
                self.conv7_d[j][i][3].bias.data = self.conv7[j][3].bias.data
            for j in range(2):
                self.conv9_d[j][i][2].weight.data = self.conv9[j][2].weight.data
                self.conv9_d[j][i][0].weight.data = self.conv9[j][0].weight.data
                self.conv9_d[j][i][0].bias.data = self.conv9[j][0].bias.data
                self.conv9_d[j][i][5].weight.data = self.conv9[j][5].weight.data
                self.conv9_d[j][i][3].weight.data = self.conv9[j][3].weight.data
                self.conv9_d[j][i][3].bias.data = self.conv9[j][3].bias.data
        x1 = self.conv1(x)
        t2 = self.conv2(x1)
        x2 = x1 + t2
        with torch.set_grad_enabled(True):
            loss_y = torch.norm(x2-x1-(1-theta)*t2-theta*self.conv2_d[0](x2))**2
            temp = loss_y
            f_y = torch.autograd.grad(loss_y, x2, retain_graph=self.graph, create_graph=self.graph)
            x2 = x2 - wd*f_y[0]
        wd = self.wd
        for i in range(it-1):
            # if flag == False:
            #     break
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x2-x1-(1-theta)*t2-theta*self.conv2_d[i+1](x2))**2
                f_y = torch.autograd.grad(loss_y, x2, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x2 = x2 - wd*f_y[0]
            wd = self.wd
        for k in range(2):
            t3 = self.conv3[k](x2)
            x3 = x2 + t3
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x3-x2-(1-theta)*t3-theta*self.conv3_d[k][0](x3))**2
                # loss_y = torch.norm(x3-x2-self.conv3_d[k][0](x3))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x3, retain_graph=self.graph, create_graph=self.graph)
                x3 = x3 - wd*f_y[0]
            wd = self.wd
            # print(torch.norm(x2),torch.norm(loss_y),torch.norm(f_y[0]))
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(x3-x2-(1-theta)*t3-theta*self.conv3_d[k][i+1](x3))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, x3, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    x3 = x3 - wd*f_y[0]
                wd = self.wd
            x2 = x3
        x4 = self.conv4(x3) + self.d4(x3)
        # x4.requires_grad=True
        # x4.retain_grad=True
        for k in range(3):
            t5 = self.conv5[k](x4)
            x5 = x4 + t5
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x5-x4-(1-theta)*t5-theta*self.conv5_d[k][0](x5))**2
                # loss_y = torch.norm(x5-x3-self.conv5_d[k][0](x5))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x5, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x5 = x5 - wd*f_y[0]
            wd = self.wd
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(x5-x4-(1-theta)*t5-theta*self.conv5_d[k][i+1](x5))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, x5, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    x5 = x5 - wd*f_y[0]
                wd = self.wd
            x4 = x5
        x6 = self.conv6(x5) + self.d6(x5)
        for k in range(5):
            t7 = self.conv7[k](x6)
            x7 = x6 + t7
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x7-x6-(1-theta)*t7-theta*self.conv7_d[k][0](x7))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x7, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x7 = x7 - wd*f_y[0]
            wd = self.wd
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(x7-x6-(1-theta)*t7-theta*self.conv7_d[k][i+1](x7))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, x7, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    x7 = x7 - wd*f_y[0]
                wd = self.wd
            x6 = x7
        x8 = self.conv8(x7) + self.d8(x7)
        # x8.requires_grad=True
        # x8.retain_grad=True
        for k in range(2):
            t9 = self.conv9[k](x8)
            output = x8 + t9
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(output-x8-(1-theta)*t9+theta*self.conv9_d[k][0](output))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, output, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                output = output - wd*f_y[0]
            wd = self.wd
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(output-x8-(1-theta)*t9+theta*self.conv9_d[k][i+1](output))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, output, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    output = output - wd*f_y[0]
                wd = self.wd
            x8 = output
        output = self.final_relu(self.final_bn(output))
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class SIR_ResNet50(nn.Module):
    def __init__(self, wd=0.1, it=1,theta=1.0, num_classes=10):
        super().__init__()

        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True))

        self.conv2 =  nn.Sequential(
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, padding=0, bias=False))

        self.d2 = nn.Sequential(
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            )

        # self.conv2_d =  nn.ModuleList(nn.Sequential(
        #     nn.GroupNorm(1,16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=1, padding=0, bias=False),
        #     nn.GroupNorm(1,16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
        #     nn.GroupNorm(1,16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 64, kernel_size=1, padding=0, bias=False)) for i in range(it))
        
        self.conv3 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, padding=0, bias=False)) for j in range(2))

        self.conv3_d = nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, padding=0, bias=False)) for i in range(it)) for j in range(2))


        self.conv4 =  nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False))

        self.d4 = nn.Sequential(
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            )

        self.conv5 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False))  for j in range(3))

        self.conv5_d = nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False)) for i in range(it)) for j in range(3))

        self.conv6 =  nn.Sequential(
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False))

        self.d6 = nn.Sequential(
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False)
            )


        self.conv7 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False)) for j in range(5))

        self.conv7_d =  nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False)) for i in range(it)) for j in range(5))

        self.conv8 =  nn.Sequential(
            nn.GroupNorm(1,512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2048, kernel_size=1, padding=0, bias=False))

        self.d8 = nn.Sequential(
            nn.GroupNorm(1,512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=3, padding=1, stride=2, bias=False),
            )
# )

        self.conv9 =  nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 256, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2048, kernel_size=1, padding=0, bias=False)) for i in range(2))

        self.conv9_d =  nn.ModuleList(nn.ModuleList(nn.Sequential(
            nn.GroupNorm(1,2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 256, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2048, kernel_size=1, padding=0, bias=False)) for i in range(it)) for i in range(2))
        self.theta =theta
        self.final_bn = nn.GroupNorm(1,2048)
        self.final_relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.graph=False
        self.wd = wd
        self.it = it

    def forward(self, x):
        wd = self.wd
        it = self.it
        theta = self.theta
        for i in range(it):
            # self.conv2_d[i][2].weight.data = self.conv2[2].weight.data
            # self.conv2_d[i][0].weight.data = self.conv2[0].weight.data
            # self.conv2_d[i][0].bias.data = self.conv2[0].bias.data
            # self.conv2_d[i][5].weight.data = self.conv2[5].weight.data
            # self.conv2_d[i][3].weight.data = self.conv2[3].weight.data
            # self.conv2_d[i][3].bias.data = self.conv2[3].bias.data
            for j in range(2):
                self.conv3_d[j][i][2].weight.data = self.conv3[j][2].weight.data
                self.conv3_d[j][i][0].weight.data = self.conv3[j][0].weight.data
                self.conv3_d[j][i][0].bias.data = self.conv3[j][0].bias.data
                self.conv3_d[j][i][5].weight.data = self.conv3[j][5].weight.data
                self.conv3_d[j][i][3].weight.data = self.conv3[j][3].weight.data
                self.conv3_d[j][i][3].bias.data = self.conv3[j][3].bias.data
            for j in range(3):
                self.conv5_d[j][i][2].weight.data = self.conv5[j][2].weight.data
                self.conv5_d[j][i][0].weight.data = self.conv5[j][0].weight.data
                self.conv5_d[j][i][0].bias.data = self.conv5[j][0].bias.data
                self.conv5_d[j][i][5].weight.data = self.conv5[j][5].weight.data
                self.conv5_d[j][i][3].weight.data = self.conv5[j][3].weight.data
                self.conv5_d[j][i][3].bias.data = self.conv5[j][3].bias.data
            for j in range(5):
                self.conv7_d[j][i][2].weight.data = self.conv7[j][2].weight.data
                self.conv7_d[j][i][0].weight.data = self.conv7[j][0].weight.data
                self.conv7_d[j][i][0].bias.data = self.conv7[j][0].bias.data
                self.conv7_d[j][i][5].weight.data = self.conv7[j][5].weight.data
                self.conv7_d[j][i][3].weight.data = self.conv7[j][3].weight.data
                self.conv7_d[j][i][3].bias.data = self.conv7[j][3].bias.data
            for j in range(2):
                self.conv9_d[j][i][2].weight.data = self.conv9[j][2].weight.data
                self.conv9_d[j][i][0].weight.data = self.conv9[j][0].weight.data
                self.conv9_d[j][i][0].bias.data = self.conv9[j][0].bias.data
                self.conv9_d[j][i][5].weight.data = self.conv9[j][5].weight.data
                self.conv9_d[j][i][3].weight.data = self.conv9[j][3].weight.data
                self.conv9_d[j][i][3].bias.data = self.conv9[j][3].bias.data
        x1 = self.conv1(x)
        # t2 = self.conv2(x1)
        # x2 = x1 + t2
        # with torch.set_grad_enabled(True):
        #     loss_y = torch.norm(x2-x1-(1-theta)*t2-theta*self.conv2_d[0](x2))**2
        #     temp = loss_y
        #     f_y = torch.autograd.grad(loss_y, x2, retain_graph=self.graph, create_graph=self.graph)
        #     x2 = x2 - wd*f_y[0]
        # wd = self.wd
        # for i in range(it-1):
        #     # if flag == False:
        #     #     break
        #     with torch.set_grad_enabled(True):
        #         loss_y = torch.norm(x2-x1-(1-theta)*t2-theta*self.conv2_d[i+1](x2))**2
        #         f_y = torch.autograd.grad(loss_y, x2, retain_graph=self.graph, create_graph=self.graph)
        #         flag = False
        #         x2 = x2 - wd*f_y[0]
        #     wd = self.wd
        x2 = self.conv2(x1) + self.d2(x1)
        for k in range(2):
            t3 = self.conv3[k](x2)
            x3 = x2 + t3
            wd = self.wd
            with torch.set_grad_enabled(True):
                #try:
                loss_y = torch.norm(x3-x2-(1-theta)*t3-theta*self.conv3_d[k][0](x3))**2
                # loss_y = torch.norm(x3-x2-self.conv3_d[k][0](x3))**2
                temp = loss_y
                try:
                    f_y = torch.autograd.grad(loss_y, x3, retain_graph=self.graph, create_graph=self.graph)
                except:
                    import ipdb; ipdb.set_trace()
                x3 = x3 - wd*f_y[0]
            wd = self.wd
            # print(torch.norm(x2),torch.norm(loss_y),torch.norm(f_y[0]))
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(x3-x2-(1-theta)*t3-theta*self.conv3_d[k][i+1](x3))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, x3, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    x3 = x3 - wd*f_y[0]
                wd = self.wd
            x2 = x3
        x4 = self.conv4(x3) + self.d4(x3)
        # x4.requires_grad=True
        # x4.retain_grad=True
        for k in range(3):
            t5 = self.conv5[k](x4)
            x5 = x4 + t5
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x5-x4-(1-theta)*t5-theta*self.conv5_d[k][0](x5))**2
                # loss_y = torch.norm(x5-x3-self.conv5_d[k][0](x5))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x5, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x5 = x5 - wd*f_y[0]
            wd = self.wd
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(x5-x4-(1-theta)*t5-theta*self.conv5_d[k][i+1](x5))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, x5, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    x5 = x5 - wd*f_y[0]
                wd = self.wd
            x4 = x5
        x6 = self.conv6(x5) + self.d6(x5)
        for k in range(5):
            t7 = self.conv7[k](x6)
            x7 = x6 + t7
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(x7-x6-(1-theta)*t7-theta*self.conv7_d[k][0](x7))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, x7, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                x7 = x7 - wd*f_y[0]
            wd = self.wd
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(x7-x6-(1-theta)*t7-theta*self.conv7_d[k][i+1](x7))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, x7, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    x7 = x7 - wd*f_y[0]
                wd = self.wd
            x6 = x7
        x8 = self.conv8(x7) + self.d8(x7)
        # x8.requires_grad=True
        # x8.retain_grad=True
        for k in range(2):
            t9 = self.conv9[k](x8)
            output = x8 + t9
            wd = self.wd
            with torch.set_grad_enabled(True):
                loss_y = torch.norm(output-x8-(1-theta)*t9+theta*self.conv9_d[k][0](output))**2
                temp = loss_y
                f_y = torch.autograd.grad(loss_y, output, retain_graph=self.graph, create_graph=self.graph)
                flag = False
                output = output - wd*f_y[0]
            wd = self.wd
            for i in range(it-1):
                # if flag == False:
                #     break
                with torch.set_grad_enabled(True):
                    loss_y = torch.norm(output-x8-(1-theta)*t9+theta*self.conv9_d[k][i+1](output))**2
                    temp = loss_y
                    f_y = torch.autograd.grad(loss_y, output, retain_graph=self.graph, create_graph=self.graph)
                    flag = False
                    output = output - wd*f_y[0]
                wd = self.wd
            x8 = output
        output = self.final_relu(self.final_bn(output))
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def sir_resnet18(wd=0.1,it=1,num_classes=10):
    return SIR_ResNet18(wd=wd,it=it,num_classes=num_classes)

def sir_resnet34(wd=0.1,it=1,num_classes=10):
    return SIR_ResNet34(wd=wd,it=it,num_classes=num_classes)

def sir_resnet50(wd=0.1,it=1,num_classes=10):
    return SIR_ResNet50(wd=wd,it=it,num_classes=num_classes)