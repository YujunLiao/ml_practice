from collections import OrderedDict

import torch
import torchvision.transforms as tf
from torch import nn
from torch.nn.modules import Module
import random
from matplotlib import pyplot  as plt

manualSeed = 4
# np.random.seed(manualSeed)
# random.seed(manualSeed)
torch.manual_seed(manualSeed)


class RegrssionDemo(Module):
    def __init__(self):
        super().__init__()
        # self.regression = nn.Sequential(OrderedDict([
        #     ('1', nn.Linear(1, 200)),
        #     ('2', nn.ReLU()),
        #     ('3', nn.Dropout()),
        #
        #     ('4', nn.Linear(200, 400)),
        #     ('5', nn.ReLU()),
        #     ('6', nn.Dropout()),
        #
        #     ('7', nn.Linear(400, 1)),
        #     ('5', nn.ReLU()),
        # ]))

        self.regression = nn.Sequential(OrderedDict([
            ('a', nn.Linear(1, 40, True)),
            ('r1', nn.ReLU()),
            ('b', nn.Linear(40, 1, True)),
            # ('r2', nn.ReLU()),
            # ('c', nn.Linear(40, 1, True)),
            # ('r3', nn.ReLU()),
            ]))
        # self.w = torch.nn.Parameter(torch.rand(1), True)
        # self.b = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

        for m in self.modules():
            if not isinstance(m, nn.Linear): continue
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        # return self.regression(x)
        # return self.w*x+self.b
        return self.regression(x)


def train():


    pass




if __name__ == '__main__':
    n = 10000
    # x = 20 * torch.randn((n, 1)) - 10
    x = torch.randint(-80, 80, (n, 1)).float()
    y = x**2 #+ 100.0*torch.randint(-10, 10, (n, 1))
    x_norm = torch.nn.functional.normalize(x, dim=0)
    pre_loss = -1
    model = RegrssionDemo()
    optim = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, 200, 0.5)
    for i in range(3000):


        # if i % 100 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         # pass
        #         y_pre = model(x)
        #         fig = plt.figure(figsize=(8, 6))
        #         fig1 = fig.add_subplot(111)
        #         fig1.plot(x.detach(), y_pre.detach(), '.', markersize=12)
        #         plt.show()

        model.train()
        optim.zero_grad()
        y_pre = model(x_norm)
        # print('y_pre', y_pre)
        loss = nn.MSELoss()(y, y_pre)
        print("     ",i, 'loss:', loss.item())

        loss.backward()
        optim.step()

        if i % 100 == 0 and pre_loss != -1:
            if abs(loss.item()-pre_loss)<0.1*pre_loss:
                for p in optim.param_groups:
                    p['lr'] *= 0.5
                    print(p['lr'])
            pre_loss = loss.item()
        #
        # scheduler.step()








    #
    # fig = plt.figure(figsize=(8, 6))
    # fig1 = fig.add_subplot(111)
    # fig1.plot(x.detach(), y_pre.detach(), '.')
    # plt.show()

