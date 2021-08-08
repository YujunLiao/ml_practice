import os

import PIL
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import Module
from collections import OrderedDict
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data.Dataset
import random
import numpy as np
import time
reproduce = True
if reproduce:
    print("deterministic")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(0)

cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

class TensorDataset(Dataset):
    def __init__(self, data):
        super(TensorDataset, self).__init__()
        self.data = data


    def __getitem__(self, item):
        img, label = self.data[item]
        img = img.resize((64, 64), PIL.Image.BILINEAR)
        return transforms.ToTensor()(img), label

    def __len__(self):
        return len(self.data)



cifar10_train_DL = DataLoader(
    TensorDataset(cifar10_trainset),
    batch_size=64,
    shuffle=False,
    num_workers=1,
    collate_fn=None,
    pin_memory=False,
 )

cifar10_test_DL = DataLoader(
    TensorDataset(cifar10_testset),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    collate_fn=None,
    pin_memory=False,
 )

# resnet_18 = torchvision.models.resnet18(pretrained=True)
# resnet_50 = torchvision.models.resnet50(pretrained=True)
alexnet = torchvision.models.alexnet(pretrained=True)
# alexnet = torchvision.models.alexnet(pretrained=False)

class Model(Module):
    def __init__(self, model, out_num):
        super(Model, self).__init__()
        self.model = model
        self.model.classifier[-1] = nn.Linear(4096, out_num, True)

        # self.fc = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(1000, 100, True)),
        #     ('r1', nn.ReLU()),
        #     ('fc2', nn.Linear(100, out_num, True)),
        # ]))
        self.init()


    def init(self):
        nn.init.kaiming_normal_(self.model.classifier[-1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.model.classifier[-1].bias, 0.)

        # for m in dict(self.named_children())['fc']:
        #     if not isinstance(m, nn.Linear): continue
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     nn.init.constant_(m.bias, 0.)

        # for m in dict(self.named_children())['model'].children():
        #     if not isinstance(m, nn.Sequential): continue
        #     for mm in m:
        #         if isinstance(mm, nn.Linear):
        #             nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
        #             nn.init.constant_(mm.bias, 0.)
        #         if isinstance(mm, nn.Conv2d):
        #             nn.reset_parameters()







    def forward(self, x):
        x = self.model(x)
        return x
        # return self.fc(x)

# model = Model(resnet_18, 10)
# model = Model(resnet_50, 10)
model = Model(alexnet, 10)
# optim = torch.optim.SGD(model.parameters(), lr=5e-6, momentum=0.9)
optim = torch.optim.SGD(model.parameters(), lr=5e-5, momentum=0.9)
# optim = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
# optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
step_lr = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(log_dir = 'logs')


pre_loss = -1
step = 0
factor_n = 10
begin_time = time.time()
for epoch in range(20):
    model.train()
    for i, (img, label) in enumerate(cifar10_train_DL):
        img, label, model = img.to(device), label.to(device), model.to(device)
        optim.zero_grad()

        logit = model(img)
        loss = nn.CrossEntropyLoss()(logit, label)
        loss.backward()
        optim.step()

        # if pre_loss != -1:
        #     if abs(loss.item()-pre_loss)<0.1*pre_loss or loss.item()-pre_loss > 0:
        #         for p in optim.param_groups:
        #             p['lr'] *= 0.5
        #             # print(p['lr'])
        # pre_loss = loss.item()
        writer.add_scalar('loss', round(loss.item(), factor_n), global_step=step)
        step += 1
        print(f"epoch:{epoch} iter:{i} lr:{list(optim.param_groups)[0]['lr']} loss:{round(loss.item(), factor_n)}")
        #
        # if i == 10:
        #     break
    step_lr.step()

    model.eval()
    loss = 0
    n = 0
    acc = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(cifar10_test_DL):
            img, label, model = img.to(device), label.to(device), model.to(device)
            logit = model(img)
            pre = torch.argmax(logit, dim=1)
            acc += torch.sum(pre==label)
            loss += nn.CrossEntropyLoss()(logit, label)
            n += len(label)
            if n == 1000:
                break
    print(f"test loss:{round(loss.item()/n, factor_n)} acc:{acc/n}")
end_time = time.time()
print(f'{end_time-begin_time} s')





