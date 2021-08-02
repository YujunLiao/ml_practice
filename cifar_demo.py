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
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 )

cifar10_test_DL = DataLoader(
    TensorDataset(cifar10_testset),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 )

resnet_18 = torchvision.models.resnet18(pretrained=True)
alexnet = torchvision.models.alexnet(pretrained=True)

class Model(Module):
    def __init__(self, model, out_num):
        super(Model, self).__init__()
        self.model = model
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1000, 100, True)),
            ('r1', nn.ReLU()),
            ('fc2', nn.Linear(100, out_num, True)),
        ]))
        self.init()


    def init(self):
        for m in dict(self.named_children())['fc']:
            if not isinstance(m, nn.Linear): continue
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0.)



    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

# alex_model = Model(alexnet, 10)
model = Model(alexnet, 10)
# optim = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
optim = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
# optim = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
step_lr = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)

device = torch.device('cuda:0')
writer = SummaryWriter(log_dir = 'logs')

pre_loss = -1
step = 0
for epoch in range(1):
    for i, (img, label) in enumerate(cifar10_train_DL):
        img, label, model = img.to(device), label.to(device), model.to(device)
        model.train()
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
        writer.add_scalar('loss', loss.item(), global_step=step)
        step += 1
        print(f"epoch:{epoch} iter:{i} lr:{list(optim.param_groups)[0]['lr']} loss:{loss.item()}")
    step_lr.step()






