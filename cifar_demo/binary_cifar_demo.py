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
from models.binary_alexnet import alexnet as get_alexnet
from pdb import set_trace as b

reproduce = False
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

# cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
# cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)



class TensorDataset(Dataset):
    def __init__(self, max_data_size = 2000):
        super(TensorDataset, self).__init__()
        self.max_data_size = max_data_size

    def __getitem__(self, item):
        # img = np.random.randint(0, 2, size=[3, 32, 32])
        # if np.mean(img) >= 0.5:
        #     img = img * 0.5
        #     label = 1
        # else:
        #     img = img*0.5
        #     label = 0

        # img = np.random.randint(0, 256, size=[3, 32, 32])/256
        # label = np.random.randint(0, 2)
        # if label == 0:
        #     img = img*0.98
        # else:
        #     img = img*0.89

        img = np.random.randint(0, 256, size=[3, 32, 32])
        label = np.random.randint(0, 10)
        img[:, label * 3, :] = np.ones([3, 32]) * 256

        # increase width
        # img[:, label * 3:label * 3+2, :] = np.ones([3, 2, 32]) * 256
        # img[:, label * 3:label * 3+3, :] = np.ones([3, 3, 32]) * 256
        # img[:, label * 3+3:label * 3+4, :] = np.ones([3, 4, 32]) * 256

        # change position
        # img[:, label * 3+3, :] = np.ones([3, 32]) * 256


        img = img / 256
        return torch.from_numpy(img).to(torch.float32), label

        # img = np.random.randint(0, 256, size=[3, 32, 32])
        # label = np.random.randint(0, 10)
        # for i in range(label):
        #     img[:,i*3,:] = np.ones([3, 32])*256
        # img = img/256
        # return torch.from_numpy(img).to(torch.float32), label

    def __len__(self):
        # return len(self.data)
        return self.max_data_size

cifar10_train_DL = DataLoader(
    TensorDataset(max_data_size = 2000),
    batch_size=100,
    shuffle=False,
    num_workers=10,
    collate_fn=None,
    pin_memory=False,
 )

cifar10_test_DL = DataLoader(
    TensorDataset(max_data_size = 400),
    batch_size=100,
    shuffle=False,
    num_workers=1,
    collate_fn=None,
    pin_memory=False,
 )

# alexnet = get_alexnet(pretrained=False)
alexnet = get_alexnet(pretrained=True)

class Model(Module):
    def __init__(self, model, out_num):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

model = Model(alexnet, 10)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
step_lr = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[75, 150], gamma=0.5)
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# step_lr = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        # loss = nn.L1Loss()(logit, label)
        # loss = nn.NLLLoss()(logit, label)
        # loss = nn.KLDivLoss()(logit, label)
        # loss = nn.MSELoss()(logit, label)
        # loss = nn.BCELoss()(logit, label)
        # loss = nn.HingeEmbeddingLoss()(logit, label)
        # loss = nn.CosineEmbeddingLoss()(logit, label)
        # loss = nn.SoftMarginLoss()(logit, label)
        # loss = nn.MultiMarginLoss()(logit, label)
        # loss = nn.MultiLabelMarginLoss()(logit, label)
        # loss = nn.MultiLabelSoftMarginLoss()(logit, label)
        # loss = nn.SoftMarginLoss()(logit, label)
        loss.backward()
        optim.step()

        # if pre_loss != -1:
        #     if abs(loss.item()-pre_loss)<0.1*pre_loss or loss.item()-pre_loss > 0:
        #         for p in optim.param_groups:
        #             p['lr'] *= 0.5
        #             # print(p['lr'])
        # pre_loss = loss.item()
        n = torch.sum(torch.argmax(logit, 1)==label)
        train_acc = n.item()/len(label)
        writer.add_scalar('loss', round(loss.item(), factor_n), global_step=step)
        writer.add_scalar('train_acc', round(train_acc, factor_n), global_step=step)
        step += 1
        print(f"epoch:{epoch} iter:{i} step:{step} lr:{list(optim.param_groups)[0]['lr']} loss:{round(loss.item(), factor_n)} train_acc:{round(train_acc, factor_n)}")
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
    print(f"test loss:{round(loss.item()/n, factor_n)} acc:{acc.item()/n}")
end_time = time.time()
print(f'{end_time-begin_time} s')

save_model = True
if save_model:
    torch.save(model.state_dict(), './checkpoint.pt')





