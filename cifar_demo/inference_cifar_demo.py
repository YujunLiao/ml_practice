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

        # img = np.random.randint(0, 256, size=[3, 32, 32])
        # label = np.random.randint(0, 10)
        # img[:,label,:] = np.ones([3, 32])*256
        # img = img/256
        # return torch.from_numpy(img).to(torch.float32), label

        img = np.random.randint(0, 256, size=[3, 32, 32])
        # label = np.random.randint(0, 10)
        # label = np.random.randint(0, 9)
        label = np.random.randint(1, 10)
        # img[:, label * 3, :] = np.ones([3, 32]) * 256

        # increase width
        # img[:, label * 3:label * 3+2, :] = np.ones([3, 2, 32]) * 256
        # img[:, label * 3:label * 3+3, :] = np.ones([3, 3, 32]) * 256
        # img[:, label * 3:label * 3+4, :] = np.ones([3, 4, 32]) * 256

        # change position
        # img[:, label * 3-1, :] = np.ones([3, 32]) * 256
        img[:, label * 3-2, :] = np.ones([3, 32]) * 256

        # img[:, label * 3+1, :] = np.ones([3, 32]) * 256
        # img[:, label * 3+2, :] = np.ones([3, 32]) * 256
        # img[:, label * 3+3, :] = np.ones([3, 32]) * 256

        # multi-lines
        # img[:, label * 3, :] = np.ones([3, 32]) * 256
        # img[:, label * 3+3, :] = np.ones([3, 32]) * 256
        # img[:, label * 3+6, :] = np.ones([3, 32]) * 256




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

cifar10_test_DL = DataLoader(
    TensorDataset(max_data_size = 400),
    batch_size=100,
    shuffle=False,
    num_workers=1,
    collate_fn=None,
    pin_memory=False,
 )

alexnet = get_alexnet(pretrained=False)

class Model(Module):
    def __init__(self, model, out_num):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

model = Model(alexnet, 10)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load('./checkpoint.pt', map_location=device))


factor_n = 10

begin_time = time.time()
model.eval()
loss = 0
n = 0
acc = 0

with torch.no_grad():
    for i, (img, label) in enumerate(cifar10_test_DL):
        img, label, model = img.to(device), label.to(device), model.to(device)
        logit = model(img)
        pre = torch.argmax(logit, dim=1)

        # for j in range(len(label)):
        #     if label[j] != pre[j]:
        #         pil_img = torchvision.transforms.ToPILImage()(img[j].cpu())
        #         pil_img.show()
        #         print(label[j], pre[j], logit[j])

        acc += torch.sum(pre==label)
        loss += nn.CrossEntropyLoss()(logit, label)
        n += len(label)
        if n == 1000:
            break
print(f"test loss:{round(loss.item()/n, factor_n)} acc:{acc.item()/n}")
end_time = time.time()
print(f'{end_time-begin_time} s')





