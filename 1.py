import torch

x = torch.ones([1,2,3])
y = torch.ones([1,2,3])

print(sum(x==y))
print(torch.sum(x==y))
print(x==y)
