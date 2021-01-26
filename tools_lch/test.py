import torch

d1 = torch.nn.Conv1d(1024, 512, 3, 1, 1)
d2 = torch.nn.Conv1d(512, 4, 1, 1, 0)
x = torch.normal(mean=0, std=1, size=(5, 1024, 100))

y = d1(x)
y = d2(y)

print(y.shape)
