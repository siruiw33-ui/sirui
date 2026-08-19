import numpy as np
import torch

#view reshape
t = np.array([[[1 ,2, 3], [3, 4, 5]],
             [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

#squeeze 只删除维度
ft1 = torch.FloatTensor([[0], [1], [2]])
print(ft1)
print(ft1.shape)

print(ft1.squeeze())
print(ft1.squeeze().shape)

#unsqueeze
ft2 = torch.Tensor([0, 1, 2])
print(ft2.shape)

print(ft2.unsqueeze(0))
print(ft2.unsqueeze(0).shape)

print(ft2.view(1, -1))
print(ft2.view(1, -1).shape)

print(ft2.unsqueeze(1))
print(ft2.unsqueeze(1).shape)

print(ft2.unsqueeze(-1).shape)
print(ft2.unsqueeze(-1).shape)

#type casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([ True, False, False, True ])
print(bt)

print(bt.long())
print(bt.float())

#concatenate
x = torch.FloatTensor([[1, 2], [4, 5]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim = 0))
print(torch.cat([x, y], dim = 1))

#stacking
x1 = torch.FloatTensor([1, 4])
y1 = torch.FloatTensor([2, 5])
z1 = torch.FloatTensor([3, 6])
print(torch.stack([x1, y1, z1]))
print(torch.stack([x1, y1, z1], dim = 1))
print(torch.cat([x1.unsqueeze(0), y1.unsqueeze(0), z1.unsqueeze(0)], dim = 0))

#ones and zeros
x2 = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x2)

print(torch.ones_like(x2))
print(torch.zeros_like(x2))

#Inplace_operation
x3 = torch.FloatTensor([[1, 2], [3, 4]])
print(x3.mul(2.))
print(x3)
print(x3.mul_(2.))
print(x3)