#创建张量
import numpy as np
import torch

tensor_list = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor_np = torch.tensor(np_array)

print(tensor_list)
print(tensor_np)

#一些特定的张量
tensor_ones = torch.ones(3, 3)
print(tensor_ones)

tensor_zeros = torch.zeros(3, 3)
print(tensor_zeros)

tensor_rand = torch.rand(3, 3)
print(tensor_rand)

tensor_randn = torch.randn(3, 3)
print(tensor_randn)

tensor_randint = torch.randint(0, 10, (3, 3))
print(tensor_randint)

#张量的数学运算
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 2], [3, 4]])

#加法
c = a+b
print(c)

#叉乘
d = torch.mm(a,b)
print(d)

#横轴相加
print(f"横轴累加{torch.cumsum(a,dim=0)}")

#纵轴累乘
print(f"纵轴累乘{torch.cumprod(a,dim=1)}")

#均值
print(a.mean)

#最大值
print(torch.max(a))
