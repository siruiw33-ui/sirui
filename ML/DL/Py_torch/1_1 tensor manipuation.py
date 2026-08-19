import numpy as np
import torch
from sympy.codegen import Print

#numpy review(1D Array)
t = np.array([0.,1.,2.,3.,4.,5.,6.])
print(t)
print('rank of t',t.ndim)
print(f'shape of t{t.shape}')

print(t[0], t[1], t[-1])
print(f't[2:5] = {t[2:5]},t[4:-1] = {t[4:-1]}')
print(f't[:2] = {t[:2]}, t[3:] = {t[3:]}')

#numpy review(2D Array)
t1 = np.array([[1.,2.,3.],[4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
print(t1)
print('rank of t1',t1.ndim)
print(f'shape of t1 :{t1.shape}')

#pytorch tensor|(1D)
t2 = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t2)
print('rank of t2',t2.dim())
print(f'shape of t2 :{t2.shape}')
print(t2[0], t2[1], t2[-1])
print(f't2[2:5] = {t2[2:5]},t[4:-1] = {t2[4:-1]}')
print(f't2[:2] = {t2[:2]}, t2[3:] = {t2[3:]}')

#pytorch tensor(2D)
t3 = torch.FloatTensor([[1.,2.,3.],[4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
print(t3)
print('rank of t3',t3.dim())
print(f'shape of t3 :{t3.shape}')
print(t3[:,1])
print(t3[:,1].size())
print([t3[:,:-1]])

#same shape
m1 = torch.FloatTensor([3,3])
m2 = torch.FloatTensor([2,2])
print(m1 + m2)
print(m1 - m2)

m3 = torch.FloatTensor([1,2])
m4 = torch.FloatTensor([3])
print(m3 + m4)
print(m3 - m4)

m5 = torch.FloatTensor([1,2])
m6 = torch.FloatTensor([[3],[4]])
print(m5 + m6)
print(m5 - m6)

print()
print('-------------')
print('Mul vs Matmul')
print('-------------')

n1 = torch.FloatTensor([[1,2],[3,4]])
n2 = torch.FloatTensor([[1],[2]])
print(f'shape of matrix n1 :{n1.shape}')
print(f'shape of matrix n2 :{n2.shape}')
print(n1.matmul(n2))#矩阵乘法，也可以用@表示
print(n1 * n2)#逐元素相乘，和下面表达相同
print(n1.mul(n2))

#mean用于计算算术平均值
a = torch.FloatTensor([1,2])
print(t.mean())

#.mean()必须用于浮点型或者复数型
a1 = torch.LongTensor([1,2])
try:
    print(a1.mean())
except Exception as exc:
    print(exc)

a2 = torch.FloatTensor([[1,2],[3,4]])
print(a2.mean())
print(a2.mean(dim = 0))
print(a2.mean(dim = 1))
print(a2.mean(dim = -1))