import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #定义一个卷积层
        self.conv1 = nn.Conv2d(3, 32, 3)

        def forward(self, x):
            #应用ReLU激活函数
            x = F.relu(self.conv1(x))
            return x





        