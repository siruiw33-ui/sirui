"""
maximum likelihood estimation MLE
经过梯度下降法进行最优化

overfitting 过拟合
采用更多数据、更少特征、正则化

regularization正则化的具体方法
早停
减少网络层数或者每层神经元数量 循环判断
权重衰减：损失函数中加上权重的平方惩罚项，迫使权重变小 nn.linear(512, 256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
dropout：训练时随机“关掉”一部分神经元，防止神经元之间过度依赖
         nn.Dropout(p = 0.5)
batch normaliza：对每一层的输入做归一化，稳定训练过程，有一定正则化效果

DNN经典、实用流程：
1.设计神经网络结构
2.训练它，检查是否过拟合
  若已经过拟合，加正则化，抑制过拟合
  若没有过拟合，增大模型，提高模型容量
3.重复步骤2，不断调整，直到找到一个既有足够容量、又不过拟合的模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]])

y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])  # 改成 LongTensor，一维

x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)  # 必须加这一行！输入3维，输出3类

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        prediction = model(x_train)
        cost = F.cross_entropy(prediction, y_train)  # 改用 cross_entropy

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


def test(model, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()

    print('Accuracy: {}%'.format(correct_count / len(y_test) * 100))


train(model, optimizer, x_train, y_train)
test(model, x_test, y_test)