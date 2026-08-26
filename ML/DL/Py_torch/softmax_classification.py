import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)
#
#
# #softmax
# z = torch.FloatTensor([1, 2, 3])
# hypothesis = F.softmax(z, dim=0)
# print(hypothesis)
#
# hypothesis.sum()
# print(hypothesis.sum())
#
#
# #cross entropy
# z1 = torch.rand(3, 5, requires_grad=True)
# hypothesis_1 = F.softmax(z1, dim=1)
# print(hypothesis_1)
#
# y = torch.randint(5, (3,)).long()
# print(y)
#
# #cross entropy loss
# y_one_hot = torch.zeros_like(hypothesis_1)
# y_one_hot.scatter_(1, y.unsqueeze(1), 1)
# print(y_one_hot)
#
# cost = (y_one_hot * -torch.log(hypothesis_1)).sum(dim=1).mean()
# print(cost)
#
# #cross_entropy loss with torch.nn.functional
# torch.log(F.softmax(z1, dim=1))
# F.log_softmax(z1, dim=1)
# (y_one_hot * -torch.log(hypothesis_1)).sum(dim=1).mean()
# F.nll_loss(F.log_softmax(z1, dim=1), y)
# F.cross_entropy(z1, y)
# print(F.cross_entropy(z1, y))

x_train = torch.FloatTensor([
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7]
])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifier()
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    prediction = model(x_train)
    cost =  F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch{epoch:4d}/{nb_epochs}, cost{cost.item():.6f}')

