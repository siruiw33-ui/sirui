import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)  # Shape: (6, 2)
y_train = torch.FloatTensor(y_data)  # Shape: (6, 1)


class BinaryClassifier(nn.Module):

  def __init__(self):
    super().__init__()

    self.linear = nn.Linear(2, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    return self.sigmoid(self.linear(x))



model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)


nb_epochs = 1000
for epoch in range(nb_epochs + 1):


  hypothesis = model(x_train)


  cost = F.binary_cross_entropy(hypothesis, y_train)


  optimizer.zero_grad()
  cost.backward()
  optimizer.step()


  if epoch % 100 == 0:
    print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')


print('\n--- 预测结果 ---')
with torch.no_grad():
  hypothesis = model(x_train)
  prediction = hypothesis >= 0.5
  correct_prediction = prediction.float() == y_train
  accuracy = correct_prediction.sum().item() / len(correct_prediction)

  print('预测概率 (Hypothesis):\n', hypothesis.numpy())
  print('预测标签 (Prediction):\n', prediction.int().numpy())
  print(f'准确率 (Accuracy): {accuracy * 100:.2f}%')