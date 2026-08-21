
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class MultivariateLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

x_train = torch.FloatTensor([[73, 80, 75],
                    [93, 88, 93],
                    [89, 91, 90],
                    [96, 98, 100],
                    [73, 66, 70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

model = MultivariateLinearRegression()

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')