import torch
from tqdm import tqdm
from torch import optim

#data definition
x_train = torch.tensor([[1.0], [2.0], [3.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0]])

#hypothesis
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train*w + b

#compute loss
cost = torch.mean((hypothesis - y_train) **2)
print(cost)


optimizer = optim.SGD([w, b], lr=0.01)

nb_epochs = 1000
pbar = tqdm(range(1, nb_epochs + 1), desc="Training Model")
for epoch in range(1, nb_epochs+1):
    hypothesis = x_train*w + b
    cost = torch.mean((hypothesis - y_train) **2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    pbar.set_postfix(
        loss=f"{cost.item():.6f}",
        w=f"{w.item():.4f}",
        b=f"{b.item():.4f}")


print(f"\nTraining Finished! Final results: w = {w.item():.4f}, b = {b.item():.4f}")
