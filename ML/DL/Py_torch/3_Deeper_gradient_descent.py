import torch

# 1. 模拟数据 (Dummy Data)
x_train = torch.FloatTensor([[1.0], [2.0], [3.0]])
y_train = torch.FloatTensor([[1.0], [2.0], [3.0]])

# 2. 模型初始化 (不开启自动求导)
W = torch.zeros(1)
lr = 0.1
nb_epochs = 10

# 3. 训练循环
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 手动推导并计算梯度 (Math: 2 * mean((W*x - y) * x))
    gradient = torch.mean(2 * (W * x_train - y_train) * x_train)

    # 手动更新 W
    W -= lr * gradient

    print(f"Epoch {epoch:4d}/{nb_epochs} W: {W.item():.3f}, Cost: {cost.item():.6f}")