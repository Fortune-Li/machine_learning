import torch

# 1. 初始化参数
batch_n = 100
input_data = 1000
hidden_layer = 100
output_data = 10

# 2. 生成模拟数据
x = torch.randn(batch_n, input_data, requires_grad=False)
y = torch.randn(batch_n, output_data, requires_grad=False)

# 3. 定义神经网络模型
model = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, output_data),
)

# 4. 设置超参数
learning_rate = 1e-4  # 学习率：梯度下降的步长
epoch_n = 40

# 5. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
# 使用了 Adam 优化器。Adam 是一种自适应学习率的优化算法，它结合了动量和梯度的自适应调整。使用 model.parameters() 来传递模型中的所有参数，并设置学习率。
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epoch_n):
    # 6. 前向传播：将输入数据 x 传入模型，获得预测值 y_pred
    y_pred = model(x)
    # 7. 计算损失：使用损失函数计算预测值和真实值 y 之间的误差
    loss = loss_fn(y_pred, y)
    print("epoch: {}, loss: {}".format(epoch, loss))
    # 8. 清除梯度：每次反向传播前需要清除旧的梯度值
    optimizer.zero_grad()
    # 9. 反向传播：计算损失函数关于模型参数的梯度
    loss.backward()
    # 10. 更新参数：使用优化器更新模型参数
    optimizer.step()
