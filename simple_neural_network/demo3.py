import torch

batch_n = 100
input_data = 1000
hidden_layer = 100
output_data = 10

x = torch.randn(batch_n, input_data, requires_grad=False)
y = torch.randn(batch_n, output_data, requires_grad=False)

# 定义模型
# 输入：torch.nn.Sequential 的输入是一个张量（tensor）。
# 具体来说，它是网络的输入数据，通常是一个形状为 (batch_size, input_features) 或 (batch_size, channels, height, width) 的张量（取决于网络类型）。
# 输出：torch.nn.Sequential 会依次将输入传递给其中的每一层，最终输出的张量是最后一层的输出。
model = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),  # 第一层：线性层，输入维度为 input_data，输出维度为 hidden_layer
    torch.nn.ReLU(),  # 激活函数：ReLU，用于增加非线性
    torch.nn.Linear(hidden_layer, output_data)  # 第二层：线性层，输入维度为 hidden_layer，输出维度为 output_data
)

print(model)

learning_rate = 1e-4
epoch_n = 10000
for epoch in range(epoch_n):
    # 1. 前向传播：将输入数据 x 传入模型，获得模型的预测输出 y_pred
    y_pred = model(x)
    # 2. 定义损失函数：均方误差损失函数（MSELoss），用于回归任务
    loss_fn = torch.nn.MSELoss()
    # 3. 计算损失：比较预测结果 y_pred 与实际值 y 之间的差异
    loss = loss_fn(y_pred, y)
    print("epoch:{},loss:{}".format(epoch, loss))
    # 4. 清零模型参数的梯度：每次反向传播前，需要将上一步的梯度清除掉
    model.zero_grad()
    # 5. 反向传播：计算梯度
    loss.backward()
    # 6. 手动更新参数：使用梯度下降法更新模型参数
    for param in model.parameters():
        # 更新模型的参数：参数 -= 学习率 * 参数的梯度
        param.data -= learning_rate * param.grad.data
