import torch

batch_n = 100  # 数据量
hidden_layer = 100  # 经过隐藏层后保留的数据特征个数
input_data = 1000  # 每个数据的数据特征
output_data = 10  # 输出的输出 表示得到的10个分类结果值

x = torch.randn(batch_n, input_data)
y = torch.randn(batch_n, output_data)

w1 = torch.randn(input_data, hidden_layer)
w2 = torch.randn(hidden_layer, output_data)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    # 前向传播
    h1 = x.mm(w1)  # (100,1000)*(1000,100)=(100,100)
    h1 = h1.clamp(min=0)  # 用于将张量中的元素限制在指定的范围内
    y_pred = h1.mm(w2)  # (100,100)*(100,10)=(100,10)

    # 损失函数 均方误差函数
    loss = (y_pred - y).pow(2).sum() / batch_n
    print("Epoch:{},Loss:{}".format(epoch, loss))

    # 反向传播
    grad_y_pred = 2 * (y_pred - y)  # 对均方误差函数进行梯度下降(偏导数）
    # h1是(batch_size,hidden_size),转置(.t())后变成(hidden_size,batch_size)
    # grad_y_pred这是损失函数对预测值 y_pred 的梯度,形状为(batch_size, output_size)
    grad_w2 = h1.t().mm(grad_y_pred)

    grad_h = grad_y_pred.clone()  # (batch_n,output_data)
    grad_h = grad_h.mm(w2.t())  # (batch_n,output_data)*(output_data,hidden_layer)
    grad_h.clamp_(min=0)
    grad_w1 = x.t().mm(grad_h)  # (input_data,batch_n)*(batch_n,hidden_layer)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# 这段代码实现了一个简单的全连接神经网络模型，包含一个输入层、一个隐藏层和一个输出层。模型的目标是通过梯度下降优化参数，最小化损失函数（均方误差）
