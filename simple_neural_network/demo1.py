import torch

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 100

# requires_grad参数 如果requires_grad=False 那么该变量在进行自动梯度计算中不会保留梯度值
# 由于x y 是输入输出因此不需要进行梯度计算
x = torch.randn(batch_n, input_data, requires_grad=False)
y = torch.randn(batch_n, output_data, requires_grad=False)

w1 = torch.randn(input_data, hidden_layer, requires_grad=True)
w2 = torch.randn(hidden_layer, output_data, requires_grad=True)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print("epoch:{}, loss:{:.4f}".format(epoch, loss.item()))

    loss.backward()  # 反向传播

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
