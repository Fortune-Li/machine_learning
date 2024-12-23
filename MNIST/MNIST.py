import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class Model(torch.nn.Module):
    # 初始化模型，定义卷积层和全连接层
    def __init__(self):
        # 调用父类的构造方法
        super(Model, self).__init__()
        # 定义卷积层部分
        self.conv1 = torch.nn.Sequential(  # conv1是torch.nn.Sequential类的一个实例
            # 第一个卷积层：输入通道数为1（灰度图像），输出通道数为64，卷积核大小为3x3，步长为1，填充为1
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # 激活函数ReLU
            torch.nn.ReLU(),
            # 第二个卷积层：输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 激活函数ReLU
            torch.nn.ReLU(),
            # 最大池化层：池化窗口大小为2x2，步长为2，进行下采样
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 定义全连接层部分
        self.dense = torch.nn.Sequential(
            # 第一个全连接层：输入大小为14*14*128（卷积输出的大小），输出大小为1024
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            # Dropout层：丢弃50%的神经元，有助于减少过拟合
            torch.nn.Dropout(0.5),
            # 第二个全连接层：输入大小为1024，输出大小为10（10个类别的分类任务）
            torch.nn.Linear(1024, 10))

    # 前向传播方法
    def forward(self, x):
        # 输入数据通过卷积层部分
        x = self.conv1(x)
        # 卷积层输出的特征图需要展平为一维向量，传入全连接层
        x = x.view(-1, 14 * 14 * 128)  # 将特征图展平成一维向量，-1表示自动推断大小
        # 输入数据通过全连接层部分
        x = self.dense(x)
        return x


# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理流程：将图片转为Tensor并进行标准化
transforms = transforms.Compose([transforms.ToTensor(),  # 将PIL图片转换为Tensor
                                 transforms.Normalize(mean=[0.5], std=[0.5]),  # 使用均值和标准差进行标准化
                                 # torchvision.transforms.RandomHorizontalFlip() # 对图片进行水平翻转
                                 ])

# 下载训练和测试数据集
data_train = datasets.MNIST(root='./data', transform=transforms, train=True, download=True)
data_test = datasets.MNIST(root='./data', transform=transforms, train=False)

# 创建训练和测试数据加载器
data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=True)

# 获取一个批次的数据（64张图片及其标签）
images, labels = next(iter(data_loader_train))

# 将图像拼接成一个网格
img = torchvision.utils.make_grid(images)
# 图片维度变为(C,H,W)
img = img.numpy().transpose((1, 2, 0))  # Matplotlib使用的数据维度必须是(H，W，C)，故转换维度为 (H, W, C)

# 反标准化（从 [-1, 1] 恢复到 [0, 1] 范围）
std = [0.5]
mean = [0.5]
img = img * std + mean  # 由于在数据预处理时，图像被标准化到 [-1, 1] 范围，img = img * std + mean 是为了将图像恢复到 [0, 1] 范围，确保显示正确的图像。
print([labels[i] for i in range(64)])

# 使用Matplotlib显示图片
plt.imshow(img)
plt.show()

# 初始化模型，并将模型移到GPU/CPU
model = Model().to(device)
# model = Model()

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(model)

epoch_n = 1

for epoch in range(epoch_n):

    running_loss = 0.0
    running_corrects = 0
    print('Epoch {}/{}'.format(epoch + 1, epoch_n))
    print('-' * 10)

    for train_data in data_loader_train:
        images_train, labels_train = train_data

        # 将数据移到GPU/CPU
        images_train, labels_train = images_train.to(device), labels_train.to(device)

        output = model(images_train)

        # torch.max(input, dim) 计算沿着 dim 维度的最大值，返回两个张量：
        # 第一个返回值：每行的最大值。第二个返回值：最大值所在的索引。这里dim=1表示按行计算最大值。
        _, pred = torch.max(output.data, 1)  # output 中的最大值的索引就是模型预测的手写数字（0-9）对应的标签（label）

        optimizer.zero_grad()
        loss = cost(output, labels_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(pred == labels_train.data)

        testing_correct = 0
        for test_data in data_loader_test:
            images_test, labels_test = test_data

            # 将数据移到GPU/CPU
            images_test, labels_test = images_test.to(device), labels_test.to(device)

            output = model(images_test)
            _, pred = torch.max(output.data, 1)
            testing_correct += torch.sum(pred == labels_test.data)
        print("Loss is:{:.4f} ,Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}%"
              .format((running_loss / len(data_train)), 100 * running_corrects / len(data_train),
                      100 * testing_correct / len(data_test)))

data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=9, shuffle=True)
x_test, y_test = next(iter(data_loader_test))
x_test, y_test = x_test.to(device), y_test.to(device)
pred = model(x_test)
_, pred = torch.max(pred.data, 1)

print("Predict Label is :", [i for i in pred.data])
print("Real Label is :", [i for i in y_test])

img = torchvision.utils.make_grid(x_test)
img = img.numpy().transpose((1, 2, 0))

std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
plt.imshow(img)
plt.show()
