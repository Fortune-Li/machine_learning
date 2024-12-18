import torch
import torchvision
from torchvision import datasets, transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import Model

data_dir = "../DogsVSCats"

# 设置计算设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理操作
data_transforms = {x: transforms.Compose([transforms.Resize([64, 64]),  # 将图像的尺寸调整为 64x64 像素
                                          transforms.ToTensor()])  # 将图像转换为 PyTorch 张量，并将像素值归一化到 [0, 1]
                   for x in ['train', 'valid']}  # 针对 'train' 和 'valid' 数据集分别定义变换

# 使用 ImageFolder 加载图像数据集，并应用对应的预处理操作
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),  # 拼接路径
                                          transform=data_transforms[x])  # 应用对应的图像预处理操作
                  for x in ['train', 'valid']}

# 创建数据加载器 DataLoader，用于批量加载数据
dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],  # 加载的数据集（对应 'train' 和 'valid'）
                                             batch_size=16,  # 每个批次的图像数量设为 16
                                             shuffle=True)  # 打乱数据集中的样本顺序
              for x in ['train', 'valid']}

# 显示一张样本图像和它的标签
x_example, y_example = next(iter(dataloader['train']))  # 从训练集加载一个批次的样本
print(x_example.shape, y_example.shape)
index_classes = image_datasets['train'].class_to_idx  # 获取类别到索引的映射
print(index_classes)
example_classes = image_datasets['train'].classes  # 获取类别的名称
# print(example_classes)

img = torchvision.utils.make_grid(x_example)  # 将样本图像合并为一个网格
# (1, 2, 0) 表示要交换张量的轴，顺序为：将原本的第 1 维（H）和第 2 维（W）保持不变，且把原本的第 0 维（C）放到最后。
img = img.numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.show()
print([example_classes[label] for label in y_example])  # 打印对应的标签

# 创建模型实例并将其移动到指定设备（GPU 或 CPU）
model = Model.Model()
model = model.to(device)
# print(model)

# 设置损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 使用 Adam 优化器

epoch_n = 10  # 设置训练的总轮数
time_start = time.time()  # 记录训练开始的时间

# 开始训练循环
for epoch in range(epoch_n):  # 循环训练多轮
    print('Epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)

    # 训练和验证阶段
    for phase in ['train', 'valid']:
        if phase == 'train':  # 如果是训练阶段
            print("Training...")
            model.train(True)  # 设置模型为训练模式
        else:
            print("Validating...")
            model.train(False)  # 设置模型为评估模式

        # 初始化每个阶段的运行损失和正确预测数
        running_loss = 0.0
        running_corrects = 0

        # 遍历每个批次的数据
        for batch, data in enumerate(dataloader[phase], 1):  # 每次迭代一个批次的数据
            X, y = data  # 获取输入图像（X）和对应的标签（y）

            X, y = X.to(device), y.to(device)  # 将输入和标签移到指定设备（GPU 或 CPU）

            y_pred = model(X)  # 将输入传入模型，得到预测结果

            _, pred = torch.max(y_pred.data, 1)  # 获取预测结果的类别标签

            optimizer.zero_grad()  # 清除梯度

            loss = loss_fn(y_pred, y)  # 计算损失

            if phase == 'train':  # 只有在训练阶段才执行反向传播
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失
            running_corrects += torch.sum(pred == y.data)  # 累加正确预测的数量

            if phase == 'train':
                print("Batch {},Train Loss:{:.4f},Train ACC:{:.4f}".format(batch, running_loss / batch,
                                                                           100 * running_corrects / (batch * 16)))

        # 计算每个阶段（训练/验证）的最终损失和准确率
        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])
        print("{},Loss:{:.4f},Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

time_end = time.time()
print('Time taken for epoch {} is {} sec'.format(epoch + 1, time_end - time_start))
