import torch
from torch import nn
from torchvision import datasets, transforms, models
import os
import time

# 数据集路径
data_dir = '../DogsVSCats'

# 检查是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：调整图片大小为 224x224，归一化到 [0,1]，然后标准化到 [-1,1]
data_transforms = {x: transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                   for x in ['train', 'valid']}

# 创建训练集和验证集的数据集对象
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),  # 数据集路径
                                          transform=data_transforms[x])  # 应用预处理方法
                  for x in ['train', 'valid']}

# 创建训练集和验证集的 DataLoader，便于批量读取数据
dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                             batch_size=16,  # 每批次 16 张图片
                                             shuffle=True)  # 打乱数据
              for x in ['train', 'valid']}

# 从训练集中取出一个批次数据，检查数据和标签的维度
X_example, Y_example = next(iter(dataloader['train']))
print(X_example.shape, Y_example.shape)  # 打印图片和标签的 shape
example_classes = image_datasets['train'].classes  # 获取类别名称
index_classes = image_datasets['train'].class_to_idx  # 获取类别对应的索引
print(example_classes)
print(index_classes)

# 指定本地权重文件路径
pretrained_weights_path = r'D:\Pycharm Project\machine_learning\DogsVSCats\resnet50-0676ba61.pth'  # 替换为你本地的路径

# 创建 ResNet50 模型，不加载官方预训练权重
model = models.resnet50(pretrained=False)

# 加载本地权重到模型
state_dict = torch.load(pretrained_weights_path)  # 加载本地权重文件
model.load_state_dict(state_dict)  # 将权重应用到模型

# 冻结 ResNet50 的所有参数（即不更新预训练权重）
for param in model.parameters():
    param.requires_grad = False

# 替换 ResNet50 的全连接层（fc），将其输出改为二分类任务
model.fc = torch.nn.Linear(in_features=2048, out_features=2)

# 将模型放到指定设备（CPU 或 GPU）
model = model.to(device)

# 定义损失函数（交叉熵损失，适用于分类任务）
loss_fn = torch.nn.CrossEntropyLoss()

# 定义优化器，只优化模型中最后一层的参数（全连接层）
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)

epoch_n = 5
time_start = time.time()

# 开始训练和验证循环
for epoch in range(epoch_n):
    print('Epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)

    for phase in ['train', 'valid']:
        if phase == 'train':
            print("Training...")
            model.train(True)  # 启用训练模式（影响 Dropout、BatchNorm 等）
        else:
            print("validating...")
            model.train(False)  # 禁用训练模式

        running_loss = 0.0  # 累计损失
        running_corrects = 0  # 累计预测正确的样本数量

        # 遍历当前阶段的所有批次数据
        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data  # 获取图片和对应标签
            X, y = X.to(device), y.to(device)  # 将数据转移到指定设备

            y_pred = model(X)  # 前向传播，获取模型预测值

            _, pred = torch.max(y_pred, 1)  # 获取预测的类别索引（取每行最大值的索引）

            optimizer.zero_grad()  # 清除优化器的梯度
            loss = loss_fn(y_pred, y)  # 计算当前批次的损失

            if phase == 'train':
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新权重

            running_loss += loss.item()  # 累计损失
            running_corrects += torch.sum(pred == y.data)  # 累计预测正确的数量

            # 打印训练阶段的每批次损失和准确率
            if phase == 'train':
                print("Batch {},Train Loss:{:.4f},Train ACC:{:.4f}%".format(batch, running_loss / batch,
                                                                            100 * running_corrects / (batch * 16)))

        # 计算当前阶段的平均损失和准确率
        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])
        print("{},Loss:{:.4f},Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

# 记录并打印整个训练过程所用时间
time_end = time.time()
print('Time taken for epoch {} is {} sec'.format(epoch + 1, time_end - time_start))
