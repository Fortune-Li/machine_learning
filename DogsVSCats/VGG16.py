import torch
from torchvision import datasets, transforms, models
import os
import time


def main():
    data_dir = "../DogsVSCats"  # 数据集路径

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置计算设备

    # 定义数据预处理操作
    data_transforms = {x: transforms.Compose([transforms.Resize((224, 224)),  # 调整图片尺寸为 224x224（VGG 模型输入大小）
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                       for x in ['train', 'valid']}  # 针对 'train' 和 'valid' 数据集分别定义变换

    # 加载数据集并应用预处理
    image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                              transform=data_transforms[x])  # 应用数据变换
                      for x in ['train', 'valid']}

    # 创建数据加载器
    dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                                 batch_size=16,
                                                 shuffle=True,
                                                 num_workers=4)  # 设置多线程加载数据
                  for x in ['train', 'valid']}

    # 获取 dataloader[phase] 中的第一个批次数据，而不会遍历整个数据集。
    X_example, Y_example = next(iter(dataloader['train']))  # 常用于快速检查数据的形状或内容，而不是用于完整训练过程。
    print(X_example.shape, Y_example.shape)
    example_classes = image_datasets['train'].classes  # 数据集中的类别名
    index_classes = image_datasets['train'].class_to_idx  # 类别名到索引的映射

    print(example_classes)
    print(index_classes)

    # 预训练模型（例如 ResNet、VGG 等）通常是在大型数据集（如 ImageNet）上训练的，它们已经学会了丰富的特征表示
    model = models.vgg16(pretrained=True)  # 加载预训练的 VGG16 模型

    # 冻结预训练模型参数
    for parma in model.parameters():  # 返回模型中所有的参数（权重和偏置）

        # 冻结这些参数，可以直接利用它们作为特征提取器，而无需重新训练
        parma.requires_grad = False  # 冻结模型的所有参数，在反向传播时不要计算这些参数的梯度，从而避免更新它们。

    # 设置 model.classifier 后，冻结的卷积层参数并不会自动解开,而分类器所在的全连接层的参数是默认可训练的。
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),  # 全连接层，输入维度为 25088，输出维度为 4096
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.5),  # 随机失活，防止过拟合
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.5),
                                           torch.nn.Linear(4096, 2))  # 最后一层，输出类别数为 2

    model = model.to(device)  # 将模型加载到计算设备上

    loss_fn = torch.nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)  # 定义优化器，只更新分类器部分的参数

    epoch_n = 5  # 定义训练的轮数
    time_start = time.time()  # 记录训练开始时间

    for epoch in range(epoch_n):  # 遍历每一轮训练
        print('Epoch {}/{}'.format(epoch, epoch_n - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:  # 分别对训练集和验证集进行操作
            if phase == 'train':
                print("Training...")
                model.train(True)  # 将模型设置为训练模式
            else:
                print("Validating...")
                model.train(False)  # 将模型设置为验证模式

            running_loss = 0.0  # 累计损失
            running_corrects = 0  # 累计正确预测样本数

            # 和next(iter(dataloader[phase])) 的作用不同
            # enumerate将 dataloader[phase] 中的每个批次（batch）进行迭代，并为每个批次生成一个带有索引的元组 (index, data)。
            for batch, data in enumerate(dataloader[phase], 1):  # 遍历整个 DataLoader 数据集。
                X, y = data  # 获取输入数据和标签
                X, y = X.to(device), y.to(device)  # 将数据加载到计算设备上

                y_pred = model(X)  # 前向传播，预测输出

                _, pred = torch.max(y_pred, 1)  # 获取预测类别

                optimizer.zero_grad()  # 清除梯度

                loss = loss_fn(y_pred, y)  # 计算损失

                if phase == 'train':
                    loss.backward()  # 反向传播，计算梯度
                    optimizer.step()  # 更新参数

                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)

                if phase == 'train':
                    print("Batch {},Train Loss: {:.4f},Train Acc: {:.4f}%"
                          .format(batch, running_loss / batch, 100 * running_corrects / (16 * batch)))

            epoch_loss = running_loss * 16 / len(image_datasets[phase])
            epoch_acc = 100 * running_corrects / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}%'.format(phase, epoch_loss, epoch_acc))

    time_end = time.time()
    print('Time: {:.4f}s'.format(time_end - time_start))


if __name__ == '__main__':  # 必须使用主模块导入，防止多线程问题
    main()
