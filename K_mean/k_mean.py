import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def auto_k_means(data, k_max=10):
    """
    自动确定聚类数的改进 k 均值算法
    :param data: 输入数据（numpy 数组）
    :param k_max: 最大可能的聚类数
    :return: 最优聚类数 k 和聚类结果
    """
    best_k = 2
    best_score = -1
    best_model = None
    scores = []

    # 尝试不同的聚类数
    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)  # 计算轮廓系数
        scores.append(score)

        if score > best_score:
            best_k = k
            best_score = score
            best_model = model

    # 返回最优 k 和对应的模型
    return best_k, best_model, scores


# 生成测试数据
data, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# 调用改进 k 均值算法
k_max = 10
best_k, best_model, scores = auto_k_means(data, k_max)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 输出最优聚类数和性能
print(f"自动确定的最优聚类数: {best_k}")
plt.plot(range(2, k_max + 1), scores, marker='o')
plt.xlabel('聚类数 k')
plt.ylabel('轮廓系数')
plt.title('聚类数与轮廓系数的关系')
plt.show()

# 可视化聚类结果
labels = best_model.labels_
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(best_model.cluster_centers_[:, 0], best_model.cluster_centers_[:, 1], c='red', marker='.', s=100,
            label='中心点')
plt.title(f'最佳聚类数 k={best_k}')
plt.legend()
plt.show()
