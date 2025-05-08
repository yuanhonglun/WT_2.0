import pickle
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


# 设置随机数种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



class SparseDataset(Dataset):
    def __init__(self, pkl_file):
        # 从 pickle 文件加载数据
        with open(pkl_file, 'rb') as f:
            self.pkl_file = pickle.load(f)

        # 检查数据是否存在
        if 'intensities' not in self.pkl_file or 'labels' not in self.pkl_file:
            raise ValueError("缺少 'features' 或 'target' 在 PKL 文件中")

        self.features = self.pkl_file['intensities']  # 数据特征
        self.target = self.pkl_file['labels']  # 目标值

        # 验证数据维度
        if self.features.shape[1] != 51302:  # 假设特征维度为 51302
            raise ValueError(f"期望特征维度 51302，但实际为 {self.features.shape[1]}")

        # 检查是否有缺失值
        if np.any(np.isnan(self.features.data)):  # 检查稀疏矩阵中的非零元素
            raise ValueError("特征中在非零元素位置存在 NaN 值")

        if np.any(np.isnan(self.target)):
            raise ValueError("目标中存在 NaN 值")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # 获取单个样本
        x = self.features[idx].toarray().astype(np.float32)  # 转换为稠密矩阵
        x = np.squeeze(x)  # 去掉多余的维度，确保形状是 (特征数,)

        y = np.array(self.target[idx], dtype=np.float32)  # 确保目标值是 numpy 数组

        # 将数据转换为 torch.Tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor

def get_data_loader(pkl_file, batch_size=64, val=True):


    # 从 pickle 文件加载完整数据集
    dataset = SparseDataset(pkl_file)
    dataset_size = len(dataset)

    if val:
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return val_loader
    else:


        train_set = []
        val_set = []
        test_set = []

        # 每100个样本一组进行处理
        for i in range(0, dataset_size, 100):
            batch_indices = list(range(i, min(i + 100, dataset_size)))  # 获取当前窗口的索引

            if len(batch_indices) == 100:  # 确保当前窗口有100个样本
                # 随机选择70个用于训练，20个用于验证，10个用于测试
                random_indices = random.sample(batch_indices, 100)  # 随机打乱当前窗口的索引
                train_indices = random_indices[:70]
                val_indices = random_indices[70:90]
                test_indices = random_indices[90:]

                train_set.extend(train_indices)
                val_set.extend(val_indices)
                test_set.extend(test_indices)

        # 创建子集
        train_dataset = torch.utils.data.Subset(dataset, train_set)
        val_dataset = torch.utils.data.Subset(dataset, val_set)
        test_dataset = torch.utils.data.Subset(dataset, test_set)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader  # 返回训练集、验证集和测试集的数据加载器


# val_loader = get_data_loader(r"D:\work\WT2.0\WT_2\test_data\result\test.pkl", batch_size=32, val=True)
#
#
# for x, y in val_loader:
#     print(x.shape, y.shape)  # 打印验证数据的形状
#     break

