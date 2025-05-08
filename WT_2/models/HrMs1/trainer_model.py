import torch
from tqdm import tqdm  # 导入 tqdm 库以显示进度条
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.all_train_losses = []  # 保存所有折的训练损失
        self.all_val_losses = []  # 保存所有折的验证损失

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc='Training', unit='batch'):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()  # 梯度清零
            output = self.model(x)  # 前向传播
            loss = self.criterion(output, y)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新权重

            total_loss += loss.item() * x.size(0)  # 累计损失

        avg_loss = total_loss / len(train_loader.dataset)  # 返回平均损失
        self.train_losses.append(avg_loss)  # 保存训练损失
        return avg_loss

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():  # 评估时不需要计算梯度
            for x, y in tqdm(data_loader, desc='Validating', unit='batch'):
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x)  # 前向传播
                loss = self.criterion(output, y)  # 计算损失
                total_loss += loss.item() * x.size(0)  # 累计损失

                all_outputs.append(output.cpu().numpy())  # 收集模型输出
                all_targets.append(y.cpu().numpy())  # 收集真实标签

        avg_loss = total_loss / len(data_loader.dataset)  # 返回平均损失
        self.val_losses.append(avg_loss)  # 保存验证损失

        # 计算 R²、MSE 和 MAE
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        r2 = r2_score(all_targets, all_outputs)
        mse = mean_squared_error(all_targets, all_outputs)
        mae = mean_absolute_error(all_targets, all_outputs)

        return avg_loss, r2, mse, mae  # 返回损失和新指标

    def train_with_validation(self, train_loader, val_loader, scheduler, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)  # 训练一个 epoch
            val_loss, r2, mse, mae = self.evaluate(val_loader)  # 验证模型并计算指标
            scheduler.step(val_loss)  # 学习率调整

            # 更新所有折的损失记录
            self.all_train_losses.append(train_loss)
            self.all_val_losses.append(val_loss)

            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '  
                  f'R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}')

            # 调整学习率
            if scheduler:
                scheduler.step(val_loss)  # 可以根据需要使用 train_loss 或 val_loss

        # 绘制所有折的总体损失图
        self.plot_total_losses(len(self.all_train_losses), 'total_losses.png')

    def plot_total_losses(self, total_epochs, out):
        plt.figure(figsize=(12, 6))
        plt.plot(range(total_epochs), self.all_train_losses, label='Total Training Loss', color='blue')
        plt.plot(range(total_epochs), self.all_val_losses, label='Total Validation Loss', color='orange')
        plt.title(f'Total Training and Validation Loss Over All Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(out)
        plt.close()