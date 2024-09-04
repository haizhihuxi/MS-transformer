import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from config_SEA import Config
import math
import matplotlib.pyplot as plt
import os


def hausdorff_distance(P, Q):
    P = P.cpu().numpy().astype(np.float32)
    Q = Q.cpu().numpy().astype(np.float32)

    def directed_hausdorff(A, B):
        max_dist = 0
        for a in A:
            min_dist = float('inf')
            for b in B:
                dist = np.linalg.norm(a - b)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > max_dist:
                max_dist = min_dist
        return max_dist

    return torch.tensor(max(directed_hausdorff(P, Q), directed_hausdorff(Q, P)), dtype=torch.float32)

def erp_distance(P, Q, gap_penalty=1.0):
    P = P.cpu().numpy().astype(np.float32)
    Q = Q.cpu().numpy().astype(np.float32)
    n = len(P)
    m = len(Q)
    ca = np.full((n, m), -1.0, dtype=np.float32)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(P[0] - Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0) + gap_penalty, np.linalg.norm(P[i] - Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1) + gap_penalty, np.linalg.norm(P[0] - Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(c(i - 1, j) + gap_penalty, c(i - 1, j - 1), c(i, j - 1) + gap_penalty),
                np.linalg.norm(P[i] - Q[j])
            )
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return torch.tensor(c(n - 1, m - 1), dtype=torch.float32)

def frechet_distance(P, Q):
    P = P.cpu().numpy().astype(np.float32)
    Q = Q.cpu().numpy().astype(np.float32)
    n = len(P)
    m = len(Q)
    ca = np.full((n, m), -1.0, dtype=np.float32)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(P[0] - Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), np.linalg.norm(P[i] - Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), np.linalg.norm(P[0] - Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                np.linalg.norm(P[i] - Q[j])
            )
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return torch.tensor(c(n - 1, m - 1), dtype=torch.float32)


def dtw_distance(P, Q):
    # 将 P 和 Q 转换为 NumPy 数组
    P = P.cpu().numpy().astype(np.float32)
    Q = Q.cpu().numpy().astype(np.float32)

    # 获取 P 和 Q 的长度
    n = len(P)
    m = len(Q)

    # 初始化 DTW 矩阵，大小为 (n, m)，初始值为无穷大
    dtw = np.full((n, m), float('inf'))
    dtw[0, 0] = 0  # 起点的 DTW 距离为 0

    # 初始化第一列
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + np.linalg.norm(P[i] - Q[0])

    # 初始化第一行
    for j in range(1, m):
        dtw[0, j] = dtw[0, j - 1] + np.linalg.norm(P[0] - Q[j])

    # 填充 DTW 矩阵
    for i in range(1, n):
        for j in range(1, m):
            cost = np.linalg.norm(P[i] - Q[j])  # 当前点之间的距离
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])  # 选择最小的累计距离

    # 返回 DTW 矩阵的右下角值，即两条轨迹的最小累计距离
    return torch.tensor(dtw[n - 1, m - 1], dtype=torch.float32)


def cosine_similarity(P, Q):
    # 将 P 和 Q 转换为 NumPy 数组
    P_np = P.numpy().astype(np.float32)
    Q_np = Q.numpy().astype(np.float32)

    # 计算速度向量
    def compute_velocity(traj):
        if len(traj) < 2:
            return np.array([[0, 0]], dtype=np.float32)  # 长度为1时返回零向量
        velocity = []
        for i in range(len(traj) - 1):
            v = traj[i + 1] - traj[i]
            velocity.append(v)
        return np.array(velocity)

    velocity_P = compute_velocity(P_np)
    velocity_Q = compute_velocity(Q_np)

    # 计算余弦相似度
    dot_product = np.sum(velocity_P * velocity_Q, axis=1)
    norm_P = np.linalg.norm(velocity_P, axis=1)
    norm_Q = np.linalg.norm(velocity_Q, axis=1)
    cosine_sim = dot_product / (norm_P * norm_Q + 1e-8)  # 加上一个小的常数防止除零

    # 返回余弦相似度的均值
    return torch.tensor(np.mean(cosine_sim), dtype=torch.float32)


class similarity_Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(similarity_Attention, self).__init__()
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 使用矩阵乘法和转置操作计算 attention_scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.hidden_dim ** 0.5
        # if mask is not None:
        #     mask = mask.unsqueeze(1).expand_as(attention_scores)
        #     attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights


class TrajectoryMatchingNetwork(nn.Module):
    def __init__(self, config):
        super(TrajectoryMatchingNetwork, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.window_size_10 = config.window_size_10
        self.window_size_5 = config.window_size_5
        self.window_size_15 = config.window_size_15
        self.window_size_20 = config.window_size_20
        self.frechet = config.frechet_judge
        self.dtw = config.dtw_judge
        self.cos = config.cos_judge
        self.erp = config.erp_judge
        self.hausdorff = config.hausdorff_judge

        # 线性层将轨迹坐标转换为嵌入向量
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.attention = similarity_Attention(self.hidden_dim)

        # 为不同的窗口大小创建全连接层
        self.fc_10 = self.create_fc_layer(self.window_size_10)
        self.fc_5 = self.create_fc_layer(self.window_size_5)
        self.fc_15 = self.create_fc_layer(self.window_size_15)
        self.fc_20 = self.create_fc_layer(self.window_size_20)
        # 为不同的窗口大小创建输出层
        self.out_10 = self.create_output_layer(self.window_size_10)
        self.out_5 = self.create_output_layer(self.window_size_5)
        self.out_15 = self.create_output_layer(self.window_size_15)
        self.out_20 = self.create_output_layer(self.window_size_20)

        # 最后的输出层

    def create_output_layer(self, window_s):
        return nn.Sequential(
            nn.Linear(self.output_dim * window_s, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def create_fc_layer(self, window_s):
        return nn.Sequential(
            nn.Linear(2 * window_s, 4 * window_s),
            nn.ReLU(),
            nn.Linear(4 * window_s, 2 * window_s),
            nn.ReLU(),
            nn.Linear(2 * window_s, self.output_dim),
            nn.ReLU()
        )

    def plot_trajectories(self, traj_1, traj_2, sim_score_10, save_dir, count, index1, index2):
        plt.figure(figsize=(9, 6), dpi=150)
        cmap = plt.colormaps.get_cmap("jet")
        color = cmap(0.5)

        # 将 sim_score_judge 转换为 numpy 数组
        sim_scores = sim_score_10.cpu().numpy()

        # 绘制轨迹 1
        plt.plot(traj_1[:, 1].cpu().numpy(), traj_1[:, 0].cpu().numpy(), linestyle="-", color=color,
                 label="Trajectory 1")
        plt.plot(traj_1[0, 1].cpu().numpy(), traj_1[0, 0].cpu().numpy(), "o", markersize=5, color=color)  # 标出起始点

        # 绘制轨迹 2
        plt.plot(traj_2[:, 1].cpu().numpy(), traj_2[:, 0].cpu().numpy(), linestyle="-.", color="red",
                 label="Trajectory 2")
        plt.plot(traj_2[0, 1].cpu().numpy(), traj_2[0, 0].cpu().numpy(), "o", markersize=5, color="red")  # 标出起始点

        # 标注相似度得分
        for i, score in enumerate(sim_scores):
            plt.text(0.5, 1.02 - i * 0.05, f'Similarity Score {i + 1}: {score:.4f}', transform=plt.gca().transAxes)
        print(f"1111traj_comparison_{count:03d}_{index1:03d}_{index2:03d}:{sim_scores}")

        plt.title(f'Similarity Scores')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()

        # 保存图片
        img_path = os.path.join(save_dir, f'traj_comparison_{count:03d}_{index1:03d}_{index2:03d}.jpg')
        plt.savefig(img_path, dpi=150)
        plt.close()

    def forward(self, traj_1, traj_2, count, index_1, index_2, masks=None):

        # 使用信息窗口大小进行计算
        sim_scores_5, true_scores_5 = self.compute_similarity_erp(traj_1, traj_2, self.window_size_5, self.fc_5,
                                                              self.out_5, count, index_1, index_2, masks)

        # 使用判断窗口大小进行计算
        sim_scores_10, true_scores_10 = self.compute_similarity_erp(traj_1, traj_2, self.window_size_10,
                                                                self.fc_10, self.out_10, count, index_1, index_2,
                                                                masks)
        # 使用判断窗口大小进行计算
        sim_scores_15, true_scores_15 = self.compute_similarity_erp(traj_1, traj_2, self.window_size_15,
                                                                self.fc_15, self.out_15, count, index_1, index_2,
                                                                masks)
        # 使用判断窗口大小进行计算
        sim_scores_20, true_scores_20 = self.compute_similarity_dtw(traj_1, traj_2, self.window_size_20,
                                                                self.fc_20, self.out_20, count, index_1, index_2,
                                                                masks)

        # 均方差损失

        mse_loss_5 = F.mse_loss(sim_scores_5, true_scores_5)
        mse_loss_10 = F.mse_loss(sim_scores_10, true_scores_10)
        mse_loss_15 = F.mse_loss(sim_scores_15, true_scores_15)
        mse_loss_20 = F.mse_loss(sim_scores_20, true_scores_20)

        total_loss = mse_loss_10 + mse_loss_5 + mse_loss_15 + mse_loss_20

        return total_loss, sim_scores_5, true_scores_5, sim_scores_10, true_scores_10, sim_scores_15, true_scores_15, sim_scores_20, true_scores_20

    def compute_similarity_erp(self, traj_1, traj_2, WindowSize, fc_layer, output_layer, count, index_1, index_2, masks):
        num_windows = traj_1.size(0) // WindowSize
        true_scores = []
        sim_scores = []

        for i in range(num_windows):
            start_idx = i * WindowSize
            end_idx = start_idx + WindowSize

            segment_a = traj_1[start_idx:end_idx, :]
            segment_b = traj_2[start_idx:end_idx, :]

            # 获取有效部分的mask
            mask_a = masks[index_1][start_idx:end_idx]
            mask_b = masks[index_2][start_idx:end_idx]
            combined_mask = mask_a.bool() & mask_b.bool()

            if combined_mask.sum().item() == 0:
                continue  # 跳过无有效部分的段

            # 将轨迹坐标转换为嵌入向量
            embed_a = F.leaky_relu(self.embedding(segment_a))
            embed_b = F.leaky_relu(self.embedding(segment_b))

            # 计算匹配得分，只要注意力得分
            _, attn_output_a = self.attention(embed_a, embed_b, embed_b)
            _, attn_output_b = self.attention(embed_b, embed_a, embed_a)

            # 全连接层进行特征提取
            feature = torch.cat((attn_output_a, attn_output_b), dim=-1)
            features = fc_layer(feature).view(-1)

            # 计算相似度
            sim_score = output_layer(features)
            sim_scores.append(sim_score)

            if self.erp:
                true_score = erp_distance(segment_a[combined_mask], segment_b[combined_mask])
                true_score = true_score.clone().detach().view(-1)
                true_scores.append(true_score)

        sim_scores = torch.cat(sim_scores) if sim_scores else torch.tensor([], dtype=torch.float32)
        true_scores = torch.cat(true_scores)
        if Config.sim_train_plot_result:
            self.plot_trajectories(traj_1, traj_2, sim_scores, Config.savedir, count, index_1, index_2)

        return sim_scores, true_scores

    def compute_similarity_dtw(self, traj_1, traj_2, WindowSize, fc_layer, output_layer, count, index_1, index_2, masks):
        num_windows = traj_1.size(0) // WindowSize
        true_scores = []
        sim_scores = []

        for i in range(num_windows):
            start_idx = i * WindowSize
            end_idx = start_idx + WindowSize

            segment_a = traj_1[start_idx:end_idx, :]
            segment_b = traj_2[start_idx:end_idx, :]

            # 获取有效部分的mask
            mask_a = masks[index_1][start_idx:end_idx]
            mask_b = masks[index_2][start_idx:end_idx]
            combined_mask = mask_a.bool() & mask_b.bool()

            if combined_mask.sum().item() == 0:
                continue  # 跳过无有效部分的段

            # 将轨迹坐标转换为嵌入向量
            embed_a = F.leaky_relu(self.embedding(segment_a))
            embed_b = F.leaky_relu(self.embedding(segment_b))

            # 计算匹配得分，只要注意力得分
            _, attn_output_a = self.attention(embed_a, embed_b, embed_b)
            _, attn_output_b = self.attention(embed_b, embed_a, embed_a)

            # 全连接层进行特征提取
            feature = torch.cat((attn_output_a, attn_output_b), dim=-1)
            features = fc_layer(feature).view(-1)

            # 计算相似度
            sim_score = output_layer(features)
            sim_scores.append(sim_score)

            if self.dtw:
                true_score = dtw_distance(segment_a[combined_mask], segment_b[combined_mask])
                true_score = true_score.clone().detach().view(-1)
                true_scores.append(true_score)


        sim_scores = torch.cat(sim_scores) if sim_scores else torch.tensor([], dtype=torch.float32)
        true_scores = torch.cat(true_scores)
        if Config.sim_train_plot_result:
            self.plot_trajectories(traj_1, traj_2, sim_scores, Config.savedir, count, index_1, index_2)

        return sim_scores, true_scores

