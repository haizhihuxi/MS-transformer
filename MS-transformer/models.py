# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for TrAISformer.
    https://arxiv.org/abs/210                                                                                                                                                                               9.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config_trAISformer import Config
import Frech_distance
import math
import matplotlib.pyplot as plt
import os
import time

logger = logging.getLogger(__name__)
random.seed(42)
torch.manual_seed(42)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 检查 config.n_embd（表示输入嵌入维度的配置参数）是否能够被 config.n_head（表示头数）整除。这是因为在多头自注意力中，每个头的维度都应该相同。
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)  # 线性层768*768
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # dropout 神经元
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask因果掩码 只关注输入序列中前面的单词对后面的单词的影响，而不允许后面的单词影响前面的单词
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        # 120*120上三角矩阵改为1 x 1 x config.max_seqlen x config.max_seqlen来将批次大小设置为1。 创建一个叫做“mask”的buffer
        self.n_head = config.n_head  # 头的数目8

        # Create an input mask that ignores the last 520 columns of the embedding during evaluation
        self.register_buffer("input_mask",
                             torch.cat([torch.ones(config.n_embd - config.n_auxil), torch.zeros(config.n_auxil)]))

    def forward(self, x, weight, state):
        # if not state:
        #     x = x * self.input_mask
        B, T, C = x.size()  # 32 120 768
        # B批量大小（batch size），T序列长度（sequence length），self.n_head注意力头的数量，C // self.n_head每个头所处理的特征数量
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B32, nh8, T120, hs96)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 计算点积
        att = q @ k.transpose(-2, -1)

        # 计算缩放因子，即特征维度的平方根的倒数
        scale_factor = 1.0 / math.sqrt(k.size(-1))
        # 应用缩放因子
        att = att * scale_factor  # q(32,8,120,96)与k转置(32,8,96,120)矩阵乘法得(32,8,120,120)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # mask无关位置,mask中为0的位置用浮点数‘-inf’填充
        att = F.softmax(att, dim=-1)  # softmax(32,8,120,120)
        # att = self.ema(att)  # 对张量应用CANet模块
        att = self.attn_drop(att)  # dropout 非0参数除以0.9，放大
        y = att @ v  # (B32, nh8, T120, T120) x (B32, nh8, T120, hs96) -> (B32, nh8, T120, hs96) #*V
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # (32,8,120,96)->(32,120,8,96)->无内存碎片，确保连续->(32,120,768)
        # output projection
        y = self.resid_drop(self.proj(y))  # 先通过线形层再drop
        return y

class CausalSelfAttention_sim_weight(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, embedding, attention, fc_judge, fc_info, out_judge, out_info):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 检查 config.n_embd（表示输入嵌入维度的配置参数）是否能够被 config.n_head（表示头数）整除。这是因为在多头自注意力中，每个头的维度都应该相同。
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)  # 线性层768*768
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.weights_layer = nn.Linear(config.max_seqlen, config.max_seqlen, bias=False)  # 添加用于学习权重的全连接层
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # dropout 神经元
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.weight_concat = nn.Linear(config.max_seqlen, config.max_seqlen)
        # torch.nn.init.kaiming_normal_(self.weight_concat.weight, mode='fan_in', nonlinearity='relu')

        # causal mask因果掩码 只关注输入序列中前面的单词对后面的单词的影响，而不允许后面的单词影响前面的单词
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        # 120*120上三角矩阵改为1 x 1 x config.max_seqlen x config.max_seqlen来将批次大小设置为1。 创建一个叫做“mask”的buffer
        self.n_head = config.n_head  # 头的数目8
        self.compary_threshold = config.traj_compary_threshold
        self.register_buffer("input_mask",
                             torch.cat([torch.ones(config.n_embd - config.n_auxil), torch.zeros(config.n_auxil)]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(Config.max_seqlen, Config.max_seqlen)
        self.device = "cuda:0"

        # 使用传入的层
        self.embedding = embedding
        self.attention = attention
        self.fc_judge = fc_judge
        self.fc_info = fc_info
        self.out_judge = out_judge
        self.out_info = out_info
        self.window_size_judge = config.window_size_judge
        self.window_size_info = config.window_size_info

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )

    def compute_similarity(self, traj_1, traj_2, WindowSize, fc_layer, output_layer):
        tag = 0
        num_windows = traj_1.size(0) // WindowSize
        sim_scores = []

        for i in range(num_windows):
            if tag == 1:
                break
            start_idx = i * WindowSize
            end_idx = start_idx + WindowSize

            segment_a = traj_1[start_idx:end_idx, :]

            # 检查 segment_b 是否足够长，不足则填充
            if end_idx <= traj_2.size(0):
                segment_b = traj_2[start_idx:end_idx, :]
            else:
                segment_b = traj_2[start_idx:, :]
                padding_size = end_idx - traj_2.size(0)
                if padding_size > 0:
                    padding = torch.zeros((padding_size, traj_2.size(1)), dtype=traj_2.dtype, device=traj_2.device)
                    segment_b = torch.cat((segment_b, padding), dim=0)
                    tag = 1

            # 创建掩码矩阵，1表示非填充，0表示填充
            mask_a = (segment_a != 0).any(dim=1).float().unsqueeze(-1)
            mask_b = (segment_b != 0).any(dim=1).float().unsqueeze(-1)

            # 将轨迹坐标转换为嵌入向量
            embed_a = F.leaky_relu(self.embedding(segment_a))
            embed_b = F.leaky_relu(self.embedding(segment_b))

            # 计算匹配得分，并传入掩码矩阵
            _, attn_output_a = self.attention(embed_a, embed_b, embed_b, mask=mask_b)
            _, attn_output_b = self.attention(embed_b, embed_a, embed_a, mask=mask_a)

            # 全连接层进行特征提取
            feature = torch.cat((attn_output_a, attn_output_b), dim=-1)
            features = fc_layer(feature).view(-1)

            # 计算相似度
            sim_score = output_layer(features)
            sim_scores.append(sim_score)

        # 确保 sim_scores 长度与 a 的长度除以 window 大小相同
        while len(sim_scores) < num_windows:
            sim_scores.append(sim_scores[-1])

        sim_scores = torch.cat(sim_scores) if sim_scores else torch.tensor([], dtype=torch.float32)

        return sim_scores

    def select_trajectories_for_comparison(self, real_trajectories, all_trajectories, threshold):
        real_trajectories = real_trajectories.cpu().numpy()

        # 收集 all_trajectories 中每个轨迹的前两个点
        all_start_points = np.array([traj[4] for traj in all_trajectories])  # (9144, 2)
        # 提取真实轨迹和所有轨迹的起始点
        real_start_points = real_trajectories[:, 4, :2]  # torch.Size([32, 2])

        # 计算所有起始点与每个真实起始点之间的距离
        distance_squared = (real_start_points[:, np.newaxis, :] - all_start_points[np.newaxis, :, :]) ** 2
        distances = np.sqrt(distance_squared.sum(axis=2))  # torch.Size([32, 9144])

        # 找到 all_trajectories 中每条轨迹到任何真实轨迹的最小距离
        min_distances = np.min(distances, axis=0)  # (9144,)

        # 选择最小距离小于或等于阈值的轨迹
        selected_indices = np.where(min_distances <= threshold)[0]

        selected_trajectories = [all_trajectories[i] for i in selected_indices]
        retain_ratio = 0.01  # 你想要保留的数据点的比例

        # 新的轨迹列表，用于存储处理后的轨迹
        retained_trajectories = []

        retain_count = int(len(selected_trajectories) * retain_ratio)
        # 生成随机但不重复的索引以选择数据点
        retained_indices = np.random.choice(len(selected_trajectories), size=retain_count, replace=False)
        # 将处理后的轨迹添加到新列表中
        for i in retained_indices:
            retained_trajectories.append(selected_trajectories[i])
        return retained_trajectories

    def plot_trajectories(self, traj_1, traj_2, sim_score_judge, save_dir, count, index_1):
        # 过滤掉填充部分（值为0的部分）
        valid_mask_1 = (traj_1[:, 0] != 0) & (traj_1[:, 1] != 0)
        valid_traj_1 = traj_1[valid_mask_1]
        valid_mask_2 = (traj_2[:, 0] != 0) & (traj_2[:, 1] != 0)
        valid_traj_2 = traj_2[valid_mask_2]

        # 检查过滤后的轨迹是否为空
        if len(valid_traj_1) == 0 or len(valid_traj_2) == 0:
            print(f"Skipping plotting for traj_1 at index {index_1} or traj_2 at index {count} due to padding.")
            return
        plt.figure(figsize=(9, 6), dpi=150)
        cmap = plt.colormaps.get_cmap("jet")
        color = cmap(0.5)
        # 将 sim_score_judge 转换为 numpy 数组
        sim_scores = sim_score_judge.cpu().numpy()
        # 绘制有效的轨迹 1
        plt.plot(valid_traj_1[:, 1].cpu().numpy(), valid_traj_1[:, 0].cpu().numpy(), linestyle="-", color=color,
                 label="Trajectory 1")
        plt.plot(valid_traj_1[0, 1].cpu().numpy(), valid_traj_1[0, 0].cpu().numpy(), "o", markersize=5,
                 color=color)  # 标出起始点

        # 绘制有效的轨迹 2
        plt.plot(valid_traj_2[:, 1].cpu().numpy(), valid_traj_2[:, 0].cpu().numpy(), linestyle="-.", color="red",
                 label="Trajectory 2")
        plt.plot(valid_traj_2[0, 1].cpu().numpy(), valid_traj_2[0, 0].cpu().numpy(), "o", markersize=5,
                 color="red")  # 标出起始点
        # 标注相似度得分
        for i, score in enumerate(sim_scores):
            plt.text(0.5, 1.02 - i * 0.05, f'Similarity Score {i + 1}: {score:.4f}', transform=plt.gca().transAxes)
        plt.title(f'Similarity Scores')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        # 保存图片
        img_path = os.path.join(save_dir, f'traj_{index_1:04d}traj_{count:04d}.jpg')
        plt.savefig(img_path, dpi=150)
        plt.close()

    # 填充斜率为0的位置
    def fill_zero_slopes(self, rate_of_change):
        non_zero_indices = (rate_of_change != 0).nonzero(as_tuple=True)[0]
        if len(non_zero_indices) == 0:
            return rate_of_change  # 如果没有非零值，直接返回原张量
        last_valid_value = rate_of_change[non_zero_indices[-1]].item()
        for i in range(len(rate_of_change)):
            if rate_of_change[i] == 0:
                if i == 0:
                    rate_of_change[i] = rate_of_change[1] if len(
                        rate_of_change) > 1 else last_valid_value
                else:
                    rate_of_change[i] = last_valid_value
            else:
                last_valid_value = rate_of_change[i].item()
        return rate_of_change

    def forward(self, x, real_x, all_data, state):
        torch.cuda.empty_cache()
        B, T, C = x.size()  # 32 120 768
        q = self.value(x)
        k = self.value(x)
        v = self.value(x)
        if state:
            # #  data 是包含所有轨迹的列表
            # traj_arrays = [np.array(all_data[idx]["traj"][:, [0, 1]], dtype=np.float32) for idx in range(len(all_data))]
            # selected_trajectories = self.select_trajectories_for_comparison(real_x, traj_arrays,
            #                                                                 Config.traj_compary_threshold)
            # print(f"the number of compared trajectory：{len(selected_trajectories)}")  # 每条轨迹不一样长
            # 记录程序开始的时刻
            start_time = time.time()
            # print(f"开始计算相似度: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            weighted_sum = torch.zeros(len(real_x), real_x.shape[1] // 5).to("cuda:0")
            count = torch.zeros((len(real_x)), 1).to("cuda:0")
            for index_1 in range(0, len(real_x - 1)):

                traj_1 = real_x[index_1, :-1, 0:2]
                for index_2 in range(0, len(real_x) - 1):
                    traj_2 = torch.tensor(real_x[index_2, :-1, 0:2]).to(traj_1.device)
                    start_distance = torch.dist(traj_1[0], traj_2[0])
                    if start_distance > 0.15:
                        continue  # 如果出发点的距离大于0.15，就跳过比较

                    sim_scores_judge = self.compute_similarity(traj_1, traj_2,
                                                               self.window_size_judge,
                                                               self.fc_judge,
                                                               self.out_judge)
                    sim_scores_info = self.compute_similarity(traj_1, traj_2,
                                                              self.window_size_info,
                                                              self.fc_info,
                                                              self.out_info)
                    # 判断是否突变
                    judge_change = (sim_scores_judge < Config.threathod_10).all() or (
                            sim_scores_judge > Config.threathod_10).all()
                    info_change = (sim_scores_info < Config.threathod_5).all() or (
                            sim_scores_info > Config.threathod_5).all()
                    if not (judge_change and info_change):
                        # 假设无效数据用0填充，可以根据具体情况调整
                        invalid_value = 0.0
                        # 计算有效数据的掩码
                        valid_mask_judge = sim_scores_judge != invalid_value
                        valid_mask_info = sim_scores_info != invalid_value
                        # 初始化变化率张量，与原始数据大小相同
                        rate_of_change_judge = torch.zeros_like(sim_scores_judge)
                        rate_of_change_info = torch.zeros_like(sim_scores_info)
                        # 计算变化率时只考虑有效数据
                        valid_sim_scores_judge = sim_scores_judge[valid_mask_judge]
                        valid_sim_scores_info = sim_scores_info[valid_mask_info]
                        # 确保有足够的数据进行diff计算
                        if len(valid_sim_scores_judge) > 1:
                            rate_of_change_valid_judge = torch.cat((
                                torch.tensor([0.0], device=sim_scores_judge.device),
                                torch.abs(torch.diff(valid_sim_scores_judge) / valid_sim_scores_judge[:-1])
                            ))
                            rate_of_change_judge[valid_mask_judge] = rate_of_change_valid_judge
                        else:
                            rate_of_change_judge[valid_mask_judge] = torch.tensor([0.0], device=sim_scores_judge.device)
                        if len(valid_sim_scores_info) > 1:
                            rate_of_change_valid_info = torch.cat((
                                torch.tensor([0.0], device=sim_scores_info.device),
                                torch.abs(torch.diff(valid_sim_scores_info) / valid_sim_scores_info[:-1])
                            ))
                            rate_of_change_info[valid_mask_info] = rate_of_change_valid_info
                        else:
                            rate_of_change_info[valid_mask_info] = torch.tensor([0.0], device=sim_scores_info.device)
                        # 填充斜率为0的位置
                        rate_of_change_judge = self.fill_zero_slopes(rate_of_change_judge)
                        rate_of_change_info = self.fill_zero_slopes(rate_of_change_info)
                        # 循环遍历 rate_of_change_judge 的索引
                        for i in range(len(rate_of_change_judge)):
                            # 更新 weighted_sum 数组的 2 * i 位置的值
                            weighted_sum[index_1][2 * i] += rate_of_change_info[2 * i] * (1 - Config.weight_10) + \
                                                            rate_of_change_judge[i] * Config.weight_10
                            # 检查 2 * i + 1 是否在 rate_of_change_info 的范围内
                            if 2 * i + 1 < len(rate_of_change_info):
                                # 更新 weighted_sum 数组的 2 * i + 1 位置的值
                                weighted_sum[index_1][2 * i + 1] += rate_of_change_info[2 * i + 1] * (
                                        1 - Config.weight_10) + \
                                                                    rate_of_change_judge[i] * Config.weight_10

                        #     # 绘制比较的轨迹并保存图片
                        #     self.plot_trajectories(traj_1, traj_2, sim_scores_judge, config_trAISformer.Config.savedir,count, index_1)
                        # print(f"weighted_sum:{weighted_sum[0]}")
                        # print(f"count:{count[0]}")
                        # print()
                        count[index_1] += 1
            weighted_sum = weighted_sum / count
            # 将 NaN 替换为 0
            data = torch.nan_to_num(weighted_sum, nan=0.0)
            # 避免除以0, 将0替换为一个很小的正数
            data[data == 0] = 1e-8
            # 计算每行的最小值和最大值
            min_vals, _ = torch.min(data, dim=1, keepdim=True)
            max_vals, _ = torch.max(data, dim=1, keepdim=True)
            # 进行归一化处理
            normalized_data = (data - min_vals) / (max_vals - min_vals)
            normalized_data = torch.nan_to_num(normalized_data, nan=1.0)
            if not torch.all(normalized_data == 1):
                seqlen = min(Config.max_seqlen, len(real_x[1]))
                weights_ori = [[None for _ in range(seqlen)] for _ in range(len(normalized_data))]
                for i in range(len(normalized_data)):
                    for j in range(len(normalized_data[1])):
                        # 计算在weights_ori中的起始列位置
                        start_col = j * 5
                        # 尽可能填充五个重复的weights[i][j], 但要检查边界
                        for n in range(5):
                            if start_col + n < seqlen:
                                weights_ori[i][start_col + n] = normalized_data[i][j]
                weights_ori = torch.tensor(weights_ori, dtype=torch.float32).to('cuda:0')
                # 检查是否全为1
                mask = torch.eq(weights_ori, 1).all(dim=1)  # 检查最后一个维度上是否全为1
                temp = weights_ori[~mask]
                changed = self.fc(weights_ori[~mask]).to(self.device)  # 非全1数据通过全连接层
                # 将结果合并回原始顺序
                output = torch.ones((weights_ori.shape[0], T)).to(self.device)
                output[~mask] = changed

                weights_ori_row = output.unsqueeze(2)
                weight_ori_line = output.unsqueeze(1).unsqueeze(2)
                end_time = time.time()
                # print(f"结束计算相似度: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
                print(f"time of one patch trajectory for comparing: {end_time - start_time:.2f} s")
            else:
                weights_ori_row = torch.ones_like(v)
                weight_ori_line = torch.ones_like(v)
        else:
            weights_ori_row = torch.ones_like(v)
            weight_ori_line = torch.ones_like(v)

        v = v * weights_ori_row
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # 计算点积
        att = q @ k.transpose(-2, -1)
        all_ones = (weight_ori_line == 1).all().item()
        if all_ones:
            weight_ori_line = torch.ones_like(att)
        att = att * weight_ori_line
        scale_factor = 1.0 / math.sqrt(k.size(-1))
        att = att * scale_factor  # q(32,8,120,96)与k转置(32,8,96,120)矩阵乘法得(32,8,120,120)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # mask无关位置,mask中为0的位置用浮点数‘-inf’填充
        att = F.softmax(att, dim=-1)  # softmax(32,8,120,120)
        att = self.attn_drop(att)  # dropout 非0参数除以0.9，放大
        att = att.float()
        y = att @ v  # (B32, nh8, T120, T120) x (B32, nh8, T120, hs96) -> (B32, nh8, T120, hs96) #*V
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))  # 先通过线形层再drop
        return y


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
        if mask is not None:
            mask = mask.expand(mask.size(0), attention_scores.size(-1))
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights


class CausalSelfAttention_adaptive_weight(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 检查 config.n_embd（表示输入嵌入维度的配置参数）是否能够被 config.n_head（表示头数）整除。这是因为在多头自注意力中，每个头的维度都应该相同。
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)  # 线性层768*768
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.weights_layer = nn.Linear(config.max_seqlen, config.max_seqlen, bias=False)  # 添加用于学习权重的全连接层
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # dropout 神经元
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.weight_concat = nn.Linear(config.max_seqlen, config.max_seqlen)
        # torch.nn.init.kaiming_normal_(self.weight_concat.weight, mode='fan_in', nonlinearity='relu')

        # causal mask因果掩码 只关注输入序列中前面的单词对后面的单词的影响，而不允许后面的单词影响前面的单词
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        # 120*120上三角矩阵改为1 x 1 x config.max_seqlen x config.max_seqlen来将批次大小设置为1。 创建一个叫做“mask”的buffer
        self.n_head = config.n_head  # 头的数目8
        self.compary_threshold = config.traj_compary_threshold
        self.register_buffer("input_mask",
                             torch.cat([torch.ones(config.n_embd - config.n_auxil), torch.zeros(config.n_auxil)]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(Config.max_seqlen, Config.max_seqlen)
        self.device = "cuda:0"

    def forward(self, x, real_x, all_data, state):
        torch.cuda.empty_cache()
        B, T, C = x.size()  # 32 120 768
        # B批量大小（batch size），T序列长度（sequence length），self.n_head注意力头的数量，C // self.n_head每个头所处理的特征数量
        # k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B32, nh8, T120, hs96)
        # q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.value(x)
        k = self.value(x)
        v = self.value(x)
        if state:
            # 训练模式权重调整
            weights = Frech_distance.self_adaption_weight(real_x)
            if not np.all(weights == 1):
                # 如果一整个batch都未突变则不执行
                seqlen = min(Config.max_seqlen, len(real_x[1]))
                weights_ori = [[None for _ in range(seqlen)] for _ in range(len(weights))]
                for i in range(len(weights)):
                    for j in range(len(weights[1])):
                        # 计算在weights_ori中的起始列位置
                        start_col = j * 5
                        # 尽可能填充五个重复的weights[i][j], 但要检查边界
                        for n in range(5):
                            if start_col + n < seqlen:
                                weights_ori[i][start_col + n] = weights[i][j]

                weights_ori = np.array(weights_ori, dtype=np.float32)  # 使用np.float32确保类型兼容
                weights_ori = torch.from_numpy(weights_ori).to('cuda:0')
                # 调整形状为(B, 1, 1, T)以便广播

                # 检查是否全为1
                mask = torch.eq(weights_ori, 1).all(dim=1)  # 检查最后一个维度上是否全为1
                temp = weights_ori[~mask]
                print(f"temp ada:{temp}")
                # 只对非全1的数据应用全连接层
                # unchanged = weights_ori[mask]  # 全为1的数据
                changed = self.fc(weights_ori[~mask]).to(self.device)  # 非全1数据通过全连接层
                # 将结果合并回原始顺序
                output = torch.ones((weights_ori.shape[0], T)).to(self.device)
                output[~mask] = changed

                weights_ori_row = output.unsqueeze(2)
                weight_ori_line = output.unsqueeze(1).unsqueeze(2)


            else:
                weights_ori_row = torch.ones_like(v)
                weight_ori_line = torch.ones_like(v)

        else:
            weights_ori_row = torch.ones_like(v)
            weight_ori_line = torch.ones_like(v)

        v = v * weights_ori_row

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 计算点积
        att = q @ k.transpose(-2, -1)
        all_ones = (weight_ori_line == 1).all().item()
        if all_ones:
            weight_ori_line = torch.ones_like(att)
        att = att * weight_ori_line

        # 计算缩放因子，即特征维度的平方根的倒数
        scale_factor = 1.0 / math.sqrt(k.size(-1))
        # 应用缩放因子
        att = att * scale_factor  # q(32,8,120,96)与k转置(32,8,96,120)矩阵乘法得(32,8,120,120)

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # mask无关位置,mask中为0的位置用浮点数‘-inf’填充
        att = F.softmax(att, dim=-1)  # softmax(32,8,120,120)
        att = self.attn_drop(att)  # dropout 非0参数除以0.9，放大
        att = att.float()
        # v = v.float()
        y = att @ v  # (B32, nh8, T120, T120) x (B32, nh8, T120, hs96) -> (B32, nh8, T120, hs96) #*V
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # (32,8,120,96)->(32,120,8,96)->无内存碎片，确保连续->(32,120,768)
        # output projection
        y = self.resid_drop(self.proj(y))  # 先通过线形层再drop
        return y



class CausalSelfAttention_newsim_weight(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 检查 config.n_embd（表示输入嵌入维度的配置参数）是否能够被 config.n_head（表示头数）整除。这是因为在多头自注意力中，每个头的维度都应该相同。
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)  # 线性层768*768
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.weights_layer = nn.Linear(config.max_seqlen, config.max_seqlen, bias=False)  # 添加用于学习权重的全连接层
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # dropout 神经元
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.weight_concat = nn.Linear(config.max_seqlen, config.max_seqlen)
        # causal mask因果掩码 只关注输入序列中前面的单词对后面的单词的影响，而不允许后面的单词影响前面的单词
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        # 120*120上三角矩阵改为1 x 1 x config.max_seqlen x config.max_seqlen来将批次大小设置为1。 创建一个叫做“mask”的buffer
        self.n_head = config.n_head  # 头的数目8
        self.compary_threshold = config.traj_compary_threshold
        self.register_buffer("input_mask",
                             torch.cat([torch.ones(config.n_embd - config.n_auxil), torch.zeros(config.n_auxil)]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(Config.max_seqlen, Config.max_seqlen)
        self.device = "cuda:0"

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )

    def forward(self, x, weight, state):
        torch.cuda.empty_cache()
        B, T, C = x.size()  # 32 120 768
        q = self.value(x)
        k = self.value(x)
        v = self.value(x)

        if state:
            # current_torch_seed = torch.initial_seed()
            # print(f"Current PyTorch seed: {current_torch_seed}")
            if random.random() > 0.6:
                weights_row = weight[:, :, 0].unsqueeze(2)
                weight_line = weight[:, :, 0].unsqueeze(1).unsqueeze(2)
            else:
                weights_row = torch.ones_like(v)
                weight_line = torch.ones_like(v)
            # weights_row = weight[:, :, 0].unsqueeze(2)
            # weight_line = weight[:, :, 0].unsqueeze(1).unsqueeze(2)
            # weights_row = torch.ones_like(v)
            # weight_line = torch.ones_like(v)
        else:
            weights_row = torch.ones_like(v)
            weight_line = torch.ones_like(v)
        v = v * weights_row
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # 计算点积
        att = q @ k.transpose(-2, -1)
        all_ones = (weight_line == 1).all().item()
        if all_ones:
            weight_line = torch.ones_like(att)
        att = att * weight_line
        scale_factor = 1.0 / math.sqrt(k.size(-1))
        att = att * scale_factor  # q(32,8,120,96)与k转置(32,8,96,120)矩阵乘法得(32,8,120,120)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # mask无关位置,mask中为0的位置用浮点数‘-inf’填充
        att = F.softmax(att, dim=-1)  # softmax(32,8,120,120)
        att = self.attn_drop(att)  # dropout 非0参数除以0.9，放大
        att = att.float()
        y = att @ v  # (B32, nh8, T120, T120) x (B32, nh8, T120, hs96) -> (B32, nh8, T120, hs96) #*V
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))  # 先通过线形层再drop
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层，将序列信息压缩，用于预测目的地信息
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x, weight, state, des):
        x = x + self.attn(self.ln1(x), weight, state)
        x = x + self.mlp(self.ln2(x))
        if state:
            auxil_x = x
            auxil_x = self.global_avg_pool(auxil_x.transpose(1, 2)).transpose(1, 2)  # (32,1,768)
            des = des + self.mlp_des(self.ln3(auxil_x))
        return x, weight, state, des

class Block_newsim_weight(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )
        self.attn = CausalSelfAttention_newsim_weight(config)

    def forward(self, x_embedding, weight, state, des):
        """将x归一化后输入自注意力模块attn再与自身相加，然后再次归一化，并通过前馈神经网络mlp进行处理"""

        x_embedding = x_embedding + self.attn(self.ln1(x_embedding), weight, state)
        x_embedding = x_embedding + self.mlp(self.ln2(x_embedding))
        if state:
            auxil_x = x_embedding
            auxil_x = self.global_avg_pool(auxil_x.transpose(1, 2)).transpose(1, 2)  # (32,1,768)
            des = des + self.mlp_des(self.ln3(auxil_x))

        return x_embedding, weight, state, des


class Block_adaptive_weight(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttention_adaptive_weight(config)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )

    def forward(self, x, real_x, all_data, state, des):
        """将x归一化后输入自注意力模块attn再与自身相加，然后再次归一化，并通过前馈神经网络mlp进行处理"""
        x = x + self.attn(self.ln1(x), real_x, all_data, state)
        x = x + self.mlp(self.ln2(x))
        if state:
            auxil_x = x
            auxil_x = self.global_avg_pool(auxil_x.transpose(1, 2)).transpose(1, 2)  # (32,1,768)
            des = des + self.mlp_des(self.ln3(auxil_x))

        return x, real_x, all_data, state, des


class Block_sim_weight(nn.Module):
    def __init__(self, config, embedding, attention, fc_judge, fc_info, out_judge, out_info):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 使用传入的层
        self.embedding = embedding
        self.attention = attention
        self.fc_judge = fc_judge
        self.fc_info = fc_info
        self.out_judge = out_judge
        self.out_info = out_info

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )
        self.attn = CausalSelfAttention_sim_weight(config, self.embedding, self.attention, self.fc_judge, self.fc_info,
                                                   self.out_judge, self.out_info)

    def forward(self, x_embedding, x_real, all_data, state, des):
        """将x归一化后输入自注意力模块attn再与自身相加，然后再次归一化，并通过前馈神经网络mlp进行处理"""

        x_embedding = x_embedding + self.attn(self.ln1(x_embedding), x_real, all_data, state)
        x_embedding = x_embedding + self.mlp(self.ln2(x_embedding))
        if state:
            auxil_x = x_embedding
            auxil_x = self.global_avg_pool(auxil_x.transpose(1, 2)).transpose(1, 2)  # (32,1,768)
            des = des + self.mlp_des(self.ln3(auxil_x))

        return x_embedding, x_real, all_data, state, des





class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x_embedding, weight, state, des):
        # 逐个模块处理输入
        for module in self.modules_list:
            x_embedding, weight, state, des = module(x_embedding, weight, state, des)
        return x_embedding, weight, state, des


class adaptiveModel(nn.Module):
    def __init__(self, config):
        super(adaptiveModel, self).__init__()
        # 创建一个包含所需顺序的块的列表
        layers = []

        for _ in range(0, 2):
            layers.append(Block(config))
        for _ in range(0, 1):
            layers.append(Block_newsim_weight(config))
        for _ in range(0, 5):
            layers.append(Block(config))

        # for _ in range(0, 8):
        #     layers.append(Block(config))



        # 使用列表创建 CustomSequential
        self.blocks = CustomSequential(*layers)

    def forward(self, x_embedding, weight, state, des):
        return self.blocks(x_embedding, weight, state, des)


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model=None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.lat_target_size = config.lat_target_size
        self.lon_target_size = config.lon_target_size
        self.full_size = config.full_size
        self.full_des_size = config.full_des_size
        self.n_head = config.n_head

        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.n_lat_target_embd = config.n_lat_target_embd
        self.n_lon_target_embd = config.n_lon_target_embd
        self.register_buffer("att_sizes", torch.tensor(
            [config.lat_size, config.lon_size, config.sog_size, config.cog_size]))  # 250,270,30,72
        self.register_buffer("destination_sizes", torch.tensor([config.lat_target_size, config.lon_target_size]))
        self.register_buffer("emb_sizes", torch.tensor(
            [config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd, config.n_lat_target_embd,
             config.n_lon_target_embd]))

        if hasattr(config, "partition_mode"):  # 数据分区的模型
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model

        if hasattr(config, "lat_min"):  # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        if hasattr(config, "mode"):  # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)  # 元素的数量250  每个元素在嵌入层中的表示的维度256
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)  # 270 256
        self.lat_target_emb = nn.Embedding(self.lat_target_size, config.n_lat_target_embd)
        self.lon_target_emb = nn.Embedding(self.lon_target_size, config.n_lon_target_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)  # 30 128
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)  # 72 128
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)  # 0.1

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.ln_auxil = nn.LayerNorm(config.n_auxil)
        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.head_auxil = nn.Linear(config.n_auxil, config.n_auxil, bias=False)
        else:
            # 将概率转化为离散经纬度的
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False)  # Classification head
            self.head_auxil = nn.Linear(config.n_auxil, self.full_des_size, bias=False)
        self.max_seqlen = config.max_seqlen
        self.blocks = adaptiveModel(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )

        # 计算模型中所有参数的数量
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def print_param_names(model):
        for name, param in model.named_parameters():
            print(name)

    # 初始化模型权重
    # 对于线性层和嵌入层，使用正态分布来初始化权重，对于归一化层，将偏置项初始化为 0，并将权重初始化为 1
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)  # 均值：0  标准差：0.02
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        区分权重衰减参数和不会权重衰减的参数
        该函数将模型的所有参数分为两类：需要进行权重衰减的参数和不需要进行权重衰减的参数
        （例如偏置项和归一化/嵌入层的权重），然后返回一个PyTorch优化器对象。
        """
        # 初始化集合以保存参数名称
        weight_decay_params = set()
        no_weight_decay_params = set()

        # 定义需要和不需要权重衰减的模块类型
        decay_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.AdaptiveAvgPool2d, torch.nn.Conv2d)
        no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.GroupNorm)

        # 遍历所有命名模块及其参数
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = f'{module_name}.{param_name}' if module_name else param_name  # 生成完整的参数名称

                if param_name.endswith('bias'):
                    # 所有偏置项不进行权重衰减
                    no_weight_decay_params.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, decay_modules):
                    # 白名单模块的权重进行权重衰减
                    weight_decay_params.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, no_decay_modules):
                    # 黑名单模块的权重不进行权重衰减
                    no_weight_decay_params.add(full_param_name)

        # 特殊处理位置嵌入参数
        no_weight_decay_params.add('pos_emb')

        # 验证所有参数都被考虑到了
        param_dict = {param_name: param for param_name, param in self.named_parameters() if param.requires_grad}
        intersect_params = weight_decay_params & no_weight_decay_params  # 取交集
        all_params = weight_decay_params | no_weight_decay_params  # 取并集

        # # 输出调试信息
        # print("参数字典中的键：", param_dict.keys())
        # print("不进行权重衰减的参数：", no_weight_decay_params)
        # print("权重衰减的参数：", weight_decay_params)

        assert len(intersect_params) == 0, "参数 %s 同时出现在了 decay 和 no_decay 集合中!" % (str(intersect_params),)
        assert len(param_dict.keys() - all_params) == 0, \
            "参数 %s 没有被分配到 decay 或 no_decay 集合中!" % (str(param_dict.keys() - all_params),)
        #
        # 检查并输出未找到的参数
        missing_params_no = [param_name for param_name in no_weight_decay_params if param_name not in param_dict]
        missing_params = [param_name for param_name in weight_decay_params if param_name not in param_dict]
        if missing_params:
            print("未找到的衰减参数：", missing_params)
            print("未找到的不衰减参数：", missing_params_no)

        # 创建 PyTorch 优化器对象
        optimizer_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(weight_decay_params)) if
                           param_name in param_dict],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(list(no_weight_decay_params)) if
                           param_name in param_dict],
                "weight_decay": 0.0
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def to_indexes(self, x, destination=None):  # uniform
        idxs = (x * self.att_sizes).long()  # 真实数据（lan，lon,sog,cog）*[250, 270,  30,  72] 每个真实值都是一个独一无二的索引
        if destination is not None:
            device = 'cuda:0'
            destination = destination.to(device)
            idxs_des = (destination * self.destination_sizes).long()
        else:
            idxs_des = None
        return idxs, idxs_des


    def forward(self, x, destination, seq_weight=None, masks=None, with_targets=False, return_loss_tuple=False,
                weight_caculate=False):  # model根据参数，直接加载forward函数
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """

        idxs, idxs_des = self.to_indexes(x, destination)

        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()  # 输入排除最后一行
            targets = idxs[:, 1:, :].contiguous()  # 目标值排除第一行

        else:
            inputs = idxs
            targets = None
        batchsize, seqlen, _ = inputs.size()  # 32 120
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."  # 检查seqlen（输入序列的长度）是否小于或等于self.max_seqlen（模型的最大序列长度）。

        # forward the GPT model
        lat_embeddings = self.lat_emb(
            inputs[:, :, 0])  # (bs, seqlen) -> (bs, seqlen, lat_size) e.t. (32,120)->(32,# 120,256)词嵌入
        lon_embeddings = self.lon_emb(inputs[:, :, 1])  # (bs, seqlen, lon_size)
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),
                                     dim=-1)  # 4-hot 在最后一个维度拼接
        position_embeddings = self.pos_emb[:, :seqlen, :]
        fea = self.drop(token_embeddings + position_embeddings)
        # 初始化一个空的张量
        batch_size = x.shape[0]
        third_dim = 512  # 自定义的第三维度大小

        # 创建新的张量
        des = torch.zeros(batch_size, 1, third_dim, device=x.device, dtype=x.dtype)
        if seq_weight is not None:
            seq_weight = seq_weight[:, :-1, :]


        fea = self.blocks(fea, seq_weight, weight_caculate, des)  # 一系列的Transforme

        fea_main = fea[0]  # 主任务
        fea_main = self.ln_f(fea_main)  # (bs, seqlen, n_embd) e.t. (32,120,768)归一化层
        logits = self.head(fea_main)  # (bs, seqlen, full_size)e,t. (32,120,622) or (bs, seqlen, n_embd)

        fea_auxil = fea[3] / 8  # 辅助任务
        fea_auxil = self.ln_auxil(fea_auxil)
        logits_auxil = self.head_auxil(fea_auxil)
        lat_logits_target, lon_logits_target = torch.split(logits_auxil, (self.lat_target_size, self.lon_target_size),
                                                           dim=-1)
        # 将经过Layer Normalization处理后的特征图通过一个全连接层（即self.head）进行提取特征并分类。这里的输出logits是模型对于输入的预测结果。
        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(logits, (
            self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

        # 将全连接层的输出按照纬度、经度、速度方向和航向的维度进行拆分，得到各自的logits。这四个logits分别表示模型对于输入的纬度、经度、速度方向和航向的预测结果

        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:
            # 交叉熵损失函数

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), targets[:, :, 2].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            # 交叉熵只计算目标索引对应位置的概率值的负对数作为顺势函数的输出
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size),
                                       targets[:, :, 3].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                                       targets[:, :, 0].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                                       targets[:, :, 1].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lat_target_loss = F.cross_entropy(lat_logits_target.view(-1, self.lat_target_size),
                                              idxs_des[:, :, 0].view(-1),
                                              reduction="none").view(batchsize, -1)
            lon_target_loss = F.cross_entropy(lon_logits_target.view(-1, self.lon_size),
                                              idxs_des[:, :, 1].view(-1),
                                              reduction="none").view(batchsize, -1)

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)

            loss_tuple_des = (lat_target_loss, lon_target_loss)
            loss_des = sum(loss_tuple_des)
            loss_des = loss_des.mean()


            if masks is not None:
                # 使用掩码（masks）进行归一化。
                loss = loss * masks
                loss = loss.sum(dim=1) / masks.sum(dim=1)  # 取每个序列长度的损失

            loss = loss.mean()  # 一个值
            loss += loss_des

        if return_loss_tuple:  # false
            return logits, loss, loss_tuple
        else:
            return logits, loss
