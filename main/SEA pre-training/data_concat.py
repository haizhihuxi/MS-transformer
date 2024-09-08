import math

import numpy as np
import os
import pickle
import time
from torch import nn
import torch.nn.functional as F
import torch
from config_SEA import Config
import similarity_models
import logging


cf = Config()

# 配置日志记录
# 配置日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logfile.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def caculate_dis(traj1, traj2):
    lat_1 = traj1[0]
    lon_1 = traj1[1]
    lat_2 = traj2[0]
    lon_2 = traj2[1]
    distance = np.sqrt((lat_2 - lat_1) ** 2 + (lon_2 - lon_1) ** 2)
    return distance


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



        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.hidden_dim ** 0.5
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)


        return attention_weights, attention_weights


class caculatenet(nn.Module):
    def __init__(self, config):
        super(caculatenet, self).__init__()
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

    def forward(self, traj_1, traj_2):

        # 使用信息窗口大小进行计算
        sim_scores_5 = self.compute_similarity(traj_1, traj_2, self.window_size_5, self.fc_5,
                                               self.out_5)

        # 使用判断窗口大小进行计算
        sim_scores_10 = self.compute_similarity(traj_1, traj_2, self.window_size_10,
                                                self.fc_10, self.out_10)
        # 使用判断窗口大小进行计算
        sim_scores_15 = self.compute_similarity(traj_1, traj_2, self.window_size_15,
                                                self.fc_15, self.out_15)
        # 使用判断窗口大小进行计算
        sim_scores_20 = self.compute_similarity(traj_1, traj_2, self.window_size_20,
                                                self.fc_20, self.out_20)

        return sim_scores_5, sim_scores_10, sim_scores_15, sim_scores_20

    def compute_similarity(self, traj_1, traj_2, WindowSize, fc_layer, output_layer):
        num_windows = traj_1.shape[0] // WindowSize
        sim_scores = []

        for i in range(num_windows):
            start_idx = i * WindowSize
            end_idx = start_idx + WindowSize

            if start_idx > traj_2.shape[0]:
                break

            segment_a = traj_1[start_idx:end_idx, :]
            segment_b = traj_2[start_idx:end_idx, :]

            # 填充 segment_a
            if segment_a.shape[0] < WindowSize:
                pad_size = WindowSize - segment_a.shape[0]
                padding = torch.zeros(pad_size, segment_a.shape[1])
                segment_a = torch.cat((segment_a, padding), dim=0)

            # 填充 segment_b
            if segment_b.shape[0] < WindowSize:
                pad_size = WindowSize - segment_b.shape[0]
                padding = torch.zeros(pad_size, segment_b.shape[1])
                segment_b = torch.cat((segment_b, padding), dim=0)

            # 创建 mask 矩阵，1 表示真实数据，0 表示填充数据
            mask_a = torch.ones(WindowSize)
            mask_b = torch.ones(WindowSize)

            if segment_a.shape[0] < WindowSize:
                mask_a[segment_a.shape[0]:] = 0
            if segment_b.shape[0] < WindowSize:
                mask_b[segment_b.shape[0]:] = 0

            # 将轨迹坐标转换为嵌入向量
            embed_a = F.leaky_relu(self.embedding(segment_a))
            embed_b = F.leaky_relu(self.embedding(segment_b))

            # 计算匹配得分，只要注意力得分
            _, attn_output_a = self.attention(embed_a, embed_b, embed_b, mask_b)
            _, attn_output_b = self.attention(embed_b, embed_a, embed_a, mask_a)



            # 全连接层进行特征提取
            feature = torch.cat((attn_output_a, attn_output_b), dim=-1)
            features = fc_layer(feature).view(-1)

            # 计算相似度
            sim_score = output_layer(features)
            sim_scores.append(sim_score)
        sim_scores = torch.cat(sim_scores)
        # 填充相似度分数，直到 num_windows
        if len(sim_scores) < num_windows:
            last_score = sim_scores[-1].unsqueeze(0)
            padding = last_score.repeat(num_windows - len(sim_scores))
            sim_scores = torch.cat((sim_scores, padding))
        return sim_scores


def fill_zero_slopes(rate_of_change):
    non_zero_indices = (rate_of_change != 0).nonzero(as_tuple=True)[0]
    if len(non_zero_indices) == 0:
        return rate_of_change  # 如果没有非零值，直接返回原张量
    last_value = rate_of_change[non_zero_indices[-1]].item()
    for i in range(len(rate_of_change)):
        if rate_of_change[i] == 0 or torch.isinf(rate_of_change[i]):
            if i == 0:
                rate_of_change[i] = rate_of_change[1] if len(rate_of_change) > 1 and not torch.isinf(
                    rate_of_change[1]) else last_value
            else:
                rate_of_change[i] = last_value
        else:
            last_value = rate_of_change[i].item()
    return rate_of_change


def WindowWeight(index, tau):
    tau_sum = 0
    for item in tau:
        tau_sum += math.exp(item)
    if index == 1:
        res = math.exp(tau[0]) / tau_sum
    if index == 2:
        res = math.exp(tau[1]) / tau_sum
    if index == 3:
        res = math.exp(tau[2]) / tau_sum
    if index == 4:
        res = math.exp(tau[3]) / tau_sum
    return res


def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)
    normalized = torch.nan_to_num(normalized, nan=1.0)
    epsilon = 1e-10
    normalized[normalized == 0] = epsilon
    return normalized


# 读取文件
file_path = './data/train_filter.pkl'
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 相似度模型=============================
sim_model = similarity_models.TrajectoryMatchingNetwork(cf)
sim_model.load_state_dict(torch.load("re-results/-window_size-5-erp-window_size-10-erp-window_size-15-erp-window_size-20-dtw/model_sim.pt"))
model = caculatenet(cf)
model.embedding.load_state_dict(sim_model.embedding.state_dict())
model.attention.load_state_dict(sim_model.attention.state_dict())
model.fc_5.load_state_dict(sim_model.fc_5.state_dict())
model.fc_10.load_state_dict(sim_model.fc_10.state_dict())
model.fc_15.load_state_dict(sim_model.fc_15.state_dict())
model.fc_20.load_state_dict(sim_model.fc_20.state_dict())
model.out_5.load_state_dict(sim_model.out_5.state_dict())
model.out_10.load_state_dict(sim_model.out_10.state_dict())
model.out_15.load_state_dict(sim_model.out_15.state_dict())
model.out_20.load_state_dict(sim_model.out_20.state_dict())
model.eval()

# num = 0
# for idx, v in enumerate(data):
#     num += 1
#     # if idx<269:
#     #     continue
#     starttime = time.time()
#     traj = v['traj'][:, 0:2]
#     if isinstance(traj, np.ndarray):
#         traj = torch.from_numpy(traj).float()
#     traj_startpoint = v['traj'][0][0:2]  # 轨迹起点
#
#     weighted_sum = torch.zeros(len(traj), 1)
#     weighted_5 = torch.zeros(len(traj), 1)
#     weighted_10 = torch.zeros(len(traj), 1)
#     weighted_15 = torch.zeros(len(traj), 1)
#     weighted_20 = torch.zeros(len(traj), 1)
#     count = 0
#
#     for compare_idx, x in enumerate(data):
#         # if compare_idx % 100 == 0:
#         #     gc.collect()
#         if compare_idx % 2 == 0:
#             continue
#
#         if idx == compare_idx:
#             continue
#         if isinstance(x['traj'], np.ndarray):
#             x['traj'] = torch.from_numpy(x['traj']).float()
#         compare_point = x['traj'][0][0:2]  # 比较轨迹起点
#         if compare_point[0] == traj_startpoint[0] and compare_point[1] == traj_startpoint[1]:
#             continue
#         distance = caculate_dis(traj_startpoint, compare_point)
#         if distance < 0.15:
#             sim_scores_5, sim_scores_10, sim_scores_15, sim_scores_20 = model(traj, x['traj'][:, 0:2])
#
#             change_5 = (sim_scores_5 < Config.threathod_5).all().item() or (
#                     sim_scores_5 > Config.threathod_5).all().item()
#             change_10 = (sim_scores_10 < Config.threathod_10).all().item() or (
#                     sim_scores_10 > Config.threathod_10).all().item()
#             change_15 = (sim_scores_15 < Config.threathod_15).all().item() or (
#                     sim_scores_15 > Config.threathod_15).all().item()
#             change_20 = (sim_scores_20 < Config.threathod_20).all().item() or (
#                     sim_scores_20 > Config.threathod_20).all().item()
#
#             if not (change_5 and change_10 and change_15 and change_20):
#                 # 5
#                 # 计算变化率
#                 rate_of_change_5 = torch.abs(torch.diff(sim_scores_5) / sim_scores_5[:-1])
#                 # 在末尾填充一个0
#                 rate_of_change_5 = torch.cat(
#                     (rate_of_change_5, torch.tensor([0.0], device=sim_scores_5.device)))
#
#                 # 10
#                 rate_of_change_10 = torch.abs(torch.diff(sim_scores_10) / sim_scores_10[:-1])
#                 rate_of_change_10 = torch.cat(
#                     (rate_of_change_10, torch.tensor([0.0], device=sim_scores_10.device)))
#
#                 # 15
#                 rate_of_change_15 = torch.abs(torch.diff(sim_scores_15) / sim_scores_15[:-1])
#                 rate_of_change_15 = torch.cat(
#                     (rate_of_change_15, torch.tensor([0.0], device=sim_scores_15.device)))
#
#                 # 20
#                 rate_of_change_20 = torch.abs(torch.diff(sim_scores_20) / sim_scores_20[:-1])
#                 rate_of_change_20 = torch.cat(
#                     (rate_of_change_20, torch.tensor([0.0], device=sim_scores_20.device)))
#
#                 # 填充斜率为0和inf的位置
#                 rate_of_change_5 = fill_zero_slopes(rate_of_change_5)
#                 rate_of_change_10 = fill_zero_slopes(rate_of_change_10)
#                 rate_of_change_15 = fill_zero_slopes(rate_of_change_15)
#                 rate_of_change_20 = fill_zero_slopes(rate_of_change_20)
#
#                 # 将 rate_of_change_judge 和 rate_of_change_info 添加到 weighted_sum 的特定位置
#                 for i in range(len(rate_of_change_5)):
#                     weighted_sum[i * 5:(i + 1) * 5] += rate_of_change_5[i] * WindowWeight(1, cf.tau_A)
#                     weighted_5[i * 5:(i + 1) * 5] += rate_of_change_5[i]
#
#                 for i in range(len(rate_of_change_10)):
#                     weighted_sum[i * 10:(i + 1) * 10] += rate_of_change_10[i] * WindowWeight(2, cf.tau_A)
#                     weighted_10[i * 10:(i + 1) * 10] += rate_of_change_10[i]
#
#                 for i in range(len(rate_of_change_15)):
#                     weighted_sum[i * 15:(i + 1) * 15] += rate_of_change_15[i] * WindowWeight(3, cf.tau_A)
#                     weighted_15[i * 15:(i + 1) * 15] += rate_of_change_15[i]
#
#                 for i in range(len(rate_of_change_20)):
#                     weighted_sum[i * 20:(i + 1) * 20] += rate_of_change_20[i] * WindowWeight(4, cf.tau_A)
#                     weighted_20[i * 20:(i + 1) * 20] += rate_of_change_20[i]
#                 count += 1
#     if count != 0:
#         weighted_sum = weighted_sum / count
#         weighted_5 = weighted_5 / count
#         weighted_10 = weighted_10 / count
#         weighted_15 = weighted_15 / count
#         weighted_20 = weighted_20 / count
#     else:
#         weighted_sum = torch.ones(len(traj), 1)
#         weighted_5 = torch.ones(len(traj), 1)
#         weighted_10 = torch.ones(len(traj), 1)
#         weighted_15 = torch.ones(len(traj), 1)
#         weighted_20 = torch.ones(len(traj), 1)
#
#     normalized_sum = normalize(weighted_sum)
#     normalized_5 = normalize(weighted_5)
#     normalized_10 = normalize(weighted_10)
#     normalized_15 = normalize(weighted_15)
#     normalized_20 = normalize(weighted_20)
#
#     v["traj"] = np.hstack([v["traj"],
#                            normalized_sum.detach().numpy(),
#                            normalized_5.detach().numpy(),
#                            normalized_10.detach().numpy(),
#                            normalized_15.detach().numpy(),
#                            normalized_20.detach().numpy()])
#
#
#     endtime = time.time()
#     logger.info(f" idx :{num}  time:{endtime - starttime:.2f}s")
#     print(f" idx :{num}  time:{endtime - starttime:.2f}s")

Data_con = [x for x in data]
save_file_path = './data/train_erp_dtw1.pkl'

# 打开文件以写入模式
with open(save_file_path, "wb") as f:
    # 使用pickle.dump将数据写入文件
    pickle.dump(Data_con, f)
